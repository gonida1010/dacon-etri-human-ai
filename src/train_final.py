"""정식 파이프라인 (통합본).

소스 4종을 만들고, 타깃별로 '모델 + 한 소스'를 순방향(last-block) 기준으로 블렌드한다.
 - model  : 멀티시드 LGBM (센서)
 - prior  : 피험자 스무딩 평균 (base rate 수축)
 - trend  : 피험자별 시간 추세 ridge 외삽
 - recent : 최근 k일 라벨 평균
과적합 방지를 위해 coarse 가중 그리드 + 마진 가드(모델보다 margin 이상 좋을 때만 비모델 소스 채택).
모든 소스는 인과적(과거만)·test 실현가능(라벨 미사용 or train 라벨만).

실행: python -m src.train_final
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .train_temporal_prior import fit_lgbm_oof_test, temporal_oof, temporal_test, clip

# 비모델 후보 소스(인과적). 이름은 train_temporal_prior._predict_temporal_one 규칙.
# prior 수축(mean_sm*), 최근평균(last*), 시간추세(ridge*) — 전부 원리적으로 동기가 있는 소스만.
CANDIDATES = ["mean_sm8", "mean_sm16", "mean_sm32", "mean_sm64",
              "last10_sm4", "last20_sm4", "ridge1", "ridge3", "ridge10"]
WGRID = [0.25, 0.5, 0.75, 1.0]   # 비모델 소스에 줄 가중(나머지는 모델)
MARGIN = 0.002                    # last-block에서 이만큼 더 좋아야 비모델 소스 채택


def ll(y, p):
    return log_loss(np.asarray(y), clip(p), labels=[0, 1])


def train():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    last = folds == (C.N_SPLITS - 1)

    # 1) 모델 소스 (멀티시드)
    oof_model, _, test_model = fit_lgbm_oof_test(Xtr, ytr, Xte, mtr, mte, folds)

    chosen, oof_final, test_final = {}, {}, {}
    rows = []
    for t in C.TARGET_COLS:
        y = ytr[t].values
        m_oof = oof_model[t].values
        m_test = test_model[t].values
        base_last = ll(y[last], m_oof[last])
        base_full = ll(y, m_oof)

        best = {"recipe": "model", "w": 0.0, "src": None,
                "last": base_last, "full": base_full,
                "oof": m_oof, "test": m_test}

        for src in CANDIDATES:
            s_oof = temporal_oof(ytr, mtr, folds, t, src)
            s_test = temporal_test(ytr, mtr, mte, t, src)
            for w in WGRID:
                bo = (1 - w) * m_oof + w * s_oof
                last_score = ll(y[last], bo[last])
                if last_score < best["last"] - MARGIN:
                    best = {"recipe": f"{1-w:.2f}*model+{w:.2f}*{src}", "w": w, "src": src,
                            "last": last_score, "full": ll(y, bo),
                            "oof": bo, "test": (1 - w) * m_test + w * s_test}

        chosen[t] = best["recipe"]
        oof_final[t] = clip(best["oof"])
        test_final[t] = clip(best["test"])
        rows.append((t, best["recipe"], best["full"], best["last"]))

    print("\n=== train_final: 타깃별 채택 recipe (full / last) ===")
    print(f"{'tgt':4s} {'recipe':26s} {'full':>8s} {'last':>8s}")
    for t, r, fu, la in rows:
        print(f"{t:4s} {r:26s} {fu:8.4f} {la:8.4f}")
    full_mean = float(np.mean([ll(ytr[t].values, oof_final[t]) for t in C.TARGET_COLS]))
    last_mean = float(np.mean([ll(ytr[t].values[last], oof_final[t][last]) for t in C.TARGET_COLS]))
    print("  ----")
    print(f"  BLEND full      : {full_mean:.4f}")
    print(f"  BLEND last-block: {last_mean:.4f}   <- 진짜 점수(LB와 일치)")

    sub = mte.copy()
    for t in C.TARGET_COLS:
        sub[t] = test_final[t]
    out = C.SUBMISSION_DIR / f"submission_final_last{last_mean:.4f}.csv"
    sub.to_csv(out, index=False)
    print(f"\n  submission saved: {out.name}")
    return last_mean


if __name__ == "__main__":
    train()
