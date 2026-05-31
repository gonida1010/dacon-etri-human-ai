"""정식 앙상블 파이프라인 (정직한 시간블록 CV 기준).

개선 적용:
 ① 다중 모델: LGBM + XGBoost + CatBoost 평균 앙상블
 ② 확률 캘리브레이션: per-target isotonic, '마지막 블록' 정직검증으로 게이팅(도움될 때만)
 ③ 약한 타깃 피처: 수면 규칙성·HR 동역학·취침전 부하 (build_dataset 에서 추가)
 ④ 수면구간 피처 (sleep_features) 유지·z-score

검증:
 - 시간블록 5-fold OOF. 그중 fold==마지막블록 = '과거→미래' 외삽 = test 와 동일 regime → 헤드라인.
 - 타깃별 (앙상블 vs 피험자 prior) 블렌드 가중을 OOF 에서 coarse 그리드로 선택(과적합 억제).

실행: python -m src.train_ensemble
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.isotonic import IsotonicRegression

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .models import fit_fold, select_topk

CLIP = C.PROB_CLIP
N_SPLITS = 5
SMOOTH = C.PRIOR_SMOOTH
MODELS = ["lgb", "xgb", "cat"]
SEED = 42
WGRID = np.linspace(0, 1, 11)  # coarse: 0,0.1,...,1.0


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def ll(y, p):
    return log_loss(y, clip(p), labels=[0, 1])


def smoothed_mean(y, idx, subj, gmean):
    df = pd.DataFrame({"s": subj[idx], "y": y[idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + SMOOTH * gmean) / (g["count"] + SMOOTH)).to_dict()


def train():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=N_SPLITS)
    last_block = folds == (N_SPLITS - 1)   # test 와 동일한 외삽 regime
    subj_tr, subj_te = mtr["subject_id"].values, mte["subject_id"].values

    # 모델별 OOF/테스트, prior
    oof = {m: pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS) for m in MODELS}
    test = {m: pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS) for m in MODELS}
    oof_prior = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    test_prior = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)

    for t in C.TARGET_COLS:
        y = ytr[t].values
        gmean = y.mean()
        for f in range(N_SPLITS):
            tr = np.where(folds != f)[0]
            va = np.where(folds == f)[0]
            pmap = smoothed_mean(y, tr, subj_tr, gmean)
            prior_tr = np.array([pmap.get(s, gmean) for s in subj_tr])
            prior_te = np.array([pmap.get(s, gmean) for s in subj_te])
            Xtr_f = Xtr.copy(); Xtr_f["subj_prior"] = prior_tr
            Xte_f = Xte.copy(); Xte_f["subj_prior"] = prior_te
            c = oof_prior.columns.get_loc(t)
            oof_prior.iloc[va, c] = prior_tr[va]
            test_prior[t] += prior_te / N_SPLITS
            cols = (select_topk(Xtr_f.iloc[tr], y[tr], C.TOP_K_FEATURES)
                    if C.TOP_K_FEATURES else list(Xtr_f.columns))
            for m in MODELS:
                va_p, te_p = fit_fold(m, Xtr_f.iloc[tr][cols], y[tr], Xtr_f.iloc[va][cols],
                                      y[va], Xte_f[cols], SEED)
                oof[m].iloc[va, c] = va_p
                test[m][t] += te_p / N_SPLITS
        print(f"  done target {t}", flush=True)

    # 앙상블(3모델 평균)
    oof_ens = sum(oof[m] for m in MODELS) / len(MODELS)
    test_ens = sum(test[m] for m in MODELS) / len(MODELS)

    # ② per-target 캘리브레이션 게이팅: isotonic 을 non-last 폴드로 적합 → last 블록에서 검증
    calib_on = {}
    oof_cal = oof_ens.copy()
    test_cal = test_ens.copy()
    for t in C.TARGET_COLS:
        nl = ~last_block
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_ens[t].values[nl], ytr[t].values[nl])
        base_ll = ll(ytr[t].values[last_block], oof_ens[t].values[last_block])
        cal_ll = ll(ytr[t].values[last_block], iso.predict(oof_ens[t].values[last_block]))
        if cal_ll < base_ll - 1e-4:
            calib_on[t] = True
            iso_full = IsotonicRegression(out_of_bounds="clip").fit(oof_ens[t].values, ytr[t].values)
            oof_cal[t] = iso_full.predict(oof_ens[t].values)
            test_cal[t] = iso_full.predict(test_ens[t].values)
        else:
            calib_on[t] = False

    # 블렌드 가중(앙상블 vs prior): 전체 OOF 로 coarse 선택
    weights, blended_oof, blended_test = {}, {}, {}
    for t in C.TARGET_COLS:
        best_w, best = 0.0, 1e9
        for w in WGRID:
            v = ll(ytr[t].values, w * oof_cal[t] + (1 - w) * oof_prior[t])
            if v < best:
                best, best_w = v, w
        weights[t] = best_w
        blended_oof[t] = best_w * oof_cal[t] + (1 - best_w) * oof_prior[t]
        blended_test[t] = best_w * test_cal[t].values + (1 - best_w) * test_prior[t].values

    # ===== 리포트 =====
    def score_df(df, mask):
        return float(np.mean([ll(ytr[t].values[mask], df[t].values[mask]) for t in C.TARGET_COLS]))

    def score_blend(mask):
        return float(np.mean([ll(ytr[t].values[mask], np.asarray(blended_oof[t])[mask]) for t in C.TARGET_COLS]))

    full = np.ones(len(Xtr), bool)
    print("\n=== 시간블록 OOF (전체 / 마지막블록=test와 동일 regime) ===")
    print(f"{'':16s} {'full':>8s} {'last-blk':>9s}")
    print(f"{'prior':16s} {score_df(oof_prior,full):8.4f} {score_df(oof_prior,last_block):9.4f}")
    for m in MODELS:
        print(f"{m:16s} {score_df(oof[m],full):8.4f} {score_df(oof[m],last_block):9.4f}")
    print(f"{'ensemble':16s} {score_df(oof_ens,full):8.4f} {score_df(oof_ens,last_block):9.4f}")
    print(f"{'ens+calib':16s} {score_df(oof_cal,full):8.4f} {score_df(oof_cal,last_block):9.4f}")
    print(f"{'BLEND(final)':16s} {score_blend(full):8.4f} {score_blend(last_block):9.4f}")
    print(f"\n  per-target weights(앙상블비중): " +
          ", ".join(f"{t}={weights[t]:.1f}{'*cal' if calib_on[t] else ''}" for t in C.TARGET_COLS))

    # 제출
    sub = mte.copy()
    for t in C.TARGET_COLS:
        sub[t] = clip(blended_test[t])
    headline = score_blend(last_block)
    out = C.SUBMISSION_DIR / f"ensemble_lastblk{headline:.4f}.csv"
    sub.to_csv(out, index=False)
    print(f"\n  submission saved: {out}")
    return headline


if __name__ == "__main__":
    train()
