"""빠른 정직 평가기 (실험 반복용, ~30초).

LGBM 단일 모델 + 피험자 prior + 블렌드를, 정직한 '시간블록 CV'로 평가한다.
출력의 핵심은 'last' 열(= 마지막 시간블록 = test와 동일한 과거→미래 regime).
실험할 때 이 도구를 돌리고 'OVERALL last' 값이 내려가면 개선이다(낮을수록 좋음).
최종 제출은 train_ensemble 로 확정한다.

실행: python -m src.evaluate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds

CLIP = C.PROB_CLIP
N_SPLITS = 5
SMOOTH = C.PRIOR_SMOOTH
WGRID = np.linspace(0, 1, 11)
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=1.0, lambda_l2=1.0, verbosity=-1, seed=42)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def ll(y, p):
    return log_loss(y, clip(p), labels=[0, 1])


def smoothed_mean(y, idx, subj, gmean):
    df = pd.DataFrame({"s": subj[idx], "y": y[idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + SMOOTH * gmean) / (g["count"] + SMOOTH)).to_dict()


def evaluate():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=N_SPLITS)
    last = folds == (N_SPLITS - 1)
    subj = mtr["subject_id"].values

    oof_m = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    oof_p = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    for t in C.TARGET_COLS:
        y = ytr[t].values; gm = y.mean()
        for f in range(N_SPLITS):
            tr = np.where(folds != f)[0]; va = np.where(folds == f)[0]
            pmap = smoothed_mean(y, tr, subj, gm)
            prior = np.array([pmap.get(s, gm) for s in subj])
            Xf = Xtr.copy(); Xf["subj_prior"] = prior
            dtr = lgb.Dataset(Xf.iloc[tr], label=y[tr], categorical_feature=["subject_id"], free_raw_data=False)
            dva = lgb.Dataset(Xf.iloc[va], label=y[va], categorical_feature=["subject_id"], free_raw_data=False)
            m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
            c = oof_m.columns.get_loc(t)
            oof_m.iloc[va, c] = m.predict(Xf.iloc[va]); oof_p.iloc[va, c] = prior[va]

    print(f"\n{'tgt':4s} | {'prior':>6s} {'model':>6s} {'blend':>6s} (full) | "
          f"{'prior':>6s} {'model':>6s} {'blend':>6s} (last) | w*  판정")
    print("-" * 86)
    fb, lb = [], []
    for t in C.TARGET_COLS:
        # 블렌드 가중: full OOF 로 선택
        best_w, best = 0.0, 1e9
        for w in WGRID:
            v = ll(ytr[t].values, w * oof_m[t] + (1 - w) * oof_p[t])
            if v < best: best, best_w = v, w
        blf = ll(ytr[t].values, best_w * oof_m[t] + (1 - best_w) * oof_p[t])
        bll = ll(ytr[t].values[last], best_w * oof_m[t].values[last] + (1 - best_w) * oof_p[t].values[last])
        pf, mf = ll(ytr[t].values, oof_p[t]), ll(ytr[t].values, oof_m[t])
        pl = ll(ytr[t].values[last], oof_p[t].values[last])
        mlv = ll(ytr[t].values[last], oof_m[t].values[last])
        verdict = "센서도움" if mlv < pl - 0.003 else ("prior유리" if mlv > pl + 0.003 else "무승부")
        print(f"{t:4s} | {pf:6.3f} {mf:6.3f} {blf:6.3f}        | "
              f"{pl:6.3f} {mlv:6.3f} {bll:6.3f}        | {best_w:.1f} {verdict}")
        fb.append(blf); lb.append(bll)
    print("-" * 86)
    print(f"OVERALL blend  full={np.mean(fb):.4f}   last={np.mean(lb):.4f}   "
          f"<- 'last'가 진짜 점수(낮을수록 좋음)")


if __name__ == "__main__":
    evaluate()
