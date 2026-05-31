"""진단: 무작위 CV vs 시간블록 CV 의 OOF 차이로 시간 누수/낙관 편향을 확인.

각 CV 에서 prior-only / model-only / blend(OOF 가중탐색) 점수를 출력한다.
시간블록 CV 점수가 LB(0.608)에 가까우면, 그것이 정직한 검증 기준이다.
실행: python -m src.diagnose
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from . import cv as CV

CLIP = 1e-6
N_SPLITS = 5
SMOOTH = 8.0
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


def run_cv(Xtr, ytr, mtr, folds_fn):
    folds = folds_fn(mtr, n_splits=N_SPLITS)
    subj = mtr["subject_id"].values
    oof_m = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    oof_p = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    for t in C.TARGET_COLS:
        y = ytr[t].values; gmean = y.mean()
        for f in range(N_SPLITS):
            tr = np.where(folds != f)[0]; va = np.where(folds == f)[0]
            pmap = smoothed_mean(y, tr, subj, gmean)
            prior = np.array([pmap.get(s, gmean) for s in subj])
            Xf = Xtr.copy(); Xf["subj_prior"] = prior
            dtr = lgb.Dataset(Xf.iloc[tr], label=y[tr], categorical_feature=["subject_id"], free_raw_data=False)
            dva = lgb.Dataset(Xf.iloc[va], label=y[va], categorical_feature=["subject_id"], free_raw_data=False)
            m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
            c = oof_m.columns.get_loc(t)
            oof_m.iloc[va, c] = m.predict(Xf.iloc[va])
            oof_p.iloc[va, c] = prior[va]
    # 점수
    prior_s = np.mean([ll(ytr[t], oof_p[t]) for t in C.TARGET_COLS])
    model_s = np.mean([ll(ytr[t], oof_m[t]) for t in C.TARGET_COLS])
    blend_per, wsum = [], {}
    for t in C.TARGET_COLS:
        best_w, best = 0.0, 1e9
        for w in np.linspace(0, 1, 41):
            v = ll(ytr[t], w * oof_m[t] + (1 - w) * oof_p[t])
            if v < best: best, best_w = v, w
        blend_per.append(best); wsum[t] = best_w
    return prior_s, model_s, float(np.mean(blend_per)), wsum


def main():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    for name, fn in [("RANDOM (현재)", CV.subject_stratified_folds),
                     ("TIME-BLOCKED (정직)", CV.subject_time_blocked_folds)]:
        ps, ms, bs, w = run_cv(Xtr, ytr, mtr, fn)
        print(f"\n=== {name} ===")
        print(f"  prior-only : {ps:.4f}")
        print(f"  model-only : {ms:.4f}")
        print(f"  blend      : {bs:.4f}   weights(model비중): {[f'{w[t]:.2f}' for t in C.TARGET_COLS]}")


if __name__ == "__main__":
    main()
