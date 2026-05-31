"""v2: 피험자 사전확률(OOF) 주입 + 센서 모델 + OOF 블렌드 가중 탐색.

핵심: 타깃이 '개인 평균 대비'로 정의되어 피험자 기준선이 지배적 신호.
 - 각 타깃에 대해 누수 없는 OOF 스무딩 피험자 평균을 피처로 추가
 - 추가로 LGBM 예측과 피험자 사전확률을 OOF 에서 최적 가중으로 블렌드
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_stratified_folds

CLIP = 1e-6
N_SPLITS = 5
SEED = 42
SMOOTH = 8.0  # 피험자 평균 스무딩 강도(전역 평균으로의 수축)

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=25,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=1.0,
    lambda_l2=1.0,
    verbosity=-1,
    seed=SEED,
)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def avg_logloss(y_true, p):
    per = {t: log_loss(y_true[t].values, clip(p[t].values), labels=[0, 1]) for t in C.TARGET_COLS}
    return float(np.mean(list(per.values()))), per


def smoothed_mean(y, idx, subj, global_mean):
    """idx 행만으로 피험자별 스무딩 평균 시리즈(subject->prior) 계산."""
    df = pd.DataFrame({"s": subj[idx], "y": y[idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + SMOOTH * global_mean) / (g["count"] + SMOOTH)).to_dict()


def train():
    Xtr, ytr, Xte, mtr, mte, feat_cols = build_dataset(use_cache=True)
    folds = subject_stratified_folds(mtr, n_splits=N_SPLITS, seed=SEED)
    subj_tr = mtr["subject_id"].values
    subj_te = mte["subject_id"].values
    cat_feature = ["subject_id"]

    oof_model = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    oof_prior = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    test_model = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)
    test_prior = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)

    for t in C.TARGET_COLS:
        y = ytr[t].values
        gmean = y.mean()
        for f in range(N_SPLITS):
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]

            prior_map = smoothed_mean(y, tr_idx, subj_tr, gmean)
            prior_tr = np.array([prior_map.get(s, gmean) for s in subj_tr])

            Xtr_f = Xtr.copy()
            Xtr_f["subj_prior"] = prior_tr
            Xte_f = Xte.copy()
            Xte_f["subj_prior"] = np.array([prior_map.get(s, gmean) for s in subj_te])

            dtr = lgb.Dataset(Xtr_f.iloc[tr_idx], label=y[tr_idx],
                              categorical_feature=cat_feature, free_raw_data=False)
            dva = lgb.Dataset(Xtr_f.iloc[va_idx], label=y[va_idx],
                              categorical_feature=cat_feature, free_raw_data=False)
            model = lgb.train(
                LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            oof_model.iloc[va_idx, oof_model.columns.get_loc(t)] = model.predict(Xtr_f.iloc[va_idx])
            oof_prior.iloc[va_idx, oof_prior.columns.get_loc(t)] = prior_tr[va_idx]
            test_model[t] += model.predict(Xte_f) / N_SPLITS
            test_prior[t] += Xte_f["subj_prior"].values / N_SPLITS

    # 타깃별 블렌드 가중 탐색 (model 비중 w, prior 비중 1-w)
    weights, blended_oof = {}, pd.DataFrame(index=Xtr.index, columns=C.TARGET_COLS, dtype=float)
    for t in C.TARGET_COLS:
        best_w, best_ll = 0.0, 1e9
        for w in np.linspace(0, 1, 21):
            p = clip(w * oof_model[t] + (1 - w) * oof_prior[t])
            ll = log_loss(ytr[t].values, p, labels=[0, 1])
            if ll < best_ll:
                best_ll, best_w = ll, w
        weights[t] = best_w
        blended_oof[t] = w_blend = best_w * oof_model[t] + (1 - best_w) * oof_prior[t]

    model_score, model_per = avg_logloss(ytr, oof_model)
    prior_score, _ = avg_logloss(ytr, oof_prior)
    blend_score, blend_per = avg_logloss(ytr, blended_oof)

    print("\n=== OOF Average Log-Loss ===")
    print(f"{'tgt':4s} {'model':>8s} {'prior':>8s} {'w*':>5s} {'blend':>8s}")
    for t in C.TARGET_COLS:
        print(f"{t:4s} {model_per[t]:8.4f} {oof_prior_ll(ytr,oof_prior,t):8.4f} "
              f"{weights[t]:5.2f} {blend_per[t]:8.4f}")
    print("  ----")
    print(f"  prior-only : {prior_score:.4f}")
    print(f"  model-only : {model_score:.4f}")
    print(f"  BLEND      : {blend_score:.4f}")

    sub = mte.copy()
    for t in C.TARGET_COLS:
        w = weights[t]
        sub[t] = clip(w * test_model[t].values + (1 - w) * test_prior[t].values)
    out = C.SUBMISSION_DIR / f"v2_blend_oof{blend_score:.4f}.csv"
    sub.to_csv(out, index=False)
    print(f"\n  submission saved: {out}")
    return blend_score


def oof_prior_ll(ytr, oof_prior, t):
    return log_loss(ytr[t].values, clip(oof_prior[t].values), labels=[0, 1])


if __name__ == "__main__":
    train()
