"""정식 학습 파이프라인 (최종 제출 생성).

구성:
 1) 피처: 일일 윈도우 통계 + 수면구간 탐지 + 중첩센서 + 피험자 z-score + 캘린더
 2) 타깃별 LGBM + 누수 없는 OOF 피험자 prior 피처
 3) 멀티시드 배깅(폴드/모델 시드)으로 OOF·테스트 예측 안정화
 4) 타깃별 (모델 vs prior) OOF 블렌드 가중 탐색
 5) 평균 Log-Loss(대회 산식) 보고 + 제출 CSV 저장

실행: python -m src.train
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
SEEDS = [42, 7, 2024]
SMOOTH = 8.0

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=1.0, verbosity=-1,
)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def avg_logloss(y_true, p):
    per = {t: log_loss(y_true[t].values, clip(p[t].values), labels=[0, 1]) for t in C.TARGET_COLS}
    return float(np.mean(list(per.values()))), per


def smoothed_mean(y, idx, subj, gmean):
    df = pd.DataFrame({"s": subj[idx], "y": y[idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + SMOOTH * gmean) / (g["count"] + SMOOTH)).to_dict()


def train():
    Xtr, ytr, Xte, mtr, mte, feat_cols = build_dataset(use_cache=True)
    subj_tr, subj_te = mtr["subject_id"].values, mte["subject_id"].values
    cat = ["subject_id"]

    oof_model = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    oof_prior = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    test_model = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)
    test_prior = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)

    for t in C.TARGET_COLS:
        y = ytr[t].values
        gmean = y.mean()
        for seed in SEEDS:
            folds = subject_stratified_folds(mtr, n_splits=N_SPLITS, seed=seed)
            params = {**LGB_PARAMS, "seed": seed}
            for f in range(N_SPLITS):
                tr_idx = np.where(folds != f)[0]
                va_idx = np.where(folds == f)[0]
                pmap = smoothed_mean(y, tr_idx, subj_tr, gmean)
                prior_tr = np.array([pmap.get(s, gmean) for s in subj_tr])
                prior_te = np.array([pmap.get(s, gmean) for s in subj_te])

                Xtr_f = Xtr.copy(); Xtr_f["subj_prior"] = prior_tr
                Xte_f = Xte.copy(); Xte_f["subj_prior"] = prior_te
                dtr = lgb.Dataset(Xtr_f.iloc[tr_idx], label=y[tr_idx],
                                  categorical_feature=cat, free_raw_data=False)
                dva = lgb.Dataset(Xtr_f.iloc[va_idx], label=y[va_idx],
                                  categorical_feature=cat, free_raw_data=False)
                m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(80, verbose=False)])
                col = oof_model.columns.get_loc(t)
                oof_model.iloc[va_idx, col] += m.predict(Xtr_f.iloc[va_idx]) / len(SEEDS)
                oof_prior.iloc[va_idx, col] += prior_tr[va_idx] / len(SEEDS)
                test_model[t] += m.predict(Xte_f) / (len(SEEDS) * N_SPLITS)
                test_prior[t] += prior_te / (len(SEEDS) * N_SPLITS)

    weights, blended = {}, pd.DataFrame(index=Xtr.index, columns=C.TARGET_COLS, dtype=float)
    for t in C.TARGET_COLS:
        best_w, best_ll = 0.0, 1e9
        for w in np.linspace(0, 1, 41):
            ll = log_loss(ytr[t].values, clip(w * oof_model[t] + (1 - w) * oof_prior[t]), labels=[0, 1])
            if ll < best_ll:
                best_ll, best_w = ll, w
        weights[t] = best_w
        blended[t] = best_w * oof_model[t] + (1 - best_w) * oof_prior[t]

    _, model_per = avg_logloss(ytr, oof_model)
    prior_score, _ = avg_logloss(ytr, oof_prior)
    model_score, _ = avg_logloss(ytr, oof_model)
    blend_score, blend_per = avg_logloss(ytr, blended)

    print("\n=== OOF Average Log-Loss (multi-seed bagged) ===")
    print(f"{'tgt':4s} {'model':>8s} {'w*':>5s} {'blend':>8s}")
    for t in C.TARGET_COLS:
        print(f"{t:4s} {model_per[t]:8.4f} {weights[t]:5.2f} {blend_per[t]:8.4f}")
    print("  ----")
    print(f"  prior-only : {prior_score:.4f}")
    print(f"  model-only : {model_score:.4f}")
    print(f"  BLEND      : {blend_score:.4f}")

    sub = mte.copy()
    for t in C.TARGET_COLS:
        w = weights[t]
        sub[t] = clip(w * test_model[t].values + (1 - w) * test_prior[t].values)
    out = C.SUBMISSION_DIR / f"submission_oof{blend_score:.4f}.csv"
    sub.to_csv(out, index=False)
    print(f"\n  submission saved: {out}")
    return blend_score


if __name__ == "__main__":
    train()
