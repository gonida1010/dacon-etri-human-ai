"""정식 학습 파이프라인 (최종 제출 생성).

 - 피처: 일일 윈도우 통계 + 수면구간 탐지 + 중첩센서 + 시간동역학 + 피험자 z-score + 캘린더
 - 타깃별 LGBM + 누수 없는 OOF 피험자 prior 피처, 멀티시드 배깅
 - 타깃별 (모델 vs prior) OOF 블렌드 가중 자동 탐색
 - 정직한 시간블록 CV의 'last-block'(=LB와 일치) 점수 보고 + 제출 CSV 1개 저장

손잡이는 config.py 참조. 실행: python -m src.train
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
N_SPLITS = C.N_SPLITS
SEEDS = C.SEEDS
SMOOTH = C.PRIOR_SMOOTH

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=1.0, verbosity=-1,
)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def sharpen(p):
    """확률 샤프닝(온도 T). T<1: 0/1 쪽으로 더 극단. T=1: 그대로. config.SHARPEN_T."""
    T = C.SHARPEN_T
    if T == 1.0:
        return p
    z = np.log(clip(p) / (1 - clip(p))) / T
    return 1.0 / (1.0 + np.exp(-z))


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

    folds = subject_time_blocked_folds(mtr, n_splits=N_SPLITS)  # 정직한 시간블록(결정적)
    for t in C.TARGET_COLS:
        y = ytr[t].values
        gmean = y.mean()
        for seed in SEEDS:
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

    # 타깃별 (모델 vs prior) 블렌드 가중 자동 탐색
    weights, blended, test_blend = {}, pd.DataFrame(index=Xtr.index, columns=C.TARGET_COLS, dtype=float), {}
    for t in C.TARGET_COLS:
        best_w, best_ll = 0.0, 1e9
        for w in np.linspace(0, 1, 41):
            ll = log_loss(ytr[t].values, clip(w * oof_model[t] + (1 - w) * oof_prior[t]), labels=[0, 1])
            if ll < best_ll:
                best_ll, best_w = ll, w
        best_w = max(best_w, C.MODEL_WEIGHT_FLOOR)   # 모델 비중 하한(config 손잡이)
        weights[t] = best_w
        blended[t] = sharpen(best_w * oof_model[t] + (1 - best_w) * oof_prior[t])
        test_blend[t] = sharpen(best_w * test_model[t].values + (1 - best_w) * test_prior[t].values)

    prior_score, _ = avg_logloss(ytr, oof_prior)
    model_score, model_per = avg_logloss(ytr, oof_model)
    blend_score, blend_per = avg_logloss(ytr, blended)
    last = folds == (N_SPLITS - 1)
    blend_last = float(np.mean([
        log_loss(ytr[t].values[last], clip(blended[t].values[last]), labels=[0, 1])
        for t in C.TARGET_COLS]))

    print("\n=== OOF Average Log-Loss (multi-seed bagged) ===")
    print(f"{'tgt':4s} {'model':>8s} {'w*':>5s} {'blend':>8s}")
    for t in C.TARGET_COLS:
        print(f"{t:4s} {model_per[t]:8.4f} {weights[t]:5.2f} {blend_per[t]:8.4f}")
    print("  ----")
    print(f"  prior-only      : {prior_score:.4f}")
    print(f"  model-only      : {model_score:.4f}")
    print(f"  BLEND full      : {blend_score:.4f}")
    print(f"  BLEND last-block: {blend_last:.4f}   <- 진짜 점수(LB와 일치, 낮을수록 좋음)")

    sub = mte.copy()
    for t in C.TARGET_COLS:
        sub[t] = clip(test_blend[t])
    out = C.SUBMISSION_DIR / f"submission_last{blend_last:.4f}.csv"
    sub.to_csv(out, index=False)
    print(f"\n  submission saved: {out.name}")
    return blend_last


if __name__ == "__main__":
    train()
