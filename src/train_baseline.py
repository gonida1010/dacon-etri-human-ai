"""LightGBM 베이스라인: 타깃별 모델 + subject-stratified KFold OOF + 배깅 테스트 예측.

- 폴드별 모델로 OOF(검증) 예측을 모아 평균 Log-Loss(대회 산식)를 계산
- 테스트 예측은 폴드 모델들의 평균(배깅)으로 안정화
- 비교용: 피험자별 타깃 평균(개인 기준선) 예측의 Log-Loss도 출력
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
    max_depth=-1,
    verbosity=-1,
    seed=SEED,
)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def avg_logloss(y_true: pd.DataFrame, p: pd.DataFrame) -> tuple[float, dict]:
    per = {}
    for t in C.TARGET_COLS:
        per[t] = log_loss(y_true[t].values, clip(p[t].values), labels=[0, 1])
    return float(np.mean(list(per.values()))), per


def train():
    Xtr, ytr, Xte, mtr, mte, feat_cols = build_dataset(use_cache=True)
    folds = subject_stratified_folds(mtr, n_splits=N_SPLITS, seed=SEED)
    cat_feature = ["subject_id"]

    oof = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    test_pred = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)

    for t in C.TARGET_COLS:
        y = ytr[t].values
        for f in range(N_SPLITS):
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]
            dtr = lgb.Dataset(Xtr.iloc[tr_idx], label=y[tr_idx],
                              categorical_feature=cat_feature, free_raw_data=False)
            dva = lgb.Dataset(Xtr.iloc[va_idx], label=y[va_idx],
                              categorical_feature=cat_feature, free_raw_data=False)
            model = lgb.train(
                LGB_PARAMS, dtr, num_boost_round=2000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
            oof.iloc[va_idx, oof.columns.get_loc(t)] = model.predict(Xtr.iloc[va_idx])
            test_pred[t] += model.predict(Xte) / N_SPLITS

    score, per = avg_logloss(ytr, oof)

    # 비교 기준선: 피험자별 타깃 평균
    base = pd.DataFrame(index=Xtr.index, columns=C.TARGET_COLS, dtype=float)
    for t in C.TARGET_COLS:
        m = mtr.assign(y=ytr[t].values).groupby("subject_id")["y"].transform("mean")
        base[t] = m.values
    base_score, _ = avg_logloss(ytr, base)

    print("\n=== OOF Average Log-Loss (lower is better) ===")
    for t in C.TARGET_COLS:
        print(f"  {t}: {per[t]:.4f}")
    print(f"  ----")
    print(f"  LGBM  OOF avg log-loss : {score:.4f}")
    print(f"  Subject-mean baseline  : {base_score:.4f}")

    # 제출 파일
    sub = mte.copy()
    for t in C.TARGET_COLS:
        sub[t] = clip(test_pred[t].values)
    out = C.SUBMISSION_DIR / f"baseline_lgbm_oof{score:.4f}.csv"
    sub.to_csv(out, index=False)
    print(f"\n  submission saved: {out}")
    return score


if __name__ == "__main__":
    train()
