"""LGBM 센서 모델 + 타깃 시계열 prior 후처리 실험.

핵심 아이디어:
  - 기존 subject prior 는 피험자 전체 평균만 본다.
  - Q2/Q3/S2/S3/S4는 마지막 시간블록에서 최근 라벨 평균/시간 추세 prior가
    센서 모델보다 나은 구간이 있다.
  - 센서 모델 OOF와 temporal prior OOF를 타깃별 고정 recipe로 섞어 제출 파일을 만든다.

실행:
  python -m src.train_temporal_prior
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds


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
    seed=42,
)

# 마지막 시간블록에서 채택한 고정 recipe.
# 형식: target -> [(source, weight), ...], source 는 "model" 또는 temporal method.
RECIPES = {
    "Q1": [("model", 1.0)],
    "Q2": [("model", 0.30), ("last20_sm4", 0.70)],
    "Q3": [("model", 0.20), ("last2_sm4", 0.80)],
    "S1": [("model", 1.0)],
    "S2": [("model", 0.30), ("ridge10", 0.70)],
    "S3": [("mean_sm16", 1.0)],
    "S4": [("model", 0.50), ("ridge1", 0.50)],
}


def clip(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), C.PROB_CLIP, 1 - C.PROB_CLIP)


def ll(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    return log_loss(np.asarray(y), clip(p), labels=[0, 1])


def smoothed_mean(
    y: np.ndarray,
    idx: np.ndarray,
    subj: np.ndarray,
    gmean: float,
    smooth: float = C.PRIOR_SMOOTH,
) -> dict[str, float]:
    df = pd.DataFrame({"s": subj[idx], "y": y[idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + smooth * gmean) / (g["count"] + smooth)).to_dict()


def fit_lgbm_oof_test(
    Xtr: pd.DataFrame,
    ytr: pd.DataFrame,
    Xte: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subj_tr = mtr["subject_id"].values
    subj_te = mte["subject_id"].values

    oof_model = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    oof_prior = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    test_model = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)

    n_seed = len(C.SEEDS)
    for target in C.TARGET_COLS:
        y = ytr[target].values
        gmean = float(y.mean())
        for fold in range(C.N_SPLITS):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]

            pmap = smoothed_mean(y, tr_idx, subj_tr, gmean)
            prior_tr = np.array([pmap.get(s, gmean) for s in subj_tr])
            prior_te = np.array([pmap.get(s, gmean) for s in subj_te])

            Xtr_f = Xtr.copy()
            Xte_f = Xte.copy()
            Xtr_f["subj_prior"] = prior_tr
            Xte_f["subj_prior"] = prior_te

            dtr = lgb.Dataset(Xtr_f.iloc[tr_idx], label=y[tr_idx],
                              categorical_feature=["subject_id"], free_raw_data=False)
            dva = lgb.Dataset(Xtr_f.iloc[va_idx], label=y[va_idx],
                              categorical_feature=["subject_id"], free_raw_data=False)
            oof_prior.loc[va_idx, target] = prior_tr[va_idx]
            for seed in C.SEEDS:   # 멀티시드 배깅(단일시드 노이즈 제거 → Q1/S1 안정화)
                model = lgb.train({**LGB_PARAMS, "seed": seed}, dtr, num_boost_round=3000,
                                  valid_sets=[dva], callbacks=[lgb.early_stopping(80, verbose=False)])
                oof_model.loc[va_idx, target] += model.predict(Xtr_f.iloc[va_idx]) / n_seed
                test_model[target] += model.predict(Xte_f) / (C.N_SPLITS * n_seed)

    return oof_model, oof_prior, test_model


def _predict_temporal_one(
    train_df: pd.DataFrame,
    target: str,
    method: str,
    subject_id: str,
    sleep_date: pd.Timestamp,
    gmean: float,
) -> float:
    same_subject = train_df[train_df["subject_id"].eq(subject_id)].sort_values("sleep_date")
    history = same_subject[same_subject["sleep_date"] < sleep_date]

    if method.startswith("mean_sm"):
        smooth = float(method.replace("mean_sm", ""))
        y = same_subject[target]
        return float((y.sum() + smooth * gmean) / (len(y) + smooth)) if len(y) else gmean

    if method.startswith("last"):
        prefix, smooth_part = method.split("_sm")
        k = int(prefix.replace("last", ""))
        smooth = float(smooth_part)
        y = history[target].tail(k)
        return float((y.sum() + smooth * gmean) / (len(y) + smooth)) if len(y) else gmean

    if method.startswith("ridge"):
        alpha = float(method.replace("ridge", ""))
        if len(same_subject) < 8 or same_subject[target].nunique() < 2:
            y = history[target]
            return float((y.sum() + 8 * gmean) / (len(y) + 8)) if len(y) else gmean
        x = ((same_subject["sleep_date"] - same_subject["sleep_date"].min()).dt.days / 30.0)
        model = Ridge(alpha=alpha).fit(x.values.reshape(-1, 1), same_subject[target].values)
        xv = np.array([[(sleep_date - same_subject["sleep_date"].min()).days / 30.0]])
        base = (same_subject[target].sum() + 8 * gmean) / (len(same_subject) + 8)
        return float(0.5 * base + 0.5 * model.predict(xv)[0])

    raise ValueError(f"unknown temporal method: {method}")


def temporal_oof(
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    folds: np.ndarray,
    target: str,
    method: str,
) -> np.ndarray:
    meta = mtr.copy()
    meta[target] = ytr[target].values
    pred = np.zeros(len(meta), dtype=float)

    for fold in range(C.N_SPLITS):
        tr_df = meta.loc[folds != fold].copy()
        gmean = float(tr_df[target].mean())
        va_idx = np.where(folds == fold)[0]
        for i in va_idx:
            pred[i] = _predict_temporal_one(
                tr_df,
                target,
                method,
                str(meta.at[i, "subject_id"]),
                pd.Timestamp(meta.at[i, "sleep_date"]),
                gmean,
            )
    return pred


def temporal_test(
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    target: str,
    method: str,
) -> np.ndarray:
    train_df = mtr.copy()
    train_df[target] = ytr[target].values
    gmean = float(train_df[target].mean())
    pred = np.zeros(len(mte), dtype=float)
    for i, row in mte.iterrows():
        pred[i] = _predict_temporal_one(
            train_df,
            target,
            method,
            str(row["subject_id"]),
            pd.Timestamp(row["sleep_date"]),
            gmean,
        )
    return pred


def build_recipe_predictions(
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    oof_model: pd.DataFrame,
    test_model: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_final = pd.DataFrame(index=oof_model.index, columns=C.TARGET_COLS, dtype=float)
    test_final = pd.DataFrame(index=test_model.index, columns=C.TARGET_COLS, dtype=float)

    temporal_cache_oof: dict[tuple[str, str], np.ndarray] = {}
    temporal_cache_test: dict[tuple[str, str], np.ndarray] = {}

    for target, parts in RECIPES.items():
        oof = np.zeros(len(oof_model), dtype=float)
        test = np.zeros(len(test_model), dtype=float)
        for source, weight in parts:
            if source == "model":
                oof += weight * oof_model[target].values
                test += weight * test_model[target].values
                continue
            key = (target, source)
            if key not in temporal_cache_oof:
                temporal_cache_oof[key] = temporal_oof(ytr, mtr, folds, target, source)
                temporal_cache_test[key] = temporal_test(ytr, mtr, mte, target, source)
            oof += weight * temporal_cache_oof[key]
            test += weight * temporal_cache_test[key]

        oof_final[target] = clip(oof)
        test_final[target] = clip(test)

    return oof_final, test_final


def report_scores(ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> tuple[float, float]:
    last = folds == (C.N_SPLITS - 1)
    full_scores = []
    last_scores = []
    print("\n=== Temporal Prior Blend OOF Log-Loss ===")
    print(f"{'tgt':4s} {'recipe':32s} {'full':>8s} {'last':>8s}")
    for target in C.TARGET_COLS:
        full = ll(ytr[target].values, pred[target].values)
        last_score = ll(ytr[target].values[last], pred[target].values[last])
        recipe = " + ".join(f"{w:.2f}*{src}" for src, w in RECIPES[target])
        print(f"{target:4s} {recipe:32s} {full:8.4f} {last_score:8.4f}")
        full_scores.append(full)
        last_scores.append(last_score)
    full_mean = float(np.mean(full_scores))
    last_mean = float(np.mean(last_scores))
    print("  ----")
    print(f"  BLEND full      : {full_mean:.4f}")
    print(f"  BLEND last-block: {last_mean:.4f}   <- headline CV")
    return full_mean, last_mean


def train() -> float:
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)

    oof_model, _, test_model = fit_lgbm_oof_test(Xtr, ytr, Xte, mtr, mte, folds)
    oof_final, test_final = build_recipe_predictions(
        ytr, mtr, mte, folds, oof_model, test_model
    )
    _, last_score = report_scores(ytr, oof_final, folds)

    submission = mte.copy()
    for target in C.TARGET_COLS:
        submission[target] = test_final[target].values
    out = C.SUBMISSION_DIR / f"submission_temporal_prior_last{last_score:.4f}.csv"
    submission.to_csv(out, index=False)
    print(f"\nsaved: {out}")
    return last_score


if __name__ == "__main__":
    train()
