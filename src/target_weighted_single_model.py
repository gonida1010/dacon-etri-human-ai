"""Target-weighted single-model source optimizer.

This is the next modeling layer before stacking/blending.  It creates new OOF
and test prediction sources from one model family while explicitly searching the
training mechanics that matter for this small temporal Dacon dataset:

- target-wise sample weighting: subject balance, class balance, recency,
  late-fold emphasis, and anchor-error emphasis
- fold-safe target/history features
- Kaggle-style numeric bin views and fold-safe smoothed target encoding
- validation-logloss early stopping with best-iteration prediction
- target-wise shrink back to the temporal anchor

The output files follow the existing `*_oof.csv` / `*_test_pred.csv` convention
so they can be fed directly into `public_aware_stack_blend.py` or
`public_score_pseudo_blend.py` as fresh single-model sources.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .residual_single_model_opt import (
    digitize_codes,
    loo_smooth_target_encode,
    quantile_edges,
    recency_fraction,
    weighted_corr,
)
from .train_temporal_prior import build_recipe_predictions, clip, fit_lgbm_oof_test, temporal_oof, temporal_test

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
EPS = C.PROB_CLIP


@dataclass(frozen=True)
class WeightProfile:
    name: str
    subject_power: float
    class_power: float
    recency_strength: float
    anchor_error_power: float
    fold_strength: float


WEIGHT_PROFILES: dict[str, WeightProfile] = {
    "uniform": WeightProfile("uniform", 0.0, 0.0, 0.0, 0.0, 0.0),
    "subject_class": WeightProfile("subject_class", 1.0, 0.8, 0.0, 0.0, 0.0),
    "recent_class": WeightProfile("recent_class", 1.0, 0.8, 1.2, 0.0, 0.35),
    "recent_anchorerr": WeightProfile("recent_anchorerr", 1.0, 0.7, 1.7, 0.8, 0.55),
    "late_anchorerr": WeightProfile("late_anchorerr", 0.8, 0.6, 2.2, 1.0, 0.80),
}


LGBM_PROFILES = {
    "smooth": {
        "learning_rate": 0.018,
        "num_leaves": 7,
        "min_child_samples": 34,
        "feature_fraction": 0.62,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 1.2,
        "lambda_l2": 8.0,
    },
    "mid": {
        "learning_rate": 0.016,
        "num_leaves": 15,
        "min_child_samples": 24,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.84,
        "bagging_freq": 1,
        "lambda_l1": 0.8,
        "lambda_l2": 4.0,
    },
    "leaf31": {
        "learning_rate": 0.012,
        "num_leaves": 31,
        "min_child_samples": 18,
        "feature_fraction": 0.76,
        "bagging_fraction": 0.86,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 6.0,
    },
}


XGB_PROFILES = {
    "smooth": {
        "learning_rate": 0.018,
        "max_depth": 2,
        "min_child_weight": 12,
        "subsample": 0.82,
        "colsample_bytree": 0.62,
        "reg_alpha": 1.2,
        "reg_lambda": 8.0,
    },
    "mid": {
        "learning_rate": 0.016,
        "max_depth": 3,
        "min_child_weight": 8,
        "subsample": 0.84,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.8,
        "reg_lambda": 5.0,
    },
}


CAT_PROFILES = {
    "smooth": {
        "learning_rate": 0.018,
        "depth": 3,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.4,
        "subsample": 0.82,
    },
    "mid": {
        "learning_rate": 0.016,
        "depth": 4,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.1,
        "subsample": 0.84,
    },
}


HISTORY_METHODS = ["mean_sm4", "mean_sm16", "last2_sm4", "last5_sm4", "last20_sm4", "ridge1", "ridge10"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def write_prediction_frame(path: Path, meta: pd.DataFrame, pred: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
    out = meta.reset_index(drop=True).copy()
    if y is not None:
        for target in TARGETS:
            out[f"label__{target}"] = y[target].values
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test.reset_index(drop=True).copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_anchor_bank(bank_dir: Path, n_train: int, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    oof_path = bank_dir / "oof_bank.csv"
    test_path = bank_dir / "test_bank.csv"
    if not oof_path.exists() or not test_path.exists():
        return None
    oof_bank = pd.read_csv(oof_path)
    test_bank = pd.read_csv(test_path)
    needed_oof = [f"anchor__{target}" for target in TARGETS]
    needed_test = [f"anchor__{target}" for target in TARGETS]
    if any(col not in oof_bank.columns for col in needed_oof) or any(col not in test_bank.columns for col in needed_test):
        return None
    anchor_oof = pd.DataFrame({target: clip(oof_bank[f"anchor__{target}"].values) for target in TARGETS})
    anchor_test = pd.DataFrame({target: clip(test_bank[f"anchor__{target}"].values) for target in TARGETS})
    if len(anchor_oof) != n_train or len(anchor_test) != n_test:
        return None
    return anchor_oof, anchor_test


def model_profile_dict(model_name: str) -> dict[str, dict]:
    if model_name == "lgbm":
        return LGBM_PROFILES
    if model_name == "xgb":
        return XGB_PROFILES
    if model_name == "cat":
        return CAT_PROFILES
    raise ValueError(model_name)


def target_auto_profile(target: str) -> WeightProfile:
    if target in {"Q2", "Q3"}:
        return WeightProfile("target_auto", 1.0, 0.8, 1.4, 0.5, 0.45)
    if target in {"S2", "S4"}:
        return WeightProfile("target_auto", 1.0, 0.7, 2.0, 0.9, 0.70)
    if target == "S3":
        return WeightProfile("target_auto", 0.8, 0.45, 0.4, 0.2, 0.20)
    return WeightProfile("target_auto", 1.0, 0.65, 0.8, 0.4, 0.35)


def resolve_weight_profile(name: str, target: str) -> WeightProfile:
    if name == "target_auto":
        return target_auto_profile(target)
    return WEIGHT_PROFILES[name]


def add_subject_time_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    both_meta = pd.concat([mtr, mte], ignore_index=True).copy()
    both_meta["sleep_date"] = pd.to_datetime(both_meta["sleep_date"])
    frames = []
    for subject, g in both_meta.sort_values(["subject_id", "sleep_date"]).groupby("subject_id", sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        part = pd.DataFrame(index=idx)
        part["subject_day_idx"] = np.arange(n, dtype=float)
        part["subject_day_frac"] = np.linspace(0.0, 1.0, n) if n > 1 else 1.0
        part["days_since_subject_start"] = (g["sleep_date"] - g["sleep_date"].min()).dt.days.astype(float).values
        part["days_until_subject_end"] = (g["sleep_date"].max() - g["sleep_date"]).dt.days.astype(float).values
        part["subject_rows_total"] = float(n)
        frames.append(part)
    time_all = pd.concat(frames).sort_index()
    n_train = len(mtr)
    cols = time_all.columns.tolist()
    return (
        pd.concat([Xtr.reset_index(drop=True), time_all.iloc[:n_train].reset_index(drop=True)], axis=1),
        pd.concat([Xte.reset_index(drop=True), time_all.iloc[n_train:].reset_index(drop=True)], axis=1),
        cols,
    )


def add_anchor_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    cols: list[str] = []
    for target in TARGETS:
        p_tr = clip(anchor_oof[target].values)
        p_te = clip(anchor_test[target].values)
        Xtr2[f"anchor_{target}_prob"] = p_tr
        Xte2[f"anchor_{target}_prob"] = p_te
        Xtr2[f"anchor_{target}_logit"] = logit(p_tr)
        Xte2[f"anchor_{target}_logit"] = logit(p_te)
        cols += [f"anchor_{target}_prob", f"anchor_{target}_logit"]
    for name, targets in {"q": ["Q1", "Q2", "Q3"], "s": ["S1", "S2", "S3", "S4"], "all": TARGETS}.items():
        Xtr2[f"anchor_{name}_mean"] = anchor_oof[targets].mean(axis=1).values
        Xte2[f"anchor_{name}_mean"] = anchor_test[targets].mean(axis=1).values
        Xtr2[f"anchor_{name}_std"] = anchor_oof[targets].std(axis=1).values
        Xte2[f"anchor_{name}_std"] = anchor_test[targets].std(axis=1).values
        cols += [f"anchor_{name}_mean", f"anchor_{name}_std"]
    return Xtr2, Xte2, cols


def add_window_pair_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, max_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if max_pairs <= 0:
        return Xtr, Xte, []
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    pairs = []
    for l_col in [c for c in Xtr.columns if c.startswith("L_") and pd.api.types.is_numeric_dtype(Xtr[c])]:
        suffix = l_col[2:]
        s_col = f"S_{suffix}"
        if s_col in Xtr.columns and pd.api.types.is_numeric_dtype(Xtr[s_col]):
            coverage = float(Xtr[[l_col, s_col]].notna().mean().mean())
            pairs.append((coverage, suffix, l_col, s_col))
    pairs = sorted(pairs, reverse=True)[:max_pairs]
    added = []
    for _, suffix, l_col, s_col in pairs:
        safe = "".join(ch if ch.isalnum() else "_" for ch in suffix)[:80]
        d_col = f"pair_delta_{safe}"
        r_col = f"pair_ratio_{safe}"
        Xtr2[d_col] = Xtr2[s_col] - Xtr2[l_col]
        Xte2[d_col] = Xte2[s_col] - Xte2[l_col]
        Xtr2[r_col] = Xtr2[s_col] / (Xtr2[l_col].abs() + 1e-3)
        Xte2[r_col] = Xte2[s_col] / (Xte2[l_col].abs() + 1e-3)
        added.extend([d_col, r_col])
    return Xtr2, Xte2, added


def add_target_history_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    cols = []
    for method in HISTORY_METHODS:
        col = f"hist_{target}_{method}"
        Xtr2[col] = temporal_oof(ytr, mtr, folds, target, method)
        Xte2[col] = temporal_test(ytr, mtr, mte, target, method)
        cols.append(col)
    return Xtr2, Xte2, cols


def numeric_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(X[c])]


def sample_weights(
    y: np.ndarray,
    anchor: np.ndarray,
    meta: pd.DataFrame,
    folds: np.ndarray,
    recency: np.ndarray,
    train_idx: np.ndarray,
    profile: WeightProfile,
) -> np.ndarray:
    idx = np.asarray(train_idx)
    w = np.ones(len(idx), dtype=float)
    if profile.subject_power:
        counts = meta.iloc[idx]["subject_id"].map(meta.iloc[idx]["subject_id"].value_counts()).astype(float).values
        subj_w = len(idx) / np.maximum(counts, 1.0)
        w *= subj_w ** profile.subject_power
    if profile.class_power:
        yt = y[idx].astype(int)
        pos = float(np.clip(yt.mean(), 0.05, 0.95))
        cls_w = np.where(yt == 1, 0.5 / pos, 0.5 / (1.0 - pos))
        w *= cls_w ** profile.class_power
    if profile.recency_strength:
        w *= np.exp(profile.recency_strength * (recency[idx] - 1.0))
    if profile.anchor_error_power:
        err = np.abs(y[idx].astype(float) - clip(anchor[idx]))
        w *= (0.25 + err) ** profile.anchor_error_power
    if profile.fold_strength:
        fold_frac = folds[idx].astype(float) / max(float(C.N_SPLITS - 1), 1.0)
        w *= 1.0 + profile.fold_strength * fold_frac
    w = np.clip(w, 0.03, 30.0)
    return w / max(float(w.mean()), 1e-12)


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
    always_cols: list[str],
    weight: np.ndarray | None,
) -> list[str]:
    always = [c for c in always_cols if c in X.columns]
    candidates = numeric_cols(X)
    if top_k <= 0 or top_k >= len(candidates):
        cols = candidates
    else:
        scored = []
        for col in candidates:
            if col in always:
                score = np.inf
            else:
                score = weighted_corr(X[col].values[train_idx], y[train_idx], weight)
            scored.append((score, col))
        cols = [c for _, c in sorted(scored, reverse=True)[:top_k]]
    for col in always:
        if col not in cols and col in X.columns:
            cols.append(col)
    if "subject_id" in X.columns:
        cols.append("subject_id")
    return list(dict.fromkeys(cols))


def select_te_source_cols(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    top_n: int,
    exclude: list[str],
) -> list[str]:
    if top_n <= 0:
        return []
    exclude_set = set(exclude)
    scored = []
    for col in numeric_cols(X):
        if col in exclude_set or col.startswith("anchor_"):
            continue
        vals = X[col].values
        finite = np.isfinite(vals[train_idx])
        if finite.sum() < 20:
            continue
        if len(np.unique(vals[train_idx][finite])) < 5:
            continue
        score = weighted_corr(vals[train_idx], y[train_idx])
        if score > 0:
            scored.append((score, col))
    return [c for _, c in sorted(scored, reverse=True)[:top_n]]


def add_fold_feature_bank(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    Xte_full: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    source_cols: list[str],
    bins: list[int],
    smooth: float,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    if mode == "none" or not source_cols:
        return X_train, X_valid, X_test, {"bank_source_count": 0, "bank_feature_count": 0}, []
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    cat_cols: list[str] = []
    added = 0
    for col in source_cols:
        train_vals = X_full[col].values[tr_idx]
        valid_vals = X_full[col].values[va_idx]
        test_vals = Xte_full[col].values
        safe = "".join(ch if ch.isalnum() else "_" for ch in col)[:80]
        for n_bins in bins:
            edges = quantile_edges(train_vals, int(n_bins))
            if edges is None:
                continue
            tr_code = digitize_codes(train_vals, edges)
            va_code = digitize_codes(valid_vals, edges)
            te_code = digitize_codes(test_vals, edges)
            if mode in {"bins", "bins_te"}:
                bname = f"bin_{safe}_{n_bins}"
                X_train[bname] = tr_code.astype(int)
                X_valid[bname] = va_code.astype(int)
                X_test[bname] = te_code.astype(int)
                cat_cols.append(bname)
                added += 1
            if mode in {"te", "bins_te"}:
                tr_te, va_te, te_te = loo_smooth_target_encode(tr_code, va_code, te_code, y[tr_idx], smooth)
                tname = f"te_{safe}_{n_bins}"
                X_train[tname] = tr_te
                X_valid[tname] = va_te
                X_test[tname] = te_te
                added += 1
    return X_train, X_valid, X_test, {"bank_source_count": len(source_cols), "bank_feature_count": added}, cat_cols


def prepare_matrix(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    cats = list(dict.fromkeys(["subject_id", *cat_cols]))
    cats = [c for c in cats if c in X_train.columns]
    if model_name == "cat":
        for df in (X_train, X_valid, X_test):
            for col in cats:
                df[col] = df[col].astype(str).fillna("__NA__")
    else:
        for df in (X_train, X_valid, X_test):
            for col in cats:
                df[col] = df[col].astype("category")
    return X_train, X_valid, X_test, cats


def fit_model(
    model_name: str,
    profile_name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    seed: int,
    rounds: int,
    early_stopping_rounds: int,
    log_period: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if model_name == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        dtr = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=cat_cols or "auto", free_raw_data=False)
        dva = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols or "auto", free_raw_data=False)
        model = lgb.train(
            {
                **params,
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "seed": seed,
            },
            dtr,
            num_boost_round=rounds,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(log_period)],
        )
        best_iter = int(model.best_iteration or model.current_iteration())
        meta = {
            "best_iteration": best_iter,
            "used_iteration": best_iter,
            "best_valid_logloss": float(model.best_score["valid"]["binary_logloss"]),
            "best_train_logloss": float(model.best_score["train"]["binary_logloss"]),
            "stop_policy": "validation_logloss_early_stopping",
            "fallback_logic": "best_iteration or current_iteration",
        }
        return clip(model.predict(X_valid, num_iteration=best_iter)), clip(model.predict(X_test, num_iteration=best_iter)), meta

    if model_name == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=rounds,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=log_period,
        )
        best_iter = int(getattr(model, "best_iteration", 0) or 0)
        kwargs = {"iteration_range": (0, best_iter + 1)} if best_iter > 0 else {}
        evals = model.evals_result()
        valid_curve = evals.get("validation_1", {}).get("logloss", [])
        train_curve = evals.get("validation_0", {}).get("logloss", [])
        meta = {
            "best_iteration": best_iter,
            "used_iteration": best_iter,
            "best_valid_logloss": float(valid_curve[best_iter]) if valid_curve and best_iter < len(valid_curve) else np.nan,
            "best_train_logloss": float(train_curve[best_iter]) if train_curve and best_iter < len(train_curve) else np.nan,
            "stop_policy": "validation_logloss_early_stopping",
            "fallback_logic": "XGBoost best_iteration via iteration_range; all trees if best_iteration missing",
        }
        return clip(model.predict_proba(X_valid, **kwargs)[:, 1]), clip(model.predict_proba(X_test, **kwargs)[:, 1]), meta

    if model_name == "cat":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        model = CatBoostClassifier(
            **params,
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=rounds,
            random_seed=seed,
            early_stopping_rounds=early_stopping_rounds,
            bootstrap_type="Bernoulli",
            verbose=log_period,
            allow_writing_files=False,
            cat_features=cat_cols,
        )
        model.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_valid, y_valid), use_best_model=True)
        best_iter = int(model.get_best_iteration() or model.tree_count_)
        best_score = model.get_best_score()
        meta = {
            "best_iteration": best_iter,
            "used_iteration": best_iter,
            "best_valid_logloss": float(best_score.get("validation", {}).get("Logloss", np.nan)),
            "best_train_logloss": float(best_score.get("learn", {}).get("Logloss", np.nan)),
            "stop_policy": "validation_logloss_early_stopping_use_best_model",
            "fallback_logic": "get_best_iteration or tree_count_",
        }
        return clip(model.predict_proba(X_valid)[:, 1]), clip(model.predict_proba(X_test)[:, 1]), meta

    raise ValueError(model_name)


def shrink_prediction(anchor: np.ndarray, pred: np.ndarray, shrink: float, mode: str) -> np.ndarray:
    anchor = clip(anchor)
    pred = clip(pred)
    if mode == "prob":
        return clip((1.0 - shrink) * anchor + shrink * pred)
    if mode == "logit":
        return clip(sigmoid((1.0 - shrink) * logit(anchor) + shrink * logit(pred)))
    raise ValueError(mode)


def fold_losses(y: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> list[float]:
    return [mean_loss(y, pred, folds == f) for f in sorted(np.unique(folds))]


def target_rank_score(full_loss: float, last_loss: float, anchor_full: float, fold_vals: list[float]) -> float:
    tail3 = fold_vals[-3:]
    return last_loss + 0.85 * max(0.0, full_loss - anchor_full) + 0.18 * max(0.0, max(tail3) - float(np.mean(tail3)))


def build_candidate(
    name: str,
    mode: str,
    search: pd.DataFrame,
    raw_oof: dict[str, dict[str, np.ndarray]],
    raw_test: dict[str, dict[str, np.ndarray]],
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = pd.DataFrame(index=anchor_oof.index, columns=TARGETS, dtype=float)
    out_test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    choices = []
    for target in TARGETS:
        rows = search[search["target"].eq(target)].copy()
        if mode == "last":
            rows = rows.sort_values(["last_logloss", "full_logloss", "rank_score"])
        elif mode == "full":
            rows = rows.sort_values(["full_logloss", "last_logloss", "rank_score"])
        elif mode == "composite":
            rows = rows.sort_values(["rank_score", "last_logloss", "full_logloss"])
        else:
            raise ValueError(mode)
        row = rows.iloc[0].to_dict()
        cfg = str(row["config_key"])
        shrink = float(row["shrink"])
        shrink_mode = str(row["shrink_mode"])
        out_oof[target] = shrink_prediction(anchor_oof[target].values, raw_oof[target][cfg], shrink, shrink_mode)
        out_test[target] = shrink_prediction(anchor_test[target].values, raw_test[target][cfg], shrink, shrink_mode)
        row["candidate"] = name
        row["selection_mode"] = mode
        choices.append(row)
    return out_oof, out_test, pd.DataFrame(choices)


def score_candidate(name: str, ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray, anchor_full: float, anchor_last: float) -> dict:
    vals = fold_losses(ytr, pred, folds)
    full = mean_loss(ytr, pred, np.ones(len(ytr), dtype=bool))
    last = vals[-1]
    return {
        "candidate": name,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "fold_std": float(np.std(vals)),
        "tail3_mean": float(np.mean(vals[-3:])),
        "tail3_worst": float(np.max(vals[-3:])),
        "rank_score": last + 0.85 * max(0.0, full - anchor_full) + 0.18 * max(0.0, max(vals[-3:]) - float(np.mean(vals[-3:]))),
        **{f"fold{i}_logloss": val for i, val in enumerate(vals)},
    }


def plot_candidate_scores(out_dir: Path, scores: pd.DataFrame) -> None:
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.scatter(scores["full_logloss"], scores["last_logloss"], s=52, alpha=0.85)
    for _, row in scores.iterrows():
        ax.annotate(str(row["candidate"]), (row["full_logloss"], row["last_logloss"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("OOF full logloss")
    ax.set_ylabel("OOF last-block logloss")
    ax.set_title("Target-weighted single-model sources")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_scores.png", dpi=160)
    fig.savefig(out_dir / "candidate_scores.svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/target_weighted_single_model")
    p.add_argument("--submission-dir", default="submissions/target_weighted_single_model")
    p.add_argument("--anchor-bank-dir", default="research/oof_sparse_greedy")
    p.add_argument("--models", nargs="+", default=["lgbm"], choices=["lgbm", "xgb", "cat"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    p.add_argument("--fold-limit", type=int, default=5)
    p.add_argument("--param-profiles", nargs="*", default=["smooth", "mid", "leaf31"])
    p.add_argument("--weight-profiles", nargs="*", default=["uniform", "subject_class", "recent_class", "recent_anchorerr", "target_auto"])
    p.add_argument("--top-k-grid", nargs="*", type=int, default=[60, 100, 160, 240])
    p.add_argument("--feature-bank", choices=["none", "bins", "te", "bins_te"], default="bins_te")
    p.add_argument("--te-top-n", type=int, default=10)
    p.add_argument("--te-bins", nargs="*", type=int, default=[4, 8])
    p.add_argument("--te-smooth", type=float, default=12.0)
    p.add_argument("--pair-feature-count", type=int, default=80)
    p.add_argument("--target-history-features", action="store_true")
    p.add_argument("--rounds", type=int, default=2600)
    p.add_argument("--early-stopping-rounds", type=int, default=140)
    p.add_argument("--shrink-grid", nargs="*", type=float, default=[0.0, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 1.0])
    p.add_argument("--shrink-modes", nargs="*", default=["logit", "prob"], choices=["logit", "prob"])
    p.add_argument("--full-guard", type=float, default=0.020)
    p.add_argument("--save-top-raw", type=int, default=8)
    p.add_argument("--log-period", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading data and temporal anchor")
    Xtr_raw, ytr, Xte_raw, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    active_folds = sorted(np.unique(folds))[: args.fold_limit]
    last_mask = folds == (C.N_SPLITS - 1)
    bank = load_anchor_bank(ROOT / args.anchor_bank_dir, len(ytr), len(mte))
    if bank is None:
        log(f"Anchor bank missing under {args.anchor_bank_dir}; rebuilding temporal anchor")
        model_oof, _, model_test = fit_lgbm_oof_test(Xtr_raw, ytr, Xte_raw, mtr, mte, folds)
        anchor_oof, anchor_test = build_recipe_predictions(ytr, mtr, mte, folds, model_oof, model_test)
    else:
        anchor_oof, anchor_test = bank
        log(f"Loaded temporal anchor from {args.anchor_bank_dir}")
    anchor_full = mean_loss(ytr, anchor_oof, np.ones(len(ytr), dtype=bool))
    anchor_last = mean_loss(ytr, anchor_oof, last_mask)
    log(f"anchor full={anchor_full:.6f} last={anchor_last:.6f}")

    Xtr_base, Xte_base, time_cols = add_subject_time_features(Xtr_raw, Xte_raw, mtr, mte)
    Xtr_base, Xte_base, anchor_cols = add_anchor_features(Xtr_base, Xte_base, anchor_oof, anchor_test)
    Xtr_base, Xte_base, pair_cols = add_window_pair_features(Xtr_base, Xte_base, args.pair_feature_count)
    always_base = [c for c in [*time_cols, *anchor_cols, *pair_cols] if c in Xtr_base.columns]
    recency = recency_fraction(mtr)

    raw_oof: dict[str, dict[str, np.ndarray]] = {t: {} for t in TARGETS}
    raw_test: dict[str, dict[str, np.ndarray]] = {t: {} for t in TARGETS}
    fit_rows: list[dict] = []
    search_rows: list[dict] = []

    for target in TARGETS:
        y = ytr[target].values.astype(int)
        Xtr_target = Xtr_base
        Xte_target = Xte_base
        hist_cols: list[str] = []
        if args.target_history_features:
            Xtr_target, Xte_target, hist_cols = add_target_history_features(Xtr_base, Xte_base, ytr, mtr, mte, folds, target)
        target_anchor_full = safe_loss(y, anchor_oof[target].values)
        target_anchor_last = safe_loss(y[last_mask], anchor_oof[target].values[last_mask])
        log(f"Training sources target={target} anchor_full={target_anchor_full:.6f} anchor_last={target_anchor_last:.6f}")
        for model_name in args.models:
            profile_map = model_profile_dict(model_name)
            for profile_name in args.param_profiles:
                if profile_name not in profile_map:
                    continue
                params = profile_map[profile_name]
                for weight_name in args.weight_profiles:
                    if weight_name != "target_auto" and weight_name not in WEIGHT_PROFILES:
                        continue
                    weight_profile = resolve_weight_profile(weight_name, target)
                    for top_k in args.top_k_grid:
                        cfg = f"{model_name}|{profile_name}|{weight_name}|top{top_k}|hist{int(args.target_history_features)}|bank{args.feature_bank}"
                        # Non-active folds stay at anchor when --fold-limit is
                        # used for smoke runs.  Full runs overwrite every fold.
                        pred_oof_accum = anchor_oof[target].values.astype(float).copy()
                        pred_test_accum = np.zeros(len(Xte_target), dtype=float)
                        fold_done = 0
                        for fold in active_folds:
                            tr_idx = np.where(folds != fold)[0]
                            va_idx = np.where(folds == fold)[0]
                            w_fit = sample_weights(y, anchor_oof[target].values, mtr, folds, recency, tr_idx, weight_profile)
                            always_cols = [c for c in [*always_base, *hist_cols] if c in Xtr_target.columns]
                            cols = select_features(Xtr_target, y, tr_idx, top_k, always_cols, w_fit)
                            X_train = Xtr_target.iloc[tr_idx][cols]
                            X_valid = Xtr_target.iloc[va_idx][cols]
                            X_test = Xte_target[cols]
                            te_sources = select_te_source_cols(Xtr_target, y, tr_idx, args.te_top_n, always_cols)
                            X_train, X_valid, X_test, bank_diag, bank_cat_cols = add_fold_feature_bank(
                                X_train,
                                X_valid,
                                X_test,
                                Xtr_target,
                                Xte_target,
                                y,
                                tr_idx,
                                va_idx,
                                te_sources,
                                args.te_bins,
                                args.te_smooth,
                                args.feature_bank,
                            )
                            X_train, X_valid, X_test, cat_cols = prepare_matrix(X_train, X_valid, X_test, model_name, bank_cat_cols)
                            seed_preds_valid = []
                            seed_preds_test = []
                            seed_meta = []
                            for seed in args.seeds:
                                p_valid, p_test, meta = fit_model(
                                    model_name,
                                    profile_name,
                                    params,
                                    X_train,
                                    y[tr_idx],
                                    w_fit,
                                    X_valid,
                                    y[va_idx],
                                    X_test,
                                    cat_cols,
                                    seed,
                                    args.rounds,
                                    args.early_stopping_rounds,
                                    args.log_period,
                                )
                                seed_preds_valid.append(p_valid)
                                seed_preds_test.append(p_test)
                                seed_meta.append(meta)
                            p_valid = clip(np.mean(seed_preds_valid, axis=0))
                            p_test = clip(np.mean(seed_preds_test, axis=0))
                            pred_oof_accum[va_idx] = p_valid
                            pred_test_accum += p_test / max(len(active_folds), 1)
                            fold_done += 1
                            fit_rows.append(
                                {
                                    "target": target,
                                    "config_key": cfg,
                                    "model": model_name,
                                    "param_profile": profile_name,
                                    "weight_profile": weight_name,
                                    "resolved_weight_profile": json.dumps(weight_profile.__dict__),
                                    "top_k": top_k,
                                    "fold": int(fold),
                                    "seed_count": len(args.seeds),
                                    "features": len(cols),
                                    "cat_features": len(cat_cols),
                                    "te_source_count": bank_diag["bank_source_count"],
                                    "te_generated_features": bank_diag["bank_feature_count"],
                                    "valid_logloss": safe_loss(y[va_idx], p_valid),
                                    "best_iteration_mean": float(np.nanmean([m["best_iteration"] for m in seed_meta])),
                                    "best_valid_logloss_mean": float(np.nanmean([m["best_valid_logloss"] for m in seed_meta])),
                                    "best_train_logloss_mean": float(np.nanmean([m["best_train_logloss"] for m in seed_meta])),
                                    "stop_policy": seed_meta[0]["stop_policy"],
                                    "fallback_logic": seed_meta[0]["fallback_logic"],
                                    "weight_min": float(w_fit.min()),
                                    "weight_max": float(w_fit.max()),
                                    "weight_std": float(w_fit.std()),
                                }
                            )
                            log(
                                f"{target} {cfg} fold={fold} valid={safe_loss(y[va_idx], p_valid):.6f} "
                                f"best_iter_mean={np.nanmean([m['best_iteration'] for m in seed_meta]):.1f}"
                            )
                        if fold_done != len(active_folds):
                            continue
                        raw_oof[target][cfg] = clip(pred_oof_accum)
                        raw_test[target][cfg] = clip(pred_test_accum)
                        fold_vals_raw = [safe_loss(y[folds == f], pred_oof_accum[folds == f]) for f in sorted(np.unique(folds)) if f in active_folds]
                        full_raw = safe_loss(y[np.isin(folds, active_folds)], pred_oof_accum[np.isin(folds, active_folds)])
                        last_raw = safe_loss(y[last_mask], pred_oof_accum[last_mask]) if (C.N_SPLITS - 1) in active_folds else np.nan
                        for shrink_mode in args.shrink_modes:
                            for shrink in args.shrink_grid:
                                p = shrink_prediction(anchor_oof[target].values, pred_oof_accum, shrink, shrink_mode)
                                full_loss = safe_loss(y, p)
                                fold_vals = [safe_loss(y[folds == f], p[folds == f]) for f in sorted(np.unique(folds))]
                                last_loss = fold_vals[-1]
                                search_rows.append(
                                    {
                                        "target": target,
                                        "config_key": cfg,
                                        "model": model_name,
                                        "param_profile": profile_name,
                                        "weight_profile": weight_name,
                                        "top_k": top_k,
                                        "shrink_mode": shrink_mode,
                                        "shrink": float(shrink),
                                        "full_logloss": full_loss,
                                        "last_logloss": last_loss,
                                        "full_delta_vs_anchor": full_loss - target_anchor_full,
                                        "last_delta_vs_anchor": last_loss - target_anchor_last,
                                        "rank_score": target_rank_score(full_loss, last_loss, target_anchor_full, fold_vals),
                                        "raw_active_full_logloss": full_raw,
                                        "raw_active_last_logloss": last_raw,
                                        **{f"fold{i}_logloss": v for i, v in enumerate(fold_vals)},
                                        **{f"active_fold{i}_raw_logloss": v for i, v in enumerate(fold_vals_raw)},
                                    }
                                )

    fit_df = pd.DataFrame(fit_rows)
    search_df = pd.DataFrame(search_rows)
    fit_df.to_csv(out_dir / "fit_diagnostics.csv", index=False)
    search_df.sort_values(["target", "rank_score", "last_logloss"]).to_csv(out_dir / "target_config_search.csv", index=False)

    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    choice_tables = []
    for mode in ["composite", "last", "full"]:
        cname = f"target_weighted_single_{mode}"
        po, pt, choices = build_candidate(cname, mode, search_df, raw_oof, raw_test, anchor_oof, anchor_test)
        choices.to_csv(out_dir / f"{cname}_choices.csv", index=False)
        candidates[cname] = (po, pt, choices)
        choice_tables.append(choices)
    pd.concat(choice_tables, ignore_index=True, sort=False).to_csv(out_dir / "target_choices_all.csv", index=False)

    score_rows = []
    for name, (po, pt, _choices) in candidates.items():
        score_rows.append(score_candidate(name, ytr, po, folds, anchor_full, anchor_last))
    scores = pd.DataFrame(score_rows).sort_values(["rank_score", "last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    plot_candidate_scores(out_dir, scores)

    for _, row in scores.iterrows():
        name = str(row["candidate"])
        po, pt, _choices = candidates[name]
        safe = name.replace("/", "_").replace(".", "p")
        stem = f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}"
        write_prediction_frame(out_dir / f"{stem}_oof.csv", mtr, po, ytr)
        write_prediction_frame(out_dir / f"{stem}_test_pred.csv", mte, pt)
        write_submission(sub_dir / f"{stem}.csv", mte, pt)

    # Save best raw source keys as individual OOF/test artifacts for correlation
    # and downstream stack/blend analysis without forcing target-wise selection.
    raw_saved = 0
    raw_best = search_df.sort_values(["rank_score", "last_logloss", "full_logloss"]).drop_duplicates(["target", "config_key"]).head(args.save_top_raw)
    for i, row in raw_best.iterrows():
        target = str(row["target"])
        cfg = str(row["config_key"])
        pred_oof = anchor_oof.copy()
        pred_test = anchor_test.copy()
        pred_oof[target] = shrink_prediction(anchor_oof[target].values, raw_oof[target][cfg], float(row["shrink"]), str(row["shrink_mode"]))
        pred_test[target] = shrink_prediction(anchor_test[target].values, raw_test[target][cfg], float(row["shrink"]), str(row["shrink_mode"]))
        safe_cfg = "".join(ch if ch.isalnum() else "_" for ch in cfg)[:120]
        name = f"raw_{target}_{safe_cfg}_sh{row['shrink']}_{row['shrink_mode']}"
        write_prediction_frame(out_dir / f"{name}_oof.csv", mtr, pred_oof, ytr)
        write_prediction_frame(out_dir / f"{name}_test_pred.csv", mte, pred_test)
        raw_saved += 1

    report = {
        "purpose": "Create new optimized single-model OOF/test sources before stacking/blending.",
        "models": args.models,
        "seeds": args.seeds,
        "fold_limit": args.fold_limit,
        "param_profiles": args.param_profiles,
        "weight_profiles": args.weight_profiles,
        "target_history_features": args.target_history_features,
        "feature_bank": args.feature_bank,
        "te_top_n": args.te_top_n,
        "te_bins": args.te_bins,
        "te_smooth": args.te_smooth,
        "pair_feature_count": args.pair_feature_count,
        "early_stopping_rounds": args.early_stopping_rounds,
        "anchor": {"full_logloss": anchor_full, "last_logloss": anchor_last},
        "candidate_scores": scores.to_dict(orient="records"),
        "raw_sources_saved": raw_saved,
        "notes": [
            "LGBM/XGB/Cat use validation logloss early stopping and best-iteration prediction.",
            "Fold-safe TE/bin views are fit inside each validation fold only.",
            "target_auto resolves to different recency/class/anchor-error weights by target.",
            "Output OOF/test files are meant to feed public_aware_stack_blend/public_score_pseudo_blend later.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Target Weighted Single Model",
        "",
        "This run creates new single-model OOF/test sources, not final submission claims.",
        "",
        "## Candidate Scores",
        "",
        scores.to_string(index=False),
        "",
        "## Best Target Choices",
        "",
        pd.concat(choice_tables, ignore_index=True, sort=False)[
            ["candidate", "target", "model", "param_profile", "weight_profile", "top_k", "shrink_mode", "shrink", "full_logloss", "last_logloss", "rank_score"]
        ].to_string(index=False),
        "",
        "## Early Stopping",
        "",
        "- Every fold/model uses validation logloss early stopping where the model supports it.",
        "- `fit_diagnostics.csv` records `best_iteration_mean`, `stop_policy`, and `fallback_logic`.",
    ]
    (out_dir / "TARGET_WEIGHTED_SINGLE_MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Target-weighted single-model candidates ===")
    print(scores.to_string(index=False))
    print("\n=== Target choices ===")
    show = ["candidate", "target", "model", "param_profile", "weight_profile", "top_k", "shrink_mode", "shrink", "full_logloss", "last_logloss", "rank_score"]
    print(pd.concat(choice_tables, ignore_index=True, sort=False)[show].to_string(index=False))
    log(f"Wrote outputs to {out_dir} and {sub_dir}")


if __name__ == "__main__":
    main()
