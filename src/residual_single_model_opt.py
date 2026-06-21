"""Residual-correction single model optimizer.

The previous diverse single-model run trained each model directly on labels and
then blended weak raw probabilities back into the temporal anchor.  That is a
poor use of a strong anchor: the model should learn where the anchor is wrong.

This script takes one model family, trains it on anchor residuals, and searches
the knobs that matter for this dataset:
- residual-based feature selection
- subject/class/recency/anchor-error sample weights
- target-wise shrinkage back onto the anchor

Default model is ExtraTreesRegressor because it is fast, robust on tiny data,
and supports sample_weight.  The same training loop also supports hist_gb and
ridge so the optimized recipe can be reused for other model families.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .train_temporal_prior import clip

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT
EPS = C.PROB_CLIP


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y, p) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def residual_target(
    y: np.ndarray,
    anchor: np.ndarray,
    mode: str,
    hessian_floor: float,
    target_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the training target and extra sample weight for residual fitting.

    prob:
      Fit probability residual directly: y - p.

    logit_newton:
      Fit a Newton step on logloss around the anchor logit:
      z = (y - p) / (p * (1 - p)), weighted by p * (1 - p).
      The fitted delta is later added in logit space.
    """
    y = np.asarray(y, dtype=float)
    anchor = clip(anchor)
    if mode == "prob":
        return y - anchor, np.ones_like(anchor, dtype=float)
    if mode == "logit_newton":
        hessian = np.clip(anchor * (1.0 - anchor), hessian_floor, None)
        z = (y - anchor) / hessian
        return np.clip(z, -target_clip, target_clip), hessian
    raise ValueError(mode)


def apply_residual_delta(
    anchor: np.ndarray,
    delta: np.ndarray,
    shrink: float,
    mode: str,
    delta_clip: float | None = None,
) -> np.ndarray:
    anchor = clip(anchor)
    delta = np.asarray(delta, dtype=float)
    if delta_clip is not None and delta_clip > 0:
        delta = np.clip(delta, -float(delta_clip), float(delta_clip))
    if mode == "prob":
        return clip(anchor + float(shrink) * delta)
    if mode == "logit_newton":
        return clip(sigmoid(logit(anchor) + float(shrink) * delta))
    raise ValueError(mode)


def load_anchor_bank(bank_dir: Path, n_train: int, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_path = bank_dir / "oof_bank.csv"
    test_path = bank_dir / "test_bank.csv"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing anchor bank under {bank_dir}; run src.oof_sparse_greedy first.")
    oof_bank = pd.read_csv(oof_path)
    test_bank = pd.read_csv(test_path)
    anchor_oof = pd.DataFrame({t: oof_bank[f"anchor__{t}"].values for t in TARGETS})
    anchor_test = pd.DataFrame({t: test_bank[f"anchor__{t}"].values for t in TARGETS})
    if len(anchor_oof) != n_train or len(anchor_test) != n_test:
        raise ValueError("Anchor bank row count does not match current dataset.")
    return anchor_oof, anchor_test


def numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(X[c])]
    return X[cols].replace([np.inf, -np.inf], np.nan).copy()


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
        part["subject_day_frac"] = np.linspace(0.0, 1.0, n) if n > 1 else 0.0
        part["days_since_subject_start"] = (
            g["sleep_date"] - g["sleep_date"].min()
        ).dt.days.astype(float).values
        part["days_until_subject_end"] = (
            g["sleep_date"].max() - g["sleep_date"]
        ).dt.days.astype(float).values
        part["sleep_ord_2024"] = (
            g["sleep_date"] - pd.Timestamp("2024-01-01")
        ).dt.days.astype(float).values
        part["subject_rows_total"] = float(n)
        part["subject_id"] = subject
        frames.append(part)
    time_all = pd.concat(frames).sort_index()
    subj_dummies = pd.get_dummies(time_all["subject_id"], prefix="sid", dtype=float)
    time_all = pd.concat([time_all.drop(columns=["subject_id"]), subj_dummies], axis=1)
    time_cols = time_all.columns.tolist()

    n_train = len(mtr)
    Xtr2 = pd.concat([Xtr.reset_index(drop=True), time_all.iloc[:n_train].reset_index(drop=True)], axis=1)
    Xte2 = pd.concat([Xte.reset_index(drop=True), time_all.iloc[n_train:].reset_index(drop=True)], axis=1)
    return Xtr2, Xte2, time_cols


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
        for frame, anchor in [(Xtr2, anchor_oof), (Xte2, anchor_test)]:
            frame[f"anchor_{target}_prob"] = anchor[target].values
            frame[f"anchor_{target}_logit"] = logit(anchor[target].values)
        cols.extend([f"anchor_{target}_prob", f"anchor_{target}_logit"])
    for name, targets in {
        "q": ["Q1", "Q2", "Q3"],
        "s": ["S1", "S2", "S3", "S4"],
        "all": TARGETS,
    }.items():
        Xtr2[f"anchor_{name}_mean"] = anchor_oof[targets].mean(axis=1).values
        Xte2[f"anchor_{name}_mean"] = anchor_test[targets].mean(axis=1).values
        Xtr2[f"anchor_{name}_std"] = anchor_oof[targets].std(axis=1).values
        Xte2[f"anchor_{name}_std"] = anchor_test[targets].std(axis=1).values
        cols.extend([f"anchor_{name}_mean", f"anchor_{name}_std"])
    return Xtr2, Xte2, cols


def recency_fraction(meta: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(meta), dtype=float)
    m = meta.sort_values(["subject_id", "sleep_date"])
    for _, g in m.groupby("subject_id", sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        out[idx] = np.linspace(0.0, 1.0, n) if n > 1 else 1.0
    return out


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 12:
        return 0.0
    xv = x[mask].astype(float)
    yv = y[mask].astype(float)
    if w is None:
        if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
            return 0.0
        val = np.corrcoef(xv, yv)[0, 1]
        return float(abs(val)) if np.isfinite(val) else 0.0
    wv = w[mask].astype(float)
    sw = float(wv.sum())
    if sw <= 0:
        return 0.0
    mx = float(np.sum(wv * xv) / sw)
    my = float(np.sum(wv * yv) / sw)
    vx = float(np.sum(wv * (xv - mx) ** 2) / sw)
    vy = float(np.sum(wv * (yv - my) ** 2) / sw)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = float(np.sum(wv * (xv - mx) * (yv - my)) / sw)
    return abs(cov / np.sqrt(vx * vy))


def select_residual_features(
    X: pd.DataFrame,
    residual: np.ndarray,
    train_idx: np.ndarray,
    top_k: int,
    always_cols: list[str],
) -> list[str]:
    always = [c for c in always_cols if c in X.columns]
    if top_k <= 0 or top_k >= X.shape[1]:
        cols = X.columns.tolist()
    else:
        scored = []
        y = residual[train_idx]
        for col in X.columns:
            if col in always:
                score = np.inf
            else:
                score = weighted_corr(X[col].values[train_idx], y)
            scored.append((score, col))
        cols = [c for _, c in sorted(scored, reverse=True)[:top_k]]
        for col in always:
            if col not in cols:
                cols.append(col)
    return list(dict.fromkeys(cols))


def select_te_source_cols(
    X: pd.DataFrame,
    fit_target: np.ndarray,
    train_idx: np.ndarray,
    top_n: int,
    exclude_cols: list[str],
) -> list[str]:
    if top_n <= 0:
        return []
    exclude = set(exclude_cols)
    scored = []
    y = fit_target[train_idx]
    for col in X.columns:
        if col in exclude or col.startswith("anchor_") or col.startswith("sid_"):
            continue
        vals = X[col].values
        finite = np.isfinite(vals[train_idx])
        if finite.sum() < 20:
            continue
        unique = np.unique(vals[train_idx][finite])
        if len(unique) < 5:
            continue
        score = weighted_corr(vals[train_idx], y)
        if score > 0:
            scored.append((score, col))
    return [c for _, c in sorted(scored, reverse=True)[:top_n]]


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < max(8, n_bins * 2) or np.nanstd(finite) <= 1e-12:
        return None
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.nanquantile(finite, qs))
    if edges.size == 0:
        return None
    return edges.astype(float)


def digitize_codes(values: np.ndarray, edges: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    codes = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    if edges is None:
        return codes
    codes[finite] = np.digitize(values[finite], edges, right=False).astype(np.int16)
    return codes


def loo_smooth_target_encode(
    train_codes: np.ndarray,
    valid_codes: np.ndarray,
    test_codes: np.ndarray,
    y_train: np.ndarray,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train, dtype=float)
    global_mean = float(np.mean(y_train)) if len(y_train) else 0.0
    df = pd.DataFrame({"code": train_codes, "y": y_train})
    stats = df.groupby("code")["y"].agg(["sum", "count"])
    sums = df["code"].map(stats["sum"]).to_numpy(dtype=float)
    counts = df["code"].map(stats["count"]).to_numpy(dtype=float)
    denom = counts - 1.0 + smooth
    train_te = np.where(
        denom > 0,
        (sums - y_train + smooth * global_mean) / denom,
        global_mean,
    )
    mapping = ((stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)).to_dict()

    def map_codes(codes: np.ndarray) -> np.ndarray:
        return pd.Series(codes).map(mapping).fillna(global_mean).to_numpy(dtype=float)

    return train_te.astype(float), map_codes(valid_codes), map_codes(test_codes)


def add_fold_feature_bank(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    Xte_full: pd.DataFrame,
    fit_target: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    source_cols: list[str],
    bins: list[int],
    smooth: float,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if mode == "none" or not source_cols:
        return X_train, X_valid, X_test, {"bank_source_count": 0, "bank_feature_count": 0}

    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    added = 0
    for col in source_cols:
        train_vals = X_full[col].values[tr_idx]
        valid_vals = X_full[col].values[va_idx]
        test_vals = Xte_full[col].values
        safe_col = "".join(ch if ch.isalnum() else "_" for ch in col)[:80]
        for n_bins in bins:
            edges = quantile_edges(train_vals, int(n_bins))
            if edges is None:
                continue
            tr_code = digitize_codes(train_vals, edges)
            va_code = digitize_codes(valid_vals, edges)
            te_code = digitize_codes(test_vals, edges)
            if mode in {"bins", "bins_te"}:
                name = f"bin_{safe_col}_{n_bins}"
                X_train[name] = tr_code.astype(float)
                X_valid[name] = va_code.astype(float)
                X_test[name] = te_code.astype(float)
                added += 1
            if mode in {"te", "bins_te"}:
                tr_te, va_te, te_te = loo_smooth_target_encode(
                    tr_code,
                    va_code,
                    te_code,
                    fit_target[tr_idx],
                    smooth,
                )
                name = f"te_{safe_col}_{n_bins}"
                X_train[name] = tr_te
                X_valid[name] = va_te
                X_test[name] = te_te
                added += 1
    return X_train, X_valid, X_test, {
        "bank_source_count": len(source_cols),
        "bank_feature_count": added,
    }


@dataclass(frozen=True)
class WeightProfile:
    name: str
    subject_power: float
    class_power: float
    recency_strength: float
    residual_power: float
    fold_strength: float


WEIGHT_PROFILES = [
    WeightProfile("uniform", 0.0, 0.0, 0.0, 0.0, 0.0),
    WeightProfile("subject_class", 1.0, 0.8, 0.0, 0.0, 0.0),
    WeightProfile("recent_class", 1.0, 0.8, 1.2, 0.0, 0.35),
    WeightProfile("recent_resid", 1.0, 0.5, 1.8, 0.8, 0.55),
]


EXTRA_TREES_PROFILES = [
    {
        "name": "smooth",
        "n_estimators": 240,
        "max_depth": 3,
        "min_samples_leaf": 20,
        "max_features": 0.35,
        "bootstrap": True,
    },
    {
        "name": "mid",
        "n_estimators": 260,
        "max_depth": 4,
        "min_samples_leaf": 14,
        "max_features": "sqrt",
        "bootstrap": False,
    },
    {
        "name": "wide",
        "n_estimators": 260,
        "max_depth": 5,
        "min_samples_leaf": 10,
        "max_features": 0.45,
        "bootstrap": True,
    },
]


HIST_GB_PROFILES = [
    {"name": "smooth", "learning_rate": 0.025, "max_iter": 140, "max_leaf_nodes": 5, "min_samples_leaf": 24, "l2_regularization": 3.0},
    {"name": "mid", "learning_rate": 0.020, "max_iter": 220, "max_leaf_nodes": 7, "min_samples_leaf": 18, "l2_regularization": 2.0},
    {"name": "tiny", "learning_rate": 0.035, "max_iter": 100, "max_leaf_nodes": 3, "min_samples_leaf": 30, "l2_regularization": 5.0},
]


RIDGE_PROFILES = [
    {"name": "ridge0p3", "alpha": 0.3},
    {"name": "ridge1", "alpha": 1.0},
    {"name": "ridge3", "alpha": 3.0},
    {"name": "ridge10", "alpha": 10.0},
    {"name": "ridge30", "alpha": 30.0},
    {"name": "ridge100", "alpha": 100.0},
]


def model_profiles(model: str) -> list[dict]:
    if model == "extra_trees":
        return EXTRA_TREES_PROFILES
    if model == "hist_gb":
        return HIST_GB_PROFILES
    if model == "ridge":
        return RIDGE_PROFILES
    raise ValueError(model)


def make_model(model: str, params: dict, seed: int):
    params = {k: v for k, v in params.items() if k != "name"}
    if model == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **params)
    if model == "hist_gb":
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=True, **params)
    if model == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(random_state=seed, **params),
        )
    raise ValueError(model)


def sample_weights(
    y: np.ndarray,
    anchor: np.ndarray,
    residual_for_weight: np.ndarray,
    meta: pd.DataFrame,
    folds: np.ndarray,
    recency: np.ndarray,
    train_idx: np.ndarray,
    profile: WeightProfile,
    base_weight: np.ndarray | None = None,
) -> np.ndarray:
    idx = np.asarray(train_idx)
    w = np.ones(len(idx), dtype=float)
    if base_weight is not None:
        w *= np.asarray(base_weight, dtype=float)[idx]
    if profile.subject_power:
        counts = meta.iloc[idx]["subject_id"].map(meta.iloc[idx]["subject_id"].value_counts()).astype(float).values
        subj_w = len(idx) / np.maximum(counts, 1.0)
        w *= subj_w ** profile.subject_power
    if profile.class_power:
        yt = y[idx].astype(int)
        pos = float(yt.mean())
        pos = float(np.clip(pos, 0.05, 0.95))
        cls = np.where(yt == 1, 0.5 / pos, 0.5 / (1.0 - pos))
        w *= cls ** profile.class_power
    if profile.recency_strength:
        # Oldest row gets exp(-strength), newest gets 1.0.
        w *= np.exp(profile.recency_strength * (recency[idx] - 1.0))
    if profile.residual_power:
        err = np.abs(residual_for_weight[idx])
        w *= (0.35 + err) ** profile.residual_power
    if profile.fold_strength:
        fold_frac = folds[idx].astype(float) / max(float(C.N_SPLITS - 1), 1.0)
        w *= 1.0 + profile.fold_strength * fold_frac
    w = np.clip(w, 0.05, 20.0)
    return w / max(float(w.mean()), 1e-12)


def fit_predict_residual(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    Xte: pd.DataFrame,
    fit_target: np.ndarray,
    residual_for_weight: np.ndarray,
    base_weight: np.ndarray,
    y: np.ndarray,
    anchor: np.ndarray,
    meta: pd.DataFrame,
    folds: np.ndarray,
    recency: np.ndarray,
    weight_profile: WeightProfile,
    top_k: int,
    always_cols: list[str],
    target: str,
    feature_bank: str,
    te_top_n: int,
    te_bins: list[int],
    te_smooth: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    delta_oof = np.zeros(len(X), dtype=float)
    delta_test = np.zeros(len(Xte), dtype=float)
    rows: list[dict] = []
    feature_cache: dict[int, list[str]] = {}
    for fold in sorted(np.unique(folds)):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]
        if fold not in feature_cache:
            feature_cache[fold] = select_residual_features(X, fit_target, tr_idx, top_k, always_cols)
        cols = feature_cache[fold]
        X_train_fold = X.iloc[tr_idx][cols]
        X_valid_fold = X.iloc[va_idx][cols]
        X_test_fold = Xte[cols]
        te_source_cols = select_te_source_cols(
            X,
            fit_target,
            tr_idx,
            te_top_n,
            always_cols,
        )
        X_train_fold, X_valid_fold, X_test_fold, bank_diag = add_fold_feature_bank(
            X_train_fold,
            X_valid_fold,
            X_test_fold,
            X,
            Xte,
            fit_target,
            tr_idx,
            va_idx,
            te_source_cols,
            te_bins,
            te_smooth,
            feature_bank,
        )
        weights = sample_weights(
            y,
            anchor,
            residual_for_weight,
            meta,
            folds,
            recency,
            tr_idx,
            weight_profile,
            base_weight=base_weight,
        )
        model = make_model(model_name, params, seed=7000 + 97 * fold + 13 * TARGETS.index(target))
        if model_name == "ridge":
            model.fit(X_train_fold, fit_target[tr_idx], ridge__sample_weight=weights)
            p_val_delta = model.predict(X_valid_fold)
            p_test_delta = model.predict(X_test_fold)
            stop_policy = "closed_form_no_early_stopping"
            stop_iteration = np.nan
        else:
            imp = SimpleImputer(strategy="median")
            Xfit = imp.fit_transform(X_train_fold)
            Xval = imp.transform(X_valid_fold)
            Xtest = imp.transform(X_test_fold)
            model.fit(Xfit, fit_target[tr_idx], sample_weight=weights)
            p_val_delta = model.predict(Xval)
            p_test_delta = model.predict(Xtest)
            if model_name == "hist_gb":
                stop_policy = "internal_validation_early_stopping"
                stop_iteration = float(getattr(model, "n_iter_", np.nan))
            else:
                stop_policy = "fixed_estimator_count_no_early_stopping"
                stop_iteration = float(getattr(model, "n_estimators", np.nan))
        delta_oof[va_idx] = p_val_delta
        delta_test += p_test_delta / C.N_SPLITS
        rows.append({
            "target": target,
            "fold": int(fold),
            "model": model_name,
            "param_profile": params["name"],
            "weight_profile": weight_profile.name,
            "top_k": int(top_k),
            "features": len(cols),
            "feature_bank": feature_bank,
            "te_source_count": bank_diag["bank_source_count"],
            "te_generated_features": bank_diag["bank_feature_count"],
            "stop_policy": stop_policy,
            "stop_iteration": stop_iteration,
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_std": float(weights.std()),
            "delta_std_val": float(np.std(p_val_delta)),
        })
    return delta_oof, delta_test, rows


def candidate_stability(ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray, name: str) -> dict:
    row = {"candidate": name}
    full_mask = np.ones(len(ytr), dtype=bool)
    row["full_logloss"] = mean_loss(ytr, pred, full_mask)
    vals = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        val = mean_loss(ytr, pred, mask)
        row[f"fold{fold}_logloss"] = val
        vals.append(val)
    row["last_logloss"] = vals[-1]
    row["tail3_mean"] = float(np.mean(vals[-3:]))
    row["tail3_worst"] = float(np.max(vals[-3:]))
    row["fold_std"] = float(np.std(vals))
    return row


def target_rank_score(full_loss: float, last_loss: float, anchor_full: float, fold_losses: list[float]) -> float:
    tail3 = fold_losses[-3:]
    return (
        last_loss
        + 0.85 * max(0.0, full_loss - anchor_full)
        + 0.20 * max(0.0, max(tail3) - float(np.mean(tail3)))
    )


def build_targetwise_candidate(
    name: str,
    search: pd.DataFrame,
    deltas_oof: dict[str, dict[str, np.ndarray]],
    deltas_test: dict[str, dict[str, np.ndarray]],
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    mode: str,
    residual_mode: str,
    delta_clip: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = pd.DataFrame(index=anchor_oof.index, columns=TARGETS, dtype=float)
    out_test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    chosen = []
    for target in TARGETS:
        rows = search[search["target"].eq(target)].copy()
        if mode == "last":
            rows = rows.sort_values(["last_logloss", "full_logloss", "rank_score"])
        elif mode == "composite":
            rows = rows.sort_values(["rank_score", "last_logloss", "full_logloss"])
        elif mode == "full":
            rows = rows.sort_values(["full_logloss", "last_logloss", "rank_score"])
        else:
            raise ValueError(mode)
        row = rows.iloc[0].to_dict()
        key = str(row["config_key"])
        shrink = float(row["shrink"])
        out_oof[target] = apply_residual_delta(
            anchor_oof[target].values,
            deltas_oof[target][key],
            shrink,
            residual_mode,
            delta_clip,
        )
        out_test[target] = apply_residual_delta(
            anchor_test[target].values,
            deltas_test[target][key],
            shrink,
            residual_mode,
            delta_clip,
        )
        row["candidate"] = name
        row["selection_mode"] = mode
        chosen.append(row)
    return out_oof, out_test, pd.DataFrame(chosen)


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test.copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def write_prediction_frame(path: Path, meta: pd.DataFrame, pred: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
    out = meta.copy()
    if y is not None:
        for target in TARGETS:
            out[f"label__{target}"] = y[target].values
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-dir", default="research/oof_sparse_greedy")
    p.add_argument("--output-dir", default="research/residual_single_model_opt")
    p.add_argument("--submission-dir", default="submissions/residual_single_model_opt")
    p.add_argument("--model", choices=["extra_trees", "hist_gb", "ridge"], default="extra_trees")
    p.add_argument("--residual-mode", choices=["prob", "logit_newton"], default="logit_newton")
    p.add_argument("--newton-hessian-floor", type=float, default=0.02)
    p.add_argument("--newton-target-clip", type=float, default=6.0)
    p.add_argument("--delta-clip", type=float, default=20.0)
    p.add_argument("--top-k-grid", nargs="*", type=int, default=[40, 80, 120, 160, 240])
    p.add_argument("--shrink-grid", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50])
    p.add_argument("--weight-profiles", nargs="*", default=[w.name for w in WEIGHT_PROFILES])
    p.add_argument("--param-profiles", nargs="*", default=None)
    p.add_argument("--feature-bank", choices=["none", "bins", "te", "bins_te"], default="none")
    p.add_argument("--te-top-n", type=int, default=12)
    p.add_argument("--te-bins", nargs="*", type=int, default=[4, 8])
    p.add_argument("--te-smooth", type=float, default=12.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset and anchor bank")
    Xtr_raw, ytr, Xte_raw, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full = np.ones(len(ytr), dtype=bool)
    last = folds == (C.N_SPLITS - 1)
    anchor_oof, anchor_test = load_anchor_bank(ROOT / args.bank_dir, len(ytr), len(mte))
    anchor_full = mean_loss(ytr, anchor_oof, full)
    anchor_last = mean_loss(ytr, anchor_oof, last)
    log(f"anchor full={anchor_full:.6f} last={anchor_last:.6f}")

    Xtr, Xte, time_cols = add_subject_time_features(Xtr_raw, Xte_raw, mtr, mte)
    Xtr, Xte, anchor_cols = add_anchor_features(Xtr, Xte, anchor_oof, anchor_test)
    Xtr = numeric_frame(Xtr)
    Xte = numeric_frame(Xte)
    always_cols = [c for c in [*anchor_cols, *time_cols] if c in Xtr.columns]
    recency = recency_fraction(mtr)

    weight_profiles = [w for w in WEIGHT_PROFILES if w.name in set(args.weight_profiles)]
    profiles = model_profiles(args.model)
    if args.param_profiles:
        wanted = set(args.param_profiles)
        profiles = [p for p in profiles if p["name"] in wanted]
    if not profiles or not weight_profiles:
        raise ValueError("No model/weight profiles selected.")

    deltas_oof: dict[str, dict[str, np.ndarray]] = {t: {} for t in TARGETS}
    deltas_test: dict[str, dict[str, np.ndarray]] = {t: {} for t in TARGETS}
    fit_rows: list[dict] = []
    search_rows: list[dict] = []
    profile_score_rows: list[dict] = []

    for target in TARGETS:
        y = ytr[target].values.astype(float)
        anchor = anchor_oof[target].values.astype(float)
        prob_residual = y - anchor
        fit_target, base_weight = residual_target(
            y,
            anchor,
            args.residual_mode,
            args.newton_hessian_floor,
            args.newton_target_clip,
        )
        target_anchor_full = safe_loss(y, anchor)
        target_anchor_last = safe_loss(y[last], anchor[last])
        log(
            f"Optimizing {args.model} {args.residual_mode} residual target={target} "
            f"anchor_full={target_anchor_full:.6f} anchor_last={target_anchor_last:.6f}"
        )
        for top_k in args.top_k_grid:
            for params in profiles:
                for weight_profile in weight_profiles:
                    key = (
                        f"{args.model}|{args.residual_mode}|{args.feature_bank}|"
                        f"{params['name']}|{weight_profile.name}|top{top_k}"
                    )
                    doof, dtest, rows = fit_predict_residual(
                        args.model,
                        params,
                        Xtr,
                        Xte,
                        fit_target,
                        prob_residual,
                        base_weight,
                        y,
                        anchor,
                        mtr,
                        folds,
                        recency,
                        weight_profile,
                        top_k,
                        always_cols,
                        target,
                        args.feature_bank,
                        args.te_top_n,
                        args.te_bins,
                        args.te_smooth,
                    )
                    deltas_oof[target][key] = doof
                    deltas_test[target][key] = dtest
                    fit_rows.extend(rows)
                    best_for_key = None
                    for shrink in args.shrink_grid:
                        pred = apply_residual_delta(anchor, doof, float(shrink), args.residual_mode, args.delta_clip)
                        full_loss = safe_loss(y, pred)
                        fold_losses = [safe_loss(y[folds == f], pred[folds == f]) for f in sorted(np.unique(folds))]
                        last_loss = fold_losses[-1]
                        row = {
                            "target": target,
                            "config_key": key,
                            "model": args.model,
                            "param_profile": params["name"],
                            "weight_profile": weight_profile.name,
                            "top_k": int(top_k),
                            "shrink": float(shrink),
                            "full_logloss": full_loss,
                            "last_logloss": last_loss,
                            "full_delta_vs_anchor": full_loss - target_anchor_full,
                            "last_delta_vs_anchor": last_loss - target_anchor_last,
                            "rank_score": target_rank_score(full_loss, last_loss, target_anchor_full, fold_losses),
                        }
                        for i, val in enumerate(fold_losses):
                            row[f"fold{i}_logloss"] = val
                        search_rows.append(row)
                        if best_for_key is None or row["rank_score"] < best_for_key["rank_score"]:
                            best_for_key = row
                    assert best_for_key is not None
                    profile_score_rows.append(best_for_key)
        tsearch = pd.DataFrame([r for r in search_rows if r["target"] == target])
        best = tsearch.sort_values(["rank_score", "last_logloss", "full_logloss"]).iloc[0]
        log(
            f"{target} best rank={best['rank_score']:.6f} full={best['full_logloss']:.6f} "
            f"last={best['last_logloss']:.6f} cfg={best['config_key']} shrink={best['shrink']}"
        )

    fit_df = pd.DataFrame(fit_rows)
    search_df = pd.DataFrame(search_rows)
    profile_df = pd.DataFrame(profile_score_rows)
    fit_df.to_csv(out_dir / "fit_diagnostics.csv", index=False)
    search_df.sort_values(["target", "rank_score", "last_logloss"]).to_csv(out_dir / "target_shrink_search.csv", index=False)
    profile_df.sort_values(["target", "rank_score", "last_logloss"]).to_csv(out_dir / "profile_best_shrink.csv", index=False)

    candidate_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]] = {
        "anchor": (anchor_oof, anchor_test, "temporal anchor"),
    }
    choices_all = []
    for mode in ["composite", "last", "full"]:
        cname = f"{args.model}_residual_{mode}"
        po, pt, choices = build_targetwise_candidate(
            cname,
            search_df,
            deltas_oof,
            deltas_test,
            ytr,
            folds,
            anchor_oof,
            anchor_test,
            mode,
            args.residual_mode,
            args.delta_clip,
        )
        choices.to_csv(out_dir / f"{cname}_target_choices.csv", index=False)
        choices_all.append(choices)
        candidate_frames[cname] = (po, pt, f"{args.model} residual correction selected by {mode}")

    score_rows = []
    for name, (po, pt, notes) in candidate_frames.items():
        row = candidate_stability(ytr, po, folds, name)
        row["notes"] = notes
        row["full_delta_vs_anchor"] = row["full_logloss"] - anchor_full
        row["last_delta_vs_anchor"] = row["last_logloss"] - anchor_last
        row["rank_score"] = (
            row["last_logloss"]
            + 0.85 * max(0.0, row["full_logloss"] - anchor_full)
            + 0.20 * max(0.0, row["tail3_worst"] - row["tail3_mean"])
        )
        score_rows.append(row)
        safe = name.replace("/", "_").replace(".", "p")
        write_prediction_frame(out_dir / f"{safe}_oof.csv", mtr, po, ytr)
        write_prediction_frame(out_dir / f"{safe}_test_pred.csv", mte, pt)
        write_submission(sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv", mte, pt)

    scores = pd.DataFrame(score_rows).sort_values(["rank_score", "last_logloss", "full_logloss"])
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.concat(choices_all, ignore_index=True).to_csv(out_dir / "target_choices_all.csv", index=False)
    report = {
        "purpose": "Optimize one single model as residual correction over the temporal anchor.",
        "model": args.model,
        "residual_mode": args.residual_mode,
        "newton_hessian_floor": args.newton_hessian_floor,
        "newton_target_clip": args.newton_target_clip,
        "delta_clip": args.delta_clip,
        "top_k_grid": args.top_k_grid,
        "shrink_grid": args.shrink_grid,
        "weight_profiles": [w.name for w in weight_profiles],
        "param_profiles": [p["name"] for p in profiles],
        "feature_bank": args.feature_bank,
        "te_top_n": args.te_top_n,
        "te_bins": args.te_bins,
        "te_smooth": args.te_smooth,
        "anchor": {"full_logloss": anchor_full, "last_logloss": anchor_last},
        "candidate_scores": scores.to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Residual single-model candidates ===")
    cols = ["candidate", "rank_score", "full_logloss", "last_logloss", "tail3_worst", "fold_std", "notes"]
    print(scores[cols].to_string(index=False))
    print("\n=== Target choices (composite) ===")
    comp = pd.read_csv(out_dir / f"{args.model}_residual_composite_target_choices.csv")
    show = ["target", "param_profile", "weight_profile", "top_k", "shrink", "full_logloss", "last_logloss", "rank_score"]
    print(comp[show].to_string(index=False))


if __name__ == "__main__":
    main()
