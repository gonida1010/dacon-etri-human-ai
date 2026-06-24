"""Target-specific models on compact raw timeline features.

This experiment is meant to break the current plateau caused by correlated
daily-aggregate/anchor sources.  It trains new OOF/test sources from compact
event-level timeline features and explicitly reports false-positive pressure.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import load_labels
from .cv import subject_time_blocked_folds
from .direction_gated_search import load_anchor
from .raw_timeline_features import build_label_timeline_features
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
EPS = C.PROB_CLIP


PROFILES: dict[str, dict] = {
    "compact": {
        "learning_rate": 0.025,
        "num_leaves": 7,
        "min_child_samples": 35,
        "feature_fraction": 0.78,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 1.2,
        "lambda_l2": 2.5,
    },
    "mid": {
        "learning_rate": 0.022,
        "num_leaves": 15,
        "min_child_samples": 24,
        "feature_fraction": 0.72,
        "bagging_fraction": 0.82,
        "bagging_freq": 1,
        "lambda_l1": 0.8,
        "lambda_l2": 1.8,
    },
    "wide": {
        "learning_rate": 0.018,
        "num_leaves": 31,
        "min_child_samples": 18,
        "feature_fraction": 0.66,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 0.6,
        "lambda_l2": 1.4,
    },
}


FP_NEG_WEIGHT = {
    "Q1": 1.05,
    "Q2": 1.35,
    "Q3": 1.55,
    "S1": 1.25,
    "S2": 1.25,
    "S3": 1.45,
    "S4": 1.30,
}


@dataclass(frozen=True)
class Config:
    target: str
    scope: str
    profile: str
    weight_profile: str
    top_k: int

    @property
    def key(self) -> str:
        return f"{self.scope}|{self.profile}|{self.weight_profile}|top{self.top_k}"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)[:170]


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


def safe_loss(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def fold_losses(y: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> list[float]:
    return [mean_loss(y, pred, folds == f) for f in sorted(np.unique(folds))]


def calendar_features(meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=meta.index)
    sleep = pd.to_datetime(meta["sleep_date"])
    life = pd.to_datetime(meta["lifelog_date"])
    for prefix, dt in [("sleep", sleep), ("life", life)]:
        out[f"{prefix}_dow"] = dt.dt.dayofweek.astype(float)
        out[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
        out[f"{prefix}_month"] = dt.dt.month.astype(float)
        out[f"{prefix}_day"] = dt.dt.day.astype(float)
        out[f"{prefix}_doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 366)
        out[f"{prefix}_doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 366)
    return out


def subject_time_features(meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=meta.index)
    out["subject_num"] = meta["subject_id"].str.extract(r"(\d+)").astype(float).values
    tmp = meta[["subject_id", "sleep_date"]].copy()
    tmp["sleep_date"] = pd.to_datetime(tmp["sleep_date"])
    first = tmp.groupby("subject_id")["sleep_date"].transform("min")
    last = tmp.groupby("subject_id")["sleep_date"].transform("max")
    span = (last - first).dt.days.replace(0, np.nan)
    out["subj_days_since_first"] = (tmp["sleep_date"] - first).dt.days.astype(float)
    out["subj_time_frac"] = out["subj_days_since_first"] / span.astype(float)
    out["subj_row_num"] = tmp.sort_values(["subject_id", "sleep_date"]).groupby("subject_id").cumcount().reindex(meta.index).astype(float)
    return out


def add_log_transforms(X: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in X.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(X[c])]
    new = {}
    for col in num_cols:
        s = X[col]
        if s.notna().mean() < 0.15:
            continue
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)
        if pd.notna(mn) and pd.notna(mx) and mn >= 0 and mx > 50:
            new[f"{col}_log1p"] = np.log1p(s)
    if not new:
        return X
    return pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)


def build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    use_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    both_meta = pd.concat([train[C.ID_COLS], test[C.ID_COLS]], ignore_index=True)
    raw = build_label_timeline_features(both_meta, use_cache=use_cache).reset_index(drop=True)
    cal = calendar_features(both_meta)
    time = subject_time_features(both_meta)
    X = pd.concat([raw, cal, time], axis=1)

    anchor_all = pd.concat([anchor_oof.reset_index(drop=True), anchor_test.reset_index(drop=True)], ignore_index=True)
    for target in TARGETS:
        X[f"anchor_{target}"] = anchor_all[target].values
        X[f"anchor_{target}_logit"] = logit(anchor_all[target].values)

    X = add_log_transforms(X)
    X["subject_id"] = both_meta["subject_id"].astype("category").values
    n_train = len(train)
    return X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True)


def scope_columns(X: pd.DataFrame, target: str, scope: str) -> list[str]:
    always_prefixes = ("sleep_", "life_", "subj_", "subject_num", "anchor_")
    always = [c for c in X.columns if c == "subject_id" or c.startswith(always_prefixes)]
    if scope == "target":
        if target.startswith("S"):
            scoped = [c for c in X.columns if c.startswith("N_")]
        else:
            scoped = [c for c in X.columns if c.startswith("L_")]
    elif scope == "all":
        scoped = [c for c in X.columns if c.startswith(("N_", "L_"))]
    elif scope == "night":
        scoped = [c for c in X.columns if c.startswith("N_")]
    elif scope == "day":
        scoped = [c for c in X.columns if c.startswith("L_")]
    else:
        raise ValueError(f"unknown scope: {scope}")
    return list(dict.fromkeys([*always, *scoped]))


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    candidates: list[str],
    always: list[str],
    top_k: int,
    weights: np.ndarray | None = None,
) -> list[str]:
    num_candidates = [
        c for c in candidates
        if c not in always and c != "subject_id" and pd.api.types.is_numeric_dtype(X[c])
    ]
    scores = []
    yt = y[train_idx].astype(float)
    if weights is None:
        wt = np.ones(len(train_idx), dtype=float)
    else:
        wt = weights.astype(float)
    y_mean = np.average(yt, weights=wt)
    y_cent = yt - y_mean
    y_var = np.average(y_cent * y_cent, weights=wt)
    for col in num_candidates:
        s = pd.to_numeric(X[col].iloc[train_idx], errors="coerce")
        if s.notna().mean() < 0.20:
            continue
        vals = s.fillna(s.median()).to_numpy(float)
        v_mean = np.average(vals, weights=wt)
        v_cent = vals - v_mean
        v_var = np.average(v_cent * v_cent, weights=wt)
        if v_var <= 1e-12 or y_var <= 1e-12:
            continue
        cov = np.average(v_cent * y_cent, weights=wt)
        scores.append((abs(cov / np.sqrt(v_var * y_var)), col))
    chosen = [c for _, c in sorted(scores, reverse=True)[:top_k]]
    return list(dict.fromkeys([c for c in [*always, *chosen] if c in X.columns]))


def fold_subject_prior_features(
    y: np.ndarray,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    train_idx: np.ndarray,
    smooth: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_meta = mtr.reset_index(drop=True)
    test_meta = mte.reset_index(drop=True)
    fit = train_meta.iloc[train_idx].copy()
    fit["y"] = y[train_idx]
    gmean = float(np.mean(y[train_idx]))
    grp = fit.groupby("subject_id")["y"].agg(["sum", "count"])
    prior = ((grp["sum"] + smooth * gmean) / (grp["count"] + smooth)).to_dict()
    count = grp["count"].to_dict()

    def base_frame(meta: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=meta.index)
        out["target_subj_prior"] = meta["subject_id"].map(prior).fillna(gmean).astype(float)
        out["target_subj_count"] = meta["subject_id"].map(count).fillna(0).astype(float)
        return out

    train_out = base_frame(train_meta)
    test_out = base_frame(test_meta)

    hist = fit.sort_values(["subject_id", "sleep_date"])
    hist_by_subject = {s: g[["sleep_date", "y"]] for s, g in hist.groupby("subject_id")}

    def recent_for(meta: pd.DataFrame, k: int) -> np.ndarray:
        vals = []
        for _, row in meta.iterrows():
            h = hist_by_subject.get(row["subject_id"])
            if h is None:
                vals.append(gmean)
                continue
            prev = h[h["sleep_date"] < row["sleep_date"]]["y"].tail(k)
            vals.append(float((prev.sum() + smooth * gmean) / (len(prev) + smooth)) if len(prev) else gmean)
        return np.asarray(vals, dtype=float)

    for k in [3, 7, 14]:
        train_out[f"target_recent{k}"] = recent_for(train_meta, k)
        test_out[f"target_recent{k}"] = recent_for(test_meta, k)
    return train_out, test_out


def sample_weights(y: np.ndarray, mtr: pd.DataFrame, profile: str, target: str) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    if profile in {"fp_guard", "recent_fp_guard"}:
        w[y == 0] *= FP_NEG_WEIGHT[target]
    elif profile == "balanced":
        pos = max(float(np.mean(y)), 1e-6)
        neg = max(1.0 - pos, 1e-6)
        w[y == 1] *= 0.5 / pos
        w[y == 0] *= 0.5 / neg
    elif profile != "uniform":
        raise ValueError(f"unknown weight profile: {profile}")

    if profile == "recent_fp_guard":
        dt = pd.to_datetime(mtr["sleep_date"])
        frac = (dt - dt.min()).dt.days / max((dt.max() - dt.min()).days, 1)
        w *= 0.85 + 0.35 * frac.to_numpy(float)
    return w


def prepare_lgb_frames(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()
    cat_cols = ["subject_id"] if "subject_id" in X_train.columns else []
    for frame in [X_train, X_valid, X_test]:
        for col in cat_cols:
            frame[col] = frame[col].astype("category")
        for col in frame.columns:
            if col not in cat_cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return X_train, X_valid, X_test, cat_cols


def fit_lgbm(
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


def blend_with_anchor(anchor: np.ndarray, model: np.ndarray, model_weight: float, mode: str) -> np.ndarray:
    if mode == "prob":
        return clip((1.0 - model_weight) * anchor + model_weight * model)
    if mode == "logit":
        return clip(sigmoid((1.0 - model_weight) * logit(anchor) + model_weight * logit(model)))
    raise ValueError(mode)


def calibrate_values(p: np.ndarray, intercept: float, temp: float) -> np.ndarray:
    return clip(sigmoid(logit(p) / temp + intercept))


def fp_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = clip(p)
    return {
        "fp_rate": float(((y == 0) & (p >= 0.5)).mean()),
        "fn_rate": float(((y == 1) & (p < 0.5)).mean()),
        "pred_pos_rate": float((p >= 0.5).mean()),
        "true_pos_rate": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def target_rank_score(
    full: float,
    last: float,
    fold_vals: list[float],
    fp: dict[str, float],
    anchor_full: float,
) -> float:
    excess_pos = max(0.0, fp["pred_pos_rate"] - fp["true_pos_rate"] - 0.06)
    return (
        full
        + 0.25 * max(0.0, last - full)
        + 0.40 * max(0.0, full - anchor_full)
        + 0.06 * excess_pos
        + 0.05 * max(0.0, max(fold_vals[-3:]) - float(np.mean(fold_vals[-3:])))
    )


def write_prediction(path: Path, meta: pd.DataFrame, pred: pd.DataFrame, labels: pd.DataFrame | None = None) -> None:
    out = meta.reset_index(drop=True).copy()
    if labels is not None:
        for target in TARGETS:
            out[f"label__{target}"] = labels[target].values
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    out.to_csv(path, index=False)


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test.reset_index(drop=True).copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    out.to_csv(path, index=False)


def build_candidate(
    name: str,
    selection: str,
    search: pd.DataFrame,
    raw_oof: dict[str, dict[str, np.ndarray]],
    raw_test: dict[str, dict[str, np.ndarray]],
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    full_guard: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = pd.DataFrame(index=anchor_oof.index, columns=TARGETS, dtype=float)
    out_test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    choices = []
    for target in TARGETS:
        rows = search[search["target"].eq(target)].copy()
        if selection == "full":
            rows = rows.sort_values(["full_logloss", "rank_score", "last_logloss"])
        elif selection == "last":
            anchor_full = float(rows["anchor_full_logloss"].iloc[0])
            guarded = rows[rows["full_logloss"] <= anchor_full + full_guard]
            rows = (guarded if not guarded.empty else rows).sort_values(["last_logloss", "rank_score", "full_logloss"])
        elif selection == "fp_guard":
            rows = rows.sort_values(["rank_score", "fp_rate", "full_logloss"])
        elif selection == "composite":
            rows = rows.sort_values(["rank_score", "full_logloss", "last_logloss"])
        else:
            raise ValueError(selection)
        row = rows.iloc[0].to_dict()
        cfg = str(row["config_key"])
        model_weight = float(row["model_weight"])
        mode = str(row["blend_mode"])
        intercept = float(row["intercept"])
        temp = float(row["temperature"])
        p_oof = blend_with_anchor(anchor_oof[target].values, raw_oof[target][cfg], model_weight, mode)
        p_test = blend_with_anchor(anchor_test[target].values, raw_test[target][cfg], model_weight, mode)
        out_oof[target] = calibrate_values(p_oof, intercept, temp)
        out_test[target] = calibrate_values(p_test, intercept, temp)
        row["candidate"] = name
        row["selection"] = selection
        choices.append(row)
    return out_oof, out_test, pd.DataFrame(choices)


def score_candidate(name: str, y: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray, anchor_full: float, anchor_last: float) -> dict:
    vals = fold_losses(y, pred, folds)
    full = mean_loss(y, pred, np.ones(len(y), dtype=bool))
    last = vals[-1]
    row = {
        "candidate": name,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "fold_std": float(np.std(vals)),
        "tail3_mean": float(np.mean(vals[-3:])),
        "tail3_worst": float(max(vals[-3:])),
        **{f"fold{i}_logloss": v for i, v in enumerate(vals)},
    }
    for target in TARGETS:
        fp = fp_summary(y[target].values, pred[target].values)
        for key, val in fp.items():
            row[f"{target}_{key}"] = val
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/raw_timeline_target_model_20260623")
    p.add_argument("--submission-dir", default="submissions/raw_timeline_target_model_20260623")
    p.add_argument("--rebuild-raw-cache", action="store_true")
    p.add_argument("--scopes", nargs="*", default=["target", "all"], choices=["target", "all", "night", "day"])
    p.add_argument("--profiles", nargs="*", default=["compact", "mid"], choices=list(PROFILES))
    p.add_argument("--weight-profiles", nargs="*", default=["uniform", "fp_guard", "recent_fp_guard"], choices=["uniform", "fp_guard", "recent_fp_guard", "balanced"])
    p.add_argument("--top-k-grid", nargs="*", type=int, default=[60, 120, 220])
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    p.add_argument("--rounds", type=int, default=1800)
    p.add_argument("--early-stopping-rounds", type=int, default=120)
    p.add_argument("--model-weight-grid", nargs="*", type=float, default=[0.0, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0])
    p.add_argument("--blend-modes", nargs="*", default=["logit", "prob"], choices=["logit", "prob"])
    p.add_argument("--intercepts", nargs="*", type=float, default=[-0.24, -0.12, 0.0, 0.08])
    p.add_argument("--temperatures", nargs="*", type=float, default=[0.90, 1.0, 1.15, 1.35])
    p.add_argument("--full-guard", type=float, default=0.010)
    p.add_argument("--log-period", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading labels, anchor, and raw timeline features")
    train, test = load_labels()
    ytr = train[TARGETS].reset_index(drop=True)
    mtr = train[C.ID_COLS].reset_index(drop=True)
    mte = test[C.ID_COLS].reset_index(drop=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    anchor_oof, anchor_test = load_anchor()
    Xtr_base, Xte_base = build_feature_matrix(train, test, anchor_oof, anchor_test, use_cache=not args.rebuild_raw_cache)
    log(f"feature matrix train={Xtr_base.shape} test={Xte_base.shape}")

    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    anchor_full = mean_loss(ytr, anchor_oof, full_mask)
    anchor_last = mean_loss(ytr, anchor_oof, last_mask)
    log(f"anchor full={anchor_full:.6f} last={anchor_last:.6f}")

    raw_oof: dict[str, dict[str, np.ndarray]] = {target: {} for target in TARGETS}
    raw_test: dict[str, dict[str, np.ndarray]] = {target: {} for target in TARGETS}
    fit_rows = []
    search_rows = []

    for target in TARGETS:
        y = ytr[target].values.astype(int)
        target_anchor_full = safe_loss(y, anchor_oof[target].values)
        target_anchor_last = safe_loss(y[last_mask], anchor_oof[target].values[last_mask])
        log(f"Target {target}: anchor_full={target_anchor_full:.6f} anchor_last={target_anchor_last:.6f}")
        configs = [
            Config(target, scope, profile, weight_profile, top_k)
            for scope in args.scopes
            for profile in args.profiles
            for weight_profile in args.weight_profiles
            for top_k in args.top_k_grid
        ]
        for cfg in configs:
            pred_oof = np.zeros(len(Xtr_base), dtype=float)
            pred_test = np.zeros(len(Xte_base), dtype=float)
            for fold in sorted(np.unique(folds)):
                tr_idx = np.where(folds != fold)[0]
                va_idx = np.where(folds == fold)[0]
                base_cols = scope_columns(Xtr_base, target, cfg.scope)
                w_all = sample_weights(y, mtr, cfg.weight_profile, target)
                prior_tr, prior_te = fold_subject_prior_features(y, mtr, mte, tr_idx)
                Xtr_fold_base = pd.concat([Xtr_base[base_cols], prior_tr], axis=1)
                Xte_fold_base = pd.concat([Xte_base[base_cols], prior_te], axis=1)
                always = [c for c in Xtr_fold_base.columns if c == "subject_id" or c.startswith(("anchor_", "sleep_", "life_", "subj_", "subject_num", "target_"))]
                cols = select_features(Xtr_fold_base, y, tr_idx, list(Xtr_fold_base.columns), always, cfg.top_k, w_all[tr_idx])
                X_train = Xtr_fold_base.iloc[tr_idx][cols]
                X_valid = Xtr_fold_base.iloc[va_idx][cols]
                X_test = Xte_fold_base[cols]
                X_train, X_valid, X_test, cat_cols = prepare_lgb_frames(X_train, X_valid, X_test)

                valid_seed_preds = []
                test_seed_preds = []
                seed_meta = []
                for seed in args.seeds:
                    pv, pt, meta = fit_lgbm(
                        PROFILES[cfg.profile],
                        X_train,
                        y[tr_idx],
                        w_all[tr_idx],
                        X_valid,
                        y[va_idx],
                        X_test,
                        cat_cols,
                        seed,
                        args.rounds,
                        args.early_stopping_rounds,
                        args.log_period,
                    )
                    valid_seed_preds.append(pv)
                    test_seed_preds.append(pt)
                    seed_meta.append(meta)
                pv = clip(np.mean(valid_seed_preds, axis=0))
                pt = clip(np.mean(test_seed_preds, axis=0))
                pred_oof[va_idx] = pv
                pred_test += pt / C.N_SPLITS
                fit_rows.append(
                    {
                        "target": target,
                        "config_key": cfg.key,
                        "scope": cfg.scope,
                        "profile": cfg.profile,
                        "weight_profile": cfg.weight_profile,
                        "top_k": cfg.top_k,
                        "fold": int(fold),
                        "features": len(cols),
                        "seed_count": len(args.seeds),
                        "valid_logloss": safe_loss(y[va_idx], pv),
                        "best_iteration_mean": float(np.nanmean([m["best_iteration"] for m in seed_meta])),
                        "best_valid_logloss_mean": float(np.nanmean([m["best_valid_logloss"] for m in seed_meta])),
                        "best_train_logloss_mean": float(np.nanmean([m["best_train_logloss"] for m in seed_meta])),
                        "stop_policy": seed_meta[0]["stop_policy"],
                        "fallback_logic": seed_meta[0]["fallback_logic"],
                        "weight_min": float(w_all[tr_idx].min()),
                        "weight_max": float(w_all[tr_idx].max()),
                    }
                )
            raw_oof[target][cfg.key] = clip(pred_oof)
            raw_test[target][cfg.key] = clip(pred_test)
            raw_full = safe_loss(y, pred_oof)
            raw_last = safe_loss(y[last_mask], pred_oof[last_mask])
            log(f"{target} {cfg.key} raw_full={raw_full:.6f} raw_last={raw_last:.6f}")

            for blend_mode in args.blend_modes:
                for model_weight in args.model_weight_grid:
                    blended = blend_with_anchor(anchor_oof[target].values, pred_oof, model_weight, blend_mode)
                    for intercept in args.intercepts:
                        for temp in args.temperatures:
                            p = calibrate_values(blended, intercept, temp)
                            fold_vals = [safe_loss(y[folds == f], p[folds == f]) for f in sorted(np.unique(folds))]
                            full = safe_loss(y, p)
                            last = fold_vals[-1]
                            fp = fp_summary(y, p)
                            search_rows.append(
                                {
                                    "target": target,
                                    "config_key": cfg.key,
                                    "scope": cfg.scope,
                                    "profile": cfg.profile,
                                    "weight_profile": cfg.weight_profile,
                                    "top_k": cfg.top_k,
                                    "blend_mode": blend_mode,
                                    "model_weight": model_weight,
                                    "intercept": intercept,
                                    "temperature": temp,
                                    "raw_full_logloss": raw_full,
                                    "raw_last_logloss": raw_last,
                                    "full_logloss": full,
                                    "last_logloss": last,
                                    "full_delta_vs_anchor": full - target_anchor_full,
                                    "last_delta_vs_anchor": last - target_anchor_last,
                                    "anchor_full_logloss": target_anchor_full,
                                    "anchor_last_logloss": target_anchor_last,
                                    "rank_score": target_rank_score(full, last, fold_vals, fp, target_anchor_full),
                                    **fp,
                                    **{f"fold{i}_logloss": val for i, val in enumerate(fold_vals)},
                                }
                            )

    fit_df = pd.DataFrame(fit_rows)
    search_df = pd.DataFrame(search_rows)
    fit_df.to_csv(out_dir / "fit_diagnostics.csv", index=False)
    search_df.sort_values(["target", "rank_score", "full_logloss"]).to_csv(out_dir / "target_config_search.csv", index=False)

    candidates = {}
    choice_tables = []
    for selection in ["composite", "fp_guard", "full", "last"]:
        name = f"raw_timeline_{selection}"
        po, pt, choices = build_candidate(name, selection, search_df, raw_oof, raw_test, anchor_oof, anchor_test, args.full_guard)
        choices.to_csv(out_dir / f"{name}_choices.csv", index=False)
        candidates[name] = (po, pt, choices)
        choice_tables.append(choices)

    score_rows = []
    fp_rows = []
    for name, (po, pt, _choices) in candidates.items():
        score_rows.append(score_candidate(name, ytr, po, folds, anchor_full, anchor_last))
        for target in TARGETS:
            fp_rows.append({"candidate": name, "target": target, **fp_summary(ytr[target].values, po[target].values)})
    scores = pd.DataFrame(score_rows).sort_values(["full_logloss", "last_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.DataFrame(fp_rows).to_csv(out_dir / "fp_fn_summary.csv", index=False)
    pd.concat(choice_tables, ignore_index=True, sort=False).to_csv(out_dir / "target_choices_all.csv", index=False)

    for _, row in scores.iterrows():
        name = str(row["candidate"])
        po, pt, _choices = candidates[name]
        stem = f"{safe_name(name)}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}"
        write_prediction(out_dir / f"{stem}_oof.csv", mtr, po, ytr)
        write_prediction(out_dir / f"{stem}_test_pred.csv", mte, pt)
        write_submission(sub_dir / f"{stem}.csv", mte, pt)

    report = {
        "purpose": "Target-specific compact raw timeline single-model sources with FP diagnostics.",
        "feature_matrix": {"train_shape": list(Xtr_base.shape), "test_shape": list(Xte_base.shape)},
        "scopes": args.scopes,
        "profiles": args.profiles,
        "weight_profiles": args.weight_profiles,
        "top_k_grid": args.top_k_grid,
        "seeds": args.seeds,
        "early_stopping_rounds": args.early_stopping_rounds,
        "anchor": {"full_logloss": anchor_full, "last_logloss": anchor_last},
        "candidate_scores": scores.to_dict(orient="records"),
        "notes": [
            "LightGBM uses validation logloss early stopping and predicts with best_iteration.",
            "Subject prior/recent label features are computed inside each fold from training rows only.",
            "Selection reports FP/FN pressure because current public-best overpredicts positives.",
            "These submissions are new source candidates; final decision should compare public/CV after run.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Raw Timeline Target Model",
        "",
        "## Candidate Scores",
        "",
        scores.to_string(index=False),
        "",
        "## FP/FN Summary",
        "",
        pd.DataFrame(fp_rows).to_string(index=False),
        "",
        "## Target Choices",
        "",
        pd.concat(choice_tables, ignore_index=True, sort=False)[
            [
                "candidate",
                "target",
                "scope",
                "profile",
                "weight_profile",
                "top_k",
                "blend_mode",
                "model_weight",
                "intercept",
                "temperature",
                "full_logloss",
                "last_logloss",
                "fp_rate",
                "fn_rate",
                "pred_pos_rate",
                "rank_score",
            ]
        ].to_string(index=False),
        "",
        "## Early Stopping",
        "",
        "- `fit_diagnostics.csv` records `best_iteration_mean`, `stop_policy`, and `fallback_logic`.",
    ]
    (out_dir / "RAW_TIMELINE_TARGET_MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Raw timeline candidate scores ===")
    print(scores[["candidate", "full_logloss", "last_logloss", "full_delta_vs_anchor", "last_delta_vs_anchor", "fold_std", "tail3_worst"]].to_string(index=False))
    print("\n=== FP/FN summary ===")
    print(pd.DataFrame(fp_rows).to_string(index=False))
    print("\n=== Target choices ===")
    print(pd.concat(choice_tables, ignore_index=True, sort=False)[
        ["candidate", "target", "scope", "profile", "weight_profile", "top_k", "blend_mode", "model_weight", "intercept", "temperature", "full_logloss", "last_logloss", "fp_rate", "fn_rate", "pred_pos_rate", "rank_score"]
    ].to_string(index=False))
    print("\nWrote outputs to", out_dir, "and", sub_dir)


if __name__ == "__main__":
    main()
