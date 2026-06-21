"""Kaggle-style ensemble pipeline for Dacon ETRI human-AI task.

What this adds over the old temporal-prior patch:
- OOF anchor features from the existing temporal-prior recipe
- Subject-hole CV from the public 0.5917 notebook idea
- LGBM / XGBoost / CatBoost target-wise models
- Fold-safe feature selection inside each validation fold
- Target-wise OOF blend search with both last-block and full-CV guards
- Submission candidates + diagnostic CSV/PNG outputs

Run examples:
  python -m src.kaggle_style_ensemble --models lgbm xgb --fold-limit 5
  python -m src.kaggle_style_ensemble --models lgbm xgb cat --fold-limit 5 --seeds 42 7 2024
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .train_temporal_prior import (
    RECIPES,
    build_recipe_predictions,
    clip,
    fit_lgbm_oof_test,
    temporal_oof,
    temporal_test,
)

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

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT
EPS = 1e-6


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS), labels=[0, 1]))


def mean_target_logloss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray | None = None) -> float:
    scores = []
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    for t in TARGETS:
        scores.append(safe_logloss(y[t].values[mask], pred[t].values[mask]))
    return float(np.mean(scores))


def subject_hole_folds(meta: pd.DataFrame, n_folds: int = 5, n_chunks: int = 10) -> np.ndarray:
    """Subject-wise chronological hole folds.

    Each subject is sorted by sleep_date and split into n_chunks. Fold k validates
    chunks k and k+n_folds. This mirrors the public notebook's idea: every fold
    sees each subject, but validation days are time-separated holes.
    """
    fold = np.full(len(meta), -1, dtype=int)
    m = meta.sort_values(["subject_id", "sleep_date"])  # keeps original indices
    for _, g in m.groupby("subject_id", sort=False):
        idx = g.index.to_numpy()
        chunks = np.array_split(idx, n_chunks)
        for k in range(n_folds):
            valid_parts = []
            for hole in (k, k + n_folds):
                if hole < len(chunks):
                    valid_parts.append(chunks[hole])
            if valid_parts:
                fold[np.concatenate(valid_parts)] = k
    assert (fold >= 0).all(), "subject_hole_folds left unassigned rows"
    return fold


def add_anchor_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    cols = []
    for t in TARGETS:
        ptr = np.clip(anchor_oof[t].values.astype(float), EPS, 1 - EPS)
        pte = np.clip(anchor_test[t].values.astype(float), EPS, 1 - EPS)
        Xtr2[f"anchor_{t}_prob"] = ptr
        Xte2[f"anchor_{t}_prob"] = pte
        Xtr2[f"anchor_{t}_logit"] = np.log(ptr / (1 - ptr))
        Xte2[f"anchor_{t}_logit"] = np.log(pte / (1 - pte))
        cols += [f"anchor_{t}_prob", f"anchor_{t}_logit"]
    # Cross-target summary: helps borrow strength without using true labels.
    Xtr2["anchor_q_mean"] = anchor_oof[["Q1", "Q2", "Q3"]].mean(axis=1).values
    Xte2["anchor_q_mean"] = anchor_test[["Q1", "Q2", "Q3"]].mean(axis=1).values
    Xtr2["anchor_s_mean"] = anchor_oof[["S1", "S2", "S3", "S4"]].mean(axis=1).values
    Xte2["anchor_s_mean"] = anchor_test[["S1", "S2", "S3", "S4"]].mean(axis=1).values
    cols += ["anchor_q_mean", "anchor_s_mean"]
    return Xtr2, Xte2, cols


def add_window_pair_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    max_pair_bases: int = 220,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create L/S same-sensor pair features.

    Public 0.5917 notes mention window-pair features. Here we implement a
    conservative version: for columns that exist both as L_* and S_*, add
    sleep-minus-lifelog delta and ratio. This is label-free and test-available.
    """
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    l_cols = [c for c in Xtr.columns if c.startswith("L_") and pd.api.types.is_numeric_dtype(Xtr[c])]
    pairs = []
    for l_col in l_cols:
        suffix = l_col[2:]
        s_col = f"S_{suffix}"
        if s_col in Xtr.columns and pd.api.types.is_numeric_dtype(Xtr[s_col]):
            # Prefer dense pairs. Sparse sensor pairs are mostly noise.
            coverage = float(Xtr[[l_col, s_col]].notna().mean().mean())
            pairs.append((coverage, suffix, l_col, s_col))
    pairs = sorted(pairs, reverse=True)[:max_pair_bases]
    new_cols = []
    for _, suffix, l_col, s_col in pairs:
        safe = suffix.replace("/", "_").replace(" ", "_")
        d_col = f"pair_delta_{safe}"
        r_col = f"pair_ratio_{safe}"
        Xtr2[d_col] = Xtr2[s_col] - Xtr2[l_col]
        Xte2[d_col] = Xte2[s_col] - Xte2[l_col]
        Xtr2[r_col] = Xtr2[s_col] / (Xtr2[l_col].abs() + 1e-3)
        Xte2[r_col] = Xte2[s_col] / (Xte2[l_col].abs() + 1e-3)
        new_cols.extend([d_col, r_col])
    return Xtr2, Xte2, new_cols


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(X[c])]


def select_fold_features(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    target: str,
    anchor_cols: list[str],
    top_k: int,
) -> list[str]:
    """Fold-safe stable feature filter by train-only correlation.

    This is deliberately simple and auditable. It avoids selecting features from
    validation labels while keeping anchor and subject_id always available.
    """
    num_cols = numeric_columns(X)
    if top_k <= 0 or top_k >= len(num_cols):
        cols = num_cols
    else:
        Xt = X.iloc[train_idx][num_cols]
        yt = pd.Series(y[train_idx], index=Xt.index)
        scores = []
        ystd = float(yt.std())
        for c in num_cols:
            s = Xt[c]
            if s.notna().sum() < 8 or float(s.std(skipna=True) or 0) == 0 or ystd == 0:
                score = 0.0
            else:
                score = abs(float(s.corr(yt)))
                if math.isnan(score):
                    score = 0.0
            scores.append((score, c))
        cols = [c for _, c in sorted(scores, reverse=True)[:top_k]]
        for c in anchor_cols:
            if c not in cols and c in X.columns:
                cols.append(c)
    if "subject_id" in X.columns:
        cols.append("subject_id")
    # de-duplicate, preserve order
    return list(dict.fromkeys(cols))


def prepare_matrix(Xtr: pd.DataFrame, Xva: pd.DataFrame, Xte: pd.DataFrame, cols: list[str], model_name: str):
    Xtrm = Xtr[cols].copy()
    Xvam = Xva[cols].copy()
    Xtem = Xte[cols].copy()
    if "subject_id" in cols:
        if model_name == "cat":
            for d in (Xtrm, Xvam, Xtem):
                d["subject_id"] = d["subject_id"].astype(str)
        else:
            for d in (Xtrm, Xvam, Xtem):
                d["subject_id"] = d["subject_id"].astype("category")
    return Xtrm, Xvam, Xtem


def fit_one_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    seed: int,
    log_period: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if model_name == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        dtr = lgb.Dataset(X_train, label=y_train, categorical_feature=["subject_id"] if "subject_id" in X_train.columns else "auto", free_raw_data=False)
        dva = lgb.Dataset(X_valid, label=y_valid, categorical_feature=["subject_id"] if "subject_id" in X_valid.columns else "auto", free_raw_data=False)
        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.018,
            num_leaves=15,
            min_child_samples=28,
            feature_fraction=0.72,
            bagging_fraction=0.82,
            bagging_freq=1,
            lambda_l1=0.8,
            lambda_l2=3.0,
            verbosity=-1,
            seed=seed,
        )
        model = lgb.train(
            params,
            dtr,
            num_boost_round=2600,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(log_period)],
        )
        best_iter = int(model.best_iteration or 0)
        meta = {
            "best_iteration": best_iter,
            "best_valid_logloss": float(model.best_score["valid"]["binary_logloss"]),
            "best_train_logloss": float(model.best_score["train"]["binary_logloss"]),
            "early_stop_metric": "binary_logloss",
            "prediction_iteration": best_iter,
            "fallback_logic": "use LightGBM best_iteration from validation logloss",
        }
        return (
            clip(model.predict(X_valid, num_iteration=best_iter)),
            clip(model.predict(X_test, num_iteration=best_iter)),
            meta,
        )

    if model_name == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=2600,
            learning_rate=0.018,
            max_depth=3,
            min_child_weight=8,
            subsample=0.82,
            colsample_bytree=0.72,
            reg_alpha=0.8,
            reg_lambda=5.0,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            early_stopping_rounds=120,
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=log_period)
        best_iter = int(getattr(model, "best_iteration", 0) or 0)
        pred_kwargs = {"iteration_range": (0, best_iter + 1)} if best_iter > 0 else {}
        evals = model.evals_result()
        valid_curve = evals.get("validation_1", {}).get("logloss", [])
        train_curve = evals.get("validation_0", {}).get("logloss", [])
        meta = {
            "best_iteration": best_iter,
            "best_valid_logloss": float(valid_curve[best_iter]) if valid_curve and best_iter < len(valid_curve) else np.nan,
            "best_train_logloss": float(train_curve[best_iter]) if train_curve and best_iter < len(train_curve) else np.nan,
            "early_stop_metric": "logloss",
            "prediction_iteration": best_iter,
            "fallback_logic": "use XGBoost best_iteration via iteration_range",
        }
        return (
            clip(model.predict_proba(X_valid, **pred_kwargs)[:, 1]),
            clip(model.predict_proba(X_test, **pred_kwargs)[:, 1]),
            meta,
        )

    if model_name == "cat":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        cat_features = ["subject_id"] if "subject_id" in X_train.columns else []
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=3200,
            learning_rate=0.018,
            depth=4,
            l2_leaf_reg=8.0,
            random_strength=1.2,
            bootstrap_type="Bernoulli",
            subsample=0.82,
            random_seed=seed,
            early_stopping_rounds=160,
            verbose=log_period,
            allow_writing_files=False,
            cat_features=cat_features,
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        best_iter = int(model.get_best_iteration() or 0)
        best_score = model.get_best_score()
        meta = {
            "best_iteration": best_iter,
            "best_valid_logloss": float(best_score.get("validation", {}).get("Logloss", np.nan)),
            "best_train_logloss": float(best_score.get("learn", {}).get("Logloss", np.nan)),
            "early_stop_metric": "Logloss",
            "prediction_iteration": best_iter,
            "fallback_logic": "CatBoost use_best_model shrinks to validation best_iteration",
        }
        return clip(model.predict_proba(X_valid)[:, 1]), clip(model.predict_proba(X_test)[:, 1]), meta

    raise ValueError(model_name)


def train_model_bank(
    model_name: str,
    Xtr: pd.DataFrame,
    ytr: pd.DataFrame,
    Xte: pd.DataFrame,
    folds: np.ndarray,
    last_mask: np.ndarray,
    anchor_cols: list[str],
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    seeds: list[int],
    top_k: int,
    fold_limit: int,
    log_period: int,
    target_history_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    oof = pd.DataFrame(0.0, index=ytr.index, columns=TARGETS)
    test = pd.DataFrame(0.0, index=Xte.index, columns=TARGETS)
    diagnostics: list[dict] = []
    active_folds = sorted(np.unique(folds))[:fold_limit]
    denom_test = len(active_folds) * len(seeds)
    for target in TARGETS:
        Xtr_target = Xtr
        Xte_target = Xte
        hist_cols: list[str] = []
        if target_history_features:
            Xtr_target = Xtr.copy()
            Xte_target = Xte.copy()
            for method in ["mean_sm4", "mean_sm16", "last2_sm4", "last5_sm4", "last20_sm4", "ridge1", "ridge10"]:
                col = f"hist_{target}_{method}"
                Xtr_target[col] = temporal_oof(ytr, mtr, folds, target, method)
                Xte_target[col] = temporal_test(ytr, mtr, mte, target, method)
                hist_cols.append(col)
            log(f"{model_name} target={target} added_target_history_features={len(hist_cols)}")
        y = ytr[target].values.astype(int)
        for fold in active_folds:
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            cols = select_fold_features(Xtr_target, y, tr_idx, target, anchor_cols + hist_cols, top_k)
            for seed in seeds:
                log(f"{model_name} target={target} fold={fold+1}/{len(active_folds)} seed={seed} features={len(cols)} train={len(tr_idx)} valid={len(va_idx)}")
                X_train, X_valid, X_test = prepare_matrix(Xtr_target.iloc[tr_idx], Xtr_target.iloc[va_idx], Xte_target, cols, model_name)
                p_val, p_test, model_meta = fit_one_model(model_name, X_train, y[tr_idx], X_valid, y[va_idx], X_test, seed, log_period)
                log(
                    f"{model_name} target={target} fold={fold+1} seed={seed} "
                    f"best_iter={model_meta['best_iteration']} "
                    f"valid_logloss={model_meta['best_valid_logloss']:.6f} "
                    f"train_logloss={model_meta['best_train_logloss']:.6f}"
                )
                oof.loc[va_idx, target] += p_val / len(seeds)
                test[target] += p_test / denom_test
            fold_pred = oof.loc[va_idx, target].values
            diagnostics.append({
                "model": model_name,
                "target": target,
                "fold": int(fold),
                "logloss": safe_logloss(y[va_idx], fold_pred),
                "last_overlap_rows": int(last_mask[va_idx].sum()),
                "features": len(cols),
                "seed_count": len(seeds),
            })
    return oof, test, diagnostics


def source_blend_grid(sources: dict[str, pd.DataFrame], step: float = 0.1) -> Iterable[tuple[str, dict[str, float], pd.DataFrame]]:
    names = list(sources)
    # single sources
    for n in names:
        yield n, {n: 1.0}, sources[n]
    # two-source grids are enough for only 450 train rows; three-way search overfits fast.
    ws = np.round(np.arange(step, 1.0, step), 4)
    for a, b in itertools.combinations(names, 2):
        for w in ws:
            pred = pd.DataFrame(clip(w * sources[a].values + (1 - w) * sources[b].values), columns=TARGETS)
            yield f"{a}{w:.1f}_{b}{1-w:.1f}", {a: float(w), b: float(1 - w)}, pred


def targetwise_blend_search(
    ytr: pd.DataFrame,
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
    last_mask: np.ndarray,
    full_guard: float,
    min_last_gain: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    out_test = pd.DataFrame(index=next(iter(test_sources.values())).index, columns=TARGETS, dtype=float)
    rows = []
    for t in TARGETS:
        base_full = safe_logloss(ytr[t].values, train_sources["anchor"][t].values)
        base_last = safe_logloss(ytr[t].values[last_mask], train_sources["anchor"][t].values[last_mask])
        candidates = []
        names = list(train_sources)
        for n in names:
            candidates.append((n, {n: 1.0}, train_sources[n][t].values, test_sources[n][t].values))
        for a, b in itertools.combinations(names, 2):
            for w in np.round(np.arange(0.1, 1.0, 0.1), 4):
                po = clip(w * train_sources[a][t].values + (1 - w) * train_sources[b][t].values)
                pt = clip(w * test_sources[a][t].values + (1 - w) * test_sources[b][t].values)
                candidates.append((f"{a}{w:.1f}_{b}{1-w:.1f}", {a: float(w), b: float(1 - w)}, po, pt))
        scored = []
        for name, weights, po, pt in candidates:
            full = safe_logloss(ytr[t].values, po)
            last = safe_logloss(ytr[t].values[last_mask], po[last_mask])
            scored.append((last, full, name, weights, po, pt))
        scored.sort(key=lambda x: (x[0], x[1]))
        # Guard: must not destroy full OOF too much compared to anchor.
        chosen = None
        fallback_reason = ""
        for last, full, name, weights, po, pt in scored:
            if full <= base_full + full_guard and last <= base_last - min_last_gain:
                chosen = (last, full, name, weights, po, pt)
                break
        if chosen is None:
            fallback_reason = f"fallback_to_anchor: no candidate passed full_guard={full_guard} and min_last_gain={min_last_gain}"
            chosen = (base_last, base_full, "anchor_guard_fallback", {"anchor": 1.0}, train_sources["anchor"][t].values, test_sources["anchor"][t].values)
        last, full, name, weights, po, pt = chosen
        if not fallback_reason:
            if name == "anchor":
                fallback_reason = "anchor retained: no blend beat the anchor under the active ranking"
            else:
                fallback_reason = "accepted: candidate improved last-block enough and stayed inside full-CV guard"
        out_oof[t] = clip(po)
        out_test[t] = clip(pt)
        rows.append({
            "target": t,
            "chosen": name,
            "weights": json.dumps(weights, ensure_ascii=False),
            "full_logloss": full,
            "last_logloss": last,
            "anchor_full": base_full,
            "anchor_last": base_last,
            "full_delta_vs_anchor": full - base_full,
            "last_delta_vs_anchor": last - base_last,
            "fallback_reason": fallback_reason,
        })
    return out_oof, out_test, pd.DataFrame(rows)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def shift_probs_to_mean(probs: np.ndarray, target_mean: float) -> np.ndarray:
    """Rank-preserving probability calibration to a desired mean.

    This keeps row ordering from the model but shifts the logit intercept so the
    group average matches the structural target rate.
    """
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    target_mean = float(np.clip(target_mean, EPS, 1 - EPS))
    z = _logit(probs)
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _sigmoid(z + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return np.clip(_sigmoid(z + (lo + hi) / 2.0), EPS, 1 - EPS)


def apply_q_balance_constraint(
    pred_oof: pd.DataFrame,
    pred_test: pd.DataFrame,
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    balance_rate: float,
    q_targets: list[str] | None = None,
    min_group_rate: float = 0.02,
    max_group_rate: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply train/test whole-period balance assumption to Q targets.

    For Q targets, the binary label is defined relative to each subject's long
    period average. In the real test case, train labels are known and test labels
    should roughly complete the subject's whole-period positive count. In OOF,
    each validation fold is calibrated using only the other folds' labels.
    """
    q_targets = q_targets or ["Q1", "Q2", "Q3"]
    out_oof = pred_oof.copy()
    out_test = pred_test.copy()
    rows = []
    subjects = sorted(mtr["subject_id"].astype(str).unique())

    for target in q_targets:
        for fold in sorted(np.unique(folds)):
            for subject in subjects:
                va_mask = (folds == fold) & (mtr["subject_id"].astype(str).values == subject)
                if not va_mask.any():
                    continue
                tr_mask = (folds != fold) & (mtr["subject_id"].astype(str).values == subject)
                n_train = int(tr_mask.sum())
                n_valid = int(va_mask.sum())
                train_pos = float(ytr.loc[tr_mask, target].sum())
                expected_valid_pos = balance_rate * (n_train + n_valid) - train_pos
                target_mean = float(np.clip(expected_valid_pos / n_valid, min_group_rate, max_group_rate))
                before = out_oof.loc[va_mask, target].values
                out_oof.loc[va_mask, target] = shift_probs_to_mean(before, target_mean)
                rows.append({
                    "split": "oof",
                    "target": target,
                    "fold": int(fold),
                    "subject_id": subject,
                    "balance_rate": balance_rate,
                    "n_train": n_train,
                    "n_calibrate": n_valid,
                    "train_pos_rate": train_pos / max(n_train, 1),
                    "target_group_mean": target_mean,
                    "before_group_mean": float(np.mean(before)),
                    "after_group_mean": float(out_oof.loc[va_mask, target].mean()),
                })

        for subject in subjects:
            te_mask = mte["subject_id"].astype(str).values == subject
            tr_mask = mtr["subject_id"].astype(str).values == subject
            if not te_mask.any():
                continue
            n_train = int(tr_mask.sum())
            n_test = int(te_mask.sum())
            train_pos = float(ytr.loc[tr_mask, target].sum())
            expected_test_pos = balance_rate * (n_train + n_test) - train_pos
            target_mean = float(np.clip(expected_test_pos / n_test, min_group_rate, max_group_rate))
            before = out_test.loc[te_mask, target].values
            out_test.loc[te_mask, target] = shift_probs_to_mean(before, target_mean)
            rows.append({
                "split": "test",
                "target": target,
                "fold": -1,
                "subject_id": subject,
                "balance_rate": balance_rate,
                "n_train": n_train,
                "n_calibrate": n_test,
                "train_pos_rate": train_pos / max(n_train, 1),
                "target_group_mean": target_mean,
                "before_group_mean": float(np.mean(before)),
                "after_group_mean": float(out_test.loc[te_mask, target].mean()),
            })

    return out_oof, out_test, pd.DataFrame(rows)


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    sub = meta_test.copy()
    for t in TARGETS:
        sub[t] = clip(pred[t].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)


def plot_scores(out_dir: Path, summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable, skip plots: {e}")
        return
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["full_logloss"], width=0.36, label="full CV")
    ax.bar(x + 0.18, summary["last_logloss"], width=0.36, label="last-block")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["candidate"], rotation=35, ha="right")
    ax.set_ylabel("Average Log-Loss (lower is better)")
    ax.set_title("Kaggle-Style Ensemble Candidate Scores")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_scores.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research/kaggle_style_ensemble")
    parser.add_argument("--submission-dir", default="submissions/kaggle_style_ensemble")
    parser.add_argument("--models", nargs="+", default=["lgbm", "xgb", "cat"], choices=["lgbm", "xgb", "cat"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 2024])
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=260)
    parser.add_argument("--subject-hole-chunks", type=int, default=10)
    parser.add_argument("--pair-features", action="store_true")
    parser.add_argument("--target-history-features", action="store_true")
    parser.add_argument("--full-guard", type=float, default=0.004)
    parser.add_argument("--min-last-gain", type=float, default=0.0)
    parser.add_argument("--q-balance-rates", nargs="*", type=float, default=[])
    parser.add_argument("--log-period", type=int, default=250)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    log("Loading data and base features")
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    last_folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    last_mask = last_folds == (C.N_SPLITS - 1)
    hole_folds = subject_hole_folds(mtr, n_folds=C.N_SPLITS, n_chunks=args.subject_hole_chunks)

    log("Building existing temporal-prior anchor OOF/test")
    anchor_model_oof, _, anchor_model_test = fit_lgbm_oof_test(Xtr, ytr, Xte, mtr, mte, last_folds)
    anchor_oof, anchor_test = build_recipe_predictions(ytr, mtr, mte, last_folds, anchor_model_oof, anchor_model_test)
    log(f"anchor full={mean_target_logloss(ytr, anchor_oof):.6f} last={mean_target_logloss(ytr, anchor_oof, last_mask):.6f}")

    Xtr2, Xte2, anchor_cols = add_anchor_features(Xtr, Xte, anchor_oof, anchor_test)
    pair_cols: list[str] = []
    if args.pair_features:
        Xtr2, Xte2, pair_cols = add_window_pair_features(Xtr2, Xte2)
        log(f"Added window-pair features: {len(pair_cols)}")

    train_sources = {"anchor": anchor_oof.reset_index(drop=True)}
    test_sources = {"anchor": anchor_test.reset_index(drop=True)}
    fold_diag = []
    for model_name in args.models:
        log(f"Training model bank: {model_name}")
        oof, test, diag = train_model_bank(
            model_name,
            Xtr2,
            ytr,
            Xte2,
            hole_folds,
            last_mask,
            anchor_cols,
            mtr,
            mte,
            args.seeds,
            args.top_k,
            args.fold_limit,
            args.log_period,
            args.target_history_features,
        )
        train_sources[model_name] = oof.reset_index(drop=True)
        test_sources[model_name] = test.reset_index(drop=True)
        fold_diag.extend(diag)
        log(f"{model_name} full={mean_target_logloss(ytr, oof):.6f} last={mean_target_logloss(ytr, oof, last_mask):.6f}")

    candidate_rows = []
    candidate_preds: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, weights, pred in source_blend_grid(train_sources, step=0.1):
        test_pred = pd.DataFrame(0.0, index=Xte.index, columns=TARGETS)
        for src, w in weights.items():
            test_pred += w * test_sources[src].values
        pred = pd.DataFrame(clip(pred.values), columns=TARGETS)
        test_pred = pd.DataFrame(clip(test_pred.values), columns=TARGETS)
        full = mean_target_logloss(ytr, pred)
        last = mean_target_logloss(ytr, pred, last_mask)
        candidate_rows.append({"candidate": name, "weights": json.dumps(weights), "full_logloss": full, "last_logloss": last})
        candidate_preds[name] = (pred, test_pred)

    guarded_oof, guarded_test, targetwise = targetwise_blend_search(
        ytr,
        train_sources,
        test_sources,
        last_mask,
        args.full_guard,
        args.min_last_gain,
    )
    candidate_rows.append({
        "candidate": "targetwise_guarded",
        "weights": "targetwise",
        "full_logloss": mean_target_logloss(ytr, guarded_oof),
        "last_logloss": mean_target_logloss(ytr, guarded_oof, last_mask),
    })
    candidate_preds["targetwise_guarded"] = (guarded_oof, guarded_test)

    q_balance_rows = []
    if args.q_balance_rates:
        log(f"Applying Q balance structural constraint rates={args.q_balance_rates}")
    base_candidate_names = list(candidate_preds)
    for rate in args.q_balance_rates:
        for cname in base_candidate_names:
            if cname not in {"targetwise_guarded", "anchor", "anchor0.8_lgbm0.2", "anchor0.7_lgbm0.3", "anchor0.6_lgbm0.4"}:
                continue
            pred_oof, pred_test = candidate_preds[cname]
            q_oof, q_test, q_diag = apply_q_balance_constraint(
                pred_oof,
                pred_test,
                ytr,
                mtr,
                mte,
                hole_folds,
                balance_rate=rate,
            )
            qname = f"qbal{str(rate).replace('.', 'p')}_{cname}"
            full = mean_target_logloss(ytr, q_oof)
            last = mean_target_logloss(ytr, q_oof, last_mask)
            candidate_rows.append({
                "candidate": qname,
                "weights": f"q_balance_rate={rate}; base={cname}",
                "full_logloss": full,
                "last_logloss": last,
            })
            candidate_preds[qname] = (q_oof, q_test)
            q_diag.insert(0, "candidate", qname)
            q_balance_rows.append(q_diag)
            log(f"q-balance candidate={qname} full={full:.6f} last={last:.6f}")

    summary = pd.DataFrame(candidate_rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    summary.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.DataFrame(fold_diag).to_csv(out_dir / "fold_diagnostics.csv", index=False)
    targetwise.to_csv(out_dir / "targetwise_blend.csv", index=False)
    if q_balance_rows:
        pd.concat(q_balance_rows, ignore_index=True).to_csv(out_dir / "q_balance_diagnostics.csv", index=False)

    # Save the top guarded/diagnostic candidates, but avoid dumping dozens of files.
    saved = []
    for i, row in summary.head(6).iterrows():
        cname = row["candidate"]
        _, test_pred = candidate_preds[cname]
        safe_name = cname.replace(".", "p").replace(":", "_").replace("/", "_")
        path = sub_dir / f"{i+1:02d}_{safe_name}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv"
        write_submission(path, mte, test_pred)
        saved.append(str(path.relative_to(ROOT)))

    plot_scores(out_dir, summary.head(12))
    report = {
        "purpose": "Kaggle-style anchor + subject-hole CV + LGBM/XGB/CatBoost blend search.",
        "models": args.models,
        "seeds": args.seeds,
        "fold_limit": args.fold_limit,
        "top_k": args.top_k,
        "subject_hole_chunks": args.subject_hole_chunks,
        "pair_features": args.pair_features,
        "pair_feature_count": len(pair_cols),
        "target_history_features": args.target_history_features,
        "full_guard": args.full_guard,
        "min_last_gain": args.min_last_gain,
        "q_balance_rates": args.q_balance_rates,
        "anchor_full": mean_target_logloss(ytr, anchor_oof),
        "anchor_last": mean_target_logloss(ytr, anchor_oof, last_mask),
        "best_candidates": summary.head(10).to_dict(orient="records"),
        "saved_submissions": saved,
        "elapsed_sec": round(time.time() - start, 2),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Candidate scores ===")
    print(summary.head(15).to_string(index=False))
    print("\n=== Target-wise choices ===")
    print(targetwise.to_string(index=False))
    print("\n=== Saved submissions ===")
    for p in saved:
        print(p)
    log(f"Done in {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()
