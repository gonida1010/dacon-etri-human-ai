"""Create pattern-filtered candidates after public/LB diagnostics.

The submitted constrained blend moved several targets whose last-fold movement
precision was weak.  This script creates small target-isolated candidates:

- Q1: keep the residual/full movement; both up/down directions are useful.
- S2: optionally keep only downward movement, because upward moves are noisy.
- S4: optionally keep only upward movement, because downward moves are noisy.
- Q2/Q3/S1/S3: anchor by default.

It writes OOF/test predictions and submission files for inspection.  It does
not train any model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def target_loss(y: pd.DataFrame, pred: pd.DataFrame, target: str, mask: np.ndarray) -> float:
    return safe_loss(y[target].values[mask], pred[target].values[mask])


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({t: clip(df[t].values) for t in TARGETS})


def load_anchor() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_bank = pd.read_csv(ROOT / "research/oof_sparse_greedy/oof_bank.csv")
    test_bank = pd.read_csv(ROOT / "research/oof_sparse_greedy/test_bank.csv")
    return (
        pd.DataFrame({t: clip(oof_bank[f"anchor__{t}"].values) for t in TARGETS}),
        pd.DataFrame({t: clip(test_bank[f"anchor__{t}"].values) for t in TARGETS}),
    )


def apply_mode(
    anchor: pd.Series,
    source: pd.Series,
    mode: str,
    alpha: float,
) -> np.ndarray:
    delta = source.values - anchor.values
    if mode == "anchor":
        return anchor.values
    if mode == "full":
        return source.values
    if mode == "up":
        return np.maximum(anchor.values, source.values)
    if mode == "down":
        return np.minimum(anchor.values, source.values)
    if mode == "up_damped":
        return clip(anchor.values + alpha * np.maximum(0.0, delta))
    if mode == "down_damped":
        return clip(anchor.values + alpha * np.minimum(0.0, delta))
    if mode == "damped":
        return clip(anchor.values + alpha * delta)
    raise ValueError(f"unknown mode: {mode}")


def build_candidate(
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    source_oof: pd.DataFrame,
    source_test: pd.DataFrame,
    q1: str,
    s2: str,
    s4: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | float]]:
    oof = anchor_oof.copy()
    test = anchor_test.copy()
    oof["Q1"] = apply_mode(anchor_oof["Q1"], source_oof["Q1"], q1, alpha)
    test["Q1"] = apply_mode(anchor_test["Q1"], source_test["Q1"], q1, alpha)
    oof["S2"] = apply_mode(anchor_oof["S2"], source_oof["S2"], s2, alpha)
    test["S2"] = apply_mode(anchor_test["S2"], source_test["S2"], s2, alpha)
    oof["S4"] = apply_mode(anchor_oof["S4"], source_oof["S4"], s4, alpha)
    test["S4"] = apply_mode(anchor_test["S4"], source_test["S4"], s4, alpha)
    return oof, test, {"Q1": q1, "S2": s2, "S4": s4, "alpha": alpha}


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="research/constrained_target_blend_logit_newton")
    p.add_argument("--source-name", default="full")
    p.add_argument("--output-dir", default="research/pattern_safe_candidates_20260622")
    p.add_argument("--submission-dir", default="submissions/pattern_safe_candidates_20260622")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading data/folds")
    _, ytr, _, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)

    log("Loading anchor and source predictions")
    anchor_oof, anchor_test = load_anchor()
    source_dir = ROOT / args.source_dir
    source_oof = read_pred(source_dir / f"{args.source_name}_oof.csv")
    source_test = read_pred(source_dir / f"{args.source_name}_test_pred.csv")

    specs = [
        ("q1_s2down_s4up", "full", "down", "up", 1.0),
        ("q1_s2down_s4full", "full", "down", "full", 1.0),
        ("q1_s2full_s4up", "full", "full", "up", 1.0),
        ("q1_s2down_only", "full", "down", "anchor", 1.0),
        ("q1_s4up_only", "full", "anchor", "up", 1.0),
        ("q1_only", "full", "anchor", "anchor", 1.0),
        ("s2down_s4up_noq1", "anchor", "down", "up", 1.0),
        ("s2down_only_noq1", "anchor", "down", "anchor", 1.0),
        ("q1_s2down75_s4up75", "full", "down_damped", "up_damped", 0.75),
        ("q1_s2down50_s4up50", "full", "down_damped", "up_damped", 0.50),
        ("q1_s2down25_s4up25", "full", "down_damped", "up_damped", 0.25),
    ]

    rows = []
    for name, q1, s2, s4, alpha in specs:
        oof, test, recipe = build_candidate(anchor_oof, anchor_test, source_oof, source_test, q1, s2, s4, alpha)
        fold_losses = [mean_loss(ytr, oof, folds == f) for f in sorted(np.unique(folds))]
        row = {
            "candidate": name,
            "full_logloss": mean_loss(ytr, oof, full_mask),
            "last_logloss": mean_loss(ytr, oof, last_mask),
            "fold_std": float(np.std(fold_losses)),
            "tail3_worst": float(max(fold_losses[-3:])),
            "test_abs_delta_mean_vs_anchor": float(np.mean(np.abs(test[TARGETS].values - anchor_test[TARGETS].values))),
            **{f"fold{i}_logloss": v for i, v in enumerate(fold_losses)},
            **{f"{target}_last_logloss": target_loss(ytr, oof, target, last_mask) for target in TARGETS},
            **{f"recipe_{k}": v for k, v in recipe.items()},
        }
        rows.append(row)
        safe = name.replace(".", "p")
        write_prediction(out_dir / f"{safe}_oof.csv", mtr, oof, ytr)
        write_prediction(out_dir / f"{safe}_test_pred.csv", mte, test)
        write_submission(
            sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv",
            mte,
            test,
        )
    scores = pd.DataFrame(rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    (out_dir / "report.json").write_text(
        json.dumps({
            "source_dir": args.source_dir,
            "source_name": args.source_name,
            "candidate_scores": scores.to_dict(orient="records"),
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== Pattern-safe candidate scores ===")
    print(scores[[
        "candidate",
        "full_logloss",
        "last_logloss",
        "fold_std",
        "tail3_worst",
        "test_abs_delta_mean_vs_anchor",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
