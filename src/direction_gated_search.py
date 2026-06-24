"""Direction-gated target search after public-feedback diagnostics.

The public result showed that a locally strong full/logit candidate still moved
some target directions that were weak in the last OOF block.  This script builds
target-level actions from existing OOF/test prediction pairs and only allows a
direction when the corresponding OOF movement is precise enough.

It does not train base models.  It is a post-processing/selection experiment.
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
ID_COLS = C.ID_COLS

DEFAULT_SOURCE_DIRS = [
    "research/constrained_target_blend_logit_newton",
    "research/pattern_safe_candidates_20260622",
    "research/residual_submission_blend_ridge_logit_newton",
    "research/residual_single_model_opt_ridge_logit_newton",
    "research/residual_submission_blend",
    "research/residual_single_model_opt_ridge",
    "research/residual_single_model_opt_hist_gb",
    "research/residual_single_model_opt_ridge_logit_te_full",
    "research/residual_submission_blend_ridge_logit_te_full",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def row_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = clip(p)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def target_loss(y: pd.DataFrame, pred: pd.DataFrame, target: str, mask: np.ndarray) -> float:
    return safe_loss(y[target].values[mask], pred[target].values[mask])


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_anchor() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_bank = pd.read_csv(ROOT / "research/oof_sparse_greedy/oof_bank.csv")
    test_bank = pd.read_csv(ROOT / "research/oof_sparse_greedy/test_bank.csv")
    return (
        pd.DataFrame({t: clip(oof_bank[f"anchor__{t}"].values) for t in TARGETS}),
        pd.DataFrame({t: clip(test_bank[f"anchor__{t}"].values) for t in TARGETS}),
    )


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path} missing targets: {missing}")
    return pd.DataFrame({t: clip(df[t].values) for t in TARGETS})


def discover_pairs(items: list[str]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for item in items:
        path = rel(item)
        if not path.exists():
            continue
        files = sorted(path.glob("*_oof.csv")) if path.is_dir() else [path]
        for oof_path in files:
            if not oof_path.name.endswith("_oof.csv") or oof_path.name.startswith("anchor_"):
                continue
            stem = oof_path.name[:-8]
            test_path = oof_path.with_name(f"{stem}_test_pred.csv")
            if not test_path.exists():
                continue
            name = f"{oof_path.parent.name}/{stem}"
            out[name] = (read_pred(oof_path), read_pred(test_path))
    return out


def apply_action(anchor: np.ndarray, source: np.ndarray, mode: str, alpha: float, threshold: float) -> np.ndarray:
    delta = source - anchor
    if mode == "full":
        return source
    if mode == "damped":
        return clip(anchor + alpha * delta)
    if mode == "up":
        return clip(anchor + alpha * np.maximum(0.0, delta))
    if mode == "down":
        return clip(anchor + alpha * np.minimum(0.0, delta))
    if mode == "up_thr":
        active = delta >= threshold
        return clip(anchor + alpha * np.where(active, delta, 0.0))
    if mode == "down_thr":
        active = delta <= -threshold
        return clip(anchor + alpha * np.where(active, delta, 0.0))
    raise ValueError(f"unknown mode: {mode}")


def action_masks(anchor: np.ndarray, source: np.ndarray, mode: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    delta = source - anchor
    if mode in {"full", "damped"}:
        return delta >= threshold, delta <= -threshold
    if mode == "up":
        return delta >= threshold, np.zeros(len(delta), dtype=bool)
    if mode == "down":
        return np.zeros(len(delta), dtype=bool), delta <= -threshold
    if mode == "up_thr":
        return delta >= threshold, np.zeros(len(delta), dtype=bool)
    if mode == "down_thr":
        return np.zeros(len(delta), dtype=bool), delta <= -threshold
    raise ValueError(f"unknown mode: {mode}")


def direction_stats(
    y: np.ndarray,
    pred: np.ndarray,
    anchor: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> dict[str, float | int]:
    base_pos = float(np.mean(y))
    up_precision = float(np.mean(y[up])) if up.any() else np.nan
    down_precision = float(np.mean(1 - y[down])) if down.any() else np.nan
    up_gain = (
        float(np.mean(row_loss(y[up], anchor[up]) - row_loss(y[up], pred[up])))
        if up.any()
        else np.nan
    )
    down_gain = (
        float(np.mean(row_loss(y[down], anchor[down]) - row_loss(y[down], pred[down])))
        if down.any()
        else np.nan
    )
    return {
        "base_positive_rate": base_pos,
        "up_n": int(up.sum()),
        "down_n": int(down.sum()),
        "up_precision_actual_1": up_precision,
        "down_precision_actual_0": down_precision,
        "up_lift_vs_base": up_precision - base_pos if np.isfinite(up_precision) else np.nan,
        "down_lift_vs_base0": down_precision - (1.0 - base_pos) if np.isfinite(down_precision) else np.nan,
        "up_logloss_gain_vs_anchor": up_gain,
        "down_logloss_gain_vs_anchor": down_gain,
    }


def build_action_table(
    y: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    thresholds: list[float],
    alphas: list[float],
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    rows = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    modes = ["full", "up", "down", "up_thr", "down_thr", "damped"]
    for target in TARGETS:
        yv = y[target].values
        anchor_train = anchor_oof[target].values
        anchor_public = anchor_test[target].values
        anchor_full = safe_loss(yv[full_mask], anchor_train[full_mask])
        anchor_last = safe_loss(yv[last_mask], anchor_train[last_mask])
        for source_name, (source_oof, source_test) in sources.items():
            src_train = source_oof[target].values
            src_public = source_test[target].values
            if np.allclose(src_train, anchor_train) and np.allclose(src_public, anchor_public):
                continue
            for threshold in thresholds:
                for mode in modes:
                    use_alphas = alphas if mode in {"up", "down", "up_thr", "down_thr", "damped"} else [1.0]
                    for alpha in use_alphas:
                        pred_train = apply_action(anchor_train, src_train, mode, alpha, threshold)
                        pred_public = apply_action(anchor_public, src_public, mode, alpha, threshold)
                        action_key = f"{target}::{source_name}::{mode}::a{alpha:g}::t{threshold:g}"
                        if action_key in arrays:
                            continue
                        up_last, down_last = action_masks(
                            anchor_train[last_mask],
                            src_train[last_mask],
                            mode,
                            threshold,
                        )
                        test_up, test_down = action_masks(anchor_public, src_public, mode, threshold)
                        stats = direction_stats(
                            yv[last_mask],
                            pred_train[last_mask],
                            anchor_train[last_mask],
                            up_last,
                            down_last,
                        )
                        full_loss = safe_loss(yv[full_mask], pred_train[full_mask])
                        last_loss = safe_loss(yv[last_mask], pred_train[last_mask])
                        test_delta = pred_public - anchor_public
                        up_bad = (
                            max(0.0, 0.5 - float(stats["up_precision_actual_1"]))
                            if np.isfinite(stats["up_precision_actual_1"])
                            else 0.0
                        )
                        down_bad = (
                            max(0.0, 0.5 - float(stats["down_precision_actual_0"]))
                            if np.isfinite(stats["down_precision_actual_0"])
                            else 0.0
                        )
                        rows.append({
                            "action_key": action_key,
                            "target": target,
                            "source": source_name,
                            "mode": mode,
                            "alpha": alpha,
                            "threshold": threshold,
                            "full_logloss": full_loss,
                            "last_logloss": last_loss,
                            "full_delta_vs_anchor": full_loss - anchor_full,
                            "last_delta_vs_anchor": last_loss - anchor_last,
                            "test_abs_delta_mean": float(np.mean(np.abs(test_delta))),
                            "test_mean_delta": float(np.mean(test_delta)),
                            "test_up_rate": float(np.mean(test_up)),
                            "test_down_rate": float(np.mean(test_down)),
                            "risk_proxy": float(np.mean(test_up)) * up_bad + float(np.mean(test_down)) * down_bad,
                            **stats,
                        })
                        arrays[action_key] = (pred_train, pred_public)
    table = pd.DataFrame(rows)
    table = table.sort_values(["target", "last_logloss", "full_logloss", "risk_proxy"]).reset_index(drop=True)
    return table, arrays


def pass_precision(row: pd.Series, min_precision: float, min_n: int) -> bool:
    up_n = int(row["up_n"])
    down_n = int(row["down_n"])
    if up_n >= min_n and float(row["up_precision_actual_1"]) < min_precision:
        return False
    if down_n >= min_n and float(row["down_precision_actual_0"]) < min_precision:
        return False
    if up_n < min_n and down_n < min_n:
        return False
    return True


def select_actions(
    table: pd.DataFrame,
    profile: dict[str, float | int | str],
) -> pd.DataFrame:
    rows = []
    min_precision = float(profile["min_precision"])
    min_n = int(profile["min_n"])
    min_last_gain = float(profile["min_last_gain"])
    full_guard = float(profile["full_guard"])
    max_test_abs = float(profile["max_test_abs"])
    max_test_dir_rate = float(profile["max_test_dir_rate"])
    risk_penalty = float(profile["risk_penalty"])
    move_penalty = float(profile["move_penalty"])
    full_penalty = float(profile["full_penalty"])

    for target in TARGETS:
        tg = table[table["target"].eq(target)].copy()
        eligible = tg[
            (tg["last_delta_vs_anchor"] <= -min_last_gain)
            & (tg["full_delta_vs_anchor"] <= full_guard)
            & (tg["test_abs_delta_mean"] <= max_test_abs)
            & (tg[["test_up_rate", "test_down_rate"]].max(axis=1) <= max_test_dir_rate)
        ].copy()
        if not eligible.empty:
            eligible = eligible[eligible.apply(lambda row: pass_precision(row, min_precision, min_n), axis=1)].copy()
        if eligible.empty:
            rows.append({
                "target": target,
                "action_key": "anchor",
                "source": "anchor",
                "mode": "anchor",
                "selected": True,
                "selector_score": 0.0,
            })
            continue
        eligible["selector_score"] = (
            eligible["last_delta_vs_anchor"]
            + full_penalty * np.maximum(0.0, eligible["full_delta_vs_anchor"])
            + risk_penalty * eligible["risk_proxy"]
            + move_penalty * eligible["test_abs_delta_mean"]
        )
        best = eligible.sort_values(["selector_score", "last_delta_vs_anchor", "full_delta_vs_anchor"]).iloc[0].to_dict()
        best["selected"] = True
        rows.append(best)
    return pd.DataFrame(rows)


def assemble(
    choices: pd.DataFrame,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = anchor_oof.copy()
    test = anchor_test.copy()
    for _, row in choices.iterrows():
        target = row["target"]
        key = row["action_key"]
        if key == "anchor":
            continue
        train_values, test_values = arrays[key]
        oof[target] = train_values
        test[target] = test_values
    return oof, test


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
    p.add_argument("--source-dirs", nargs="*", default=DEFAULT_SOURCE_DIRS)
    p.add_argument("--output-dir", default="research/direction_gated_search_20260622")
    p.add_argument("--submission-dir", default="submissions/direction_gated_search_20260622")
    p.add_argument("--thresholds", nargs="*", type=float, default=[0.03, 0.05, 0.08])
    p.add_argument("--alphas", nargs="*", type=float, default=[0.5, 0.75, 1.0])
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

    log("Loading sources")
    anchor_oof, anchor_test = load_anchor()
    sources = discover_pairs(args.source_dirs)
    log(f"Loaded {len(sources)} source pairs")

    log("Building direction action table")
    actions, arrays = build_action_table(ytr, folds, anchor_oof, anchor_test, sources, args.thresholds, args.alphas)
    actions.to_csv(out_dir / "target_action_scores.csv", index=False)

    profiles = [
        {
            "name": "precision55_move03",
            "min_precision": 0.55,
            "min_n": 3,
            "min_last_gain": 0.0002,
            "full_guard": 0.006,
            "max_test_abs": 0.03,
            "max_test_dir_rate": 0.35,
            "risk_penalty": 1.5,
            "move_penalty": 0.10,
            "full_penalty": 0.50,
        },
        {
            "name": "precision55_move05",
            "min_precision": 0.55,
            "min_n": 3,
            "min_last_gain": 0.0002,
            "full_guard": 0.006,
            "max_test_abs": 0.05,
            "max_test_dir_rate": 0.35,
            "risk_penalty": 1.0,
            "move_penalty": 0.05,
            "full_penalty": 0.35,
        },
        {
            "name": "precision60_move03",
            "min_precision": 0.60,
            "min_n": 3,
            "min_last_gain": 0.0002,
            "full_guard": 0.006,
            "max_test_abs": 0.03,
            "max_test_dir_rate": 0.35,
            "risk_penalty": 2.0,
            "move_penalty": 0.15,
            "full_penalty": 0.75,
        },
        {
            "name": "precision55_lowrisk",
            "min_precision": 0.55,
            "min_n": 3,
            "min_last_gain": 0.0002,
            "full_guard": 0.004,
            "max_test_abs": 0.025,
            "max_test_dir_rate": 0.25,
            "risk_penalty": 3.0,
            "move_penalty": 0.25,
            "full_penalty": 1.0,
        },
    ]

    score_rows = []
    all_choices = []
    for profile in profiles:
        choices = select_actions(actions, profile)
        oof, test = assemble(choices, arrays, anchor_oof, anchor_test)
        fold_losses = [mean_loss(ytr, oof, folds == f) for f in sorted(np.unique(folds))]
        row = {
            "candidate": profile["name"],
            "full_logloss": mean_loss(ytr, oof, full_mask),
            "last_logloss": mean_loss(ytr, oof, last_mask),
            "fold_std": float(np.std(fold_losses)),
            "tail3_worst": float(max(fold_losses[-3:])),
            "test_abs_delta_mean_vs_anchor": float(np.mean(np.abs(test[TARGETS].values - anchor_test[TARGETS].values))),
            **{f"fold{i}_logloss": v for i, v in enumerate(fold_losses)},
        }
        score_rows.append(row)
        safe = profile["name"].replace(".", "p")
        choices.insert(0, "candidate", profile["name"])
        all_choices.append(choices)
        write_prediction(out_dir / f"{safe}_oof.csv", mtr, oof, ytr)
        write_prediction(out_dir / f"{safe}_test_pred.csv", mte, test)
        write_submission(
            sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv",
            mte,
            test,
        )

    scores = pd.DataFrame(score_rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.concat(all_choices, ignore_index=True, sort=False).to_csv(out_dir / "selected_actions.csv", index=False)
    (out_dir / "report.json").write_text(
        json.dumps({
            "source_dirs": args.source_dirs,
            "thresholds": args.thresholds,
            "alphas": args.alphas,
            "candidate_scores": scores.to_dict(orient="records"),
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== Direction-gated candidate scores ===")
    print(scores[[
        "candidate",
        "full_logloss",
        "last_logloss",
        "fold_std",
        "tail3_worst",
        "test_abs_delta_mean_vs_anchor",
    ]].to_string(index=False))
    print("\n=== Selected actions ===")
    cols = [
        "candidate",
        "target",
        "source",
        "mode",
        "alpha",
        "threshold",
        "last_delta_vs_anchor",
        "full_delta_vs_anchor",
        "test_abs_delta_mean",
        "up_precision_actual_1",
        "down_precision_actual_0",
        "risk_proxy",
    ]
    choices = pd.concat(all_choices, ignore_index=True, sort=False)
    print(choices[cols].to_string(index=False))


if __name__ == "__main__":
    main()
