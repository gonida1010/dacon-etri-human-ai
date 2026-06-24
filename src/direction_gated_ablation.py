"""Ablation candidates for direction-gated post-processing.

This builds explicit recipes after inspecting direction precision.  The goal is
to separate the high-confidence Q1/S2/S4 moves from smaller Q2/Q3/S1 additions
instead of relying on one automatic selector.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .direction_gated_search import (
    DEFAULT_SOURCE_DIRS,
    assemble,
    build_action_table,
    discover_pairs,
    load_anchor,
    mean_loss,
    write_prediction,
    write_submission,
)

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[TARGETS].copy()


def recipe_choices(actions: pd.DataFrame, recipe: dict[str, str]) -> pd.DataFrame:
    rows = []
    by_key = actions.set_index("action_key", drop=False)
    for target in TARGETS:
        key = recipe.get(target, "anchor")
        if key == "anchor":
            rows.append({
                "target": target,
                "action_key": "anchor",
                "source": "anchor",
                "mode": "anchor",
                "selected": True,
            })
            continue
        if key not in by_key.index:
            raise KeyError(f"missing action key for {target}: {key}")
        row = by_key.loc[key].to_dict()
        row["selected"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def build_recipes() -> dict[str, dict[str, str]]:
    q1_down = "Q1::residual_single_model_opt_ridge_logit_newton/ridge_residual_last::down_thr::a1::t0.08"
    q1_orig_up = "Q1::constrained_target_blend_logit_newton/last::up_thr::a1::t0.08"
    q2_up = "Q2::residual_single_model_opt_ridge_logit_te_full/ridge_residual_last::up_thr::a1::t0.03"
    q2_tiny = "Q2::constrained_target_blend_logit_newton/last::up_thr::a1::t0.05"
    q3_up = "Q3::residual_single_model_opt_ridge_logit_te_full/ridge_residual_last::up_thr::a0.75::t0.08"
    s1_down = "S1::residual_single_model_opt_ridge/ridge_residual_full::down_thr::a1::t0.08"
    s2_tight = "S2::residual_single_model_opt_ridge/ridge_residual_composite::down_thr::a0.75::t0.08"
    s2_mid = "S2::residual_single_model_opt_ridge/ridge_residual_composite::down_thr::a1::t0.08"
    s2_wide = "S2::residual_single_model_opt_ridge/ridge_residual_composite::down_thr::a1::t0.05"
    s4_tight = "S4::residual_single_model_opt_ridge_logit_newton/ridge_residual_last::up_thr::a1::t0.05"
    s4_mid = "S4::residual_single_model_opt_ridge_logit_newton/ridge_residual_last::up_thr::a1::t0.03"

    return {
        "core_q1down_s2tight_s4tight": {"Q1": q1_down, "S2": s2_tight, "S4": s4_tight},
        "core_q1down_s2mid_s4tight": {"Q1": q1_down, "S2": s2_mid, "S4": s4_tight},
        "core_q1down_s2wide_s4tight": {"Q1": q1_down, "S2": s2_wide, "S4": s4_tight},
        "core_q1up_s2tight_s4tight": {"Q1": q1_orig_up, "S2": s2_tight, "S4": s4_tight},
        "core_q1down_s2tight_s4mid": {"Q1": q1_down, "S2": s2_tight, "S4": s4_mid},
        "core_plus_s1": {"Q1": q1_down, "S1": s1_down, "S2": s2_tight, "S4": s4_tight},
        "core_plus_q2": {"Q1": q1_down, "Q2": q2_up, "S2": s2_tight, "S4": s4_tight},
        "core_plus_q2tiny": {"Q1": q1_down, "Q2": q2_tiny, "S2": s2_tight, "S4": s4_tight},
        "core_plus_q3": {"Q1": q1_down, "Q3": q3_up, "S2": s2_tight, "S4": s4_tight},
        "core_plus_q2q3": {"Q1": q1_down, "Q2": q2_up, "Q3": q3_up, "S2": s2_tight, "S4": s4_tight},
        "core_plus_s1_q2q3": {
            "Q1": q1_down,
            "Q2": q2_up,
            "Q3": q3_up,
            "S1": s1_down,
            "S2": s2_tight,
            "S4": s4_tight,
        },
        "core_plus_s1_q2tiny_q3": {
            "Q1": q1_down,
            "Q2": q2_tiny,
            "Q3": q3_up,
            "S1": s1_down,
            "S2": s2_tight,
            "S4": s4_tight,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dirs", nargs="*", default=DEFAULT_SOURCE_DIRS)
    p.add_argument("--output-dir", default="research/direction_gated_ablation_20260622")
    p.add_argument("--submission-dir", default="submissions/direction_gated_ablation_20260622")
    p.add_argument("--submitted", default="submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv")
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

    log("Loading sources/actions")
    anchor_oof, anchor_test = load_anchor()
    sources = discover_pairs(args.source_dirs)
    actions, arrays = build_action_table(ytr, folds, anchor_oof, anchor_test, sources, args.thresholds, args.alphas)
    actions.to_csv(out_dir / "target_action_scores.csv", index=False)

    submitted_path = ROOT / args.submitted
    submitted = read_submission(submitted_path) if submitted_path.exists() else None

    rows = []
    selected = []
    shifts = []
    for name, recipe in build_recipes().items():
        choices = recipe_choices(actions, recipe)
        oof, test = assemble(choices, arrays, anchor_oof, anchor_test)
        fold_losses = [mean_loss(ytr, oof, folds == f) for f in sorted(np.unique(folds))]
        test_values = test[TARGETS].values
        anchor_values = anchor_test[TARGETS].values
        row = {
            "candidate": name,
            "full_logloss": mean_loss(ytr, oof, full_mask),
            "last_logloss": mean_loss(ytr, oof, last_mask),
            "fold_std": float(np.std(fold_losses)),
            "tail3_worst": float(max(fold_losses[-3:])),
            "test_abs_delta_mean_vs_anchor": float(np.mean(np.abs(test_values - anchor_values))),
            **{f"fold{i}_logloss": v for i, v in enumerate(fold_losses)},
        }
        if submitted is not None:
            submitted_values = submitted[TARGETS].values
            row["mean_abs_diff_to_submitted"] = float(np.mean(np.abs(test_values - submitted_values)))
            row["flat_corr_to_submitted"] = float(np.corrcoef(test_values.ravel(), submitted_values.ravel())[0, 1])
        rows.append(row)

        choices.insert(0, "candidate", name)
        selected.append(choices)
        for target in TARGETS:
            delta = test[target].values - anchor_test[target].values
            shifts.append({
                "candidate": name,
                "target": target,
                "test_mean_delta": float(np.mean(delta)),
                "test_abs_delta_mean": float(np.mean(np.abs(delta))),
                "test_up_rate": float(np.mean(delta > 1e-12)),
                "test_down_rate": float(np.mean(delta < -1e-12)),
            })

        safe = name.replace(".", "p")
        write_prediction(out_dir / f"{safe}_oof.csv", mtr, oof, ytr)
        write_prediction(out_dir / f"{safe}_test_pred.csv", mte, test)
        write_submission(
            sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv",
            mte,
            test,
        )

    scores = pd.DataFrame(rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    choices = pd.concat(selected, ignore_index=True, sort=False)
    shift_df = pd.DataFrame(shifts)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    choices.to_csv(out_dir / "selected_actions.csv", index=False)
    shift_df.to_csv(out_dir / "test_target_shift_summary.csv", index=False)
    (out_dir / "report.json").write_text(
        json.dumps({
            "submitted": str(submitted_path),
            "candidate_scores": scores.to_dict(orient="records"),
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== Direction-gated ablation scores ===")
    show_cols = [
        "candidate",
        "full_logloss",
        "last_logloss",
        "fold_std",
        "tail3_worst",
        "test_abs_delta_mean_vs_anchor",
    ]
    if "mean_abs_diff_to_submitted" in scores.columns:
        show_cols += ["mean_abs_diff_to_submitted", "flat_corr_to_submitted"]
    print(scores[show_cols].to_string(index=False))

    print("\n=== Test target shifts ===")
    print(shift_df.to_string(index=False))


if __name__ == "__main__":
    main()
