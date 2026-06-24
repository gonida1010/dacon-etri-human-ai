"""Constrained target-wise blend search for residual sources.

This is a lightweight post-processing search.  It does not train base models.

The previous `ridge_knn_blend_full` result found a strong full-period candidate,
but target-level diagnostics showed specific last-block losses on Q2/S1/S3.  This
script keeps the same source family and creates safer target-wise portfolios:

- hard last-delta guards per target
- full-delta caps per target
- full/last trade-off selectors

The goal is to generate a small frontier of submit/research candidates rather
than one overfit target-wise pick.
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
from .kaggle_last_mile import build_knn_source, targetwise_guarded_blend
from .residual_submission_blend import (
    candidate_stability,
    fold_losses,
    load_anchor_bank,
    load_pred,
    mean_loss,
    rank_score,
    safe_loss,
    target_candidates,
    write_prediction_frame,
    write_submission,
)

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def token_float(x: float) -> str:
    sign = "m" if x < 0 else ""
    return sign + str(abs(float(x))).replace(".", "p").rstrip("0").rstrip("p")


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "p").replace("-", "m")


def load_sources(
    ridge_dir: Path,
    bank_dir: Path,
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    full_guard: float,
    min_last_gain: float,
    weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    anchor_oof, anchor_test = load_anchor_bank(bank_dir, len(ytr), len(mte))
    sources_oof = {
        "anchor": anchor_oof,
        "ridge_residual_composite": load_pred(ridge_dir / "ridge_residual_composite_oof.csv"),
        "ridge_residual_full": load_pred(ridge_dir / "ridge_residual_full_oof.csv"),
        "ridge_residual_last": load_pred(ridge_dir / "ridge_residual_last_oof.csv"),
    }
    sources_test = {
        "anchor": anchor_test,
        "ridge_residual_composite": load_pred(ridge_dir / "ridge_residual_composite_test_pred.csv"),
        "ridge_residual_full": load_pred(ridge_dir / "ridge_residual_full_test_pred.csv"),
        "ridge_residual_last": load_pred(ridge_dir / "ridge_residual_last_test_pred.csv"),
    }

    knn_sources = {}
    for k in [3, 5, 8]:
        for scale in [7.0, 14.0, 30.0]:
            name = f"knn_k{k}_s{str(scale).replace('.', 'p')}"
            knn_sources[name] = build_knn_source(ytr, mtr, mte, folds, k=k, scale_days=scale, smooth=4.0)
    knn_oof, knn_test, knn_choices = targetwise_guarded_blend(
        ytr,
        folds,
        anchor_oof,
        anchor_test,
        knn_sources,
        full_guard,
        min_last_gain,
        weights,
        "knn_targetwise_guarded",
    )
    sources_oof["knn_targetwise_guarded"] = knn_oof
    sources_test["knn_targetwise_guarded"] = knn_test
    return anchor_oof, anchor_test, sources_oof, sources_test, knn_choices


def build_target_search(
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    sources_oof: dict[str, pd.DataFrame],
    sources_test: dict[str, pd.DataFrame],
    weights: list[float],
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    rows = []
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for target in TARGETS:
        y = ytr[target].values.astype(int)
        anchor_full = safe_loss(y, anchor_oof[target].values)
        anchor_folds = fold_losses(y, anchor_oof[target].values, folds)
        anchor_last = anchor_folds[-1]
        anchor_tail_worst = float(np.max(anchor_folds[-3:]))
        for source, po, pt in target_candidates(target, sources_oof, sources_test, weights):
            full_loss = safe_loss(y, po)
            losses = fold_losses(y, po, folds)
            last_loss = losses[-1]
            row = {
                "target": target,
                "source": source,
                "full_logloss": full_loss,
                "last_logloss": last_loss,
                "full_delta_vs_anchor": full_loss - anchor_full,
                "last_delta_vs_anchor": last_loss - anchor_last,
                "tail3_worst": float(np.max(losses[-3:])),
                "tail3_worst_delta_vs_anchor": float(np.max(losses[-3:]) - anchor_tail_worst),
                "fold_std": float(np.std(losses)),
                "rank_score": rank_score(full_loss, last_loss, anchor_full, losses),
            }
            for i, loss in enumerate(losses):
                row[f"fold{i}_logloss"] = loss
            rows.append(row)
            arrays[(target, source)] = (po, pt)
    search = pd.DataFrame(rows)
    return search.sort_values(["target", "full_logloss", "last_logloss"]), arrays


def choose_target_rows(
    search: pd.DataFrame,
    mode: str,
    last_guard: float,
    full_cap: float,
    alpha: float,
    tail_guard: float | None,
) -> pd.DataFrame:
    chosen = []
    for target in TARGETS:
        rows = search[search["target"].eq(target)].copy()
        if mode == "full":
            rows = rows.sort_values(["full_logloss", "last_logloss", "rank_score"])
        elif mode == "last":
            rows = rows.sort_values(["last_logloss", "full_logloss", "rank_score"])
        elif mode == "composite":
            rows = rows.sort_values(["rank_score", "last_logloss", "full_logloss"])
        elif mode == "last_guard":
            eligible = rows[
                rows["last_delta_vs_anchor"].le(last_guard)
                & rows["full_delta_vs_anchor"].le(full_cap)
            ]
            rows = (eligible if not eligible.empty else rows[rows["source"].eq("anchor")])
            rows = rows.sort_values(["full_logloss", "last_logloss", "rank_score"])
        elif mode == "positive_last_penalty":
            rows["selector_score"] = (
                rows["full_delta_vs_anchor"]
                + alpha * np.maximum(0.0, rows["last_delta_vs_anchor"] - last_guard)
            )
            if tail_guard is not None:
                rows["selector_score"] += 0.25 * np.maximum(
                    0.0,
                    rows["tail3_worst_delta_vs_anchor"] - tail_guard,
                )
            rows = rows.sort_values(["selector_score", "full_logloss", "last_logloss"])
        elif mode == "tradeoff_cap":
            eligible = rows[rows["full_delta_vs_anchor"].le(full_cap)].copy()
            rows = eligible if not eligible.empty else rows[rows["source"].eq("anchor")].copy()
            rows["selector_score"] = rows["full_delta_vs_anchor"] + alpha * rows["last_delta_vs_anchor"]
            if tail_guard is not None:
                rows["selector_score"] += 0.25 * np.maximum(
                    0.0,
                    rows["tail3_worst_delta_vs_anchor"] - tail_guard,
                )
            rows = rows.sort_values(["selector_score", "full_logloss", "last_logloss"])
        else:
            raise ValueError(mode)
        row = rows.iloc[0].to_dict()
        row["selection_mode"] = mode
        row["selector_last_guard"] = last_guard
        row["selector_full_cap"] = full_cap
        row["selector_alpha"] = alpha
        row["selector_tail_guard"] = tail_guard
        chosen.append(row)
    return pd.DataFrame(chosen)


def assemble_prediction(
    choices: pd.DataFrame,
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.DataFrame(index=anchor_oof.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    for _, row in choices.iterrows():
        po, pt = arrays[(row["target"], row["source"])]
        oof[row["target"]] = po
        test[row["target"]] = pt
    return oof, test


def candidate_specs(args: argparse.Namespace) -> list[tuple[str, str, float, float, float, float | None]]:
    specs: list[tuple[str, str, float, float, float, float | None]] = [
        ("full", "full", 0.0, args.full_cap, 0.0, None),
        ("composite", "composite", 0.0, args.full_cap, 0.0, None),
        ("last", "last", 0.0, args.full_cap, 0.0, None),
    ]
    for guard in args.last_guard_grid:
        specs.append((
            f"last_guard_{token_float(guard)}",
            "last_guard",
            float(guard),
            args.full_cap,
            0.0,
            args.tail_guard,
        ))
    for alpha in args.penalty_alpha_grid:
        specs.append((
            f"positive_last_penalty_a{token_float(alpha)}",
            "positive_last_penalty",
            args.penalty_last_guard,
            args.full_cap,
            float(alpha),
            args.tail_guard,
        ))
    for alpha in args.tradeoff_alpha_grid:
        specs.append((
            f"tradeoff_cap_a{token_float(alpha)}",
            "tradeoff_cap",
            args.penalty_last_guard,
            args.full_cap,
            float(alpha),
            args.tail_guard,
        ))
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-dir", default="research/oof_sparse_greedy")
    p.add_argument("--ridge-dir", default="research/residual_single_model_opt_ridge_logit_newton")
    p.add_argument("--output-dir", default="research/constrained_target_blend")
    p.add_argument("--submission-dir", default="submissions/constrained_target_blend")
    p.add_argument("--knn-full-guard", type=float, default=0.003)
    p.add_argument("--knn-min-last-gain", type=float, default=0.0001)
    p.add_argument("--weights", nargs="*", type=float, default=[0.2, 0.35, 0.5, 0.65, 0.8])
    p.add_argument("--full-cap", type=float, default=0.004)
    p.add_argument("--last-guard-grid", nargs="*", type=float, default=[0.0, 0.002, 0.004, 0.006, 0.008])
    p.add_argument("--penalty-last-guard", type=float, default=0.0)
    p.add_argument("--penalty-alpha-grid", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--tradeoff-alpha-grid", nargs="*", type=float, default=[0.15, 0.25, 0.35, 0.5])
    p.add_argument("--tail-guard", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset and residual/KNN sources")
    _, ytr, _, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    anchor_oof, anchor_test, sources_oof, sources_test, knn_choices = load_sources(
        ROOT / args.ridge_dir,
        ROOT / args.bank_dir,
        ytr,
        mtr,
        mte,
        folds,
        args.knn_full_guard,
        args.knn_min_last_gain,
        args.weights,
    )
    anchor_full = mean_loss(ytr, anchor_oof, full_mask)
    anchor_last = mean_loss(ytr, anchor_oof, last_mask)
    knn_choices.to_csv(out_dir / "knn_targetwise_choices.csv", index=False)

    log("Building target candidate table")
    search, arrays = build_target_search(ytr, folds, anchor_oof, sources_oof, sources_test, args.weights)
    search.to_csv(out_dir / "target_blend_search.csv", index=False)

    score_rows = []
    choices_all = []
    for name, mode, last_guard, full_cap, alpha, tail_guard in candidate_specs(args):
        choices = choose_target_rows(search, mode, last_guard, full_cap, alpha, tail_guard)
        po, pt = assemble_prediction(choices, arrays, anchor_oof, anchor_test)
        row = candidate_stability(ytr, po, folds, name)
        row["full_delta_vs_anchor"] = row["full_logloss"] - anchor_full
        row["last_delta_vs_anchor"] = row["last_logloss"] - anchor_last
        row["rank_score"] = rank_score(row["full_logloss"], row["last_logloss"], anchor_full, [
            row[f"fold{f}_logloss"] for f in sorted(np.unique(folds))
        ])
        row["selector_mode"] = mode
        row["selector_last_guard"] = last_guard
        row["selector_full_cap"] = full_cap
        row["selector_alpha"] = alpha
        score_rows.append(row)
        choices_all.append(choices.assign(candidate=name))

        safe = safe_name(name)
        choices.to_csv(out_dir / f"{safe}_target_choices.csv", index=False)
        write_prediction_frame(out_dir / f"{safe}_oof.csv", mtr, po, ytr)
        write_prediction_frame(out_dir / f"{safe}_test_pred.csv", mte, pt)
        write_submission(sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv", mte, pt)

    scores = pd.DataFrame(score_rows).sort_values(["rank_score", "full_logloss", "last_logloss"])
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.concat(choices_all, ignore_index=True).to_csv(out_dir / "target_choices_all.csv", index=False)
    report = {
        "anchor": {"full_logloss": anchor_full, "last_logloss": anchor_last},
        "sources": list(sources_oof),
        "args": vars(args),
        "candidate_scores": scores.to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Constrained target blend candidates ===")
    print(scores[[
        "candidate",
        "selector_mode",
        "selector_last_guard",
        "selector_alpha",
        "rank_score",
        "full_logloss",
        "last_logloss",
        "tail3_worst",
        "fold_std",
    ]].to_string(index=False))
    print("\n=== Best-full choices ===")
    best = scores.sort_values(["full_logloss", "last_logloss"]).iloc[0]["candidate"]
    best_choices = pd.read_csv(out_dir / f"{safe_name(best)}_target_choices.csv")
    print(best_choices[["target", "source", "full_logloss", "last_logloss", "full_delta_vs_anchor", "last_delta_vs_anchor"]].to_string(index=False))


if __name__ == "__main__":
    main()
