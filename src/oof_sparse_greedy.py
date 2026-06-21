"""OOF prediction bank + guarded sparse greedy ensemble.

This is the next step after the structural/sequence searches:
- keep the temporal-prior anchor as the fallback for every target
- build a wide OOF/test prediction bank from conservative temporal sources
- add sequence-smoothed anchor variants as candidate sources
- select target-wise sparse blends only when last-block improves and full-CV
  stays inside the guard

Run:
  python -m src.oof_sparse_greedy
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
from .sequence_smoothing_search import apply_smoothing
from .train_temporal_prior import (
    build_recipe_predictions,
    clip,
    fit_lgbm_oof_test,
    temporal_oof,
    temporal_test,
)

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT
EPS = C.PROB_CLIP

TEMPORAL_METHODS = [
    "mean_sm4",
    "mean_sm8",
    "mean_sm16",
    "mean_sm32",
    "mean_sm64",
    "last2_sm2",
    "last2_sm4",
    "last3_sm4",
    "last5_sm4",
    "last10_sm4",
    "last20_sm4",
    "last30_sm4",
    "ridge0.5",
    "ridge1",
    "ridge3",
    "ridge10",
    "ridge30",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def temp_scale(p: np.ndarray, alpha: float) -> np.ndarray:
    return clip(sigmoid(alpha * logit(p)))


def source_name_float(prefix: str, *values: float) -> str:
    parts = [prefix]
    for value in values:
        parts.append(f"{value:g}".replace(".", "p").replace("-", "m"))
    return "_".join(parts)


def empty_source_like(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=TARGETS, dtype=float)


def add_source(
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
    name: str,
    oof: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_sources[name] = pd.DataFrame(clip(oof.values), index=oof.index, columns=TARGETS)
    test_sources[name] = pd.DataFrame(clip(test.values), index=test.index, columns=TARGETS)


def build_bank(
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    model_oof: pd.DataFrame,
    model_test: pd.DataFrame,
    emission_powers: list[float],
    transition_blends: list[float],
    transition_smooth: float,
    calibration_alphas: list[float],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    train_sources: dict[str, pd.DataFrame] = {}
    test_sources: dict[str, pd.DataFrame] = {}
    add_source(train_sources, test_sources, "anchor", base_oof, base_test)
    add_source(train_sources, test_sources, "model_lgbm_anchor", model_oof, model_test)

    for alpha in calibration_alphas:
        name = source_name_float("anchor_temp", alpha)
        oof = pd.DataFrame(
            {target: temp_scale(base_oof[target].values, alpha) for target in TARGETS},
            index=base_oof.index,
        )
        test = pd.DataFrame(
            {target: temp_scale(base_test[target].values, alpha) for target in TARGETS},
            index=base_test.index,
        )
        add_source(train_sources, test_sources, name, oof, test)

    temporal_cache_oof: dict[tuple[str, str], np.ndarray] = {}
    temporal_cache_test: dict[tuple[str, str], np.ndarray] = {}
    for method in TEMPORAL_METHODS:
        log(f"Building temporal source: {method}")
        oof = empty_source_like(base_oof.index)
        test = empty_source_like(base_test.index)
        for target in TARGETS:
            key = (target, method)
            temporal_cache_oof[key] = temporal_oof(ytr, mtr, folds, target, method)
            temporal_cache_test[key] = temporal_test(ytr, mtr, mte, target, method)
            oof[target] = temporal_cache_oof[key]
            test[target] = temporal_cache_test[key]
        add_source(train_sources, test_sources, method, oof, test)

    # Sequence sources are intentionally target-local. Non-target columns stay at
    # anchor values so target-wise selection can safely pick each source.
    sequence_details: list[pd.DataFrame] = []
    for target in TARGETS:
        log(f"Building sequence source grid for {target}")
        for ep in emission_powers:
            for tb in transition_blends:
                name = source_name_float(f"seq_{target}", ep, tb)
                sm_oof, sm_test, detail = apply_smoothing(
                    base_oof,
                    base_test,
                    ytr,
                    mtr,
                    mte,
                    folds,
                    {target: (float(ep), float(tb))},
                    transition_smooth,
                )
                detail["source"] = name
                sequence_details.append(detail)
                add_source(train_sources, test_sources, name, sm_oof, sm_test)

    detail_df = pd.concat(sequence_details, ignore_index=True) if sequence_details else pd.DataFrame()
    return train_sources, test_sources, detail_df


def source_scores(
    ytr: pd.DataFrame,
    train_sources: dict[str, pd.DataFrame],
    folds: np.ndarray,
) -> pd.DataFrame:
    last = folds == (C.N_SPLITS - 1)
    rows = []
    anchor = train_sources["anchor"]
    for source, pred in train_sources.items():
        for target in TARGETS:
            if not source_relevant_for_target(source, target):
                continue
            full = safe_loss(ytr[target].values, pred[target].values)
            last_score = safe_loss(ytr[target].values[last], pred[target].values[last])
            rows.append({
                "source": source,
                "target": target,
                "full_logloss": full,
                "last_logloss": last_score,
                "full_delta_vs_anchor": full - safe_loss(ytr[target].values, anchor[target].values),
                "last_delta_vs_anchor": last_score - safe_loss(ytr[target].values[last], anchor[target].values[last]),
            })
    return pd.DataFrame(rows).sort_values(["target", "last_logloss", "full_logloss"])


def write_wide_bank(
    path: Path,
    meta: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    y: pd.DataFrame | None = None,
) -> None:
    frames = [meta.reset_index(drop=True).copy()]
    if y is not None:
        labels = pd.DataFrame(index=meta.index)
        for target in TARGETS:
            labels[f"label__{target}"] = y[target].values
        frames.append(labels.reset_index(drop=True))
    pred_cols = {}
    for source, pred in sources.items():
        for target in TARGETS:
            pred_cols[f"{source}__{target}"] = clip(pred[target].values)
    frames.append(pd.DataFrame(pred_cols))
    out = pd.concat(frames, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def source_relevant_for_target(source: str, target: str) -> bool:
    if not source.startswith("seq_"):
        return True
    return source.startswith(f"seq_{target}_")


def candidate_pool_for_target(
    target: str,
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
) -> list[dict]:
    pool = []
    for source, pred in train_sources.items():
        if not source_relevant_for_target(source, target):
            continue
        pool.append({
            "source": source,
            "oof": pred[target].values,
            "test": test_sources[source][target].values,
        })
    return pool


def greedy_target(
    y: np.ndarray,
    last_mask: np.ndarray,
    anchor_oof: np.ndarray,
    anchor_test: np.ndarray,
    pool: list[dict],
    weights: list[float],
    full_guard: float,
    min_last_gain: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    current_oof = clip(anchor_oof)
    current_test = clip(anchor_test)
    anchor_full = safe_loss(y, current_oof)
    anchor_last = safe_loss(y[last_mask], current_oof[last_mask])
    current_full = anchor_full
    current_last = anchor_last
    steps: list[dict] = []
    used: set[str] = {"anchor"}

    for step in range(1, max_steps + 1):
        best = None
        for cand in pool:
            if cand["source"] in used:
                continue
            cand_oof = clip(cand["oof"])
            cand_test = clip(cand["test"])
            for w in weights:
                trial_oof = clip((1.0 - w) * current_oof + w * cand_oof)
                full = safe_loss(y, trial_oof)
                last = safe_loss(y[last_mask], trial_oof[last_mask])
                if full > anchor_full + full_guard:
                    continue
                if last > current_last - min_last_gain:
                    continue
                row = {
                    "step": step,
                    "source": cand["source"],
                    "weight_new_source": float(w),
                    "full_logloss": full,
                    "last_logloss": last,
                    "full_delta_vs_anchor": full - anchor_full,
                    "last_delta_vs_anchor": last - anchor_last,
                    "last_gain_vs_previous": current_last - last,
                    "oof": trial_oof,
                    "test": clip((1.0 - w) * current_test + w * cand_test),
                }
                if best is None or (row["last_logloss"], row["full_logloss"]) < (best["last_logloss"], best["full_logloss"]):
                    best = row
        if best is None:
            break
        current_oof = best.pop("oof")
        current_test = best.pop("test")
        current_full = best["full_logloss"]
        current_last = best["last_logloss"]
        used.add(best["source"])
        steps.append(best)

    if not steps:
        steps.append({
            "step": 0,
            "source": "anchor",
            "weight_new_source": 0.0,
            "full_logloss": current_full,
            "last_logloss": current_last,
            "full_delta_vs_anchor": 0.0,
            "last_delta_vs_anchor": 0.0,
            "last_gain_vs_previous": 0.0,
        })
    return current_oof, current_test, steps


def best_single_target(
    target: str,
    ytr: pd.DataFrame,
    last_mask: np.ndarray,
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
    full_guard: float,
    min_last_gain: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    anchor = train_sources["anchor"][target].values
    anchor_full = safe_loss(ytr[target].values, anchor)
    anchor_last = safe_loss(ytr[target].values[last_mask], anchor[last_mask])
    chosen = {
        "target": target,
        "source": "anchor",
        "full_logloss": anchor_full,
        "last_logloss": anchor_last,
        "full_delta_vs_anchor": 0.0,
        "last_delta_vs_anchor": 0.0,
        "reason": "fallback_anchor",
    }
    out_oof = clip(anchor)
    out_test = clip(test_sources["anchor"][target].values)
    for source, pred in train_sources.items():
        po = clip(pred[target].values)
        full = safe_loss(ytr[target].values, po)
        last = safe_loss(ytr[target].values[last_mask], po[last_mask])
        if full <= anchor_full + full_guard and last <= anchor_last - min_last_gain:
            if (last, full) < (chosen["last_logloss"], chosen["full_logloss"]):
                chosen = {
                    "target": target,
                    "source": source,
                    "full_logloss": full,
                    "last_logloss": last,
                    "full_delta_vs_anchor": full - anchor_full,
                    "last_delta_vs_anchor": last - anchor_last,
                    "reason": "accepted_single_source",
                }
                out_oof = po
                out_test = clip(test_sources[source][target].values)
    return out_oof, out_test, chosen


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    sub = meta_test.reset_index(drop=True).copy()
    for target in TARGETS:
        sub[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/oof_sparse_greedy")
    p.add_argument("--submission-dir", default="submissions/oof_sparse_greedy")
    p.add_argument("--full-guard", type=float, default=0.006)
    p.add_argument("--min-last-gain", type=float, default=0.0002)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--weight-grid", nargs="*", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    p.add_argument("--emission-powers", nargs="*", type=float, default=[0.5, 0.65, 0.8, 1.0, 1.25, 1.5])
    p.add_argument("--transition-blends", nargs="*", type=float, default=[0.15, 0.3, 0.45, 0.6, 0.8, 1.0])
    p.add_argument("--transition-smooth", type=float, default=2.0)
    p.add_argument("--calibration-alphas", nargs="*", type=float, default=[0.85, 0.92, 1.08, 1.15])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset")
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)

    log("Building temporal anchor OOF/test")
    model_oof, _, model_test = fit_lgbm_oof_test(Xtr, ytr, Xte, mtr, mte, folds)
    anchor_oof, anchor_test = build_recipe_predictions(ytr, mtr, mte, folds, model_oof, model_test)
    anchor_full = mean_loss(ytr, anchor_oof, full_mask)
    anchor_last = mean_loss(ytr, anchor_oof, last_mask)
    log(f"anchor full={anchor_full:.6f} last={anchor_last:.6f}")

    train_sources, test_sources, sequence_detail = build_bank(
        ytr,
        mtr,
        mte,
        folds,
        anchor_oof,
        anchor_test,
        model_oof,
        model_test,
        args.emission_powers,
        args.transition_blends,
        args.transition_smooth,
        args.calibration_alphas,
    )
    log(f"source bank built: {len(train_sources)} sources")

    write_wide_bank(out_dir / "oof_bank.csv", mtr, train_sources, ytr)
    write_wide_bank(out_dir / "test_bank.csv", mte, test_sources)
    if not sequence_detail.empty:
        sequence_detail.to_csv(out_dir / "sequence_source_detail.csv", index=False)
    scores = source_scores(ytr, train_sources, folds)
    scores.to_csv(out_dir / "source_scores.csv", index=False)

    single_oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    single_test = pd.DataFrame(index=mte.index, columns=TARGETS, dtype=float)
    single_rows = []
    greedy_oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    greedy_test = pd.DataFrame(index=mte.index, columns=TARGETS, dtype=float)
    greedy_rows = []

    for target in TARGETS:
        log(f"Selecting target {target}")
        soof, stest, srow = best_single_target(
            target,
            ytr,
            last_mask,
            train_sources,
            test_sources,
            args.full_guard,
            args.min_last_gain,
        )
        single_oof[target] = soof
        single_test[target] = stest
        single_rows.append(srow)

        goof, gtest, steps = greedy_target(
            ytr[target].values,
            last_mask,
            train_sources["anchor"][target].values,
            test_sources["anchor"][target].values,
            candidate_pool_for_target(target, train_sources, test_sources),
            args.weight_grid,
            args.full_guard,
            args.min_last_gain,
            args.max_steps,
        )
        greedy_oof[target] = goof
        greedy_test[target] = gtest
        for row in steps:
            greedy_rows.append({"target": target, **row})

    single_df = pd.DataFrame(single_rows)
    greedy_df = pd.DataFrame(greedy_rows)
    single_df.to_csv(out_dir / "targetwise_best_single.csv", index=False)
    greedy_df.to_csv(out_dir / "targetwise_greedy_steps.csv", index=False)

    candidate_rows = [
        {"candidate": "anchor", "full_logloss": anchor_full, "last_logloss": anchor_last, "notes": "temporal_prior_anchor"},
        {
            "candidate": "targetwise_best_single_guarded",
            "full_logloss": mean_loss(ytr, single_oof, full_mask),
            "last_logloss": mean_loss(ytr, single_oof, last_mask),
            "notes": "per-target best single source under full/last guards",
        },
        {
            "candidate": "targetwise_sparse_greedy",
            "full_logloss": mean_loss(ytr, greedy_oof, full_mask),
            "last_logloss": mean_loss(ytr, greedy_oof, last_mask),
            "notes": f"stagewise blend max_steps={args.max_steps}",
        },
    ]
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["last_logloss", "full_logloss"])
    candidate_df["full_delta_vs_anchor"] = candidate_df["full_logloss"] - anchor_full
    candidate_df["last_delta_vs_anchor"] = candidate_df["last_logloss"] - anchor_last
    candidate_df.to_csv(out_dir / "candidate_scores.csv", index=False)

    write_submission(sub_dir / f"00_anchor_last{anchor_last:.6f}_full{anchor_full:.6f}.csv", mte, anchor_test)
    for name, pred in [
        ("targetwise_best_single_guarded", single_test),
        ("targetwise_sparse_greedy", greedy_test),
    ]:
        row = candidate_df[candidate_df["candidate"].eq(name)].iloc[0]
        write_submission(sub_dir / f"{name}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv", mte, pred)

    report = {
        "purpose": "OOF/test prediction bank and guarded target-wise sparse greedy ensemble.",
        "full_guard": args.full_guard,
        "min_last_gain": args.min_last_gain,
        "max_steps": args.max_steps,
        "weight_grid": args.weight_grid,
        "emission_powers": args.emission_powers,
        "transition_blends": args.transition_blends,
        "transition_smooth": args.transition_smooth,
        "calibration_alphas": args.calibration_alphas,
        "source_count": len(train_sources),
        "anchor_full": anchor_full,
        "anchor_last": anchor_last,
        "candidate_scores": candidate_df.to_dict(orient="records"),
        "best_source_by_target": scores.groupby("target").head(5).to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Sparse greedy candidate scores ===")
    print(candidate_df.to_string(index=False))
    print("\n=== Target-wise best single choices ===")
    print(single_df.to_string(index=False))
    print("\n=== Target-wise greedy steps ===")
    print(greedy_df.to_string(index=False))
    if float(candidate_df.iloc[0]["last_logloss"]) <= 0.55:
        print("\nLOCAL <= 0.55 candidate found. Review before submission.")
    else:
        print("\nNo <= 0.55 local candidate yet. Do not submit from this run.")


if __name__ == "__main__":
    main()
