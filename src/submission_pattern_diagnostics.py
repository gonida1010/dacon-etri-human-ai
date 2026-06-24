"""Diagnose OOF pattern transfer and submission CSV shifts.

This script is intentionally diagnostic only.  It compares generated OOF/test
prediction files against the temporal anchor and answers two questions:

- when a candidate moves probability up/down in OOF, was that movement right?
- does the submitted/test CSV apply the same movement too often by target or
  subject, creating public-LB false-positive/false-negative risk?
"""
from __future__ import annotations

import argparse
import math
import re
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
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
ID_COLS = C.ID_COLS


DEFAULT_RESEARCH_DIRS = [
    "research/constrained_target_blend_logit_newton",
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


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path} missing targets: {missing}")
    return pd.DataFrame({t: clip(df[t].values) for t in TARGETS})


def read_meta_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [*ID_COLS, *TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[[*ID_COLS, *TARGETS]].copy()
    for t in TARGETS:
        out[t] = clip(out[t].values)
    return out


def load_anchor() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_bank = pd.read_csv(ROOT / "research/oof_sparse_greedy/oof_bank.csv")
    test_bank = pd.read_csv(ROOT / "research/oof_sparse_greedy/test_bank.csv")
    anchor_oof = pd.DataFrame({t: clip(oof_bank[f"anchor__{t}"].values) for t in TARGETS})
    anchor_test = pd.DataFrame({t: clip(test_bank[f"anchor__{t}"].values) for t in TARGETS})
    return anchor_oof, anchor_test


def candidate_name(directory: Path, stem: str) -> str:
    return f"{directory.name}/{stem}"


def discover_oof_test_pairs(paths: list[str]) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for item in paths:
        p = rel(item)
        if not p.exists():
            continue
        files = sorted(p.glob("*_oof.csv")) if p.is_dir() else [p]
        for oof_path in files:
            if not oof_path.name.endswith("_oof.csv") or oof_path.name.startswith("anchor_"):
                continue
            stem = oof_path.name[:-8]
            test_path = oof_path.with_name(f"{stem}_test_pred.csv")
            if not test_path.exists():
                continue
            pairs[candidate_name(oof_path.parent, stem)] = (oof_path, test_path)
    return pairs


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def flat_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    av = a[TARGETS].to_numpy(dtype=float).ravel()
    bv = b[TARGETS].to_numpy(dtype=float).ravel()
    if np.std(av) == 0 or np.std(bv) == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def local_score_rows(
    y: pd.DataFrame,
    folds: np.ndarray,
    preds: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    full = np.ones(len(y), dtype=bool)
    last = folds == (C.N_SPLITS - 1)
    for name, (oof, _test) in preds.items():
        fold_losses = [mean_loss(y, oof, folds == f) for f in sorted(np.unique(folds))]
        rows.append({
            "candidate": name,
            "full_logloss": mean_loss(y, oof, full),
            "last_logloss": mean_loss(y, oof, last),
            "fold0_logloss": fold_losses[0],
            "fold1_logloss": fold_losses[1],
            "fold2_logloss": fold_losses[2],
            "fold3_logloss": fold_losses[3],
            "fold4_logloss": fold_losses[4],
            "fold_std": float(np.std(fold_losses)),
            "tail3_worst": float(max(fold_losses[-3:])),
        })
    return pd.DataFrame(rows).sort_values(["full_logloss", "last_logloss"]).reset_index(drop=True)


def split_pattern_rows(
    y: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    preds: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    split_masks = {
        "full": np.ones(len(y), dtype=bool),
        "last": folds == (C.N_SPLITS - 1),
    }
    for name, (oof, test) in preds.items():
        for target in TARGETS:
            test_delta = test[target].values - anchor_test[target].values
            test_up_rate = float(np.mean(test_delta >= threshold))
            test_down_rate = float(np.mean(test_delta <= -threshold))
            for split_name, mask in split_masks.items():
                yy = y[target].values[mask]
                pp = oof[target].values[mask]
                aa = anchor_oof[target].values[mask]
                delta = pp - aa
                up = delta >= threshold
                down = delta <= -threshold
                base_pos = float(np.mean(yy))
                up_precision = float(np.mean(yy[up])) if up.any() else np.nan
                down_precision = float(np.mean(1 - yy[down])) if down.any() else np.nan
                all_gain = float(np.mean(row_loss(yy, aa) - row_loss(yy, pp)))
                up_gain = float(np.mean(row_loss(yy[up], aa[up]) - row_loss(yy[up], pp[up]))) if up.any() else np.nan
                down_gain = float(np.mean(row_loss(yy[down], aa[down]) - row_loss(yy[down], pp[down]))) if down.any() else np.nan
                neutral = ~(up | down)
                neutral_gain = (
                    float(np.mean(row_loss(yy[neutral], aa[neutral]) - row_loss(yy[neutral], pp[neutral])))
                    if neutral.any()
                    else np.nan
                )
                up_bad = float(max(0.0, 0.5 - up_precision)) if np.isfinite(up_precision) else 0.0
                down_bad = float(max(0.0, 0.5 - down_precision)) if np.isfinite(down_precision) else 0.0
                rows.append({
                    "candidate": name,
                    "target": target,
                    "split": split_name,
                    "threshold": threshold,
                    "base_positive_rate": base_pos,
                    "oof_mean_delta": float(np.mean(delta)),
                    "oof_abs_delta_mean": float(np.mean(np.abs(delta))),
                    "oof_up_rate": float(np.mean(up)),
                    "oof_down_rate": float(np.mean(down)),
                    "oof_up_n": int(up.sum()),
                    "oof_down_n": int(down.sum()),
                    "up_precision_actual_1": up_precision,
                    "down_precision_actual_0": down_precision,
                    "up_lift_vs_base": up_precision - base_pos if np.isfinite(up_precision) else np.nan,
                    "down_lift_vs_base0": down_precision - (1.0 - base_pos) if np.isfinite(down_precision) else np.nan,
                    "false_positive_n": int(((yy == 0) & up).sum()),
                    "false_negative_n": int(((yy == 1) & down).sum()),
                    "logloss_gain_vs_anchor": all_gain,
                    "up_logloss_gain_vs_anchor": up_gain,
                    "down_logloss_gain_vs_anchor": down_gain,
                    "neutral_logloss_gain_vs_anchor": neutral_gain,
                    "test_mean_delta": float(np.mean(test_delta)),
                    "test_abs_delta_mean": float(np.mean(np.abs(test_delta))),
                    "test_up_rate": test_up_rate,
                    "test_down_rate": test_down_rate,
                    "test_up_n": int((test_delta >= threshold).sum()),
                    "test_down_n": int((test_delta <= -threshold).sum()),
                    "test_up_over_oof": test_up_rate - float(np.mean(up)),
                    "test_down_over_oof": test_down_rate - float(np.mean(down)),
                    "public_fp_risk_proxy": test_up_rate * up_bad,
                    "public_fn_risk_proxy": test_down_rate * down_bad,
                })
    return pd.DataFrame(rows)


def target_distribution_rows(
    meta_test: pd.DataFrame,
    anchor_test: pd.DataFrame,
    preds: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for name, (_oof, test) in preds.items():
        for target in TARGETS:
            p = test[target].values
            a = anchor_test[target].values
            d = p - a
            rows.append({
                "candidate": name,
                "target": target,
                "test_mean": float(np.mean(p)),
                "anchor_mean": float(np.mean(a)),
                "train_like_anchor_delta": float(np.mean(d)),
                "abs_delta_mean": float(np.mean(np.abs(d))),
                "delta_p05": float(np.quantile(d, 0.05)),
                "delta_p50": float(np.quantile(d, 0.50)),
                "delta_p95": float(np.quantile(d, 0.95)),
                "prob_p05": float(np.quantile(p, 0.05)),
                "prob_p50": float(np.quantile(p, 0.50)),
                "prob_p95": float(np.quantile(p, 0.95)),
                "prob_lt_0p2": float(np.mean(p < 0.2)),
                "prob_gt_0p8": float(np.mean(p > 0.8)),
            })
    return pd.DataFrame(rows)


def subject_shift_rows(
    meta_test: pd.DataFrame,
    anchor_test: pd.DataFrame,
    preds: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    subjects = meta_test["subject_id"].astype(str).values
    for name, (_oof, test) in preds.items():
        for target in TARGETS:
            for subject in sorted(np.unique(subjects)):
                mask = subjects == subject
                p = test.loc[mask, target].values
                a = anchor_test.loc[mask, target].values
                rows.append({
                    "candidate": name,
                    "target": target,
                    "subject_id": subject,
                    "n": int(mask.sum()),
                    "test_mean": float(np.mean(p)),
                    "anchor_mean": float(np.mean(a)),
                    "mean_delta": float(np.mean(p - a)),
                    "abs_delta_mean": float(np.mean(np.abs(p - a))),
                    "prob_lt_0p2": float(np.mean(p < 0.2)),
                    "prob_gt_0p8": float(np.mean(p > 0.8)),
                })
    return pd.DataFrame(rows)


def parse_local_from_name(path: Path) -> tuple[float, float]:
    text = path.name
    m_last = re.search(r"last([0-9]+(?:p|\.)[0-9]+)", text)
    m_full = re.search(r"full([0-9]+(?:p|\.)[0-9]+)", text)
    def conv(m: re.Match[str] | None) -> float:
        if not m:
            return float("nan")
        return float(m.group(1).replace("p", "."))
    return conv(m_last), conv(m_full)


def scan_submission_csvs(pattern: str, submitted: pd.DataFrame, anchor_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paths = sorted(ROOT.glob(pattern))
    for path in paths:
        try:
            sub = read_meta_pred(path)
        except Exception:
            continue
        last_local, full_local = parse_local_from_name(path)
        pred = sub[TARGETS]
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "local_last_from_name": last_local,
            "local_full_from_name": full_local,
            "flat_corr_to_submitted": flat_corr(pred, submitted[TARGETS]),
            "flat_corr_to_anchor": flat_corr(pred, anchor_test),
            "mean_abs_diff_to_submitted": float(np.mean(np.abs(pred.to_numpy() - submitted[TARGETS].to_numpy()))),
            "mean_abs_diff_to_anchor": float(np.mean(np.abs(pred.to_numpy() - anchor_test.to_numpy()))),
            "max_abs_diff_to_submitted": float(np.max(np.abs(pred.to_numpy() - submitted[TARGETS].to_numpy()))),
            "prob_lt_0p05": float(np.mean(pred.to_numpy() < 0.05)),
            "prob_gt_0p95": float(np.mean(pred.to_numpy() > 0.95)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mean_abs_diff_to_submitted", "path"]).reset_index(drop=True)
    return out


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    if df.empty:
        return "No rows."
    view = df[cols].copy()
    if n is not None:
        view = view.head(n)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    try:
        return view.to_markdown(index=False)
    except Exception:
        return "```\n" + view.to_string(index=False) + "\n```"


def plot_submitted_target_delta(dist: pd.DataFrame, submitted_name: str, out_dir: Path) -> None:
    plot = dist[dist["candidate"].eq(submitted_name)].copy()
    if plot.empty:
        return
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x, plot["train_like_anchor_delta"], color="#457b9d")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["target"])
    ax.set_title("Submitted Candidate Test Mean Delta vs Anchor")
    ax.set_ylabel("mean probability delta")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "submitted_target_mean_delta.png", dpi=180)
    plt.close(fig)


def plot_subject_delta_heatmap(subjects: pd.DataFrame, submitted_name: str, out_dir: Path) -> None:
    plot = subjects[subjects["candidate"].eq(submitted_name)].copy()
    if plot.empty:
        return
    pivot = plot.pivot(index="subject_id", columns="target", values="mean_delta").reindex(columns=TARGETS)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    vals = pivot.to_numpy(dtype=float)
    limit = max(0.02, float(np.nanpercentile(np.abs(vals), 95)))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Submitted Candidate Subject x Target Mean Delta vs Anchor")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "submitted_subject_target_delta_heatmap.png", dpi=180)
    plt.close(fig)


def plot_pattern_scatter(patterns: pd.DataFrame, out_dir: Path) -> None:
    plot = patterns[patterns["split"].eq("last")].copy()
    plot = plot[plot["oof_up_n"].ge(3) | plot["oof_down_n"].ge(3)]
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    risk = plot["public_fp_risk_proxy"] + plot["public_fn_risk_proxy"]
    sc = ax.scatter(
        plot["logloss_gain_vs_anchor"],
        plot["test_abs_delta_mean"],
        c=risk,
        cmap="magma_r",
        s=38,
        alpha=0.75,
    )
    ax.axvline(0, color="#333333", lw=1, ls="--")
    ax.set_title("OOF Last Gain vs Test Movement, by Candidate Target")
    ax.set_xlabel("OOF last logloss gain vs anchor, higher is better")
    ax.set_ylabel("test abs delta mean vs anchor")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="false-direction risk proxy")
    fig.tight_layout()
    fig.savefig(out_dir / "oof_gain_vs_test_movement.png", dpi=180)
    plt.close(fig)


def build_report(
    out_dir: Path,
    submitted_public: float | None,
    submitted_name: str,
    local_scores: pd.DataFrame,
    patterns: pd.DataFrame,
    dist: pd.DataFrame,
    subjects: pd.DataFrame,
    submissions: pd.DataFrame,
) -> None:
    submitted_patterns = patterns[
        patterns["candidate"].eq(submitted_name) & patterns["split"].eq("last")
    ].sort_values("target")
    strong = submitted_patterns.sort_values("logloss_gain_vs_anchor", ascending=False)
    false_up = submitted_patterns.sort_values(["public_fp_risk_proxy", "test_up_rate"], ascending=False)
    false_down = submitted_patterns.sort_values(["public_fn_risk_proxy", "test_down_rate"], ascending=False)
    submitted_dist = dist[dist["candidate"].eq(submitted_name)].sort_values("target")
    subject_worst = subjects[subjects["candidate"].eq(submitted_name)].copy()
    if not subject_worst.empty:
        subject_worst["abs_mean_delta"] = subject_worst["mean_delta"].abs()
        subject_worst = subject_worst.sort_values(["abs_mean_delta"], ascending=False)
    nearest = submissions.sort_values("mean_abs_diff_to_submitted").head(18) if not submissions.empty else submissions

    lines = [
        "# Submission Pattern Diagnostics",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Submitted candidate key: `{submitted_name}`",
    ]
    if submitted_public is not None and math.isfinite(submitted_public):
        lines.append(f"Observed public score supplied by user: `{submitted_public:.10f}`")
    lines.extend([
        "",
        "## Read",
        "",
        "- OOF gains are real only where the candidate's probability movement has high directional precision.",
        "- Public risk rises when low-precision OOF movements appear often in test CSV rows.",
        "- The submitted candidate is extremely close to the full logit blend; it mainly changes Q2/S1 by a small amount, so it did not test a meaningfully new public pattern.",
        "",
        "## Local OOF Scores, Paired Artifacts",
        "",
        md_table(local_scores, [
            "candidate",
            "full_logloss",
            "last_logloss",
            "fold_std",
            "tail3_worst",
        ], 20),
        "",
        "## Submitted Candidate Target Pattern, Last Fold",
        "",
        md_table(submitted_patterns, [
            "target",
            "logloss_gain_vs_anchor",
            "base_positive_rate",
            "oof_up_rate",
            "up_precision_actual_1",
            "oof_down_rate",
            "down_precision_actual_0",
            "test_mean_delta",
            "test_up_rate",
            "test_down_rate",
            "public_fp_risk_proxy",
            "public_fn_risk_proxy",
        ]),
        "",
        "## Stronger True-Looking Patterns In Submitted Candidate",
        "",
        md_table(strong, [
            "target",
            "logloss_gain_vs_anchor",
            "up_precision_actual_1",
            "down_precision_actual_0",
            "test_mean_delta",
            "test_abs_delta_mean",
        ], 7),
        "",
        "## False-Positive Risk, Upward Probability Pushes",
        "",
        md_table(false_up, [
            "target",
            "base_positive_rate",
            "oof_up_n",
            "up_precision_actual_1",
            "false_positive_n",
            "test_up_n",
            "test_up_rate",
            "public_fp_risk_proxy",
        ], 7),
        "",
        "## False-Negative Risk, Downward Probability Pushes",
        "",
        md_table(false_down, [
            "target",
            "base_positive_rate",
            "oof_down_n",
            "down_precision_actual_0",
            "false_negative_n",
            "test_down_n",
            "test_down_rate",
            "public_fn_risk_proxy",
        ], 7),
        "",
        "## Submitted Candidate Test Distribution",
        "",
        md_table(submitted_dist, [
            "target",
            "test_mean",
            "anchor_mean",
            "train_like_anchor_delta",
            "abs_delta_mean",
            "prob_p05",
            "prob_p50",
            "prob_p95",
        ]),
        "",
        "## Largest Subject-Level Test Shifts In Submitted Candidate",
        "",
        md_table(subject_worst, [
            "subject_id",
            "target",
            "n",
            "test_mean",
            "anchor_mean",
            "mean_delta",
            "abs_delta_mean",
        ], 24),
        "",
        "## CSVs Closest To Submitted File",
        "",
        md_table(nearest, [
            "path",
            "local_last_from_name",
            "local_full_from_name",
            "flat_corr_to_submitted",
            "mean_abs_diff_to_submitted",
            "mean_abs_diff_to_anchor",
        ], 18),
        "",
        "## Generated Figures",
        "",
        "- `submitted_target_mean_delta.png`",
        "- `submitted_subject_target_delta_heatmap.png`",
        "- `oof_gain_vs_test_movement.png`",
        "",
    ])
    text = "\n".join(lines)
    (out_dir / "PATTERN_DIAGNOSTICS_LATEST.md").write_text(text, encoding="utf-8")
    (out_dir / f"PATTERN_DIAGNOSTICS_{datetime.now().strftime('%Y%m%d')}.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/submission_pattern_diagnostics_20260622")
    p.add_argument("--research-dirs", nargs="*", default=DEFAULT_RESEARCH_DIRS)
    p.add_argument("--submission-glob", default="submissions/**/*.csv")
    p.add_argument(
        "--submitted",
        default="submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv",
    )
    p.add_argument("--submitted-public", type=float, default=float("nan"))
    p.add_argument("--delta-threshold", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = rel(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading data/folds")
    _, ytr, _, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    anchor_oof, anchor_test = load_anchor()

    log("Loading paired OOF/test artifacts")
    pairs = discover_oof_test_pairs(args.research_dirs)
    preds: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "anchor": (anchor_oof, anchor_test),
    }
    for name, (oof_path, test_path) in pairs.items():
        preds[name] = (read_pred(oof_path), read_pred(test_path))

    submitted_path = rel(args.submitted)
    submitted = read_meta_pred(submitted_path)
    submitted_key = f"{submitted_path.parent.name}/{submitted_path.name}"
    # Use the paired artifact name when available; it is cleaner in reports.
    exact_stem = submitted_path.name.replace(".csv", "")
    exact_paired_name = f"{submitted_path.parent.name}/{exact_stem}"
    stripped_stem = re.sub(r"_last[0-9.]+_full[0-9.]+\.csv$", "", submitted_path.name)
    stripped_stem = stripped_stem.replace(".csv", "")
    stripped_paired_name = f"{submitted_path.parent.name}/{stripped_stem}"
    if exact_paired_name in preds:
        submitted_name = exact_paired_name
    elif stripped_paired_name in preds:
        submitted_name = stripped_paired_name
    else:
        submitted_name = submitted_key
        preds[submitted_name] = (
            pd.DataFrame({t: anchor_oof[t].values for t in TARGETS}),
            submitted[TARGETS].reset_index(drop=True),
        )

    log(f"Loaded {len(preds)} paired candidates")
    local_scores = local_score_rows(ytr, folds, preds)
    local_scores.to_csv(out_dir / "paired_local_scores.csv", index=False)

    log("Computing OOF movement precision and test shift diagnostics")
    patterns = split_pattern_rows(
        ytr,
        folds,
        anchor_oof,
        anchor_test,
        preds,
        threshold=args.delta_threshold,
    )
    patterns.to_csv(out_dir / "oof_delta_pattern_diagnostics.csv", index=False)

    dist = target_distribution_rows(mte, anchor_test, preds)
    dist.to_csv(out_dir / "test_target_distribution.csv", index=False)

    subjects = subject_shift_rows(mte, anchor_test, preds)
    subjects.to_csv(out_dir / "test_subject_target_shift.csv", index=False)

    log("Scanning submission CSV files")
    submissions = scan_submission_csvs(args.submission_glob, submitted, anchor_test)
    submissions.to_csv(out_dir / "submission_csv_distance_scan.csv", index=False)

    plot_submitted_target_delta(dist, submitted_name, out_dir)
    plot_subject_delta_heatmap(subjects, submitted_name, out_dir)
    plot_pattern_scatter(patterns, out_dir)

    submitted_public = args.submitted_public if math.isfinite(args.submitted_public) else None
    build_report(
        out_dir,
        submitted_public,
        submitted_name,
        local_scores,
        patterns,
        dist,
        subjects,
        submissions,
    )

    print(f"Wrote diagnostics to {out_dir.relative_to(ROOT)}")
    print("\nSubmitted candidate pattern:")
    view = patterns[
        patterns["candidate"].eq(submitted_name) & patterns["split"].eq("last")
    ][[
        "target",
        "logloss_gain_vs_anchor",
        "up_precision_actual_1",
        "down_precision_actual_0",
        "test_mean_delta",
        "test_up_rate",
        "test_down_rate",
        "public_fp_risk_proxy",
        "public_fn_risk_proxy",
    ]]
    print(view.to_string(index=False))


if __name__ == "__main__":
    main()
