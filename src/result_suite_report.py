"""Build a research report from completed OOF/submission-result CSVs.

This script does not train models.  It only reads existing research artifacts,
creates comparison tables/plots, and writes a markdown report for deciding the
next experiment branch.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
ANCHOR_FULL = 0.595829751093375
ANCHOR_LAST = 0.5932822067983344


CANDIDATE_RUNS = {
    "anchor_bank_sparse": "research/oof_sparse_greedy",
    "ridge_prob": "research/residual_single_model_opt_ridge",
    "ridge_logit_newton": "research/residual_single_model_opt_ridge_logit_newton",
    "ridge_logit_te_smoke": "research/residual_single_model_opt_ridge_logit_te_smoke",
    "ridge_logit_te_full": "research/residual_single_model_opt_ridge_logit_te_full",
    "blend_ridge_knn": "research/residual_submission_blend",
    "blend_ridge_logit_newton": "research/residual_submission_blend_ridge_logit_newton",
    "constrained_logit_blend": "research/constrained_target_blend_logit_newton",
    "blend_logit_te_smoke": "research/residual_submission_blend_ridge_logit_te_smoke",
    "blend_logit_te_full": "research/residual_submission_blend_ridge_logit_te_full",
    "hist_gb_residual": "research/residual_single_model_opt_hist_gb",
    "extra_trees_residual": "research/residual_single_model_opt",
    "kaggle_last_mile": "research/kaggle_last_mile",
    "sequence_smoothing": "research/sequence_smoothing_search",
    "structural_balance": "research/structural_balance_search",
}


TARGET_CHOICE_RUNS = {
    "ridge_logit_newton": "research/residual_single_model_opt_ridge_logit_newton",
    "ridge_logit_te_full": "research/residual_single_model_opt_ridge_logit_te_full",
    "blend_ridge_knn": "research/residual_submission_blend",
    "blend_ridge_logit_newton": "research/residual_submission_blend_ridge_logit_newton",
    "constrained_logit_blend": "research/constrained_target_blend_logit_newton",
    "blend_logit_te_full": "research/residual_submission_blend_ridge_logit_te_full",
}


KEY_CANDIDATES = [
    ("constrained_logit_blend", "full"),
    ("constrained_logit_blend", "last_guard_0p008"),
    ("constrained_logit_blend", "last_guard_0p006"),
    ("constrained_logit_blend", "positive_last_penalty_a0p25"),
    ("constrained_logit_blend", "positive_last_penalty_a0p5"),
    ("constrained_logit_blend", "tradeoff_cap_a0p15"),
    ("constrained_logit_blend", "tradeoff_cap_a0p25"),
    ("constrained_logit_blend", "tradeoff_cap_a0p35"),
    ("constrained_logit_blend", "tradeoff_cap_a0p5"),
    ("ridge_logit_newton", "ridge_residual_full"),
    ("ridge_logit_newton", "ridge_residual_composite"),
    ("ridge_logit_newton", "ridge_residual_last"),
    ("blend_ridge_logit_newton", "ridge_knn_blend_full"),
    ("blend_ridge_logit_newton", "ridge_knn_blend_composite"),
    ("ridge_logit_te_full", "ridge_residual_full"),
    ("ridge_logit_te_full", "ridge_residual_composite"),
    ("blend_ridge_knn", "ridge_knn_blend_full"),
    ("blend_ridge_knn", "ridge_knn_blend_composite"),
    ("blend_logit_te_full", "ridge_knn_blend_full"),
    ("blend_logit_te_full", "ridge_knn_blend_composite"),
]

ALIASES = {
    ("anchor", "anchor"): "anchor",
    ("constrained_logit_blend", "full"): "constr full",
    ("constrained_logit_blend", "last_guard_0p008"): "guard 0.008",
    ("constrained_logit_blend", "last_guard_0p006"): "guard 0.006",
    ("constrained_logit_blend", "positive_last_penalty_a0p25"): "penalty .25",
    ("constrained_logit_blend", "positive_last_penalty_a0p5"): "penalty .50",
    ("constrained_logit_blend", "tradeoff_cap_a0p15"): "trade .15",
    ("constrained_logit_blend", "tradeoff_cap_a0p25"): "trade .25",
    ("constrained_logit_blend", "tradeoff_cap_a0p35"): "trade .35",
    ("constrained_logit_blend", "tradeoff_cap_a0p5"): "trade .50",
    ("ridge_logit_newton", "ridge_residual_full"): "logit ridge full",
    ("ridge_logit_newton", "ridge_residual_composite"): "logit ridge comp",
    ("ridge_logit_newton", "ridge_residual_last"): "logit ridge last",
    ("blend_ridge_logit_newton", "ridge_knn_blend_full"): "logit blend full",
    ("blend_ridge_logit_newton", "ridge_knn_blend_composite"): "logit blend comp",
    ("blend_ridge_logit_newton", "ridge_knn_blend_last"): "logit blend last",
    ("blend_ridge_knn", "ridge_knn_blend_full"): "old blend full",
    ("blend_ridge_knn", "ridge_knn_blend_composite"): "old blend comp",
    ("ridge_logit_te_full", "ridge_residual_full"): "TE ridge full",
    ("ridge_logit_te_full", "ridge_residual_composite"): "TE ridge comp",
    ("blend_logit_te_full", "ridge_knn_blend_full"): "TE blend full",
    ("blend_logit_te_full", "ridge_knn_blend_composite"): "TE blend comp",
}

LABEL_OFFSETS = {
    "anchor": (6, 6),
    "constr full": (8, 14),
    "guard 0.008": (10, -24),
    "guard 0.006": (8, 3),
    "penalty .25": (8, 6),
    "penalty .50": (18, -18),
    "trade .15": (8, -18),
    "trade .25": (-72, -12),
    "trade .35": (6, 6),
    "trade .50": (6, -12),
    "logit blend full": (5, -16),
    "logit ridge full": (5, 8),
    "old blend full": (5, 5),
    "TE blend full": (5, 8),
    "TE ridge full": (5, -14),
    "logit blend comp": (8, -16),
    "logit ridge comp": (8, 6),
    "logit ridge last": (8, 8),
    "old blend comp": (8, 8),
    "TE blend comp": (8, 8),
    "TE ridge comp": (8, -14),
}

CORE_FOLD_CANDIDATES = [
    ("anchor", "anchor"),
    ("constrained_logit_blend", "full"),
    ("constrained_logit_blend", "last_guard_0p008"),
    ("constrained_logit_blend", "positive_last_penalty_a0p25"),
    ("constrained_logit_blend", "tradeoff_cap_a0p15"),
    ("constrained_logit_blend", "tradeoff_cap_a0p25"),
    ("blend_ridge_logit_newton", "ridge_knn_blend_full"),
    ("ridge_logit_newton", "ridge_residual_full"),
    ("blend_ridge_knn", "ridge_knn_blend_full"),
    ("blend_ridge_logit_newton", "ridge_knn_blend_composite"),
    ("ridge_logit_newton", "ridge_residual_composite"),
]

SUBMISSION_FRONTIER_CANDIDATES = [
    ("constrained_logit_blend", "full"),
    ("constrained_logit_blend", "last_guard_0p008"),
    ("constrained_logit_blend", "last_guard_0p006"),
    ("constrained_logit_blend", "positive_last_penalty_a0p25"),
    ("constrained_logit_blend", "tradeoff_cap_a0p15"),
    ("constrained_logit_blend", "tradeoff_cap_a0p25"),
    ("ridge_logit_newton", "ridge_residual_full"),
]

CONSTRAINED_FRONTIER_CANDIDATES = [
    ("constrained_logit_blend", "full"),
    ("constrained_logit_blend", "last_guard_0p008"),
    ("constrained_logit_blend", "last_guard_0p006"),
    ("constrained_logit_blend", "positive_last_penalty_a0p25"),
    ("constrained_logit_blend", "positive_last_penalty_a0p5"),
    ("constrained_logit_blend", "tradeoff_cap_a0p15"),
    ("constrained_logit_blend", "tradeoff_cap_a0p25"),
    ("constrained_logit_blend", "tradeoff_cap_a0p35"),
    ("constrained_logit_blend", "tradeoff_cap_a0p5"),
]

CANDIDATE_FRONTIER_LABELS = {
    "anchor",
    "constr full",
    "guard 0.008",
    "penalty .25",
    "trade .15",
    "logit ridge full",
    "old blend full",
    "logit blend comp",
    "logit ridge comp",
}


def alias_for(run: str, candidate: str) -> str:
    return ALIASES.get((run, candidate), f"{run} / {candidate}")


def rel(path: str) -> Path:
    return ROOT / path


def load_candidate_scores() -> pd.DataFrame:
    frames = []
    for run, directory in CANDIDATE_RUNS.items():
        path = rel(directory) / "candidate_scores.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "candidate" not in df.columns:
            continue
        df.insert(0, "run", run)
        df.insert(1, "path", str(path.relative_to(ROOT)))
        for col in ["full_delta_vs_anchor", "last_delta_vs_anchor"]:
            if col not in df.columns:
                base = ANCHOR_FULL if col.startswith("full") else ANCHOR_LAST
                metric = "full_logloss" if col.startswith("full") else "last_logloss"
                df[col] = df[metric] - base
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No candidate_scores.csv files found")
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.sort_values(["full_logloss", "last_logloss", "run", "candidate"])


def load_anchor_target_scores(source_scores_path: Path) -> pd.DataFrame:
    if source_scores_path.exists():
        src = pd.read_csv(source_scores_path)
        anchor = src[src["source"].eq("anchor")][["target", "full_logloss", "last_logloss"]].copy()
        if len(anchor) == len(TARGETS):
            return anchor.rename(columns={
                "full_logloss": "anchor_full",
                "last_logloss": "anchor_last",
            })
    raise FileNotFoundError(f"Missing anchor target scores: {source_scores_path}")


def load_target_choices(anchor_target: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for run, directory in TARGET_CHOICE_RUNS.items():
        path = rel(directory) / "target_choices_all.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not {"target", "candidate", "full_logloss", "last_logloss"}.issubset(df.columns):
            continue
        df.insert(0, "run", run)
        df.insert(1, "path", str(path.relative_to(ROOT)))
        df = df.merge(anchor_target, on="target", how="left")
        df["full_delta_vs_anchor"] = df["full_logloss"] - df["anchor_full"]
        df["last_delta_vs_anchor"] = df["last_logloss"] - df["anchor_last"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def key_rows(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for run, cand in KEY_CANDIDATES:
        row = scores[scores["run"].eq(run) & scores["candidate"].eq(cand)]
        if not row.empty:
            parts.append(row.iloc[[0]])
    anchor_rows = scores[scores["candidate"].eq("anchor")]
    if not anchor_rows.empty:
        anchor = anchor_rows.iloc[[0]].copy()
        anchor["run"] = "anchor"
        anchor["full_delta_vs_anchor"] = 0.0
        anchor["last_delta_vs_anchor"] = 0.0
    else:
        anchor = pd.DataFrame([{
            "run": "anchor",
            "candidate": "anchor",
            "full_logloss": ANCHOR_FULL,
            "last_logloss": ANCHOR_LAST,
            "full_delta_vs_anchor": 0.0,
            "last_delta_vs_anchor": 0.0,
            "rank_score": np.nan,
            "fold_std": np.nan,
            "tail3_worst": np.nan,
        }])
    out = pd.concat([anchor] + parts, ignore_index=True, sort=False)
    out["label"] = out["run"] + " / " + out["candidate"]
    out["alias"] = [alias_for(r, c) for r, c in zip(out["run"], out["candidate"])]
    return out


def save_candidate_frontier(scores: pd.DataFrame, key: pd.DataFrame, out_dir: Path) -> None:
    stable = scores[scores["full_logloss"].le(0.62) & scores["last_logloss"].le(0.62)].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.scatter(stable["full_logloss"], stable["last_logloss"], s=30, alpha=0.25, color="#7d8597", label="stable-zone candidates")
    ax.axvline(ANCHOR_FULL, color="#444444", lw=1, ls="--")
    ax.axhline(ANCHOR_LAST, color="#444444", lw=1, ls="--")
    key_stable = key[key["full_logloss"].le(0.62) & key["last_logloss"].le(0.62)].copy()
    ax.scatter(key_stable["full_logloss"], key_stable["last_logloss"], s=78, color="#d62828", zorder=3, label="key candidates")
    for _, row in key_stable.iterrows():
        if row["alias"] not in CANDIDATE_FRONTIER_LABELS:
            continue
        offset = LABEL_OFFSETS.get(row["alias"], (5, 5))
        ax.annotate(
            row["alias"],
            (row["full_logloss"], row["last_logloss"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    xpad = 0.0015
    ypad = 0.0015
    ax.set_xlim(stable["full_logloss"].min() - xpad, min(0.605, stable["full_logloss"].max() + xpad))
    ax.set_ylim(stable["last_logloss"].min() - ypad, min(0.606, stable["last_logloss"].max() + ypad))
    ax.set_title("Stable OOF Frontier, full <= 0.62 and last <= 0.62")
    ax.set_xlabel("full_logloss, lower is better")
    ax.set_ylabel("last_logloss, lower is better")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_frontier.png", dpi=180)
    plt.close(fig)


def save_submission_frontier(key: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for run, cand in SUBMISSION_FRONTIER_CANDIDATES:
        row = key[key["run"].eq(run) & key["candidate"].eq(cand)]
        if not row.empty:
            rows.append(row.iloc[[0]])
    plot = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    colors = {
        "constr full": "#d62828",
        "guard 0.008": "#f77f00",
        "guard 0.006": "#fcbf49",
        "penalty .25": "#2a9d8f",
        "trade .15": "#457b9d",
        "trade .25": "#6a4c93",
        "logit blend full": "#8d99ae",
        "logit ridge full": "#606c38",
    }
    for _, row in plot.iterrows():
        alias = row["alias"]
        ax.scatter(row["full_logloss"], row["last_logloss"], s=90, color=colors.get(alias, "#777777"), zorder=3)
        offset = LABEL_OFFSETS.get(alias, (7, 7))
        ax.annotate(alias, (row["full_logloss"], row["last_logloss"]), xytext=offset, textcoords="offset points", fontsize=9)
    ax.set_xlim(plot["full_logloss"].min() - 0.0002, plot["full_logloss"].max() + 0.00025)
    ax.set_ylim(plot["last_logloss"].min() - 0.0004, plot["last_logloss"].max() + 0.00035)
    ax.set_title("Submission Candidate Frontier, constrained logit blend focus")
    ax.set_xlabel("full_logloss, lower is better")
    ax.set_ylabel("last_logloss, lower is better")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "submission_frontier.png", dpi=180)
    plt.close(fig)


def save_constrained_frontier(key: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for run, cand in CONSTRAINED_FRONTIER_CANDIDATES:
        row = key[key["run"].eq(run) & key["candidate"].eq(cand)]
        if not row.empty:
            rows.append(row.iloc[[0]])
    plot = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if plot.empty:
        return
    base = plot[plot["candidate"].eq("full")]
    if base.empty:
        return
    base = base.iloc[0]
    plot["full_cost_vs_full"] = plot["full_logloss"] - base["full_logloss"]
    plot["last_gain_vs_full"] = base["last_logloss"] - plot["last_logloss"]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    color_map = {
        "constr full": "#d62828",
        "guard 0.008": "#f77f00",
        "guard 0.006": "#fcbf49",
        "penalty .25": "#2a9d8f",
        "penalty .50": "#52b788",
        "trade .15": "#457b9d",
        "trade .25": "#6a4c93",
        "trade .35": "#9d4edd",
        "trade .50": "#7209b7",
    }
    for _, row in plot.iterrows():
        alias = row["alias"]
        ax.scatter(
            row["full_cost_vs_full"],
            row["last_gain_vs_full"],
            s=92,
            color=color_map.get(alias, "#777777"),
            zorder=3,
        )
        offset = LABEL_OFFSETS.get(alias, (7, 7))
        ax.annotate(alias, (row["full_cost_vs_full"], row["last_gain_vs_full"]), xytext=offset, textcoords="offset points", fontsize=9)
    ax.axhline(0.0, color="#555555", lw=1, ls="--")
    ax.axvline(0.0, color="#555555", lw=1, ls="--")
    ax.set_title("Constrained Blend Tradeoff vs constr full")
    ax.set_xlabel("full logloss cost vs constr full, lower is safer")
    ax.set_ylabel("last logloss gain vs constr full, higher is better")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "constrained_frontier.png", dpi=180)
    plt.close(fig)


def save_delta_bars(key: pd.DataFrame, out_dir: Path) -> None:
    plot = key[key["candidate"].ne("anchor")].copy()
    plot = plot.sort_values(["full_delta_vs_anchor", "last_delta_vs_anchor"])
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10.5, 6.7))
    ax.barh(y - 0.18, plot["full_delta_vs_anchor"], height=0.34, label="full delta", color="#277da1")
    ax.barh(y + 0.18, plot["last_delta_vs_anchor"], height=0.34, label="last delta", color="#f94144")
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.text(row["full_delta_vs_anchor"], i - 0.18, f"{row['full_delta_vs_anchor']:+.4f}", va="center", ha="right" if row["full_delta_vs_anchor"] < 0 else "left", fontsize=7)
        ax.text(row["last_delta_vs_anchor"], i + 0.18, f"{row['last_delta_vs_anchor']:+.4f}", va="center", ha="right" if row["last_delta_vs_anchor"] < 0 else "left", fontsize=7)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["alias"], fontsize=8)
    ax.set_title("Key Candidate Delta vs Anchor")
    ax.set_xlabel("logloss delta vs anchor lower is better")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "key_candidate_deltas.png", dpi=180)
    plt.close(fig)


def save_fold_curves(key: pd.DataFrame, out_dir: Path) -> None:
    fold_cols = [f"fold{i}_logloss" for i in range(5)]
    rows = []
    for run, cand in CORE_FOLD_CANDIDATES:
        row = key[key["run"].eq(run) & key["candidate"].eq(cand)]
        if not row.empty:
            rows.append(row.iloc[[0]])
    plot = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    plot = plot[plot[fold_cols].notna().all(axis=1)] if not plot.empty else plot
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(5)
    for _, row in plot.iterrows():
        ax.plot(x, [row[c] for c in fold_cols], marker="o", lw=1.8, label=row["alias"])
    ax.set_title("Fold Logloss Curves, Core Candidates Only")
    ax.set_xlabel("subject-time fold")
    ax.set_ylabel("mean target logloss")
    ax.set_xticks(x)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_dir / "key_fold_curves.png", dpi=180)
    plt.close(fig)


def save_one_target_heatmap(chosen: pd.DataFrame, metric: str, title: str, filename: str, out_dir: Path) -> None:
    chosen = chosen.copy()
    chosen["row"] = [alias_for(r, c) for r, c in zip(chosen["run"], chosen["candidate"])]
    pivot = chosen.pivot_table(index="row", columns="target", values=metric, aggfunc="first")
    desired_rows = [alias_for(r, c) for r, c in CORE_FOLD_CANDIDATES if alias_for(r, c) in set(pivot.index)]
    desired_rows.extend([r for r in pivot.index if r not in desired_rows])
    pivot = pivot.reindex(index=desired_rows, columns=TARGETS)
    fig, ax = plt.subplots(figsize=(9.5, max(3.8, 0.42 * len(pivot))))
    vals = pivot.to_numpy(dtype=float)
    limit = max(0.006, float(np.nanpercentile(np.abs(vals), 95)))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def save_target_heatmaps(choices: pd.DataFrame, out_dir: Path) -> None:
    if choices.empty:
        return
    legacy = choices[
        choices["run"].isin(["ridge_logit_newton", "blend_ridge_logit_newton", "ridge_logit_te_full", "blend_ridge_knn", "blend_logit_te_full"])
        & choices["candidate"].isin(["ridge_residual_full", "ridge_residual_composite", "ridge_knn_blend_full", "ridge_knn_blend_composite"])
    ].copy()
    constrained = choices[
        choices["run"].eq("constrained_logit_blend")
        & choices["candidate"].isin([
            "full",
            "last_guard_0p008",
            "positive_last_penalty_a0p25",
            "tradeoff_cap_a0p15",
            "tradeoff_cap_a0p25",
        ])
    ].copy()
    chosen = pd.concat([constrained, legacy], ignore_index=True, sort=False)
    if chosen.empty:
        return
    save_one_target_heatmap(
        chosen,
        "last_delta_vs_anchor",
        "Target-wise LAST delta vs anchor",
        "target_last_delta_heatmap.png",
        out_dir,
    )
    save_one_target_heatmap(
        chosen,
        "full_delta_vs_anchor",
        "Target-wise FULL delta vs anchor",
        "target_full_delta_heatmap.png",
        out_dir,
    )


def summarize_correlation(corr_dir: Path, out_dir: Path) -> pd.DataFrame:
    scores_path = corr_dir / "source_scores.csv"
    pairs_path = corr_dir / "source_correlations_long.csv"
    if not scores_path.exists() or not pairs_path.exists():
        return pd.DataFrame()
    scores = pd.read_csv(scores_path)
    pairs = pd.read_csv(pairs_path)
    rows = []
    for target, g in scores.groupby("target"):
        pg = pairs[pairs["target"].eq(target)]
        rows.append({
            "target": target,
            "source_count": int(g["source"].nunique()),
            "anchor_clone_count": int(g[g["full_logloss"].round(12).eq(g[g["source"].eq("anchor")]["full_logloss"].iloc[0].round(12))]["source"].nunique()) if not g[g["source"].eq("anchor")].empty else 0,
            "pairs_corr_ge_0p999": int((pg["abs_corr"] >= 0.999).sum()) if not pg.empty else 0,
            "pairs_corr_lt_0p98": int((pg["abs_corr"] < 0.98).sum()) if not pg.empty else 0,
            "best_full_source": g.sort_values("full_logloss").iloc[0]["source"],
            "best_full_logloss": float(g["full_logloss"].min()),
            "best_last_source": g.sort_values("last_logloss").iloc[0]["source"],
            "best_last_logloss": float(g["last_logloss"].min()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "source_correlation_summary.csv", index=False)
    return out


def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
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


def build_report(
    scores: pd.DataFrame,
    key: pd.DataFrame,
    choices: pd.DataFrame,
    corr_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    best_full = scores.sort_values(["full_logloss", "last_logloss"]).head(12)
    best_last = scores.sort_values(["last_logloss", "full_logloss"]).head(12)
    key_view = key.sort_values(["full_logloss", "last_logloss"])
    run_date = datetime.now().strftime("%Y-%m-%d")
    stamp = datetime.now().strftime("%Y%m%d")
    best = best_full.iloc[0]
    no_te_ridge = scores[
        scores["run"].eq("ridge_logit_newton") & scores["candidate"].eq("ridge_residual_full")
    ]
    old_blend = scores[
        scores["run"].eq("blend_ridge_knn") & scores["candidate"].eq("ridge_knn_blend_full")
    ]
    no_te_line = ""
    if not no_te_ridge.empty:
        r = no_te_ridge.iloc[0]
        no_te_line = (
            f"- No-TE Ridge full reference: full `{r['full_logloss']:.6f}`, "
            f"last `{r['last_logloss']:.6f}`."
        )
    old_blend_line = ""
    if not old_blend.empty:
        r = old_blend.iloc[0]
        old_blend_line = (
            f"- Old blend full reference: full `{r['full_logloss']:.6f}`, "
            f"last `{r['last_logloss']:.6f}`."
        )
    lines = [
        f"# Result Analysis Report, {run_date}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Scope: completed OOF/result artifacts only. No model training is performed by this report.",
        "",
        "## Executive Findings",
        "",
        f"- Best loaded local candidate is `{best['run']} / {best['candidate']}`: full `{best['full_logloss']:.6f}`, last `{best['last_logloss']:.6f}`.",
        no_te_line,
        old_blend_line,
        "- `bins_te` did not improve the Ridge residual family. It improves some last-block targets, but worsens full stability.",
        "- The no-TE logit blend remains the best full-logloss base; constrained target portfolios trade tiny full cost for better last-block OOF.",
        "- OOF source-correlation output shows many exact anchor clones from earlier source banks, so future blend search must deduplicate sources before greedy/stacking.",
        "",
        "## Key Candidate Comparison",
        "",
        md_table(key_view, [
            "run",
            "candidate",
            "full_logloss",
            "last_logloss",
            "full_delta_vs_anchor",
            "last_delta_vs_anchor",
            "rank_score",
            "fold_std",
        ]),
        "",
        "## Top by Full Logloss",
        "",
        md_table(best_full, ["run", "candidate", "full_logloss", "last_logloss", "full_delta_vs_anchor", "last_delta_vs_anchor"], 12),
        "",
        "## Top by Last Logloss",
        "",
        md_table(best_last, ["run", "candidate", "full_logloss", "last_logloss", "full_delta_vs_anchor", "last_delta_vs_anchor"], 12),
        "",
        "## TE Feature Bank Read",
        "",
        "- Smoke `bins_te`: `ridge_residual_full` full `0.594343`, last `0.591565`; worse than no-TE logit Ridge full candidate.",
        "- Full `bins_te`: `ridge_residual_full` full `0.593726`, last `0.591578`; still worse than no-TE logit Ridge full candidate.",
        "- Full `bins_te` composite: full `0.597277`, last `0.587759`; last improves but full is worse than anchor.",
        "- Interpretation: current fold-safe TE/bin bank is too noisy for this 450-row dataset. Keep the implementation, but do not use it globally in the next submit path.",
        "",
        "## Blend Read",
        "",
        "- No-TE logit blend full candidate: full `0.587732`, last `0.588717`.",
        "- Constrained `last_guard_0p008`: full `0.587743`, last `0.588383`; almost no full cost versus full base, with better last.",
        "- Constrained `positive_last_penalty_a0p25`: full `0.587837`, last `0.587745`; stronger last gain with still small full cost.",
        "- Constrained `tradeoff_cap_a0p15`: full `0.587898`, last `0.586630`; best balanced attack candidate before full cost starts rising.",
        "- Old blend full candidate: full `0.592591`, last `0.590445`.",
        "- New TE blend full candidate: full `0.593723`, last `0.591800`.",
        "- New TE blend composite: full `0.596128`, last `0.588013`.",
        "- Interpretation: no-TE logit residual is useful as a blend source; TE source is not adding useful diversity.",
        "",
        "## Correlation Summary",
        "",
        md_table(corr_summary, [
            "target",
            "source_count",
            "anchor_clone_count",
            "pairs_corr_ge_0p999",
            "pairs_corr_lt_0p98",
            "best_full_source",
            "best_full_logloss",
            "best_last_source",
            "best_last_logloss",
        ]) if not corr_summary.empty else "No correlation summary available.",
        "",
        "## Figures",
        "",
        "- `candidate_frontier.png`: full/last scatter over all loaded candidates.",
        "- `submission_frontier.png`: clean submit-candidate-only full/last scatter.",
        "- `constrained_frontier.png`: full-cost/last-gain tradeoff among constrained portfolios.",
        "- `key_candidate_deltas.png`: full/last delta against anchor for major candidates.",
        "- `key_fold_curves.png`: fold stability curves.",
        "- `target_last_delta_heatmap.png`: target-wise last-block gain/loss.",
        "- `target_full_delta_heatmap.png`: target-wise full-period gain/loss.",
        "",
        "## Next Research Step",
        "",
        "1. Primary submit-candidate path: `constrained_logit_blend / last_guard_0p008` when public-risk control matters, or `constrained_logit_blend / full` when pure full OOF is preferred.",
        "2. Attack candidate: `constrained_logit_blend / tradeoff_cap_a0p15`; it buys much better last OOF for a still small full cost.",
        "3. Keep `ridge_logit_newton / ridge_residual_full` as the no-blend fallback candidate.",
        "4. Freeze `bins_te` as an analysis-only branch for now; do not use it in the next submit candidate.",
        "5. Next modeling work should be source deduplication plus target-specific constraints for Q1/S2/S3, not more global TE.",
        "",
    ]
    text = "\n".join(lines)
    (out_dir / f"RESULT_ANALYSIS_{stamp}.md").write_text(text, encoding="utf-8")
    (out_dir / "RESULT_ANALYSIS_LATEST.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/result_analysis_20260621")
    p.add_argument("--correlation-dir", default="research/oof_source_correlation_logit_te_full")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = rel(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = load_candidate_scores()
    scores.to_csv(out_dir / "all_candidate_scores.csv", index=False)

    anchor_target = load_anchor_target_scores(rel(args.correlation_dir) / "source_scores.csv")
    anchor_target.to_csv(out_dir / "anchor_target_scores.csv", index=False)

    choices = load_target_choices(anchor_target)
    if not choices.empty:
        choices.to_csv(out_dir / "target_choice_summary.csv", index=False)

    key = key_rows(scores)
    key.to_csv(out_dir / "key_candidate_scores.csv", index=False)

    corr_summary = summarize_correlation(rel(args.correlation_dir), out_dir)

    save_candidate_frontier(scores, key, out_dir)
    save_submission_frontier(key, out_dir)
    save_constrained_frontier(key, out_dir)
    save_delta_bars(key, out_dir)
    save_fold_curves(key, out_dir)
    save_target_heatmaps(choices, out_dir)
    build_report(scores, key, choices, corr_summary, out_dir)

    print(f"Wrote report artifacts to {out_dir.relative_to(ROOT)}")
    print("\nTop full candidates:")
    print(scores[["run", "candidate", "full_logloss", "last_logloss"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
