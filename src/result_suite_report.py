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
    "blend_logit_te_full": "research/residual_submission_blend_ridge_logit_te_full",
}


KEY_CANDIDATES = [
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
    return out


def save_candidate_frontier(scores: pd.DataFrame, key: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(scores["full_logloss"], scores["last_logloss"], s=28, alpha=0.35, color="#577590")
    ax.axvline(ANCHOR_FULL, color="#444444", lw=1, ls="--")
    ax.axhline(ANCHOR_LAST, color="#444444", lw=1, ls="--")
    ax.scatter(key["full_logloss"], key["last_logloss"], s=58, color="#d62828", zorder=3)
    for _, row in key.iterrows():
        ax.annotate(
            row["candidate"].replace("ridge_", "").replace("_", " "),
            (row["full_logloss"], row["last_logloss"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("OOF Candidate Frontier: full vs last logloss")
    ax.set_xlabel("full_logloss lower is better")
    ax.set_ylabel("last_logloss lower is better")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_frontier.png", dpi=180)
    plt.close(fig)


def save_delta_bars(key: pd.DataFrame, out_dir: Path) -> None:
    plot = key[key["candidate"].ne("anchor")].copy()
    plot = plot.sort_values("full_delta_vs_anchor")
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - 0.18, plot["full_delta_vs_anchor"], height=0.34, label="full delta", color="#277da1")
    ax.barh(y + 0.18, plot["last_delta_vs_anchor"], height=0.34, label="last delta", color="#f94144")
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=8)
    ax.set_title("Key Candidate Delta vs Anchor")
    ax.set_xlabel("logloss delta vs anchor lower is better")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "key_candidate_deltas.png", dpi=180)
    plt.close(fig)


def save_fold_curves(key: pd.DataFrame, out_dir: Path) -> None:
    fold_cols = [f"fold{i}_logloss" for i in range(5)]
    plot = key.dropna(subset=["run"]).copy()
    plot = plot[plot["candidate"].ne("anchor") & plot[fold_cols].notna().all(axis=1)]
    if plot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(5)
    for _, row in plot.iterrows():
        label = f"{row['run']} / {row['candidate']}"
        ax.plot(x, [row[c] for c in fold_cols], marker="o", lw=1.6, label=label)
    ax.set_title("Fold Logloss Curves for Key Candidates")
    ax.set_xlabel("subject-time fold")
    ax.set_ylabel("mean target logloss")
    ax.set_xticks(x)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / "key_fold_curves.png", dpi=180)
    plt.close(fig)


def save_target_heatmap(choices: pd.DataFrame, out_dir: Path) -> None:
    if choices.empty:
        return
    chosen = choices[
        choices["run"].isin(["ridge_logit_newton", "ridge_logit_te_full", "blend_ridge_knn", "blend_logit_te_full"])
        & choices["candidate"].isin(["ridge_residual_full", "ridge_residual_composite", "ridge_knn_blend_full", "ridge_knn_blend_composite"])
    ].copy()
    if chosen.empty:
        return
    chosen["row"] = chosen["run"] + " / " + chosen["candidate"]
    pivot = chosen.pivot_table(index="row", columns="target", values="last_delta_vs_anchor", aggfunc="first")
    pivot = pivot.reindex(columns=TARGETS)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(pivot))))
    vals = pivot.to_numpy(dtype=float)
    limit = max(0.001, float(np.nanmax(np.abs(vals))))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("Target-wise last delta vs anchor")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "target_last_delta_heatmap.png", dpi=180)
    plt.close(fig)


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
    lines = [
        "# Result Analysis Report, 2026-06-21",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Scope: completed OOF/result artifacts only. No model training is performed by this report.",
        "",
        "## Executive Findings",
        "",
        "- Best stable local candidate remains `ridge_logit_newton / ridge_residual_full`: full `0.587810`, last `0.590089`.",
        "- `bins_te` did not improve the Ridge residual family. It improves some last-block targets, but worsens full stability.",
        "- `blend_ridge_knn / ridge_knn_blend_full` remains the best blend-style stable candidate: full `0.592591`, last `0.590445`.",
        "- `blend_logit_te_full` is weaker than the old no-TE blend on both full and composite stability.",
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
        "- Old blend full candidate: full `0.592591`, last `0.590445`.",
        "- New TE blend full candidate: full `0.593723`, last `0.591800`.",
        "- New TE blend composite: full `0.596128`, last `0.588013`.",
        "- Interpretation: TE source is not adding useful diversity; it shifts target choices toward unstable last-block gains.",
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
        "- `key_candidate_deltas.png`: full/last delta against anchor for major candidates.",
        "- `key_fold_curves.png`: fold stability curves.",
        "- `target_last_delta_heatmap.png`: target-wise last-block gain/loss.",
        "",
        "## Next Research Step",
        "",
        "1. Freeze `bins_te` as an analysis-only branch for now; do not use it in the next submit candidate.",
        "2. Promote no-TE `ridge_logit_newton / ridge_residual_full` into blend search as the primary residual source.",
        "3. Add deduplicated source selection before stacking/blending: remove sources with identical predictions or corr >= 0.999 against anchor/another better source.",
        "4. Run a ridge-logit no-TE blend that uses `research/residual_single_model_opt_ridge_logit_newton` instead of the old prob residual dir.",
        "5. If that blend does not beat full `0.587810`, next implementation target is target-specific Ridge residual constraints, especially Q1 and S2, not more TE.",
        "",
    ]
    (out_dir / "RESULT_ANALYSIS_20260621.md").write_text("\n".join(lines), encoding="utf-8")


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
    save_delta_bars(key, out_dir)
    save_fold_curves(key, out_dir)
    save_target_heatmap(choices, out_dir)
    build_report(scores, key, choices, corr_summary, out_dir)

    print(f"Wrote report artifacts to {out_dir.relative_to(ROOT)}")
    print("\nTop full candidates:")
    print(scores[["run", "candidate", "full_logloss", "last_logloss"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
