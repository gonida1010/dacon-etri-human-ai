"""Public-score-constrained pseudo posterior blend search.

The public leaderboard exposes only aggregate logloss, but each submitted
prediction still imposes one linear constraint on the hidden public labels:

    score = mean(-y log(p) - (1-y) log(1-p))

This script uses the known public scores as soft constraints to estimate a
test-time Bernoulli posterior around the temporal anchor.  It then searches
existing OOF/test sources for candidates that improve the pseudo-public
objective while keeping subject-time CV guards.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .direction_gated_search import DEFAULT_SOURCE_DIRS, discover_pairs, load_anchor
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
EPS = C.PROB_CLIP


@dataclass(frozen=True)
class KnownPublic:
    name: str
    path: Path
    score: float


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def soft_logloss(q: np.ndarray, p: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    p = clip(p)
    return float(np.mean(-(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))))


def soft_logloss_by_target(q: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    return {t: soft_logloss(q[t].values, pred[t].values) for t in TARGETS}


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def target_loss(y: pd.DataFrame, pred: pd.DataFrame, target: str, mask: np.ndarray) -> float:
    return safe_loss(y[target].values[mask], pred[target].values[mask])


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path} missing targets: {missing}")
    return pd.DataFrame({t: clip(df[t].values) for t in TARGETS})


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


def flatten_frame(df: pd.DataFrame) -> np.ndarray:
    return df[TARGETS].to_numpy(dtype=float).ravel()


def unflatten(values: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float).reshape(-1, len(TARGETS))
    return pd.DataFrame(arr, columns=TARGETS, index=index)


def parse_known_public(items: list[str]) -> list[KnownPublic]:
    out: list[KnownPublic] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"known public item must be PATH=SCORE, got {item}")
        path_s, score_s = item.rsplit("=", 1)
        path = rel(path_s)
        if not path.exists():
            log(f"Skipping missing public constraint: {path}")
            continue
        out.append(KnownPublic(name=path.stem, path=path, score=float(score_s)))
    return out


def constraint_terms(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = clip(flatten_frame(pred))
    a = np.log((1.0 - p) / p)
    c = -np.log(1.0 - p)
    return a, c


def estimate_pseudo_posterior(
    anchor_test: pd.DataFrame,
    known: list[KnownPublic],
    ridge: float,
    max_abs_lambda: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = clip(flatten_frame(anchor_test))
    prior_logit = logit(prior)
    a_rows = []
    c_means = []
    scores = []
    names = []
    for item in known:
        pred = read_pred(item.path)
        a, c = constraint_terms(pred)
        a_rows.append(a)
        c_means.append(float(np.mean(c)))
        scores.append(float(item.score))
        names.append(item.name)

    if not a_rows:
        q = prior.copy()
        diagnostics = pd.DataFrame()
        return unflatten(q, anchor_test.index), diagnostics

    a_mat = np.vstack(a_rows)
    c_means_arr = np.asarray(c_means, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    scale = np.maximum(1e-4, np.abs(scores_arr))

    def q_from_lam(lam: np.ndarray) -> np.ndarray:
        return clip(sigmoid(prior_logit - lam @ a_mat))

    def residual(lam: np.ndarray) -> np.ndarray:
        q = q_from_lam(lam)
        pred_scores = c_means_arr + (a_mat @ q) / q.size
        res = (pred_scores - scores_arr) / scale
        if ridge > 0:
            res = np.r_[res, np.sqrt(ridge) * lam]
        return res

    bounds = (-max_abs_lambda * np.ones(len(known)), max_abs_lambda * np.ones(len(known)))
    fit = least_squares(residual, np.zeros(len(known)), bounds=bounds, max_nfev=5000, xtol=1e-11, ftol=1e-11)
    lam = np.asarray(fit.x, dtype=float)
    q = q_from_lam(lam)

    rows = []
    for i, item in enumerate(known):
        matched_score = c_means_arr[i] + float(np.dot(a_mat[i], q)) / q.size
        anchor_score = c_means_arr[i] + float(np.dot(a_mat[i], prior)) / prior.size
        rows.append(
            {
                "constraint": item.name,
                "path": str(item.path.relative_to(ROOT) if item.path.is_relative_to(ROOT) else item.path),
                "public_score": item.score,
                "anchor_expected_score": anchor_score,
                "pseudo_expected_score": matched_score,
                "residual": matched_score - item.score,
                "lambda": lam[i],
                "ridge": ridge,
                "success": bool(fit.success),
                "cost": float(fit.cost),
            }
        )
    return unflatten(q, anchor_test.index), pd.DataFrame(rows)


def blend_pred(a: pd.DataFrame, b: pd.DataFrame, w: float, mode: str) -> pd.DataFrame:
    out = pd.DataFrame(index=a.index, columns=TARGETS, dtype=float)
    for target in TARGETS:
        if mode == "logit":
            z = w * logit(a[target].values) + (1.0 - w) * logit(b[target].values)
            out[target] = clip(sigmoid(z))
        else:
            out[target] = clip(w * a[target].values + (1.0 - w) * b[target].values)
    return out


def targetwise_pseudo_select(
    name: str,
    y: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    q_public: pd.DataFrame,
    pool: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    full_guard: float,
    last_guard: float,
    cv_weight: float,
    move_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    oof = anchor_oof.copy()
    test = anchor_test.copy()
    rows = []
    for target in TARGETS:
        base_full = target_loss(y, anchor_oof, target, full_mask)
        base_last = target_loss(y, anchor_oof, target, last_mask)
        base_pseudo = soft_logloss(q_public[target].values, anchor_test[target].values)
        best = {
            "target": target,
            "source": "anchor",
            "full_logloss": base_full,
            "last_logloss": base_last,
            "pseudo_public_logloss": base_pseudo,
            "selector_score": base_pseudo,
            "test_abs_delta_mean": 0.0,
        }
        for source, (poof, ptest) in pool.items():
            full = target_loss(y, poof, target, full_mask)
            last = target_loss(y, poof, target, last_mask)
            if full > base_full + full_guard:
                continue
            if last > base_last + last_guard:
                continue
            pseudo = soft_logloss(q_public[target].values, ptest[target].values)
            delta = ptest[target].values - anchor_test[target].values
            score = (
                pseudo
                + cv_weight * max(0.0, full - base_full)
                + 0.35 * cv_weight * max(0.0, last - base_last)
                + move_penalty * float(np.mean(np.abs(delta)))
            )
            if score < best["selector_score"]:
                best = {
                    "target": target,
                    "source": source,
                    "full_logloss": full,
                    "last_logloss": last,
                    "pseudo_public_logloss": pseudo,
                    "selector_score": score,
                    "test_abs_delta_mean": float(np.mean(np.abs(delta))),
                }
        if best["source"] != "anchor":
            oof[target] = pool[best["source"]][0][target].values
            test[target] = pool[best["source"]][1][target].values
        rows.append(best)
    return oof, test, pd.DataFrame(rows).assign(candidate=name)


def fit_simplex_to_pseudo(
    name: str,
    y: pd.DataFrame,
    folds: np.ndarray,
    q_public: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    pool: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    source_names: list[str],
    cv_weight: float,
    l2: float,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    oof = pd.DataFrame(index=y.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    rows = []
    for target in TARGETS:
        names = ["anchor"] + [n for n in source_names if n in pool and n != "anchor"]
        names = names[: min(len(names), 18)]
        x_test = np.column_stack([pool[n][1][target].values if n != "anchor" else anchor_test[target].values for n in names])
        x_oof = np.column_stack([pool[n][0][target].values if n != "anchor" else anchor_oof[target].values for n in names])
        x0 = np.zeros(len(names))
        x0[0] = 1.0
        base_full = target_loss(y, anchor_oof, target, full_mask)
        base_last = target_loss(y, anchor_oof, target, last_mask)

        def pred_from(w: np.ndarray, x: np.ndarray) -> np.ndarray:
            if mode == "logit":
                return clip(sigmoid(logit(x) @ w))
            return clip(x @ w)

        def obj(w: np.ndarray) -> float:
            p_test = pred_from(w, x_test)
            p_oof = pred_from(w, x_oof)
            full = safe_loss(y[target].values[full_mask], p_oof[full_mask])
            last = safe_loss(y[target].values[last_mask], p_oof[last_mask])
            cv_pen = cv_weight * max(0.0, full - base_full) + 0.35 * cv_weight * max(0.0, last - base_last)
            return soft_logloss(q_public[target].values, p_test) + cv_pen + l2 * float(np.sum((w - x0) ** 2))

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * len(names)
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
        w = np.asarray(res.x if res.success else x0, dtype=float)
        w = np.maximum(0.0, w)
        w = w / max(1e-12, float(w.sum()))
        oof[target] = pred_from(w, x_oof)
        test[target] = pred_from(w, x_test)
        rows.append(
            {
                "candidate": name,
                "target": target,
                "mode": mode,
                "success": bool(res.success),
                "sources": json.dumps({n: float(v) for n, v in zip(names, w) if v > 1e-4}),
                "full_logloss": target_loss(y, oof, target, full_mask),
                "last_logloss": target_loss(y, oof, target, last_mask),
                "pseudo_public_logloss": soft_logloss(q_public[target].values, test[target].values),
            }
        )
    return (
        pd.DataFrame({t: clip(oof[t].values) for t in TARGETS}),
        pd.DataFrame({t: clip(test[t].values) for t in TARGETS}),
        pd.DataFrame(rows),
    )


def candidate_score_row(
    name: str,
    y: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    q_public: pd.DataFrame,
    known: list[KnownPublic],
    oof: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    fold_losses = [mean_loss(y, oof, folds == f) for f in sorted(np.unique(folds))]
    full = mean_loss(y, oof, full_mask)
    last = mean_loss(y, oof, last_mask)
    anchor_full = mean_loss(y, anchor_oof, full_mask)
    anchor_last = mean_loss(y, anchor_oof, last_mask)
    pseudo_by_target = soft_logloss_by_target(q_public, test)
    delta = test[TARGETS].values - anchor_test[TARGETS].values
    row: dict[str, float | str] = {
        "candidate": name,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "fold_std": float(np.std(fold_losses)),
        "tail3_worst": float(max(fold_losses[-3:])),
        "pseudo_public_logloss": float(np.mean(list(pseudo_by_target.values()))),
        "test_abs_delta_mean_vs_anchor": float(np.mean(np.abs(delta))),
        "test_up_rate_vs_anchor": float(np.mean(delta > 1e-12)),
        "test_down_rate_vs_anchor": float(np.mean(delta < -1e-12)),
    }
    for i, val in enumerate(fold_losses):
        row[f"fold{i}_logloss"] = val
    for target, val in pseudo_by_target.items():
        row[f"pseudo_{target}"] = val
        row[f"delta_mean_{target}"] = float(np.mean(test[target].values - anchor_test[target].values))
        row[f"delta_abs_{target}"] = float(np.mean(np.abs(test[target].values - anchor_test[target].values)))
        row[f"up_rate_{target}"] = float(np.mean(test[target].values > anchor_test[target].values + 1e-12))
    for item in known:
        submitted = read_pred(item.path)
        row[f"corr_to_{item.name}"] = flat_corr(test, submitted)
        row[f"mean_abs_diff_to_{item.name}"] = float(np.mean(np.abs(test[TARGETS].values - submitted[TARGETS].values)))
    row["selector_score"] = (
        float(row["pseudo_public_logloss"])
        + 0.35 * max(0.0, full - anchor_full)
        + 0.18 * max(0.0, last - anchor_last)
        + 0.05 * float(row["fold_std"])
        + 0.03 * float(row["test_abs_delta_mean_vs_anchor"])
    )
    return row


def flat_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    av = flatten_frame(a)
    bv = flatten_frame(b)
    if np.std(av) == 0 or np.std(bv) == 0:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def build_report(
    out_dir: Path,
    scores: pd.DataFrame,
    constraints: pd.DataFrame,
    target_choices: pd.DataFrame,
    selected: pd.Series,
    known: list[KnownPublic],
) -> None:
    def to_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        work = df.copy()
        for col in work.columns:
            if pd.api.types.is_float_dtype(work[col]):
                work[col] = work[col].map(lambda x: f"{x:.6f}")
            else:
                work[col] = work[col].astype(str)
        header = "| " + " | ".join(work.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in work.to_numpy(dtype=str)]
        return "\n".join([header, sep, *rows])

    lines = [
        "# Public-score pseudo blend report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Known public constraints",
        "",
    ]
    if constraints.empty:
        lines.append("- none")
    else:
        for row in constraints.itertuples(index=False):
            lines.append(
                f"- `{row.constraint}` public={row.public_score:.10f}, "
                f"pseudo_expected={row.pseudo_expected_score:.10f}, residual={row.residual:+.6g}"
            )
    lines += [
        "",
        "## Selected candidate",
        "",
        f"- candidate: `{selected['candidate']}`",
        f"- pseudo_public_logloss: `{selected['pseudo_public_logloss']:.9f}`",
        f"- local full/last: `{selected['full_logloss']:.6f}` / `{selected['last_logloss']:.6f}`",
        f"- delta vs anchor full/last: `{selected['full_delta_vs_anchor']:+.6f}` / `{selected['last_delta_vs_anchor']:+.6f}`",
        "",
        "## Top candidates",
        "",
        to_markdown_table(
            scores.head(20)[
                ["candidate", "pseudo_public_logloss", "full_logloss", "last_logloss", "selector_score"]
            ]
        ),
        "",
    ]
    if not target_choices.empty:
        lines += [
            "## Target choices",
            "",
            to_markdown_table(target_choices.tail(7)),
            "",
        ]
    if known:
        lines += [
            "## Public feedback interpretation",
            "",
            "- The newer `target_select_public_tight` score is lower than the prior `last_guard_0p008`, so public feedback rewards the restricted movement pattern.",
            "- The search therefore optimizes a pseudo-public posterior first, then rejects candidates whose CV full/last degradation is too large.",
            "",
        ]
    (out_dir / "PUBLIC_PSEUDO_BLEND_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plot_scores(out_dir: Path, scores: pd.DataFrame) -> None:
    plot_df = scores.head(30).copy()
    plot_df["rank_label"] = [f"R{i}" for i in range(1, len(plot_df) + 1)]
    min_pseudo = float(plot_df["pseudo_public_logloss"].min())
    plot_df["pseudo_excess_x1e4"] = (plot_df["pseudo_public_logloss"] - min_pseudo) * 10000.0

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.scatter(plot_df["full_logloss"], plot_df["pseudo_excess_x1e4"], s=48, alpha=0.82)
    for _, row in plot_df.head(12).iterrows():
        ax.annotate(
            row["rank_label"],
            (row["full_logloss"], row["pseudo_excess_x1e4"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
        )
    ax.set_xlabel("OOF full logloss")
    ax.set_ylabel("Pseudo-public excess vs best (x1e-4)")
    ax.set_title("Candidate tradeoff: lower-left is better")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_tradeoff.png", dpi=160)
    fig.savefig(out_dir / "candidate_tradeoff.svg")
    plt.close(fig)

    legend_cols = ["rank_label", "candidate", "pseudo_public_logloss", "full_logloss", "last_logloss"]
    plot_df.head(12)[legend_cols].to_csv(out_dir / "candidate_tradeoff_legend.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-dirs",
        nargs="*",
        default=DEFAULT_SOURCE_DIRS
        + [
            "research/direction_gated_ablation_20260622",
            "research/direction_gated_search_20260622",
            "research/diverse_single_stack",
            "research/kaggle_last_mile",
            "research/public_aware_stack_blend_20260622",
            "research/residual_single_model_opt_ridge_logit_te_smoke",
            "research/residual_submission_blend_ridge_logit_te_smoke",
            "research/residual_single_model_opt_ridge_logit_newton",
            "research/residual_submission_blend_ridge_logit_newton",
        ],
    )
    p.add_argument(
        "--known-public",
        nargs="*",
        default=[
            "submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv=0.5920118473",
            "submissions/public_aware_stack_blend_20260622/target_select_public_tight_last0.572477_full0.592635.csv=0.5905116492",
            "submissions/fast_temporal_stack/02_guarded_targetwise.csv=0.5935970063",
        ],
    )
    p.add_argument("--ridge", type=float, default=0.002)
    p.add_argument("--max-abs-lambda", type=float, default=2.0)
    p.add_argument("--output-dir", default="research/public_score_pseudo_blend_20260622")
    p.add_argument("--submission-dir", default="submissions/public_score_pseudo_blend_20260622")
    p.add_argument("--top-source-count", type=int, default=28)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading data/folds/anchor")
    _, ytr, _, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    anchor_oof, anchor_test = load_anchor()
    known = parse_known_public(args.known_public)

    log("Estimating public-score pseudo posterior")
    q_public, constraints = estimate_pseudo_posterior(anchor_test, known, args.ridge, args.max_abs_lambda)
    constraints.to_csv(out_dir / "public_constraint_fit.csv", index=False)
    write_prediction(out_dir / "pseudo_public_posterior.csv", mte, q_public)

    log("Loading OOF/test sources")
    sources = discover_pairs(args.source_dirs)
    sources["anchor"] = (anchor_oof, anchor_test)
    log(f"Loaded {len(sources)} paired sources")

    base_rows = []
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    for name, (oof, test) in sources.items():
        row = candidate_score_row(name, ytr, folds, anchor_oof, anchor_test, q_public, known, oof, test)
        base_rows.append(row)
    base_scores = pd.DataFrame(base_rows).sort_values(["selector_score", "pseudo_public_logloss"]).reset_index(drop=True)
    base_scores.to_csv(out_dir / "base_source_scores.csv", index=False)

    eligible = base_scores[
        (base_scores["full_logloss"] <= mean_loss(ytr, anchor_oof, full_mask) + 0.020)
        & (base_scores["last_logloss"] <= mean_loss(ytr, anchor_oof, last_mask) + 0.030)
    ]["candidate"].tolist()
    top_names = [n for n in eligible if n != "anchor"][: args.top_source_count]
    log(f"Using {len(top_names)} top sources for blend search")

    pool: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        n: pair for n, pair in sources.items() if n == "anchor" or n in top_names
    }
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for n in top_names:
        candidates[f"base__{n.replace('/', '__')}"] = sources[n]

    log("Building pairwise and anchor-shrink blends")
    weights = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]
    for n in top_names[:18]:
        oof, test = sources[n]
        for w in [0.45, 0.60, 0.75, 0.88, 0.94]:
            cname = f"anchor_logit_{n.replace('/', '__')}_w{str(w).replace('.', 'p')}"
            candidates[cname] = (
                blend_pred(oof, anchor_oof, w, "logit"),
                blend_pred(test, anchor_test, w, "logit"),
            )
    for i, a_name in enumerate(top_names[:14]):
        for b_name in top_names[i + 1 : 14]:
            a_oof, a_test = sources[a_name]
            b_oof, b_test = sources[b_name]
            for w in weights:
                cname = f"pair_logit_{a_name.replace('/', '__')}__{b_name.replace('/', '__')}_w{str(w).replace('.', 'p')}"
                candidates[cname] = (
                    blend_pred(a_oof, b_oof, w, "logit"),
                    blend_pred(a_test, b_test, w, "logit"),
                )

    log("Fitting pseudo-public targetwise selections")
    choice_tables = []
    profiles = [
        ("pseudo_target_tight", 0.004, 0.006, 0.45, 0.030),
        ("pseudo_target_balanced", 0.007, 0.012, 0.28, 0.020),
        ("pseudo_target_public_heavy", 0.012, 0.020, 0.12, 0.012),
        ("pseudo_target_public_max", 0.020, 0.035, 0.04, 0.006),
    ]
    source_pool = {n: sources[n] for n in top_names}
    for name, full_guard, last_guard, cv_weight, move_penalty in profiles:
        poof, ptest, choices = targetwise_pseudo_select(
            name,
            ytr,
            folds,
            anchor_oof,
            anchor_test,
            q_public,
            source_pool,
            full_guard,
            last_guard,
            cv_weight,
            move_penalty,
        )
        candidates[name] = (poof, ptest)
        choice_tables.append(choices)
        for w in [0.70, 0.82, 0.90, 0.96]:
            cname = f"{name}_anchorlogit_w{str(w).replace('.', 'p')}"
            candidates[cname] = (
                blend_pred(poof, anchor_oof, w, "logit"),
                blend_pred(ptest, anchor_test, w, "logit"),
            )

    log("Fitting pseudo-public simplex blends")
    simplex_rows = []
    for mode in ["prob", "logit"]:
        for cv_weight in [0.02, 0.08, 0.18]:
            for l2 in [0.002, 0.01, 0.04]:
                name = f"pseudo_simplex_{mode}_cv{str(cv_weight).replace('.', 'p')}_l2{str(l2).replace('.', 'p')}"
                poof, ptest, details = fit_simplex_to_pseudo(
                    name,
                    ytr,
                    folds,
                    q_public,
                    anchor_oof,
                    anchor_test,
                    source_pool,
                    top_names,
                    cv_weight,
                    l2,
                    mode,
                )
                candidates[name] = (poof, ptest)
                simplex_rows.append(details)
                for w in [0.82, 0.92]:
                    cname = f"{name}_anchorlogit_w{str(w).replace('.', 'p')}"
                    candidates[cname] = (
                        blend_pred(poof, anchor_oof, w, "logit"),
                        blend_pred(ptest, anchor_test, w, "logit"),
                    )

    if choice_tables:
        pd.concat(choice_tables, ignore_index=True, sort=False).to_csv(out_dir / "targetwise_choices.csv", index=False)
    if simplex_rows:
        pd.concat(simplex_rows, ignore_index=True, sort=False).to_csv(out_dir / "simplex_details.csv", index=False)

    log(f"Scoring {len(candidates)} generated candidates")
    rows = []
    for name, (oof, test) in candidates.items():
        rows.append(candidate_score_row(name, ytr, folds, anchor_oof, anchor_test, q_public, known, oof, test))
    scores = pd.DataFrame(rows).sort_values(["selector_score", "pseudo_public_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    plot_scores(out_dir, scores)

    # Save top candidates that keep CV in a reasonable band.  Also keep the
    # absolute pseudo-public top candidate for inspection.
    anchor_full = mean_loss(ytr, anchor_oof, full_mask)
    anchor_last = mean_loss(ytr, anchor_oof, last_mask)
    save_rows = scores[
        (scores["full_logloss"] <= anchor_full + 0.014)
        & (scores["last_logloss"] <= anchor_last + 0.010)
    ].head(12)
    pseudo_best = scores.head(1)
    save_rows = pd.concat([pseudo_best, save_rows], ignore_index=True).drop_duplicates("candidate").head(12)

    for _, row in save_rows.iterrows():
        name = str(row["candidate"])
        poof, ptest = candidates[name]
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:160]
        stem = f"{safe_name}_pseudo{row['pseudo_public_logloss']:.6f}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}"
        write_prediction(out_dir / f"{stem}_oof.csv", mtr, poof, ytr)
        write_prediction(out_dir / f"{stem}_test_pred.csv", mte, ptest)
        write_submission(sub_dir / f"{stem}.csv", mte, ptest)

    selected = save_rows.iloc[0] if not save_rows.empty else scores.iloc[0]
    target_choices = pd.concat(choice_tables, ignore_index=True, sort=False) if choice_tables else pd.DataFrame()
    build_report(out_dir, scores, constraints, target_choices, selected, known)
    report = {
        "known_public": [{"path": str(k.path), "score": k.score} for k in known],
        "selected_candidate": str(selected["candidate"]),
        "selected_submission_dir": str(sub_dir),
        "selected_metrics": {
            "pseudo_public_logloss": float(selected["pseudo_public_logloss"]),
            "full_logloss": float(selected["full_logloss"]),
            "last_logloss": float(selected["last_logloss"]),
            "selector_score": float(selected["selector_score"]),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    log("Top candidates:")
    print(scores.head(12)[["candidate", "pseudo_public_logloss", "full_logloss", "last_logloss", "selector_score"]].to_string(index=False))
    log(f"Wrote report: {out_dir / 'PUBLIC_PSEUDO_BLEND_REPORT.md'}")
    log(f"Wrote submissions: {sub_dir}")


if __name__ == "__main__":
    main()
