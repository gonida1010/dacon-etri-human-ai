"""Blend optimized residual single-model outputs with proven prior sources.

Inputs are OOF/test prediction files from residual_single_model_opt plus the
public-confirmed same-subject KNN guarded family.  The search is target-wise and
uses OOF only; no submission score is assumed.
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
from .kaggle_last_mile import build_knn_source, targetwise_guarded_blend
from .train_temporal_prior import clip

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT
EPS = C.PROB_CLIP


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y, p) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_anchor_bank(bank_dir: Path, n_train: int, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_bank = pd.read_csv(bank_dir / "oof_bank.csv")
    test_bank = pd.read_csv(bank_dir / "test_bank.csv")
    return (
        pd.DataFrame({t: oof_bank[f"anchor__{t}"].values for t in TARGETS}),
        pd.DataFrame({t: test_bank[f"anchor__{t}"].values for t in TARGETS}),
    )


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"{path} missing targets: {missing}")
    return pd.DataFrame({t: clip(df[t].values) for t in TARGETS})


def blend_values(a: np.ndarray, b: np.ndarray, w: float, mode: str) -> np.ndarray:
    if mode == "logit":
        return clip(sigmoid(w * logit(a) + (1.0 - w) * logit(b)))
    return clip(w * a + (1.0 - w) * b)


def blend_three_values(a: np.ndarray, b: np.ndarray, c: np.ndarray, wa: float, wb: float, mode: str) -> np.ndarray:
    wc = 1.0 - wa - wb
    if mode == "logit":
        return clip(sigmoid(wa * logit(a) + wb * logit(b) + wc * logit(c)))
    return clip(wa * a + wb * b + wc * c)


def fold_losses(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> list[float]:
    return [safe_loss(y[folds == f], p[folds == f]) for f in sorted(np.unique(folds))]


def rank_score(full: float, last: float, anchor_full: float, folds_scores: list[float]) -> float:
    tail = folds_scores[-3:]
    return last + 0.85 * max(0.0, full - anchor_full) + 0.20 * max(0.0, max(tail) - float(np.mean(tail)))


def target_candidates(
    target: str,
    sources_oof: dict[str, pd.DataFrame],
    sources_test: dict[str, pd.DataFrame],
    weights: list[float],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    names = list(sources_oof)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name in names:
        out.append((name, sources_oof[name][target].values, sources_test[name][target].values))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for mode in ["arith", "logit"]:
                for w in weights:
                    cname = f"{mode}:{a}:{w:.2f}+{b}:{1-w:.2f}"
                    out.append((
                        cname,
                        blend_values(sources_oof[a][target].values, sources_oof[b][target].values, w, mode),
                        blend_values(sources_test[a][target].values, sources_test[b][target].values, w, mode),
                    ))
    triple_sets = [
        ("anchor", "ridge_residual_composite", "knn_targetwise_guarded"),
        ("ridge_residual_full", "ridge_residual_composite", "knn_targetwise_guarded"),
    ]
    triple_weights = [(0.2, 0.2), (0.2, 0.35), (0.35, 0.2), (0.35, 0.35), (0.5, 0.2), (0.2, 0.5)]
    for a, b, c in triple_sets:
        if a not in sources_oof or b not in sources_oof or c not in sources_oof:
            continue
        for mode in ["arith", "logit"]:
            for wa, wb in triple_weights:
                if wa + wb >= 1.0:
                    continue
                wc = 1.0 - wa - wb
                cname = f"{mode}:{a}:{wa:.2f}+{b}:{wb:.2f}+{c}:{wc:.2f}"
                out.append((
                    cname,
                    blend_three_values(
                        sources_oof[a][target].values,
                        sources_oof[b][target].values,
                        sources_oof[c][target].values,
                        wa,
                        wb,
                        mode,
                    ),
                    blend_three_values(
                        sources_test[a][target].values,
                        sources_test[b][target].values,
                        sources_test[c][target].values,
                        wa,
                        wb,
                        mode,
                    ),
                ))
    return out


def build_targetwise(
    mode: str,
    search: pd.DataFrame,
    candidate_arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.DataFrame(index=anchor_oof.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    chosen = []
    for target in TARGETS:
        rows = search[search["target"].eq(target)].copy()
        if mode == "composite":
            rows = rows.sort_values(["rank_score", "last_logloss", "full_logloss"])
        elif mode == "last":
            rows = rows.sort_values(["last_logloss", "full_logloss", "rank_score"])
        elif mode == "full":
            rows = rows.sort_values(["full_logloss", "last_logloss", "rank_score"])
        else:
            raise ValueError(mode)
        row = rows.iloc[0].to_dict()
        pred_oof, pred_test = candidate_arrays[(target, row["source"])]
        oof[target] = pred_oof
        test[target] = pred_test
        row["selection_mode"] = mode
        chosen.append(row)
    return oof, test, pd.DataFrame(chosen)


def candidate_stability(ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray, name: str) -> dict:
    row = {"candidate": name}
    full_mask = np.ones(len(ytr), dtype=bool)
    row["full_logloss"] = mean_loss(ytr, pred, full_mask)
    vals = []
    for fold in sorted(np.unique(folds)):
        val = mean_loss(ytr, pred, folds == fold)
        row[f"fold{fold}_logloss"] = val
        vals.append(val)
    row["last_logloss"] = vals[-1]
    row["tail3_mean"] = float(np.mean(vals[-3:]))
    row["tail3_worst"] = float(np.max(vals[-3:]))
    row["fold_std"] = float(np.std(vals))
    return row


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test.copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def write_prediction_frame(path: Path, meta: pd.DataFrame, pred: pd.DataFrame, y: pd.DataFrame | None = None) -> None:
    out = meta.copy()
    if y is not None:
        for target in TARGETS:
            out[f"label__{target}"] = y[target].values
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-dir", default="research/oof_sparse_greedy")
    p.add_argument("--ridge-dir", default="research/residual_single_model_opt_ridge")
    p.add_argument("--output-dir", default="research/residual_submission_blend")
    p.add_argument("--submission-dir", default="submissions/residual_submission_blend")
    p.add_argument("--full-guard", type=float, default=0.003)
    p.add_argument("--min-last-gain", type=float, default=0.0001)
    p.add_argument("--weights", nargs="*", type=float, default=[0.2, 0.35, 0.5, 0.65, 0.8])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset and residual sources")
    _, ytr, _, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full = np.ones(len(ytr), dtype=bool)
    last = folds == (C.N_SPLITS - 1)
    anchor_oof, anchor_test = load_anchor_bank(ROOT / args.bank_dir, len(ytr), len(mte))
    anchor_full = mean_loss(ytr, anchor_oof, full)
    anchor_last = mean_loss(ytr, anchor_oof, last)

    ridge_dir = ROOT / args.ridge_dir
    sources_oof = {
        "anchor": anchor_oof,
        "ridge_residual_composite": load_pred(ridge_dir / "ridge_residual_composite_oof.csv"),
        "ridge_residual_full": load_pred(ridge_dir / "ridge_residual_full_oof.csv"),
    }
    sources_test = {
        "anchor": anchor_test,
        "ridge_residual_composite": load_pred(ridge_dir / "ridge_residual_composite_test_pred.csv"),
        "ridge_residual_full": load_pred(ridge_dir / "ridge_residual_full_test_pred.csv"),
    }

    log("Rebuilding KNN guarded source")
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
        args.full_guard,
        args.min_last_gain,
        args.weights,
        "knn_targetwise_guarded",
    )
    knn_choices.to_csv(out_dir / "knn_targetwise_choices.csv", index=False)
    sources_oof["knn_targetwise_guarded"] = knn_oof
    sources_test["knn_targetwise_guarded"] = knn_test

    log("Searching target-wise source blends")
    rows = []
    arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for target in TARGETS:
        y = ytr[target].values.astype(int)
        target_anchor_full = safe_loss(y, anchor_oof[target].values)
        for source, po, pt in target_candidates(target, sources_oof, sources_test, args.weights):
            full_loss = safe_loss(y, po)
            losses = fold_losses(y, po, folds)
            last_loss = losses[-1]
            row = {
                "target": target,
                "source": source,
                "full_logloss": full_loss,
                "last_logloss": last_loss,
                "full_delta_vs_anchor": full_loss - target_anchor_full,
                "rank_score": rank_score(full_loss, last_loss, target_anchor_full, losses),
            }
            for i, loss in enumerate(losses):
                row[f"fold{i}_logloss"] = loss
            rows.append(row)
            arrays[(target, source)] = (po, pt)
    search = pd.DataFrame(rows).sort_values(["target", "rank_score", "last_logloss"])
    search.to_csv(out_dir / "target_blend_search.csv", index=False)

    candidates = {}
    choices_all = []
    for mode in ["composite", "last", "full"]:
        name = f"ridge_knn_blend_{mode}"
        po, pt, choices = build_targetwise(mode, search, arrays, anchor_oof, anchor_test)
        choices.to_csv(out_dir / f"{name}_target_choices.csv", index=False)
        choices_all.append(choices.assign(candidate=name))
        candidates[name] = (po, pt)

    score_rows = []
    for name, (po, pt) in candidates.items():
        row = candidate_stability(ytr, po, folds, name)
        row["full_delta_vs_anchor"] = row["full_logloss"] - anchor_full
        row["last_delta_vs_anchor"] = row["last_logloss"] - anchor_last
        row["rank_score"] = rank_score(row["full_logloss"], row["last_logloss"], anchor_full, [
            row[f"fold{f}_logloss"] for f in sorted(np.unique(folds))
        ])
        score_rows.append(row)
        safe = name.replace("/", "_").replace(".", "p")
        write_prediction_frame(out_dir / f"{safe}_oof.csv", mtr, po, ytr)
        write_prediction_frame(out_dir / f"{safe}_test_pred.csv", mte, pt)
        write_submission(sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv", mte, pt)

    scores = pd.DataFrame(score_rows).sort_values(["rank_score", "last_logloss", "full_logloss"])
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.concat(choices_all, ignore_index=True).to_csv(out_dir / "target_choices_all.csv", index=False)
    report = {
        "anchor": {"full_logloss": anchor_full, "last_logloss": anchor_last},
        "sources": list(sources_oof),
        "candidate_scores": scores.to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Ridge residual + KNN blend candidates ===")
    print(scores[["candidate", "rank_score", "full_logloss", "last_logloss", "tail3_worst", "fold_std"]].to_string(index=False))
    print("\n=== Composite choices ===")
    comp = pd.read_csv(out_dir / "ridge_knn_blend_composite_target_choices.csv")
    print(comp[["target", "source", "full_logloss", "last_logloss", "rank_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
