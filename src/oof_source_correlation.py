"""Target-wise OOF source correlation diagnostics.

Kaggle-style notebooks often inspect OOF correlations before blending.  This
script brings that workflow to the Dacon ETRI prediction banks:

- read `research/oof_sparse_greedy/oof_bank.csv`
- optionally read residual optimizer `*_oof.csv` files from extra directories
- score every source by full/last logloss
- write target-wise correlation matrices and a long pair table

It does not create submissions.  It is a research diagnostic for deciding which
sources are genuinely diverse enough to blend.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .train_temporal_prior import clip

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y, p) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def source_name_from_oof_path(path: Path) -> str:
    name = path.stem
    return name[:-4] if name.endswith("_oof") else name


def load_bank_sources(path: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    sources: dict[str, pd.DataFrame] = {}
    if not path.exists():
        return sources, None
    bank = pd.read_csv(path)
    label_cols = [f"label__{t}" for t in TARGETS]
    labels = None
    if all(c in bank.columns for c in label_cols):
        labels = pd.DataFrame({t: bank[f"label__{t}"].astype(int).values for t in TARGETS})
    for col in bank.columns:
        if "__" not in col or col.startswith("label__"):
            continue
        source, target = col.rsplit("__", 1)
        if target not in TARGETS:
            continue
        if source not in sources:
            sources[source] = pd.DataFrame(index=bank.index, columns=TARGETS, dtype=float)
        sources[source][target] = clip(bank[col].values)
    return sources, labels


def load_oof_file(path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame | None]:
    df = pd.read_csv(path)
    if not all(t in df.columns for t in TARGETS):
        raise ValueError(f"{path} does not contain all target columns")
    pred = pd.DataFrame({t: clip(df[t].values) for t in TARGETS})
    label_cols = [f"label__{t}" for t in TARGETS]
    labels = None
    if all(c in df.columns for c in label_cols):
        labels = pd.DataFrame({t: df[f"label__{t}"].astype(int).values for t in TARGETS})
    return source_name_from_oof_path(path), pred, labels


def load_extra_sources(paths: list[str]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    sources: dict[str, pd.DataFrame] = {}
    labels = None
    for item in paths:
        p = ROOT / item if not Path(item).is_absolute() else Path(item)
        files = sorted(p.glob("*_oof.csv")) if p.is_dir() else [p]
        for file in files:
            if not file.exists() or file.name.startswith("anchor_"):
                continue
            raw_name, pred, lab = load_oof_file(file)
            prefix = p.name if p.is_dir() else file.parent.name
            name = f"{prefix}__{raw_name}"
            sources[name] = pred
            if labels is None and lab is not None:
                labels = lab
    return sources, labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="research/oof_sparse_greedy/oof_bank.csv")
    p.add_argument("--extra-oof", nargs="*", default=[])
    p.add_argument("--output-dir", default="research/oof_source_correlation")
    p.add_argument("--min-abs-corr", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading labels/folds")
    _, ytr, _, mtr, _, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full = np.ones(len(ytr), dtype=bool)
    last = folds == (C.N_SPLITS - 1)

    log("Loading OOF sources")
    sources, labels_from_bank = load_bank_sources(ROOT / args.bank)
    extra_sources, labels_from_extra = load_extra_sources(args.extra_oof)
    sources.update(extra_sources)
    labels = labels_from_bank if labels_from_bank is not None else labels_from_extra
    if labels is None:
        labels = ytr
    if not sources:
        raise ValueError("No OOF sources found")

    score_rows = []
    pair_rows = []
    for target in TARGETS:
        target_sources = {
            name: pred[target].values
            for name, pred in sources.items()
            if target in pred.columns and pred[target].notna().all()
        }
        matrix = pd.DataFrame(target_sources).corr()
        matrix.to_csv(out_dir / f"corr_matrix_{target}.csv")
        for name, pred in target_sources.items():
            score_rows.append({
                "target": target,
                "source": name,
                "full_logloss": safe_loss(labels[target].values[full], pred[full]),
                "last_logloss": safe_loss(labels[target].values[last], pred[last]),
                "pred_std": float(np.std(pred)),
                "pred_mean": float(np.mean(pred)),
            })
        names = list(target_sources)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                corr = float(matrix.loc[a, b])
                if abs(corr) < args.min_abs_corr:
                    continue
                pair_rows.append({
                    "target": target,
                    "source_a": a,
                    "source_b": b,
                    "corr": corr,
                    "abs_corr": abs(corr),
                })

    scores = pd.DataFrame(score_rows).sort_values(["target", "full_logloss", "last_logloss"])
    pair_cols = ["target", "source_a", "source_b", "corr", "abs_corr"]
    pairs = pd.DataFrame(pair_rows, columns=pair_cols)
    if not pairs.empty:
        pairs = pairs.sort_values(["target", "abs_corr"], ascending=[True, False])
    scores.to_csv(out_dir / "source_scores.csv", index=False)
    pairs.to_csv(out_dir / "source_correlations_long.csv", index=False)

    print("\n=== Source scores head ===")
    print(scores.head(40).to_string(index=False))
    print("\n=== Highest source correlations ===")
    print(pairs.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
