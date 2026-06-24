"""Public-feedback-aware OOF stack/blend search.

This applies the useful Kaggle notebook ideas to the Dacon target setup:
- fold-safe meta stackers over OOF prediction sources
- simplex weighted blends over logit/probability source predictions
- target-wise blend/selection with an explicit penalty for directions that
  matched the bad public-feedback submission

It does not use public labels.  The supplied public score is only used as a
warning that the submitted file's movement pattern should be penalized.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .direction_gated_search import DEFAULT_SOURCE_DIRS, discover_pairs, load_anchor
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
EPS = C.PROB_CLIP


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def read_submission(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
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


def fold_losses(ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> list[float]:
    return [mean_loss(ytr, pred, folds == f) for f in sorted(np.unique(folds))]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def source_score(
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    name: str,
    oof: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> dict:
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    full = safe_loss(ytr[target].values[full_mask], oof[target].values[full_mask])
    last = safe_loss(ytr[target].values[last_mask], oof[target].values[last_mask])
    base_full = safe_loss(ytr[target].values[full_mask], anchor_oof[target].values[full_mask])
    base_last = safe_loss(ytr[target].values[last_mask], anchor_oof[target].values[last_mask])
    delta = test[target].values - anchor_test[target].values
    return {
        "source": name,
        "target": target,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_anchor": full - base_full,
        "last_delta_vs_anchor": last - base_last,
        "test_abs_delta_mean": float(np.mean(np.abs(delta))),
        "test_mean_delta": float(np.mean(delta)),
        "test_up_rate": float(np.mean(delta > 1e-12)),
        "test_down_rate": float(np.mean(delta < -1e-12)),
        "public_risk": public_direction_risk(target, delta),
    }


def public_direction_risk(target: str, delta: np.ndarray) -> float:
    up = np.maximum(0.0, delta)
    down = np.maximum(0.0, -delta)
    abs_delta = np.abs(delta)
    if target == "Q1":
        return 0.05 * float(abs_delta.mean())
    if target == "Q2":
        return 0.35 * float(abs_delta.mean()) + 0.20 * float(down.mean())
    if target == "Q3":
        return 0.15 * float(abs_delta.mean()) + 0.10 * float(up.mean())
    if target == "S1":
        return 0.30 * float(abs_delta.mean()) + 0.20 * float(down.mean())
    if target == "S2":
        return 1.60 * float(up.mean()) + 0.05 * float(down.mean())
    if target == "S3":
        return 2.00 * float(up.mean()) + 0.20 * float(abs_delta.mean())
    if target == "S4":
        return 2.50 * float(down.mean()) + 0.02 * float(up.mean())
    return float(abs_delta.mean())


def bad_alignment_risk(
    target: str,
    candidate_test: np.ndarray,
    anchor_test: np.ndarray,
    submitted_test: pd.DataFrame | None,
) -> float:
    if submitted_test is None:
        return 0.0
    cand_delta = candidate_test - anchor_test
    bad_delta = submitted_test[target].values - anchor_test
    same = np.maximum(0.0, cand_delta * bad_delta)
    if target in {"S3", "S4", "Q2", "S1"}:
        return float(np.mean(same))
    return 0.05 * float(np.mean(same))


def dedupe_sources(
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    corr_threshold: float,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    rows = []
    keep_by_target: dict[str, list[str]] = {t: [] for t in TARGETS}
    for target in TARGETS:
        scored = []
        for name, (oof, test) in sources.items():
            row = source_score(ytr, folds, anchor_oof, anchor_test, name, oof, test, target)
            row["rank_score"] = (
                row["last_logloss"]
                + 0.60 * max(0.0, row["full_delta_vs_anchor"])
                + 6.0 * row["public_risk"]
                + 0.05 * row["test_abs_delta_mean"]
            )
            scored.append(row)
        scored = sorted(scored, key=lambda r: (r["rank_score"], r["last_logloss"], r["full_logloss"]))
        kept: list[str] = []
        kept_arrays: list[np.ndarray] = []
        for row in scored:
            arr = sources[row["source"]][0][target].values
            if all(abs(corr(arr, prev)) < corr_threshold for prev in kept_arrays):
                kept.append(row["source"])
                kept_arrays.append(arr)
            if len(kept) >= 24:
                break
        keep_by_target[target] = kept
        for row in scored:
            row["kept_for_target"] = row["source"] in kept
            rows.append(row)

    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, pair in sources.items():
        if any(name in keep_by_target[t] for t in TARGETS):
            out[name] = pair
    if "anchor" in sources:
        out["anchor"] = sources["anchor"]
    return out, pd.DataFrame(rows)


def top_sources_for_target(source_scores: pd.DataFrame, target: str, max_sources: int) -> list[str]:
    rows = source_scores[(source_scores["target"].eq(target)) & (source_scores["kept_for_target"])].copy()
    rows = rows.sort_values(["rank_score", "last_logloss", "full_logloss"])
    names = ["anchor"]
    for name in rows["source"].tolist():
        if name not in names:
            names.append(name)
        if len(names) >= max_sources:
            break
    return names


def feature_matrix(
    target: str,
    frame_map: dict[str, pd.DataFrame],
    names: list[str],
    index: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    vals = []
    for name in names:
        arr = frame_map[name][target].values
        if index is not None:
            arr = arr[index]
        vals.append(arr)
    p = np.column_stack(vals)
    z = logit(p)
    if mode == "logit":
        return z
    if mode == "both":
        return np.column_stack([p, z, z - z[:, [0]]])
    raise ValueError(mode)


def fit_logreg_stacker(
    name: str,
    ytr: pd.DataFrame,
    folds: np.ndarray,
    source_scores: pd.DataFrame,
    oof_map: dict[str, pd.DataFrame],
    test_map: dict[str, pd.DataFrame],
    max_sources: int,
    c_value: float,
    class_weight: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(0.0, index=test_map["anchor"].index, columns=TARGETS, dtype=float)
    rows = []
    for target in TARGETS:
        y = ytr[target].values.astype(int)
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            names = top_sources_for_target(source_scores, target, max_sources)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=class_weight,
                ),
            )
            Xtr = feature_matrix(target, oof_map, names, tr_idx, "both")
            Xva = feature_matrix(target, oof_map, names, va_idx, "both")
            Xte = feature_matrix(target, test_map, names, None, "both")
            model.fit(Xtr, y[tr_idx])
            p_va = clip(model.predict_proba(Xva)[:, 1])
            p_te = clip(model.predict_proba(Xte)[:, 1])
            oof.loc[va_idx, target] = p_va
            test[target] += p_te / C.N_SPLITS
            rows.append({
                "candidate": name,
                "target": target,
                "fold": int(fold),
                "sources": json.dumps(names),
                "fold_logloss": safe_loss(y[va_idx], p_va),
            })
    return (
        pd.DataFrame({t: clip(oof[t].values) for t in TARGETS}),
        pd.DataFrame({t: clip(test[t].values) for t in TARGETS}),
        pd.DataFrame(rows),
    )


def simplex_fit(X: np.ndarray, y: np.ndarray, mode: str, l2: float) -> np.ndarray:
    n = X.shape[1]
    x0 = np.ones(n) / n

    def predict(w: np.ndarray) -> np.ndarray:
        if mode == "logit":
            return sigmoid(logit(X) @ w)
        return X @ w

    def obj(w: np.ndarray) -> float:
        return safe_loss(y, predict(w)) + l2 * float(np.sum((w - x0) ** 2))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    if not res.success:
        return x0
    w = np.maximum(0.0, np.asarray(res.x, dtype=float))
    return w / max(w.sum(), 1e-12)


def fit_simplex_blend(
    name: str,
    ytr: pd.DataFrame,
    folds: np.ndarray,
    source_scores: pd.DataFrame,
    oof_map: dict[str, pd.DataFrame],
    test_map: dict[str, pd.DataFrame],
    max_sources: int,
    mode: str,
    l2: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(0.0, index=test_map["anchor"].index, columns=TARGETS, dtype=float)
    rows = []
    for target in TARGETS:
        y = ytr[target].values.astype(int)
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            names = top_sources_for_target(source_scores, target, max_sources)
            Xtr = np.column_stack([oof_map[n][target].values[tr_idx] for n in names])
            Xva = np.column_stack([oof_map[n][target].values[va_idx] for n in names])
            Xte = np.column_stack([test_map[n][target].values for n in names])
            w = simplex_fit(Xtr, y[tr_idx], mode, l2)
            if mode == "logit":
                p_va = sigmoid(logit(Xva) @ w)
                p_te = sigmoid(logit(Xte) @ w)
            else:
                p_va = Xva @ w
                p_te = Xte @ w
            oof.loc[va_idx, target] = clip(p_va)
            test[target] += clip(p_te) / C.N_SPLITS
            rows.append({
                "candidate": name,
                "target": target,
                "fold": int(fold),
                "sources": json.dumps(names),
                "weights": json.dumps({n: float(v) for n, v in zip(names, w) if v > 1e-4}),
                "fold_logloss": safe_loss(y[va_idx], p_va),
            })
    return (
        pd.DataFrame({t: clip(oof[t].values) for t in TARGETS}),
        pd.DataFrame({t: clip(test[t].values) for t in TARGETS}),
        pd.DataFrame(rows),
    )


def blend_pred(a: pd.DataFrame, b: pd.DataFrame, w: float, mode: str) -> pd.DataFrame:
    out = pd.DataFrame(index=a.index, columns=TARGETS, dtype=float)
    for target in TARGETS:
        if mode == "logit":
            out[target] = clip(sigmoid(w * logit(a[target].values) + (1.0 - w) * logit(b[target].values)))
        else:
            out[target] = clip(w * a[target].values + (1.0 - w) * b[target].values)
    return out


def temperature_pred(pred: pd.DataFrame, anchor: pd.DataFrame, alpha: float, temp: float) -> pd.DataFrame:
    out = pd.DataFrame(index=pred.index, columns=TARGETS, dtype=float)
    for target in TARGETS:
        z = logit(anchor[target].values) + alpha * (logit(pred[target].values) - logit(anchor[target].values)) / temp
        out[target] = clip(sigmoid(z))
    return out


def candidate_score_row(
    name: str,
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    submitted_test: pd.DataFrame | None,
    oof: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    losses = fold_losses(ytr, oof, folds)
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    anchor_full = mean_loss(ytr, anchor_oof, full_mask)
    anchor_last = mean_loss(ytr, anchor_oof, last_mask)
    risk = 0.0
    align = 0.0
    for target in TARGETS:
        delta = test[target].values - anchor_test[target].values
        risk += public_direction_risk(target, delta)
        align += bad_alignment_risk(target, test[target].values, anchor_test[target].values, submitted_test)
    risk /= len(TARGETS)
    align /= len(TARGETS)
    full = mean_loss(ytr, oof, full_mask)
    last = mean_loss(ytr, oof, last_mask)
    move = float(np.mean(np.abs(test[TARGETS].values - anchor_test[TARGETS].values)))
    return {
        "candidate": name,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "fold_std": float(np.std(losses)),
        "tail3_worst": float(max(losses[-3:])),
        "test_abs_delta_mean_vs_anchor": move,
        "public_direction_risk": risk,
        "bad_submission_alignment": align,
        "selector_score": (
            last
            + 0.55 * max(0.0, full - anchor_full)
            + 2.5 * risk
            + 7.5 * align
            + 0.08 * move
            + 0.08 * max(0.0, max(losses[-3:]) - np.mean(losses[-3:]))
        ),
        **{f"fold{i}_logloss": v for i, v in enumerate(losses)},
    }


def select_targetwise(
    name: str,
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    submitted_test: pd.DataFrame | None,
    pool: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    full_guard: float,
    min_last_gain: float,
    risk_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_mask = np.ones(len(ytr), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    oof = anchor_oof.copy()
    test = anchor_test.copy()
    rows = []
    for target in TARGETS:
        base_full = safe_loss(ytr[target].values[full_mask], anchor_oof[target].values[full_mask])
        base_last = safe_loss(ytr[target].values[last_mask], anchor_oof[target].values[last_mask])
        best = {
            "target": target,
            "source": "anchor",
            "full_logloss": base_full,
            "last_logloss": base_last,
            "full_delta_vs_anchor": 0.0,
            "last_delta_vs_anchor": 0.0,
            "public_risk": 0.0,
            "bad_alignment": 0.0,
            "selector_score": base_last,
        }
        for source, (poof, ptest) in pool.items():
            full = safe_loss(ytr[target].values[full_mask], poof[target].values[full_mask])
            last = safe_loss(ytr[target].values[last_mask], poof[target].values[last_mask])
            delta = ptest[target].values - anchor_test[target].values
            risk = public_direction_risk(target, delta)
            align = bad_alignment_risk(target, ptest[target].values, anchor_test[target].values, submitted_test)
            if full > base_full + full_guard:
                continue
            if last > base_last - min_last_gain:
                continue
            if risk > risk_limit:
                continue
            score = last + 0.45 * max(0.0, full - base_full) + 3.0 * risk + 8.0 * align + 0.05 * float(np.mean(np.abs(delta)))
            if score < best["selector_score"]:
                best = {
                    "target": target,
                    "source": source,
                    "full_logloss": full,
                    "last_logloss": last,
                    "full_delta_vs_anchor": full - base_full,
                    "last_delta_vs_anchor": last - base_last,
                    "public_risk": risk,
                    "bad_alignment": align,
                    "selector_score": score,
                }
        if best["source"] != "anchor":
            oof[target] = pool[best["source"]][0][target].values
            test[target] = pool[best["source"]][1][target].values
        rows.append(best)
    return oof, test, pd.DataFrame(rows).assign(candidate=name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dirs", nargs="*", default=DEFAULT_SOURCE_DIRS + [
        "research/direction_gated_ablation_20260622",
        "research/direction_gated_search_20260622",
        "research/diverse_single_stack",
        "research/kaggle_last_mile",
    ])
    p.add_argument("--submitted", default="submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv")
    p.add_argument("--submitted-public", type=float, default=0.5920118473)
    p.add_argument("--output-dir", default="research/public_aware_stack_blend_20260622")
    p.add_argument("--submission-dir", default="submissions/public_aware_stack_blend_20260622")
    p.add_argument("--corr-threshold", type=float, default=0.9992)
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
    anchor_oof, anchor_test = load_anchor()
    submitted_test = read_submission(ROOT / args.submitted)

    log("Loading OOF/test source pairs")
    sources = discover_pairs(args.source_dirs)
    sources["anchor"] = (anchor_oof, anchor_test)
    log(f"Loaded {len(sources)} raw sources")

    log("De-duplicating sources by target")
    kept_sources, source_scores = dedupe_sources(
        ytr, folds, anchor_oof, anchor_test, sources, args.corr_threshold
    )
    source_scores.to_csv(out_dir / "source_target_scores.csv", index=False)
    oof_map = {name: pair[0] for name, pair in kept_sources.items()}
    test_map = {name: pair[1] for name, pair in kept_sources.items()}
    log(f"Kept {len(kept_sources)} sources after target dedupe")

    pool: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        name: pair for name, pair in kept_sources.items() if name != "anchor"
    }
    meta_rows = []
    for max_sources in [8, 12, 16]:
        for c_value in [0.05, 0.15, 0.5]:
            for class_weight in [None, "balanced"]:
                name = f"logreg_stack_top{max_sources}_C{str(c_value).replace('.', 'p')}_{class_weight or 'plain'}"
                log(f"Fitting {name}")
                poof, ptest, details = fit_logreg_stacker(
                    name, ytr, folds, source_scores, oof_map, test_map, max_sources, c_value, class_weight
                )
                pool[name] = (poof, ptest)
                meta_rows.append(details)
        for mode in ["prob", "logit"]:
            for l2 in [0.001, 0.01, 0.05]:
                name = f"simplex_{mode}_top{max_sources}_l2{str(l2).replace('.', 'p')}"
                log(f"Fitting {name}")
                poof, ptest, details = fit_simplex_blend(
                    name, ytr, folds, source_scores, oof_map, test_map, max_sources, mode, l2
                )
                pool[name] = (poof, ptest)
                meta_rows.append(details)

    if meta_rows:
        pd.concat(meta_rows, ignore_index=True, sort=False).to_csv(out_dir / "meta_fit_details.csv", index=False)

    log("Building stack/blend variants")
    expanded_pool = dict(pool)
    for base_name, (poof, ptest) in list(pool.items()):
        if base_name == "anchor":
            continue
        if not (base_name.startswith("logreg_stack") or base_name.startswith("simplex")):
            continue
        for alpha in [0.35, 0.50, 0.70]:
            for temp in [0.9, 1.0, 1.15]:
                name = f"{base_name}_anchlogit_a{str(alpha).replace('.', 'p')}_t{str(temp).replace('.', 'p')}"
                expanded_pool[name] = (
                    temperature_pred(poof, anchor_oof, alpha, temp),
                    temperature_pred(ptest, anchor_test, alpha, temp),
                )

    candidate_pool: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    choice_tables = []
    profiles = [
        ("target_select_public_tight", 0.004, 0.0002, 0.0025),
        ("target_select_public_balanced", 0.006, 0.0002, 0.0040),
        ("target_select_public_aggressive", 0.010, 0.0002, 0.0060),
    ]
    for name, full_guard, min_last_gain, risk_limit in profiles:
        poof, ptest, choices = select_targetwise(
            name,
            ytr,
            folds,
            anchor_oof,
            anchor_test,
            submitted_test,
            expanded_pool,
            full_guard,
            min_last_gain,
            risk_limit,
        )
        candidate_pool[name] = (poof, ptest)
        choice_tables.append(choices)

    for base in list(candidate_pool):
        poof, ptest = candidate_pool[base]
        for w in [0.65, 0.80, 0.90]:
            name = f"{base}_logit_anchorblend_w{str(w).replace('.', 'p')}"
            candidate_pool[name] = (
                blend_pred(poof, anchor_oof, w, "logit"),
                blend_pred(ptest, anchor_test, w, "logit"),
            )

    for name, pair in expanded_pool.items():
        if name.startswith("logreg_stack") or name.startswith("simplex"):
            candidate_pool[name] = pair

    log("Scoring candidates")
    rows = []
    for name, (poof, ptest) in candidate_pool.items():
        rows.append(candidate_score_row(name, ytr, folds, anchor_oof, anchor_test, submitted_test, poof, ptest))
    scores = pd.DataFrame(rows).sort_values(["selector_score", "last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    if choice_tables:
        pd.concat(choice_tables, ignore_index=True, sort=False).to_csv(out_dir / "target_selection_choices.csv", index=False)

    # Save a focused top set plus the selected one.
    saved = []
    for _, row in scores.head(12).iterrows():
        name = str(row["candidate"])
        poof, ptest = candidate_pool[name]
        safe = name.replace("/", "_").replace(":", "_").replace(".", "p")
        write_prediction(out_dir / f"{safe}_oof.csv", mtr, poof, ytr)
        write_prediction(out_dir / f"{safe}_test_pred.csv", mte, ptest)
        sub_path = sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv"
        write_submission(sub_path, mte, ptest)
        saved.append(str(sub_path.relative_to(ROOT)))

    best = scores.iloc[0].to_dict()
    (out_dir / "report.json").write_text(
        json.dumps({
            "submitted": args.submitted,
            "submitted_public": args.submitted_public,
            "best": best,
            "saved": saved,
        }, indent=2),
        encoding="utf-8",
    )

    print("\n=== Public-aware stack/blend candidates ===")
    cols = [
        "candidate",
        "selector_score",
        "full_logloss",
        "last_logloss",
        "full_delta_vs_anchor",
        "last_delta_vs_anchor",
        "tail3_worst",
        "test_abs_delta_mean_vs_anchor",
        "public_direction_risk",
        "bad_submission_alignment",
    ]
    print(scores[cols].head(30).to_string(index=False))
    print("\n=== Saved submissions ===")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
