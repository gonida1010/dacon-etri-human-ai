"""Diverse non-LGB/XGB/Cat single-model bank + stacking/blending.

This deliberately excludes the three tree boosters already explored
(LightGBM/XGBoost/CatBoost) and tests different model families:
- ElasticNet-style logistic regression
- HistGradientBoosting from sklearn
- ExtraTrees
- feature-distance KNN

The script also builds:
- target-wise guarded anchor blends for every single model
- fold-safe OOF stack over the diverse models
- logit/arithmetic submission blends with the public-confirmed KNN prior family
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_anchor_bank(bank_dir: Path, n_train: int, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_path = bank_dir / "oof_bank.csv"
    test_path = bank_dir / "test_bank.csv"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing OOF bank files under {bank_dir}")
    oof_bank = pd.read_csv(oof_path)
    test_bank = pd.read_csv(test_path)
    anchor_oof = pd.DataFrame({t: oof_bank[f"anchor__{t}"].values for t in TARGETS})
    anchor_test = pd.DataFrame({t: test_bank[f"anchor__{t}"].values for t in TARGETS})
    assert len(anchor_oof) == n_train and len(anchor_test) == n_test
    return anchor_oof, anchor_test


def add_anchor_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr2 = Xtr.copy()
    Xte2 = Xte.copy()
    for t in TARGETS:
        Xtr2[f"anchor_{t}"] = anchor_oof[t].values
        Xte2[f"anchor_{t}"] = anchor_test[t].values
        Xtr2[f"anchor_{t}_logit"] = logit(anchor_oof[t].values)
        Xte2[f"anchor_{t}_logit"] = logit(anchor_test[t].values)
    Xtr2["anchor_q_mean"] = anchor_oof[["Q1", "Q2", "Q3"]].mean(axis=1).values
    Xte2["anchor_q_mean"] = anchor_test[["Q1", "Q2", "Q3"]].mean(axis=1).values
    Xtr2["anchor_s_mean"] = anchor_oof[["S1", "S2", "S3", "S4"]].mean(axis=1).values
    Xte2["anchor_s_mean"] = anchor_test[["S1", "S2", "S3", "S4"]].mean(axis=1).values
    return Xtr2, Xte2


def numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in X.columns if c != "subject_id" and pd.api.types.is_numeric_dtype(X[c])]
    out = X[cols].replace([np.inf, -np.inf], np.nan).copy()
    return out


def select_top_corr(X: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, top_k: int, always: list[str]) -> list[str]:
    Xt = X.iloc[train_idx]
    yt = pd.Series(y[train_idx], index=Xt.index)
    scores = []
    y_std = float(yt.std())
    for c in X.columns:
        s = Xt[c]
        if c in always:
            score = np.inf
        elif s.notna().sum() < 12 or float(s.std(skipna=True) or 0) == 0 or y_std == 0:
            score = 0.0
        else:
            score = abs(float(s.corr(yt)))
            if not np.isfinite(score):
                score = 0.0
        scores.append((score, c))
    cols = [c for _, c in sorted(scores, reverse=True)[:top_k]]
    for c in always:
        if c in X.columns and c not in cols:
            cols.append(c)
    return list(dict.fromkeys(cols))


def make_model(kind: str, seed: int):
    if kind == "logreg_enet":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.15,
                C=0.08,
                max_iter=2500,
                random_state=seed,
            ),
        )
    if kind == "hist_gb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.025,
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=24,
                l2_regularization=2.5,
                early_stopping=True,
                random_state=seed,
            ),
        )
    if kind == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=5,
                min_samples_leaf=12,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if kind == "feature_knn":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=17, weights="distance", metric="minkowski", p=2),
        )
    raise ValueError(kind)


def fit_single_bank(
    kind: str,
    Xtr: pd.DataFrame,
    ytr: pd.DataFrame,
    Xte: pd.DataFrame,
    folds: np.ndarray,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.DataFrame(index=Xtr.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(0.0, index=Xte.index, columns=TARGETS, dtype=float)
    rows = []
    always = [c for c in Xtr.columns if c.startswith("anchor_")]
    for target in TARGETS:
        y = ytr[target].values.astype(int)
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            cols = select_top_corr(Xtr, y, tr_idx, top_k, always)
            model = make_model(kind, seed=9100 + 31 * fold + TARGETS.index(target))
            model.fit(Xtr.iloc[tr_idx][cols], y[tr_idx])
            p_val = clip(model.predict_proba(Xtr.iloc[va_idx][cols])[:, 1])
            p_test = clip(model.predict_proba(Xte[cols])[:, 1])
            oof.loc[va_idx, target] = p_val
            test[target] += p_test / C.N_SPLITS
            rows.append({
                "model": kind,
                "target": target,
                "fold": int(fold),
                "features": len(cols),
                "fold_logloss": safe_loss(y[va_idx], p_val),
            })
    return pd.DataFrame(clip(oof.values), columns=TARGETS), pd.DataFrame(clip(test.values), columns=TARGETS), pd.DataFrame(rows)


def targetwise_anchor_blend(
    name: str,
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    model_oof: pd.DataFrame,
    model_test: pd.DataFrame,
    weights: list[float],
    full_guard: float,
    min_last_gain: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last = folds == (C.N_SPLITS - 1)
    out_oof = pd.DataFrame(index=anchor_oof.index, columns=TARGETS, dtype=float)
    out_test = pd.DataFrame(index=anchor_test.index, columns=TARGETS, dtype=float)
    rows = []
    for target in TARGETS:
        base_full = safe_loss(ytr[target].values, anchor_oof[target].values)
        base_last = safe_loss(ytr[target].values[last], anchor_oof[target].values[last])
        best = {
            "target": target,
            "source": "anchor",
            "weight": 0.0,
            "full": base_full,
            "last": base_last,
            "oof": anchor_oof[target].values,
            "test": anchor_test[target].values,
        }
        for w in weights:
            po = clip((1 - w) * anchor_oof[target].values + w * model_oof[target].values)
            full = safe_loss(ytr[target].values, po)
            last_score = safe_loss(ytr[target].values[last], po[last])
            if full <= base_full + full_guard and last_score <= base_last - min_last_gain:
                if (last_score, full) < (best["last"], best["full"]):
                    best = {
                        "target": target,
                        "source": name,
                        "weight": float(w),
                        "full": full,
                        "last": last_score,
                        "oof": po,
                        "test": clip((1 - w) * anchor_test[target].values + w * model_test[target].values),
                    }
        out_oof[target] = best["oof"]
        out_test[target] = best["test"]
        rows.append({
            "candidate": f"{name}_anchblend",
            "target": target,
            "source": best["source"],
            "weight": best["weight"],
            "full_logloss": best["full"],
            "last_logloss": best["last"],
            "full_delta_vs_anchor": best["full"] - base_full,
            "last_delta_vs_anchor": best["last"] - base_last,
        })
    return out_oof, out_test, pd.DataFrame(rows)


def fold_safe_stack(
    ytr: pd.DataFrame,
    folds: np.ndarray,
    test_frames: dict[str, pd.DataFrame],
    oof_frames: dict[str, pd.DataFrame],
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    test = pd.DataFrame(0.0, index=next(iter(test_frames.values())).index, columns=TARGETS, dtype=float)
    rows = []
    names = list(oof_frames)
    for target in TARGETS:
        X = np.column_stack([logit(oof_frames[n][target].values) for n in names])
        Xte = np.column_stack([logit(test_frames[n][target].values) for n in names])
        y = ytr[target].values.astype(int)
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000),
            )
            model.fit(X[tr_idx], y[tr_idx])
            p_val = clip(model.predict_proba(X[va_idx])[:, 1])
            p_test = clip(model.predict_proba(Xte)[:, 1])
            oof.loc[va_idx, target] = p_val
            test[target] += p_test / C.N_SPLITS
            rows.append({
                "target": target,
                "fold": int(fold),
                "sources": json.dumps(names),
                "fold_logloss": safe_loss(y[va_idx], p_val),
            })
    return pd.DataFrame(clip(oof.values), columns=TARGETS), pd.DataFrame(clip(test.values), columns=TARGETS), pd.DataFrame(rows)


def blend_frames(a: pd.DataFrame, b: pd.DataFrame, w: float, mode: str) -> pd.DataFrame:
    if mode == "logit":
        vals = sigmoid(w * logit(a.values) + (1 - w) * logit(b.values))
    else:
        vals = w * a.values + (1 - w) * b.values
    return pd.DataFrame(clip(vals), columns=TARGETS)


def candidate_stability(ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray, name: str) -> dict:
    row = {"candidate": name}
    row["full_logloss"] = mean_loss(ytr, pred, np.ones(len(ytr), dtype=bool))
    vals = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        val = mean_loss(ytr, pred, mask)
        row[f"fold{fold}_logloss"] = val
        vals.append(val)
    row["last_logloss"] = vals[-1]
    row["tail3_mean"] = float(np.mean(vals[-3:]))
    row["tail3_worst"] = float(np.max(vals[-3:]))
    row["fold_std"] = float(np.std(vals))
    return row


def rank_score(row: dict, anchor_full: float) -> float:
    return (
        row["last_logloss"]
        + 1.5 * max(0.0, row["full_logloss"] - anchor_full)
        + 0.35 * max(0.0, row["tail3_worst"] - row["tail3_mean"])
    )


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test.copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-dir", default="research/oof_sparse_greedy")
    p.add_argument("--output-dir", default="research/diverse_single_stack")
    p.add_argument("--submission-dir", default="submissions/diverse_single_stack")
    p.add_argument("--models", nargs="*", default=["logreg_enet", "hist_gb", "extra_trees", "feature_knn"])
    p.add_argument("--top-k", type=int, default=140)
    p.add_argument("--full-guard", type=float, default=0.003)
    p.add_argument("--min-last-gain", type=float, default=0.0001)
    p.add_argument("--blend-weights", nargs="*", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    p.add_argument("--stack-c", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset and anchor bank")
    Xtr_raw, ytr, Xte_raw, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full = np.ones(len(ytr), dtype=bool)
    last = folds == (C.N_SPLITS - 1)
    anchor_oof, anchor_test = load_anchor_bank(ROOT / args.bank_dir, len(ytr), len(mte))

    Xtr_aug, Xte_aug = add_anchor_features(Xtr_raw, Xte_raw, anchor_oof, anchor_test)
    Xtr = numeric_frame(Xtr_aug)
    Xte = numeric_frame(Xte_aug)

    model_oofs: dict[str, pd.DataFrame] = {"anchor": anchor_oof}
    model_tests: dict[str, pd.DataFrame] = {"anchor": anchor_test}
    diagnostics = []
    candidate_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]] = {
        "anchor": (anchor_oof, anchor_test, "baseline anchor"),
    }

    for kind in args.models:
        log(f"Training diverse single model: {kind}")
        oof, test, diag = fit_single_bank(kind, Xtr, ytr, Xte, folds, args.top_k)
        model_oofs[kind] = oof
        model_tests[kind] = test
        diagnostics.append(diag)
        candidate_frames[kind] = (oof, test, "raw diverse single model")
        log(f"{kind} raw full={mean_loss(ytr, oof, full):.6f} last={mean_loss(ytr, oof, last):.6f}")

        blend_oof, blend_test, choices = targetwise_anchor_blend(
            kind,
            ytr,
            folds,
            anchor_oof,
            anchor_test,
            oof,
            test,
            args.blend_weights,
            args.full_guard,
            args.min_last_gain,
        )
        choices.to_csv(out_dir / f"{kind}_targetwise_blend.csv", index=False)
        candidate_frames[f"{kind}_anchblend"] = (blend_oof, blend_test, "target-wise guarded anchor blend")
        log(f"{kind}_anchblend full={mean_loss(ytr, blend_oof, full):.6f} last={mean_loss(ytr, blend_oof, last):.6f}")

    log("Rebuilding public-confirmed KNN guarded source")
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
        args.blend_weights,
        "knn_targetwise_guarded",
    )
    knn_choices.to_csv(out_dir / "knn_targetwise_choices.csv", index=False)
    candidate_frames["knn_targetwise_guarded"] = (knn_oof, knn_test, "public-confirmed KNN guarded family")

    log("Fold-safe stacking over diverse single models")
    stack_inputs = {k: v for k, v in model_oofs.items() if k in {"anchor", *args.models}}
    stack_tests = {k: model_tests[k] for k in stack_inputs}
    stack_oof, stack_test, stack_diag = fold_safe_stack(ytr, folds, stack_tests, stack_inputs, args.stack_c)
    stack_diag.to_csv(out_dir / "stack_diagnostics.csv", index=False)
    candidate_frames["diverse_oof_stack"] = (stack_oof, stack_test, "fold-safe stack over diverse single models")

    log("Submission-level blends with KNN guarded")
    for name, (po, pt, _) in list(candidate_frames.items()):
        if name in {"anchor", "knn_targetwise_guarded"}:
            continue
        for mode in ["arith", "logit"]:
            for w in [0.2, 0.35, 0.5]:
                cname = f"subblend_{mode}_{name}_{int(w*100)}_{int((1-w)*100)}knn"
                candidate_frames[cname] = (
                    blend_frames(po, knn_oof, w, mode),
                    blend_frames(pt, knn_test, w, mode),
                    f"submission-level {mode} blend with KNN guarded",
                )

    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_csv(out_dir / "single_model_fold_diagnostics.csv", index=False)

    anchor_full = mean_loss(ytr, anchor_oof, full)
    score_rows = []
    stability_rows = []
    for name, (po, pt, notes) in candidate_frames.items():
        row = candidate_stability(ytr, po, folds, name)
        row["notes"] = notes
        row["full_delta_vs_anchor"] = row["full_logloss"] - anchor_full
        row["last_delta_vs_anchor"] = row["last_logloss"] - mean_loss(ytr, anchor_oof, last)
        row["rank_score"] = rank_score(row, anchor_full)
        score_rows.append(row)
        stability_rows.append(row.copy())
        safe_name = name.replace("/", "_").replace(".", "p")
        write_submission(sub_dir / f"{safe_name}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv", mte, pt)

    scores = pd.DataFrame(score_rows).sort_values(["rank_score", "last_logloss", "full_logloss"])
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    pd.DataFrame(stability_rows).sort_values(["last_logloss", "full_logloss"]).to_csv(
        out_dir / "candidate_stability.csv", index=False
    )
    report = {
        "purpose": "Non-LGB/XGB/Cat diverse single models plus OOF stacking and submission blending.",
        "models": args.models,
        "top_k": args.top_k,
        "best_by_rank_score": scores.head(12).to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Diverse single/stack/blend candidates by rank_score ===")
    cols = ["candidate", "rank_score", "full_logloss", "last_logloss", "tail3_worst", "fold_std", "notes"]
    print(scores[cols].head(30).to_string(index=False))
    print("\n=== Raw single model diagnostics ===")
    raw = scores[scores["candidate"].isin(args.models)]
    print(raw[["candidate", "full_logloss", "last_logloss", "rank_score"]].to_string(index=False))
    print("\nNo public submission is automatically recommended; compare against public-confirmed KNN guarded first.")


if __name__ == "__main__":
    main()
