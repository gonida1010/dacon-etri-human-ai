"""Fast last-mile Kaggle-style algorithms over the OOF prediction bank.

Adds the methods that were still missing from the handoff:
- fold-safe OOF meta stackers over prediction-bank columns
- ExtraTrees/RandomForest meta models over the same bank
- same-subject date-nearest KNN priors
- Q-target rank-level patches instead of hard mean shifts
- fold/tail stability diagnostics for deciding submission risk

Run after src.oof_sparse_greedy has produced research/oof_sparse_greedy/*.csv.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .cv import subject_time_blocked_folds
from .train_temporal_prior import clip

TARGETS = C.TARGET_COLS
Q_TARGETS = ["Q1", "Q2", "Q3"]
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


def source_relevant(source: str, target: str) -> bool:
    if not source.startswith("seq_"):
        return True
    return source.startswith(f"seq_{target}_")


def source_names(bank: pd.DataFrame, target: str) -> list[str]:
    suffix = f"__{target}"
    names = []
    for col in bank.columns:
        if col.endswith(suffix) and not col.startswith("label__"):
            src = col[: -len(suffix)]
            if source_relevant(src, target):
                names.append(src)
    return names


def source_values(frame: pd.DataFrame, source: str, target: str) -> np.ndarray:
    return clip(frame[f"{source}__{target}"].values)


def labels_from_bank(bank: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({target: bank[f"label__{target}"].astype(int).values for target in TARGETS})


def meta_from_bank(bank: pd.DataFrame) -> pd.DataFrame:
    out = bank[["subject_id", "sleep_date", "lifelog_date"]].copy()
    out["sleep_date"] = pd.to_datetime(out["sleep_date"])
    out["lifelog_date"] = pd.to_datetime(out["lifelog_date"])
    return out


def empty_pred(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=TARGETS, dtype=float)


def reconstruct_best_single(bank: pd.DataFrame, test: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = empty_pred(bank.index)
    out_test = empty_pred(test.index)
    for _, row in table.iterrows():
        target = str(row["target"])
        source = str(row["source"])
        oof[target] = source_values(bank, source, target)
        out_test[target] = source_values(test, source, target)
    return oof, out_test


def reconstruct_sparse_greedy(bank: pd.DataFrame, test: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = empty_pred(bank.index)
    out_test = empty_pred(test.index)
    for target in TARGETS:
        current_oof = source_values(bank, "anchor", target)
        current_test = source_values(test, "anchor", target)
        target_steps = steps[(steps["target"].eq(target)) & (steps["step"].astype(int) > 0)].sort_values("step")
        for _, row in target_steps.iterrows():
            source = str(row["source"])
            w = float(row["weight_new_source"])
            current_oof = clip((1.0 - w) * current_oof + w * source_values(bank, source, target))
            current_test = clip((1.0 - w) * current_test + w * source_values(test, source, target))
        oof[target] = current_oof
        out_test[target] = current_test
    return oof, out_test


def select_sources_on_train(
    bank: pd.DataFrame,
    y: np.ndarray,
    target: str,
    train_idx: np.ndarray,
    top_n: int,
) -> list[str]:
    scored = []
    for source in source_names(bank, target):
        pred = source_values(bank, source, target)
        scored.append((safe_loss(y[train_idx], pred[train_idx]), source))
    chosen = ["anchor"]
    for _, source in sorted(scored, key=lambda x: x[0]):
        if source not in chosen:
            chosen.append(source)
        if len(chosen) >= top_n:
            break
    return chosen


def feature_matrix(frame: pd.DataFrame, sources: list[str], target: str, mode: str) -> np.ndarray:
    vals = [source_values(frame, source, target) for source in sources]
    X = np.column_stack(vals)
    if mode == "logit":
        return logit(X)
    if mode == "both":
        return np.column_stack([X, logit(X)])
    return X


def fit_meta_candidate(
    name: str,
    model_kind: str,
    bank: pd.DataFrame,
    test: pd.DataFrame,
    ytr: pd.DataFrame,
    folds: np.ndarray,
    top_sources: int,
    c_value: float,
    tree_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    oof = empty_pred(bank.index)
    out_test = empty_pred(test.index)
    rows = []
    for target in TARGETS:
        y = ytr[target].values.astype(int)
        test_acc = np.zeros(len(test), dtype=float)
        fold_count = 0
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            sources = select_sources_on_train(bank, y, target, tr_idx, top_sources)
            if model_kind.startswith("logreg"):
                mode = "logit"
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000),
                )
            elif model_kind == "extratrees":
                mode = "both"
                model = ExtraTreesClassifier(
                    n_estimators=tree_estimators,
                    max_depth=4,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    random_state=1000 + fold + TARGETS.index(target),
                    n_jobs=-1,
                )
            elif model_kind == "rf":
                mode = "both"
                model = RandomForestClassifier(
                    n_estimators=tree_estimators,
                    max_depth=4,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    random_state=2000 + fold + TARGETS.index(target),
                    n_jobs=-1,
                )
            else:
                raise ValueError(model_kind)
            Xtr = feature_matrix(bank.iloc[tr_idx], sources, target, mode)
            Xva = feature_matrix(bank.iloc[va_idx], sources, target, mode)
            Xte = feature_matrix(test, sources, target, mode)
            model.fit(Xtr, y[tr_idx])
            p_va = clip(model.predict_proba(Xva)[:, 1])
            p_te = clip(model.predict_proba(Xte)[:, 1])
            oof.loc[va_idx, target] = p_va
            test_acc += p_te
            fold_count += 1
            rows.append({
                "candidate": name,
                "target": target,
                "fold": int(fold),
                "sources": json.dumps(sources),
                "fold_logloss": safe_loss(y[va_idx], p_va),
            })
        out_test[target] = clip(test_acc / max(fold_count, 1))
    return oof, out_test, rows


def knn_predict_one(
    y: pd.Series,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    subject: str,
    date: pd.Timestamp,
    k: int,
    scale_days: float,
    smooth: float,
) -> float:
    idx = train_idx[meta.iloc[train_idx]["subject_id"].astype(str).values == subject]
    if len(idx) == 0:
        return float(y.iloc[train_idx].mean())
    dates = pd.to_datetime(meta.iloc[idx]["sleep_date"])
    dist = np.abs((dates - date).dt.days.values.astype(float))
    order = np.argsort(dist)[: min(k, len(idx))]
    chosen = idx[order]
    weights = np.exp(-dist[order] / scale_days)
    vals = y.iloc[chosen].values.astype(float)
    gmean = float(y.iloc[train_idx].mean())
    return float((weights @ vals + smooth * gmean) / (weights.sum() + smooth))


def build_knn_source(
    ytr: pd.DataFrame,
    meta: pd.DataFrame,
    meta_test: pd.DataFrame,
    folds: np.ndarray,
    k: int,
    scale_days: float,
    smooth: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = empty_pred(meta.index)
    out_test = empty_pred(meta_test.index)
    all_idx = np.arange(len(meta))
    for target in TARGETS:
        y = ytr[target]
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            va_idx = np.where(folds == fold)[0]
            vals = []
            for i in va_idx:
                vals.append(knn_predict_one(y, meta, tr_idx, str(meta.at[i, "subject_id"]), pd.Timestamp(meta.at[i, "sleep_date"]), k, scale_days, smooth))
            oof.loc[va_idx, target] = vals
        vals = []
        for i, row in meta_test.iterrows():
            vals.append(knn_predict_one(y, meta, all_idx, str(row["subject_id"]), pd.Timestamp(row["sleep_date"]), k, scale_days, smooth))
        out_test[target] = vals
    return pd.DataFrame(clip(oof.values), columns=TARGETS), pd.DataFrame(clip(out_test.values), columns=TARGETS)


def targetwise_guarded_blend(
    ytr: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    full_guard: float,
    min_last_gain: float,
    weights: list[float],
    candidate_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    last = folds == (C.N_SPLITS - 1)
    out_oof = empty_pred(anchor_oof.index)
    out_test = empty_pred(anchor_test.index)
    rows = []
    for target in TARGETS:
        base_full = safe_loss(ytr[target].values, anchor_oof[target].values)
        base_last = safe_loss(ytr[target].values[last], anchor_oof[target].values[last])
        best = {
            "source": "anchor",
            "weight": 0.0,
            "full": base_full,
            "last": base_last,
            "oof": anchor_oof[target].values,
            "test": anchor_test[target].values,
        }
        for source, (src_oof, src_test) in sources.items():
            for w in weights:
                po = clip((1 - w) * anchor_oof[target].values + w * src_oof[target].values)
                full = safe_loss(ytr[target].values, po)
                last_score = safe_loss(ytr[target].values[last], po[last])
                if full <= base_full + full_guard and last_score <= base_last - min_last_gain:
                    if (last_score, full) < (best["last"], best["full"]):
                        best = {
                            "source": source,
                            "weight": float(w),
                            "full": full,
                            "last": last_score,
                            "oof": po,
                            "test": clip((1 - w) * anchor_test[target].values + w * src_test[target].values),
                        }
        out_oof[target] = best["oof"]
        out_test[target] = best["test"]
        rows.append({
            "candidate": candidate_name,
            "target": target,
            "source": best["source"],
            "weight": best["weight"],
            "full_logloss": best["full"],
            "last_logloss": best["last"],
            "full_delta_vs_anchor": best["full"] - base_full,
            "last_delta_vs_anchor": best["last"] - base_last,
        })
    return out_oof, out_test, pd.DataFrame(rows)


def rank_patch_group(
    probs: np.ndarray,
    drift: float,
    frac: float,
    delta: float,
    threshold: float,
) -> np.ndarray:
    out = clip(probs).copy()
    if len(out) == 0 or abs(drift) < threshold:
        return out
    n = max(1, int(np.ceil(len(out) * frac)))
    strength = delta * min(abs(drift) / 0.3, 1.0)
    z = logit(out)
    if drift > 0:
        idx = np.argsort(out)[-n:]
        z[idx] += strength
    else:
        idx = np.argsort(out)[:n]
        z[idx] -= strength
    return clip(sigmoid(z))


def history_drift(
    y: pd.Series,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    subject: str,
    start_date: pd.Timestamp,
    recent_k: int,
) -> float:
    hist_idx = train_idx[
        (meta.iloc[train_idx]["subject_id"].astype(str).values == subject)
        & (pd.to_datetime(meta.iloc[train_idx]["sleep_date"]).values < np.datetime64(start_date))
    ]
    if len(hist_idx) < 4:
        hist_idx = train_idx[meta.iloc[train_idx]["subject_id"].astype(str).values == subject]
    if len(hist_idx) < 4:
        return 0.0
    hist = meta.iloc[hist_idx].copy()
    hist["y"] = y.iloc[hist_idx].values.astype(float)
    hist = hist.sort_values("sleep_date")
    recent = float(hist["y"].tail(recent_k).mean())
    base = float(hist["y"].mean())
    return recent - base


def apply_q_rank_patch(
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    ytr: pd.DataFrame,
    meta: pd.DataFrame,
    meta_test: pd.DataFrame,
    folds: np.ndarray,
    recent_k: int,
    frac: float,
    delta: float,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = base_oof.copy()
    out_test = base_test.copy()
    rows = []
    subjects = sorted(meta["subject_id"].astype(str).unique())
    all_idx = np.arange(len(meta))
    for target in Q_TARGETS:
        y = ytr[target]
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            for subject in subjects:
                mask = (folds == fold) & (meta["subject_id"].astype(str).values == subject)
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                start_date = pd.Timestamp(meta.loc[idx, "sleep_date"].min())
                drift = history_drift(y, meta, tr_idx, subject, start_date, recent_k)
                before = out_oof.loc[idx, target].values
                after = rank_patch_group(before, drift, frac, delta, threshold)
                out_oof.loc[idx, target] = after
                rows.append({
                    "split": "oof",
                    "target": target,
                    "fold": int(fold),
                    "subject_id": subject,
                    "rows": len(idx),
                    "drift": drift,
                    "before_mean": float(np.mean(before)),
                    "after_mean": float(np.mean(after)),
                })
        for subject in subjects:
            idx = np.where(meta_test["subject_id"].astype(str).values == subject)[0]
            if len(idx) == 0:
                continue
            start_date = pd.Timestamp(meta_test.loc[idx, "sleep_date"].min())
            drift = history_drift(y, meta, all_idx, subject, start_date, recent_k)
            before = out_test.loc[idx, target].values
            after = rank_patch_group(before, drift, frac, delta, threshold)
            out_test.loc[idx, target] = after
            rows.append({
                "split": "test",
                "target": target,
                "fold": -1,
                "subject_id": subject,
                "rows": len(idx),
                "drift": drift,
                "before_mean": float(np.mean(before)),
                "after_mean": float(np.mean(after)),
            })
    return out_oof, out_test, pd.DataFrame(rows)


def candidate_stability(ytr: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray, name: str) -> dict:
    row = {"candidate": name}
    full_mask = np.ones(len(ytr), dtype=bool)
    row["full_logloss"] = mean_loss(ytr, pred, full_mask)
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        row[f"fold{fold}_logloss"] = mean_loss(ytr, pred, mask)
    tail_cols = [f"fold{fold}_logloss" for fold in sorted(np.unique(folds))[-3:]]
    row["tail3_mean"] = float(np.mean([row[c] for c in tail_cols]))
    row["tail3_worst"] = float(np.max([row[c] for c in tail_cols]))
    row["last_logloss"] = row[f"fold{int(np.max(folds))}_logloss"]
    return row


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank-dir", default="research/oof_sparse_greedy")
    p.add_argument("--output-dir", default="research/kaggle_last_mile")
    p.add_argument("--submission-dir", default="submissions/kaggle_last_mile")
    p.add_argument("--full-guard", type=float, default=0.006)
    p.add_argument("--min-last-gain", type=float, default=0.0002)
    p.add_argument("--top-sources", type=int, default=18)
    p.add_argument("--logreg-c", nargs="*", type=float, default=[0.03, 0.1, 0.3])
    p.add_argument("--tree-estimators", type=int, default=240)
    p.add_argument("--blend-weights", nargs="*", type=float, default=[0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    p.add_argument("--knn-k", nargs="*", type=int, default=[3, 5, 8])
    p.add_argument("--knn-scales", nargs="*", type=float, default=[7.0, 14.0, 30.0])
    p.add_argument("--rank-fracs", nargs="*", type=float, default=[0.10, 0.20])
    p.add_argument("--rank-deltas", nargs="*", type=float, default=[0.35, 0.70, 1.05])
    p.add_argument("--rank-recent-k", nargs="*", type=int, default=[3, 5, 8])
    p.add_argument("--rank-threshold", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bank_dir = ROOT / args.bank_dir
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading OOF/test bank")
    bank = pd.read_csv(bank_dir / "oof_bank.csv")
    test = pd.read_csv(bank_dir / "test_bank.csv")
    best_single_table = pd.read_csv(bank_dir / "targetwise_best_single.csv")
    greedy_steps = pd.read_csv(bank_dir / "targetwise_greedy_steps.csv")

    meta = meta_from_bank(bank)
    meta_test = meta_from_bank(test)
    ytr = labels_from_bank(bank)
    folds = subject_time_blocked_folds(meta, n_splits=C.N_SPLITS)
    full = np.ones(len(bank), dtype=bool)
    last = folds == (C.N_SPLITS - 1)

    anchor_oof = pd.DataFrame({target: source_values(bank, "anchor", target) for target in TARGETS})
    anchor_test = pd.DataFrame({target: source_values(test, "anchor", target) for target in TARGETS})
    best_single_oof, best_single_test = reconstruct_best_single(bank, test, best_single_table)
    sparse_oof, sparse_test = reconstruct_sparse_greedy(bank, test, greedy_steps)

    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]] = {
        "anchor": (anchor_oof, anchor_test, "existing anchor"),
        "oof_sparse_best_single": (best_single_oof, best_single_test, "existing best single guarded"),
        "oof_sparse_greedy": (sparse_oof, sparse_test, "existing sparse greedy"),
    }
    diagnostics = []

    log("Running fold-safe logistic meta-stackers")
    for c_value in args.logreg_c:
        name = f"meta_logreg_c{str(c_value).replace('.', 'p')}"
        oof, pred_test, rows = fit_meta_candidate(
            name, "logreg", bank, test, ytr, folds, args.top_sources, c_value, args.tree_estimators
        )
        candidates[name] = (oof, pred_test, f"logistic meta stacker C={c_value}")
        diagnostics.extend(rows)
        log(f"{name} full={mean_loss(ytr, oof, full):.6f} last={mean_loss(ytr, oof, last):.6f}")

    log("Running ExtraTrees/RandomForest meta models")
    for kind in ["extratrees", "rf"]:
        name = f"meta_{kind}"
        oof, pred_test, rows = fit_meta_candidate(
            name, kind, bank, test, ytr, folds, args.top_sources, 0.1, args.tree_estimators
        )
        candidates[name] = (oof, pred_test, f"{kind} meta model over bank columns")
        diagnostics.extend(rows)
        log(f"{name} full={mean_loss(ytr, oof, full):.6f} last={mean_loss(ytr, oof, last):.6f}")

    log("Building KNN-like subject/date priors")
    knn_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for k in args.knn_k:
        for scale in args.knn_scales:
            name = f"knn_k{k}_s{str(scale).replace('.', 'p')}"
            oof, pred_test = build_knn_source(ytr, meta, meta_test, folds, k=k, scale_days=scale, smooth=4.0)
            knn_sources[name] = (oof, pred_test)
            log(f"{name} full={mean_loss(ytr, oof, full):.6f} last={mean_loss(ytr, oof, last):.6f}")
    knn_oof, knn_test, knn_rows = targetwise_guarded_blend(
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
    knn_rows.to_csv(out_dir / "knn_targetwise_choices.csv", index=False)
    candidates["knn_targetwise_guarded"] = (knn_oof, knn_test, "target-wise guarded blend of KNN priors")
    log(f"knn_targetwise_guarded full={mean_loss(ytr, knn_oof, full):.6f} last={mean_loss(ytr, knn_oof, last):.6f}")

    log("Searching Q rank-level patches")
    rank_rows = []
    patch_bases = {
        "anchor": (anchor_oof, anchor_test),
        "best_single": (best_single_oof, best_single_test),
        "sparse_greedy": (sparse_oof, sparse_test),
    }
    for base_name, (base_oof, base_test) in patch_bases.items():
        base_full = mean_loss(ytr, base_oof, full)
        best_row = None
        best_pack = None
        for recent_k in args.rank_recent_k:
            for frac in args.rank_fracs:
                for delta in args.rank_deltas:
                    po, pt, detail = apply_q_rank_patch(
                        base_oof,
                        base_test,
                        ytr,
                        meta,
                        meta_test,
                        folds,
                        recent_k=recent_k,
                        frac=frac,
                        delta=delta,
                        threshold=args.rank_threshold,
                    )
                    full_score = mean_loss(ytr, po, full)
                    last_score = mean_loss(ytr, po, last)
                    row = {
                        "base": base_name,
                        "recent_k": recent_k,
                        "frac": frac,
                        "delta": delta,
                        "full_logloss": full_score,
                        "last_logloss": last_score,
                        "full_delta_vs_base": full_score - base_full,
                    }
                    rank_rows.append(row)
                    if full_score <= base_full + args.full_guard:
                        if best_row is None or (last_score, full_score) < (best_row["last_logloss"], best_row["full_logloss"]):
                            best_row = row
                            best_pack = (po, pt, detail)
        if best_row is not None and best_pack is not None:
            cname = (
                f"rankpatch_{base_name}_k{best_row['recent_k']}"
                f"_f{str(best_row['frac']).replace('.', 'p')}"
                f"_d{str(best_row['delta']).replace('.', 'p')}"
            )
            candidates[cname] = (best_pack[0], best_pack[1], f"Q rank patch over {base_name}")
            best_pack[2].to_csv(out_dir / f"{cname}_detail.csv", index=False)
            log(f"{cname} full={best_row['full_logloss']:.6f} last={best_row['last_logloss']:.6f}")
    pd.DataFrame(rank_rows).sort_values(["base", "last_logloss", "full_logloss"]).to_csv(out_dir / "rank_patch_search.csv", index=False)

    log("Scoring candidates and writing submissions")
    score_rows = []
    stability_rows = []
    for name, (po, pt, notes) in candidates.items():
        row = {
            "candidate": name,
            "full_logloss": mean_loss(ytr, po, full),
            "last_logloss": mean_loss(ytr, po, last),
            "notes": notes,
        }
        row["full_delta_vs_anchor"] = row["full_logloss"] - mean_loss(ytr, anchor_oof, full)
        row["last_delta_vs_anchor"] = row["last_logloss"] - mean_loss(ytr, anchor_oof, last)
        score_rows.append(row)
        stability_rows.append(candidate_stability(ytr, po, folds, name))
        safe_name = name.replace(".", "p").replace("/", "_")
        write_submission(sub_dir / f"{safe_name}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv", meta_test, pt)

    scores = pd.DataFrame(score_rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    stability = pd.DataFrame(stability_rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    stability.to_csv(out_dir / "candidate_stability.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "meta_diagnostics.csv", index=False)
    report = {
        "purpose": "Last-mile Kaggle-style algorithms over existing OOF prediction bank.",
        "top_sources": args.top_sources,
        "full_guard": args.full_guard,
        "min_last_gain": args.min_last_gain,
        "best_candidates": scores.head(10).to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Last-mile candidate scores ===")
    print(scores.head(20).to_string(index=False))
    print("\n=== Candidate stability ===")
    print(stability.head(20).to_string(index=False))
    print("\n=== Suggested submit shortlist ===")
    shortlist = scores.head(6).copy()
    for _, row in shortlist.iterrows():
        print(f"{row['candidate']}: last={row['last_logloss']:.6f} full={row['full_logloss']:.6f}")


if __name__ == "__main__":
    main()
