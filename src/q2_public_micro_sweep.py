"""Last-mile Q2 micro-sweep using recent public feedback.

This is intentionally narrower than the generic blend search.  The latest
public result improved after a guarded Q2-only change, while broader target
movement did not give reliable public feedback.  This script therefore keeps
all non-Q2 targets fixed at a reference submission and only sweeps Q2 between
nearby OOF/test sources.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .direction_gated_search import discover_pairs, load_anchor, read_pred
from .public_aware_stack_blend import public_direction_risk
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
EPS = C.PROB_CLIP


@dataclass(frozen=True)
class KnownPublic:
    path: Path
    score: float

    @property
    def name(self) -> str:
        return self.path.stem


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace(" ", "_")
        .replace("-", "m")
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def blend_values(base: np.ndarray, source: np.ndarray, weight: float, mode: str) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    source = np.asarray(source, dtype=float)
    if mode == "prob":
        return clip(base + weight * (source - base))
    if mode == "logit":
        return clip(sigmoid(logit(base) + weight * (logit(source) - logit(base))))
    raise ValueError(f"unknown blend mode: {mode}")


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def target_loss(y: pd.DataFrame, pred: pd.DataFrame, target: str, mask: np.ndarray) -> float:
    return safe_loss(y[target].values[mask], pred[target].values[mask])


def fold_losses(y: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> list[float]:
    return [mean_loss(y, pred, folds == f) for f in sorted(np.unique(folds))]


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


def parse_known_public(items: list[str]) -> list[KnownPublic]:
    out: list[KnownPublic] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"known public item must be path=score: {item}")
        path_s, score_s = item.rsplit("=", 1)
        out.append(KnownPublic(rel(path_s), float(score_s)))
    return out


def find_pair(pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name in pairs:
        return pairs[name]
    matches = [key for key in pairs if key.endswith(name) or key.split("/", 1)[-1] == name]
    if len(matches) != 1:
        raise ValueError(f"could not uniquely resolve pair {name!r}; matches={matches[:10]}")
    return pairs[matches[0]]


def test_features(pred: pd.DataFrame, anchor_test: pd.DataFrame, reference_test: pd.DataFrame | None = None) -> dict[str, float]:
    delta = pred[TARGETS].values - anchor_test[TARGETS].values
    feats: dict[str, float] = {
        "total_abs": float(np.mean(np.abs(delta))),
        "total_mean": float(np.mean(delta)),
    }
    if reference_test is not None:
        ref = reference_test[TARGETS].values
        feats["ref_abs"] = float(np.mean(np.abs(pred[TARGETS].values - ref)))
    else:
        feats["ref_abs"] = 0.0
    risk_sum = 0.0
    for target in TARGETS:
        d = pred[target].values - anchor_test[target].values
        feats[f"{target}_abs"] = float(np.mean(np.abs(d)))
        feats[f"{target}_mean"] = float(np.mean(d))
        feats[f"{target}_up_rate"] = float(np.mean(d > 1e-12))
        risk = public_direction_risk(target, d)
        feats[f"{target}_risk"] = float(risk)
        risk_sum += float(risk)
    feats["risk_mean"] = risk_sum / len(TARGETS)
    return feats


def fit_public_proxy(
    known: list[KnownPublic],
    anchor_test: pd.DataFrame,
    reference_test: pd.DataFrame,
    ridge: float,
) -> tuple[list[str], np.ndarray, dict[str, dict[str, float]], pd.DataFrame]:
    rows = []
    feature_map: dict[str, dict[str, float]] = {}
    for item in known:
        pred = read_pred(item.path)
        feats = test_features(pred, anchor_test, reference_test)
        feature_map[item.name] = feats
        rows.append({"constraint": item.name, "public_score": item.score, **feats})
    fit_df = pd.DataFrame(rows)
    # Keep this deliberately low-dimensional.  There are only a few public
    # observations, so high-dimensional target features overfit immediately.
    feature_cols = [
        "Q2_abs",
        "Q2_mean",
        "Q2_risk",
        "S2_abs",
        "S2_mean",
        "S4_abs",
        "S4_mean",
        "total_abs",
        "risk_mean",
        "ref_abs",
    ]
    x_raw = fit_df[feature_cols].values.astype(float)
    y = fit_df["public_score"].values.astype(float)
    mu = x_raw.mean(axis=0)
    sigma = x_raw.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    x = (x_raw - mu) / sigma
    x = np.column_stack([np.ones(len(x)), x])
    eye = np.eye(x.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge * eye, x.T @ y)
    fit_df["public_proxy_fit"] = x @ beta
    fit_df["public_proxy_residual"] = fit_df["public_proxy_fit"] - fit_df["public_score"]
    meta = {
        "_mu": {c: float(v) for c, v in zip(feature_cols, mu)},
        "_sigma": {c: float(v) for c, v in zip(feature_cols, sigma)},
    }
    feature_map["_meta"] = meta
    return feature_cols, beta, feature_map, fit_df


def predict_public_proxy(features: dict[str, float], feature_cols: list[str], beta: np.ndarray, meta: dict[str, dict[str, float]]) -> float:
    x = []
    for col in feature_cols:
        x.append((features[col] - meta["_mu"][col]) / meta["_sigma"][col])
    row = np.array([1.0, *x], dtype=float)
    return float(row @ beta)


def candidate_row(
    name: str,
    y: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    reference_test: pd.DataFrame,
    pred_oof: pd.DataFrame,
    pred_test: pd.DataFrame,
    feature_cols: list[str],
    beta: np.ndarray,
    proxy_meta: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    losses = fold_losses(y, pred_oof, folds)
    anchor_full = mean_loss(y, anchor_oof, full_mask)
    anchor_last = mean_loss(y, anchor_oof, last_mask)
    full = mean_loss(y, pred_oof, full_mask)
    last = mean_loss(y, pred_oof, last_mask)
    feats = test_features(pred_test, anchor_test, reference_test)
    q2_full = target_loss(y, pred_oof, "Q2", full_mask)
    q2_last = target_loss(y, pred_oof, "Q2", last_mask)
    proxy = predict_public_proxy(feats, feature_cols, beta, proxy_meta)
    ref_abs = float(np.mean(np.abs(pred_test[TARGETS].values - reference_test[TARGETS].values)))
    tail3_mean = float(np.mean(losses[-3:]))
    tail3_worst = float(max(losses[-3:]))
    # Public feedback is useful but sparse.  Penalize public-proxy winners that
    # move far away from the actual best public submission or damage OOF.
    decision_score = (
        proxy
        + 0.35 * max(0.0, full - anchor_full)
        + 0.12 * max(0.0, last - anchor_last)
        + 0.18 * ref_abs
        + 0.05 * max(0.0, tail3_worst - tail3_mean)
    )
    return {
        "candidate": name,
        "public_proxy": proxy,
        "decision_score": decision_score,
        "full_logloss": full,
        "last_logloss": last,
        "q2_full_logloss": q2_full,
        "q2_last_logloss": q2_last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "fold_std": float(np.std(losses)),
        "tail3_mean": tail3_mean,
        "tail3_worst": tail3_worst,
        "ref_abs_delta_mean": ref_abs,
        **{f"fold{i}_logloss": v for i, v in enumerate(losses)},
        **{f"feat_{k}": v for k, v in feats.items()},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-dirs",
        nargs="*",
        default=[
            "research/guarded_lgbm_integration_20260623_v2",
            "research/guarded_lgbm_kaggle_base_20260623",
            "research/public_aware_stack_blend_20260622",
            "research/public_aware_stack_blend_with_lgbm_source_20260622",
        ],
    )
    p.add_argument(
        "--reference-name",
        default=(
            "guarded_lgbm_integration_20260623_v2/"
            "public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5"
        ),
    )
    p.add_argument("--known-public", nargs="*", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=[-0.20, -0.10, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.15])
    p.add_argument("--modes", nargs="*", default=["logit", "prob"])
    p.add_argument("--public-ridge", type=float, default=2.0)
    p.add_argument("--max-ref-delta", type=float, default=0.0040)
    p.add_argument("--max-q2-abs", type=float, default=0.0140)
    p.add_argument("--save-top", type=int, default=12)
    p.add_argument("--output-dir", default="research/q2_public_micro_sweep_20260623")
    p.add_argument("--submission-dir", default="submissions/q2_public_micro_sweep_20260623")
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

    log("Loading prediction pairs")
    pairs = discover_pairs(args.source_dirs)
    if not pairs:
        raise RuntimeError("no OOF/test prediction pairs found")
    ref_oof, ref_test = find_pair(pairs, args.reference_name)

    known = parse_known_public(args.known_public)
    feature_cols, beta, proxy_state, fit_df = fit_public_proxy(known, anchor_test, ref_test, args.public_ridge)
    fit_df.to_csv(out_dir / "public_proxy_fit.csv", index=False)

    log(f"Generating Q2 micro blends from {len(pairs)} sources")
    rows = []
    candidate_store: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    base_row = candidate_row(
        "reference_public_best",
        ytr,
        folds,
        anchor_oof,
        anchor_test,
        ref_test,
        ref_oof,
        ref_test,
        feature_cols,
        beta,
        proxy_state["_meta"],
    )
    rows.append(base_row)
    candidate_store["reference_public_best"] = (ref_oof, ref_test)

    for source_name, (src_oof, src_test) in pairs.items():
        if source_name == args.reference_name:
            continue
        source_q2_abs = float(np.mean(np.abs(src_test["Q2"].values - anchor_test["Q2"].values)))
        if source_q2_abs > args.max_q2_abs * 1.35:
            continue
        for mode in args.modes:
            for weight in args.weights:
                poof = ref_oof.copy()
                ptest = ref_test.copy()
                poof["Q2"] = blend_values(ref_oof["Q2"].values, src_oof["Q2"].values, weight, mode)
                ptest["Q2"] = blend_values(ref_test["Q2"].values, src_test["Q2"].values, weight, mode)
                feats = test_features(ptest, anchor_test, ref_test)
                if feats["ref_abs"] > args.max_ref_delta:
                    continue
                if feats["Q2_abs"] > args.max_q2_abs:
                    continue
                name = f"q2micro__{safe_name(source_name)}__{mode}__w{str(weight).replace('-', 'm').replace('.', 'p')}"
                row = candidate_row(
                    name,
                    ytr,
                    folds,
                    anchor_oof,
                    anchor_test,
                    ref_test,
                    poof,
                    ptest,
                    feature_cols,
                    beta,
                    proxy_state["_meta"],
                )
                row["source"] = source_name
                row["mode"] = mode
                row["weight"] = weight
                row["source_q2_abs"] = source_q2_abs
                rows.append(row)
                candidate_store[name] = (poof, ptest)

    scores = pd.DataFrame(rows)
    scores = scores.sort_values(["decision_score", "public_proxy", "last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)

    saved_rows = pd.concat(
        [
            scores.head(args.save_top),
            scores.sort_values(["last_logloss", "full_logloss"]).head(args.save_top),
            scores.sort_values(["full_logloss", "last_logloss"]).head(args.save_top),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("candidate")

    saved = []
    for _, row in saved_rows.iterrows():
        name = str(row["candidate"])
        poof, ptest = candidate_store[name]
        safe = safe_name(name)
        write_prediction(out_dir / f"{safe}_oof.csv", mtr, poof, ytr)
        write_prediction(out_dir / f"{safe}_test_pred.csv", mte, ptest)
        sub_path = sub_dir / (
            f"{safe}_proxy{row['public_proxy']:.6f}_last{row['last_logloss']:.6f}"
            f"_full{row['full_logloss']:.6f}.csv"
        )
        write_submission(sub_path, mte, ptest)
        saved.append(str(sub_path.relative_to(ROOT)))

    report = {
        "reference_name": args.reference_name,
        "source_dirs": args.source_dirs,
        "known_public": [{"path": str(item.path.relative_to(ROOT)), "score": item.score} for item in known],
        "proxy_features": feature_cols,
        "saved": saved,
        "best": scores.iloc[0].to_dict() if not scores.empty else {},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Q2 public micro-sweep candidates ===")
    cols = [
        "candidate",
        "decision_score",
        "public_proxy",
        "full_logloss",
        "last_logloss",
        "q2_full_logloss",
        "q2_last_logloss",
        "ref_abs_delta_mean",
        "feat_Q2_abs",
        "feat_Q2_mean",
        "feat_Q2_risk",
    ]
    print(scores[cols].head(40).to_string(index=False))
    print("\n=== Public proxy fit ===")
    print(fit_df[["constraint", "public_score", "public_proxy_fit", "public_proxy_residual"]].to_string(index=False))
    print("\n=== Saved submissions ===")
    for item in saved:
        print(item)


if __name__ == "__main__":
    main()
