"""Subject-wise sequence smoothing over temporal-anchor probabilities.

This tests whether each target benefits from a simple two-state Markov prior
over consecutive sleep dates. It is batch-test compatible: labels are not used
inside the validation/test block, only feature-based probabilities and the last
known train label before the block.

Run:
  python -m src.sequence_smoothing_search
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
from .train_temporal_prior import build_recipe_predictions, clip, fit_lgbm_oof_test

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT
EPS = C.PROB_CLIP


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def transition_matrix(y: pd.Series, meta: pd.DataFrame, idx: np.ndarray, smooth: float) -> np.ndarray:
    counts = np.full((2, 2), smooth, dtype=float)
    df = meta.iloc[idx].copy()
    df["y"] = y.iloc[idx].astype(int).values
    for _, g in df.sort_values(["subject_id", "sleep_date"]).groupby("subject_id"):
        vals = g["y"].to_numpy(int)
        if len(vals) < 2:
            continue
        for a, b in zip(vals[:-1], vals[1:]):
            counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def initial_prob(y: pd.Series, meta: pd.DataFrame, idx: np.ndarray, subject: str, start_date: pd.Timestamp) -> np.ndarray:
    hist = meta.iloc[idx].copy()
    hist["y"] = y.iloc[idx].astype(int).values
    hist = hist[(hist["subject_id"].astype(str) == subject) & (hist["sleep_date"] < start_date)]
    if len(hist):
        last_y = int(hist.sort_values("sleep_date")["y"].iloc[-1])
        out = np.array([0.04, 0.04], dtype=float)
        out[last_y] = 0.96
        return out
    p = float(y.iloc[idx].mean()) if len(idx) else float(y.mean())
    return np.array([1 - p, p], dtype=float)


def smooth_sequence(
    probs: np.ndarray,
    trans: np.ndarray,
    init: np.ndarray,
    emission_power: float,
    transition_blend: float,
) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    uniform_trans = np.full((2, 2), 0.5, dtype=float)
    trans = transition_blend * trans + (1 - transition_blend) * uniform_trans
    trans = trans / trans.sum(axis=1, keepdims=True)

    emit = np.column_stack([(1 - probs) ** emission_power, probs ** emission_power])
    emit = np.clip(emit, EPS, None)
    emit = emit / emit.sum(axis=1, keepdims=True)
    n = len(probs)
    if n == 0:
        return probs

    alpha = np.zeros((n, 2), dtype=float)
    scale = np.zeros(n, dtype=float)
    alpha[0] = init * emit[0]
    scale[0] = alpha[0].sum()
    alpha[0] /= max(scale[0], EPS)
    for i in range(1, n):
        alpha[i] = alpha[i - 1] @ trans * emit[i]
        scale[i] = alpha[i].sum()
        alpha[i] /= max(scale[i], EPS)

    beta = np.ones((n, 2), dtype=float)
    for i in range(n - 2, -1, -1):
        beta[i] = trans @ (emit[i + 1] * beta[i + 1])
        beta[i] /= max(beta[i].sum(), EPS)

    post = alpha * beta
    post = post / post.sum(axis=1, keepdims=True)
    return clip(post[:, 1])


def apply_smoothing(
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    target_params: dict[str, tuple[float, float]],
    trans_smooth: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = base_oof.copy()
    out_test = base_test.copy()
    rows: list[dict] = []
    subjects = sorted(mtr["subject_id"].astype(str).unique())

    for target, (emission_power, transition_blend) in target_params.items():
        for fold in sorted(np.unique(folds)):
            tr_idx = np.where(folds != fold)[0]
            trans = transition_matrix(ytr[target], mtr, tr_idx, smooth=trans_smooth)
            for subject in subjects:
                va_mask = (folds == fold) & (mtr["subject_id"].astype(str).values == subject)
                if not va_mask.any():
                    continue
                order = mtr.loc[va_mask].sort_values("sleep_date").index.to_numpy()
                start_date = pd.Timestamp(mtr.loc[order, "sleep_date"].min())
                init = initial_prob(ytr[target], mtr, tr_idx, subject, start_date)
                before = base_oof.loc[order, target].values
                after = smooth_sequence(before, trans, init, emission_power, transition_blend)
                out_oof.loc[order, target] = after
                rows.append({
                    "split": "oof",
                    "target": target,
                    "fold": int(fold),
                    "subject_id": subject,
                    "rows": len(order),
                    "emission_power": emission_power,
                    "transition_blend": transition_blend,
                    "p00": trans[0, 0],
                    "p11": trans[1, 1],
                    "before_mean": float(np.mean(before)),
                    "after_mean": float(np.mean(after)),
                })

        trans = transition_matrix(ytr[target], mtr, np.arange(len(mtr)), smooth=trans_smooth)
        for subject in subjects:
            te_mask = mte["subject_id"].astype(str).values == subject
            if not te_mask.any():
                continue
            order = mte.loc[te_mask].sort_values("sleep_date").index.to_numpy()
            start_date = pd.Timestamp(mte.loc[order, "sleep_date"].min())
            init = initial_prob(ytr[target], mtr, np.arange(len(mtr)), subject, start_date)
            before = base_test.loc[order, target].values
            after = smooth_sequence(before, trans, init, emission_power, transition_blend)
            out_test.loc[order, target] = after
            rows.append({
                "split": "test",
                "target": target,
                "fold": -1,
                "subject_id": subject,
                "rows": len(order),
                "emission_power": emission_power,
                "transition_blend": transition_blend,
                "p00": trans[0, 0],
                "p11": trans[1, 1],
                "before_mean": float(np.mean(before)),
                "after_mean": float(np.mean(after)),
            })

    return out_oof, out_test, pd.DataFrame(rows)


def write_submission(path: Path, meta_test: pd.DataFrame, pred: pd.DataFrame) -> None:
    out = meta_test.copy()
    for target in TARGETS:
        out[target] = clip(pred[target].values)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research/sequence_smoothing_search")
    p.add_argument("--submission-dir", default="submissions/sequence_smoothing_search")
    p.add_argument("--emission-powers", nargs="*", type=float, default=[0.65, 0.8, 1.0, 1.25])
    p.add_argument("--transition-blends", nargs="*", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--transition-smooth", type=float, default=2.0)
    p.add_argument("--full-guard", type=float, default=0.006)
    p.add_argument("--min-last-gain", type=float, default=0.0002)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.output_dir
    sub_dir = ROOT / args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading dataset")
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    full = np.ones(len(ytr), dtype=bool)
    last = folds == (C.N_SPLITS - 1)

    log("Building temporal anchor OOF/test")
    oof_model, _, test_model = fit_lgbm_oof_test(Xtr, ytr, Xte, mtr, mte, folds)
    base_oof, base_test = build_recipe_predictions(ytr, mtr, mte, folds, oof_model, test_model)
    base_full = mean_loss(ytr, base_oof, full)
    base_last = mean_loss(ytr, base_oof, last)
    log(f"anchor full={base_full:.6f} last={base_last:.6f}")

    search_rows: list[dict] = []
    chosen_last: dict[str, tuple[float, float]] = {}
    chosen_guarded: dict[str, tuple[float, float]] = {}

    for target in TARGETS:
        base_t_full = loss(ytr[target].values, base_oof[target].values)
        base_t_last = loss(ytr[target].values[last], base_oof[target].values[last])
        best_last = {"target": target, "emission_power": None, "transition_blend": None, "full": base_t_full, "last": base_t_last}
        best_guarded = best_last.copy()
        log(f"Searching sequence smoothing for {target}")
        for ep in args.emission_powers:
            for tb in args.transition_blends:
                sm_oof, _, _ = apply_smoothing(
                    base_oof,
                    base_test,
                    ytr,
                    mtr,
                    mte,
                    folds,
                    {target: (float(ep), float(tb))},
                    args.transition_smooth,
                )
                full_score = loss(ytr[target].values, sm_oof[target].values)
                last_score = loss(ytr[target].values[last], sm_oof[target].values[last])
                row = {
                    "target": target,
                    "emission_power": float(ep),
                    "transition_blend": float(tb),
                    "full_logloss": full_score,
                    "last_logloss": last_score,
                    "full_delta_vs_anchor": full_score - base_t_full,
                    "last_delta_vs_anchor": last_score - base_t_last,
                }
                search_rows.append(row)
                if last_score < best_last["last"]:
                    best_last = {"target": target, "emission_power": float(ep), "transition_blend": float(tb), "full": full_score, "last": last_score}
                if (
                    last_score < base_t_last - args.min_last_gain
                    and full_score <= base_t_full + args.full_guard
                    and last_score < best_guarded["last"]
                ):
                    best_guarded = {"target": target, "emission_power": float(ep), "transition_blend": float(tb), "full": full_score, "last": last_score}
        if best_last["emission_power"] is not None:
            chosen_last[target] = (best_last["emission_power"], best_last["transition_blend"])
        if best_guarded["emission_power"] is not None:
            chosen_guarded[target] = (best_guarded["emission_power"], best_guarded["transition_blend"])
        log(
            f"{target} anchor_last={base_t_last:.6f} "
            f"last_best=({best_last['emission_power']},{best_last['transition_blend']}) {best_last['last']:.6f} "
            f"guarded=({best_guarded['emission_power']},{best_guarded['transition_blend']}) {best_guarded['last']:.6f}"
        )

    pd.DataFrame(search_rows).sort_values(["target", "last_logloss"]).to_csv(
        out_dir / "sequence_param_search.csv", index=False
    )

    score_rows = [{"candidate": "anchor", "params": "{}", "full_logloss": base_full, "last_logloss": base_last}]
    write_submission(sub_dir / f"00_anchor_last{base_last:.6f}_full{base_full:.6f}.csv", mte, base_test)

    detail_tables = []
    for name, params in [("targetwise_guarded", chosen_guarded), ("targetwise_lastbest", chosen_last)]:
        if not params:
            continue
        sm_oof, sm_test, detail = apply_smoothing(
            base_oof,
            base_test,
            ytr,
            mtr,
            mte,
            folds,
            params,
            args.transition_smooth,
        )
        detail["candidate"] = name
        detail_tables.append(detail)
        full_score = mean_loss(ytr, sm_oof, full)
        last_score = mean_loss(ytr, sm_oof, last)
        score_rows.append({
            "candidate": name,
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
            "full_logloss": full_score,
            "last_logloss": last_score,
            "full_delta_vs_anchor": full_score - base_full,
            "last_delta_vs_anchor": last_score - base_last,
        })
        write_submission(sub_dir / f"{name}_last{last_score:.6f}_full{full_score:.6f}.csv", mte, sm_test)
        log(f"candidate {name} full={full_score:.6f} last={last_score:.6f}")

    scores = pd.DataFrame(score_rows).sort_values(["last_logloss", "full_logloss"])
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    if detail_tables:
        pd.concat(detail_tables, ignore_index=True).to_csv(out_dir / "subject_sequence_smoothing.csv", index=False)
    report = {
        "purpose": "Two-state Markov sequence smoothing over temporal-anchor probabilities.",
        "base_full": base_full,
        "base_last": base_last,
        "chosen_guarded": chosen_guarded,
        "chosen_last": chosen_last,
        "best_candidate": scores.iloc[0].to_dict(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Sequence smoothing candidates ===")
    print(scores.head(20).to_string(index=False))
    if float(scores.iloc[0]["last_logloss"]) <= 0.55:
        print("\nLOCAL <= 0.55 candidate found. Review before submission.")
    else:
        print("\nNo <= 0.55 local candidate yet. Do not submit from this run.")


if __name__ == "__main__":
    main()
