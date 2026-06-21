"""Search subject-level whole-period balance constraints for Q targets.

The Q targets are defined relative to each subject's long-period mean. That
creates a structural count constraint: train labels are known, and the future
test block should complete the subject's long-period positive rate. This script
does not assume that the rate is exactly 0.5. It searches the rate by honest
subject-time-blocked OOF, then writes only diagnostics and candidate files.

Run:
  python -m src.structural_balance_search
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
from .train_temporal_prior import (
    build_recipe_predictions,
    clip,
    fit_lgbm_oof_test,
    ll,
)

TARGETS = C.TARGET_COLS
Q_TARGETS = ["Q1", "Q2", "Q3"]
ROOT = C.PROJECT_ROOT
EPS = C.PROB_CLIP


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def shift_to_mean(prob: np.ndarray, mean: float) -> np.ndarray:
    """Keep ranks, shift logit intercept so the group mean becomes `mean`."""
    mean = float(np.clip(mean, EPS, 1 - EPS))
    z = logit(prob)
    lo, hi = -30.0, 30.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if sigmoid(z + mid).mean() < mean:
            lo = mid
        else:
            hi = mid
    return clip(sigmoid(z + (lo + hi) / 2))


def target_loss(y: pd.DataFrame, pred: pd.DataFrame, target: str, mask: np.ndarray) -> float:
    return float(log_loss(y[target].values[mask], clip(pred[target].values[mask]), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([target_loss(y, pred, t, mask) for t in TARGETS]))


def calibrate_q_targets(
    pred_oof: pd.DataFrame,
    pred_test: pd.DataFrame,
    ytr: pd.DataFrame,
    mtr: pd.DataFrame,
    mte: pd.DataFrame,
    folds: np.ndarray,
    rates: dict[str, float],
    min_rate: float,
    max_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_oof = pred_oof.copy()
    out_test = pred_test.copy()
    rows: list[dict] = []
    subj_train = mtr["subject_id"].astype(str).values
    subj_test = mte["subject_id"].astype(str).values
    subjects = sorted(np.unique(subj_train))

    for target, balance_rate in rates.items():
        for fold in sorted(np.unique(folds)):
            for subject in subjects:
                va = (folds == fold) & (subj_train == subject)
                if not va.any():
                    continue
                tr = (folds != fold) & (subj_train == subject)
                n_hist = int(tr.sum())
                n_future = int(va.sum())
                hist_pos = float(ytr.loc[tr, target].sum())
                expected_future = balance_rate * (n_hist + n_future) - hist_pos
                future_mean = float(np.clip(expected_future / max(n_future, 1), min_rate, max_rate))
                before = out_oof.loc[va, target].values
                after = shift_to_mean(before, future_mean)
                out_oof.loc[va, target] = after
                rows.append({
                    "split": "oof",
                    "target": target,
                    "fold": int(fold),
                    "subject_id": subject,
                    "balance_rate": balance_rate,
                    "history_rows": n_hist,
                    "future_rows": n_future,
                    "history_pos_rate": hist_pos / max(n_hist, 1),
                    "target_future_mean": future_mean,
                    "before_mean": float(np.mean(before)),
                    "after_mean": float(np.mean(after)),
                })

        for subject in subjects:
            te = subj_test == subject
            if not te.any():
                continue
            tr = subj_train == subject
            n_hist = int(tr.sum())
            n_future = int(te.sum())
            hist_pos = float(ytr.loc[tr, target].sum())
            expected_future = balance_rate * (n_hist + n_future) - hist_pos
            future_mean = float(np.clip(expected_future / max(n_future, 1), min_rate, max_rate))
            before = out_test.loc[te, target].values
            after = shift_to_mean(before, future_mean)
            out_test.loc[te, target] = after
            rows.append({
                "split": "test",
                "target": target,
                "fold": -1,
                "subject_id": subject,
                "balance_rate": balance_rate,
                "history_rows": n_hist,
                "future_rows": n_future,
                "history_pos_rate": hist_pos / max(n_hist, 1),
                "target_future_mean": future_mean,
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
    p.add_argument("--output-dir", default="research/structural_balance_search")
    p.add_argument("--submission-dir", default="submissions/structural_balance_search")
    p.add_argument("--rate-min", type=float, default=0.35)
    p.add_argument("--rate-max", type=float, default=0.70)
    p.add_argument("--rate-step", type=float, default=0.01)
    p.add_argument("--min-future-rate", type=float, default=0.02)
    p.add_argument("--max-future-rate", type=float, default=0.98)
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

    rates = np.round(np.arange(args.rate_min, args.rate_max + 0.5 * args.rate_step, args.rate_step), 6)
    rate_rows: list[dict] = []
    target_best: dict[str, dict] = {}

    for target in Q_TARGETS:
        log(f"Searching balance rate for {target}")
        target_base_full = target_loss(ytr, base_oof, target, full)
        target_base_last = target_loss(ytr, base_oof, target, last)
        best_last = {
            "target": target,
            "rate": None,
            "full": target_base_full,
            "last": target_base_last,
            "kind": "anchor",
        }
        best_guarded = best_last.copy()
        for rate in rates:
            cal_oof, _, _ = calibrate_q_targets(
                base_oof,
                base_test,
                ytr,
                mtr,
                mte,
                folds,
                {target: float(rate)},
                args.min_future_rate,
                args.max_future_rate,
            )
            full_score = target_loss(ytr, cal_oof, target, full)
            last_score = target_loss(ytr, cal_oof, target, last)
            row = {
                "target": target,
                "rate": float(rate),
                "full_logloss": full_score,
                "last_logloss": last_score,
                "full_delta_vs_anchor": full_score - target_base_full,
                "last_delta_vs_anchor": last_score - target_base_last,
            }
            rate_rows.append(row)
            if last_score < best_last["last"]:
                best_last = {**row, "kind": "last_best"}
            if (
                last_score < target_base_last - args.min_last_gain
                and full_score <= target_base_full + args.full_guard
                and last_score < best_guarded["last"]
            ):
                best_guarded = {**row, "kind": "guarded"}
        target_best[target] = {
            "base": best_last if best_last["kind"] == "anchor" else {
                "target": target,
                "rate": None,
                "full": target_base_full,
                "last": target_base_last,
                "kind": "anchor",
            },
            "last_best": best_last,
            "guarded": best_guarded,
        }
        log(
            f"{target} anchor_last={target_base_last:.6f} "
            f"last_best_rate={best_last.get('rate')} last={best_last.get('last_logloss', best_last['last']):.6f} "
            f"guarded_rate={best_guarded.get('rate')} guarded_last={best_guarded.get('last_logloss', best_guarded['last']):.6f}"
        )

    pd.DataFrame(rate_rows).sort_values(["target", "last_logloss"]).to_csv(
        out_dir / "q_rate_search.csv", index=False
    )

    candidates: list[tuple[str, dict[str, float]]] = []
    guarded_rates = {
        t: float(pack["guarded"]["rate"])
        for t, pack in target_best.items()
        if pack["guarded"].get("rate") is not None
    }
    lastbest_rates = {
        t: float(pack["last_best"]["rate"])
        for t, pack in target_best.items()
        if pack["last_best"].get("rate") is not None
    }
    if guarded_rates:
        candidates.append(("targetwise_guarded", guarded_rates))
    if lastbest_rates:
        candidates.append(("targetwise_lastbest", lastbest_rates))
    for rate in rates:
        candidates.append((f"allq_rate_{str(float(rate)).replace('.', 'p')}", {t: float(rate) for t in Q_TARGETS}))

    score_rows = [{"candidate": "anchor", "rates": "{}", "full_logloss": base_full, "last_logloss": base_last}]
    write_submission(
        sub_dir / f"00_anchor_last{base_last:.6f}_full{base_full:.6f}.csv",
        mte,
        base_test,
    )

    detail_tables = []
    for name, rate_map in candidates:
        cal_oof, cal_test, detail = calibrate_q_targets(
            base_oof,
            base_test,
            ytr,
            mtr,
            mte,
            folds,
            rate_map,
            args.min_future_rate,
            args.max_future_rate,
        )
        detail["candidate"] = name
        detail_tables.append(detail)
        full_score = mean_loss(ytr, cal_oof, full)
        last_score = mean_loss(ytr, cal_oof, last)
        score_rows.append({
            "candidate": name,
            "rates": json.dumps(rate_map, ensure_ascii=False, sort_keys=True),
            "full_logloss": full_score,
            "last_logloss": last_score,
            "full_delta_vs_anchor": full_score - base_full,
            "last_delta_vs_anchor": last_score - base_last,
        })
        if last_score <= base_last - args.min_last_gain or name.startswith("targetwise"):
            safe_name = name.replace(".", "p")
            write_submission(
                sub_dir / f"{safe_name}_last{last_score:.6f}_full{full_score:.6f}.csv",
                mte,
                cal_test,
            )
            log(f"candidate {name} full={full_score:.6f} last={last_score:.6f}")

    scores = pd.DataFrame(score_rows).sort_values(["last_logloss", "full_logloss"])
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    if detail_tables:
        pd.concat(detail_tables, ignore_index=True).to_csv(out_dir / "subject_calibration.csv", index=False)
    report = {
        "purpose": "Q target whole-period balance search with subject-time-blocked OOF.",
        "base_full": base_full,
        "base_last": base_last,
        "rate_min": args.rate_min,
        "rate_max": args.rate_max,
        "rate_step": args.rate_step,
        "target_best": target_best,
        "best_candidate": scores.iloc[0].to_dict(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Structural balance candidates ===")
    print(scores.head(20).to_string(index=False))
    best_last = float(scores.iloc[0]["last_logloss"])
    if best_last <= 0.55:
        print("\nLOCAL <= 0.55 candidate found. Review before submission.")
    else:
        print("\nNo <= 0.55 local candidate yet. Do not submit from this run.")


if __name__ == "__main__":
    main()
