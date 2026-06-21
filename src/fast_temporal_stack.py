"""Fast OOF temporal source search for leaderboard-oriented candidates.

This ports the Kaggle workflow to this Dacon task:
- build an OOF/test prediction bank from model, subject priors, recent means, and ridge trends
- search target-wise blends on the last time block, the local proxy that matched LB before
- save diagnostics and a few submission candidates

Run:
  python -m src.fast_temporal_stack
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds
from .train_temporal_prior import fit_lgbm_oof_test, temporal_oof, temporal_test, clip, RECIPES

TARGETS = C.TARGET_COLS
ROOT = C.PROJECT_ROOT
OUT_DIR = ROOT / "research" / "fast_temporal_stack"
SUB_DIR = C.SUBMISSION_DIR / "fast_temporal_stack"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "mean_sm4", "mean_sm8", "mean_sm16", "mean_sm32", "mean_sm64",
    "last2_sm2", "last2_sm4", "last3_sm4", "last5_sm4", "last10_sm4", "last20_sm4", "last30_sm4",
    "ridge0.5", "ridge1", "ridge3", "ridge10", "ridge30",
]
WGRID = np.round(np.linspace(0, 1, 21), 3)


def progress(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ll(y, p) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def col_values(pred, target: str) -> np.ndarray:
    if isinstance(pred, pd.DataFrame):
        return pred[target].values
    return np.asarray(pred)[:, TARGETS.index(target)]


def mean_loss(y_df: pd.DataFrame, pred_df, mask: np.ndarray) -> float:
    return float(np.mean([ll(y_df[t].values[mask], col_values(pred_df, t)[mask]) for t in TARGETS]))


def recipe_pred(target: str, recipe, sources: dict[str, dict[str, np.ndarray]], split: str) -> np.ndarray:
    n = len(next(iter(sources.values()))[split])
    out = np.zeros(n, dtype=float)
    for src, weight in recipe:
        out += weight * sources[src][split]
    return clip(out)


def save_submission(name: str, meta_test: pd.DataFrame, pred: pd.DataFrame, records: list[dict], scores: dict) -> None:
    sub = meta_test.copy()
    for t in TARGETS:
        sub[t] = clip(pred[t].values)
    path = SUB_DIR / name
    sub.to_csv(path, index=False)
    records.append({"file": str(path.relative_to(ROOT)), **scores})
    progress(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    progress("Loading dataset")
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=C.N_SPLITS)
    last = folds == (C.N_SPLITS - 1)
    full = np.ones(len(ytr), dtype=bool)

    progress("Training model OOF/test bank")
    oof_model, _, test_model = fit_lgbm_oof_test(Xtr, ytr, Xte, mtr, mte, folds)

    target_rows = []
    final_last = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    final_guarded = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    test_last = pd.DataFrame(index=mte.index, columns=TARGETS, dtype=float)
    test_guarded = pd.DataFrame(index=mte.index, columns=TARGETS, dtype=float)
    baseline_oof = pd.DataFrame(index=ytr.index, columns=TARGETS, dtype=float)
    baseline_test = pd.DataFrame(index=mte.index, columns=TARGETS, dtype=float)

    progress("Building temporal source bank and searching target-wise blends")
    for target in TARGETS:
        progress(f"Target {target}")
        sources = {
            "model": {"oof": oof_model[target].values, "test": test_model[target].values}
        }
        for method in METHODS:
            sources[method] = {
                "oof": temporal_oof(ytr, mtr, folds, target, method),
                "test": temporal_test(ytr, mtr, mte, target, method),
            }

        base_oof = recipe_pred(target, RECIPES[target], sources, "oof")
        base_test = recipe_pred(target, RECIPES[target], sources, "test")
        baseline_oof[target] = base_oof
        baseline_test[target] = base_test
        base_full = ll(ytr[target].values, base_oof)
        base_last = ll(ytr[target].values[last], base_oof[last])

        candidates = []
        for src, pack in sources.items():
            pred = clip(pack["oof"])
            candidates.append({
                "target": target, "recipe": src, "src": src, "weight": 1.0,
                "full": ll(ytr[target].values, pred),
                "last": ll(ytr[target].values[last], pred[last]),
                "oof": pred, "test": clip(pack["test"]),
            })
        for src, pack in sources.items():
            if src == "model":
                continue
            for w in WGRID:
                pred = clip((1 - w) * sources["model"]["oof"] + w * pack["oof"])
                candidates.append({
                    "target": target, "recipe": f"{1-w:.2f}*model+{w:.2f}*{src}", "src": src, "weight": float(w),
                    "full": ll(ytr[target].values, pred),
                    "last": ll(ytr[target].values[last], pred[last]),
                    "oof": pred,
                    "test": clip((1 - w) * sources["model"]["test"] + w * pack["test"]),
                })

        best_last = min(candidates, key=lambda r: r["last"])
        # Guarded: use best-last only when it beats current recipe on last block. Public-oriented margin is intentionally small.
        guarded = best_last if best_last["last"] < base_last - 0.0002 else {
            "recipe": "baseline_recipe", "full": base_full, "last": base_last, "oof": base_oof, "test": base_test
        }
        final_last[target] = best_last["oof"]
        test_last[target] = best_last["test"]
        final_guarded[target] = guarded["oof"]
        test_guarded[target] = guarded["test"]

        for row in sorted(candidates, key=lambda r: r["last"])[:12]:
            target_rows.append({k: row[k] for k in ["target", "recipe", "src", "weight", "full", "last"] if k in row})
        target_rows.append({"target": target, "recipe": "BASELINE_CURRENT", "src": "baseline", "weight": np.nan, "full": base_full, "last": base_last})
        target_rows.append({"target": target, "recipe": "CHOSEN_LASTBEST", "src": best_last["src"], "weight": best_last.get("weight", np.nan), "full": best_last["full"], "last": best_last["last"]})
        target_rows.append({"target": target, "recipe": "CHOSEN_GUARDED", "src": guarded.get("src", "baseline"), "weight": guarded.get("weight", np.nan), "full": guarded["full"], "last": guarded["last"]})

    records = []
    scores = []
    for name, oof, test in [
        ("01_lastbest_targetwise.csv", final_last, test_last),
        ("02_guarded_targetwise.csv", final_guarded, test_guarded),
        ("03_baseline_recipe_rebuilt.csv", baseline_oof, baseline_test),
    ]:
        full_score = mean_loss(ytr, oof, full)
        last_score = mean_loss(ytr, oof, last)
        save_submission(name, mte, test, records, {"full_logloss": full_score, "last_logloss": last_score})
        scores.append({"candidate": name, "full_logloss": full_score, "last_logloss": last_score})

    # Blend baseline and lastbest. This mirrors public-LB blending while keeping generated OOF diagnostics.
    for w in [0.25, 0.50, 0.75]:
        oof = pd.DataFrame(clip(w * final_last.values + (1 - w) * baseline_oof.values), columns=TARGETS)
        test = pd.DataFrame(clip(w * test_last.values + (1 - w) * baseline_test.values), columns=TARGETS)
        name = f"04_blend_lastbest{int(w*100):02d}_baseline{int((1-w)*100):02d}.csv"
        full_score = mean_loss(ytr, oof, full)
        last_score = mean_loss(ytr, oof, last)
        save_submission(name, mte, test, records, {"full_logloss": full_score, "last_logloss": last_score, "lastbest_weight": w})
        scores.append({"candidate": name, "full_logloss": full_score, "last_logloss": last_score})

    pd.DataFrame(target_rows).to_csv(OUT_DIR / "target_source_search.csv", index=False)
    pd.DataFrame(scores).sort_values("last_logloss").to_csv(OUT_DIR / "candidate_scores.csv", index=False)
    report = {
        "purpose": "Fast public-oriented temporal source search using last-block as LB proxy.",
        "scores": scores,
        "outputs": records,
        "target_search": str((OUT_DIR / "target_source_search.csv").relative_to(ROOT)),
        "candidate_scores": str((OUT_DIR / "candidate_scores.csv").relative_to(ROOT)),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nCandidate scores:")
    print(pd.DataFrame(scores).sort_values("last_logloss").to_string(index=False))
    print("\nSubmit-first shortlist:")
    print("1. submissions/fast_temporal_stack/02_guarded_targetwise.csv")
    print("2. submissions/fast_temporal_stack/01_lastbest_targetwise.csv")
    print("3. submissions/fast_temporal_stack/04_blend_lastbest50_baseline50.csv")


if __name__ == "__main__":
    main()
