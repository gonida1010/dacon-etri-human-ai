"""Fast leaderboard candidates from existing submission CSVs.

This is intentionally public-LB oriented: no labels are used. It creates a small
set of probability blends from already generated candidates so we can spend the
3 daily submissions on meaningfully different files.

Run:
  python -m src.blend_existing_submissions
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

TARGETS = C.TARGET_COLS
ID_COLS = C.ID_COLS
ROOT = C.PROJECT_ROOT
SUB_DIR = C.SUBMISSION_DIR
OUT_DIR = SUB_DIR / "leaderboard_blends"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPS = 1e-6


def progress(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, EPS, 1.0 - EPS)


def logit(x: np.ndarray) -> np.ndarray:
    x = clip(x)
    return np.log(x / (1.0 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def shrink(p: np.ndarray, alpha: float) -> np.ndarray:
    return clip(0.5 + alpha * (p - 0.5))


def load_sub(name: str) -> pd.DataFrame:
    path = SUB_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in [*ID_COLS, *TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[ID_COLS + TARGETS].copy()


def save(name: str, base: pd.DataFrame, arr: np.ndarray, meta: dict, records: list[dict]) -> None:
    out = base[ID_COLS].copy()
    out[TARGETS] = clip(arr)
    path = OUT_DIR / name
    out.to_csv(path, index=False)
    stat = out[TARGETS].agg(["mean", "std", "min", "max"]).round(6).to_dict()
    record = {"file": str(path.relative_to(ROOT)), **meta, "stats": stat}
    records.append(record)
    progress(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    progress("Loading existing submissions")
    names = {
        "temporal": "submission_temporal_prior_last0.5933.csv",
        "robust": "submission_robust_fwd0.5899.csv",
        "final": "submission_final_last0.5957.csv",
        "last": "submission_last0.6025.csv",
    }
    dfs = {k: load_sub(v) for k, v in names.items()}
    base = dfs["temporal"]
    arr = {k: v[TARGETS].to_numpy(float) for k, v in dfs.items()}

    records: list[dict] = []
    progress("Writing direct copies with clear names")
    save("01_direct_robust_fwd0p5899.csv", base, arr["robust"], {"kind": "direct", "source": "robust"}, records)
    save("02_direct_temporal_prior0p5933.csv", base, arr["temporal"], {"kind": "direct", "source": "temporal"}, records)

    progress("Writing arithmetic/logit blends")
    for w in [0.20, 0.35, 0.50, 0.65, 0.80]:
        a = w * arr["robust"] + (1 - w) * arr["temporal"]
        save(f"blend_arith_robust{int(w*100):02d}_temporal{int((1-w)*100):02d}.csv", base, a,
             {"kind": "arith_blend", "robust_weight": w, "temporal_weight": 1 - w}, records)
        g = sigmoid(w * logit(arr["robust"]) + (1 - w) * logit(arr["temporal"]))
        save(f"blend_logit_robust{int(w*100):02d}_temporal{int((1-w)*100):02d}.csv", base, g,
             {"kind": "logit_blend", "robust_weight": w, "temporal_weight": 1 - w}, records)

    progress("Writing target-wise mosaics")
    # Robust differs mostly on Q2/Q3/S2/S3/S4. Keep strong unchanged Q1/S1 from temporal/robust equivalently.
    mosaics = {
        "mosaic_robust_q2q3s2s3s4.csv": {"Q2": "robust", "Q3": "robust", "S2": "robust", "S3": "robust", "S4": "robust"},
        "mosaic_robust_q2q3_s2s4_temporal_s3.csv": {"Q2": "robust", "Q3": "robust", "S2": "robust", "S4": "robust"},
        "mosaic_final_q3_robust_q2s2s3s4.csv": {"Q2": "robust", "Q3": "final", "S2": "robust", "S3": "robust", "S4": "robust"},
        "mosaic_last_s3_robust_q2q3s2s4.csv": {"Q2": "robust", "Q3": "robust", "S2": "robust", "S3": "last", "S4": "robust"},
    }
    for fname, spec in mosaics.items():
        a = arr["temporal"].copy()
        for target, src in spec.items():
            a[:, TARGETS.index(target)] = arr[src][:, TARGETS.index(target)]
        save(fname, base, a, {"kind": "target_mosaic", "spec": spec}, records)

    progress("Writing calibration/shrink variants")
    for src in ["temporal", "robust"]:
        for alpha in [0.85, 0.92, 1.08]:
            save(f"cal_{src}_alpha{str(alpha).replace('.', 'p')}.csv", base, shrink(arr[src], alpha),
                 {"kind": "global_shrink", "source": src, "alpha": alpha}, records)

    report_path = OUT_DIR / "blend_report.json"
    report_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in r.items() if k != "stats"} for r in records]).to_csv(
        OUT_DIR / "blend_index.csv", index=False
    )
    progress(f"report: {report_path.relative_to(ROOT)}")
    print("\nSubmit-first shortlist:")
    print("1. submissions/leaderboard_blends/01_direct_robust_fwd0p5899.csv")
    print("2. submissions/leaderboard_blends/blend_logit_robust65_temporal35.csv")
    print("3. submissions/leaderboard_blends/mosaic_robust_q2q3_s2s4_temporal_s3.csv")


if __name__ == "__main__":
    main()
