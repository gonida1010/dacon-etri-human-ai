"""Guarded target-wise blend using raw timeline sources.

This is a last-mile blender over already generated OOF/test predictions.  The
raw timeline model improved full-CV strongly but can over-predict positives on
Q2/Q3/S3, so this script searches target-wise blend actions with explicit
false-positive and test-movement penalties.
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
from .cv import subject_time_blocked_folds
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
ID_COLS = C.ID_COLS
EPS = C.PROB_CLIP


DEFAULT_BASE_RUN = ROOT / "research" / "guarded_lgbm_integration_20260623_v2"
DEFAULT_RAW_RUN = ROOT / "research" / "raw_timeline_target_model_20260623_full"

BASE_STEM = "public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5"
TIGHT_STEM = "public_aware_stack_blend_20260622_target_select_public_tight_logit_anchorblend_w0p8__balanced_Q2_only_5"

RAW_STEMS = {
    "raw_full": "raw_timeline_full",
    "raw_comp": "raw_timeline_composite",
    "raw_last": "raw_timeline_last",
}

TARGET_FULL_GUARD = {
    "Q1": 0.0005,
    "Q2": 0.0040,
    "Q3": 0.0100,
    "S1": 0.0100,
    "S2": 0.0120,
    "S3": 0.0040,
    "S4": 0.0120,
}

FP_PRESSURE_WEIGHT = {
    "Q1": 0.6,
    "Q2": 3.0,
    "Q3": 2.6,
    "S1": 0.8,
    "S2": 0.8,
    "S3": 2.4,
    "S4": 1.0,
}

TEST_MOVE_WEIGHT = {
    "Q1": 0.3,
    "Q2": 2.5,
    "Q3": 1.8,
    "S1": 0.8,
    "S2": 1.6,
    "S3": 2.2,
    "S4": 1.4,
}


@dataclass(frozen=True)
class SourcePair:
    oof: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class Action:
    target: str
    source: str
    mode: str
    alpha: float

    @property
    def key(self) -> str:
        return f"{self.target}__{self.source}__{self.mode}__a{fmt_num(self.alpha)}"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_num(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)[:180]


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {directory / pattern}, found {len(matches)}")
    return matches[0]


def read_oof(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    labels = df[[f"label__{t}" for t in TARGETS]].rename(columns={f"label__{t}": t for t in TARGETS})
    return df[ID_COLS].copy(), labels.astype(int), df[TARGETS].apply(clip)


def read_test(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    return df[ID_COLS].copy(), df[TARGETS].apply(clip)


def load_pair(run_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof_path = find_one(run_dir, f"{stem}*_oof.csv")
    test_path = find_one(run_dir, f"{stem}*_test_pred.csv")
    meta, y, oof = read_oof(oof_path)
    meta_test, test = read_test(test_path)
    return meta, y, oof, test


def safe_loss(y: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> float:
    return float(log_loss(np.asarray(y), clip(np.asarray(pred)), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def fold_losses(y: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> list[float]:
    return [mean_loss(y, pred, folds == fold) for fold in sorted(np.unique(folds))]


def logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def blend(base: np.ndarray, source: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    if alpha <= 0:
        return clip(base)
    if mode == "prob":
        return clip((1.0 - alpha) * base + alpha * source)
    if mode == "logit":
        return clip(sigmoid((1.0 - alpha) * logit(base) + alpha * logit(source)))
    raise ValueError(f"unknown mode: {mode}")


def fp_fn(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    pred_label = pred >= 0.5
    yt = y.astype(bool)
    fp = float(np.mean(pred_label & ~yt))
    fn = float(np.mean(~pred_label & yt))
    return fp, fn, float(np.mean(pred_label)), float(np.mean(yt))


def target_fold_losses(y: pd.Series, pred: np.ndarray, folds: np.ndarray) -> list[float]:
    return [safe_loss(y.values[folds == fold], pred[folds == fold]) for fold in sorted(np.unique(folds))]


def action_metrics(
    action: Action,
    base: SourcePair,
    source: SourcePair,
    y: pd.DataFrame,
    folds: np.ndarray,
    base_target_stats: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    target = action.target
    if action.source == "base":
        pred = base.oof[target].values
        test_pred = base.test[target].values
    else:
        pred = blend(base.oof[target].values, source.oof[target].values, action.alpha, action.mode)
        test_pred = blend(base.test[target].values, source.test[target].values, action.alpha, action.mode)

    fold_vals = target_fold_losses(y[target], pred, folds)
    full = safe_loss(y[target].values, pred)
    last = fold_vals[-1]
    fp, fn, pred_pos, true_pos = fp_fn(y[target].values, pred)
    base_stats = base_target_stats[target]
    test_pos = float(np.mean(test_pred >= 0.5))
    test_mean_delta = float(np.mean(test_pred - base.test[target].values))
    test_abs_delta = float(np.mean(np.abs(test_pred - base.test[target].values)))

    fp_pressure = max(0.0, pred_pos - true_pos)
    test_pos_inflation = max(0.0, test_pos - base_stats["base_test_pos_rate"])
    full_regret = max(0.0, full - base_stats["full"] - TARGET_FULL_GUARD[target])
    tail_regret = max(0.0, max(fold_vals[-3:]) - base_stats["tail3_worst"])

    public_tail_score = (
        last
        + 0.90 * full_regret
        + 0.15 * tail_regret
        + FP_PRESSURE_WEIGHT[target] * fp_pressure
        + TEST_MOVE_WEIGHT[target] * test_pos_inflation
        + 0.30 * TEST_MOVE_WEIGHT[target] * test_abs_delta
    )
    private_full_score = (
        full
        + 0.20 * max(0.0, last - base_stats["last"])
        + 0.12 * tail_regret
        + 0.50 * FP_PRESSURE_WEIGHT[target] * fp_pressure
        + 0.10 * TEST_MOVE_WEIGHT[target] * test_abs_delta
    )
    balanced_score = (
        full
        + 0.55 * last
        + 0.35 * full_regret
        + 0.20 * tail_regret
        + 0.70 * FP_PRESSURE_WEIGHT[target] * fp_pressure
        + 0.40 * TEST_MOVE_WEIGHT[target] * test_pos_inflation
        + 0.20 * TEST_MOVE_WEIGHT[target] * test_abs_delta
    )
    fp_suppress_score = (
        full
        + 0.25 * last
        + 1.20 * FP_PRESSURE_WEIGHT[target] * fp_pressure
        + 0.50 * TEST_MOVE_WEIGHT[target] * test_pos_inflation
        + 0.15 * TEST_MOVE_WEIGHT[target] * test_abs_delta
    )

    row: dict[str, float | str] = {
        "target": target,
        "source": action.source,
        "mode": action.mode,
        "alpha": action.alpha,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_base": full - base_stats["full"],
        "last_delta_vs_base": last - base_stats["last"],
        "fold_std": float(np.std(fold_vals)),
        "tail3_worst": float(max(fold_vals[-3:])),
        "fp_rate": fp,
        "fn_rate": fn,
        "pred_pos_rate": pred_pos,
        "true_pos_rate": true_pos,
        "test_pos_rate": test_pos,
        "base_test_pos_rate": base_stats["base_test_pos_rate"],
        "test_pos_delta": test_pos - base_stats["base_test_pos_rate"],
        "test_mean_delta": test_mean_delta,
        "test_abs_delta_mean": test_abs_delta,
        "fp_pressure": fp_pressure,
        "test_pos_inflation": test_pos_inflation,
        "full_regret": full_regret,
        "tail_regret": tail_regret,
        "public_tail_score": public_tail_score,
        "private_full_score": private_full_score,
        "balanced_score": balanced_score,
        "fp_suppress_score": fp_suppress_score,
    }
    row.update({f"fold{i}_logloss": value for i, value in enumerate(fold_vals)})
    return row


def build_action_table(
    base: SourcePair,
    raw_sources: dict[str, SourcePair],
    y: pd.DataFrame,
    folds: np.ndarray,
    alphas: list[float],
    modes: list[str],
) -> pd.DataFrame:
    base_target_stats: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        fold_vals = target_fold_losses(y[target], base.oof[target].values, folds)
        base_target_stats[target] = {
            "full": safe_loss(y[target].values, base.oof[target].values),
            "last": fold_vals[-1],
            "tail3_worst": float(max(fold_vals[-3:])),
            "base_test_pos_rate": float(np.mean(base.test[target].values >= 0.5)),
        }

    rows = []
    for target in TARGETS:
        base_action = Action(target=target, source="base", mode="prob", alpha=0.0)
        rows.append(action_metrics(base_action, base, base, y, folds, base_target_stats))
        for source_name, source in raw_sources.items():
            for mode in modes:
                for alpha in alphas:
                    if alpha <= 0:
                        continue
                    action = Action(target=target, source=source_name, mode=mode, alpha=float(alpha))
                    rows.append(action_metrics(action, base, source, y, folds, base_target_stats))
    return pd.DataFrame(rows)


def select_actions(actions: pd.DataFrame, selector: str) -> pd.DataFrame:
    choices = []
    for target in TARGETS:
        rows = actions[actions["target"].eq(target)].copy()
        if selector == "public_tail":
            if target in {"Q1", "Q2", "S3"}:
                rows = rows[rows["source"].eq("base")]
            else:
                rows = rows[rows["full_regret"].le(TARGET_FULL_GUARD[target] + 0.006)]
                rows = rows[rows["tail_regret"].le(0.010)]
            sort_cols = ["public_tail_score", "last_logloss", "full_logloss"]
        elif selector == "private_full":
            if target == "Q2":
                rows = rows[rows["test_pos_rate"].le(rows["base_test_pos_rate"] + 0.04)]
            if target in {"Q3", "S3"}:
                rows = rows[rows["test_pos_rate"].le(rows["base_test_pos_rate"] + 0.02)]
            sort_cols = ["private_full_score", "full_logloss", "last_logloss"]
        elif selector == "balanced":
            if target in {"Q2", "Q3", "S3"}:
                rows = rows[rows["test_pos_inflation"].le(0.03)]
            rows = rows[rows["tail_regret"].le(0.018)]
            sort_cols = ["balanced_score", "full_logloss", "last_logloss"]
        elif selector == "fp_suppress":
            if target in {"Q2", "Q3", "S3"}:
                rows = rows[rows["pred_pos_rate"].le(rows["true_pos_rate"] + 0.18)]
                rows = rows[rows["test_pos_rate"].le(rows["base_test_pos_rate"] + 0.01)]
            sort_cols = ["fp_suppress_score", "full_logloss", "last_logloss"]
        else:
            raise ValueError(f"unknown selector: {selector}")

        if rows.empty:
            rows = actions[(actions["target"].eq(target)) & (actions["source"].eq("base"))].copy()
        choices.append(rows.sort_values(sort_cols).iloc[0])
    out = pd.DataFrame(choices).reset_index(drop=True)
    out["selector"] = selector
    return out


def manual_choices(actions: pd.DataFrame) -> pd.DataFrame:
    desired = {
        "Q1": ("base", "prob", 0.0),
        "Q2": ("base", "prob", 0.0),
        "Q3": ("raw_full", "logit", 0.50),
        "S1": ("raw_last", "logit", 0.60),
        "S2": ("raw_last", "logit", 0.75),
        "S3": ("base", "prob", 0.0),
        "S4": ("raw_last", "logit", 0.55),
    }
    rows = []
    for target, (source, mode, alpha) in desired.items():
        mask = (
            actions["target"].eq(target)
            & actions["source"].eq(source)
            & actions["mode"].eq(mode)
            & np.isclose(actions["alpha"].astype(float), alpha)
        )
        if not mask.any():
            mask = actions["target"].eq(target) & actions["source"].eq("base")
        rows.append(actions[mask].iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["selector"] = "manual_public_tail"
    return out


def fixed_target_choices(actions: pd.DataFrame, desired: dict[str, tuple[str, str, float]], name: str) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        source, mode, alpha = desired.get(target, ("base", "prob", 0.0))
        mask = (
            actions["target"].eq(target)
            & actions["source"].eq(source)
            & actions["mode"].eq(mode)
            & np.isclose(actions["alpha"].astype(float), alpha)
        )
        if not mask.any():
            mask = actions["target"].eq(target) & actions["source"].eq("base")
        rows.append(actions[mask].iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["selector"] = name
    return out


def action_pred(base: SourcePair, raw_sources: dict[str, SourcePair], row: pd.Series, is_test: bool) -> np.ndarray:
    target = row["target"]
    source_name = row["source"]
    pair = base if source_name == "base" else raw_sources[str(source_name)]
    base_arr = base.test[target].values if is_test else base.oof[target].values
    src_arr = pair.test[target].values if is_test else pair.oof[target].values
    return blend(base_arr, src_arr, float(row["alpha"]), str(row["mode"]))


def build_candidate(
    name: str,
    choices: pd.DataFrame,
    base: SourcePair,
    raw_sources: dict[str, SourcePair],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = base.oof.copy()
    test = base.test.copy()
    choice_rows = []
    for _, row in choices.iterrows():
        target = str(row["target"])
        oof[target] = action_pred(base, raw_sources, row, is_test=False)
        test[target] = action_pred(base, raw_sources, row, is_test=True)
        choice = row.to_dict()
        choice["candidate"] = name
        choice_rows.append(choice)
    return oof, test, pd.DataFrame(choice_rows)


def global_candidate_choices(actions: pd.DataFrame, source: str, mode: str, alpha: float, name: str) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        mask = (
            actions["target"].eq(target)
            & actions["source"].eq(source)
            & actions["mode"].eq(mode)
            & np.isclose(actions["alpha"].astype(float), alpha)
        )
        rows.append(actions[mask].iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["selector"] = name
    return out


def base_choices(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions[actions["source"].eq("base")].copy().reset_index(drop=True)
    out["selector"] = "base_public_best"
    return out


def candidate_row(
    name: str,
    pred: pd.DataFrame,
    y: pd.DataFrame,
    folds: np.ndarray,
    base_full: float,
    base_last: float,
) -> dict[str, float | str]:
    vals = fold_losses(y, pred, folds)
    full = mean_loss(y, pred, np.ones(len(y), dtype=bool))
    last = vals[-1]
    row: dict[str, float | str] = {
        "candidate": name,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_base": full - base_full,
        "last_delta_vs_base": last - base_last,
        "fold_std": float(np.std(vals)),
        "tail3_mean": float(np.mean(vals[-3:])),
        "tail3_worst": float(max(vals[-3:])),
    }
    row.update({f"fold{i}_logloss": value for i, value in enumerate(vals)})
    for target in TARGETS:
        fp, fn, pred_pos, true_pos = fp_fn(y[target].values, pred[target].values)
        row[f"{target}_fp_rate"] = fp
        row[f"{target}_fn_rate"] = fn
        row[f"{target}_pred_pos_rate"] = pred_pos
        row[f"{target}_true_pos_rate"] = true_pos
    return row


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--raw-run-dir", type=Path, default=DEFAULT_RAW_RUN)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research" / "raw_timeline_guarded_blend_20260624")
    parser.add_argument("--submission-dir", type=Path, default=ROOT / "submissions" / "raw_timeline_guarded_blend_20260624")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00])
    parser.add_argument("--modes", nargs="+", default=["logit", "prob"], choices=["logit", "prob"])
    args = parser.parse_args()

    out_dir = args.output_dir
    sub_dir = args.submission_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    log("Loading base and raw sources")
    meta, y, base_oof, base_test = load_pair(args.base_run_dir, BASE_STEM)
    meta_tight, y_tight, tight_oof, tight_test = load_pair(args.base_run_dir, TIGHT_STEM)
    if not meta.equals(meta_tight) or not y.equals(y_tight):
        raise ValueError("base and tight OOF rows do not align")

    raw_sources: dict[str, SourcePair] = {}
    for name, stem in RAW_STEMS.items():
        meta_raw, y_raw, raw_oof, raw_test = load_pair(args.raw_run_dir, stem)
        if not meta.equals(meta_raw) or not y.equals(y_raw):
            raise ValueError(f"{name} OOF rows do not align with base")
        raw_sources[name] = SourcePair(raw_oof, raw_test)

    meta_test, _ = read_test(find_one(args.base_run_dir, f"{BASE_STEM}*_test_pred.csv"))
    base = SourcePair(base_oof, base_test)
    tight = SourcePair(tight_oof, tight_test)
    folds = subject_time_blocked_folds(meta, n_splits=C.N_SPLITS)
    base_full = mean_loss(y, base.oof, np.ones(len(y), dtype=bool))
    base_last = mean_loss(y, base.oof, folds == C.N_SPLITS - 1)

    log("Building action table")
    actions = build_action_table(base, raw_sources, y, folds, args.alphas, args.modes)
    actions.to_csv(out_dir / "action_scores.csv", index=False)

    candidate_defs: dict[str, pd.DataFrame] = {
        "base_public_best": base_choices(actions),
        "selector_public_tail": select_actions(actions, "public_tail"),
        "selector_private_full": select_actions(actions, "private_full"),
        "selector_balanced": select_actions(actions, "balanced"),
        "selector_fp_suppress": select_actions(actions, "fp_suppress"),
        "manual_public_tail": manual_choices(actions),
        "manual_v2_q3_only": fixed_target_choices(
            actions,
            {"Q3": ("raw_full", "logit", 0.50)},
            "manual_v2_q3_only",
        ),
        "manual_v2_q3_s1": fixed_target_choices(
            actions,
            {
                "Q3": ("raw_full", "logit", 0.50),
                "S1": ("raw_last", "logit", 0.60),
            },
            "manual_v2_q3_s1",
        ),
        "manual_v2_no_s2": fixed_target_choices(
            actions,
            {
                "Q3": ("raw_full", "logit", 0.50),
                "S1": ("raw_last", "logit", 0.60),
                "S4": ("raw_last", "logit", 0.55),
            },
            "manual_v2_no_s2",
        ),
        "global_raw_full_logit_a0p80": global_candidate_choices(actions, "raw_full", "logit", 0.80, "global_raw_full_logit_a0p80"),
        "global_raw_last_logit_a0p50": global_candidate_choices(actions, "raw_last", "logit", 0.50, "global_raw_last_logit_a0p50"),
        "global_raw_comp_logit_a0p65": global_candidate_choices(actions, "raw_comp", "logit", 0.65, "global_raw_comp_logit_a0p65"),
    }

    # Add the previous tight public-aware family as a reference candidate.
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]] = {
        "base_public_best": (base.oof, base.test, candidate_defs["base_public_best"]),
        "reference_public_tight": (tight.oof, tight.test, None),
    }
    for name, choices in candidate_defs.items():
        if name == "base_public_best":
            continue
        oof, test, choice_df = build_candidate(name, choices, base, raw_sources)
        candidates[name] = (oof, test, choice_df)

    log("Scoring and writing candidates")
    score_rows = []
    choice_frames = []
    for name, (oof, test, choices) in candidates.items():
        row = candidate_row(name, oof, y, folds, base_full, base_last)
        score_rows.append(row)
        if choices is not None:
            choice_frames.append(choices.assign(candidate=name))
        oof_path = out_dir / f"{safe_name(name)}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}_oof.csv"
        test_path = out_dir / f"{safe_name(name)}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}_test_pred.csv"
        sub_path = sub_dir / f"{safe_name(name)}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv"
        write_prediction(oof_path, meta, oof, y)
        write_submission(test_path, meta_test, test)
        write_submission(sub_path, meta_test, test)

    scores = pd.DataFrame(score_rows).sort_values(["last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    choices_all = pd.concat(choice_frames, ignore_index=True)
    choices_all.to_csv(out_dir / "candidate_choices.csv", index=False)

    report = {
        "base_full": base_full,
        "base_last": base_last,
        "candidate_scores": scores.to_dict(orient="records"),
        "top_actions_by_public_tail": actions.sort_values(["target", "public_tail_score"]).groupby("target").head(5).to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Raw Timeline Guarded Blend",
        "",
        "## Candidate Scores",
        "",
        scores[["candidate", "full_logloss", "last_logloss", "full_delta_vs_base", "last_delta_vs_base", "fold_std", "tail3_worst"]].to_string(index=False),
        "",
        "## Target Choices",
        "",
        choices_all[["candidate", "target", "source", "mode", "alpha", "full_logloss", "last_logloss", "fp_rate", "fn_rate", "pred_pos_rate", "test_pos_rate", "public_tail_score", "private_full_score", "balanced_score"]].to_string(index=False),
        "",
        "## Notes",
        "",
        "- `base_public_best` is the previously submitted public-aware family baseline.",
        "- Direct raw timeline submissions are not selected; raw sources are used target-wise with FP/test-movement penalties.",
        "- Review candidate_scores and candidate_choices before any submission decision.",
    ]
    (out_dir / "RAW_TIMELINE_GUARDED_BLEND_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print("\n=== Raw timeline guarded blend candidates ===")
    print(scores[["candidate", "full_logloss", "last_logloss", "full_delta_vs_base", "last_delta_vs_base", "fold_std", "tail3_worst"]].to_string(index=False))
    print(f"\nWrote outputs to {out_dir} and {sub_dir}")


if __name__ == "__main__":
    main()
