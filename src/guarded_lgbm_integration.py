"""Guarded integration of the target-weighted LGBM source.

The generic public-aware selector rejected the new LGBM source because the raw
Q2/S2 test movement is large.  This script extracts only the useful target-wise
signal with explicit shrink and movement caps, then evaluates combinations on
the existing OOF folds.
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
from .public_aware_stack_blend import bad_alignment_risk, public_direction_risk
from .train_temporal_prior import clip

ROOT = C.PROJECT_ROOT
TARGETS = C.TARGET_COLS
EPS = C.PROB_CLIP


@dataclass(frozen=True)
class Action:
    target: str
    source: str
    mode: str
    alpha: float
    cap: float
    anchor_ref: str

    @property
    def key(self) -> str:
        return (
            f"{self.target}__{self.source.replace('/', '_')}__{self.mode}"
            f"__a{fmt_num(self.alpha)}__cap{fmt_num(self.cap)}__{self.anchor_ref}"
        )


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_num(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def safe_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace(" ", "_")
    )


def safe_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(np.asarray(y), clip(p), labels=[0, 1]))


def mean_loss(y: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> float:
    return float(np.mean([safe_loss(y[t].values[mask], pred[t].values[mask]) for t in TARGETS]))


def target_loss(y: pd.DataFrame, pred: pd.DataFrame, target: str, mask: np.ndarray) -> float:
    return safe_loss(y[target].values[mask], pred[target].values[mask])


def fold_losses(y: pd.DataFrame, pred: pd.DataFrame, folds: np.ndarray) -> list[float]:
    return [mean_loss(y, pred, folds == f) for f in sorted(np.unique(folds))]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_submission(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return read_pred(path)


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


def load_ranked_names(source_dir: Path, top_n: int) -> list[str]:
    score_path = source_dir / "candidate_scores.csv"
    if not score_path.exists():
        return []
    scores = pd.read_csv(score_path)
    if "candidate" not in scores.columns:
        return []
    sort_cols = [c for c in ["selector_score", "rank_score", "last_logloss", "full_logloss"] if c in scores.columns]
    if sort_cols:
        scores = scores.sort_values(sort_cols)
    return [f"{source_dir.name}/{name}" for name in scores["candidate"].astype(str).head(top_n).tolist()]


def load_base_pool(base_dirs: list[str], top_n: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    pairs = discover_pairs(base_dirs)
    ranked: list[str] = []
    for item in base_dirs:
        ranked.extend(load_ranked_names(rel(item), top_n))
    if not ranked:
        return pairs
    selected = {name: pairs[name] for name in ranked if name in pairs}
    for name in [
        "public_aware_stack_blend_20260622/target_select_public_tight",
        "public_aware_stack_blend_20260622/target_select_public_aggressive",
        "public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p8",
        "public_aware_stack_blend_with_lgbm_source_20260622/target_select_public_tight",
        "public_aware_stack_blend_with_lgbm_source_20260622/target_select_public_aggressive",
    ]:
        if name in pairs:
            selected[name] = pairs[name]
    return selected


def load_lgbm_sources(lgbm_dir: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    pairs = discover_pairs([lgbm_dir])
    wanted = {
        "target_weighted_single_composite",
        "target_weighted_single_full",
        "target_weighted_single_last",
    }
    out = {}
    for name, pair in pairs.items():
        stem = name.split("/", 1)[-1]
        if any(stem.startswith(w) for w in wanted):
            out[name] = pair
    return out


def blend_values(base: np.ndarray, source: np.ndarray, alpha: float, mode: str) -> np.ndarray:
    if mode == "logit":
        return clip(sigmoid(logit(base) + alpha * (logit(source) - logit(base))))
    if mode == "prob":
        return clip(base + alpha * (source - base))
    raise ValueError(f"unknown blend mode: {mode}")


def cap_against_anchor(pred: np.ndarray, anchor: np.ndarray, cap: float, mode: str) -> tuple[np.ndarray, float]:
    delta = np.asarray(pred, dtype=float) - np.asarray(anchor, dtype=float)
    move = float(np.mean(np.abs(delta)))
    if cap <= 0 or move <= cap:
        return clip(pred), 1.0
    scale = cap / max(move, 1e-12)
    if mode == "logit":
        capped = sigmoid(logit(anchor) + scale * (logit(pred) - logit(anchor)))
    else:
        capped = anchor + scale * (pred - anchor)
    return clip(capped), float(scale)


def apply_action_to_pair(
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    lgbm_oof: pd.DataFrame,
    lgbm_test: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    action: Action,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    out_oof = base_oof.copy()
    out_test = base_test.copy()
    target = action.target
    if action.anchor_ref == "anchor":
        train_ref = anchor_oof[target].values
        test_ref = anchor_test[target].values
    elif action.anchor_ref == "base":
        train_ref = base_oof[target].values
        test_ref = base_test[target].values
    else:
        raise ValueError(f"unknown anchor_ref: {action.anchor_ref}")

    p_train = blend_values(train_ref, lgbm_oof[target].values, action.alpha, action.mode)
    p_test = blend_values(test_ref, lgbm_test[target].values, action.alpha, action.mode)
    p_train, train_cap_scale = cap_against_anchor(p_train, anchor_oof[target].values, action.cap, action.mode)
    p_test, test_cap_scale = cap_against_anchor(p_test, anchor_test[target].values, action.cap, action.mode)
    out_oof[target] = p_train
    out_test[target] = p_test
    return out_oof, out_test, {
        "train_cap_scale": train_cap_scale,
        "test_cap_scale": test_cap_scale,
        "test_abs_delta_mean": float(np.mean(np.abs(p_test - anchor_test[target].values))),
        "test_mean_delta": float(np.mean(p_test - anchor_test[target].values)),
        "public_risk": public_direction_risk(target, p_test - anchor_test[target].values),
    }


def apply_actions(
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    action_pairs: list[tuple[Action, tuple[pd.DataFrame, pd.DataFrame]]],
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float | str]]]:
    oof = base_oof.copy()
    test = base_test.copy()
    diagnostics: list[dict[str, float | str]] = []
    for action, (src_oof, src_test) in action_pairs:
        before_oof = oof
        before_test = test
        oof, test, diag = apply_action_to_pair(
            before_oof,
            before_test,
            src_oof,
            src_test,
            anchor_oof,
            anchor_test,
            action,
        )
        diagnostics.append({"action_key": action.key, "target": action.target, "source": action.source, **diag})
    return oof, test, diagnostics


def target_action_row(
    y: pd.DataFrame,
    folds: np.ndarray,
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    submitted_test: pd.DataFrame | None,
    action: Action,
    src_pair: tuple[pd.DataFrame, pd.DataFrame],
) -> dict:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    src_oof, src_test = src_pair
    poof, ptest, diag = apply_action_to_pair(
        base_oof,
        base_test,
        src_oof,
        src_test,
        anchor_oof,
        anchor_test,
        action,
    )
    target = action.target
    base_full = target_loss(y, base_oof, target, full_mask)
    base_last = target_loss(y, base_oof, target, last_mask)
    anchor_full = target_loss(y, anchor_oof, target, full_mask)
    anchor_last = target_loss(y, anchor_oof, target, last_mask)
    full = target_loss(y, poof, target, full_mask)
    last = target_loss(y, poof, target, last_mask)
    delta = ptest[target].values - anchor_test[target].values
    align = bad_alignment_risk(target, ptest[target].values, anchor_test[target].values, submitted_test)
    return {
        "action_key": action.key,
        "target": target,
        "source": action.source,
        "mode": action.mode,
        "alpha": action.alpha,
        "cap": action.cap,
        "anchor_ref": action.anchor_ref,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_base": full - base_full,
        "last_delta_vs_base": last - base_last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "test_abs_delta_mean": float(np.mean(np.abs(delta))),
        "test_mean_delta": float(np.mean(delta)),
        "test_up_rate": float(np.mean(delta > 1e-12)),
        "test_down_rate": float(np.mean(delta < -1e-12)),
        "public_risk": public_direction_risk(target, delta),
        "bad_alignment": align,
        **diag,
    }


def candidate_row(
    name: str,
    y: pd.DataFrame,
    folds: np.ndarray,
    anchor_oof: pd.DataFrame,
    anchor_test: pd.DataFrame,
    submitted_test: pd.DataFrame | None,
    oof: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    full_mask = np.ones(len(y), dtype=bool)
    last_mask = folds == (C.N_SPLITS - 1)
    losses = fold_losses(y, oof, folds)
    anchor_full = mean_loss(y, anchor_oof, full_mask)
    anchor_last = mean_loss(y, anchor_oof, last_mask)
    full = mean_loss(y, oof, full_mask)
    last = mean_loss(y, oof, last_mask)
    risk = 0.0
    align = 0.0
    for target in TARGETS:
        delta = test[target].values - anchor_test[target].values
        risk += public_direction_risk(target, delta)
        align += bad_alignment_risk(target, test[target].values, anchor_test[target].values, submitted_test)
    risk /= len(TARGETS)
    align /= len(TARGETS)
    move = float(np.mean(np.abs(test[TARGETS].values - anchor_test[TARGETS].values)))
    tail3_worst = float(max(losses[-3:]))
    tail3_mean = float(np.mean(losses[-3:]))
    return {
        "candidate": name,
        "full_logloss": full,
        "last_logloss": last,
        "full_delta_vs_anchor": full - anchor_full,
        "last_delta_vs_anchor": last - anchor_last,
        "fold_std": float(np.std(losses)),
        "tail3_mean": tail3_mean,
        "tail3_worst": tail3_worst,
        "test_abs_delta_mean_vs_anchor": move,
        "public_direction_risk": risk,
        "bad_submission_alignment": align,
        "selector_score": (
            last
            + 0.45 * max(0.0, full - anchor_full)
            + 2.8 * risk
            + 8.0 * align
            + 0.06 * move
            + 0.08 * max(0.0, tail3_worst - tail3_mean)
        ),
        **{f"fold{i}_logloss": v for i, v in enumerate(losses)},
    }


def action_grid(lgbm_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> list[tuple[Action, tuple[pd.DataFrame, pd.DataFrame]]]:
    out: list[tuple[Action, tuple[pd.DataFrame, pd.DataFrame]]] = []
    for source_name, pair in lgbm_sources.items():
        stem = source_name.split("/", 1)[-1]
        for mode in ["logit", "prob"]:
            if stem.startswith("target_weighted_single_full"):
                q2_alphas = [0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
                s2_alphas = [0.05, 0.08, 0.12, 0.18, 0.25]
                q3_alphas = [0.03, 0.05, 0.08]
            elif stem.startswith("target_weighted_single_composite"):
                q2_alphas = [0.08, 0.12, 0.18, 0.25, 0.35]
                s2_alphas = [0.03, 0.05, 0.08, 0.12, 0.18]
                q3_alphas = [0.03, 0.05]
            else:
                q2_alphas = [0.05, 0.08, 0.12, 0.18, 0.25]
                s2_alphas = [0.03, 0.05, 0.08, 0.12]
                q3_alphas = [0.03, 0.05]
            for anchor_ref in ["base", "anchor"]:
                for alpha in q2_alphas:
                    for cap in [0.004, 0.006, 0.008, 0.010, 0.014, 0.018, 0.024]:
                        action = Action("Q2", source_name, mode, alpha, cap, anchor_ref)
                        out.append((action, pair))
                for alpha in s2_alphas:
                    for cap in [0.004, 0.006, 0.008, 0.010, 0.014, 0.018]:
                        action = Action("S2", source_name, mode, alpha, cap, anchor_ref)
                        out.append((action, pair))
                for alpha in q3_alphas:
                    for cap in [0.003, 0.005, 0.008]:
                        action = Action("Q3", source_name, mode, alpha, cap, anchor_ref)
                        out.append((action, pair))
    return out


def filter_actions(action_scores: pd.DataFrame, profile: str) -> pd.DataFrame:
    rows = action_scores.copy()
    if profile == "tight":
        return rows[
            (rows["last_delta_vs_base"] <= -0.00015)
            & (rows["full_delta_vs_base"] <= 0.0015)
            & (rows["public_risk"] <= rows["target"].map({"Q2": 0.0045, "S2": 0.0045, "Q3": 0.0020}).fillna(0.0020))
            & (rows["test_abs_delta_mean"] <= rows["cap"] + 1e-12)
        ].copy()
    if profile == "balanced":
        return rows[
            (rows["last_delta_vs_base"] <= -0.00010)
            & (rows["full_delta_vs_base"] <= 0.0030)
            & (rows["public_risk"] <= rows["target"].map({"Q2": 0.0075, "S2": 0.0070, "Q3": 0.0030}).fillna(0.0030))
            & (rows["test_abs_delta_mean"] <= rows["cap"] + 1e-12)
        ].copy()
    if profile == "q2_force":
        return rows[
            (rows["target"].eq("Q2"))
            & (rows["last_delta_vs_base"] <= -0.0010)
            & (rows["full_delta_vs_base"] <= 0.0040)
            & (rows["test_abs_delta_mean"] <= rows["cap"] + 1e-12)
            & (rows["public_risk"] <= 0.0100)
        ].copy()
    raise ValueError(f"unknown profile: {profile}")


def choose_best_actions(action_scores: pd.DataFrame, profile: str, per_target: int) -> dict[str, list[str]]:
    eligible = filter_actions(action_scores, profile)
    if eligible.empty:
        return {target: [] for target in ["Q2", "S2", "Q3"]}
    eligible["action_selector"] = (
        eligible["last_delta_vs_base"]
        + 0.45 * np.maximum(0.0, eligible["full_delta_vs_base"])
        + 2.5 * eligible["public_risk"]
        + 6.5 * eligible["bad_alignment"]
        + 0.04 * eligible["test_abs_delta_mean"]
    )
    out: dict[str, list[str]] = {}
    for target in ["Q2", "S2", "Q3"]:
        tg = eligible[eligible["target"].eq(target)].copy()
        out[target] = tg.sort_values(["action_selector", "last_delta_vs_base", "full_delta_vs_base"])["action_key"].head(per_target).tolist()
    return out


def build_candidate_specs(best_by_profile: dict[str, dict[str, list[str]]]) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    for profile, by_target in best_by_profile.items():
        for target, keys in by_target.items():
            for i, key in enumerate(keys[:8]):
                specs.append((f"{profile}_{target}_only_{i}", [key]))
        q2_keys = by_target.get("Q2", [])[:8]
        s2_keys = by_target.get("S2", [])[:8]
        q3_keys = by_target.get("Q3", [])[:5]
        for i, q2 in enumerate(q2_keys[:6]):
            for j, s2 in enumerate(s2_keys[:6]):
                specs.append((f"{profile}_Q2S2_{i}_{j}", [q2, s2]))
        for i, q2 in enumerate(q2_keys[:5]):
            for j, s2 in enumerate(s2_keys[:5]):
                for k, q3 in enumerate(q3_keys[:3]):
                    specs.append((f"{profile}_Q2S2Q3_{i}_{j}_{k}", [q2, s2, q3]))
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dirs", nargs="*", default=[
        "research/public_aware_stack_blend_20260622",
        "research/public_aware_stack_blend_with_lgbm_source_20260622",
    ])
    p.add_argument("--lgbm-dir", default="research/target_weighted_single_model_lgbm_20260622")
    p.add_argument("--submitted", default="submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv")
    p.add_argument("--output-dir", default="research/guarded_lgbm_integration_20260623")
    p.add_argument("--submission-dir", default="submissions/guarded_lgbm_integration_20260623")
    p.add_argument("--base-top-n", type=int, default=10)
    p.add_argument("--per-target-actions", type=int, default=12)
    p.add_argument("--save-top", type=int, default=16)
    p.add_argument("--save-top-each", type=int, default=12)
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
    submitted_test = read_submission(rel(args.submitted))

    log("Loading base and LGBM sources")
    base_pool = load_base_pool(args.base_dirs, args.base_top_n)
    lgbm_sources = load_lgbm_sources(args.lgbm_dir)
    if not base_pool:
        raise RuntimeError("no base source pairs found")
    if not lgbm_sources:
        raise RuntimeError("no target-weighted LGBM source pairs found")
    log(f"Loaded {len(base_pool)} base sources and {len(lgbm_sources)} LGBM sources")

    action_pairs = action_grid(lgbm_sources)
    action_by_key = {action.key: (action, pair) for action, pair in action_pairs}
    candidate_pool: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float | str]]]] = {}
    action_rows = []

    log("Scoring guarded target actions")
    for base_name, (base_oof, base_test) in base_pool.items():
        base_short = safe_name(base_name)
        candidate_pool[f"base__{base_short}"] = (base_oof, base_test, [])
        for action, pair in action_pairs:
            row = target_action_row(
                ytr,
                folds,
                base_oof,
                base_test,
                anchor_oof,
                anchor_test,
                submitted_test,
                action,
                pair,
            )
            row["base"] = base_name
            row["base_safe"] = base_short
            action_rows.append(row)

    action_scores = pd.DataFrame(action_rows)
    action_scores.to_csv(out_dir / "target_action_scores.csv", index=False)

    log("Building target-action combinations")
    combo_rows = []
    all_choice_rows = []
    for base_name, (base_oof, base_test) in base_pool.items():
        base_short = safe_name(base_name)
        base_actions = action_scores[action_scores["base"].eq(base_name)].copy()
        best_by_profile = {
            profile: choose_best_actions(base_actions, profile, args.per_target_actions)
            for profile in ["tight", "balanced", "q2_force"]
        }
        for profile, by_target in best_by_profile.items():
            for target, keys in by_target.items():
                for rank, key in enumerate(keys):
                    selected = base_actions[base_actions["action_key"].eq(key)].iloc[0].to_dict()
                    all_choice_rows.append({
                        "base": base_name,
                        "profile": profile,
                        "target": target,
                        "rank": rank,
                        **selected,
                    })

        specs = build_candidate_specs(best_by_profile)
        for spec_name, keys in specs:
            pairs = [action_by_key[key] for key in keys if key in action_by_key]
            if not pairs:
                continue
            poof, ptest, diagnostics = apply_actions(base_oof, base_test, pairs, anchor_oof, anchor_test)
            name = f"{base_short}__{spec_name}"
            candidate_pool[name] = (poof, ptest, diagnostics)
            row = candidate_row(name, ytr, folds, anchor_oof, anchor_test, submitted_test, poof, ptest)
            row["base"] = base_name
            row["actions"] = json.dumps(keys)
            combo_rows.append(row)

    log("Scoring all candidates")
    full_rows = []
    for name, (poof, ptest, diagnostics) in candidate_pool.items():
        row = candidate_row(name, ytr, folds, anchor_oof, anchor_test, submitted_test, poof, ptest)
        row["base"] = name.split("__", 1)[-1] if name.startswith("base__") else ""
        row["actions"] = json.dumps([d["action_key"] for d in diagnostics])
        full_rows.append(row)

    scores = pd.DataFrame(full_rows).sort_values(["selector_score", "last_logloss", "full_logloss"]).reset_index(drop=True)
    combo_scores = pd.DataFrame(combo_rows).sort_values(["selector_score", "last_logloss", "full_logloss"]).reset_index(drop=True)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    combo_scores.to_csv(out_dir / "combo_candidate_scores.csv", index=False)
    if all_choice_rows:
        pd.DataFrame(all_choice_rows).to_csv(out_dir / "selected_target_actions.csv", index=False)

    save_rows = pd.concat(
        [
            scores.head(args.save_top),
            scores.sort_values(["last_logloss", "full_logloss"]).head(args.save_top_each),
            scores.sort_values(["full_logloss", "last_logloss"]).head(args.save_top_each),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("candidate")

    saved = []
    for _, row in save_rows.iterrows():
        name = str(row["candidate"])
        poof, ptest, _ = candidate_pool[name]
        safe = safe_name(name)
        write_prediction(out_dir / f"{safe}_oof.csv", mtr, poof, ytr)
        write_prediction(out_dir / f"{safe}_test_pred.csv", mte, ptest)
        sub_path = sub_dir / f"{safe}_last{row['last_logloss']:.6f}_full{row['full_logloss']:.6f}.csv"
        write_submission(sub_path, mte, ptest)
        saved.append(str(sub_path.relative_to(ROOT)))

    report = {
        "base_dirs": args.base_dirs,
        "lgbm_dir": args.lgbm_dir,
        "submitted": args.submitted,
        "best": scores.iloc[0].to_dict() if not scores.empty else {},
        "saved": saved,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Guarded LGBM integration candidates ===")
    cols = [
        "candidate",
        "selector_score",
        "full_logloss",
        "last_logloss",
        "full_delta_vs_anchor",
        "last_delta_vs_anchor",
        "fold_std",
        "tail3_worst",
        "test_abs_delta_mean_vs_anchor",
        "public_direction_risk",
        "bad_submission_alignment",
    ]
    print(scores[cols].head(30).to_string(index=False))
    print("\n=== Best selected target actions ===")
    if all_choice_rows:
        choices = pd.DataFrame(all_choice_rows)
        choice_cols = [
            "base",
            "profile",
            "target",
            "rank",
            "source",
            "mode",
            "alpha",
            "cap",
            "anchor_ref",
            "full_delta_vs_base",
            "last_delta_vs_base",
            "test_abs_delta_mean",
            "public_risk",
            "bad_alignment",
        ]
        print(choices[choice_cols].head(80).to_string(index=False))
    print("\n=== Saved submissions ===")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
