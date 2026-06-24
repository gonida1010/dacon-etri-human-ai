"""physio 피처 평가: 컴팩트 physio 전용 per-target 모델 OOF + public-best OOF 와 블렌딩.

목적: physio 신호가 (1) 단독으로 얼마의 logloss 를 내는지, (2) 기존 public-best 와
탈상관이라 블렌드 시 per-target logloss 를 낮추는지 정직하게 측정.

CV: subject_time_blocked_folds(5). 'last'=fold4(과거→미래 regime, test 유사).
실행: python -m src.physio_eval
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .cv import subject_time_blocked_folds
from .physio_features import build_physio_features

CLIP = 1e-3
SEEDS = [42, 7, 2024]
BASE_OOF = ("research/guarded_lgbm_integration_20260623_v2/"
            "public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5_oof.csv")
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=1.0, lambda_l2=2.0, verbosity=-1)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def ll(y, p):
    return log_loss(y, clip(p), labels=[0, 1])


def logit(p):
    p = clip(p)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smoothed_prior(y, tr_idx, subj, gmean, smooth=8.0):
    df = pd.DataFrame({"s": subj[tr_idx], "y": y[tr_idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    pmap = ((g["sum"] + smooth * gmean) / (g["count"] + smooth)).to_dict()
    return np.array([pmap.get(s, gmean) for s in subj])


def main():
    train = pd.read_csv(C.TRAIN_CSV, parse_dates=["sleep_date", "lifelog_date"])
    physio = build_physio_features(use_cache=True)
    physio["sleep_date"] = pd.to_datetime(physio["sleep_date"])

    df = train.merge(physio, on=["subject_id", "sleep_date"], how="left")
    feat_cols = [c for c in physio.columns if c not in ("subject_id", "sleep_date")]
    print(f"rows={len(df)}  physio feats={len(feat_cols)}  "
          f"physio coverage={df[feat_cols[0]].notna().mean():.2f}")

    meta = df[["subject_id", "sleep_date"]].copy()
    folds = subject_time_blocked_folds(df, n_splits=5)
    last = folds == 4
    subj = df["subject_id"].values
    X = df[feat_cols].astype(float)

    # baseline public-best OOF, aligned by (subject_id, sleep_date)
    base = pd.read_csv(BASE_OOF, parse_dates=["sleep_date"])
    base = meta.merge(base, on=["subject_id", "sleep_date"], how="left")

    print(f"\n{'tgt':>3} | {'base_full':>9} {'base_last':>9} | {'phys_full':>9} {'phys_last':>9} | "
          f"{'a*':>4} {'bl_full':>8} {'bl_last':>8} | {'d_full':>7} {'d_last':>7}")
    print("-" * 105)
    agg = {k: [] for k in ["base_full", "base_last", "phys_full", "phys_last", "bl_full", "bl_last"]}
    blend_oof = {}
    for t in C.TARGET_COLS:
        y = df[t].values.astype(float)
        gm = y.mean()
        # physio standalone OOF (subject prior + physio model)
        oof = np.zeros(len(df))
        for f in range(5):
            tr = np.where(folds != f)[0]
            va = np.where(folds == f)[0]
            prior = smoothed_prior(y, tr, subj, gm)
            Xf = X.copy()
            Xf["subj_prior"] = prior
            preds = np.zeros(len(va))
            for sd in SEEDS:
                p = dict(PARAMS, seed=sd)
                m = lgb.train(p, lgb.Dataset(Xf.iloc[tr], label=y[tr]), num_boost_round=160)
                preds += m.predict(Xf.iloc[va])
            oof[va] = preds / len(SEEDS)

        bp = base[t].values
        base_full, base_last = ll(y, bp), ll(y[last], bp[last])
        phys_full, phys_last = ll(y, oof), ll(y[last], oof[last])

        # logit blend search vs base (minimize full+last avg, fold-safe-ish via global search)
        best = (1e9, 0.0, None)
        for a in np.linspace(0, 1, 41):
            bl = sigmoid((1 - a) * logit(bp) + a * logit(oof))
            score = 0.5 * ll(y, bl) + 0.5 * ll(y[last], bl[last])
            if score < best[0]:
                best = (score, a, bl)
        a, bl = best[1], best[2]
        bl_full, bl_last = ll(y, bl), ll(y[last], bl[last])
        blend_oof[t] = bl
        for k, v in [("base_full", base_full), ("base_last", base_last), ("phys_full", phys_full),
                     ("phys_last", phys_last), ("bl_full", bl_full), ("bl_last", bl_last)]:
            agg[k].append(v)
        print(f"{t:>3} | {base_full:9.4f} {base_last:9.4f} | {phys_full:9.4f} {phys_last:9.4f} | "
              f"{a:4.2f} {bl_full:8.4f} {bl_last:8.4f} | {bl_full-base_full:+7.4f} {bl_last-base_last:+7.4f}")
    print("-" * 105)
    print(f"{'AVG':>3} | {np.mean(agg['base_full']):9.4f} {np.mean(agg['base_last']):9.4f} | "
          f"{np.mean(agg['phys_full']):9.4f} {np.mean(agg['phys_last']):9.4f} | "
          f"     {np.mean(agg['bl_full']):8.4f} {np.mean(agg['bl_last']):8.4f} | "
          f"{np.mean(agg['bl_full'])-np.mean(agg['base_full']):+7.4f} "
          f"{np.mean(agg['bl_last'])-np.mean(agg['base_last']):+7.4f}")


if __name__ == "__main__":
    main()
