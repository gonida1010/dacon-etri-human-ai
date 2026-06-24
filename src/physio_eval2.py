"""결정적 실험: 실제 전체 피처 모델에 physio 를 '추가'했을 때 한계 기여 측정.

build_dataset 의 전체 피처(+subject prior)로 per-target LGBM(멀티시드) OOF 를,
physio 포함/미포함 두 번 돌려 full/last logloss 비교. 같은 폴드/시드.
실행: python -m src.physio_eval2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .cv import subject_time_blocked_folds
from .build_dataset import build_dataset
from .physio_features import build_physio_features

CLIP = 1e-3
SEEDS = [42, 7, 2024]
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=1.0, lambda_l2=1.0, verbosity=-1)


def ll(y, p):
    return log_loss(y, np.clip(p, CLIP, 1 - CLIP), labels=[0, 1])


def smoothed_prior(y, tr_idx, subj, gmean, smooth=8.0):
    d = pd.DataFrame({"s": subj[tr_idx], "y": y[tr_idx]})
    g = d.groupby("s")["y"].agg(["sum", "count"])
    pmap = ((g["sum"] + smooth * gmean) / (g["count"] + smooth)).to_dict()
    return np.array([pmap.get(s, gmean) for s in subj])


def run(Xtr, ytr, meta, folds, extra=None):
    subj = meta["subject_id"].values
    last = folds == 4
    base_cols = [c for c in Xtr.columns if c != "subject_id"]
    res = {}
    for t in C.TARGET_COLS:
        y = ytr[t].values.astype(float)
        gm = y.mean()
        oof = np.zeros(len(Xtr))
        for f in range(5):
            tr = np.where(folds != f)[0]
            va = np.where(folds == f)[0]
            prior = smoothed_prior(y, tr, subj, gm)
            Xf = Xtr[base_cols].copy()
            Xf["subj_prior"] = prior
            if extra is not None:
                for c in extra.columns:
                    Xf[c] = extra[c].values
            pr = np.zeros(len(va))
            for sd in SEEDS:
                m = lgb.train(dict(PARAMS, seed=sd),
                              lgb.Dataset(Xf.iloc[tr], label=y[tr]), num_boost_round=300)
                pr += m.predict(Xf.iloc[va])
            oof[va] = pr / len(SEEDS)
        res[t] = (ll(y, oof), ll(y[last], oof[last]), oof)
    return res


def main():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=5)
    physio = build_physio_features(use_cache=True)
    physio["sleep_date"] = pd.to_datetime(physio["sleep_date"])
    pcols = [c for c in physio.columns if c not in ("subject_id", "sleep_date")]
    extra = mtr.merge(physio, on=["subject_id", "sleep_date"], how="left")[pcols].reset_index(drop=True)
    extra.columns = [f"PH_{c}" for c in extra.columns]

    print("Running WITHOUT physio ...", flush=True)
    r0 = run(Xtr, ytr, mtr, folds, extra=None)
    print("Running WITH physio ...", flush=True)
    r1 = run(Xtr, ytr, mtr, folds, extra=extra)

    print(f"\n{'tgt':>3} | {'noP_full':>8} {'noP_last':>8} | {'wP_full':>8} {'wP_last':>8} | "
          f"{'d_full':>7} {'d_last':>7}")
    print("-" * 70)
    df = dl = wf = wl = 0
    for t in C.TARGET_COLS:
        a = r0[t]; b = r1[t]
        df += a[0]; dl += a[1]; wf += b[0]; wl += b[1]
        print(f"{t:>3} | {a[0]:8.4f} {a[1]:8.4f} | {b[0]:8.4f} {b[1]:8.4f} | "
              f"{b[0]-a[0]:+7.4f} {b[1]-a[1]:+7.4f}")
    print("-" * 70)
    print(f"{'AVG':>3} | {df/7:8.4f} {dl/7:8.4f} | {wf/7:8.4f} {wl/7:8.4f} | "
          f"{(wf-df)/7:+7.4f} {(wl-dl)/7:+7.4f}")


if __name__ == "__main__":
    main()
