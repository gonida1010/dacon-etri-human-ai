"""최종 강건 제출본: global_raw_full_a0.80(전타깃 개선+안정) 기반,
physio(S1/S2/S4)를 얹어 full-OOF 가 개선되고 최악폴드가 나빠지지 않을 때만 채택.

평가산식=Average Log-Loss, Private=test 100% → full-OOF + 폴드안정성을 private 대리로 사용.
실행: python -m src.final_robust_submission
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
SEEDS = [42, 7, 2024, 11, 99]
S_TARGETS = ["S1", "S2", "S4"]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_child_samples=20,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=1.0, lambda_l2=2.0, verbosity=-1)
RAW = ("research/raw_timeline_guarded_blend_20260624_v2/"
       "global_raw_full_logit_a0p80_last0.578422_full0.583519")


def ll(y, p):
    return log_loss(y, np.clip(p, CLIP, 1 - CLIP), labels=[0, 1])


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smoothed_prior_map(y, subj, gm, smooth=8.0):
    d = pd.DataFrame({"s": subj, "y": y})
    g = d.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + smooth * gm) / (g["count"] + smooth)).to_dict()


def main():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=5)
    subj_tr = mtr["subject_id"].values
    physio = build_physio_features(use_cache=True)
    physio["sleep_date"] = pd.to_datetime(physio["sleep_date"])
    pcols = [c for c in physio.columns if c not in ("subject_id", "sleep_date")]
    Ptr = mtr.merge(physio, on=["subject_id", "sleep_date"], how="left")[pcols].reset_index(drop=True).astype(float)
    Pte = mte.merge(physio, on=["subject_id", "sleep_date"], how="left")[pcols].reset_index(drop=True).astype(float)

    raw_oof = mtr.merge(pd.read_csv(RAW + "_oof.csv", parse_dates=["sleep_date"]),
                        on=["subject_id", "sleep_date"], how="left")
    raw_test = mte.merge(pd.read_csv(RAW + "_test_pred.csv", parse_dates=["sleep_date"]),
                         on=["subject_id", "sleep_date"], how="left")

    final_oof = {t: raw_oof[t].values.copy() for t in C.TARGET_COLS}
    final_test = {t: raw_test[t].values.copy() for t in C.TARGET_COLS}

    def fold_stats(oof_map):
        per = []
        for f in range(5):
            m = folds == f
            per.append(np.mean([ll(ytr[t].values[m], oof_map[t][m]) for t in C.TARGET_COLS]))
        full = np.mean([ll(ytr[t].values, oof_map[t]) for t in C.TARGET_COLS])
        return full, np.std(per), max(per)

    f0, sd0, w0 = fold_stats(final_oof)
    print(f"RAW base candidate: full={f0:.4f} foldstd={sd0:.4f} worst={w0:.4f}\n")

    print(f"{'tgt':>3} | {'raw_full':>8} {'phys_full':>9} | {'a*':>4} {'new_full':>8} | {'d_full':>7} adopted")
    for t in S_TARGETS:
        y = ytr[t].values.astype(float)
        gm = y.mean()
        oof = np.zeros(len(Xtr))
        for f in range(5):
            tr = np.where(folds != f)[0]
            va = np.where(folds == f)[0]
            pm = smoothed_prior_map(y[tr], subj_tr[tr], gm)
            prior = np.array([pm.get(s, gm) for s in subj_tr])
            Xf = Ptr.copy(); Xf["subj_prior"] = prior
            pr = np.zeros(len(va))
            for sd in SEEDS:
                m = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(Xf.iloc[tr], label=y[tr]), num_boost_round=160)
                pr += m.predict(Xf.iloc[va])
            oof[va] = pr / len(SEEDS)
        # test preds (train on all)
        pm = smoothed_prior_map(y, subj_tr, gm)
        prior_all = np.array([pm.get(s, gm) for s in subj_tr])
        prior_te = np.array([pm.get(s, gm) for s in mte["subject_id"].values])
        Xa = Ptr.copy(); Xa["subj_prior"] = prior_all
        Xb = Pte.copy(); Xb["subj_prior"] = prior_te
        tp = np.zeros(len(Xte))
        for sd in SEEDS:
            m = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(Xa, label=y), num_boost_round=160)
            tp += m.predict(Xb)
        tp /= len(SEEDS)

        rp = raw_oof[t].values
        raw_full = ll(y, rp); phys_full = ll(y, oof)
        # choose alpha minimizing full, but candidate must not raise overall worst-fold
        best = (raw_full, 0.0, None)
        for a in np.linspace(0, 0.5, 26):
            bl = sigmoid((1 - a) * logit(rp) + a * logit(oof))
            cand = dict(final_oof); cand[t] = bl
            _, _, w = fold_stats(cand)
            full_t = ll(y, bl)
            if w <= w0 + 1e-4 and full_t < best[0]:
                best = (full_t, a, bl)
        a, bl = best[1], best[2]
        adopt = a > 0
        if adopt:
            final_oof[t] = bl
            final_test[t] = sigmoid((1 - a) * logit(raw_test[t].values) + a * logit(tp))
        print(f"{t:>3} | {raw_full:8.4f} {phys_full:9.4f} | {a:4.2f} {ll(y,final_oof[t]):8.4f} | "
              f"{ll(y,final_oof[t])-raw_full:+7.4f} {adopt}")

    f1, sd1, w1 = fold_stats(final_oof)
    print(f"\nFINAL: full={f1:.4f} foldstd={sd1:.4f} worst={w1:.4f}  (raw was {f0:.4f}/{sd0:.4f}/{w0:.4f})")

    sub = mte[C.ID_COLS].copy()
    for t in C.TARGET_COLS:
        sub[t] = final_test[t]
    out = C.SUBMISSION_DIR / "FINAL_robust_20260624"
    out.mkdir(exist_ok=True)
    fn = out / f"final_robust_full{f1:.4f}_foldstd{sd1:.4f}_worst{w1:.4f}.csv"
    sub.to_csv(fn, index=False)
    print(f"wrote {fn}")
    print("test positivity:", {t: round(float((final_test[t] > 0.5).mean()), 3) for t in C.TARGET_COLS})


if __name__ == "__main__":
    main()
