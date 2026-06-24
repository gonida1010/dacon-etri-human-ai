"""보수적 제출 후보: public-best 기반, S1/S2/S4 에만 physio-증강 모델을 블렌딩.

physio 는 객관 수면 메트릭(S1/S2/S4)에서만 도움(검증됨) → 거기에만 적용, 나머지는 public-best 유지.
블렌드 비율은 full-OOF(=public 의 더 나은 대리) 기준으로 선택, last/안정성으로 검증.
실행: python -m src.physio_submission
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
PHYSIO_TARGETS = ["S1", "S2", "S4"]
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              lambda_l1=1.0, lambda_l2=1.0, verbosity=-1)
BASE = ("research/guarded_lgbm_integration_20260623_v2/"
        "public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5")


def ll(y, p):
    return log_loss(y, np.clip(p, CLIP, 1 - CLIP), labels=[0, 1])


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smoothed_prior(y, tr_idx, subj, gmean, smooth=8.0):
    d = pd.DataFrame({"s": subj[tr_idx], "y": y[tr_idx]})
    g = d.groupby("s")["y"].agg(["sum", "count"])
    pmap = ((g["sum"] + smooth * gmean) / (g["count"] + smooth)).to_dict()
    return np.array([pmap.get(s, gmean) for s in subj])


def main():
    Xtr, ytr, Xte, mtr, mte, _ = build_dataset(use_cache=True)
    folds = subject_time_blocked_folds(mtr, n_splits=5)
    last = folds == 4
    subj_tr = mtr["subject_id"].values

    physio = build_physio_features(use_cache=True)
    physio["sleep_date"] = pd.to_datetime(physio["sleep_date"])
    pcols = [c for c in physio.columns if c not in ("subject_id", "sleep_date")]
    ph_tr = mtr.merge(physio, on=["subject_id", "sleep_date"], how="left")[pcols].reset_index(drop=True)
    ph_te = mte.merge(physio, on=["subject_id", "sleep_date"], how="left")[pcols].reset_index(drop=True)
    ph_tr.columns = [f"PH_{c}" for c in ph_tr.columns]
    ph_te.columns = [f"PH_{c}" for c in ph_te.columns]

    base_cols = [c for c in Xtr.columns if c != "subject_id"]

    # public-best OOF + test, aligned
    base_oof = pd.read_csv(BASE + "_oof.csv", parse_dates=["sleep_date"])
    base_oof = mtr.merge(base_oof, on=["subject_id", "sleep_date"], how="left")
    base_test = pd.read_csv(BASE + "_test_pred.csv", parse_dates=["sleep_date"])
    base_test = mte.merge(base_test, on=["subject_id", "sleep_date"], how="left")

    final_oof = {t: base_oof[t].values.copy() for t in C.TARGET_COLS}
    final_test = {t: base_test[t].values.copy() for t in C.TARGET_COLS}

    print(f"{'tgt':>3} | {'base_full':>9} {'phys_full':>9} | {'a*':>4} {'bl_full':>8} {'bl_last':>8} "
          f"{'tail_w':>7} | {'d_full':>7} | test_pos b->bl")
    print("-" * 100)
    for t in PHYSIO_TARGETS:
        y = ytr[t].values.astype(float)
        gm = y.mean()
        oof = np.zeros(len(Xtr))
        test_acc = np.zeros(len(Xte))
        # OOF
        for f in range(5):
            tr = np.where(folds != f)[0]
            va = np.where(folds == f)[0]
            prior = smoothed_prior(y, tr, subj_tr, gm)
            Xf = Xtr[base_cols].copy()
            Xf["subj_prior"] = prior
            for c in ph_tr.columns:
                Xf[c] = ph_tr[c].values
            pr = np.zeros(len(va))
            for sd in SEEDS:
                m = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(Xf.iloc[tr], label=y[tr]),
                              num_boost_round=300)
                pr += m.predict(Xf.iloc[va])
            oof[va] = pr / len(SEEDS)
        # full-train -> test
        prior_all = smoothed_prior(y, np.arange(len(Xtr)), subj_tr, gm)
        prior_te = smoothed_prior(y, np.arange(len(Xtr)), subj_tr, gm)  # subject prior maps test by id
        # build test prior via subject map
        d = pd.DataFrame({"s": subj_tr, "y": y})
        g = d.groupby("s")["y"].agg(["sum", "count"])
        pmap = ((g["sum"] + 8.0 * gm) / (g["count"] + 8.0)).to_dict()
        prior_te = np.array([pmap.get(s, gm) for s in mte["subject_id"].values])
        Xa = Xtr[base_cols].copy(); Xa["subj_prior"] = prior_all
        Xb = Xte[base_cols].copy(); Xb["subj_prior"] = prior_te
        for c in ph_tr.columns:
            Xa[c] = ph_tr[c].values
            Xb[c] = ph_te[c].values
        for sd in SEEDS:
            m = lgb.train(dict(PARAMS, seed=sd), lgb.Dataset(Xa, label=y), num_boost_round=300)
            test_acc += m.predict(Xb)
        test_pred = test_acc / len(SEEDS)

        bp = base_oof[t].values
        base_full = ll(y, bp)
        phys_full = ll(y, oof)
        # choose alpha by full-OOF (public proxy), tie-break: not worsening last beyond +0.003
        best = (1e9, 0.0)
        for a in np.linspace(0, 0.6, 31):
            bl = sigmoid((1 - a) * logit(bp) + a * logit(oof))
            f_full = ll(y, bl)
            f_last = ll(y[last], bl[last])
            base_last = ll(y[last], bp[last])
            if f_last > base_last + 0.004:
                continue
            if f_full < best[0]:
                best = (f_full, a)
        a = best[1]
        bl_oof = sigmoid((1 - a) * logit(bp) + a * logit(oof))
        bl_test = sigmoid((1 - a) * logit(base_test[t].values) + a * logit(test_pred))
        final_oof[t] = bl_oof
        final_test[t] = bl_test
        # per-fold stability
        fold_lls = [ll(y[folds == f], bl_oof[folds == f]) for f in range(5)]
        tail_w = max(fold_lls)
        print(f"{t:>3} | {base_full:9.4f} {phys_full:9.4f} | {a:4.2f} {ll(y,bl_oof):8.4f} "
              f"{ll(y[last],bl_oof[last]):8.4f} {tail_w:7.4f} | {ll(y,bl_oof)-base_full:+7.4f} | "
              f"{(base_test[t].values>0.5).mean():.3f}->{(bl_test>0.5).mean():.3f}")

    # overall
    of = np.mean([ll(ytr[t].values, final_oof[t]) for t in C.TARGET_COLS])
    ol = np.mean([ll(ytr[t].values[last], final_oof[t][last]) for t in C.TARGET_COLS])
    bf = np.mean([ll(ytr[t].values, base_oof[t].values) for t in C.TARGET_COLS])
    bl_ = np.mean([ll(ytr[t].values[last], base_oof[t].values[last]) for t in C.TARGET_COLS])
    print("-" * 100)
    print(f"OVERALL  base full={bf:.4f} last={bl_:.4f}  ->  blend full={of:.4f} last={ol:.4f}  "
          f"(d_full={of-bf:+.4f}, d_last={ol-bl_:+.4f})")

    # write submission
    sub = mte[C.ID_COLS].copy()
    for t in C.TARGET_COLS:
        sub[t] = final_test[t]
    outdir = C.SUBMISSION_DIR / "physio_s_blend_20260624"
    outdir.mkdir(exist_ok=True)
    fn = outdir / f"physio_s124_blend_last{ol:.4f}_full{of:.4f}.csv"
    sub.to_csv(fn, index=False)
    print(f"\nwrote {fn}")
    print("test positivity (final):")
    print({t: round(float((final_test[t] > 0.5).mean()), 3) for t in C.TARGET_COLS})


if __name__ == "__main__":
    main()
