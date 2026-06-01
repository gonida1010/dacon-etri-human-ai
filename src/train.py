"""정식 학습 파이프라인 (최종 제출 생성).

구성:
 1) 피처: 일일 윈도우 통계 + 수면구간 탐지 + 중첩센서 + 피험자 z-score + 캘린더
 2) 타깃별 LGBM + 누수 없는 OOF 피험자 prior 피처
 3) 멀티시드 배깅(폴드/모델 시드)으로 OOF·테스트 예측 안정화
 4) 타깃별 (모델 vs prior) OOF 블렌드 가중 탐색
 5) 평균 Log-Loss(대회 산식) 보고 + 제출 CSV 저장

실행: python -m src.train
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

from . import config as C
from .build_dataset import build_dataset
from .cv import subject_time_blocked_folds

CLIP = C.PROB_CLIP
N_SPLITS = C.N_SPLITS
SEEDS = C.SEEDS
SMOOTH = C.PRIOR_SMOOTH

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=25,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=1.0, verbosity=-1,
)


def clip(p):
    return np.clip(p, CLIP, 1 - CLIP)


def sharpen(p):
    """확률 샤프닝(온도 T). T<1: 0/1 쪽으로 더 극단(자신감↑). T=1: 그대로."""
    T = C.SHARPEN_T
    if T == 1.0:
        return p
    z = np.log(clip(p) / (1 - clip(p))) / T
    return 1.0 / (1.0 + np.exp(-z))


def avg_logloss(y_true, p):
    per = {t: log_loss(y_true[t].values, clip(p[t].values), labels=[0, 1]) for t in C.TARGET_COLS}
    return float(np.mean(list(per.values()))), per


SMOOTH_GRID = [0.5, 2, 4, 8, 16, 32, 64, 128]


def smoothed_mean(y, idx, subj, gmean, sm=SMOOTH):
    df = pd.DataFrame({"s": subj[idx], "y": y[idx]})
    g = df.groupby("s")["y"].agg(["sum", "count"])
    return ((g["sum"] + sm * gmean) / (g["count"] + sm)).to_dict()


def best_smooth(y, folds, subj):
    """타깃별 최적 prior 스무딩(개인평균 신뢰도). 지속성↑타깃은 작게, 노이즈 타깃은 크게.

    Q2/Q3처럼 개인평균이 무의미하면 큰 SMOOTH(전역평균 쪽), S1/S3처럼 지속적이면 작게.
    """
    best_sm, best_ll = SMOOTH, 1e9
    for sm in SMOOTH_GRID:
        oof = np.zeros(len(y))
        for f in range(N_SPLITS):
            tr = np.where(folds != f)[0]; va = np.where(folds == f)[0]
            gm = y[tr].mean()
            pmap = smoothed_mean(y, tr, subj, gm, sm)
            oof[va] = [pmap.get(s, gm) for s in subj[va]]
        v = log_loss(y, clip(oof), labels=[0, 1])
        if v < best_ll:
            best_ll, best_sm = v, sm
    return best_sm


def train():
    Xtr, ytr, Xte, mtr, mte, feat_cols = build_dataset(use_cache=True)
    subj_tr, subj_te = mtr["subject_id"].values, mte["subject_id"].values
    cat = ["subject_id"]

    oof_model = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    oof_prior = pd.DataFrame(0.0, index=Xtr.index, columns=C.TARGET_COLS)
    test_model = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)
    test_prior = pd.DataFrame(0.0, index=Xte.index, columns=C.TARGET_COLS)

    folds = subject_time_blocked_folds(mtr, n_splits=N_SPLITS)  # 정직한 시간블록(결정적)
    for t in C.TARGET_COLS:
        y = ytr[t].values
        gmean = y.mean()
        sm_t = best_smooth(y, folds, subj_tr)   # 타깃별 최적 스무딩
        for seed in SEEDS:
            params = {**LGB_PARAMS, "seed": seed}
            for f in range(N_SPLITS):
                tr_idx = np.where(folds != f)[0]
                va_idx = np.where(folds == f)[0]
                pmap = smoothed_mean(y, tr_idx, subj_tr, gmean, sm_t)
                prior_tr = np.array([pmap.get(s, gmean) for s in subj_tr])
                prior_te = np.array([pmap.get(s, gmean) for s in subj_te])

                Xtr_f = Xtr.copy(); Xtr_f["subj_prior"] = prior_tr
                Xte_f = Xte.copy(); Xte_f["subj_prior"] = prior_te
                dtr = lgb.Dataset(Xtr_f.iloc[tr_idx], label=y[tr_idx],
                                  categorical_feature=cat, free_raw_data=False)
                dva = lgb.Dataset(Xtr_f.iloc[va_idx], label=y[va_idx],
                                  categorical_feature=cat, free_raw_data=False)
                m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(80, verbose=False)])
                col = oof_model.columns.get_loc(t)
                oof_model.iloc[va_idx, col] += m.predict(Xtr_f.iloc[va_idx]) / len(SEEDS)
                oof_prior.iloc[va_idx, col] += prior_tr[va_idx] / len(SEEDS)
                test_model[t] += m.predict(Xte_f) / (len(SEEDS) * N_SPLITS)
                test_prior[t] += prior_te / (len(SEEDS) * N_SPLITS)

    weights, blended = {}, pd.DataFrame(index=Xtr.index, columns=C.TARGET_COLS, dtype=float)
    for t in C.TARGET_COLS:
        best_w, best_ll = 0.0, 1e9
        for w in np.linspace(0, 1, 41):
            ll = log_loss(ytr[t].values, clip(w * oof_model[t] + (1 - w) * oof_prior[t]), labels=[0, 1])
            if ll < best_ll:
                best_ll, best_w = ll, w
        best_w = max(best_w, C.MODEL_WEIGHT_FLOOR)   # 모델 비중 하한(과감 실험용)
        weights[t] = best_w
        blended[t] = sharpen(best_w * oof_model[t] + (1 - best_w) * oof_prior[t])

    _, model_per = avg_logloss(ytr, oof_model)
    prior_score, _ = avg_logloss(ytr, oof_prior)
    model_score, _ = avg_logloss(ytr, oof_model)
    blend_score, blend_per = avg_logloss(ytr, blended)

    print("\n=== OOF Average Log-Loss (multi-seed bagged) ===")
    print(f"{'tgt':4s} {'model':>8s} {'w*':>5s} {'blend':>8s}")
    for t in C.TARGET_COLS:
        print(f"{t:4s} {model_per[t]:8.4f} {weights[t]:5.2f} {blend_per[t]:8.4f}")
    # 정직한 헤드라인: 마지막 시간블록(=test와 동일 regime)
    last = folds == (N_SPLITS - 1)
    blend_last = float(np.mean([
        log_loss(ytr[t].values[last], clip(blended[t].values[last]), labels=[0, 1])
        for t in C.TARGET_COLS]))
    print("  ----")
    print(f"  prior-only      : {prior_score:.4f}")
    print(f"  model-only      : {model_score:.4f}")
    print(f"  BLEND full      : {blend_score:.4f}")
    print(f"  BLEND last-block: {blend_last:.4f}   <- 진짜 점수(LB와 일치, 낮을수록 좋음)")

    # 진단용 3종 제출 생성: 실제 공개LB로 'CV가 맞는지' 삼각측량
    def save(name, getter):
        s = mte.copy()
        for t in C.TARGET_COLS:
            s[t] = clip(getter(t))
        p = C.SUBMISSION_DIR / name
        s.to_csv(p, index=False)
        print(f"  saved: {p.name}")

    print("\n=== 진단용 제출 3종 (이 순서로 공개LB에 올리고 점수 알려주세요) ===")
    save(f"A_blend_last{blend_last:.4f}.csv",
         lambda t: sharpen(weights[t] * test_model[t].values + (1 - weights[t]) * test_prior[t].values))
    save("B_model_only.csv", lambda t: test_model[t].values)      # 모델 단독(공격적)
    save("C_prior_only.csv", lambda t: test_prior[t].values)      # 피험자 prior 단독(보수적)
    print("\n해석: A<B<C 면 우리 전략 타당. B가 A보다 좋으면 prior가 해로움(덜 섞어야).")
    print("      C가 의외로 좋으면 모델 자체가 과적합. 셋 다 0.60근처면 데이터 천장 확정.")
    return blend_score


if __name__ == "__main__":
    train()
