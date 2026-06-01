"""눈으로 신호를 찾는 EDA 시각화. 결과 PNG 는 eda/ 폴더에 저장.

핵심 질문: "각 타깃(0/1)을 실제로 구분하는 피처는 무엇인가?"
 - 피처별 단변량 판별력 = AUC. 0.5=무신호, 0.5에서 멀수록 신호 강함.
 - 피험자 효과를 제거한 *_zsubj(개인 대비 편차) 피처의 AUC = '개인 평균을 넘는 진짜 신호'.

실행: python -m src.eda    →  eda/*.png 생성 + 콘솔에 타깃별 상위 피처 출력
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 한글 폰트(맥). 없으면 무시.
for _f in ["AppleGothic", "Nanum Gothic", "Apple SD Gothic Neo"]:
    try:
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import roc_auc_score

from . import config as C
from .build_dataset import build_dataset, load_labels
from .cv import subject_time_blocked_folds

warnings.filterwarnings("ignore")
EDA_DIR = C.PROJECT_ROOT / "eda"
EDA_DIR.mkdir(exist_ok=True)


def feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """단변량 AUC. NaN 제거 후 두 클래스 모두 있어야 계산. 신호 없으면 0.5."""
    m = ~np.isnan(x)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return 0.5
    try:
        a = roc_auc_score(y[m], x[m])
    except ValueError:
        return 0.5
    return a


def _usable(s: pd.Series) -> bool:
    """스파이크/희소 피처 제외: 결측 40%↓ 이고 서로 다른 값이 충분히 많아야(연속적) 함."""
    x = s.values.astype(float)
    m = ~np.isnan(x)
    if m.mean() < 0.4:
        return False
    return np.unique(x[m]).size >= 12


def compute_auc_table(X: pd.DataFrame, y: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """피처별 단변량 AUC를 (1) 전체 train, (2) 마지막 시간블록(=test regime) 양쪽으로 계산.

    스파이크(z-score 인공물)·희소 피처는 제외. 'gen' = 두 AUC가 같은 방향이고 둘 다 0.5에서
    멀 때만 큰 값 → 진짜 일반화 신호의 강도.
    """
    num_cols = [c for c in X.columns
                if X[c].dtype.kind in "fc" and c != "subject_num" and _usable(X[c])]
    last = subject_time_blocked_folds(meta, n_splits=C.N_SPLITS) == (C.N_SPLITS - 1)
    rows_full, rows_last = {}, {}
    for t in C.TARGET_COLS:
        yt = y[t].values
        rows_full[t] = pd.Series({c: feature_auc(X[c].values.astype(float), yt) for c in num_cols})
        rows_last[t] = pd.Series({c: feature_auc(X[c].values[last].astype(float), yt[last]) for c in num_cols})
    full = pd.DataFrame(rows_full)
    lastd = pd.DataFrame(rows_last).add_suffix("_last")
    tbl = pd.concat([full, lastd], axis=1)
    for t in C.TARGET_COLS:
        same = np.sign(full[t] - 0.5) == np.sign(lastd[f"{t}_last"] - 0.5)
        tbl[f"{t}_gen"] = np.minimum((full[t] - 0.5).abs(), (lastd[f"{t}_last"] - 0.5).abs()) * same
    return tbl


def plot_top_features(auc_tbl: pd.DataFrame):
    """타깃별 '일반화되는' 상위 판별 피처 막대그래프 (full·last 둘 다 강한 것만)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.ravel()
    for i, t in enumerate(C.TARGET_COLS):
        s = auc_tbl[f"{t}_gen"].sort_values(ascending=False).head(15)[::-1]
        labels = [c[:34] for c in s.index]
        colors = ["#2c7fb8" if auc_tbl.loc[c, t] > 0.5 else "#de2d26" for c in s.index]
        axes[i].barh(labels, s.values, color=colors)
        axes[i].set_title(f"{t}  (일반화 신호강도 = min|full-.5|,|last-.5|)", fontsize=10)
        axes[i].axvline(0.05, color="gray", ls="--", lw=0.8)
        axes[i].tick_params(labelsize=7)
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.5, "파란색=값↑일수록 1\n빨간색=값↑일수록 0\n점선(0.05)=약한 신호 기준\n\n"
                  "full(전체)·last(미래블록) AUC가\n둘 다 0.5에서 멀어야 진짜 신호\n(스파이크·희소 피처는 제외함)",
                  fontsize=11)
    fig.suptitle("타깃별 '일반화되는' 단변량 판별력 (가짜 신호 제외)", fontsize=15)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "01_top_features_per_target.png", dpi=120)
    plt.close(fig)


def plot_separation(X: pd.DataFrame, y: pd.DataFrame, auc_tbl: pd.DataFrame):
    """각 타깃의 상위 6개 피처를 0/1 클래스별 히스토그램으로 (분리 정도 눈으로 확인)."""
    for t in C.TARGET_COLS:
        top = auc_tbl[f"{t}_gen"].sort_values(ascending=False).head(6).index
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, c in zip(axes.ravel(), top):
            x = X[c].values.astype(float)
            yt = y[t].values
            m = ~np.isnan(x)
            ax.hist(x[m & (yt == 1)], bins=25, alpha=0.55, label="1", color="#2c7fb8", density=True)
            ax.hist(x[m & (yt == 0)], bins=25, alpha=0.55, label="0", color="#de2d26", density=True)
            ax.set_title(f"{c[:36]}\nfull={auc_tbl.loc[c, t]:.3f} last={auc_tbl.loc[c, t+'_last']:.3f}", fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(f"{t}: 일반화 신호 상위 피처의 클래스별 분포 (파랑=1, 빨강=0; 겹침 적을수록 신호↑)", fontsize=13)
        fig.tight_layout()
        fig.savefig(EDA_DIR / f"02_separation_{t}.png", dpi=110)
        plt.close(fig)


def plot_subject_target_rate(y: pd.DataFrame, meta: pd.DataFrame):
    rate = meta.assign(**{t: y[t].values for t in C.TARGET_COLS}).groupby("subject_id")[C.TARGET_COLS].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(rate.values, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(C.TARGET_COLS))); ax.set_xticklabels(C.TARGET_COLS)
    ax.set_yticks(range(len(rate))); ax.set_yticklabels(rate.index)
    for i in range(len(rate)):
        for j in range(len(C.TARGET_COLS)):
            ax.text(j, i, f"{rate.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="rate of 1 (높을수록 그 사람은 대부분 1)")
    ax.set_title("피험자별 타깃 1 비율 (0/1로 치우친 사람일수록 prior로 잘 맞음)")
    fig.tight_layout(); fig.savefig(EDA_DIR / "03_subject_target_rate.png", dpi=120); plt.close(fig)


def plot_timeline(train, test):
    fig, ax = plt.subplots(figsize=(14, 5))
    subs = sorted(set(train.subject_id))
    for i, s in enumerate(subs):
        a = train[train.subject_id == s].sleep_date
        b = test[test.subject_id == s].sleep_date
        ax.scatter(a, [i] * len(a), c="#2c7fb8", s=12, label="train" if i == 0 else "")
        ax.scatter(b, [i] * len(b), c="#de2d26", s=12, marker="x", label="test" if i == 0 else "")
    ax.set_yticks(range(len(subs))); ax.set_yticklabels(subs)
    ax.legend(); ax.set_title("피험자별 train(파랑)·test(빨강X) 날짜 분포 — test는 대체로 '뒤쪽(미래)'")
    fig.tight_layout(); fig.savefig(EDA_DIR / "04_train_test_timeline.png", dpi=120); plt.close(fig)


def plot_target_corr(y: pd.DataFrame):
    corr = y[C.TARGET_COLS].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(7)); ax.set_xticklabels(C.TARGET_COLS)
    ax.set_yticks(range(7)); ax.set_yticklabels(C.TARGET_COLS)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("타깃 간 상관 (높으면 함께 예측 가능)")
    fig.tight_layout(); fig.savefig(EDA_DIR / "05_target_correlation.png", dpi=120); plt.close(fig)


def main():
    X, y, _, meta, _, _ = build_dataset(use_cache=True)
    train, test = load_labels()
    print("AUC 계산 중...", flush=True)
    auc_tbl = compute_auc_table(X, y, meta)

    # 진짜 일반화 신호: 전체와 last 모두 같은 방향으로 0.5에서 멀리 떨어진 피처
    print("\n=== 타깃별 '일반화되는' 신호 top-8  (full / last 둘 다 강해야 진짜) ===")
    print("    형식:  full_AUC  last_AUC  feature   (last가 0.5 근처면 가짜/과적합)")
    for t in C.TARGET_COLS:
        f = auc_tbl[t]; l = auc_tbl[f"{t}_last"]
        # 같은 방향(둘 다 >0.5 또는 둘 다 <0.5)이고 last도 강한 것 우선
        same_dir = np.sign(f - 0.5) == np.sign(l - 0.5)
        gen = (np.minimum((f - 0.5).abs(), (l - 0.5).abs())) * same_dir
        order = gen.sort_values(ascending=False).head(8).index
        print(f"\n[{t}]")
        for c in order:
            print(f"   full={f[c]:.3f}  last={l[c]:.3f}  {c}")
    auc_tbl.to_csv(EDA_DIR / "feature_auc_table.csv")

    plot_top_features(auc_tbl)
    plot_separation(X, y, auc_tbl)
    plot_subject_target_rate(y, meta)
    plot_timeline(train, test)
    plot_target_corr(y)
    print(f"\n저장 완료 → {EDA_DIR}/  (01~05 png + feature_auc_table.csv)")


if __name__ == "__main__":
    main()
