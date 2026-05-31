"""라벨 행(subject, sleep_date, lifelog_date)에 일일 센서 피처 + 캘린더 + 피험자별 정규화를 결합.

- lifelog_date 의 일일 피처 → 접두사 L_ (취침 전 낮/저녁 활동)
- sleep_date  의 일일 피처 → 접두사 S_ (수면 야간/기상 아침)
- 피험자별 z-score: 타깃 Q1~Q3 가 '개인 평균 대비'로 정의되므로 핵심 신호.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .sensor_features import build_daily_features
from .sleep_features import build_sleep_features
from .nested_features import build_nested_features


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(C.TRAIN_CSV, parse_dates=["sleep_date", "lifelog_date"])
    test = pd.read_csv(C.SAMPLE_CSV, parse_dates=["sleep_date", "lifelog_date"])
    return train, test


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dt = df["sleep_date"]
    out["cal_dow"] = dt.dt.dayofweek.astype("float")
    out["cal_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float")
    out["cal_month"] = dt.dt.month.astype("float")
    out["cal_day"] = dt.dt.day.astype("float")
    out["cal_dow_sin"] = np.sin(2 * np.pi * out["cal_dow"] / 7)
    out["cal_dow_cos"] = np.cos(2 * np.pi * out["cal_dow"] / 7)
    doy = dt.dt.dayofyear
    out["cal_doy_sin"] = np.sin(2 * np.pi * doy / 366)
    out["cal_doy_cos"] = np.cos(2 * np.pi * doy / 366)
    return out


def _merge_daily(labels: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in daily.columns if c not in ("subject_id", "date")]
    blocks = []
    for prefix, date_col in [("L", "lifelog_date"), ("S", "sleep_date")]:
        right = daily.rename(columns={c: f"{prefix}_{c}" for c in feat_cols})
        merged = labels.merge(
            right,
            left_on=["subject_id", date_col],
            right_on=["subject_id", "date"],
            how="left",
        )
        blocks.append(merged[[f"{prefix}_{c}" for c in feat_cols]].reset_index(drop=True))
    return pd.concat(blocks, axis=1)


def _subject_zscore(df: pd.DataFrame, feat_cols: list[str], subj: pd.Series) -> pd.DataFrame:
    """피험자별 평균/표준편차로 z-score (라벨 미사용, train+test 전체 통계 = transductive)."""
    g = df[feat_cols].groupby(subj.values)
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    z = (df[feat_cols] - mean) / std
    z.columns = [f"{c}_zsubj" for c in feat_cols]
    return z


def build_dataset(use_cache: bool = True):
    """train/test 피처 행렬과 피처 컬럼 목록을 반환."""
    daily = build_daily_features(use_cache=use_cache)
    nested = build_nested_features(use_cache=use_cache)
    daily = daily.merge(nested, on=["subject_id", "date"], how="outer")
    sleep = build_sleep_features(use_cache=use_cache)
    train, test = load_labels()

    n_train = len(train)
    both = pd.concat([train, test], ignore_index=True)

    sensor = _merge_daily(both, daily)
    cal = _calendar(both)

    # 수면구간 피처: (subject_id, sleep_date) 기준으로 라벨 행에 직접 결합
    slp = both.merge(sleep, on=["subject_id", "sleep_date"], how="left")
    slp_cols = [c for c in sleep.columns if c not in ("subject_id", "sleep_date")]
    slp = slp[slp_cols].reset_index(drop=True)

    # 피험자별 z-score 는 'count' 류를 제외한 연속 피처에만 적용 (센서+수면구간)
    feats_for_z = pd.concat([sensor, slp], axis=1)
    z_src = [c for c in feats_for_z.columns
             if not c.endswith("_count") and not c.endswith("_act_count")]
    zsubj = _subject_zscore(feats_for_z, z_src, both["subject_id"])

    X = pd.concat([sensor, slp, cal, zsubj], axis=1)
    X["subject_num"] = both["subject_id"].str.extract(r"(\d+)").astype("float").values
    X["subject_id"] = both["subject_id"].astype("category").values

    feat_cols = [c for c in X.columns if c != "subject_id"] + ["subject_id"]

    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    y_train = train[C.TARGET_COLS].reset_index(drop=True)

    meta_train = train[C.ID_COLS].reset_index(drop=True)
    meta_test = test[C.ID_COLS].reset_index(drop=True)

    return X_train, y_train, X_test, meta_train, meta_test, feat_cols


if __name__ == "__main__":
    Xtr, ytr, Xte, mtr, mte, cols = build_dataset(use_cache=True)
    print("X_train", Xtr.shape, "X_test", Xte.shape, "n_feats", len(cols))
    print("non-null feature coverage (train, mean over cols): "
          f"{Xtr[cols[:-1]].notna().mean().mean():.3f}")
