"""센서 parquet → (subject_id, date) 단위 일일 윈도우 피처 집계.

각 센서의 고빈도 시계열을 하루 중 시각 윈도우(full/day/eve/night/morn)별 통계로 요약한다.
결과는 (subject_id, date)를 인덱스로 하는 넓은 형식의 일일 피처 테이블이며 cache/daily_features.parquet 로 저장된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

STATS = ["mean", "std", "min", "max", "sum", "count"]


def _add_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df = df.copy()
    df["date"] = ts.dt.normalize()
    df["hour"] = ts.dt.hour
    return df


def _window_agg(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    """(subject_id,date) × 윈도우별로 value_cols 통계 집계 → 넓은 테이블."""
    parts = []
    for win, (h0, h1) in C.WINDOWS.items():
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        if sub.empty:
            continue
        g = sub.groupby(["subject_id", "date"])[value_cols].agg(STATS)
        g.columns = [f"{prefix}_{win}_{c}_{s}" for c, s in g.columns]
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    return out


def aggregate_numeric(name: str, value_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path(name), columns=["subject_id", "timestamp", *value_cols])
    df = _add_date_hour(df)
    return _window_agg(df, value_cols, name)


def aggregate_activity() -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path("mActivity"), columns=["subject_id", "timestamp", "m_activity"])
    df = _add_date_hour(df)
    df["is_still"] = df["m_activity"].isin(C.ACTIVITY_STILL).astype("float")
    df["is_move"] = df["m_activity"].isin(C.ACTIVITY_MOVE).astype("float")
    df["is_vehicle"] = df["m_activity"].isin(C.ACTIVITY_VEHICLE).astype("float")
    parts = []
    for win, (h0, h1) in C.WINDOWS.items():
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        if sub.empty:
            continue
        g = sub.groupby(["subject_id", "date"]).agg(
            **{
                f"mActivity_{win}_still_frac": ("is_still", "mean"),
                f"mActivity_{win}_move_frac": ("is_move", "mean"),
                f"mActivity_{win}_vehicle_frac": ("is_vehicle", "mean"),
                f"mActivity_{win}_count": ("is_still", "size"),
            }
        )
        parts.append(g)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def aggregate_hr() -> pd.DataFrame:
    """wHr: heart_rate 가 타임스탬프별 분 단위 값 배열. 행 요약 후 윈도우 집계."""
    df = pd.read_parquet(C.sensor_path("wHr"), columns=["subject_id", "timestamp", "heart_rate"])
    arr = df["heart_rate"].apply(lambda a: np.asarray(a, dtype="float"))
    df["hr_mean"] = arr.apply(lambda a: a.mean() if a.size else np.nan)
    df["hr_min"] = arr.apply(lambda a: a.min() if a.size else np.nan)
    df["hr_max"] = arr.apply(lambda a: a.max() if a.size else np.nan)
    df["hr_std"] = arr.apply(lambda a: a.std() if a.size else np.nan)
    df = _add_date_hour(df)
    parts = []
    for win, (h0, h1) in C.WINDOWS.items():
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        if sub.empty:
            continue
        g = sub.groupby(["subject_id", "date"]).agg(
            **{
                f"wHr_{win}_mean": ("hr_mean", "mean"),
                f"wHr_{win}_min": ("hr_min", "min"),
                f"wHr_{win}_max": ("hr_max", "max"),
                f"wHr_{win}_std": ("hr_mean", "std"),
                f"wHr_{win}_restmin": ("hr_min", "mean"),
                f"wHr_{win}_count": ("hr_mean", "size"),
            }
        )
        parts.append(g)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def build_daily_features(use_cache: bool = True) -> pd.DataFrame:
    cache = C.CACHE_DIR / "daily_features.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    tables = []
    for name, cols in C.NUMERIC_SENSORS.items():
        print(f"  [numeric] {name} ...", flush=True)
        tables.append(aggregate_numeric(name, cols))
    print("  [activity] mActivity ...", flush=True)
    tables.append(aggregate_activity())
    print("  [hr] wHr ...", flush=True)
    tables.append(aggregate_hr())

    daily = pd.concat(tables, axis=1).reset_index()
    daily.to_parquet(cache, index=False)
    print(f"  saved {cache}  shape={daily.shape}", flush=True)
    return daily


if __name__ == "__main__":
    d = build_daily_features(use_cache=False)
    print(d.shape)
    print(d.columns.tolist()[:20])
