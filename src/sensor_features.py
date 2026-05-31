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


# 행동 시간동역학을 적용할 일일 피처(누적 활동량·각성·사회부하·수면환경)
TEMPORAL_DAILY = [
    "wPedo_day_step_sum", "wPedo_day_distance_sum",   # 낮 신체 활동(피로 누적)
    "wHr_eve_mean", "wHr_day_mean",                    # 각성/심박 부하
    "mActivity_day_move_frac",                          # 낮 이동 비율
    "mUsage_eve_use_total_sum", "mUsage_full_use_total_sum",  # 스마트폰/앱 사용(정신부하)
    "mScreenStatus_full_m_screen_use_mean",            # 화면 사용
    "amb_conversation_eve_mean", "amb_speech_full_mean",  # 사회적 상호작용(스트레스)
    "mGps_day_gps_speed_mean_mean",                    # 이동성
    "mLight_day_m_light_mean",                          # 빛 노출
]


def add_daily_temporal(daily: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """일일 행동 피처에 시간동역학 추가: 전날(lag1)·최근평균(roll3/7, 현재 제외)·최근 대비 편차.

    피로/스트레스(Q2·Q3)는 누적 활동·각성·수면빚의 함수 → 단일 날보다 추세가 중요.
    daily 는 (subject_id, date) 키. 결과는 동일 키에 컬럼 추가.
    """
    cols = cols or TEMPORAL_DAILY
    cols = [c for c in cols if c in daily.columns]
    daily = daily.sort_values(["subject_id", "date"]).reset_index(drop=True)
    g = daily.groupby("subject_id")
    new = {}
    for c in cols:
        new[f"{c}_lag1"] = g[c].shift(1)
        r7 = g[c].transform(lambda s: s.rolling(7, min_periods=2).mean().shift(1))
        r3 = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        new[f"{c}_roll7"] = r7
        new[f"{c}_vs_roll7"] = daily[c] - r7
        new[f"{c}_roll3"] = r3
    return pd.concat([daily, pd.DataFrame(new, index=daily.index)], axis=1)


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
