"""중첩(리스트/배열) 센서 → (subject_id, date) 윈도우 피처.

- mAmbience: 오디오 장면 확률(Silence=조용함/수면, Speech·Conversation=사회활동, Music)
- mUsageStats: 앱 사용시간 합·앱 개수(취침 전 스마트폰 사용 = 각성/스트레스 프록시)
- mWifi / mBle: 주변 AP·기기 수(장소 안정성·사회적 근접)
- mGps: 속도(이동성)·고도 변동
결과 캐시 cache/nested_features.parquet, (subject_id, date) 키.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .sensor_features import _add_date_hour, _window_agg

AMB_LABELS = ["Silence", "Speech", "Conversation", "Music", "Narration, monologue"]


def _amb_probs(row) -> dict:
    d = {}
    for pair in row:
        try:
            d[pair[0]] = float(pair[1])
        except (ValueError, TypeError):
            pass
    return d


def aggregate_ambience() -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path("mAmbience"), columns=["subject_id", "timestamp", "m_ambience"])
    probs = df["m_ambience"].apply(_amb_probs)
    for lab in AMB_LABELS:
        key = "amb_" + lab.split(",")[0].lower().replace(" ", "")
        df[key] = probs.apply(lambda d: d.get(lab, 0.0))
    df = _add_date_hour(df)
    cols = [c for c in df.columns if c.startswith("amb_")]
    # 평균만(합/최댓값은 노이즈) → _window_agg 대신 직접 mean
    parts = []
    for win, (h0, h1) in C.WINDOWS.items():
        sub = df[(df["hour"] >= h0) & (df["hour"] < h1)]
        if sub.empty:
            continue
        g = sub.groupby(["subject_id", "date"])[cols].mean()
        g.columns = [f"{c}_{win}_mean" for c in g.columns]
        parts.append(g)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def aggregate_usage() -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path("mUsageStats"), columns=["subject_id", "timestamp", "m_usage_stats"])
    df["use_total"] = df["m_usage_stats"].apply(lambda a: float(sum(x["total_time"] for x in a)))
    df["use_napps"] = df["m_usage_stats"].apply(len).astype(float)
    df = _add_date_hour(df)
    return _window_agg(df, ["use_total", "use_napps"], "mUsage")


def aggregate_wifi() -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path("mWifi"), columns=["subject_id", "timestamp", "m_wifi"])
    df["wifi_n"] = df["m_wifi"].apply(len).astype(float)
    df = _add_date_hour(df)
    return _window_agg(df, ["wifi_n"], "mWifi")


def aggregate_ble() -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path("mBle"), columns=["subject_id", "timestamp", "m_ble"])
    df["ble_n"] = df["m_ble"].apply(len).astype(float)
    df = _add_date_hour(df)
    return _window_agg(df, ["ble_n"], "mBle")


def aggregate_gps() -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path("mGps"), columns=["subject_id", "timestamp", "m_gps"])
    spd = df["m_gps"].apply(lambda a: np.array([p["speed"] for p in a], dtype=float))
    alt = df["m_gps"].apply(lambda a: np.array([p["altitude"] for p in a], dtype=float))
    df["gps_speed_mean"] = spd.apply(lambda a: a.mean() if a.size else np.nan)
    df["gps_speed_max"] = spd.apply(lambda a: a.max() if a.size else np.nan)
    df["gps_alt_std"] = alt.apply(lambda a: a.std() if a.size else np.nan)
    df = _add_date_hour(df)
    return _window_agg(df, ["gps_speed_mean", "gps_speed_max", "gps_alt_std"], "mGps")


def build_nested_features(use_cache: bool = True) -> pd.DataFrame:
    cache = C.CACHE_DIR / "nested_features.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    tables = []
    for label, fn in [("ambience", aggregate_ambience), ("usage", aggregate_usage),
                      ("wifi", aggregate_wifi), ("ble", aggregate_ble), ("gps", aggregate_gps)]:
        print(f"  [nested] {label} ...", flush=True)
        tables.append(fn())
    nested = pd.concat(tables, axis=1).reset_index()
    nested.to_parquet(cache, index=False)
    print(f"  saved {cache}  shape={nested.shape}")
    return nested


if __name__ == "__main__":
    n = build_nested_features(use_cache=False)
    print(n.shape)
    print([c for c in n.columns][:25])
