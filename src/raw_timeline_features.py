"""Compact raw timeline features from the original sensor parquet files.

The existing dataset builder compresses high-frequency streams into broad daily
aggregates.  This module adds a smaller, event-oriented feature table focused on
the two places where the current submissions fail:

- night state detection for S1-S4
- daytime/evening stress/activity state detection for Q2/Q3

All features are keyed to the competition rows by subject_id plus either
sleep_date (night features) or lifelog_date (day/evening features).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


NIGHT_WINDOWS = {
    "pre": (-6.0, 0.0),
    "mid": (0.0, 3.0),
    "late": (3.0, 6.0),
    "wake": (6.0, 12.0),
    "all": (-6.0, 12.0),
}

DAY_WINDOWS = {
    "full": (0, 24),
    "morn": (6, 9),
    "day": (9, 18),
    "eve": (18, 24),
    "late": (21, 24),
}


def _night_axis(ts: pd.Series) -> pd.Series:
    h = ts.dt.hour + ts.dt.minute / 60.0
    return h.where(h < 18, h - 24)


def _night_date(ts: pd.Series) -> pd.Series:
    d = ts.dt.normalize()
    h = ts.dt.hour
    nd = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    nd[h >= 18] = d[h >= 18] + pd.Timedelta(days=1)
    nd[h < 12] = d[h < 12]
    return nd


def _load_night(name: str, value_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path(name), columns=["subject_id", "timestamp", *value_cols])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    nd = _night_date(df["timestamp"])
    df = df[nd.notna()].copy()
    df["sleep_date"] = nd[nd.notna()].values
    df["axis"] = _night_axis(df["timestamp"]).values
    return df


def _load_day(name: str, value_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path(name), columns=["subject_id", "timestamp", *value_cols])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    return df


def _longest_gap(active_axis: np.ndarray, start: float = -6.0, end: float = 12.0) -> tuple[float, float, float]:
    active_axis = np.sort(np.asarray(active_axis, dtype=float))
    active_axis = active_axis[np.isfinite(active_axis)]
    bounds = np.concatenate([[start], active_axis[(active_axis >= start) & (active_axis <= end)], [end]])
    if len(bounds) < 2:
        return np.nan, np.nan, np.nan
    gaps = np.diff(bounds)
    k = int(np.argmax(gaps))
    return float(bounds[k]), float(bounds[k + 1]), float(gaps[k])


def _count_between(axis: np.ndarray, lo: float, hi: float) -> int:
    axis = np.asarray(axis, dtype=float)
    return int(((axis >= lo) & (axis < hi)).sum())


def _window_scalar_stats(
    df: pd.DataFrame,
    key_date_col: str,
    time_col: str,
    value_col: str,
    prefix: str,
    windows: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    rows = []
    for win, (lo, hi) in windows.items():
        sub = df[(df[time_col] >= lo) & (df[time_col] < hi)]
        if sub.empty:
            continue
        g = sub.groupby(["subject_id", key_date_col])[value_col].agg(["mean", "std", "min", "max", "sum", "count"])
        g.columns = [f"{prefix}_{win}_{stat}" for stat in g.columns]
        rows.append(g)
    return pd.concat(rows, axis=1) if rows else pd.DataFrame()


def _screen_night() -> pd.DataFrame:
    df = _load_night("mScreenStatus", ["m_screen_use"])
    scalar = _window_scalar_stats(df, "sleep_date", "axis", "m_screen_use", "nt_screen", NIGHT_WINDOWS)

    def per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("axis")
        axis = g["axis"].to_numpy(float)
        val = g["m_screen_use"].to_numpy(float)
        on_axis = axis[val > 0.5]
        off_onset, off_wake, off_tst = _longest_gap(on_axis)
        return pd.Series(
            {
                "nt_screen_transitions": float(np.sum(val[1:] != val[:-1])) if len(val) > 1 else 0.0,
                "nt_screen_first_on": float(on_axis.min()) if len(on_axis) else np.nan,
                "nt_screen_last_on": float(on_axis.max()) if len(on_axis) else np.nan,
                "nt_screen_longest_off_onset": off_onset,
                "nt_screen_longest_off_wake": off_wake,
                "nt_screen_longest_off_tst": off_tst,
                "nt_screen_on_pre_count": _count_between(on_axis, -6.0, 0.0),
                "nt_screen_on_mid_count": _count_between(on_axis, 0.0, 3.0),
                "nt_screen_on_late_count": _count_between(on_axis, 3.0, 6.0),
                "nt_screen_on_wake_count": _count_between(on_axis, 6.0, 12.0),
            }
        )

    detail = df.groupby(["subject_id", "sleep_date"]).apply(per_group, include_groups=False)
    return pd.concat([scalar, detail], axis=1)


def _charge_night() -> pd.DataFrame:
    df = _load_night("mACStatus", ["m_charging"])
    scalar = _window_scalar_stats(df, "sleep_date", "axis", "m_charging", "nt_charge", NIGHT_WINDOWS)

    def per_group(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("axis")
        axis = g["axis"].to_numpy(float)
        val = g["m_charging"].to_numpy(float)
        charge_axis = axis[val > 0.5]
        onset, wake, dur = _longest_gap(axis[val <= 0.5])
        return pd.Series(
            {
                "nt_charge_first": float(charge_axis.min()) if len(charge_axis) else np.nan,
                "nt_charge_last": float(charge_axis.max()) if len(charge_axis) else np.nan,
                "nt_charge_longest_on_onset": onset,
                "nt_charge_longest_on_wake": wake,
                "nt_charge_longest_on_tst": dur,
                "nt_charge_at_midnight": float(np.mean(val[(axis >= 0.0) & (axis < 1.0)])) if np.any((axis >= 0.0) & (axis < 1.0)) else np.nan,
                "nt_charge_at_4h": float(np.mean(val[(axis >= 4.0) & (axis < 5.0)])) if np.any((axis >= 4.0) & (axis < 5.0)) else np.nan,
            }
        )

    detail = df.groupby(["subject_id", "sleep_date"]).apply(per_group, include_groups=False)
    return pd.concat([scalar, detail], axis=1)


def _activity_night() -> pd.DataFrame:
    df = _load_night("mActivity", ["m_activity"])
    df["still"] = df["m_activity"].isin(C.ACTIVITY_STILL).astype(float)
    df["move"] = df["m_activity"].isin(C.ACTIVITY_MOVE).astype(float)
    df["vehicle"] = df["m_activity"].isin(C.ACTIVITY_VEHICLE).astype(float)
    parts = []
    for col in ["still", "move", "vehicle"]:
        parts.append(_window_scalar_stats(df, "sleep_date", "axis", col, f"nt_activity_{col}", NIGHT_WINDOWS))
    return pd.concat(parts, axis=1)


def _pedo_night() -> pd.DataFrame:
    df = _load_night("wPedo", ["step", "distance", "speed", "burned_calories"])
    for col in ["step", "distance", "speed", "burned_calories"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["active"] = (df["step"].fillna(0) > 0).astype(float)
    parts = [
        _window_scalar_stats(df, "sleep_date", "axis", "step", "nt_pedo_step", NIGHT_WINDOWS),
        _window_scalar_stats(df, "sleep_date", "axis", "distance", "nt_pedo_distance", NIGHT_WINDOWS),
        _window_scalar_stats(df, "sleep_date", "axis", "speed", "nt_pedo_speed", NIGHT_WINDOWS),
        _window_scalar_stats(df, "sleep_date", "axis", "burned_calories", "nt_pedo_cal", NIGHT_WINDOWS),
        _window_scalar_stats(df, "sleep_date", "axis", "active", "nt_pedo_active", NIGHT_WINDOWS),
    ]

    def per_group(g: pd.DataFrame) -> pd.Series:
        axis = g["axis"].to_numpy(float)
        active_axis = axis[g["step"].fillna(0).to_numpy(float) > 0]
        onset, wake, dur = _longest_gap(active_axis)
        return pd.Series(
            {
                "nt_pedo_longest_no_step_onset": onset,
                "nt_pedo_longest_no_step_wake": wake,
                "nt_pedo_longest_no_step_tst": dur,
                "nt_pedo_step_burst_count": float(np.sum(g["step"].fillna(0).to_numpy(float) >= 10.0)),
            }
        )

    parts.append(df.groupby(["subject_id", "sleep_date"]).apply(per_group, include_groups=False))
    return pd.concat(parts, axis=1)


def _hr_row_summaries(series: pd.Series) -> pd.DataFrame:
    arr = series.apply(lambda a: np.asarray(a, dtype=float))
    out = pd.DataFrame(index=series.index)
    out["hr_mean"] = arr.apply(lambda a: float(np.nanmean(a)) if a.size else np.nan)
    out["hr_min"] = arr.apply(lambda a: float(np.nanmin(a)) if a.size else np.nan)
    out["hr_max"] = arr.apply(lambda a: float(np.nanmax(a)) if a.size else np.nan)
    out["hr_std"] = arr.apply(lambda a: float(np.nanstd(a)) if a.size >= 2 else np.nan)
    out["hr_rmssd"] = arr.apply(lambda a: float(np.sqrt(np.nanmean(np.diff(a) ** 2))) if a.size >= 3 else np.nan)
    return out


def _hr_night() -> pd.DataFrame:
    df = _load_night("wHr", ["heart_rate"])
    summary = _hr_row_summaries(df["heart_rate"])
    df = pd.concat([df.drop(columns=["heart_rate"]), summary], axis=1)
    parts = []
    for col in ["hr_mean", "hr_min", "hr_max", "hr_std", "hr_rmssd"]:
        parts.append(_window_scalar_stats(df, "sleep_date", "axis", col, f"nt_hr_{col}", NIGHT_WINDOWS))

    def per_group(g: pd.DataFrame) -> pd.Series:
        base = float(np.nanmin(g["hr_min"].values)) if g["hr_min"].notna().any() else np.nan
        axis = g["axis"].to_numpy(float)
        mean = g["hr_mean"].to_numpy(float)
        low5 = mean <= base + 5.0 if np.isfinite(base) else np.zeros(len(g), dtype=bool)
        low10 = mean <= base + 10.0 if np.isfinite(base) else np.zeros(len(g), dtype=bool)
        high15 = mean >= base + 15.0 if np.isfinite(base) else np.zeros(len(g), dtype=bool)
        onset, wake, dur = _longest_gap(axis[high15])
        return pd.Series(
            {
                "nt_hr_low5_ratio": float(np.mean(low5)) if len(low5) else np.nan,
                "nt_hr_low10_ratio": float(np.mean(low10)) if len(low10) else np.nan,
                "nt_hr_high15_ratio": float(np.mean(high15)) if len(high15) else np.nan,
                "nt_hr_longest_low_onset": onset,
                "nt_hr_longest_low_wake": wake,
                "nt_hr_longest_low_tst": dur,
            }
        )

    parts.append(df.groupby(["subject_id", "sleep_date"]).apply(per_group, include_groups=False))
    return pd.concat(parts, axis=1)


def _light_night() -> pd.DataFrame:
    parts = []
    for name, col, prefix in [("mLight", "m_light", "nt_mlight"), ("wLight", "w_light", "nt_wlight")]:
        df = _load_night(name, [col])
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df["dark"] = (df[col].fillna(np.inf) <= 1.0).astype(float)
        parts.append(_window_scalar_stats(df, "sleep_date", "axis", col, prefix, NIGHT_WINDOWS))
        parts.append(_window_scalar_stats(df, "sleep_date", "axis", "dark", f"{prefix}_dark", NIGHT_WINDOWS))
    return pd.concat(parts, axis=1)


def build_night_timeline_features(use_cache: bool = True) -> pd.DataFrame:
    cache = C.CACHE_DIR / "raw_night_timeline_features.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    tables = [
        _screen_night(),
        _charge_night(),
        _activity_night(),
        _pedo_night(),
        _hr_night(),
        _light_night(),
    ]
    out = pd.concat(tables, axis=1).reset_index()
    out.to_parquet(cache, index=False)
    return out


def _activity_day() -> pd.DataFrame:
    df = _load_day("mActivity", ["m_activity"])
    df["still"] = df["m_activity"].isin(C.ACTIVITY_STILL).astype(float)
    df["move"] = df["m_activity"].isin(C.ACTIVITY_MOVE).astype(float)
    df["vehicle"] = df["m_activity"].isin(C.ACTIVITY_VEHICLE).astype(float)
    parts = []
    for col in ["still", "move", "vehicle"]:
        parts.append(_window_scalar_stats(df, "date", "hour", col, f"day_activity_{col}", DAY_WINDOWS))
    return pd.concat(parts, axis=1)


def _screen_day() -> pd.DataFrame:
    df = _load_day("mScreenStatus", ["m_screen_use"])
    return _window_scalar_stats(df, "date", "hour", "m_screen_use", "day_screen", DAY_WINDOWS)


def _usage_day() -> pd.DataFrame:
    df = _load_day("mUsageStats", ["m_usage_stats"])
    df["use_total"] = df["m_usage_stats"].apply(lambda a: float(sum(x.get("total_time", 0.0) for x in a)))
    df["use_napps"] = df["m_usage_stats"].apply(len).astype(float)
    total = _window_scalar_stats(df, "date", "hour", "use_total", "day_usage_total", DAY_WINDOWS)
    apps = _window_scalar_stats(df, "date", "hour", "use_napps", "day_usage_napps", DAY_WINDOWS)
    return pd.concat([total, apps], axis=1)


def _pedo_day() -> pd.DataFrame:
    df = _load_day("wPedo", ["step", "distance", "speed", "burned_calories"])
    parts = []
    for col in ["step", "distance", "speed", "burned_calories"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        parts.append(_window_scalar_stats(df, "date", "hour", col, f"day_pedo_{col}", DAY_WINDOWS))
    return pd.concat(parts, axis=1)


def _hr_day() -> pd.DataFrame:
    df = _load_day("wHr", ["heart_rate"])
    summary = _hr_row_summaries(df["heart_rate"])
    df = pd.concat([df.drop(columns=["heart_rate"]), summary], axis=1)
    parts = []
    for col in ["hr_mean", "hr_min", "hr_max", "hr_std", "hr_rmssd"]:
        parts.append(_window_scalar_stats(df, "date", "hour", col, f"day_hr_{col}", DAY_WINDOWS))
    return pd.concat(parts, axis=1)


def _gps_day() -> pd.DataFrame:
    df = _load_day("mGps", ["m_gps"])
    gps = df["m_gps"]
    df["gps_speed_mean"] = gps.apply(lambda a: float(np.mean([p.get("speed", np.nan) for p in a])) if len(a) else np.nan)
    df["gps_speed_max"] = gps.apply(lambda a: float(np.nanmax([p.get("speed", np.nan) for p in a])) if len(a) else np.nan)
    df["gps_alt_std"] = gps.apply(lambda a: float(np.nanstd([p.get("altitude", np.nan) for p in a])) if len(a) else np.nan)
    parts = []
    for col in ["gps_speed_mean", "gps_speed_max", "gps_alt_std"]:
        parts.append(_window_scalar_stats(df, "date", "hour", col, f"day_gps_{col}", DAY_WINDOWS))
    return pd.concat(parts, axis=1)


def _ambience_day() -> pd.DataFrame:
    labels = ["Silence", "Speech", "Conversation", "Music", "Narration, monologue"]
    df = _load_day("mAmbience", ["m_ambience"])
    def get_prob(row, label: str) -> float:
        out = 0.0
        for pair in row:
            try:
                if pair[0] == label:
                    out = float(pair[1])
                    break
            except (TypeError, ValueError, IndexError):
                continue
        return out
    parts = []
    for lab in labels:
        short = lab.split(",")[0].lower().replace(" ", "")
        col = f"amb_{short}"
        df[col] = df["m_ambience"].apply(lambda row, label=lab: get_prob(row, label))
        parts.append(_window_scalar_stats(df, "date", "hour", col, f"day_{col}", DAY_WINDOWS))
    return pd.concat(parts, axis=1)


def build_lifelog_timeline_features(use_cache: bool = True) -> pd.DataFrame:
    cache = C.CACHE_DIR / "raw_lifelog_timeline_features.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    tables = [
        _screen_day(),
        _usage_day(),
        _activity_day(),
        _pedo_day(),
        _hr_day(),
        _gps_day(),
        _ambience_day(),
    ]
    out = pd.concat(tables, axis=1).reset_index()
    out.to_parquet(cache, index=False)
    return out


def build_label_timeline_features(labels: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Return compact raw-timeline features aligned to competition rows."""
    labels = labels[["subject_id", "sleep_date", "lifelog_date"]].copy()
    labels["sleep_date"] = pd.to_datetime(labels["sleep_date"])
    labels["lifelog_date"] = pd.to_datetime(labels["lifelog_date"])

    night = build_night_timeline_features(use_cache=use_cache)
    night["sleep_date"] = pd.to_datetime(night["sleep_date"])
    night_cols = [c for c in night.columns if c not in {"subject_id", "sleep_date"}]
    night = night.rename(columns={c: f"N_{c}" for c in night_cols})

    day = build_lifelog_timeline_features(use_cache=use_cache)
    day["date"] = pd.to_datetime(day["date"])
    day_cols = [c for c in day.columns if c not in {"subject_id", "date"}]
    day = day.rename(columns={c: f"L_{c}" for c in day_cols})

    out = labels.merge(night, on=["subject_id", "sleep_date"], how="left")
    out = out.merge(day, left_on=["subject_id", "lifelog_date"], right_on=["subject_id", "date"], how="left")
    out = out.drop(columns=["date"], errors="ignore")
    return out.drop(columns=["subject_id", "sleep_date", "lifelog_date"])


if __name__ == "__main__":
    train = pd.read_csv(C.TRAIN_CSV, parse_dates=["sleep_date", "lifelog_date"])
    test = pd.read_csv(C.SAMPLE_CSV, parse_dates=["sleep_date", "lifelog_date"])
    both = pd.concat([train[C.ID_COLS], test[C.ID_COLS]], ignore_index=True)
    feats = build_label_timeline_features(both, use_cache=False)
    print(feats.shape)
    print(feats.notna().mean().describe())
