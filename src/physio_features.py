"""생리학적 수면추정 + 제대로 된 HRV 피처 (분 단위 night panel 기반).

기존 sleep_features.py 의 '최장 gap' 휴리스틱을 넘어, 분 단위 actigraphy + HR 로
수면/각성 hypnogram 을 만들고 TST/SE/SOL/WASO/각성수 를 직접 추정한다.
- S1~S4 는 NSF 객관 임계값(TST 7~9h / SE≥85% / SOL≤30m / WASO≤20m) → 임계 피처 동봉.
- HRV(RR 기반 RMSSD/SDNN/pNN50): 취침 전(pre-sleep) & 수면 중 윈도우 → Q2(피로)/Q3(스트레스).
- 피험자 내 편차(오늘 − 본인 중앙값): Q1~Q3 가 '개인 평균 대비' 라벨이므로 핵심.

야간 축: 18:00 → 다음날 12:00. minute = round(axis_hours*60), 범위 -360..+720.
결과: (subject_id, sleep_date) 키 테이블. 캐시 cache/physio_features.parquet.

실행: python -m src.physio_features        (전체 빌드, use_cache=False)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

NIGHT_MIN = -360   # 18:00
NIGHT_MAX = 720    # 12:00 (exclusive)
GRID = np.arange(NIGHT_MIN, NIGHT_MAX)
MOVE_CODES = {1, 2, 7, 8}


def _night_minute(df: pd.DataFrame) -> pd.DataFrame:
    """타임스탬프를 (night=sleep_date, minute=야간축 분) 으로 변환. 낮 12~18시는 제외."""
    ts = df["timestamp"]
    h = ts.dt.hour
    d = ts.dt.normalize()
    nd = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    nd[h >= 18] = d[h >= 18] + pd.Timedelta(days=1)
    nd[h < 12] = d[h < 12]
    keep = nd.notna()
    df = df[keep].copy()
    df["night"] = nd[keep]
    hh = ts[keep].dt.hour + ts[keep].dt.minute / 60.0
    axis = hh.where(hh < 18, hh - 24)
    df["minute"] = (axis * 60).round().astype(int)
    return df


def _hrv_per_minute() -> pd.DataFrame:
    """분 단위 HR 배열 → 분당 HR/HRV 통계. RR(ms)=60000/bpm 변환 후 RMSSD/SDNN/pNN50."""
    df = pd.read_parquet(C.sensor_path("wHr"), columns=["subject_id", "timestamp", "heart_rate"])
    df = _night_minute(df)

    def stats(a):
        a = np.asarray(a, dtype=float)
        a = a[(a > 25) & (a < 220)]
        if a.size < 3:
            m = a.mean() if a.size else np.nan
            return (m, a.min() if a.size else np.nan, np.nan, np.nan, np.nan, np.nan)
        rr = 60000.0 / a                      # 순간 RR(ms)
        drr = np.diff(rr)
        rmssd = np.sqrt(np.mean(drr ** 2))
        sdnn = rr.std()
        pnn50 = np.mean(np.abs(drr) > 50.0)
        return (a.mean(), a.min(), rmssd, sdnn, pnn50, a.std())

    s = df["heart_rate"].apply(stats)
    out = pd.DataFrame(s.tolist(), index=df.index,
                       columns=["hrm", "hrmin", "rmssd", "sdnn", "pnn50", "hrstd"])
    out[["subject_id", "night", "minute"]] = df[["subject_id", "night", "minute"]].values
    # 분 단위로 집계(같은 minute 중복 시 평균)
    g = out.groupby(["subject_id", "night", "minute"]).agg(
        hrm=("hrm", "mean"), hrmin=("hrmin", "min"), rmssd=("rmssd", "mean"),
        sdnn=("sdnn", "mean"), pnn50=("pnn50", "mean"))
    return g


def _minute_series(name: str, col: str, agg: str) -> pd.Series:
    df = pd.read_parquet(C.sensor_path(name), columns=["subject_id", "timestamp", col])
    df = _night_minute(df)
    return df.groupby(["subject_id", "night", "minute"])[col].agg(agg)


def _estimate_night(subject, night, screen, move, step, light, hrv) -> dict | None:
    """한 밤의 수면 지표 추정. hrv 는 분단위 DataFrame(hrm,hrmin,rmssd,sdnn,pnn50)."""
    def grab(s, fill=np.nan):
        try:
            v = s.loc[(subject, night)]
        except KeyError:
            return pd.Series(fill, index=GRID)
        return v.reindex(GRID)

    sc = grab(screen, 0).fillna(0).values
    mv = grab(move, 0).fillna(0).values
    st = grab(step, 0).fillna(0).values
    lt = grab(light).values
    try:
        H = hrv.loc[(subject, night)].reindex(GRID)
        hrm = H["hrm"].values
        rmssd = H["rmssd"].values
        sdnn = H["sdnn"].values
        pnn50 = H["pnn50"].values
    except KeyError:
        hrm = np.full(GRID.size, np.nan)
        rmssd = sdnn = pnn50 = hrm
    if np.isfinite(hrm).sum() < 30:
        return None

    rest = np.nanpercentile(hrm, 10)
    hr_high = np.where(np.isfinite(hrm), hrm > rest + 8, False)
    awake = ((st > 0) | (mv > 0) | (sc > 0) | hr_high).astype(float)
    quiet = 1 - awake
    sm = pd.Series(quiet).rolling(15, center=True, min_periods=1).mean().values
    asleep = (sm >= 0.5).astype(int)

    # 최장 수면 런
    best = (0, 0, 0)
    i = 0
    n = len(asleep)
    while i < n:
        if asleep[i]:
            j = i
            while j < n and asleep[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    L, s, e = best
    if L < 90:
        return None
    onset = GRID[s]
    wake = GRID[e - 1]
    aw_win = awake[s:e]
    waso = int(aw_win.sum())
    tst = (L - waso) / 60.0
    nawak = int(((aw_win[1:] == 1) & (aw_win[:-1] == 0)).sum())
    frag = nawak / max(tst, 0.5)

    # 취침(bedtime) 프록시 = onset 직전 마지막 화면 사용 시각 → SOL
    pre = np.where(sc[:s] > 0)[0]
    bed = GRID[pre.max()] if pre.size else onset
    sol = max(0, onset - bed)
    tib = wake - bed
    se = tst / (tib / 60.0) if tib > 0 else np.nan

    def wmean(arr, a, b):
        seg = arr[a:b]
        seg = seg[np.isfinite(seg)]
        return seg.mean() if seg.size else np.nan

    hr_sleep = wmean(hrm, s, e)
    hr_eve = wmean(hrm, max(0, s - 120), s)        # 취침 전 2h
    hr_drop = (hr_eve - hr_sleep) if np.isfinite(hr_eve) and np.isfinite(hr_sleep) else np.nan
    hr_morn = wmean(hrm, e, min(n, e + 60))

    # HRV: 취침 전 2h(pre-sleep) & 수면 중
    rmssd_pre = wmean(rmssd, max(0, s - 120), s)
    rmssd_slp = wmean(rmssd, s, e)
    sdnn_pre = wmean(sdnn, max(0, s - 120), s)
    sdnn_slp = wmean(sdnn, s, e)
    pnn50_slp = wmean(pnn50, s, e)
    rmssd_ratio = (rmssd_slp / rmssd_pre) if np.isfinite(rmssd_pre) and rmssd_pre > 0 else np.nan

    light_eve = wmean(lt, max(0, s - 120), s)
    light_slp = wmean(lt, s, e)

    return dict(
        subject_id=subject, sleep_date=pd.Timestamp(night),
        ph_tst=tst, ph_se=se, ph_sol=float(sol), ph_waso=float(waso),
        ph_nawak=float(nawak), ph_frag=frag,
        ph_onset_h=onset / 60.0, ph_wake_h=wake / 60.0, ph_tib=tib / 60.0,
        ph_rest=rest, ph_hr_sleep=hr_sleep, ph_hr_eve=hr_eve, ph_hr_drop=hr_drop,
        ph_hr_morn=hr_morn,
        ph_rmssd_pre=rmssd_pre, ph_rmssd_slp=rmssd_slp, ph_rmssd_ratio=rmssd_ratio,
        ph_sdnn_pre=sdnn_pre, ph_sdnn_slp=sdnn_slp, ph_pnn50_slp=pnn50_slp,
        ph_light_eve=light_eve, ph_light_slp=light_slp,
        # NSF 임계 피처(연속 + 지시)
        ph_nsf_tst_ok=float(7.0 <= tst <= 9.0),
        ph_nsf_tst_dist=min(abs(tst - 7.0), abs(tst - 9.0)) if not (7 <= tst <= 9) else 0.0,
        ph_nsf_se_ok=float(se >= 0.85) if np.isfinite(se) else np.nan,
        ph_nsf_sol_ok=float(sol <= 30),
        ph_nsf_waso_ok=float(waso <= 20),
    )


# 피험자 내 편차를 만들 연속 피처
DEV_COLS = ["ph_tst", "ph_se", "ph_sol", "ph_waso", "ph_frag", "ph_onset_h", "ph_wake_h",
            "ph_rest", "ph_hr_sleep", "ph_hr_drop", "ph_rmssd_pre", "ph_rmssd_slp",
            "ph_sdnn_pre", "ph_pnn50_slp", "ph_tib"]


def _add_within_subject(df: pd.DataFrame) -> pd.DataFrame:
    """오늘 − 본인 중앙값 (그리고 z). 라벨이 '개인 평균 대비'이므로 직접적 신호."""
    new = {}
    g = df.groupby("subject_id")
    for c in DEV_COLS:
        med = g[c].transform("median")
        std = g[c].transform("std").replace(0, np.nan)
        new[f"{c}_dev"] = df[c] - med
        new[f"{c}_z"] = (df[c] - med) / std
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """전날(lag1)·최근평균(roll7, 현재 제외)·추세 대비 편차. 수면빚/회복 동역학."""
    df = df.sort_values(["subject_id", "sleep_date"]).reset_index(drop=True)
    g = df.groupby("subject_id")
    new = {}
    for c in ["ph_tst", "ph_se", "ph_hr_drop", "ph_rest", "ph_rmssd_slp", "ph_sol", "ph_waso"]:
        new[f"{c}_lag1"] = g[c].shift(1)
        r7 = g[c].transform(lambda s: s.rolling(7, min_periods=2).mean().shift(1))
        new[f"{c}_roll7"] = r7
        new[f"{c}_vs_roll7"] = df[c] - r7
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def build_physio_features(use_cache: bool = True) -> pd.DataFrame:
    cache = C.CACHE_DIR / "physio_features.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    print("  [physio] loading minute streams ...", flush=True)
    screen = _minute_series("mScreenStatus", "m_screen_use", "max")
    act = pd.read_parquet(C.sensor_path("mActivity"), columns=["subject_id", "timestamp", "m_activity"])
    act = _night_minute(act)
    act["move"] = act["m_activity"].isin(MOVE_CODES).astype(float)
    move = act.groupby(["subject_id", "night", "minute"])["move"].max()
    step = _minute_series("wPedo", "step", "sum")
    light = _minute_series("wLight", "w_light", "mean")
    print("  [physio] computing per-minute HRV (per-second arrays) ...", flush=True)
    hrv = _hrv_per_minute()

    print("  [physio] estimating nightly metrics ...", flush=True)
    nights = hrv.index.droplevel("minute").unique()
    rows = []
    for k, (subject, night) in enumerate(nights):
        r = _estimate_night(subject, night, screen, move, step, light, hrv)
        if r:
            rows.append(r)
        if k % 500 == 0:
            print(f"    {k}/{len(nights)}", flush=True)
    out = pd.DataFrame(rows)
    out = _add_within_subject(out)
    out = _add_temporal(out)
    out.to_parquet(cache, index=False)
    print(f"  [physio] saved {cache} shape={out.shape}", flush=True)
    return out


if __name__ == "__main__":
    df = build_physio_features(use_cache=False)
    print(df.shape)
    print([c for c in df.columns if not c.endswith(("_dev", "_z", "_lag1", "_roll7", "_vs_roll7"))])
