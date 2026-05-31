"""수면구간 탐지 피처: 폰(화면/충전)·워치(심박/걸음)·활동/조도로 야간 수면 블록을 추정.

S1~S4(총수면시간 TST / 수면효율 SE / 입면지연 SOL / 각성 WASO)의 직접 프록시를 만든다.
야간 축(night-axis): 시각 t 를 18시 이후는 -24 보정 → 18:00=-6 ... 자정=0 ... 정오=+12 로 단조 정렬.
각 이벤트는 그것이 속한 '수면일(sleep_date)'에 귀속(18시 이후=다음날, 12시 이전=당일, 낮 12~18시는 야간 제외).
결과는 (subject_id, sleep_date) 키의 야간 피처 테이블. 캐시 cache/sleep_features.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _night_axis(ts: pd.Series) -> pd.Series:
    """시각(소수 시간)을 야간 단조 축으로 변환. 18시 이후는 -24 보정."""
    h = ts.dt.hour + ts.dt.minute / 60.0
    return h.where(h < 18, h - 24)


def _night_date(ts: pd.Series) -> pd.Series:
    """이벤트가 귀속될 수면일. 18시 이후→다음날, 12시 이전→당일, 낮(12~18)→제외(NaT)."""
    d = ts.dt.normalize()
    h = ts.dt.hour
    nd = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
    nd[h >= 18] = d[h >= 18] + pd.Timedelta(days=1)
    nd[h < 12] = d[h < 12]
    return nd


def _load_night(name: str, value_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(C.sensor_path(name), columns=["subject_id", "timestamp", *value_cols])
    nd = _night_date(df["timestamp"])
    df = df[nd.notna()].copy()
    df["night"] = nd[nd.notna()].values
    df["axis"] = _night_axis(df["timestamp"]).values
    return df


def _longest_off_block(g: pd.DataFrame) -> pd.Series:
    """화면 사용(use==1) 이벤트 사이 최장 '꺼짐' 구간 = 주 수면 블록."""
    on = np.sort(g.loc[g["m_screen_use"] == 1, "axis"].values)
    bounds = np.concatenate([[-6.0], on, [12.0]])
    if len(bounds) < 2:
        return pd.Series({"onset": np.nan, "wake": np.nan, "tst": np.nan})
    gaps = np.diff(bounds)
    k = int(np.argmax(gaps))
    onset, wake = bounds[k], bounds[k + 1]
    return pd.Series({"onset": onset, "wake": wake, "tst": wake - onset})


def _block_from_active(active_axis: np.ndarray) -> pd.Series:
    """'활동(깨어있음)' 이벤트 시각 배열로 최장 비활동(수면) 블록을 찾는다."""
    on = np.sort(active_axis)
    bounds = np.concatenate([[-6.0], on, [12.0]])
    gaps = np.diff(bounds)
    if len(gaps) == 0:
        return pd.Series({"onset": np.nan, "wake": np.nan, "tst": np.nan})
    k = int(np.argmax(gaps))
    return pd.Series({"onset": bounds[k], "wake": bounds[k + 1], "tst": bounds[k + 1] - bounds[k]})


TEMPORAL_COLS = ["slp_tst_consensus", "slp_onset_consensus", "slp_wake_consensus",
                 "slp_tst_hr", "slp_eff_proxy", "slp_hr_min", "slp_hr_mean",
                 "slp_step_sum", "slp_waso_screen_events", "slp_waso_hr"]


def _add_temporal(sleep: pd.DataFrame) -> pd.DataFrame:
    """시간 동역학 피처: 전날 값(lag) + 최근 평균(rolling, 현재일 제외) + 최근평균 대비 편차.

    수면빚·수면 모멘텀·규칙성을 포착. 센서는 모든 날 존재하므로 라벨 유무와 무관하게 계산 가능.
    """
    sleep = sleep.sort_values(["subject_id", "sleep_date"]).reset_index(drop=True)
    g = sleep.groupby("subject_id")
    new = {}
    for c in TEMPORAL_COLS:
        if c not in sleep.columns:
            continue
        new[f"{c}_lag1"] = g[c].shift(1)
        roll7 = g[c].transform(lambda s: s.rolling(7, min_periods=2).mean().shift(1))
        roll3 = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        new[f"{c}_roll7"] = roll7
        new[f"{c}_roll3"] = roll3
        new[f"{c}_vs_roll7"] = sleep[c] - roll7   # 최근 평소 대비 오늘
    return pd.concat([sleep, pd.DataFrame(new, index=sleep.index)], axis=1)


def build_sleep_features(use_cache: bool = True) -> pd.DataFrame:
    cache = C.CACHE_DIR / "sleep_features.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    parts = []

    # 화면: 사용 비율, 사용량, 수면 블록(onset/wake/tst), 야간 각성(WASO 프록시)
    scr = _load_night("mScreenStatus", ["m_screen_use"])
    block = scr.groupby(["subject_id", "night"]).apply(_longest_off_block, include_groups=False)
    waso = scr[(scr["axis"] >= 0) & (scr["axis"] <= 6) & (scr["m_screen_use"] == 1)] \
        .groupby(["subject_id", "night"]).size().rename("slp_waso_screen_events")
    scr_agg = scr.groupby(["subject_id", "night"]).agg(
        slp_screen_on_ratio=("m_screen_use", "mean"),
        slp_screen_on_sum=("m_screen_use", "sum"),
    )
    block = block.rename(columns={"onset": "slp_onset_h", "wake": "slp_wake_h", "tst": "slp_tst_h"})
    parts += [block, scr_agg, waso]

    # 충전: 야간 충전 비율(취침 중 거치 프록시)
    chg = _load_night("mACStatus", ["m_charging"])
    parts.append(chg.groupby(["subject_id", "night"]).agg(slp_charge_ratio=("m_charging", "mean")))

    # 활동: 정지 비율(수면), 이동 비율
    act = _load_night("mActivity", ["m_activity"])
    act["still"] = act["m_activity"].isin(C.ACTIVITY_STILL).astype(float)
    parts.append(act.groupby(["subject_id", "night"]).agg(
        slp_still_ratio=("still", "mean"), slp_act_count=("still", "size")))

    # 걸음: 야간 총 걸음 + 워치기반 수면블록(걸음>0 = 깨어있음)
    pedo = _load_night("wPedo", ["step"])
    pedo["active"] = (pedo["step"] > 0).astype(float)
    parts.append(pedo.groupby(["subject_id", "night"]).agg(
        slp_step_sum=("step", "sum"), slp_step_active=("active", "sum")))
    step_block = pedo[pedo["step"] > 0].groupby(["subject_id", "night"])["axis"] \
        .apply(lambda s: _block_from_active(s.values))
    if not step_block.empty:
        step_block = step_block.unstack().rename(
            columns={"onset": "slp_onset_steps", "wake": "slp_wake_steps", "tst": "slp_tst_steps"})
        parts.append(step_block)

    # 심박: 안정심박·평균·변동 + 심박기반 수면블록(안정심박+10 이상 = 깨어있음)
    hr = _load_night("wHr", ["heart_rate"])
    arr = hr["heart_rate"].apply(lambda a: np.asarray(a, dtype=float))
    hr["hr_mean"] = arr.apply(lambda a: a.mean() if a.size else np.nan)
    hr["hr_min"] = arr.apply(lambda a: a.min() if a.size else np.nan)
    parts.append(hr.groupby(["subject_id", "night"]).agg(
        slp_hr_min=("hr_min", "min"), slp_hr_mean=("hr_mean", "mean"),
        slp_hr_std=("hr_mean", "std")))

    def _hr_block(g):
        thr = np.nanmin(g["hr_min"].values) + 10.0   # 그 밤의 안정심박 + 10bpm
        awake = g.loc[g["hr_mean"] >= thr, "axis"].values
        b = _block_from_active(awake)
        b["waso_hr"] = int(((g["axis"] >= b["onset"]) & (g["axis"] <= b["wake"])
                            & (g["hr_mean"] >= thr)).sum()) if np.isfinite(b["onset"]) else 0
        return b
    hr_block = hr.groupby(["subject_id", "night"]).apply(_hr_block, include_groups=False)
    hr_block = hr_block.rename(columns={"onset": "slp_onset_hr", "wake": "slp_wake_hr",
                                        "tst": "slp_tst_hr", "waso_hr": "slp_waso_hr"})
    parts.append(hr_block)

    # 조도: 야간 어둠(폰+워치)
    for nm, col, out in [("mLight", "m_light", "slp_mlight_mean"), ("wLight", "w_light", "slp_wlight_mean")]:
        lt = _load_night(nm, [col])
        parts.append(lt.groupby(["subject_id", "night"]).agg(**{out: (col, "mean")}))

    sleep = pd.concat(parts, axis=1)
    # 파생: 수면효율 프록시 = 1 - 화면사용비율 ; 입면지연 프록시 = onset(클수록 늦게 잠)
    sleep["slp_eff_proxy"] = 1 - sleep["slp_screen_on_ratio"]
    # 합의(consensus): 화면/걸음/심박 3종 수면블록의 중앙값 → 더 견고한 TST/onset/wake
    for base in ["tst", "onset", "wake"]:
        srcs = [f"slp_{base}_h", f"slp_{base}_steps", f"slp_{base}_hr"]
        srcs = [c for c in srcs if c in sleep.columns]
        sleep[f"slp_{base}_consensus"] = sleep[srcs].median(axis=1)
    # 센서간 불일치(추정 신뢰도 프록시): 화면 vs 심박 TST 차이
    if {"slp_tst_h", "slp_tst_hr"}.issubset(sleep.columns):
        sleep["slp_tst_disagree"] = (sleep["slp_tst_h"] - sleep["slp_tst_hr"]).abs()
    sleep = sleep.reset_index().rename(columns={"night": "sleep_date"})

    sleep = _add_temporal(sleep)
    sleep.to_parquet(cache, index=False)
    print(f"  saved {cache}  shape={sleep.shape}")
    return sleep


if __name__ == "__main__":
    s = build_sleep_features(use_cache=False)
    print(s.shape)
    print(s.columns.tolist())
    print(s[["subject_id", "sleep_date", "slp_onset_h", "slp_wake_h", "slp_tst_h",
             "slp_hr_min", "slp_charge_ratio"]].head(8).to_string())
