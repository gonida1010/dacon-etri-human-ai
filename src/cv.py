"""교차검증 분할.

실제 과제는 '동일 10명 피험자의 다른 날'을 맞추는 것(train/test 피험자 동일, 기간 겹침).
따라서 subject-out 이 아니라 각 피험자의 날짜를 폴드로 분산시키는 subject-stratified KFold 를 사용한다.
각 폴드는 모든 피험자를 포함하되 서로 다른 날들을 검증에 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def subject_stratified_folds(meta: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """행별 폴드 인덱스(0..n_splits-1) 반환. 피험자별로 날짜순 셔플 후 라운드로빈 배정.

    주의: 무작위 분할은 같은 피험자의 인접일이 train/val 에 섞여 시간 누수 → OOF 낙관.
    실제 test 가 뒤쪽 날짜 블록이므로 subject_time_blocked_folds 사용을 권장.
    """
    rng = np.random.default_rng(seed)
    fold = np.full(len(meta), -1, dtype=int)
    for _, idx in meta.groupby("subject_id").groups.items():
        idx = np.array(idx)
        perm = rng.permutation(len(idx))
        fold[idx[perm]] = np.arange(len(idx)) % n_splits
    assert (fold >= 0).all()
    return fold


def subject_time_blocked_folds(meta: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """피험자별로 날짜를 정렬해 '연속 시간 블록'으로 폴드 배정(시간 누수 차단).

    각 피험자의 날들을 시간순 N 블록으로 나누고 블록=폴드. 검증일이 학습일과 시간적으로
    분리되어, test 가 뒤쪽 블록인 실제 구조를 더 정직하게 모사한다. seed 미사용(결정적).
    """
    fold = np.full(len(meta), -1, dtype=int)
    order = meta.sort_values(["subject_id", "sleep_date"]).index
    m = meta.loc[order]
    for _, idx in m.groupby("subject_id").groups.items():
        idx = np.array(idx)  # 이미 날짜순
        n = len(idx)
        # 균등 연속 블록 분할
        block = (np.arange(n) * n_splits // n)
        fold[idx] = block
    assert (fold >= 0).all()
    return fold
