"""교차검증 분할.

실제 과제는 '동일 10명 피험자의 다른 날'을 맞추는 것(train/test 피험자 동일, 기간 겹침).
따라서 subject-out 이 아니라 각 피험자의 날짜를 폴드로 분산시키는 subject-stratified KFold 를 사용한다.
각 폴드는 모든 피험자를 포함하되 서로 다른 날들을 검증에 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def subject_stratified_folds(meta: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """행별 폴드 인덱스(0..n_splits-1) 반환. 피험자별로 날짜순 셔플 후 라운드로빈 배정."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(meta), -1, dtype=int)
    for _, idx in meta.groupby("subject_id").groups.items():
        idx = np.array(idx)
        perm = rng.permutation(len(idx))
        fold[idx[perm]] = np.arange(len(idx)) % n_splits
    assert (fold >= 0).all()
    return fold
