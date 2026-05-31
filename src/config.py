"""공통 설정: 경로 해석(로컬/Colab 호환), 타깃, 센서, 시간 윈도우 정의."""
from __future__ import annotations

from pathlib import Path


def resolve_project_root() -> Path:
    """현재 디렉터리부터 위로 올라가며 `data/`를 가진 폴더를 프로젝트 루트로 사용.

    로컬과 Colab(예: /content/drive/MyDrive/...) 양쪽에서 동작하도록 절대경로/사용자명 미사용.
    """
    here = Path(__file__).resolve()
    for cand in [Path.cwd().resolve(), *here.parents]:
        if (cand / "data").is_dir():
            return cand
    return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
SENSOR_DIR = DATA_DIR / "ch2025_data_items"
CACHE_DIR = PROJECT_ROOT / "cache"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
CACHE_DIR.mkdir(exist_ok=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "ch2026_metrics_train.csv"
SAMPLE_CSV = DATA_DIR / "ch2026_submission_sample.csv"

TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date"]

EPS = 1e-15  # log-loss clipping (대회 산식과 동일하게 양끝 클리핑)

# 센서 파일명(접두사 제거 후) → 사용할 처리 방식
# 'numeric': 스칼라 수치 컬럼들을 윈도우 통계로 집계
# 'activity': 활동 코드 → 클래스 비율
# 'hr': heart_rate 배열 → 윈도우 통계
NUMERIC_SENSORS = {
    "mACStatus": ["m_charging"],
    "mLight": ["m_light"],
    "wLight": ["w_light"],
    "mScreenStatus": ["m_screen_use"],
    "wPedo": [
        "step", "step_frequency", "running_step", "walking_step",
        "distance", "speed", "burned_calories",
    ],
}

# 시간(시각) 윈도우 정의: 이름 -> (시작시, 끝시) [끝 배타적]
WINDOWS = {
    "full": (0, 24),
    "day": (9, 18),
    "eve": (18, 24),
    "night": (0, 6),
    "morn": (6, 9),
}

# Google Activity Recognition 코드
# 0 IN_VEHICLE, 1 ON_BICYCLE, 2 ON_FOOT, 3 STILL, 4 UNKNOWN, 5 TILTING, 7 WALKING, 8 RUNNING
ACTIVITY_STILL = {3}
ACTIVITY_MOVE = {2, 7, 8, 1}
ACTIVITY_VEHICLE = {0}


def sensor_path(name: str) -> Path:
    return SENSOR_DIR / f"ch2025_{name}.parquet"
