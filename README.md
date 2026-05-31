# ETRI Human Understanding AI 2026 — 라이프로그 기반 수면·감정·스트레스 예측

ETRI 휴먼이해 인공지능 논문경진대회(DACON, ICTC 2026 Workshop)의 코드 저장소이다.
멀티모달 라이프로그(스마트폰·웨어러블 센서) 데이터로부터 7개의 이진 지표를 예측한다.

---

## 1. 과제 정의

각 행은 한 명의 피험자가 보낸 하루(`subject_id`, `sleep_date`, `lifelog_date`)이며,
그 하루에 대해 7개 이진 지표의 **확률**을 예측한다.

| 지표 | 의미 | 1의 뜻 |
|------|------|--------|
| Q1 | 기상 직후 주관적 수면의 질 | 개인 평균보다 좋음 |
| Q2 | 취침 직전 피로도 | 피로 낮음(좋음) |
| Q3 | 취침 직전 스트레스 | 스트레스 낮음(좋음) |
| S1 | 총수면시간(TST) 가이드라인 | 권장 충족 |
| S2 | 수면효율(SE) 가이드라인 | 권장 충족 |
| S3 | 입면지연(SOL) 가이드라인 | 권장 충족 |
| S4 | 수면 중 각성(WASO) 가이드라인 | 권장 충족 |

- **Q1~Q3 (설문)**: 5점 척도 응답을 **그 피험자 본인의 전체 기간 평균과 비교**해 이진화한 값이다.
  따라서 "그날이 그 사람 평소보다 좋았는가/나빴는가"가 본질이며, 피험자별 기준선이 매우 중요하다.
- **S1~S4 (수면센서)**: Withings 수면 분석기 기반 지표가 미국수면재단(NSF) 가이드라인을 충족하는지 여부이다.

## 2. 평가 산식

7개 지표 각각의 Log-Loss를 계산한 뒤 평균한다(**낮을수록 좋음**).

```
Score = (1/7) Σ_j  [ -(1/N) Σ_i ( y_ij·log p_ij + (1-y_ij)·log(1-p_ij) ) ]
```

Log-Loss는 정답에 가까운 확률을 얼마나 **자신 있게** 맞혔는지를 평가한다.
정답을 맞히되 과도한 확신으로 틀리면 큰 벌점을 받으므로, **잘 보정된(calibrated) 확률**을 내는 것이 핵심이다.
Public 점수는 테스트의 44% 표본, Private 점수는 100%로 산정된다.

## 3. 데이터 구성

- 라벨: `data/ch2026_metrics_train.csv` (450일) / 제출 양식 `data/ch2026_submission_sample.csv` (250일).
- 피험자는 `id01`~`id10` 10명이며, **train과 test에 동일한 10명**이 등장하고 기간(2024-06~11)도 겹친다(인터리빙).
  즉 "처음 보는 사람"이 아니라 "같은 사람의 다른 날"을 예측하는 문제다.
- 센서 12종(700일분, `data/ch2025_data_items/*.parquet`):
  - 스칼라: `mACStatus`(충전), `mActivity`(활동코드), `mLight`/`wLight`(조도),
    `mScreenStatus`(화면사용), `wPedo`(걸음·거리·속도·칼로리), `wHr`(심박 배열).
  - 중첩: `mAmbience`(오디오 장면 확률), `mUsageStats`(앱 사용시간),
    `mWifi`/`mBle`(주변 AP·기기), `mGps`(속도·고도; 위경도는 마스킹).

## 4. 접근 방법

### 4.1 핵심 통찰
1. **피험자 기준선이 지배적 신호다.** 타깃이 개인 평균 대비로 정의되므로,
   피험자별 타깃 평균(prior)만으로도 강한 베이스라인이 된다. 모델은 이 기준선에서 출발해
   센서 신호로 그날의 편차를 보정해야 한다. 본 코드는 누수 없는 OOF prior를 피처로 주입하고,
   모델 예측과 prior를 타깃별 최적 가중으로 블렌드한다.
2. **검증은 subject-out이 아니라 subject-stratified로 한다.** 테스트에 같은 피험자가 있으므로,
   각 피험자의 날들을 폴드에 분산시켜 "같은 사람의 다른 날"을 모사한다. (처음 보는 피험자를
   가정한 subject-out CV는 실제 과제와 어긋나 Local-LB 괴리를 만든다.)

### 4.2 피처
- **일일 윈도우 통계**(`sensor_features.py`): 각 센서를 하루의 시각 윈도우
  (full/day/eve/night/morn)별 mean·std·min·max·sum·count로 요약.
- **수면구간 탐지**(`sleep_features.py`): 화면 OFF·충전·정지·저심박·무걸음을 결합해
  야간 수면 블록을 추정 → 입면시각·기상시각·총수면시간·수면효율·각성 프록시(S1~S4 직격).
- **중첩 센서**(`nested_features.py`): 오디오 장면 확률(조용함/대화/음악),
  앱 사용시간·앱 개수, 주변 WiFi/BLE 수, GPS 속도/고도.
- **피험자 z-score**: 각 연속 피처를 피험자별 평균/표준편차로 표준화(개인 대비 편차).
- **캘린더**: 요일·주말·월·순환(sin/cos) 특성.

### 4.3 모델
- 타깃별 LightGBM(이진), 작은 데이터에 맞춘 강한 정규화(num_leaves=15, min_child_samples=25,
  feature_fraction=0.6, L1/L2), 폴드 검증 Log-Loss로 early stopping.
- 폴드 시드 3종 × 5-fold 멀티시드 배깅으로 OOF·테스트 예측을 안정화.
- 타깃별 (모델 vs prior) OOF 블렌드 가중을 그리드 탐색.

## 5. 코드 구조

```
src/
  config.py           # 경로(로컬/Colab 자동해석)·타깃·센서·윈도우 정의
  sensor_features.py  # parquet → (subject,date)×윈도우 통계  → cache/daily_features.parquet
  sleep_features.py   # 야간 수면구간 탐지                    → cache/sleep_features.parquet
  nested_features.py  # 중첩 센서(오디오/앱/WiFi/BLE/GPS)     → cache/nested_features.parquet
  build_dataset.py    # 라벨 행에 피처 결합 + z-score + 캘린더
  cv.py               # subject-stratified KFold
  train.py            # 정식 파이프라인(멀티시드 배깅+블렌드) → submissions/*.csv
  train_v2.py         # 단일 시드 실험본
  train_baseline.py   # 최소 베이스라인(prior 없음)
```

실행 방법과 점검 방법은 `test_readme.md`(개인 열람용)를 참고한다.

## 6. 결과(검증 OOF 평균 Log-Loss)

| 단계 | OOF | 비고 |
|------|-----|------|
| 단순 LGBM | 0.6095 | prior·수면·중첩 미사용 |
| + 피험자 prior 주입·블렌드 | 0.6008 | |
| + 수면구간 탐지 피처 | 0.5958 | S1·Q1 개선 |
| + 중첩 센서 피처 | 0.5887 | Q1 큰 개선 |
| + 멀티시드 배깅 | **0.5817** | 현재 |

참고 기준선: 누수 없는 피험자 prior 단독 ≈ 0.615.

## 7. 환경

- Python 3.14, 로컬 가상환경 `.venv`. 주요 패키지: pandas, pyarrow, numpy, scikit-learn, lightgbm 4.6.
- macOS에서 lightgbm 실행에 `brew install libomp` 필요.
- 학습은 GPU 불필요(수십 초~수 분, CPU). 무거운 부분은 센서 parquet 집계뿐이며 캐시로 재사용한다.
