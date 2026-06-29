# ETRI Human Understanding AI 2026 — 라이프로그 기반 수면·감정·스트레스 예측

ETRI 휴먼이해 인공지능 논문경진대회(DACON, ICTC 2026 Workshop) 코드 저장소.
멀티모달 라이프로그(스마트폰·웨어러블 센서)로부터 7개 이진 지표의 **확률**을 예측한다.

> 본 README는 적용한 방법·피처·모델과 그 성능, 그리고 데이터 구조에서 도출한 연구 결과를 기록한다.
> 핵심 분석은 [§7 연구 결과·한계](#7-연구-결과--한계-분석)에 정리했다.

---

## 1. 과제 정의

각 행은 한 피험자의 하루(`subject_id`, `sleep_date`, `lifelog_date`)이며, 7개 이진 지표의 확률을 예측한다.

| 지표 | 의미 | 1의 뜻 | 라벨 정의 |
|------|------|--------|-----------|
| Q1 | 기상 직후 주관적 수면의 질 | 개인 평균보다 좋음 | **본인 전기간 평균 대비** |
| Q2 | 취침 직전 피로도 | 피로 낮음(좋음) | **본인 전기간 평균 대비** |
| Q3 | 취침 직전 스트레스 | 스트레스 낮음(좋음) | **본인 전기간 평균 대비** |
| S1 | 총수면시간(TST) | NSF 권장 충족 | 객관 임계값(7–9h) |
| S2 | 수면효율(SE) | NSF 권장 충족 | 객관 임계값(≥85%) |
| S3 | 입면지연(SOL) | NSF 권장 충족 | 객관 임계값(≤30m) |
| S4 | 수면 중 각성(WASO) | NSF 권장 충족 | 객관 임계값(≤20m) |

이 라벨 정의는 전략 전체를 좌우한다(공식 `data/ch2026_metrics_description.pdf` 해독):
- **Q1~Q3 (설문)**: 5점 척도를 **그 피험자 본인의 전체 기간 평균과 비교**해 이진화. 즉 "그날이
  평소보다 좋았나"가 본질이며, 피험자별 prevalence가 구조적으로 ≈0.5 → **본질적으로 거의 무작위**.
- **S1~S4 (수면센서)**: **Withings 침대형 수면 분석기**가 측정한 TST/SE/SOL/WASO가 미국수면재단(NSF)
  가이드라인을 충족하는지. 우리는 이 침대센서를 갖지 못하고, **손목 웨어러블+폰으로 역추정**만 가능.

## 2. 평가 산식 — Average Log-Loss

```
Score = (1/7) Σ_j  [ -(1/N) Σ_i ( y_ij·log p_ij + (1-y_ij)·log(1-p_ij) ) ]   (낮을수록 좋음)
```

정답을 맞히되 과도한 확신으로 틀리면 큰 벌점 → **잘 보정된(calibrated) 확률**이 핵심.
**Public = 테스트의 사전샘플 44%, Private = 100%.** 최종 점수는 Private 기준이며, 이 분리가 검증 전략에 큰 영향을 준다([§7](#7-연구-결과--한계-분석)).

## 3. 데이터 구성

- 라벨: `data/ch2026_metrics_train.csv`(450일) / 제출 양식 `data/ch2026_submission_sample.csv`(250일).
- 피험자 `id01`~`id10` 10명, **train·test 동일 인물**, 기간(2024-06~11) 겹침(인터리빙).
  → "처음 보는 사람"이 아니라 "같은 사람의 다른(주로 나중) 날"을 예측.
- 센서 12종(700일분, `data/ch2025_data_items/*.parquet`): 충전/활동/조도/화면/걸음/**심박(초단위 배열)**/
  오디오 장면/앱 사용/WiFi·BLE/GPS. 웨어러블(HR·걸음) 야간 커버리지는 test에서도 94~96%로 양호.

## 4. 접근 방법

### 4.1 핵심 통찰
1. **피험자 기준선이 지배적 신호.** 타깃이 개인 평균 대비라, 피험자별 prior만으로 강한 베이스라인.
   누수 없는 OOF prior를 피처로 주입하고, 모델 예측과 prior를 타깃별 최적 가중으로 블렌드.
2. **검증은 '시간'을 존중.** test는 각 피험자의 뒤쪽 날짜에 몰려 부분적 미래예측. 무작위 CV는 인접일
   누수로 OOF를 낙관시킨다(실측: 무작위 OOF 0.58 → 실제 LB 0.61). → **피험자별 연속 시간블록 CV**
   (`subject_time_blocked_folds`) 사용.

### 4.2 피처
- **일일 윈도우 통계**(`sensor_features.py`): 센서별 시각 윈도우(full/day/eve/night/morn)의 통계.
- **수면구간 탐지**(`sleep_features.py`): 화면OFF·충전·정지·저심박·무걸음으로 야간 수면블록 추정 → 입면/기상/TST/SE/각성 프록시.
- **중첩 센서**(`nested_features.py`): 오디오 장면·앱 사용·WiFi/BLE 수·GPS.
- **생리학적 수면추정**(`physio_features.py`, *본 대회에서 신규 추가*): 분 단위 actigraphy로
  수면/각성 hypnogram을 만들어 TST/SE/SOL/WASO/각성수를 직접 추정 + NSF 임계 피처 +
  **RR기반 제대로 된 HRV**(RMSSD/SDNN/pNN50, 취침 전·수면 중 윈도우) + 피험자내 편차/시간동역학.
- **시간 동역학**: 전날(lag)·최근평균(rolling)·추세 대비 편차(수면빚·규칙성).
- **피험자 z-score · 캘린더**.

### 4.3 모델 / 앙상블
- 타깃별 **LightGBM/XGBoost/CatBoost** 강정규화 앙상블 + OOF prior 블렌드 + 게이팅 캘리브레이션.
- 이후 다수의 스택/블렌드 계열(`public_aware_stack_blend`, `raw_timeline_target_model`,
  `guarded_lgbm_integration`, `constrained_target_blend` 등)로 소스 다양화.

## 5. 코드 구조 (요약)

```
src/
  config.py · cv.py · models.py            # 설정 / 정직 CV / 단일 적합기
  sensor_features.py · sleep_features.py    # 일일 통계 / 수면구간 (→ cache/*.parquet)
  nested_features.py · build_dataset.py     # 중첩센서 / 피처 결합·z-score·캘린더
  physio_features.py                        # ★신규: actigraphy 수면추정 + RR기반 HRV
  physio_eval.py · physio_eval2.py          # physio 단독/한계기여 OOF 평가
  final_robust_submission.py                # ★최종 강건 제출본 빌더
  train_ensemble.py · raw_timeline_*.py …   # 정식 파이프라인 / 소스·블렌드 계열
research/   # 실험 분석 노트(특히 AI_HANDOFF_FINDINGS_20260624.md = 최종 회고 근거)
```

## 6. 결과 (검증 평균 Log-Loss)

타깃별 OOF Log-Loss (public-best 기준):

| Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | 평균 |
|---|---|---|---|---|---|---|---|
| 0.585 | **0.693** | 0.666 | 0.489 | 0.563 | 0.529 | 0.625 | **0.5926** |

- **Q2 = 0.693 = ln2 → 완전 무작위.** 거대한 앙상블(0.5926)이 단순 subject_prior(0.5936)와 거의 동일.
- 최종 강건 후보 `global_raw_full_a0.80`(base를 raw_timeline 쪽 0.8 logit 블렌드): full-OOF **0.5835**
  (6/7 타깃 개선·Q2 −0.020·S2 −0.023), 폴드 표준편차 0.0161→**0.0088**, 최악폴드 0.619→0.599.

| 후보 | full-OOF | 폴드 σ | 최악폴드 |
|---|---|---|---|
| 피험자 prior 단독 | 0.5936 | — | — |
| base public-best(스택/블렌드) | 0.5926 | 0.0161 | 0.619 |
| **강건 블렌드 `global_raw_full_a0.80`** | **0.5835** | **0.0088** | **0.599** |

## 7. 연구 결과 · 한계 분석

데이터 구조와 다양한 실험에서 얻은 결과를 정리한다(근거: `research/AI_HANDOFF_FINDINGS_20260624.md`).

1. **검증(CV) 지표와 리더보드의 단절.** full-OOF를 0.5926→0.5835로 0.009 개선해도 Public 점수는
   사실상 무변동(0.5903→0.5907)이었다. Public이 테스트의 44%(약 110행)에 불과해 비대표이고,
   train↔test 분포 차이로 인해 어떤 CV 지표도 리더보드를 신뢰성 있게 예측하지 못했다. → 미세
   최적화는 전이되지 않으며, **full-OOF + 폴드 안정성으로 강건한 단일 제출을 고르는 것**이 합리적.
2. **Q1/Q2/Q3는 본질적으로 거의 무작위.** "본인 평균 대비" 정의상 동전던지기에 가깝다(Q2 OOF=0.693=ln2).
   HRV·캘리브레이션·전이성 가정 등 어떤 기법도 Q2/Q3를 의미 있게 낮추지 못했다.
3. **S1~S4의 정답은 데이터에 없는 센서에서 유래.** 침대형 Withings가 측정한 TST/SE/SOL/WASO 라벨을
   손목+폰으로 역추정하는 데는 천장이 있다. 신규 생리학적 수면추정 피처(actigraphy+RR기반 HRV)는
   S1/S2/S4에서만, full-OOF 기준 −0.004~−0.007 수준으로 기여했다(검증 상관: S1↔추정TST +0.31,
   S3↔resting HR −0.29). 기존 수면 피처와 상당부분 중복되어 한계 기여는 작았다.
4. **블렌딩 포화.** 대부분의 소스가 동일한 일일집계+피험자 prior에서 파생돼 상호 상관이 높아,
   스택/블렌드만으로는 없던 신호를 만들지 못했다. 가장 탈상관된 소스는 raw-timeline 모델로,
   이를 base와 0.8 logit 블렌드한 `global_raw_full_a0.80`이 full-OOF·폴드 안정성을 동시에 개선했다.

**요약:** 이 데이터의 현실적 천장은 full-OOF ~0.583 부근이며, 더 내리려면 추가 블렌딩이 아니라
근본적으로 새로운 신호가 필요하다. Q는 정의상 무작위, S는 측정 센서 부재라는 구조적 제약이 명확하다.

## 8. 환경

- Python 3.14, `.venv`. 주요 패키지: pandas, pyarrow, numpy, scikit-learn, lightgbm 4.6, xgboost 3.2, catboost 1.2.
- macOS는 lightgbm용 `brew install libomp` 필요. GPU 불필요(CPU 수십 초~수 분, 무거운 부분은 parquet 집계 → 캐시 재사용).
- 재현: `python -m src.physio_features`(피처 빌드) → `python -m src.physio_eval2`(physio 한계기여)
  → `python -m src.final_robust_submission`(최종 강건 제출본).
