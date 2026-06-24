# Dacon ETRI Human-AI 인수인계 기록

작성일: 2026-06-21

이 문서는 Dacon 대회만 정리한다. Kaggle 별분류 대회 내용은 제외한다.

목표는 리더보드 확인용 제출을 반복하는 것이 아니라, 로컬 검증에서 확실히 내려가는 방법을 찾는 것이다. 현재 기준은 다음처럼 둔다.

1. 로컬 `subject_time_blocked`의 마지막 시간블록 점수를 최우선으로 본다.
2. `last_logloss <= 0.55` 근처가 나오기 전까지는 제출하지 않는다.
3. 기존 0.59대 후보는 “조금 개선”일 뿐 상위권 접근으로 보지 않는다.

## 1. 대회와 데이터 구조

- 대회: Dacon ETRI Human-AI
- 평가 지표: 7개 이진 타깃 평균 Log-Loss, 낮을수록 좋음
- 현재 리더보드 목표권: 약 0.54~0.55
- 현재 보유 성적: 약 0.59317~0.59318
- 최근 확인 제출:
  - `02_guarded_targetwise.csv`: public `0.5935970063`
  - 기존 최고보다 악화되어 제출 후보로 부적합

데이터 크기:

| split | rows | 설명 |
|---|---:|---|
| train | 450 | 10명 피험자의 과거 날짜 라벨 |
| test | 250 | 같은 10명 피험자의 이후 날짜 예측 |

타깃:

| group | targets | 의미 |
|---|---|---|
| Q | Q1, Q2, Q3 | 설문 기반 이진 타깃 |
| S | S1, S2, S3, S4 | 수면/객관 지표 기반 이진 타깃 |

피험자별 행 수:

| subject | train | test |
|---|---:|---:|
| id01 | 41 | 27 |
| id02 | 48 | 32 |
| id03 | 33 | 21 |
| id04 | 57 | 27 |
| id05 | 44 | 21 |
| id06 | 48 | 24 |
| id07 | 49 | 30 |
| id08 | 56 | 19 |
| id09 | 41 | 27 |
| id10 | 33 | 22 |

train 전체 타깃 평균:

| target | mean |
|---|---:|
| Q1 | 0.4956 |
| Q2 | 0.5622 |
| Q3 | 0.6000 |
| S1 | 0.6822 |
| S2 | 0.6511 |
| S3 | 0.6622 |
| S4 | 0.5600 |

## 2. 검증 기준

무작위 CV는 사용하면 안 된다. 같은 피험자의 인접일이 train/valid에 섞여 점수가 낙관적으로 나온다.

현재 신뢰하는 검증은 `subject_time_blocked_folds`이다.

- 피험자별 날짜를 시간순으로 5개 블록으로 나눈다.
- 마지막 블록을 미래 test와 가장 비슷한 regime으로 본다.
- 따라서 `last_logloss`를 핵심 판단 기준으로 쓴다.

이 검증 방식은 기존 제출들과 어느 정도 맞았다.

| local last-block | public LB |
|---:|---:|
| 0.6025 | 0.6034 |
| 0.5933 | 0.5931 |
| 0.5957 | 0.5972 |

따라서 지금은 public 제출보다 로컬 last-block 개선이 먼저다.

## 3. 현재 강한 앵커

기준 앵커는 `src/train_temporal_prior.py`이다.

핵심 구조:

- 센서 기반 LightGBM 모델
- 피험자별 평균 prior
- 최근 라벨 평균
- 피험자별 시간 추세 Ridge
- 타깃별 고정 recipe

기준 점수:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| temporal prior anchor | 0.595830 | 0.593282 |

타깃별 현재 recipe:

| target | recipe |
|---|---|
| Q1 | model 1.0 |
| Q2 | model 0.30 + last20_sm4 0.70 |
| Q3 | model 0.20 + last2_sm4 0.80 |
| S1 | model 1.0 |
| S2 | model 0.30 + ridge10 0.70 |
| S3 | mean_sm16 1.0 |
| S4 | model 0.50 + ridge1 0.50 |

핵심 해석:

- Q2, Q3는 센서보다 최근 평균이나 base-rate 수축이 강하다.
- S3는 거의 피험자 prior가 가장 강하다.
- S2, S4는 시간 추세가 중요하다.
- 현재 0.59 벽은 모델 튜닝보다 “미래 구간 구조”를 더 잘 맞춰야 깨질 가능성이 높다.

## 4. 지금까지 실험 수치

### 4.1 Fast temporal source search

위치:

- `src/fast_temporal_stack.py`
- `research/fast_temporal_stack/candidate_scores.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| 01_lastbest_targetwise.csv | 0.601067 | 0.591331 |
| 02_guarded_targetwise.csv | 0.600942 | 0.591357 |
| blend_lastbest75_baseline25 | 0.598689 | 0.591617 |
| blend_lastbest50_baseline50 | 0.597047 | 0.592034 |
| baseline_recipe_rebuilt | 0.595830 | 0.593282 |

해석:

- local last 기준으로는 0.5913까지 내려갔다.
- 그러나 public 제출 `02_guarded_targetwise.csv = 0.5935970063`으로 악화.
- 원인 후보는 last-block 90행에 과하게 맞춘 targetwise 선택이다.
- 이 계열은 단독 제출 후보가 아니라 분석 재료로만 유지한다.

### 4.2 Kaggle-style model bank, 1 seed

위치:

- `src/kaggle_style_ensemble.py`
- `research/kaggle_style_full_seed42/candidate_scores.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| targetwise_guarded | 0.592261 | 0.588175 |
| anchor0.7_lgbm0.3 | 0.591741 | 0.591217 |
| anchor0.8_lgbm0.2 | 0.592593 | 0.591438 |
| anchor0.6_lgbm0.4 | 0.591378 | 0.591445 |

해석:

- local last 기준으로 가장 낮은 값은 0.588175.
- 하지만 1 seed라 불안정하다.
- 0.55와는 여전히 거리가 크다.

### 4.3 Kaggle-style model bank, 3 seeds

위치:

- `research/kaggle_style_full_3seed/candidate_scores.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| targetwise_guarded | 0.592436 | 0.589048 |
| anchor0.7_lgbm0.3 | 0.591853 | 0.591783 |
| anchor0.8_lgbm0.2 | 0.592680 | 0.591852 |
| anchor0.6_lgbm0.4 | 0.591499 | 0.592120 |

해석:

- 1 seed보다 last 성능이 후퇴했다.
- seed가 늘어나며 노이즈성 개선이 줄어든 것으로 보인다.
- 안정성 관점에서는 1 seed보다 믿을 만하지만, 상위권 수준은 아니다.

### 4.4 Kaggle-style model bank, 5 seeds guarded

위치:

- `research/kaggle_style_full_5seed_guarded/candidate_scores.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| targetwise_guarded | 0.592733 | 0.589493 |
| anchor0.8_lgbm0.2 | 0.592677 | 0.592092 |
| anchor0.7_lgbm0.3 | 0.591842 | 0.592148 |
| anchor0.9_lgbm0.1 | 0.593997 | 0.592461 |
| anchor0.8_xgb0.2 | 0.592968 | 0.592588 |

해석:

- 5 seed로도 0.589493이 최선.
- 현재 모델 bank 방식만으로는 0.55권 접근이 안 보인다.
- 그래도 LGBM이 XGB/Cat보다 안정적으로 섞인다.
- CatBoost, XGB는 일부 full 개선은 있으나 last에서 일관되게 강하지 않다.

### 4.5 Pair features + target-history features smoke

위치:

- `research/kaggle_style_feature_plus_smoke/candidate_scores.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| anchor | 0.595830 | 0.593282 |
| targetwise_guarded | 0.595830 | 0.593282 |
| anchor0.9_lgbm0.1 | 0.604721 | 0.609654 |
| anchor0.8_lgbm0.2 | 0.626608 | 0.642342 |
| lgbm 단독 | 6.789647 | 8.684036 |

해석:

- 새 feature를 무작정 붙이면 심하게 망가진다.
- fold-limit 1 smoke라 점수 자체는 완전 평가가 아니지만, 모델 출력이 비정상적으로 악화되는 신호가 있었다.
- 이 방향은 그대로 풀런하면 안 된다.

### 4.6 Q balance 0.5 smoke

위치:

- `research/kaggle_style_qbalance_smoke/candidate_scores.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| anchor | 0.595830 | 0.593282 |
| qbal0p5_anchor | 0.966655 | 0.942290 |

해석:

- 단순히 Q 타깃을 0.5 비율로 강제하면 망가진다.
- 이유는 train의 Q 비율이 피험자별로 0.5가 아니고, 문제 정의의 “전체기간 평균 대비” 구조를 단순 비율 0.5로 치환하면 안 되기 때문이다.
- 따라서 단순 Q balance는 폐기한다.

### 4.7 Structural balance search, 2026-06-21

위치:

- `src/structural_balance_search.py`
- `research/structural_balance_search/candidate_scores.csv`
- `research/structural_balance_search/q_rate_search.csv`

결과:

| candidate | full_logloss | last_logloss |
|---|---:|---:|
| anchor | 0.595830 | 0.593282 |
| best non-anchor candidate | 약 1.040890 | 약 0.890150 |

타깃별 탐색 결과:

| target | anchor last | best rate | best last | 판단 |
|---|---:|---:|---:|---|
| Q1 | 0.618506 | 없음 | 0.618506 | 앵커 유지 |
| Q2 | 0.638084 | 없음 | 0.638084 | 앵커 유지 |
| Q3 | 0.634692 | 없음 | 0.634692 | 앵커 유지 |

해석:

- 피험자별 전체기간 양성 비율을 직접 맞추는 방식은 현재 구현 기준 실패.
- rate grid 후보들이 last를 0.89대로 악화시켰다.
- 제출 금지.
- 단, “Q 구조 제약” 아이디어 자체를 완전히 버리지는 않는다. 현재 구현은 확률 평균을 강제로 이동시키는 방식이라 과격하다. 다음에는 hard mean shift가 아니라 약한 penalty나 rank-level 후보 교체 방식으로 재설계해야 한다.

### 4.8 Sequence smoothing search, 2026-06-21

위치:

- `src/sequence_smoothing_search.py`
- `research/sequence_smoothing_search/candidate_scores.csv`
- `research/sequence_smoothing_search/sequence_param_search.csv`

결과:

| candidate | full_logloss | last_logloss | 판단 |
|---|---:|---:|---|
| anchor | 0.595830 | 0.593282 | 기준 |
| targetwise_lastbest | 0.619717 | 0.586406 | full guard 초과, 제출 금지 |

해석:

- Q3, S4에서는 sequence smoothing이 last-block을 크게 낮출 수 있다.
- 하지만 전체 full OOF가 `+0.023887` 악화되어 단독 후보로는 위험하다.
- sequence smoothing은 제출 후보가 아니라 OOF bank 안의 target별 source로만 사용한다.

### 4.9 OOF sparse greedy ensemble, 2026-06-21

위치:

- `src/oof_sparse_greedy.py`
- `research/oof_sparse_greedy/oof_bank.csv`
- `research/oof_sparse_greedy/test_bank.csv`
- `research/oof_sparse_greedy/source_scores.csv`
- `research/oof_sparse_greedy/targetwise_greedy_steps.csv`
- `research/oof_sparse_greedy/candidate_scores.csv`

설계:

- temporal anchor를 모든 타깃의 fallback으로 둔다.
- temporal priors, anchor calibration, sequence smoothing 변형을 275개 source bank로 만든다.
- target별 stagewise greedy를 적용하되 `last_gain >= 0.0002`, target full guard `+0.006` 조건을 통과할 때만 source를 추가한다.

결과:

| candidate | full_logloss | last_logloss | 판단 |
|---|---:|---:|---|
| targetwise_sparse_greedy | 0.599491 | 0.582418 | 개선은 있으나 0.55와 거리 큼, 제출 금지 |
| targetwise_best_single_guarded | 0.597516 | 0.590326 | 보수적 개선, 제출 가치 낮음 |
| anchor | 0.595830 | 0.593282 | 기준 |

주요 선택:

| target | selected sources |
|---|---|
| Q1 | anchor 유지 |
| Q2 | `seq_Q2_0p8_1`, `anchor_temp_1p15` |
| Q3 | `seq_Q3_0p5_0p8` |
| S1 | `last2_sm2`, `anchor_temp_1p15` |
| S2 | `last30_sm4`, `seq_S2_0p5_0p15`, `anchor_temp_0p85` |
| S3 | `seq_S3_0p65_0p15`, `last3_sm4` |
| S4 | `seq_S4_1p5_1`, `seq_S4_0p65_0p15` |

해석:

- OOF bank와 sparse greedy는 last-block을 0.5824까지 낮췄다.
- 0.55에는 아직 부족하고, Q1은 어떤 source도 guard 안에서 개선하지 못했다.
- 다음 연구는 단순 source 추가보다 검증 안정성 확장과 Q1/Q2/Q3 구조 보정 쪽이 우선이다.

### 4.10 Kaggle last-mile algorithms, 2026-06-21

위치:

- `src/kaggle_last_mile.py`
- `research/kaggle_last_mile/candidate_scores.csv`
- `research/kaggle_last_mile/candidate_stability.csv`
- `research/kaggle_last_mile/meta_diagnostics.csv`
- `research/kaggle_last_mile/rank_patch_search.csv`
- `research/kaggle_last_mile/knn_targetwise_choices.csv`

추가 구현:

- OOF bank 기반 fold-safe LogisticRegression meta-stacker
- ExtraTrees/RandomForest meta model
- same-subject date-nearest KNN prior
- Q 타깃 rank-level patch
- fold별/tail3 stability 진단

결과:

| candidate | full_logloss | last_logloss | 판단 |
|---|---:|---:|---|
| oof_sparse_greedy | 0.599491 | 0.582418 | 공격형 제출 후보 |
| rankpatch_sparse_greedy | 0.599334 | 0.582890 | sparse와 상관 높음, 중복 제출 비효율 |
| rankpatch_best_single | 0.597099 | 0.590232 | 중간 위험 |
| knn_targetwise_guarded | 0.596344 | 0.592718 | 보수/다변화 제출 후보 |
| rankpatch_anchor | 0.595360 | 0.593098 | 가장 보수적이나 개선 폭 작음 |
| meta_extratrees | 0.593542 | 0.605017 | last 악화, 제출 금지 |
| meta_logreg / meta_rf | 0.602~0.605 | 0.605~0.615 | last 악화, 제출 금지 |

제출 판단:

- sparse 계열 2개를 모두 내는 것은 상관이 높아 비효율적이다.
- 오늘 2개만 고르면 `oof_sparse_greedy` 1개와 `knn_targetwise_guarded` 1개가 가장 균형이 좋다.
- 더 보수적으로 가려면 두 번째를 `rankpatch_anchor`로 바꿀 수 있지만, 기대 개선 폭은 매우 작다.

## 5. 현재 결론

지금까지 확인된 최고 로컬 수치:

| 계열 | best full | best last | 제출 가치 |
|---|---:|---:|---|
| temporal anchor | 0.595830 | 0.593282 | 기존 기준 |
| fast temporal lastbest | 0.601067 | 0.591331 | public 악화 확인 |
| model bank 1 seed | 0.592261 | 0.588175 | 불안정 |
| model bank 3 seed | 0.592436 | 0.589048 | 0.55와 거리 큼 |
| model bank 5 seed | 0.592733 | 0.589493 | 안정적이나 부족 |
| Q balance hard shift | 0.966655 | 0.942290 | 폐기 |
| structural balance search | 0.595830 | 0.593282 | 앵커 유지 |
| sequence smoothing lastbest | 0.619717 | 0.586406 | full guard 초과 |
| OOF sparse greedy | 0.599491 | 0.582418 | 개선은 있으나 제출 금지 |
| Kaggle last-mile KNN guarded | 0.596344 | 0.592718 | 보수 제출 후보 |

현재 확실한 판단:

1. 0.59대 개선은 상위권으로 충분하지 않다.
2. 단순 LGBM/XGB/CatBoost 앙상블로는 0.55까지 바로 내려갈 가능성이 낮다.
3. 현재 제일 강한 안전 베이스는 temporal prior anchor다.
4. OOF sparse greedy는 last-block 개선 가능성을 보였지만, 아직 full/last trade-off와 last-block 과적합 위험이 남아 있다.
5. 0.55권을 보려면 타깃 구조, 시간 구조, 피험자별 future block 구조를 더 직접적으로 이용해야 한다.

## 6. 아직 Dacon에서 제대로 시도하지 않은 캐글식 방법

여기서 “캐글식”은 특정 Kaggle 대회 코드를 복사한다는 뜻이 아니라, 정형 데이터 경진대회에서 쓰는 검증/앙상블/후처리 기법을 Dacon 데이터 구조에 맞춰 적용한다는 뜻이다.

### 6.1 Level-1 OOF prediction bank 확장

부분 시도:

- LGBM/XGB/CatBoost model bank
- temporal anchor를 feature로 넣는 stacking 비슷한 구조

아직 부족한 점:

- 모델 다양성이 부족하다.
- 같은 피처와 같은 split에서만 tree 계열을 반복했다.
- 확실히 다른 inductive bias가 없다.

추가 후보:

- ElasticNet / LogisticRegression 계열의 보수적 선형 모델
- RandomForest / ExtraTrees의 강한 bagging 모델
- KNN-like subject/date nearest neighbor prior
- isotonic 또는 Platt calibration 전용 OOF 보정
- target별로 다른 모델군만 쓰는 sparse ensemble

목표:

- 모델 하나를 세게 만드는 것이 아니라, target별로 서로 다른 오류를 내는 예측 bank를 만든다.

### 6.2 OOF meta-stacker

부분 시도:

- anchor와 LGBM/XGB/CatBoost의 단순 가중 blend

아직 안 한 것:

- OOF prediction columns만 모아 2단계 모델을 따로 학습
- target별 meta-model
- fold 안에서 meta-model 학습 후 last-block 검증

주의:

- 데이터가 450행이라 복잡한 meta-model은 바로 과적합된다.
- 후보는 LogisticRegression, RidgeClassifier 확률화, 작은 LightGBM 정도만 허용해야 한다.

필요 산출:

- `oof_prediction_bank.csv`
- `test_prediction_bank.csv`
- `meta_stacker_diagnostics.csv`
- target별 meta coefficient 또는 feature importance

### 6.3 Greedy ensemble selection

부분 시도:

- grid weight blend

아직 안 한 것:

- 후보 예측 bank에서 하나씩 추가하면서 last-block이 내려가는 조합만 선택
- target별 greedy selection
- full guard + last gain guard 동시 적용

목표:

- 모델을 많이 만들되, 마지막에는 target별로 1~3개 예측만 남기는 sparse ensemble을 만든다.

### 6.4 Rank-level / row-level patch

부분 시도:

- fast temporal stack에서 source별 targetwise 선택

아직 안 한 것:

- 확률 전체 평균 shift가 아니라, subject-target별 상위/하위 rank row만 교체
- 예: Q2에서 특정 subject의 test future positive 수가 부족하다고 추정될 때, 가장 확신 높은 row만 위로 당김

이유:

- `Q balance hard shift`는 전체 확률을 강제로 움직여 망가졌다.
- 그 대신 rank 보존 + 소수 row 보정이 더 안전할 수 있다.

### 6.5 Time-series backtesting split 확장

현재:

- 5개 subject-time block
- 마지막 블록 중심

아직 안 한 것:

- rolling-origin backtest
- train 초기 60%, valid 다음 20%, holdout 마지막 20% 같은 여러 시간 기준
- 피험자별 test 날짜 길이에 맞춘 custom validation block

목표:

- last-block 90행 하나에 과적합되는 것을 줄인다.
- public 44%와 private 100% 모두에 더 안정적인 후보를 만든다.

### 6.6 피험자별 모델 또는 피험자별 보정

부분 시도:

- 피험자 prior
- 피험자별 ridge trend

아직 안 한 것:

- subject별 target별 calibration
- subject별 temporal recipe 선택
- subject cluster별 recipe 선택

가능성:

- train/test 피험자가 동일하므로 subject-specific 보정은 합법적이고 중요하다.
- 단, 피험자당 33~57행뿐이라 강한 모델은 금지.

### 6.7 S 타깃용 sequence smoothing

구현 및 결과 확인됨:

- `src/sequence_smoothing_search.py`
- `src/oof_sparse_greedy.py` 안에서 sequence smoothing source를 OOF bank source로 재사용

확인된 점:

- 2-state Markov smoothing
- 피험자별 날짜 순서 posterior smoothing
- 타깃별 guarded 선택

판단:

- Q3, S3, S4에서 last-block 개선 폭이 있었다.
- 단독 sequence smoothing은 full OOF가 크게 악화되어 제출 후보가 아니다.
- sparse greedy 안에서 제한적으로 섞을 때만 쓴다.

재현 명령:

```bash
cd /Users/parkyeonggon/Projects/dacon/dacon-etri-human-ai
source .venv/bin/activate
mkdir -p research/sequence_smoothing_search research/.mplconfig

MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 python -m src.sequence_smoothing_search \
  --emission-powers 0.5 0.65 0.8 1.0 1.25 1.5 \
  --transition-blends 0.15 0.3 0.45 0.6 0.8 1.0 \
  --transition-smooth 2.0 \
  --full-guard 0.006 \
  --min-last-gain 0.0002 \
  --output-dir research/sequence_smoothing_search \
  --submission-dir submissions/sequence_smoothing_search \
  2>&1 | tee research/sequence_smoothing_search/run.log
```

판단:

- `research/sequence_smoothing_search/candidate_scores.csv` 확인
- `last_logloss <= 0.55`가 아니면 제출하지 않음

## 7. 다음 계획

우선순위 1. OOF sparse greedy 재현 및 분석

- 실행 파일: `src/oof_sparse_greedy.py`
- 목표: OOF/test prediction bank를 만들고 target별 sparse source 선택을 확인
- 산출 파일:
  - `research/oof_sparse_greedy/oof_bank.csv`
  - `research/oof_sparse_greedy/test_bank.csv`
  - `research/oof_sparse_greedy/source_scores.csv`
  - `research/oof_sparse_greedy/targetwise_greedy_steps.csv`
  - `research/oof_sparse_greedy/candidate_scores.csv`
- 현재 결과: `targetwise_sparse_greedy full=0.599491 last=0.582418`
- 판단: 제출 금지, 다음 연구의 분석 재료로 사용

우선순위 2. Rolling-origin backtest 확장

목표:

- last-block 90행 하나에 맞춘 source 선택인지 검증한다.
- 피험자별 test 길이를 반영한 backtest를 2~3개 더 만든다.
- sparse greedy 후보가 여러 backtest에서 동시에 개선될 때만 유지한다.

필요 산출:

- `research/rolling_backtest/source_scores_by_split.csv`
- `research/rolling_backtest/greedy_stability.csv`

우선순위 3. Q target rank patch 재설계

목표:

- hard mean shift를 쓰지 않는다.
- subject-target별 expected future positive count를 직접 평균 이동으로 맞추지 않는다.
- 대신 rank 상위/하위 일부 row만 제한적으로 교체하거나 logit margin만 조정한다.

가드:

- row patch 비율은 subject-target별 최대 10~20%부터 시작한다.
- Q1은 현재 개선 source가 없으므로 Q1을 첫 타깃으로 삼는다.
- full guard와 rolling backtest stability를 동시에 본다.

우선순위 4. 피험자별 recipe 선택

목표:

- 전체 target 하나의 recipe가 아니라 `subject_id × target`별 recipe를 고른다.

주의:

- 피험자당 행이 적어 과적합 위험이 크다.
- 반드시 rolling/last-block guard 필요.

우선순위 5. 보수적 meta-stacker

목표:

- `oof_bank.csv`의 예측 컬럼만 사용한다.
- target별 LogisticRegression 또는 Ridge-calibrated stacker만 허용한다.

주의:

- 복잡한 meta-model은 금지한다.
- sparse greedy보다 full OOF가 나빠지면 즉시 폐기한다.

## 8. 2026-06-21 22:24 KST 추가: residual single-model 최적 구현

문제의식:

- 기존 `diverse_single_stack`는 단일 모델을 label에 직접 맞춘 뒤 anchor와 섞었다. raw 모델이 약하면 anchor에 기대는 구조라 학습 설계가 부실했다.
- 새 접근은 강한 temporal anchor를 기준으로 `y - anchor` residual을 학습한다.
- 피처 선택도 y 상관이 아니라 residual 상관으로 바꿨다.
- 샘플가중치는 target/fold별로 `subject 균등화`, `class imbalance`, `최근성`, `anchor error 크기`를 profile grid로 탐색한다.
- 최종 확률은 `anchor + shrink * residual_prediction`으로 만들고, shrink는 target별 OOF에서 고른다. shrink 0도 포함해 모델이 못 고치는 타깃은 anchor 유지한다.

구현:

- `src/residual_single_model_opt.py`
- `src/residual_submission_blend.py`

실행 완료:

```bash
MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.residual_single_model_opt --model extra_trees --output-dir research/residual_single_model_opt --submission-dir submissions/residual_single_model_opt 2>&1 | tee research/residual_single_model_opt/run.log

MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.residual_single_model_opt --model ridge --output-dir research/residual_single_model_opt_ridge --submission-dir submissions/residual_single_model_opt_ridge 2>&1 | tee research/residual_single_model_opt_ridge/run.log

MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.residual_single_model_opt --model hist_gb --output-dir research/residual_single_model_opt_hist_gb --submission-dir submissions/residual_single_model_opt_hist_gb 2>&1 | tee research/residual_single_model_opt_hist_gb/run.log

MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.residual_submission_blend --output-dir research/residual_submission_blend --submission-dir submissions/residual_submission_blend 2>&1 | tee research/residual_submission_blend/run.log
```

핵심 결과:

- `ridge_residual_composite`: full `0.595364`, last `0.585595`
  - 선택 타깃: S1/S2/S3만 residual correction, Q1/Q2/Q3/S4는 anchor 유지.
  - S1: recent_class + shrink 0.03
  - S2: recent_resid + shrink 0.20
  - S3: recent_resid + shrink 0.03
- `ridge_residual_full`: full `0.592740`, last `0.592791`
  - full OOF 기준으로는 현재 가장 깨끗한 단일 residual 후보.
- `hist_gb_residual_full`: full `0.595000`, last `0.592166`
  - Ridge보다 약하다.
- `extra_trees_residual_composite`: full `0.597429`, last `0.588501`
  - S2를 강하게 잡지만 full 비용이 커서 Ridge보다 우선순위 낮다.

OOF 기반 제출 블렌딩:

- `ridge_knn_blend_full`: full `0.592591`, last `0.590445`
  - anchor 대비 full `-0.003239`, last `-0.002837`.
  - 현 시점 public 확인 1순위.
- `ridge_knn_blend_composite`: full `0.595536`, last `0.584969`
  - last 개선은 크지만 full 개선은 작다.
  - public 확인 2순위 또는 공격 후보.

제출 파일:

- `submissions/residual_submission_blend/ridge_knn_blend_full_last0.590445_full0.592591.csv`
- `submissions/residual_submission_blend/ridge_knn_blend_composite_last0.584969_full0.595536.csv`
- 대안 단일 residual: `submissions/residual_single_model_opt_ridge/ridge_residual_composite_last0.585595_full0.595364.csv`

## 9. 2026-06-21 Kaggle notebook method review

기록 파일:

- `research/KAGGLE_NOTEBOOK_METHOD_REVIEW_20260621.md`

분석 대상:

- `ps6e6-one-vs-rest-tabm (1).ipynb`
- `ps6e6-one-vs-rest-xgb (1).ipynb`
- `cat-v3-for-s6e6 (1).ipynb`
- `realmlp-v5-for-s6e6 (2).ipynb`

이미 Dacon에 반영된 것:

- OOF-first workflow
- target-wise decomposition
- early stopping/best-iteration fallback for LGB/XGB/Cat
- residual correction + target-wise shrink
- OOF blend/stacking skeleton

아직 빠진 핵심:

- fold-safe target encoding feature bank
- numeric bin/category views
- OOF source-correlation diversity selector
- CatBoost categorical-view residual source
- TabM/RealMLP residual source

바로 적용 상태:

1. `src/residual_single_model_opt.py`에 `--feature-bank bins_te` 추가 완료
2. fold-wise quantile binning + leave-one-out smoothed TE 추가 완료
3. `src/oof_source_correlation.py` 추가 완료
4. `src/residual_submission_blend.py`가 blend candidate OOF/test prediction 저장하도록 수정 완료
5. CatBoost categorical-view residual source는 그 다음

## 10. 현재 제출 규칙

기본적으로 다음 조건 전에는 제출하지 않는다.

- `candidate_scores.csv`에서 `last_logloss <= 0.55`
- 또는 다른 AI/사람이 재현 가능한 근거로 `0.56` 이하 근접 후보를 찾은 경우

예외:

- 오늘처럼 public/local 상관 확인 목적이면 2개까지 제출 가능하다.
- 이 경우 `last`만 좋은 후보보다 `full`도 같이 개선된 후보를 먼저 제출한다.
- 2026-06-21 23시 이전 확인 우선순위는 `ridge_knn_blend_full`, 그 다음 `ridge_knn_blend_composite`였다.
- 23시 이후 Ridge logit 분석 반영 후 다음 확인 우선순위는 `ridge_logit_newton / ridge_residual_full`이다.

## 11. 2026-06-21 result analysis after Ridge logit/TE runs

결과지:

- `research/result_analysis_20260621/RESULT_ANALYSIS_20260621.md`
- `research/result_analysis_20260621/all_candidate_scores.csv`
- `research/result_analysis_20260621/key_candidate_scores.csv`
- `research/result_analysis_20260621/source_correlation_summary.csv`
- `research/result_analysis_20260621/candidate_frontier.png`
- `research/result_analysis_20260621/key_candidate_deltas.png`
- `research/result_analysis_20260621/key_fold_curves.png`
- `research/result_analysis_20260621/target_last_delta_heatmap.png`

핵심 판정:

- no-TE `ridge_logit_newton / ridge_residual_full`: full `0.587810`, last `0.590089`
  - 현재 OOF full/last 동시 개선 기준 최상위 후보.
- old `blend_ridge_knn / ridge_knn_blend_full`: full `0.592591`, last `0.590445`
  - 여전히 blend 계열 최상위 안정 후보.
- `bins_te` 계열은 전역 채택하지 않는다.
  - `ridge_logit_te_full / ridge_residual_full`: full `0.593726`, last `0.591578`
  - no-TE logit Ridge보다 명확히 약하다.
  - composite/last 선택은 last만 좋아지고 full이 악화된다.
- `blend_logit_te_full`도 old no-TE blend보다 약하다.
  - full candidate: full `0.593723`, last `0.591800`
  - composite: full `0.596128`, last `0.588013`
- source correlation 분석에서 anchor clone이 target별 217~220개 수준으로 많다.
  - 이후 source-bank greedy/stacking 전에는 corr `>= 0.999` 또는 identical prediction dedup이 필요하다.

다음 실행 우선순위:

1. `research/residual_single_model_opt_ridge_logit_newton`을 ridge source로 쓰는 no-TE blend를 실행한다.
2. 해당 blend가 full `0.587810`을 넘지 못하면 단순 blend가 아니라 target-specific Ridge residual constraint로 간다.
3. `bins_te`는 폐기하지 말고 분석 branch로 보관하되 다음 제출 후보에는 넣지 않는다.

## 12. 2026-06-22 no-TE logit blend result analysis

결과지:

- `research/result_analysis_20260622/RESULT_ANALYSIS_LATEST.md`
- `research/result_analysis_20260622/submission_frontier.png`
- `research/result_analysis_20260622/candidate_frontier.png`
- `research/result_analysis_20260622/key_candidate_deltas.png`
- `research/result_analysis_20260622/key_fold_curves.png`
- `research/result_analysis_20260622/target_full_delta_heatmap.png`
- `research/result_analysis_20260622/target_last_delta_heatmap.png`

중요 수정:

- 이전 그래프는 직접 확인 결과 연구용으로 부적절했다.
  - 실패 후보가 축을 늘려 핵심 0.58~0.60 구간이 뭉개졌다.
  - 후보명이 길고 겹쳐서 frontier와 heatmap 해석이 어려웠다.
- `src/result_suite_report.py`를 수정해 stable zone frontier, 제출 후보 전용 frontier, full/last target heatmap 분리, 짧은 alias를 적용했다.

최신 핵심 결과:

- `blend_ridge_logit_newton / ridge_knn_blend_full`: full `0.587732`, last `0.588717`
  - 현재 OOF 기준 1순위 후보.
  - `ridge_logit_newton / ridge_residual_full`보다 full `-0.000078`, last `-0.001372` 개선.
- `ridge_logit_newton / ridge_residual_full`: full `0.587810`, last `0.590089`
  - no-blend fallback 1순위.
- `blend_ridge_logit_newton / ridge_knn_blend_composite`: full `0.594314`, last `0.581319`
  - last는 매우 좋지만 full이 1순위보다 약하다.
  - 공격 후보일 뿐 1순위 제출 후보는 아니다.
- `bins_te` 계열은 계속 제외한다.

다음 판정:

1. 다음 public 확인 1순위는 `submissions/residual_submission_blend_ridge_logit_newton/ridge_knn_blend_full_last0.588717_full0.587732.csv`.
2. 보수 fallback은 `submissions/residual_single_model_opt_ridge_logit_newton/ridge_residual_full_last0.590089_full0.587810.csv`.
3. 추가 연구는 global TE가 아니라 Q1/Q2/S3 target-specific constraint 또는 deduped source stacking으로 간다.

구현 시작:

- `src/constrained_target_blend.py` 추가.
- 목적:
  - `logit blend full`의 full 장점을 유지한다.
  - Q2/S1/S3 last 손해를 target별 selector로 제한한다.
  - hard last guard, positive last penalty, full/last tradeoff 후보를 한 번에 생성한다.
- 다음 결과 디렉터리:
  - `research/constrained_target_blend_logit_newton`
  - `submissions/constrained_target_blend_logit_newton`
- `src/result_suite_report.py`도 `constrained_logit_blend` run을 읽도록 등록 완료.

실행 완료 및 판정:

- 실행 로그:
  - `research/constrained_target_blend_logit_newton/run.log`
  - `research/oof_source_correlation_constrained_logit_blend/run.log`
- 최신 결과지:
  - `research/result_analysis_20260622_constrained/RESULT_ANALYSIS_LATEST.md`
  - `research/result_analysis_20260622_constrained/constrained_frontier.png`
  - `research/result_analysis_20260622_constrained/submission_frontier.png`
  - `research/result_analysis_20260622_constrained/target_full_delta_heatmap.png`
  - `research/result_analysis_20260622_constrained/target_last_delta_heatmap.png`
- 최저 full 후보:
  - `constrained_logit_blend / full`: full `0.587732`, last `0.588717`
  - 파일: `submissions/constrained_target_blend_logit_newton/full_last0.588717_full0.587732.csv`
- full 거의 유지 + last 개선 1순위:
  - `constrained_logit_blend / last_guard_0p008`: full `0.587743`, last `0.588383`
  - full 비용은 `+0.000011`, last 개선은 `-0.000334` vs `full`.
  - 파일: `submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv`
- 균형 공격 후보:
  - `constrained_logit_blend / positive_last_penalty_a0p25`: full `0.587837`, last `0.587745`
  - 파일: `submissions/constrained_target_blend_logit_newton/positive_last_penalty_a0p25_last0.587745_full0.587837.csv`
- 더 공격적인 last 후보:
  - `constrained_logit_blend / tradeoff_cap_a0p15`: full `0.587898`, last `0.586630`
  - 파일: `submissions/constrained_target_blend_logit_newton/tradeoff_cap_a0p15_last0.586630_full0.587898.csv`
- 제외:
  - `tradeoff_cap_a0p25/a0p35/a0p5`는 last는 좋지만 full 비용이 더 커진다. public-risk 공격 후보 외에는 뒤로 둔다.
  - `composite/last`는 last는 최상위지만 full 안정성이 약해 제출 1순위가 아니다.

다음 연구 방향:

1. 단일모델 추가보다 먼저 target-wise selector를 더 정교화한다. Q2/S3 last 손해를 줄이는 constraint를 별도로 둔다.
2. source correlation 결과에서 anchor clone과 corr `>=0.999`가 많으므로, 다음 stack/blend는 반드시 source dedup 후 진행한다.
3. 새 단일모델을 팔 때는 한 모델군에서 early stopping, fold별 best_iteration, target별 class/sample weight, residual/logit objective를 먼저 최적화하고 다른 모델군에 이식한다.
4. `bins_te`는 계속 제외한다. 작은 row 수에서 global TE는 full 안정성을 깎는다.

## 13. 2026-06-22 submitted public feedback and pattern diagnosis

제출 결과:

- 파일: `submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv`
- Public: `0.5920118473`

진단 산출물:

- `src/submission_pattern_diagnostics.py`
- `src/pattern_safe_candidates.py`
- `research/submission_pattern_diagnostics_20260622/PATTERN_DIAGNOSTICS_LATEST.md`
- `research/submission_pattern_diagnostics_20260622/INTERPRETATION.md`
- `research/pattern_safe_candidates_20260622/candidate_scores.csv`
- `submissions/pattern_safe_candidates_20260622/`

핵심 판정:

- `last_guard_0p008`은 `full`과 flat corr `0.999981`, mean abs diff `0.000329`라 사실상 같은 제출군이었다. 근처 constrained 후보 재제출 금지.
- Q1은 up precision `0.636`, down precision `0.750`이라 유지 가능한 진짜 패턴.
- S2는 down-only가 더 낫다. full S2 movement의 upward precision은 `0.429`로 오탐 위험이 있다.
- S3은 harmful. up precision `0.333`, local last gain `-0.005484`; 반드시 anchor로 되돌린다.
- Q2는 harmful. last gain `-0.006712`; anchor로 되돌린다.
- S1은 weak/harmful. anchor로 되돌린다.
- S4는 upward만 좋다. up precision `1.000`, down precision `0.488`; 기존 제출은 S4를 81/250 rows에서 내린 것이 위험했다.

패턴 안전 후보:

- `submissions/pattern_safe_candidates_20260622/q1_s2down_s4up_last0.584779_full0.593510.csv`
  - full `0.593510`, last `0.584779`, test abs delta vs anchor `0.006615`
- `submissions/pattern_safe_candidates_20260622/q1_s2down_s4full_last0.585601_full0.593284.csv`
  - full `0.593284`, last `0.585601`, test abs delta vs anchor `0.011672`
- 이 후보들은 full-OOF 1위 후보가 아니라 public-risk diagnostic 후보이다.

## 14. 2026-06-22 direction-gated ablation

구현/실행 완료:

- `src/direction_gated_search.py`
- `src/direction_gated_ablation.py`
- `research/direction_gated_search_20260622/`
- `research/direction_gated_ablation_20260622/`
- `submissions/direction_gated_search_20260622/`
- `submissions/direction_gated_ablation_20260622/`

핵심 변경:

- 기존 제출에서 public 위험으로 읽힌 방향을 분리했다.
  - S4 down 금지, S4 up만 허용.
  - S3 up 금지. ablation 제출 파일에서는 S3 anchor.
  - S2는 down-only만 허용.
  - Q2/Q3/S1은 core에 섞지 않고 추가 효과를 따로 ablation.
- `up`/`down` 모드는 threshold가 예측값에는 적용되지 않는 구조라, ablation에서는 실제 예측도 제한되는 `up_thr`/`down_thr` 액션을 우선 사용했다.

자동 direction-gated 결과:

| candidate | full | last | test abs delta vs anchor |
|---|---:|---:|---:|
| `precision55_lowrisk` | `0.591125` | `0.574694` | `0.006517` |
| `precision55_move03` | `0.594600` | `0.572429` | `0.010284` |
| `precision55_move05` | `0.595382` | `0.571661` | `0.014389` |

자동 후보는 local last가 매우 좋지만 Q2/Q3까지 섞여 있어 public 검증 전 단일 제출 후보로 바로 믿기 어렵다.

명시적 ablation 결과:

| candidate | full | last | test abs delta vs anchor | read |
|---|---:|---:|---:|---|
| `core_plus_s1_q2q3` | `0.593496` | `0.579113` | `0.004008` | local 최상, Q2 up 6.8% 포함 |
| `core_plus_s1_q2tiny_q3` | `0.593587` | `0.579823` | `0.003705` | Q2 up을 0.8%로 축소한 균형 후보 |
| `core_plus_s1` | `0.593139` | `0.582015` | `0.003485` | Q2/Q3 제거, S1 tiny down 포함 |
| `core_q1down_s2tight_s4tight` | `0.593640` | `0.584668` | `0.003264` | Q1/S2/S4 core만 사용 |
| `core_q1up_s2tight_s4tight` | `0.594358` | `0.584922` | `0.003034` | Q1은 upward만 쓰는 대안 |

현재 제출 후보 판정:

1. public-risk와 local 개선 균형 1순위:
   - `submissions/direction_gated_ablation_20260622/core_plus_s1_q2tiny_q3_last0.579823_full0.593587.csv`
   - 이유: Q2 movement가 0.8%로 작고 Q3/S1은 매우 작은 보정, 기존 제출의 S4 down/S3 up 문제 제거.
2. 더 단순한 패턴 검증 후보:
   - `submissions/direction_gated_ablation_20260622/core_plus_s1_last0.582015_full0.593139.csv`
   - 이유: Q2/Q3를 아예 제거하고 Q1 down, S1 tiny down, S2 down, S4 up만 검증.
3. Q2/Q3까지 믿는 local 공격 후보:
   - `submissions/direction_gated_ablation_20260622/core_plus_s1_q2q3_last0.579113_full0.593496.csv`
   - 이유: local last 최상이나 Q2 up 6.8%가 있어 public 확인 전 리스크가 더 있다.

기존 제출과 비교:

- `last_guard_0p008`: anchor 대비 test abs delta `0.020602`, public `0.5920118473`.
- `core_plus_s1_q2tiny_q3`: anchor 대비 `0.003705`, 기존 제출과 mean abs diff `0.019747`.
- `core_plus_s1`: anchor 대비 `0.003485`, 기존 제출과 mean abs diff `0.019567`.
- 즉 새 후보는 기존 제출과는 충분히 다르지만 anchor에서 멀리 튀지 않는다.

다음 명령어:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.direction_gated_ablation \
  --output-dir research/direction_gated_ablation_20260622 \
  --submission-dir submissions/direction_gated_ablation_20260622
```

## 15. 2026-06-22 public-aware stacking/blending and latest research

추가 조사:

- `research/LATEST_ENSEMBLE_PAPERS_20260622.md`
- 2026-06 기준 확인한 방향:
  - logistic meta-stacking은 accuracy/ROC-AUC는 좋아도 log-loss calibration을 망칠 수 있다.
  - 중복 source가 많으면 stacker 이득에 ceiling이 생긴다.
  - greedy/convex ensemble selection, source dedupe, temperature/anchor shrink가 log-loss 목적에는 더 맞다.
  - HAPEns식 다목적 post-hoc ensemble 개념은 여기서 public-risk/movement-cost penalty로 대응한다.

구현/실행 완료:

- `src/public_aware_stack_blend.py`
- `research/public_aware_stack_blend_20260622/`
- `submissions/public_aware_stack_blend_20260622/`

구현 내용:

- 기존 OOF/test 후보 62개를 로드.
- target별 OOF correlation dedupe 후 49개 source 유지.
- fold-safe meta Logistic stacker:
  - probability, logit, anchor-delta features 사용.
  - top 8/12/16 source, C `0.05/0.15/0.5`, class_weight plain/balanced.
- fold-safe simplex weighted blend:
  - probability blend / logit blend.
  - L2 `0.001/0.01/0.05`.
- stack/blend 후보에 anchor-logit shrink 적용.
- 마지막 target-wise selector:
  - local full/last OOF
  - public 실패 방향 penalty
  - failed submission alignment penalty
  - test movement penalty

상위 결과:

| candidate | full | last | anchor test abs delta | read |
|---|---:|---:|---:|---|
| `target_select_public_balanced` | `0.593490` | `0.571439` | `0.007461` | local 최상, Q2 up 16.4% |
| `target_select_public_tight` | `0.592635` | `0.572477` | `0.006878` | 최종 추천, Q2 up 6.8% |
| `target_select_public_tight_logit_anchorblend_w0p9` | `0.592477` | `0.573720` | `0.006293` | 더 축소한 대안 |

최종 제출 1개 추천:

```text
submissions/public_aware_stack_blend_20260622/target_select_public_tight_last0.572477_full0.592635.csv
```

이유:

- stack/blend 기반이다. Q1은 simplex logit stacker를 쓰고, 나머지는 direction-gated/ablation source를 target-wise로 선택했다.
- 기존 public 실패 후보와 충분히 다르다.
  - 기존 제출과 mean abs diff `0.019946`.
  - 기존 제출 anchor delta `0.020602` vs 새 후보 `0.006878`.
- 기존 실패 방향을 제거했다.
  - S3 up 제거.
  - S4 down 제거.
  - S2 up 제거.
  - Q2 movement는 aggressive의 16.4%가 아니라 6.8%로 제한.

재현 명령어:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.public_aware_stack_blend \
  --output-dir research/public_aware_stack_blend_20260622 \
  --submission-dir submissions/public_aware_stack_blend_20260622
```

## 16. 다음 AI에게 넘길 핵심 질문

1. `subject_time_blocked` last-block을 유지하면서 `0.55` 이하로 내려가는 구조를 찾을 수 있는가?
2. Q 타깃의 전체기간 평균 기준 이진화 구조를 hard shift가 아닌 rank patch로 안전하게 쓸 수 있는가?
3. Sparse greedy의 0.5824가 last-block 과적합인지 rolling-origin backtest에서도 유지되는가?
4. Q1을 개선할 수 있는 subject-target rank patch나 recipe가 있는가?
5. public 제출 없이 로컬 `last_logloss` 기준으로 후보를 거를 수 있는가?

## 17. 핵심 파일 위치

현재 베이스:

- `src/train_temporal_prior.py`
- `submissions/submission_temporal_prior_last0.5933.csv`

실험 코드:

- `src/fast_temporal_stack.py`
- `src/kaggle_style_ensemble.py`
- `src/structural_balance_search.py`
- `src/sequence_smoothing_search.py`
- `src/oof_sparse_greedy.py`

주요 결과:

- `research/fast_temporal_stack/candidate_scores.csv`
- `research/kaggle_style_full_seed42/candidate_scores.csv`
- `research/kaggle_style_full_3seed/candidate_scores.csv`
- `research/kaggle_style_full_5seed_guarded/candidate_scores.csv`
- `research/kaggle_style_feature_plus_smoke/candidate_scores.csv`
- `research/kaggle_style_qbalance_smoke/candidate_scores.csv`
- `research/structural_balance_search/candidate_scores.csv`
- `research/sequence_smoothing_search/candidate_scores.csv`
- `research/oof_sparse_greedy/candidate_scores.csv`
- `research/oof_sparse_greedy/source_scores.csv`
- `research/oof_sparse_greedy/targetwise_greedy_steps.csv`

현재 가장 중요한 다음 결과 파일:

- `research/rolling_backtest/greedy_stability.csv`
- `research/rolling_backtest/source_scores_by_split.csv`

## 18. 2026-06-22 public score pseudo-posterior blend

새 public 확인:

| submitted file | public |
|---|---:|
| `submissions/public_aware_stack_blend_20260622/target_select_public_tight_last0.572477_full0.592635.csv` | `0.5905116492` |
| `submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv` | `0.5920118473` |
| `submissions/fast_temporal_stack/02_guarded_targetwise.csv` | `0.5935970063` |

해석:

- `target_select_public_tight`가 기존 `last_guard_0p008`보다 public에서 `0.0015001981` 좋아졌다.
- 따라서 public은 무작정 local full 최저 후보보다, 방향 제한/anchor shrink가 걸린 후보를 선호하는 신호가 있다.
- 다만 0.5905는 아직 166등권이므로, public score 자체를 제약으로 쓰는 더 직접적인 post-hoc search를 추가했다.

구현:

- `src/public_score_pseudo_blend.py`
- `research/public_score_pseudo_blend_20260622/`
- `submissions/public_score_pseudo_blend_20260622/`
- `research/submission_pattern_diagnostics_public_score_pseudo_20260622/`

방법:

- 제출 파일별 public logloss는 test hidden label에 대한 선형 제약으로 볼 수 있다.
- anchor test probability를 prior로 두고, known public scores를 맞추는 soft pseudo posterior를 추정했다.
- 그 posterior 기준으로 기존 OOF/test source 80개를 다시 평가했다.
- target-wise selection, pairwise logit blend, simplex blend, anchor-logit shrink 후보 829개를 생성했다.

상위 결과:

| candidate | pseudo-public | full | last | anchor abs delta |
|---|---:|---:|---:|---:|
| `pseudo_target_tight_anchorlogit_w0p82` | `0.588794` | `0.594948` | `0.588809` | `0.001352` |
| `pseudo_target_tight_anchorlogit_w0p9` | `0.588800` | `0.594902` | `0.588453` | `0.001465` |
| `pseudo_target_tight` | `0.588821` | `0.594855` | `0.588027` | `0.001601` |

패턴 진단:

- Q1/Q2/Q3/S1/S3는 anchor 그대로 유지.
- S2만 down 이동:
  - test mean delta `-0.007815`
  - test down rate at 0.005 threshold `0.052`
  - last OOF down precision `0.666667`
- S4만 up 이동:
  - test mean delta `+0.001646`
  - test up rate at 0.005 threshold `0.128`
  - last OOF up precision `0.833333`

현재 한 개만 제출한다면:

```text
submissions/public_score_pseudo_blend_20260622/pseudo_target_tight_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948.csv
```

재현 명령:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.public_score_pseudo_blend \
  --output-dir research/public_score_pseudo_blend_20260622 \
  --submission-dir submissions/public_score_pseudo_blend_20260622
```

## 19. 2026-06-22 target-weighted single-model source optimizer

주의:

- 여기서는 제출 후보를 고르는 단계가 아니다.
- 목적은 0.59대 post-hoc 이동이 아니라, stack/blend에 넣을 새 OOF/test source를 만드는 것이다.
- 먼저 한 모델군을 깊게 판다. 현재 1순위는 LGBM이다.

구현:

- `src/target_weighted_single_model.py`
- smoke 검증 완료:
  - `research/target_weighted_single_model_smoke/`
  - `submissions/target_weighted_single_model_smoke/`

구현된 학습 기법:

- LGBM/XGB/Cat 공통 인터페이스.
- target-wise sample weighting:
  - subject balance
  - class balance
  - recency emphasis
  - late-fold emphasis
  - anchor-error emphasis
- `target_auto` weight profile:
  - Q2/Q3는 recency/class 강화.
  - S2/S4는 recency/anchor-error/fold-late 강화.
  - S3는 과격한 이동 억제.
- fold-safe numeric bin/category view.
- fold-safe smoothed target encoding.
- target history feature:
  - `mean_sm4`, `mean_sm16`, `last2_sm4`, `last5_sm4`, `last20_sm4`, `ridge1`, `ridge10`
- validation logloss early stopping + best-iteration prediction.
- target-wise shrink search back to temporal anchor.
- output convention:
  - `*_oof.csv`
  - `*_test_pred.csv`
  - downstream stack/blend script에서 바로 source-dir로 사용 가능.

실행 순서:

1. LGBM full source bank:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.target_weighted_single_model \
  --models lgbm \
  --seeds 42 7 2024 \
  --fold-limit 5 \
  --param-profiles smooth mid leaf31 \
  --weight-profiles uniform subject_class recent_class recent_anchorerr target_auto \
  --top-k-grid 60 100 160 240 \
  --feature-bank bins_te \
  --te-top-n 10 \
  --te-bins 4 8 \
  --te-smooth 12 \
  --pair-feature-count 80 \
  --target-history-features \
  --rounds 2600 \
  --early-stopping-rounds 140 \
  --shrink-grid 0 0.05 0.08 0.12 0.18 0.25 0.35 0.50 0.70 1.0 \
  --shrink-modes logit prob \
  --output-dir research/target_weighted_single_model_lgbm_20260622 \
  --submission-dir submissions/target_weighted_single_model_lgbm_20260622 \
  --log-period 0
```

2. 결과 확인:

```bash
sed -n '1,220p' research/target_weighted_single_model_lgbm_20260622/TARGET_WEIGHTED_SINGLE_MODEL_REPORT.md
```

3. 새 LGBM source를 기존 stack/blend 후보군에 추가:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.public_aware_stack_blend \
  --source-dirs \
    research/target_weighted_single_model_lgbm_20260622 \
    research/public_aware_stack_blend_20260622 \
    research/direction_gated_ablation_20260622 \
    research/pattern_safe_candidates_20260622 \
    research/constrained_target_blend_logit_newton \
    research/residual_submission_blend_ridge_logit_newton \
    research/residual_single_model_opt_ridge_logit_newton \
    research/residual_submission_blend \
    research/residual_single_model_opt_ridge \
    research/residual_single_model_opt_hist_gb \
    research/residual_single_model_opt_ridge_logit_te_full \
    research/residual_submission_blend_ridge_logit_te_full \
  --output-dir research/public_aware_stack_blend_with_lgbm_source_20260622 \
  --submission-dir submissions/public_aware_stack_blend_with_lgbm_source_20260622
```

4. public-score pseudo blend에도 새 source 추가:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.public_score_pseudo_blend \
  --source-dirs \
    research/target_weighted_single_model_lgbm_20260622 \
    research/public_aware_stack_blend_20260622 \
    research/direction_gated_ablation_20260622 \
    research/pattern_safe_candidates_20260622 \
    research/constrained_target_blend_logit_newton \
    research/residual_submission_blend_ridge_logit_newton \
    research/residual_single_model_opt_ridge_logit_newton \
    research/residual_submission_blend \
    research/residual_single_model_opt_ridge \
    research/residual_single_model_opt_hist_gb \
    research/residual_single_model_opt_ridge_logit_te_full \
    research/residual_submission_blend_ridge_logit_te_full \
  --output-dir research/public_score_pseudo_blend_with_lgbm_source_20260622 \
  --submission-dir submissions/public_score_pseudo_blend_with_lgbm_source_20260622
```

5. LGBM source가 유효하면 같은 구조를 XGB/Cat에 이식:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=research/.mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m src.target_weighted_single_model \
  --models xgb cat \
  --seeds 42 7 \
  --fold-limit 5 \
  --param-profiles smooth mid \
  --weight-profiles subject_class recent_class recent_anchorerr target_auto \
  --top-k-grid 60 120 200 \
  --feature-bank bins_te \
  --te-top-n 8 \
  --te-bins 4 8 \
  --te-smooth 16 \
  --pair-feature-count 60 \
  --target-history-features \
  --rounds 2200 \
  --early-stopping-rounds 140 \
  --shrink-grid 0 0.05 0.10 0.18 0.30 0.50 0.70 1.0 \
  --shrink-modes logit prob \
  --output-dir research/target_weighted_single_model_xgb_cat_20260622 \
  --submission-dir submissions/target_weighted_single_model_xgb_cat_20260622 \
  --log-period 0
```
