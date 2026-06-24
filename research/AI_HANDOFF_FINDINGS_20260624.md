# ETRI Human-AI — 연구 요약 (다른 AI 전달용) · 2026-06-24

대회 개요는 안다고 가정. 아래는 내가 검증한 사실 + 시도한 기법 + 성능만 정리.

## 0. 확정된 평가/구조 (전략의 전제)
- 평가산식 = **Average Log-Loss** (7개 타깃 평균, 낮을수록 좋음, 확률 제출). 산식 사진으로 확정.
- **Public = test의 44%(~110행) 사전샘플 / Private = 100%(250행). 최종 순위는 private.**
- 라벨 정의(공식 PDF 해독):
  - **Q1/Q2/Q3 = 피험자 본인 기간평균 대비 위/아래** → 피험자별 prevalence≈0.5, 본질적으로 거의 무작위.
  - **S1~S4 = NSF 객관 임계값**(TST 7–9h / SE≥85% / SOL≤30m / WASO≤20m)을 **Withings 침대센서**가 측정해 이진화. 우리는 손목+폰으로 역추정만 가능(천장 존재).

## 1. 타깃별 현재 실력 (public-best OOF, full)
| Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 | 평균 |
|---|---|---|---|---|---|---|---|
|0.585|**0.693**|0.666|0.489|0.563|0.529|0.625|0.5926|
- **Q2=0.693=ln2 → 완전 무작위.** Q3도 거의 무작위. 전체 앙상블(0.5926)이 단순 subject_prior(0.5936)와 거의 동일 → 블렌딩 포화.

## 2. 가장 중요한 실증: CV↔public 단절
| 후보 | full-OOF | foldstd | public |
|---|---|---|---|
| base public-best | 0.5926 | 0.0161 | 0.5903 |
| **global_raw_full_a0.80 (강건 후보)** | **0.5835** | **0.0088** | **0.5907** |
| manual_public_tail | 0.5889 | — | 0.5922 (악화) |
| q2_big_probe | ~0.5926 | — | 0.59028 |

→ **full-OOF를 0.009 개선해도 public은 0.5903→0.5907로 무변동.** last-OOF는 더 무관. public 110행이 비대표라 어떤 CV 지표도 public을 신뢰성 있게 예측 못함. (이전 팀이 "last fold"를 public 대리로 삼아 최적화한 건 오판.)

## 3. 시도한 기법과 결과
1. **생리학적 수면추정 피처** (`src/physio_features.py`): 분 단위 actigraphy 수면/각성 판정 → TST/SE/SOL/WASO/각성수 직접 추정 + NSF 임계 피처 + **RR기반 제대로 된 HRV**(RMSSD/SDNN/pNN50, 사전수면·수면중 윈도우) + 피험자내 편차.
   - 검증 상관: S1↔TST +0.31(NSF임계로 S1 0.78 vs 0.62 분리), S3↔resting HR −0.29, hr_drop −0.18~−0.20(S1/S2/Q1/Q2).
   - **전체 모델에 추가 시 S1/S2/S4만 개선**(full −0.0037/−0.0068/−0.0055), Q1/Q3/S3 악화. 보수적 블렌드 시 전체 −0.0005(노이즈). **기존 sleep_features와 상당부분 중복**이 한계.
2. **온도/캘리브레이션**: Q2 full은 0.5로 수축 원함(−0.009)이나 last는 sharpen 원함 → 폴드 상충, 전이 불확실.
3. **전이성(Q→0.5 회귀) 가설**: 라벨이 전기간 개인평균 기준이니 test Q평균이 0.5로 회귀할 것이라 가정 → **OOF에서 실패(w=0 최적)**. 폐기.
4. **강건 블렌드 재발견**: `global_raw_full_a0.80` = base를 raw_timeline_full 쪽 0.8 logit 블렌드. full-OOF 0.5835(6/7 타깃 개선, Q2 −0.020·S2 −0.023 포함), foldstd 절반. **이전 AI가 만들고 안 낸 것**. → 제출했더니 public 0.5907(무변동). private 강건성 베팅으로는 유효.

## 4. 결론 / 다른 AI에게 묻고 싶은 것
- 내 판단: **0.54는 이 데이터로 도달 불가.** Q들은 거의 무작위, S들은 우리가 못 가진 센서가 만든 정답. 현실 천장 ~0.583(full-OOF), public은 ~0.59에서 노이즈.
- 만약 0.54 팀이 실재한다면 가능한 경로는 (a) S1~S4를 평균 0.40까지 떨어뜨리는 손목→침대센서 재구성 비법, 또는 (b) 우리가 못 본 외부/추가 신호. 둘 다 현재 데이터로는 근거 못 찾음.
- **검증 요청:** ① 0.54가 정말 같은 mean-logloss/같은 데이터 기준인가? ② S메트릭(특히 S3 SOL / S4 WASO)을 손목 actigraphy로 0.45 미만으로 떨어뜨린 사례가 있는가? ③ Q2/Q3에서 0.66 미만으로 내린 재현 가능한 신호가 있는가?

## 핵심 파일
- 피처/평가: `src/physio_features.py`, `src/physio_eval.py`, `src/physio_eval2.py`, `src/final_robust_submission.py`
- 강건 최종본: `submissions/FINAL_robust_20260624/RECOMMENDED_final_robust_full0.5835_foldstd0.0088_worst0.599.csv` (public 0.5907)
- 진단 근거: 본 문서 + `research/AI_HANDOFF_ROOT_CAUSE_20260623.md`
