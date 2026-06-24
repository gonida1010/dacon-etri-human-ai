# Dacon ETRI Human-AI Current Root Cause Handoff - 2026-06-23

This document is for another AI/engineer to continue the work without relying on chat history.

## Current Public Scores

Known actual submissions:

- `submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv`
  - public: `0.5920118473`
- `submissions/public_aware_stack_blend_20260622/target_select_public_tight_last0.572477_full0.592635.csv`
  - public: `0.5905116492`
- `submissions/guarded_lgbm_integration_20260623_v2/public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5_last0.570607_full0.592648.csv`
  - public: `0.590284128`
- `submissions/q2_big_probe_safe_upload.csv`
  - public: `0.5902817192`

Interpretation:

- Q2-only movement direction was slightly rewarded.
- The improvement is now effectively saturated: `0.590284128 -> 0.5902817192` is only `0.0000024088`.
- Further leaderboard-only micro movement is unlikely to close the gap to `0.54`.

## Original Data Inventory

Project root:

`/Users/parkyeonggon/Projects/dacon/dacon-etri-human-ai`

Raw files:

- `data/ch2026_metrics_train.csv`
  - shape: `450 x 10`
  - columns: `subject_id, sleep_date, lifelog_date, Q1, Q2, Q3, S1, S2, S3, S4`
  - subjects: `10` (`id01` to `id10`)
  - sleep_date: `2024-06-04` to `2024-11-15`
- `data/ch2026_submission_sample.csv`
  - shape: `250 x 10`
  - same subjects
  - sleep_date: `2024-07-07` to `2024-11-20`
- Sensor parquet directory:
  - `data/ch2025_data_items/`
  - all sensor files cover the same 10 subjects.
  - sensor date range is roughly `2024-06-03` to `2024-11-19`.

Sensor files:

- `ch2025_mACStatus.parquet`: `939,896` rows, columns `subject_id,timestamp,m_charging`
- `ch2025_mActivity.parquet`: `961,062` rows, columns `subject_id,timestamp,m_activity`
- `ch2025_mAmbience.parquet`: `476,577` rows, columns `subject_id,timestamp,m_ambience`
- `ch2025_mBle.parquet`: `21,830` rows, columns `subject_id,timestamp,m_ble`
- `ch2025_mGps.parquet`: `800,611` rows, columns `subject_id,timestamp,m_gps`
- `ch2025_mLight.parquet`: `96,258` rows, columns `subject_id,timestamp,m_light`
- `ch2025_mScreenStatus.parquet`: `939,653` rows, columns `subject_id,timestamp,m_screen_use`
- `ch2025_mUsageStats.parquet`: `45,197` rows, columns `subject_id,timestamp,m_usage_stats`
- `ch2025_mWifi.parquet`: `76,336` rows, columns `subject_id,timestamp,m_wifi`
- `ch2025_wHr.parquet`: `382,918` rows, columns `subject_id,timestamp,heart_rate`
- `ch2025_wLight.parquet`: `633,741` rows, columns `subject_id,timestamp,w_light`
- `ch2025_wPedo.parquet`: `748,100` rows, pedometer columns

## Current Feature Pipeline

Main files:

- `src/build_dataset.py`
- `src/sensor_features.py`
- `src/sleep_features.py`
- `src/nested_features.py`
- `src/cv.py`

Current `build_dataset(use_cache=True)` result:

- `X_train`: `450 x 2730`
- `X_test`: `250 x 2730`
- mean non-null coverage over train features: `0.769`
- mean non-null coverage over test features: `0.780`

Feature construction:

- Daily aggregates from sensor rows by `(subject_id, date)` and windows:
  - `full`: 0-24
  - `day`: 9-18
  - `eve`: 18-24
  - `night`: 0-6
  - `morn`: 6-9
- `L_` prefix joins `lifelog_date`.
- `S_` prefix joins `sleep_date`.
- Sleep features estimate night blocks using screen, activity, pedometer, HR, light.
- Subject-level z-score is transductive: train+test sensor statistics are used without labels.

Important check:

- Cached files are:
  - `cache/daily_features.parquet`
  - `cache/nested_features.parquet`
  - `cache/sleep_features.parquet`
- `daily_features.parquet` is intentionally pre-temporal; `build_dataset()` adds lag/rolling on top.
- Cache is not the immediate error, but regenerated features should still be compared if changing feature code.

## Current CV

Main CV:

- `src/cv.py::subject_time_blocked_folds`
- Each subject's labeled days are sorted and split into 5 contiguous blocks.
- Folds contain all subjects, but calendar date ranges overlap across subjects.

Fold target means are unstable, especially Q2:

- fold0 Q2 mean: `0.463`
- fold1 Q2 mean: `0.538`
- fold2 Q2 mean: `0.670`
- fold3 Q2 mean: `0.473`
- fold4 Q2 mean: `0.682`

This means optimizing "last fold" is not a clean public/private proxy. It is just one temporal stress test.

## Current Failure Pattern

Generated diagnostics:

- `research/current_failure_analysis_20260623/target_fold_scores.csv`
- `research/current_failure_analysis_20260623/fp_fn_summary.csv`
- `research/current_failure_analysis_20260623/top_oof_errors_submitted_public_best.csv`

Key OOF pattern for latest public-best:

| target | fp_rate | fn_rate | pred_pos_rate | true_pos_rate |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 0.1356 | 0.1689 | 0.4622 | 0.4956 |
| Q2 | 0.3378 | 0.1156 | 0.7844 | 0.5622 |
| Q3 | 0.3511 | 0.0511 | 0.9000 | 0.6000 |
| S1 | 0.2000 | 0.0444 | 0.8378 | 0.6822 |
| S2 | 0.1956 | 0.0867 | 0.7600 | 0.6511 |
| S3 | 0.2133 | 0.0311 | 0.8444 | 0.6622 |
| S4 | 0.2422 | 0.1156 | 0.6867 | 0.5600 |

Interpretation:

- Most targets overpredict positive class.
- Q2/Q3 are especially bad:
  - Q2 true positive rate is `0.562`, but submitted OOF predicts positive at `0.784`.
  - Q3 true positive rate is `0.600`, but submitted OOF predicts positive at `0.900`.
- The existing anchor/model pipeline is too close to subject prior and not good enough at finding negative days.

Top OOF error concentration:

- Q2: many errors from `id08`, `id06`
- S4: many errors from `id02`, `id06`
- S3: many errors from `id07`, `id02`, `id01`
- Q3: many errors from `id04`, `id05`, `id02`

This suggests subject-specific calibration alone is not enough. Need subject-specific *state detection* from raw time-series patterns.

## Why Current Blending Is Stuck

Already tried many source/blend families:

- temporal anchor
- sparse greedy OOF bank
- public-aware stack/blend
- residual ridge/logit/HistGB variants
- target-weighted single LGBM/XGB/Cat style variants
- guarded target-specific blending
- public pseudo-score blending
- Q2-only public micro/big probes
- Kaggle-style subject-hole CV + anchor feature + LGBM/XGB/Cat source integration

Observed:

- Broad blending changes many targets but does not improve public reliably.
- Q2-only movement improves public, but only by tiny amounts.
- Current sources are highly correlated because most derive from the same coarse daily features and subject prior.
- Stacking correlated predictions cannot create the missing raw signal.

Conclusion:

The bottleneck is not "more blending." The bottleneck is weak single-model/raw-feature signal. Need improve feature extraction and target-specific state detection before stacking/blending can help.

## Likely Root Causes

1. Coarse daily aggregation loses the key signal.
   - Raw data is high-frequency, but current features are mostly mean/std/min/max/sum/count by broad windows.
   - Sleep targets likely need event-level features: screen-off/on blocks, HR stability, step bursts, charging intervals, wearable non-wear, light changes.

2. The sleep block heuristic is too simple.
   - `sleep_features.py` uses longest screen-off / active-event gaps and simple HR threshold.
   - This may be insufficient for S1-S4.
   - Need validate against OOF errors: inspect actual night timelines for top wrong rows.

3. Positive overprediction is severe.
   - Q2/Q3/S1/S2/S3/S4 are often predicted positive too frequently.
   - Need target-specific negative-day detectors.

4. CV-public mismatch is still unresolved.
   - Public score can reward tiny Q2 shifts while OOF full/last do not map cleanly.
   - Last fold alone should not drive final decisions.
   - Need several validation schemes:
     - subject time-blocked
     - month-blocked
     - public-like test-date distribution
     - subject-hole/subject-chunk CV

5. Feature dimensionality is too large for 450 rows.
   - `2730` features for `450` rows.
   - Many missing/constant-derived columns.
   - Strong feature selection and target-specific feature families are mandatory.

## High-Priority Next Research

1. Build raw night timeline diagnostics for top OOF errors.
   - For each top wrong row in `research/current_failure_analysis_20260623/top_oof_errors_submitted_public_best.csv`, plot or tabulate:
     - screen use by minute/hour
     - charging
     - steps
     - HR mean/min/std
     - light
     - activity still/move
   - Compare false positives vs true positives/true negatives for S targets.

2. Replace broad daily aggregation with compact event features.
   - Night axis: `18:00` to `12:00` next day.
   - Candidate features:
     - first/last screen event
     - longest no-screen block
     - screen-on bursts after midnight
     - charging interval duration and start/end
     - HR stable low-duration
     - HR variance inside candidate sleep block
     - step bursts inside sleep block
     - non-wear or missing HR periods
     - light exposure before sleep/wake
   - Keep features compact, not thousands of columns.

3. Target-specific modeling instead of one general feature soup.
   - Q2/Q3 likely use daytime/evening stress/activity/phone-use patterns.
   - S1-S4 likely use night block features.
   - Do not feed all features to all targets by default.

4. Recalibrate positive rates.
   - Especially Q2/Q3/S3.
   - Use OOF-calibrated per-target intercept/temperature or isotonic calibration.
   - Calibration must be fold-safe.

5. Revisit label/metric definition.
   - File: `data/ch2026_metrics_description.pdf`
   - Current environment lacked `pypdf` and `pdftotext`; someone should read this manually or install a PDF parser.
   - Need verify what Q1/Q2/Q3/S1-S4 semantically mean and whether thresholds can be directly approximated from sensors.

## Key Files For Another AI

Read these first:

- `data/ch2026_metrics_train.csv`
- `data/ch2026_submission_sample.csv`
- `data/ch2026_metrics_description.pdf`
- `data/ch2025_data_items/*.parquet`
- `src/build_dataset.py`
- `src/sensor_features.py`
- `src/sleep_features.py`
- `src/nested_features.py`
- `src/cv.py`

Current best/diagnostics:

- `research/current_failure_analysis_20260623/fp_fn_summary.csv`
- `research/current_failure_analysis_20260623/target_fold_scores.csv`
- `research/current_failure_analysis_20260623/top_oof_errors_submitted_public_best.csv`
- `research/Q2_PUBLIC_MICRO_SWEEP_ANALYSIS_20260623.md`
- `research/TARGET_WEIGHTED_LGBM_ANALYSIS_20260623.md`
- `research/KAGGLE_NOTEBOOK_METHOD_REVIEW_20260621.md`

Important result directories:

- `research/oof_sparse_greedy/`
- `research/public_aware_stack_blend_20260622/`
- `research/public_aware_stack_blend_with_lgbm_source_20260622/`
- `research/guarded_lgbm_integration_20260623_v2/`
- `research/q2_public_micro_sweep_20260623/`
- `research/target_weighted_single_model_lgbm_20260622/`
- `research/kaggle_style_full_seed42/`
- `research/kaggle_style_full_3seed/`
- `research/kaggle_style_full_5seed_guarded/`

Relevant scripts:

- `src/train_temporal_prior.py`
- `src/oof_sparse_greedy.py`
- `src/public_aware_stack_blend.py`
- `src/target_weighted_single_model.py`
- `src/guarded_lgbm_integration.py`
- `src/q2_public_micro_sweep.py`
- `src/kaggle_style_ensemble.py`

## Suggested Prompt To Another AI

You are continuing a Dacon ETRI Human-AI competition project. The current public score is stuck around `0.59028`, while top competitors are reportedly near `0.54`. Do not focus on more submission blending first. Inspect the raw sensor data and implement better compact target-specific features.

Start by reading:

- `research/AI_HANDOFF_ROOT_CAUSE_20260623.md`
- `src/build_dataset.py`
- `src/sleep_features.py`
- `src/sensor_features.py`
- `src/nested_features.py`
- `research/current_failure_analysis_20260623/top_oof_errors_submitted_public_best.csv`

Main problem:

- Current model overpredicts positives, especially Q2/Q3/S3.
- Blending is saturated because all sources come from similar coarse daily aggregates and subject prior.
- Need event-level/night-timeline features from raw parquet, especially for S1-S4 and negative-day detection.

Deliverable requested:

- Add a new compact raw-timeline feature builder.
- Re-run OOF with subject/time CV and at least one alternative CV.
- Show per-target FP/FN improvements, not only aggregate logloss.
- Only then create a new stack/blend candidate.

