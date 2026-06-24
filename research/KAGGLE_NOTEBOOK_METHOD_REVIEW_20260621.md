# Kaggle Notebook Method Review for Dacon ETRI, 2026-06-21

Scope:

- Source notebooks:
  - `/Users/parkyeonggon/Downloads/ps6e6-one-vs-rest-tabm (1).ipynb`
  - `/Users/parkyeonggon/Downloads/ps6e6-one-vs-rest-xgb (1).ipynb`
  - `/Users/parkyeonggon/Downloads/cat-v3-for-s6e6 (1).ipynb`
  - `/Users/parkyeonggon/Downloads/realmlp-v5-for-s6e6 (2).ipynb`
- Kaggle star-classification content itself is not reused as a task. Only research methods are mapped to this Dacon ETRI binary multi-target task.

## 1. Methods Found in the Notebooks

### OOF-first research workflow

- All notebooks save OOF predictions and test predictions separately.
- Model or feature decisions are made from OOF diagnostics.
- Test predictions are fold-averaged.

Dacon status:

- Already mostly implemented.
- Current scripts use `subject_time_blocked_folds`, OOF `full_logloss`, `last_logloss`, and target-wise choice tables.

### One-vs-rest / target-wise decomposition

- XGB/TabM notebooks train class-specific binary models and assemble final probabilities.
- This is not directly needed for Dacon because Dacon targets are already binary, but the equivalent is target-wise model/source/shrink selection.

Dacon status:

- Already partly implemented through target-wise residual correction and target-wise blend search.

### Fold-safe target encoding

- RealMLP and tabular notebooks use `TargetEncoder` and interaction categories.
- This is a major method: categorical views are encoded inside CV rather than globally leaking labels.

Dacon status:

- Missing.
- Existing temporal priors are fold-safe, but there is no general fold-safe target encoding feature bank for sensor/time/category views.

### Numeric bin/category views

- CatBoost and RealMLP notebooks create categorical views from numeric columns:
  - floor/rounded numeric columns
  - quantile bins
  - compact interaction categories
  - binned continuous values used as categorical or target-encoded features

Dacon status:

- Mostly missing.
- Current Dacon features are mostly continuous daily aggregates, sleep aggregates, subject z-scores, and calendar features.
- CatBoost-style native categorical/bin views are not yet represented.

### Source diversity via OOF correlation

- TabM/XGB notebooks inspect OOF correlations per class/source before combining predictions.

Dacon status:

- Partly missing.
- We have source score tables, but no formal source-correlation selector for target-wise blending.

### Bias/intercept tuning

- Star notebooks tune class bias in log-probability space for balanced accuracy.

Dacon status:

- Direct argmax bias tuning is not applicable because Dacon metric is binary logloss.
- Applicable adaptation: target-wise logit intercept calibration or shrink in logit space, OOF-only.

### TabM / RealMLP

- TabM notebook uses `TabM_D_Classifier`, PWL numerical embeddings, and internal model ensembling.
- RealMLP notebook uses `n_ens=8`, categorical embedding/one-hot, robust numeric preprocessing, dropout schedule, and class weights.

Dacon status:

- Missing.
- Not first priority because train size is only 450 rows. If used, they should predict anchor residual/logit residual, not raw labels.

## 2. What Has Already Been Implemented in Dacon

- `src/residual_single_model_opt.py`
  - anchor residual learning
  - target-wise shrink
  - shrink 0 fallback to anchor
  - subject/class/recency/residual sample weights
  - `logit_newton` residual mode
  - overflow-safe sigmoid and delta clipping
- `src/residual_submission_blend.py`
  - OOF-based target-wise blend over anchor, Ridge residual, and KNN guarded source
- Existing booster scripts:
  - LGBM/XGB/CatBoost use validation early stopping and best-iteration prediction where applicable.

## 3. Missing Methods to Apply Next

Priority 1. Fold-safe target encoding feature bank

- Add target-wise, fold-safe target encoding over selected sensor/time/bin columns.
- Validation rows must be transformed using encoders fit only on training folds.
- Training rows should use leave-one-out smoothed encoding to avoid memorization.
- Encoding target should follow residual mode:
  - `prob`: encode probability residual
  - `logit_newton`: encode Newton residual target

Priority 2. Numeric bin/category views

- Add quantile-bin views for selected continuous features.
- Use bins as:
  - numeric category codes
  - fold-safe target encoding keys
- Candidate source columns should be selected fold-wise from residual correlation, not globally from validation labels.

Priority 3. OOF source correlation selector

- Build target-wise OOF correlation tables across anchor, temporal, KNN, residual, TE residual, and future CatBoost/NN sources.
- Penalize highly correlated sources in blend search.
- This is safer than greedy last-block source picking.

Priority 4. CatBoost categorical-view residual source

- Train CatBoost on continuous features plus bin/category views.
- Use logloss early stopping and OOF.
- Treat output as residual/logit residual source first; avoid raw-label submission until it proves stable.

Priority 5. TabM / RealMLP residual source

- Only after TE/bin/CatBoost feature bank exists.
- Use small configs and residual targets, not raw labels.

## 4. Implemented Immediately

Implemented in `src/residual_single_model_opt.py`:

- `--feature-bank none|bins|te|bins_te`
- `--te-top-n`
- `--te-bins`
- `--te-smooth`
- fold-wise quantile binning from training rows only
- leave-one-out smoothed TE for training rows
- train-fold-fitted TE for validation/test rows
- diagnostics columns for generated bin/TE feature count

Implemented source-correlation diagnostics:

- `src/oof_source_correlation.py`
- Reads OOF prediction files and/or `oof_bank.csv`
- Writes target-wise correlation and source diversity diagnostics.

Implemented in `src/residual_submission_blend.py`:

- target-wise blend candidates now save `*_oof.csv`
- target-wise blend candidates now save `*_test_pred.csv`
- these files can be passed into `src.oof_source_correlation`

Implemented in `src/target_weighted_single_model.py`:

- new single-model OOF/test source generator before stacking/blending
- LGBM/XGB/Cat support with shared target-wise weighting interface
- target-wise sample weights:
  - subject balance
  - class balance
  - recency emphasis
  - late-fold emphasis
  - anchor-error emphasis
- fold-safe numeric bin views and smoothed target encoding
- optional target-history features from temporal priors
- validation-logloss early stopping with best-iteration prediction
- target-wise shrink search back to temporal anchor
- OOF/test artifacts saved in the `*_oof.csv` and `*_test_pred.csv` convention

## 5. Notes for Future Runs

- Do not use Kaggle notebook random `StratifiedKFold` for Dacon final scoring. Keep `subject_time_blocked_folds`.
- Do not use balanced-accuracy bias tuning directly. Dacon optimizes binary logloss.
- Do not let target encoding use validation labels.
- Do not submit raw deep tabular/large CatBoost outputs before OOF full/last stability is proven.
- First run the optimized LGBM source bank, then feed its OOF/test files into the existing public-aware stack/blend scripts.
