# Q2 Public Micro Sweep Analysis - 2026-06-23

## Context

Latest actual public feedback:

- Submitted: `submissions/guarded_lgbm_integration_20260623_v2/public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5_last0.570607_full0.592648.csv`
- Public score: `0.590284128`
- Previous known public: `target_select_public_tight_last0.572477_full0.592635.csv` scored `0.5905116492`

The improvement came from a guarded Q2-only modification. Broader pseudo-public blending selected S2/S4 movement, but that contradicted the latest actual feedback and had worse local full CV, so it is not reliable enough for a remaining submission.

## Competitor Notebook Usage

Attached notebook `[Public_0.5917] LGBM+XGB anchor/Subject-hole CV/...` uses:

- subject-hole CV
- anchor probability/logit features
- LGBM/XGB targetwise OOF sources
- targetwise LGBM/XGB blending
- conservative feature selection / early stopping

These ideas are already represented in this repo through `src/kaggle_style_ensemble.py` and `src/target_weighted_single_model.py`. Including the generated Kaggle-style sources into the guarded LGBM integration did not beat the current Q2-only public-aware source. The useful part is not the whole notebook pipeline, but the targetwise Q2 source signal already extracted from `target_weighted_single_model_lgbm_20260622`.

## New Implementation

Added:

- `src/q2_public_micro_sweep.py`

What it does:

- Starts from the latest actual best public submission as the reference.
- Keeps Q1/Q3/S1/S2/S3/S4 fixed.
- Sweeps only Q2 by logit/prob interpolation or weak extrapolation against nearby OOF/test sources.
- Fits a low-dimensional public proxy using known public scores:
  - `last_guard_0p008`: `0.5920118473`
  - `target_select_public_tight`: `0.5905116492`
  - latest Q2-only guarded: `0.590284128`
  - `02_guarded_targetwise`: `0.5935970063`
- Records CV full, CV last, Q2-specific CV, fold stability, distance from current public best, and proxy estimate.

## Result

Best micro-sweep candidate:

`submissions/q2_public_micro_sweep_20260623/q2micro__public_aware_stack_blend_20260622_target_select_public_tight_logit_anchorblend_w0p65__prob__wm0p2_proxy0.590184_last0.570231_full0.592721.csv`

Compared with the latest submitted public-best file:

- Only Q2 changes.
- Q2 mean delta vs current file: `+0.000765650`
- Q2 mean absolute delta vs current file: `0.002315581`
- Q2 max absolute delta vs current file: `0.013457754`
- Q2 up-rate vs current file: `0.540`
- Other targets: exactly unchanged.

Local metrics:

- Reference latest public-best local: full `0.592648`, last `0.570607`
- New micro-sweep local: full `0.592721`, last `0.570231`
- Full CV worsens by `+0.000073`
- Last CV improves by `-0.000376`
- Public proxy improves from `0.590288` to `0.590184`

Public proxy fit residuals:

- `last_guard_0p008`: `+0.000042`
- `target_select_public_tight`: `+0.000261`
- latest Q2-only guarded: `+0.000004`
- `02_guarded_targetwise`: `-0.000307`

The proxy is not a guarantee because it is fit from only four public observations, but the candidate is constrained enough that the experiment isolates Q2 direction rather than mixing many target changes.

## Interpretation

Blending is not globally broken. The issue is that broad target blending and pseudo-public search overfit noisy public feedback. The latest actual public result indicates Q2 movement is the only currently validated lever. S2/S4 pseudo candidates are not trusted because they improve proxy but degrade local full and contradict the latest Q2-only public improvement pattern.

## Submission Decision

If using one remaining submission to probe public structure, use the top Q2 micro-sweep candidate above. Expected improvement is small, around `0.0001` by proxy. It is not a safe private-CV improvement candidate because full CV worsens slightly.

If preserving submissions is more important, wait and instead run a new true model training experiment focused on Q2/S2 feature learning rather than post-processing.

