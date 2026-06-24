# Target Weighted LGBM Run Analysis - 2026-06-23

## Completed Outputs Checked

- `research/target_weighted_single_model_lgbm_20260622`
- `submissions/target_weighted_single_model_lgbm_20260622`
- `research/public_aware_stack_blend_with_lgbm_source_20260622`
- `submissions/public_aware_stack_blend_with_lgbm_source_20260622`

Missing output:

- `research/public_score_pseudo_blend_with_lgbm_source_20260622`
- `submissions/public_score_pseudo_blend_with_lgbm_source_20260622`

That means the public-score pseudo blend with the new LGBM source was not completed, or did not write outputs.

## Single Model Result

Anchor baseline:

- full logloss: `0.595830`
- last logloss: `0.593282`

Target-weighted LGBM candidates:

| candidate | full | last | full delta | last delta | fold std | tail3 worst |
|---|---:|---:|---:|---:|---:|---:|
| `target_weighted_single_composite` | 0.592353 | 0.586688 | -0.003477 | -0.006594 | 0.006320 | 0.604356 |
| `target_weighted_single_last` | 0.593006 | 0.586345 | -0.002823 | -0.006937 | 0.007348 | 0.607136 |
| `target_weighted_single_full` | 0.591406 | 0.587857 | -0.004424 | -0.005425 | 0.005970 | 0.602325 |

Read:

- The new LGBM source is real CV signal, not just a submission blend.
- `target_weighted_single_full` is the most stable local source.
- `target_weighted_single_last` is most last-fold aggressive but has worse fold dispersion.
- `target_weighted_single_composite` is the best compromise.

## Target-Level Signal

Strong useful signal:

- `Q2`: `0.692880/0.638084` anchor to `0.669502/0.604338`.
  - full delta `-0.023377`
  - last delta `-0.033747`
  - source config: LGBM smooth, `recent_anchorerr`, top 240, logit shrink 0.70.

Moderate useful signal with risk:

- `S2`: composite gives last gain but weak full stability.
  - composite: `0.560148/0.611811`, full delta `+0.000027`, last delta `-0.011005`
  - full candidate: `0.557499/0.615890`, full delta `-0.002621`, last delta `-0.006926`
  - This should be used through smaller shrink or movement cap.

Weak or conservative-only:

- `Q3`: full improves, last worsens. Use only tiny shrink if used.
- `Q1`: tiny last gain, full risk. Not a core source.
- `S1`: tiny last gain, raw model overfits. Existing public-aware source is much better.
- `S3`: tiny gain. Existing public-aware source is better.
- `S4`: no useful LGBM signal. Best shrink is zero, anchor/public-aware only.

## Early Stopping Check

`fit_diagnostics.csv` has 2100 rows and every row records:

- `stop_policy = validation_logloss_early_stopping`
- `fallback_logic = best_iteration or current_iteration`

Mean best iteration by target:

| target | count | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| Q1 | 300 | 19.10 | 7.67 | 1.00 | 89.33 |
| Q2 | 300 | 21.18 | 3.00 | 1.00 | 70.00 |
| Q3 | 300 | 28.65 | 24.33 | 1.00 | 81.00 |
| S1 | 300 | 76.34 | 71.33 | 1.00 | 209.33 |
| S2 | 300 | 63.92 | 45.33 | 1.00 | 313.00 |
| S3 | 300 | 103.19 | 87.33 | 1.33 | 321.33 |
| S4 | 300 | 4.99 | 3.67 | 1.00 | 23.67 |

Read:

- Fallback is implemented.
- S4 stops almost immediately, confirming no useful single-model LGBM signal.
- Q2 has useful signal but stops early in many folds, so it should be guarded against test-distribution overmovement.

## Public-Aware Stack/Blend Result

Top candidates:

| candidate | full | last | test abs delta | public risk | selector |
|---|---:|---:|---:|---:|---:|
| `target_select_public_balanced` | 0.593490 | 0.571440 | 0.007379 | 0.000713 | 0.576610 |
| `target_select_public_aggressive` | 0.593490 | 0.571440 | 0.007379 | 0.000713 | 0.576610 |
| `target_select_public_tight` | 0.592636 | 0.572477 | 0.006796 | 0.000509 | 0.577092 |

Important: these are effectively the previous public-aware candidates again. The new LGBM source did not materially change the top saved target selections.

Selected sources for `target_select_public_tight`:

| target | selected source |
|---|---|
| Q1 | `public_aware_stack_blend_20260622/target_select_public_aggressive` |
| Q2 | `public_aware_stack_blend_20260622/target_select_public_tight` |
| Q3 | `public_aware_stack_blend_20260622/target_select_public_aggressive` |
| S1 | `public_aware_stack_blend_20260622/target_select_public_aggressive_logit_anchorblend_w0p65` |
| S2 | `public_aware_stack_blend_20260622/target_select_public_aggressive` |
| S3 | `public_aware_stack_blend_20260622/target_select_public_aggressive` |
| S4 | `public_aware_stack_blend_20260622/target_select_public_aggressive` |

Read:

- The stack/blend run included the LGBM source in the candidate pool.
- The final public-aware target selector mostly rejected it.
- The main reason is public-risk/test movement, especially for Q2 and S2.

## Why The LGBM Source Was Rejected

Q2 LGBM source:

- local: `0.669502/0.604338`, excellent versus anchor
- test abs delta mean: `0.063123`
- public risk: `0.029021`
- rank score: `0.781618`
- kept for target: false

This is too much raw movement for the current public-aware penalty, even though local CV is very good.

S2 LGBM source:

- `target_weighted_single_full`: `0.557499/0.615890`, public risk `0.006134`, rank `0.653437`
- `target_weighted_single_composite`: `0.560148/0.611811`, public risk `0.011026`, rank `0.679619`

Existing public-aware S2 candidates are much stronger on last fold:

- `public_aware_stack_blend_20260622/target_select_public_aggressive`: `0.562622/0.568421`
- `direction_gated_ablation_20260622/core_q1down_s2tight_s4mid`: `0.555477/0.581033`

So S2 should not take the raw LGBM source directly. It may still be useful as a capped/shrunk auxiliary source.

## Next Implementation Direction

Do not run generic stack/blend again as-is. It already rejects the useful new LGBM source.

Next code should implement guarded extraction of the new LGBM signal:

1. Q2-specific gated integration:
   - source: target-weighted LGBM Q2 signal
   - apply logit shrink grid and explicit test movement cap
   - compare against public-aware Q2 baseline
   - keep only candidates with lower full/last and acceptable movement

2. S2-specific cautious integration:
   - source: `target_weighted_single_full` first, composite only as more aggressive option
   - smaller shrink than Q2
   - movement cap and public-direction guard

3. Q3 tiny-shrink only:
   - use only if full improves without last regression beyond a small guard

4. S4 remains public-aware/anchor:
   - LGBM best shrink was zero.

5. After guarded integration, rerun public-score pseudo blend if needed:
   - current evidence shows that output is missing.

## Submission Status

No new submission should be claimed from this run alone.

Reason:

- The single-model source improved CV.
- But the top public-aware submissions are essentially unchanged from the earlier public-aware run.
- The missing next step is controlled Q2/S2 signal extraction, not another direct generic blend.
