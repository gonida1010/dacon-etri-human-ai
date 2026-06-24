# Public-score pseudo blend report

Generated: 2026-06-23T20:32:57

## Known public constraints

- `last_guard_0p008_last0.588383_full0.587743` public=0.5920118473, pseudo_expected=0.5907949395, residual=-0.00121691
- `target_select_public_tight_last0.572477_full0.592635` public=0.5905116492, pseudo_expected=0.5917652960, residual=+0.00125365
- `public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5_last0.570607_full0.592648` public=0.5902841280, pseudo_expected=0.5918348851, residual=+0.00155076
- `02_guarded_targetwise` public=0.5935970063, pseudo_expected=0.5917350163, residual=-0.00186199

## Selected candidate

- candidate: `pseudo_target_tight_anchorlogit_w0p9`
- pseudo_public_logloss: `0.588561410`
- local full/last: `0.594902` / `0.588453`
- delta vs anchor full/last: `-0.000928` / `-0.004829`

## Top candidates

| candidate | pseudo_public_logloss | full_logloss | last_logloss | selector_score |
| --- | --- | --- | --- | --- |
| pseudo_target_tight_anchorlogit_w0p9 | 0.588561 | 0.594902 | 0.588453 | 0.589298 |
| pseudo_target_balanced_anchorlogit_w0p9 | 0.588561 | 0.594902 | 0.588453 | 0.589298 |
| pseudo_target_public_heavy_anchorlogit_w0p9 | 0.588561 | 0.594902 | 0.588453 | 0.589298 |
| pseudo_target_public_max_anchorlogit_w0p9 | 0.588561 | 0.594902 | 0.588453 | 0.589298 |
| pseudo_target_tight_anchorlogit_w0p96 | 0.588559 | 0.594873 | 0.588195 | 0.589299 |
| pseudo_target_balanced_anchorlogit_w0p96 | 0.588559 | 0.594873 | 0.588195 | 0.589299 |
| pseudo_target_public_heavy_anchorlogit_w0p96 | 0.588559 | 0.594873 | 0.588195 | 0.589299 |
| pseudo_target_public_max_anchorlogit_w0p96 | 0.588559 | 0.594873 | 0.588195 | 0.589299 |
| pseudo_target_tight | 0.588561 | 0.594855 | 0.588027 | 0.589302 |
| pseudo_target_balanced | 0.588561 | 0.594855 | 0.588027 | 0.589302 |
| pseudo_target_public_heavy | 0.588561 | 0.594855 | 0.588027 | 0.589302 |
| pseudo_target_public_max | 0.588561 | 0.594855 | 0.588027 | 0.589302 |
| pseudo_target_tight_anchorlogit_w0p82 | 0.588573 | 0.594948 | 0.588809 | 0.589306 |
| pseudo_target_balanced_anchorlogit_w0p82 | 0.588573 | 0.594948 | 0.588809 | 0.589306 |
| pseudo_target_public_heavy_anchorlogit_w0p82 | 0.588573 | 0.594948 | 0.588809 | 0.589306 |
| pseudo_target_public_max_anchorlogit_w0p82 | 0.588573 | 0.594948 | 0.588809 | 0.589306 |
| pseudo_target_tight_anchorlogit_w0p7 | 0.588612 | 0.595029 | 0.589367 | 0.589338 |
| pseudo_target_balanced_anchorlogit_w0p7 | 0.588612 | 0.595029 | 0.589367 | 0.589338 |
| pseudo_target_public_heavy_anchorlogit_w0p7 | 0.588612 | 0.595029 | 0.589367 | 0.589338 |
| pseudo_target_public_max_anchorlogit_w0p7 | 0.588612 | 0.595029 | 0.589367 | 0.589338 |

## Target choices

| target | source | full_logloss | last_logloss | pseudo_public_logloss | selector_score | test_abs_delta_mean | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | anchor | 0.590425 | 0.618506 | 0.597655 | 0.597655 | 0.000000 | pseudo_target_public_max |
| Q2 | anchor | 0.692880 | 0.638084 | 0.650629 | 0.650629 | 0.000000 | pseudo_target_public_max |
| Q3 | anchor | 0.662681 | 0.634692 | 0.638558 | 0.638558 | 0.000000 | pseudo_target_public_max |
| S1 | anchor | 0.492352 | 0.449544 | 0.480808 | 0.480808 | 0.000000 | pseudo_target_public_max |
| S2 | direction_gated_ablation_20260622/core_q1down_s2tight_s4tight | 0.555449 | 0.588833 | 0.557641 | 0.557696 | 0.009201 | pseudo_target_public_max |
| S3 | anchor | 0.541551 | 0.600679 | 0.555675 | 0.555675 | 0.000000 | pseudo_target_public_max |
| S4 | pattern_safe_candidates_20260622/q1_s2down25_s4up25 | 0.628649 | 0.585849 | 0.638958 | 0.638970 | 0.002004 | pseudo_target_public_max |

## Public feedback interpretation

- The newer `target_select_public_tight` score is lower than the prior `last_guard_0p008`, so public feedback rewards the restricted movement pattern.
- The search therefore optimizes a pseudo-public posterior first, then rejects candidates whose CV full/last degradation is too large.
