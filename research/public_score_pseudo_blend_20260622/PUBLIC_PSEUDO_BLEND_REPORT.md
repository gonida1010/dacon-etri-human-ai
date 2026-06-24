# Public-score pseudo blend report

Generated: 2026-06-22T18:00:49

## Known public constraints

- `last_guard_0p008_last0.588383_full0.587743` public=0.5920118473, pseudo_expected=0.5910109383, residual=-0.00100091
- `target_select_public_tight_last0.572477_full0.592635` public=0.5905116492, pseudo_expected=0.5932056179, residual=+0.00269397
- `02_guarded_targetwise` public=0.5935970063, pseudo_expected=0.5916997234, residual=-0.00189728

## Selected candidate

- candidate: `pseudo_target_tight_anchorlogit_w0p82`
- pseudo_public_logloss: `0.588794436`
- local full/last: `0.594948` / `0.588809`
- delta vs anchor full/last: `-0.000882` / `-0.004474`

## Top candidates

| candidate | pseudo_public_logloss | full_logloss | last_logloss | selector_score |
| --- | --- | --- | --- | --- |
| pseudo_target_tight_anchorlogit_w0p82 | 0.588794 | 0.594948 | 0.588809 | 0.589527 |
| pseudo_target_balanced_anchorlogit_w0p82 | 0.588794 | 0.594948 | 0.588809 | 0.589527 |
| pseudo_target_public_heavy_anchorlogit_w0p82 | 0.588794 | 0.594948 | 0.588809 | 0.589527 |
| pseudo_target_public_max_anchorlogit_w0p82 | 0.588794 | 0.594948 | 0.588809 | 0.589527 |
| pseudo_target_tight_anchorlogit_w0p7 | 0.588807 | 0.595029 | 0.589367 | 0.589533 |
| pseudo_target_balanced_anchorlogit_w0p7 | 0.588807 | 0.595029 | 0.589367 | 0.589533 |
| pseudo_target_public_heavy_anchorlogit_w0p7 | 0.588807 | 0.595029 | 0.589367 | 0.589533 |
| pseudo_target_public_max_anchorlogit_w0p7 | 0.588807 | 0.595029 | 0.589367 | 0.589533 |
| pseudo_target_tight_anchorlogit_w0p9 | 0.588800 | 0.594902 | 0.588453 | 0.589537 |
| pseudo_target_balanced_anchorlogit_w0p9 | 0.588800 | 0.594902 | 0.588453 | 0.589537 |
| pseudo_target_public_heavy_anchorlogit_w0p9 | 0.588800 | 0.594902 | 0.588453 | 0.589537 |
| pseudo_target_public_max_anchorlogit_w0p9 | 0.588800 | 0.594902 | 0.588453 | 0.589537 |
| pseudo_target_tight_anchorlogit_w0p96 | 0.588811 | 0.594873 | 0.588195 | 0.589551 |
| pseudo_target_balanced_anchorlogit_w0p96 | 0.588811 | 0.594873 | 0.588195 | 0.589551 |
| pseudo_target_public_heavy_anchorlogit_w0p96 | 0.588811 | 0.594873 | 0.588195 | 0.589551 |
| pseudo_target_public_max_anchorlogit_w0p96 | 0.588811 | 0.594873 | 0.588195 | 0.589551 |
| pseudo_target_tight | 0.588821 | 0.594855 | 0.588027 | 0.589563 |
| pseudo_target_balanced | 0.588821 | 0.594855 | 0.588027 | 0.589563 |
| pseudo_target_public_heavy | 0.588821 | 0.594855 | 0.588027 | 0.589563 |
| pseudo_target_public_max | 0.588821 | 0.594855 | 0.588027 | 0.589563 |

## Target choices

| target | source | full_logloss | last_logloss | pseudo_public_logloss | selector_score | test_abs_delta_mean | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | anchor | 0.590425 | 0.618506 | 0.597692 | 0.597692 | 0.000000 | pseudo_target_public_max |
| Q2 | anchor | 0.692880 | 0.638084 | 0.650598 | 0.650598 | 0.000000 | pseudo_target_public_max |
| Q3 | anchor | 0.662681 | 0.634692 | 0.638550 | 0.638550 | 0.000000 | pseudo_target_public_max |
| S1 | anchor | 0.492352 | 0.449544 | 0.481325 | 0.481325 | 0.000000 | pseudo_target_public_max |
| S2 | direction_gated_ablation_20260622/core_q1down_s2tight_s4tight | 0.555449 | 0.588833 | 0.558544 | 0.558599 | 0.009201 | pseudo_target_public_max |
| S3 | anchor | 0.541551 | 0.600679 | 0.555764 | 0.555764 | 0.000000 | pseudo_target_public_max |
| S4 | pattern_safe_candidates_20260622/q1_s2down25_s4up25 | 0.628649 | 0.585849 | 0.639276 | 0.639288 | 0.002004 | pseudo_target_public_max |

## Public feedback interpretation

- The newer `target_select_public_tight` score is lower than the prior `last_guard_0p008`, so public feedback rewards the restricted movement pattern.
- The search therefore optimizes a pseudo-public posterior first, then rejects candidates whose CV full/last degradation is too large.
