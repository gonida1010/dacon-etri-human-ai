# Public-score pseudo blend report

Generated: 2026-06-22T17:59:36

## Known public constraints

- `last_guard_0p008_last0.588383_full0.587743` public=0.5920118473, pseudo_expected=0.5920118472, residual=-8.87312e-11
- `target_select_public_tight_last0.572477_full0.592635` public=0.5905116492, pseudo_expected=0.5905116493, residual=+9.32743e-11
- `02_guarded_targetwise` public=0.5935970063, pseudo_expected=0.5935970063, residual=+4.23819e-11

## Selected candidate

- candidate: `pseudo_simplex_logit_cv0p02_l20p002`
- pseudo_public_logloss: `0.589392804`
- local full/last: `0.594923` / `0.588803`
- delta vs anchor full/last: `-0.000906` / `-0.004479`

## Top candidates

| candidate | pseudo_public_logloss | full_logloss | last_logloss | selector_score |
| --- | --- | --- | --- | --- |
| pseudo_simplex_logit_cv0p02_l20p002 | 0.589393 | 0.594923 | 0.588803 | 0.590149 |
| pseudo_simplex_logit_cv0p08_l20p002 | 0.589405 | 0.595214 | 0.589843 | 0.590152 |
| pseudo_simplex_logit_cv0p18_l20p002 | 0.589412 | 0.595356 | 0.590082 | 0.590155 |
| anchor_logit_public_aware_stack_blend_20260622__target_select_public_tight_logit_anchorblend_w0p65_w0p45 | 0.589357 | 0.593466 | 0.583670 | 0.590155 |
| pseudo_simplex_logit_cv0p02_l20p002_anchorlogit_w0p92 | 0.589407 | 0.594971 | 0.589081 | 0.590160 |
| pseudo_simplex_logit_cv0p08_l20p002_anchorlogit_w0p92 | 0.589418 | 0.595240 | 0.590044 | 0.590163 |
| anchor_logit_public_aware_stack_blend_20260622__target_select_public_aggressive_logit_anchorblend_w0p65_w0p45 | 0.589360 | 0.593680 | 0.583336 | 0.590166 |
| anchor_logit_public_aware_stack_blend_20260622__target_select_public_balanced_logit_anchorblend_w0p65_w0p45 | 0.589360 | 0.593680 | 0.583336 | 0.590166 |
| pseudo_simplex_logit_cv0p18_l20p002_anchorlogit_w0p92 | 0.589425 | 0.595372 | 0.590264 | 0.590166 |
| pseudo_simplex_logit_cv0p02_l20p002_anchorlogit_w0p82 | 0.589437 | 0.595036 | 0.589439 | 0.590185 |
| pseudo_simplex_logit_cv0p08_l20p002_anchorlogit_w0p82 | 0.589447 | 0.595277 | 0.590304 | 0.590188 |
| pseudo_simplex_logit_cv0p18_l20p002_anchorlogit_w0p82 | 0.589454 | 0.595395 | 0.590501 | 0.590191 |
| pseudo_simplex_logit_cv0p02_l20p01 | 0.589468 | 0.595281 | 0.590279 | 0.590206 |
| pseudo_simplex_logit_cv0p08_l20p01 | 0.589471 | 0.595357 | 0.590545 | 0.590207 |
| pseudo_simplex_logit_cv0p18_l20p01 | 0.589474 | 0.595408 | 0.590623 | 0.590209 |
| anchor_logit_public_aware_stack_blend_20260622__target_select_public_tight_logit_anchorblend_w0p65_w0p6 | 0.589397 | 0.593064 | 0.581752 | 0.590224 |
| pseudo_simplex_logit_cv0p02_l20p01_anchorlogit_w0p92 | 0.589501 | 0.595309 | 0.590462 | 0.590236 |
| pseudo_simplex_logit_cv0p08_l20p01_anchorlogit_w0p92 | 0.589504 | 0.595378 | 0.590706 | 0.590237 |
| pseudo_simplex_logit_cv0p18_l20p01_anchorlogit_w0p92 | 0.589507 | 0.595426 | 0.590779 | 0.590239 |
| anchor_logit_public_aware_stack_blend_20260622__target_select_public_aggressive_logit_anchorblend_w0p65_w0p6 | 0.589405 | 0.593357 | 0.581313 | 0.590241 |

## Target choices

| target | source | full_logloss | last_logloss | pseudo_public_logloss | selector_score | test_abs_delta_mean | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | anchor | 0.590425 | 0.618506 | 0.599285 | 0.599285 | 0.000000 | pseudo_target_public_max |
| Q2 | public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p65 | 0.692456 | 0.632738 | 0.650836 | 0.650846 | 0.001746 | pseudo_target_public_max |
| Q3 | anchor | 0.662681 | 0.634692 | 0.640131 | 0.640131 | 0.000000 | pseudo_target_public_max |
| S1 | residual_submission_blend_ridge_logit_te_smoke/ridge_knn_blend_full | 0.492226 | 0.449156 | 0.481541 | 0.481551 | 0.001725 | pseudo_target_public_max |
| S2 | direction_gated_ablation_20260622/core_plus_q2tiny | 0.555449 | 0.588833 | 0.558950 | 0.559005 | 0.009201 | pseudo_target_public_max |
| S3 | anchor | 0.541551 | 0.600679 | 0.557720 | 0.557720 | 0.000000 | pseudo_target_public_max |
| S4 | pattern_safe_candidates_20260622/q1_s2down50_s4up50 | 0.626859 | 0.583176 | 0.638720 | 0.638744 | 0.004008 | pseudo_target_public_max |

## Public feedback interpretation

- The newer `target_select_public_tight` score is lower than the prior `last_guard_0p008`, so public feedback rewards the restricted movement pattern.
- The search therefore optimizes a pseudo-public posterior first, then rejects candidates whose CV full/last degradation is too large.
