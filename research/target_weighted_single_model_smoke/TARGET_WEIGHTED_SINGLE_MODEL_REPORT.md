# Target Weighted Single Model

This run creates new single-model OOF/test sources, not final submission claims.

## Candidate Scores

                       candidate  full_logloss  last_logloss  full_delta_vs_anchor  last_delta_vs_anchor  fold_std  tail3_mean  tail3_worst  rank_score  fold0_logloss  fold1_logloss  fold2_logloss  fold3_logloss  fold4_logloss
target_weighted_single_composite       0.59569      0.593282             -0.000139                   0.0  0.013819    0.601707     0.622422    0.597011       0.592005       0.582211       0.622422       0.589416       0.593282
     target_weighted_single_last       0.59569      0.593282             -0.000139                   0.0  0.013819    0.601707     0.622422    0.597011       0.592005       0.582211       0.622422       0.589416       0.593282
     target_weighted_single_full       0.59569      0.593282             -0.000139                   0.0  0.013819    0.601707     0.622422    0.597011       0.592005       0.582211       0.622422       0.589416       0.593282

## Best Target Choices

                       candidate target model param_profile weight_profile  top_k shrink_mode  shrink  full_logloss  last_logloss  rank_score
target_weighted_single_composite     Q1  lgbm        smooth        uniform     20       logit    0.00      0.590425      0.618506    0.622909
target_weighted_single_composite     Q2  lgbm        smooth        uniform     20       logit    0.00      0.692880      0.638084    0.654447
target_weighted_single_composite     Q3  lgbm        smooth        uniform     20       logit    0.00      0.662681      0.634692    0.643253
target_weighted_single_composite     S1  lgbm        smooth        uniform     20       logit    0.00      0.492352      0.449544    0.454510
target_weighted_single_composite     S2  lgbm        smooth        uniform     20       logit    0.00      0.560120      0.622815    0.632413
target_weighted_single_composite     S3  lgbm        smooth        uniform     20       logit    0.00      0.541551      0.600679    0.608057
target_weighted_single_composite     S4  lgbm        smooth        uniform     20       logit    0.25      0.629823      0.588654    0.593616
     target_weighted_single_last     Q1  lgbm        smooth        uniform     20       logit    0.00      0.590425      0.618506    0.622909
     target_weighted_single_last     Q2  lgbm        smooth        uniform     20       logit    0.00      0.692880      0.638084    0.654447
     target_weighted_single_last     Q3  lgbm        smooth        uniform     20       logit    0.00      0.662681      0.634692    0.643253
     target_weighted_single_last     S1  lgbm        smooth        uniform     20       logit    0.00      0.492352      0.449544    0.454510
     target_weighted_single_last     S2  lgbm        smooth        uniform     20       logit    0.00      0.560120      0.622815    0.632413
     target_weighted_single_last     S3  lgbm        smooth        uniform     20       logit    0.00      0.541551      0.600679    0.608057
     target_weighted_single_last     S4  lgbm        smooth        uniform     20       logit    0.25      0.629823      0.588654    0.593616
     target_weighted_single_full     Q1  lgbm        smooth        uniform     20       logit    0.00      0.590425      0.618506    0.622909
     target_weighted_single_full     Q2  lgbm        smooth        uniform     20       logit    0.00      0.692880      0.638084    0.654447
     target_weighted_single_full     Q3  lgbm        smooth        uniform     20       logit    0.00      0.662681      0.634692    0.643253
     target_weighted_single_full     S1  lgbm        smooth        uniform     20       logit    0.00      0.492352      0.449544    0.454510
     target_weighted_single_full     S2  lgbm        smooth        uniform     20       logit    0.00      0.560120      0.622815    0.632413
     target_weighted_single_full     S3  lgbm        smooth        uniform     20       logit    0.00      0.541551      0.600679    0.608057
     target_weighted_single_full     S4  lgbm        smooth        uniform     20       logit    0.25      0.629823      0.588654    0.593616

## Early Stopping

- Every fold/model uses validation logloss early stopping where the model supports it.
- `fit_diagnostics.csv` records `best_iteration_mean`, `stop_policy`, and `fallback_logic`.