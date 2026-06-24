# Target Weighted Single Model

This run creates new single-model OOF/test sources, not final submission claims.

## Candidate Scores

                       candidate  full_logloss  last_logloss  full_delta_vs_anchor  last_delta_vs_anchor  fold_std  tail3_mean  tail3_worst  rank_score  fold0_logloss  fold1_logloss  fold2_logloss  fold3_logloss  fold4_logloss
target_weighted_single_composite      0.592353      0.586688             -0.003477             -0.006594  0.006320    0.594632     0.604356    0.588438       0.589676       0.588332       0.604356       0.592852       0.586688
     target_weighted_single_last      0.593006      0.586345             -0.002823             -0.006937  0.007348    0.595444     0.607136    0.588450       0.589871       0.588993       0.607136       0.592852       0.586345
     target_weighted_single_full      0.591406      0.587857             -0.004424             -0.005425  0.005970    0.594447     0.602325    0.589275       0.588384       0.585560       0.602325       0.593160       0.587857

## Best Target Choices

                       candidate target model param_profile   weight_profile  top_k shrink_mode  shrink  full_logloss  last_logloss  rank_score
target_weighted_single_composite     Q1  lgbm        leaf31      target_auto    100        prob    0.05      0.590962      0.617525    0.622010
target_weighted_single_composite     Q2  lgbm        smooth recent_anchorerr    240       logit    0.70      0.669502      0.604338    0.610708
target_weighted_single_composite     Q3  lgbm        leaf31          uniform    100       logit    0.35      0.660825      0.635618    0.641673
target_weighted_single_composite     S1  lgbm        smooth    subject_class    160       logit    0.05      0.492591      0.448887    0.454102
target_weighted_single_composite     S2  lgbm        smooth      target_auto    240       logit    0.25      0.560148      0.611811    0.620212
target_weighted_single_composite     S3  lgbm           mid recent_anchorerr     60       logit    0.08      0.541644      0.599985    0.607201
target_weighted_single_composite     S4  lgbm        smooth          uniform     60        prob    0.00      0.630799      0.588654    0.593616
     target_weighted_single_last     Q1  lgbm           mid    subject_class    160        prob    0.12      0.592765      0.617074    0.622618
     target_weighted_single_last     Q2  lgbm        smooth recent_anchorerr    240       logit    0.70      0.669502      0.604338    0.610708
     target_weighted_single_last     Q3  lgbm           mid      target_auto    240       logit    0.08      0.661214      0.634613    0.642655
     target_weighted_single_last     S1  lgbm        smooth recent_anchorerr     60        prob    0.08      0.494665      0.448074    0.455648
     target_weighted_single_last     S2  lgbm        smooth      target_auto    240       logit    0.25      0.560148      0.611811    0.620212
     target_weighted_single_last     S3  lgbm           mid recent_anchorerr     60       logit    0.12      0.541953      0.599852    0.607209
     target_weighted_single_last     S4  lgbm        smooth          uniform     60        prob    0.00      0.630799      0.588654    0.593616
     target_weighted_single_full     Q1  lgbm        smooth          uniform     60       logit    0.00      0.590425      0.618506    0.622909
     target_weighted_single_full     Q2  lgbm        smooth recent_anchorerr    240       logit    0.70      0.669502      0.604338    0.610708
     target_weighted_single_full     Q3  lgbm        smooth recent_anchorerr    240        prob    0.50      0.657869      0.637555    0.642834
     target_weighted_single_full     S1  lgbm        smooth          uniform     60        prob    0.00      0.492352      0.449544    0.454510
     target_weighted_single_full     S2  lgbm        leaf31      target_auto    160       logit    0.12      0.557499      0.615890    0.625065
     target_weighted_single_full     S3  lgbm        smooth recent_anchorerr    240       logit    0.05      0.541391      0.600515    0.607867
     target_weighted_single_full     S4  lgbm        smooth          uniform     60        prob    0.00      0.630799      0.588654    0.593616

## Early Stopping

- Every fold/model uses validation logloss early stopping where the model supports it.
- `fit_diagnostics.csv` records `best_iteration_mean`, `stop_policy`, and `fallback_logic`.