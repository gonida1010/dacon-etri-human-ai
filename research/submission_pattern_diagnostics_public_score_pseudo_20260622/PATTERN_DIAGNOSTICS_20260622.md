# Submission Pattern Diagnostics

Generated: 2026-06-22 18:02:07

Submitted candidate key: `public_score_pseudo_blend_20260622/pseudo_target_tight_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948`
Observed public score supplied by user: `0.0000000000`

## Read

- OOF gains are real only where the candidate's probability movement has high directional precision.
- Public risk rises when low-precision OOF movements appear often in test CSV rows.
- The submitted candidate is extremely close to the full logit blend; it mainly changes Q2/S1 by a small amount, so it did not test a meaningfully new public pattern.

## Local OOF Scores, Paired Artifacts

```
                                                                                candidate full_logloss last_logloss fold_std tail3_worst
                                               constrained_target_blend_logit_newton/full     0.587732     0.588717 0.015432    0.616208
                                   constrained_target_blend_logit_newton/last_guard_0p008     0.587743     0.588383 0.015408    0.616251
                                   constrained_target_blend_logit_newton/last_guard_0p006     0.587774     0.588102 0.015387    0.616309
                        constrained_target_blend_logit_newton/positive_last_penalty_a0p25     0.587837     0.587745 0.015354    0.616332
                                 constrained_target_blend_logit_newton/tradeoff_cap_a0p15     0.587898     0.586630 0.015405    0.616703
                         constrained_target_blend_logit_newton/positive_last_penalty_a0p5     0.587911     0.587499 0.015321    0.616349
                                 constrained_target_blend_logit_newton/tradeoff_cap_a0p25     0.588271     0.584942 0.015421    0.617307
                                 constrained_target_blend_logit_newton/tradeoff_cap_a0p35     0.588345     0.584695 0.015409    0.617324
                                   constrained_target_blend_logit_newton/last_guard_0p004     0.588346     0.587181 0.015281    0.617033
                                   constrained_target_blend_logit_newton/last_guard_0p002     0.588411     0.586975 0.015270    0.617124
                        constrained_target_blend_logit_newton/positive_last_penalty_a0p75     0.588496     0.586676 0.015244    0.617060
                           constrained_target_blend_logit_newton/positive_last_penalty_a1     0.588583     0.586289 0.015323    0.617463
                                       constrained_target_blend_logit_newton/last_guard_0     0.588584     0.586274 0.015277    0.617361
                                  constrained_target_blend_logit_newton/tradeoff_cap_a0p5     0.588696     0.583872 0.015569    0.617980
      public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p8     0.592405     0.575071 0.015610    0.620281
     public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p65     0.592465     0.577300 0.015172    0.620268
      public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p9     0.592477     0.573720 0.015928    0.620395
                             public_aware_stack_blend_20260622/target_select_public_tight     0.592635     0.572477 0.016266    0.620591
public_aware_stack_blend_20260622/target_select_public_aggressive_logit_anchorblend_w0p65     0.592981     0.576593 0.015355    0.620496
  public_aware_stack_blend_20260622/target_select_public_balanced_logit_anchorblend_w0p65     0.592981     0.576593 0.015355    0.620496
```

## Submitted Candidate Target Pattern, Last Fold

```
target logloss_gain_vs_anchor base_positive_rate oof_up_rate up_precision_actual_1 oof_down_rate down_precision_actual_0 test_mean_delta test_up_rate test_down_rate public_fp_risk_proxy public_fn_risk_proxy
    Q1              -0.000000           0.588235    0.000000                            0.000000                               -0.000000     0.000000       0.000000             0.000000             0.000000
    Q2               0.000000           0.682353    0.000000                            0.000000                               -0.000000     0.000000       0.000000             0.000000             0.000000
    Q3               0.000000           0.647059    0.000000                            0.000000                               -0.000000     0.000000       0.000000             0.000000             0.000000
    S1              -0.000000           0.741176    0.000000                            0.000000                               -0.000000     0.000000       0.000000             0.000000             0.000000
    S2               0.029003           0.564706    0.000000                            0.141176                0.666667       -0.007815     0.000000       0.052000             0.000000             0.000000
    S3               0.000000           0.635294    0.000000                            0.000000                               -0.000000     0.000000       0.000000             0.000000             0.000000
    S4               0.002312           0.541176    0.141176              0.833333      0.000000                                0.001646     0.128000       0.000000             0.000000             0.000000
```

## Stronger True-Looking Patterns In Submitted Candidate

```
target logloss_gain_vs_anchor up_precision_actual_1 down_precision_actual_0 test_mean_delta test_abs_delta_mean
    S2               0.029003                                      0.666667       -0.007815            0.007815
    S4               0.002312              0.833333                                0.001646            0.001646
    S3               0.000000                                                     -0.000000            0.000000
    Q2               0.000000                                                     -0.000000            0.000000
    Q3               0.000000                                                     -0.000000            0.000000
    S1              -0.000000                                                     -0.000000            0.000000
    Q1              -0.000000                                                     -0.000000            0.000000
```

## False-Positive Risk, Upward Probability Pushes

```
target base_positive_rate  oof_up_n up_precision_actual_1  false_positive_n  test_up_n test_up_rate public_fp_risk_proxy
    S4           0.541176        12              0.833333                 2         32     0.128000             0.000000
    Q1           0.588235         0                                       0          0     0.000000             0.000000
    Q2           0.682353         0                                       0          0     0.000000             0.000000
    Q3           0.647059         0                                       0          0     0.000000             0.000000
    S1           0.741176         0                                       0          0     0.000000             0.000000
    S2           0.564706         0                                       0          0     0.000000             0.000000
    S3           0.635294         0                                       0          0     0.000000             0.000000
```

## False-Negative Risk, Downward Probability Pushes

```
target base_positive_rate  oof_down_n down_precision_actual_0  false_negative_n  test_down_n test_down_rate public_fn_risk_proxy
    S2           0.564706          12                0.666667                 4           13       0.052000             0.000000
    Q1           0.588235           0                                         0            0       0.000000             0.000000
    Q2           0.682353           0                                         0            0       0.000000             0.000000
    Q3           0.647059           0                                         0            0       0.000000             0.000000
    S1           0.741176           0                                         0            0       0.000000             0.000000
    S3           0.635294           0                                         0            0       0.000000             0.000000
    S4           0.541176           0                                         0            0       0.000000             0.000000
```

## Submitted Candidate Test Distribution

```
target test_mean anchor_mean train_like_anchor_delta abs_delta_mean prob_p05 prob_p50 prob_p95
    Q1  0.493527    0.493527               -0.000000       0.000000 0.178895 0.514747 0.813202
    Q2  0.585894    0.585894               -0.000000       0.000000 0.410294 0.578476 0.760021
    Q3  0.626437    0.626437               -0.000000       0.000000 0.449979 0.596115 0.725764
    S1  0.708386    0.708386               -0.000000       0.000000 0.299429 0.768121 0.934638
    S2  0.648349    0.656164               -0.007815       0.007815 0.336216 0.633682 0.925633
    S3  0.673428    0.673428               -0.000000       0.000000 0.276593 0.694659 0.884306
    S4  0.562701    0.561054                0.001646       0.001646 0.326516 0.538057 0.807617
```

## Largest Subject-Level Test Shifts In Submitted Candidate

```
subject_id target  n test_mean anchor_mean mean_delta abs_delta_mean
      id07     S2 30  0.594621    0.629703  -0.035082       0.035082
      id08     S2 19  0.557955    0.580918  -0.022963       0.022963
      id10     S2 22  0.398982    0.407210  -0.008228       0.008228
      id09     S4 27  0.608601    0.602737   0.005865       0.005865
      id06     S4 24  0.804559    0.798856   0.005703       0.005703
      id03     S2 21  0.557136    0.559923  -0.002787       0.002787
      id06     S2 24  0.894213    0.896897  -0.002684       0.002684
      id05     S2 21  0.345893    0.348293  -0.002400       0.002400
      id02     S2 32  0.912177    0.914106  -0.001929       0.001929
      id04     S2 27  0.681147    0.682956  -0.001810       0.001810
      id02     S4 32  0.698858    0.697731   0.001126       0.001126
      id03     S4 21  0.316219    0.315507   0.000712       0.000712
      id01     S4 27  0.532742    0.532069   0.000673       0.000673
      id08     S4 19  0.507931    0.507314   0.000617       0.000617
      id05     S4 21  0.431934    0.431355   0.000579       0.000579
      id10     S4 22  0.693399    0.692867   0.000532       0.000532
      id07     S4 30  0.471854    0.471596   0.000257       0.000257
      id04     S4 27  0.496805    0.496660   0.000145       0.000145
      id08     S3 19  0.661049    0.661049  -0.000000       0.000000
      id01     S3 27  0.799922    0.799922  -0.000000       0.000000
      id09     S3 27  0.694659    0.694659  -0.000000       0.000000
      id08     S1 19  0.436491    0.436491  -0.000000       0.000000
      id07     Q1 30  0.529652    0.529652  -0.000000       0.000000
      id10     Q3 22  0.510450    0.510450  -0.000000       0.000000
```

## CSVs Closest To Submitted File

```
                                                                                                                                                    path local_last_from_name local_full_from_name flat_corr_to_submitted mean_abs_diff_to_submitted mean_abs_diff_to_anchor
                    submissions/public_score_pseudo_blend_20260622/pseudo_target_balanced_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948.csv             0.588809             0.594948               1.000000                   0.000000                0.001352
                submissions/public_score_pseudo_blend_20260622/pseudo_target_public_heavy_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948.csv             0.588809             0.594948               1.000000                   0.000000                0.001352
                  submissions/public_score_pseudo_blend_20260622/pseudo_target_public_max_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948.csv             0.588809             0.594948               1.000000                   0.000000                0.001352
                       submissions/public_score_pseudo_blend_20260622/pseudo_target_tight_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948.csv             0.588809             0.594948               1.000000                   0.000000                0.001352
                     submissions/public_score_pseudo_blend_20260622/pseudo_target_balanced_anchorlogit_w0p9_pseudo0.588800_last0.588453_full0.594902.csv             0.588453             0.594902               0.999972                   0.000114                0.001465
                 submissions/public_score_pseudo_blend_20260622/pseudo_target_public_heavy_anchorlogit_w0p9_pseudo0.588800_last0.588453_full0.594902.csv             0.588453             0.594902               0.999972                   0.000114                0.001465
                   submissions/public_score_pseudo_blend_20260622/pseudo_target_public_max_anchorlogit_w0p9_pseudo0.588800_last0.588453_full0.594902.csv             0.588453             0.594902               0.999972                   0.000114                0.001465
                        submissions/public_score_pseudo_blend_20260622/pseudo_target_tight_anchorlogit_w0p9_pseudo0.588800_last0.588453_full0.594902.csv             0.588453             0.594902               0.999972                   0.000114                0.001465
                     submissions/public_score_pseudo_blend_20260622/pseudo_target_balanced_anchorlogit_w0p7_pseudo0.588807_last0.589367_full0.595029.csv             0.589367             0.595029               0.999922                   0.000180                0.001172
                 submissions/public_score_pseudo_blend_20260622/pseudo_target_public_heavy_anchorlogit_w0p7_pseudo0.588807_last0.589367_full0.595029.csv             0.589367             0.595029               0.999922                   0.000180                0.001172
                   submissions/public_score_pseudo_blend_20260622/pseudo_target_public_max_anchorlogit_w0p7_pseudo0.588807_last0.589367_full0.595029.csv             0.589367             0.595029               0.999922                   0.000180                0.001172
                        submissions/public_score_pseudo_blend_20260622/pseudo_target_tight_anchorlogit_w0p7_pseudo0.588807_last0.589367_full0.595029.csv             0.589367             0.595029               0.999922                   0.000180                0.001172
submissions/public_score_pseudo_blend_20260622_ridge0/pseudo_simplex_logit_cv0p18_l20p002_anchorlogit_w0p82_pseudo0.589454_last0.590501_full0.595395.csv             0.590501             0.595395               0.999202                   0.000972                0.001445
submissions/public_score_pseudo_blend_20260622_ridge0/pseudo_simplex_logit_cv0p02_l20p002_anchorlogit_w0p82_pseudo0.589437_last0.589439_full0.595036.csv             0.589439             0.595036               0.999216                   0.001006                0.001661
submissions/public_score_pseudo_blend_20260622_ridge0/pseudo_simplex_logit_cv0p18_l20p002_anchorlogit_w0p92_pseudo0.589425_last0.590264_full0.595372.csv             0.590264             0.595372               0.999032                   0.001020                0.001547
                  submissions/public_score_pseudo_blend_20260622_ridge0/pseudo_simplex_logit_cv0p18_l20p002_pseudo0.589412_last0.590082_full0.595356.csv             0.590082             0.595356               0.998914                   0.001050                0.001620
submissions/public_score_pseudo_blend_20260622_ridge0/pseudo_simplex_logit_cv0p02_l20p002_anchorlogit_w0p92_pseudo0.589407_last0.589081_full0.594971.csv             0.589081             0.594971               0.999045                   0.001058                0.001789
submissions/public_score_pseudo_blend_20260622_ridge0/pseudo_simplex_logit_cv0p08_l20p002_anchorlogit_w0p82_pseudo0.589447_last0.590304_full0.595277.csv             0.590304             0.595277               0.999193                   0.001090                0.001563
```

## Generated Figures

- `submitted_target_mean_delta.png`
- `submitted_subject_target_delta_heatmap.png`
- `oof_gain_vs_test_movement.png`
