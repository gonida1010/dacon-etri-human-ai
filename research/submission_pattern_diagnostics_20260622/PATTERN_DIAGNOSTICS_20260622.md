# Submission Pattern Diagnostics

Generated: 2026-06-22 17:43:36

Submitted candidate key: `constrained_target_blend_logit_newton/last_guard_0p008`
Observed public score supplied by user: `0.5920118473`

## Read

- OOF gains are real only where the candidate's probability movement has high directional precision.
- Public risk rises when low-precision OOF movements appear often in test CSV rows.
- The submitted candidate is extremely close to the full logit blend; it mainly changes Q2/S1 by a small amount, so it did not test a meaningfully new public pattern.

## Local OOF Scores, Paired Artifacts

```
                                                                           candidate full_logloss last_logloss fold_std tail3_worst
                                          constrained_target_blend_logit_newton/full     0.587732     0.588717 0.015432    0.616208
                   residual_submission_blend_ridge_logit_newton/ridge_knn_blend_full     0.587732     0.588717 0.015432    0.616208
                              constrained_target_blend_logit_newton/last_guard_0p008     0.587743     0.588383 0.015408    0.616251
                              constrained_target_blend_logit_newton/last_guard_0p006     0.587774     0.588102 0.015387    0.616309
                    residual_single_model_opt_ridge_logit_newton/ridge_residual_full     0.587810     0.590089 0.015406    0.615949
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
                                 direction_gated_search_20260622/precision55_lowrisk     0.591125     0.574694 0.015294    0.618952
 public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p8     0.592405     0.575071 0.015610    0.620281
public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p65     0.592465     0.577300 0.015172    0.620268
 public_aware_stack_blend_20260622/target_select_public_tight_logit_anchorblend_w0p9     0.592477     0.573720 0.015928    0.620395
```

## Submitted Candidate Target Pattern, Last Fold

```
target logloss_gain_vs_anchor base_positive_rate oof_up_rate up_precision_actual_1 oof_down_rate down_precision_actual_0 test_mean_delta test_up_rate test_down_rate public_fp_risk_proxy public_fn_risk_proxy
    Q1               0.011973           0.588235    0.129412              0.636364      0.047059                0.750000        0.001879     0.024000       0.004000             0.000000             0.000000
    Q2              -0.006712           0.682353    0.011765              0.000000      0.011765                0.000000       -0.003942     0.000000       0.004000             0.000000             0.002000
    Q3               0.000000           0.647059    0.000000                            0.000000                                0.000000     0.000000       0.000000             0.000000             0.000000
    S1              -0.000523           0.741176    0.000000                            0.094118                0.500000       -0.018234     0.000000       0.016000             0.000000             0.000000
    S2               0.030326           0.564706    0.082353              0.428571      0.223529                0.578947       -0.009370     0.044000       0.080000             0.003143             0.000000
    S3              -0.005484           0.635294    0.070588              0.333333      0.329412                0.535714        0.005414     0.052000       0.032000             0.008667             0.000000
    S4               0.004716           0.541176    0.058824              1.000000      0.482353                0.487805       -0.027384     0.048000       0.324000             0.000000             0.003951
```

## Stronger True-Looking Patterns In Submitted Candidate

```
target logloss_gain_vs_anchor up_precision_actual_1 down_precision_actual_0 test_mean_delta test_abs_delta_mean
    S2               0.030326              0.428571                0.578947       -0.009370            0.031568
    Q1               0.011973              0.636364                0.750000        0.001879            0.017820
    S4               0.004716              1.000000                0.487805       -0.027384            0.043415
    Q3               0.000000                                                      0.000000            0.000000
    S1              -0.000523                                      0.500000       -0.018234            0.019682
    S3              -0.005484              0.333333                0.535714        0.005414            0.021748
    Q2              -0.006712              0.000000                0.000000       -0.003942            0.009984
```

## False-Positive Risk, Upward Probability Pushes

```
target base_positive_rate  oof_up_n up_precision_actual_1  false_positive_n  test_up_n test_up_rate public_fp_risk_proxy
    S3           0.635294         6              0.333333                 4         13     0.052000             0.008667
    S2           0.564706         7              0.428571                 4         11     0.044000             0.003143
    S4           0.541176         5              1.000000                 0         12     0.048000             0.000000
    Q1           0.588235        11              0.636364                 4          6     0.024000             0.000000
    Q2           0.682353         1              0.000000                 1          0     0.000000             0.000000
    Q3           0.647059         0                                       0          0     0.000000             0.000000
    S1           0.741176         0                                       0          0     0.000000             0.000000
```

## False-Negative Risk, Downward Probability Pushes

```
target base_positive_rate  oof_down_n down_precision_actual_0  false_negative_n  test_down_n test_down_rate public_fn_risk_proxy
    S4           0.541176          41                0.487805                21           81       0.324000             0.003951
    Q2           0.682353           1                0.000000                 1            1       0.004000             0.002000
    S2           0.564706          19                0.578947                 8           20       0.080000             0.000000
    S3           0.635294          28                0.535714                13            8       0.032000             0.000000
    S1           0.741176           8                0.500000                 4            4       0.016000             0.000000
    Q1           0.588235           4                0.750000                 1            1       0.004000             0.000000
    Q3           0.647059           0                                         0            0       0.000000             0.000000
```

## Submitted Candidate Test Distribution

```
target test_mean anchor_mean train_like_anchor_delta abs_delta_mean prob_p05 prob_p50 prob_p95
    Q1  0.495407    0.493527                0.001879       0.017820 0.170573 0.526163 0.819808
    Q2  0.581953    0.585894               -0.003942       0.009984 0.410258 0.573401 0.757530
    Q3  0.626437    0.626437                0.000000       0.000000 0.449979 0.596115 0.725764
    S1  0.690152    0.708386               -0.018234       0.019682 0.268053 0.750034 0.935582
    S2  0.646794    0.656164               -0.009370       0.031568 0.318273 0.645815 0.923169
    S3  0.678842    0.673428                0.005414       0.021748 0.284739 0.707639 0.886094
    S4  0.533670    0.561054               -0.027384       0.043415 0.282364 0.519536 0.809506
```

## Largest Subject-Level Test Shifts In Submitted Candidate

```
subject_id target  n test_mean anchor_mean mean_delta abs_delta_mean
      id04     S4 27  0.432782    0.496660  -0.063879       0.065293
      id07     S4 30  0.416182    0.471596  -0.055415       0.057925
      id07     S2 30  0.582262    0.629703  -0.047441       0.057654
      id02     S4 32  0.653063    0.697731  -0.044669       0.055645
      id03     S4 21  0.277360    0.315507  -0.038147       0.045103
      id05     S4 21  0.394363    0.431355  -0.036992       0.042638
      id03     Q1 21  0.714099    0.678723   0.035377       0.035377
      id10     S1 22  0.458572    0.492990  -0.034418       0.034418
      id10     S4 22  0.661514    0.692867  -0.031353       0.036538
      id05     S1 21  0.562685    0.592194  -0.029508       0.031490
      id09     S4 27  0.628016    0.602737   0.025280       0.031875
      id06     Q1 24  0.237738    0.261563  -0.023825       0.023825
      id09     S1 27  0.640006    0.663043  -0.023037       0.023662
      id01     S1 27  0.606129    0.629098  -0.022968       0.022968
      id03     S1 21  0.744748    0.765968  -0.021220       0.024790
      id01     S4 27  0.511475    0.532069  -0.020593       0.027157
      id07     S3 30  0.782724    0.763009   0.019715       0.024165
      id07     S1 30  0.791213    0.810628  -0.019414       0.019414
      id02     Q1 32  0.529214    0.510596   0.018618       0.019820
      id04     S3 27  0.551646    0.569802  -0.018156       0.023963
      id04     S2 27  0.665012    0.682956  -0.017945       0.027335
      id08     S1 19  0.418752    0.436491  -0.017739       0.022220
      id04     S1 27  0.769397    0.786945  -0.017548       0.017548
      id03     S3 21  0.560156    0.542766   0.017390       0.030523
```

## CSVs Closest To Submitted File

```
                                                                                                       path local_last_from_name local_full_from_name flat_corr_to_submitted mean_abs_diff_to_submitted mean_abs_diff_to_anchor
           submissions/constrained_target_blend_logit_newton/last_guard_0p008_last0.588383_full0.587743.csv             0.588383             0.587743               1.000000                   0.000000                0.020602
           submissions/constrained_target_blend_logit_newton/last_guard_0p006_last0.588102_full0.587774.csv             0.588102             0.587774               0.999981                   0.000329                0.020273
                       submissions/constrained_target_blend_logit_newton/full_last0.588717_full0.587732.csv             0.588717             0.587732               0.999981                   0.000329                0.020932
submissions/residual_submission_blend_ridge_logit_newton/ridge_knn_blend_full_last0.588717_full0.587732.csv             0.588717             0.587732               0.999981                   0.000329                0.020932
submissions/constrained_target_blend_logit_newton/positive_last_penalty_a0p25_last0.587745_full0.587837.csv             0.587745             0.587837               0.999902                   0.000555                0.020263
 submissions/constrained_target_blend_logit_newton/positive_last_penalty_a0p5_last0.587499_full0.587911.csv             0.587499             0.587911               0.999739                   0.000804                0.020205
           submissions/constrained_target_blend_logit_newton/last_guard_0p004_last0.587181_full0.588346.csv             0.587181             0.588346               0.999711                   0.001508                0.019580
           submissions/constrained_target_blend_logit_newton/last_guard_0p002_last0.586975_full0.588411.csv             0.586975             0.588411               0.999616                   0.001837                0.019251
submissions/constrained_target_blend_logit_newton/positive_last_penalty_a0p75_last0.586676_full0.588496.csv             0.586676             0.588496               0.999107                   0.001889                0.020265
 submissions/residual_single_model_opt_ridge_logit_newton/ridge_residual_full_last0.590089_full0.587810.csv             0.590089             0.587810               0.999696                   0.001999                0.022496
               submissions/constrained_target_blend_logit_newton/last_guard_0_last0.586274_full0.588584.csv             0.586274             0.588584               0.999395                   0.002668                0.018715
   submissions/constrained_target_blend_logit_newton/positive_last_penalty_a1_last0.586289_full0.588583.csv             0.586289             0.588583               0.999183                   0.002894                0.018873
         submissions/constrained_target_blend_logit_newton/tradeoff_cap_a0p15_last0.586630_full0.587898.csv             0.586630             0.587898               0.999611                   0.003009                0.020919
         submissions/constrained_target_blend_logit_newton/tradeoff_cap_a0p25_last0.584942_full0.588271.csv             0.584942             0.588271               0.998238                   0.006289                0.020287
         submissions/constrained_target_blend_logit_newton/tradeoff_cap_a0p35_last0.584695_full0.588345.csv             0.584695             0.588345               0.998078                   0.006537                0.020229
          submissions/constrained_target_blend_logit_newton/tradeoff_cap_a0p5_last0.583872_full0.588696.csv             0.583872             0.588696               0.996985                   0.008436                0.020769
                submissions/pattern_safe_candidates_20260622/q1_s2down_s4full_last0.585601_full0.593284.csv             0.585601             0.593284               0.993597                   0.008930                0.011672
                  submissions/pattern_safe_candidates_20260622/q1_s2full_s4up_last0.585744_full0.593427.csv             0.585744             0.593427               0.992323                   0.012402                0.008200
```

## Generated Figures

- `submitted_target_mean_delta.png`
- `submitted_subject_target_delta_heatmap.png`
- `oof_gain_vs_test_movement.png`
