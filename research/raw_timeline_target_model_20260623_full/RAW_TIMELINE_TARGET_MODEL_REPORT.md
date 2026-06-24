# Raw Timeline Target Model

## Candidate Scores

             candidate  full_logloss  last_logloss  full_delta_vs_anchor  last_delta_vs_anchor  fold_std  tail3_mean  tail3_worst  fold0_logloss  fold1_logloss  fold2_logloss  fold3_logloss  fold4_logloss  Q1_fp_rate  Q1_fn_rate  Q1_pred_pos_rate  Q1_true_pos_rate  Q1_brier  Q2_fp_rate  Q2_fn_rate  Q2_pred_pos_rate  Q2_true_pos_rate  Q2_brier  Q3_fp_rate  Q3_fn_rate  Q3_pred_pos_rate  Q3_true_pos_rate  Q3_brier  S1_fp_rate  S1_fn_rate  S1_pred_pos_rate  S1_true_pos_rate  S1_brier  S2_fp_rate  S2_fn_rate  S2_pred_pos_rate  S2_true_pos_rate  S2_brier  S3_fp_rate  S3_fn_rate  S3_pred_pos_rate  S3_true_pos_rate  S3_brier  S4_fp_rate  S4_fn_rate  S4_pred_pos_rate  S4_true_pos_rate  S4_brier
     raw_timeline_full      0.584434      0.586415             -0.011396             -0.006868  0.007583    0.588015     0.597139       0.584559       0.574108       0.597139       0.580492       0.586415    0.137778    0.166667          0.466667          0.495556  0.202468    0.377778    0.028889          0.911111          0.562222  0.240084    0.300000    0.077778          0.822222               0.6  0.228466    0.155556    0.075556          0.762222          0.682222  0.158719    0.191111    0.080000          0.762222          0.651111  0.181588    0.208889    0.055556          0.815556          0.662222  0.176919    0.217778    0.146667          0.631111              0.56  0.216692
raw_timeline_composite      0.586764      0.590669             -0.009066             -0.002613  0.010546    0.591631     0.605283       0.584623       0.575265       0.605283       0.578939       0.590669    0.160000    0.153333          0.502222          0.495556  0.202586    0.297778    0.193333          0.666667          0.562222  0.242004    0.242222    0.157778          0.684444               0.6  0.231602    0.146667    0.080000          0.748889          0.682222  0.158577    0.168889    0.113333          0.706667          0.651111  0.182603    0.168889    0.102222          0.728889          0.662222  0.178301    0.204444    0.153333          0.611111              0.56  0.217043
 raw_timeline_fp_guard      0.586764      0.590669             -0.009066             -0.002613  0.010546    0.591631     0.605283       0.584623       0.575265       0.605283       0.578939       0.590669    0.160000    0.153333          0.502222          0.495556  0.202586    0.297778    0.193333          0.666667          0.562222  0.242004    0.242222    0.157778          0.684444               0.6  0.231602    0.146667    0.080000          0.748889          0.682222  0.158577    0.168889    0.113333          0.706667          0.651111  0.182603    0.168889    0.102222          0.728889          0.662222  0.178301    0.204444    0.153333          0.611111              0.56  0.217043
     raw_timeline_last      0.595367      0.572793             -0.000463             -0.020489  0.018730    0.600268     0.628044       0.592522       0.583223       0.628044       0.599967       0.572793    0.160000    0.153333          0.502222          0.495556  0.202586    0.326667    0.140000          0.748889          0.562222  0.251670    0.322222    0.073333          0.848889               0.6  0.230241    0.204444    0.042222          0.844444          0.682222  0.163071    0.142222    0.160000          0.633333          0.651111  0.187989    0.244444    0.046667          0.860000          0.662222  0.183568    0.246667    0.131111          0.675556              0.56  0.218850

## FP/FN Summary

             candidate target  fp_rate  fn_rate  pred_pos_rate  true_pos_rate    brier
raw_timeline_composite     Q1 0.160000 0.153333       0.502222       0.495556 0.202586
raw_timeline_composite     Q2 0.297778 0.193333       0.666667       0.562222 0.242004
raw_timeline_composite     Q3 0.242222 0.157778       0.684444       0.600000 0.231602
raw_timeline_composite     S1 0.146667 0.080000       0.748889       0.682222 0.158577
raw_timeline_composite     S2 0.168889 0.113333       0.706667       0.651111 0.182603
raw_timeline_composite     S3 0.168889 0.102222       0.728889       0.662222 0.178301
raw_timeline_composite     S4 0.204444 0.153333       0.611111       0.560000 0.217043
 raw_timeline_fp_guard     Q1 0.160000 0.153333       0.502222       0.495556 0.202586
 raw_timeline_fp_guard     Q2 0.297778 0.193333       0.666667       0.562222 0.242004
 raw_timeline_fp_guard     Q3 0.242222 0.157778       0.684444       0.600000 0.231602
 raw_timeline_fp_guard     S1 0.146667 0.080000       0.748889       0.682222 0.158577
 raw_timeline_fp_guard     S2 0.168889 0.113333       0.706667       0.651111 0.182603
 raw_timeline_fp_guard     S3 0.168889 0.102222       0.728889       0.662222 0.178301
 raw_timeline_fp_guard     S4 0.204444 0.153333       0.611111       0.560000 0.217043
     raw_timeline_full     Q1 0.137778 0.166667       0.466667       0.495556 0.202468
     raw_timeline_full     Q2 0.377778 0.028889       0.911111       0.562222 0.240084
     raw_timeline_full     Q3 0.300000 0.077778       0.822222       0.600000 0.228466
     raw_timeline_full     S1 0.155556 0.075556       0.762222       0.682222 0.158719
     raw_timeline_full     S2 0.191111 0.080000       0.762222       0.651111 0.181588
     raw_timeline_full     S3 0.208889 0.055556       0.815556       0.662222 0.176919
     raw_timeline_full     S4 0.217778 0.146667       0.631111       0.560000 0.216692
     raw_timeline_last     Q1 0.160000 0.153333       0.502222       0.495556 0.202586
     raw_timeline_last     Q2 0.326667 0.140000       0.748889       0.562222 0.251670
     raw_timeline_last     Q3 0.322222 0.073333       0.848889       0.600000 0.230241
     raw_timeline_last     S1 0.204444 0.042222       0.844444       0.682222 0.163071
     raw_timeline_last     S2 0.142222 0.160000       0.633333       0.651111 0.187989
     raw_timeline_last     S3 0.244444 0.046667       0.860000       0.662222 0.183568
     raw_timeline_last     S4 0.246667 0.131111       0.675556       0.560000 0.218850

## Target Choices

             candidate target  scope profile  weight_profile  top_k blend_mode  model_weight  intercept  temperature  full_logloss  last_logloss  fp_rate  fn_rate  pred_pos_rate  rank_score
raw_timeline_composite     Q1 target compact         uniform     60      logit          0.00       0.06         1.00      0.590762      0.612457 0.160000 0.153333       0.502222    0.597260
raw_timeline_composite     Q2    all     mid        fp_guard     60      logit          0.70       0.06         1.15      0.675647      0.679731 0.297778 0.193333       0.666667    0.680362
raw_timeline_composite     Q3    all compact recent_fp_guard    220       prob          0.70      -0.08         0.90      0.654098      0.649363 0.242222 0.157778       0.684444    0.657245
raw_timeline_composite     S1    all     mid recent_fp_guard    220      logit          0.20      -0.30         0.90      0.486453      0.465496 0.146667 0.080000       0.748889    0.487903
raw_timeline_composite     S2 target compact        fp_guard    220       prob          0.90      -0.08         0.90      0.542352      0.544437 0.168889 0.113333       0.706667    0.543994
raw_timeline_composite     S3 target     mid recent_fp_guard     60      logit          0.35      -0.18         0.90      0.534570      0.601279 0.168889 0.102222       0.728889    0.554470
raw_timeline_composite     S4    all     mid recent_fp_guard     60      logit          0.50       0.06         0.90      0.623463      0.581924 0.204444 0.153333       0.611111    0.624854
 raw_timeline_fp_guard     Q1 target compact         uniform     60      logit          0.00       0.06         1.00      0.590762      0.612457 0.160000 0.153333       0.502222    0.597260
 raw_timeline_fp_guard     Q2    all     mid        fp_guard     60      logit          0.70       0.06         1.15      0.675647      0.679731 0.297778 0.193333       0.666667    0.680362
 raw_timeline_fp_guard     Q3    all compact recent_fp_guard    220       prob          0.70      -0.08         0.90      0.654098      0.649363 0.242222 0.157778       0.684444    0.657245
 raw_timeline_fp_guard     S1    all     mid recent_fp_guard    220      logit          0.20      -0.30         0.90      0.486453      0.465496 0.146667 0.080000       0.748889    0.487903
 raw_timeline_fp_guard     S2 target compact        fp_guard    220       prob          0.90      -0.08         0.90      0.542352      0.544437 0.168889 0.113333       0.706667    0.543994
 raw_timeline_fp_guard     S3 target     mid recent_fp_guard     60      logit          0.35      -0.18         0.90      0.534570      0.601279 0.168889 0.102222       0.728889    0.554470
 raw_timeline_fp_guard     S4    all     mid recent_fp_guard     60      logit          0.50       0.06         0.90      0.623463      0.581924 0.204444 0.153333       0.611111    0.624854
     raw_timeline_full     Q1 target compact         uniform     60      logit          0.00       0.00         1.00      0.590425      0.618506 0.137778 0.166667       0.466667    0.598668
     raw_timeline_full     Q2 target     mid         uniform    120      logit          0.90       0.06         1.35      0.671579      0.662229 0.377778 0.028889       0.911111    0.690304
     raw_timeline_full     Q3 target compact recent_fp_guard    220       prob          0.50       0.06         0.90      0.648205      0.623302 0.300000 0.077778       0.822222    0.659460
     raw_timeline_full     S1    all     mid recent_fp_guard    220      logit          0.10      -0.30         0.90      0.485952      0.461001 0.155556 0.075556       0.762222    0.488321
     raw_timeline_full     S2 target compact        fp_guard    220      logit          0.70       0.00         0.90      0.540559      0.558816 0.191111 0.080000       0.762222    0.548762
     raw_timeline_full     S3 target     mid recent_fp_guard     60      logit          0.35       0.00         0.90      0.531039      0.601393 0.208889 0.055556       0.815556    0.556983
     raw_timeline_full     S4    all     mid recent_fp_guard     60      logit          0.35       0.06         0.90      0.623277      0.579654 0.217778 0.146667       0.631111    0.625415
     raw_timeline_last     Q1 target compact         uniform     60      logit          0.00       0.06         1.00      0.590762      0.612457 0.160000 0.153333       0.502222    0.597260
     raw_timeline_last     Q2 target compact         uniform     60      logit          0.00       0.06         0.90      0.697596      0.628082 0.326667 0.140000       0.748889    0.711895
     raw_timeline_last     Q3    all     mid        fp_guard     60      logit          0.35       0.06         0.90      0.651285      0.618076 0.322222 0.073333       0.848889    0.665178
     raw_timeline_last     S1    all     mid recent_fp_guard    120      logit          0.10       0.06         0.90      0.495185      0.447942 0.204444 0.042222       0.844444    0.504118
     raw_timeline_last     S2 target compact        fp_guard    220       prob          1.00      -0.30         1.00      0.555612      0.532524 0.142222 0.160000       0.633333    0.557700
     raw_timeline_last     S3 target compact         uniform     60      logit          0.00      -0.08         1.15      0.549904      0.597138 0.244444 0.046667       0.860000    0.575049
     raw_timeline_last     S4    all compact         uniform    120      logit          0.50       0.06         0.90      0.627227      0.573335 0.246667 0.131111       0.675556    0.632212

## Early Stopping

- `fit_diagnostics.csv` records `best_iteration_mean`, `stop_policy`, and `fallback_logic`.