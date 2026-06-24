# Raw Timeline Target Model

## Candidate Scores

             candidate  full_logloss  last_logloss  full_delta_vs_anchor  last_delta_vs_anchor  fold_std  tail3_mean  tail3_worst  fold0_logloss  fold1_logloss  fold2_logloss  fold3_logloss  fold4_logloss  Q1_fp_rate  Q1_fn_rate  Q1_pred_pos_rate  Q1_true_pos_rate  Q1_brier  Q2_fp_rate  Q2_fn_rate  Q2_pred_pos_rate  Q2_true_pos_rate  Q2_brier  Q3_fp_rate  Q3_fn_rate  Q3_pred_pos_rate  Q3_true_pos_rate  Q3_brier  S1_fp_rate  S1_fn_rate  S1_pred_pos_rate  S1_true_pos_rate  S1_brier  S2_fp_rate  S2_fn_rate  S2_pred_pos_rate  S2_true_pos_rate  S2_brier  S3_fp_rate  S3_fn_rate  S3_pred_pos_rate  S3_true_pos_rate  S3_brier  S4_fp_rate  S4_fn_rate  S4_pred_pos_rate  S4_true_pos_rate  S4_brier
     raw_timeline_full      0.594151      0.595263             -0.001679              0.001981  0.008995    0.598405     0.610447       0.592753       0.583461       0.610447       0.589503       0.595263    0.137778    0.166667          0.466667          0.495556  0.202468    0.346667    0.113333          0.795556          0.562222  0.245760    0.346667    0.055556          0.891111               0.6  0.235201    0.186667    0.060000          0.808889          0.682222  0.160563    0.226667    0.048889          0.828889          0.651111  0.188945    0.253333    0.013333          0.902222          0.662222  0.180786    0.228889    0.131111          0.657778              0.56  0.219842
raw_timeline_composite      0.594689      0.595842             -0.001140              0.002560  0.009589    0.599046     0.612573       0.592248       0.584834       0.612573       0.588723       0.595842    0.137778    0.166667          0.466667          0.495556  0.202468    0.313333    0.160000          0.715556          0.562222  0.246953    0.346667    0.055556          0.891111               0.6  0.235201    0.186667    0.060000          0.808889          0.682222  0.160563    0.220000    0.060000          0.811111          0.651111  0.188890    0.244444    0.046667          0.860000          0.662222  0.181037    0.228889    0.131111          0.657778              0.56  0.219842
 raw_timeline_fp_guard      0.594689      0.595842             -0.001140              0.002560  0.009589    0.599046     0.612573       0.592248       0.584834       0.612573       0.588723       0.595842    0.137778    0.166667          0.466667          0.495556  0.202468    0.313333    0.160000          0.715556          0.562222  0.246953    0.346667    0.055556          0.891111               0.6  0.235201    0.186667    0.060000          0.808889          0.682222  0.160563    0.220000    0.060000          0.811111          0.651111  0.188890    0.244444    0.046667          0.860000          0.662222  0.181037    0.228889    0.131111          0.657778              0.56  0.219842
     raw_timeline_last      0.596019      0.591167              0.000189             -0.002115  0.013261    0.601105     0.621724       0.594015       0.583379       0.621724       0.590425       0.591167    0.137778    0.166667          0.466667          0.495556  0.202468    0.315556    0.148889          0.728889          0.562222  0.249609    0.346667    0.055556          0.891111               0.6  0.235201    0.202222    0.044444          0.840000          0.682222  0.162266    0.220000    0.060000          0.811111          0.651111  0.188890    0.244444    0.046667          0.860000          0.662222  0.181037    0.228889    0.131111          0.657778              0.56  0.219842

## FP/FN Summary

             candidate target  fp_rate  fn_rate  pred_pos_rate  true_pos_rate    brier
raw_timeline_composite     Q1 0.137778 0.166667       0.466667       0.495556 0.202468
raw_timeline_composite     Q2 0.313333 0.160000       0.715556       0.562222 0.246953
raw_timeline_composite     Q3 0.346667 0.055556       0.891111       0.600000 0.235201
raw_timeline_composite     S1 0.186667 0.060000       0.808889       0.682222 0.160563
raw_timeline_composite     S2 0.220000 0.060000       0.811111       0.651111 0.188890
raw_timeline_composite     S3 0.244444 0.046667       0.860000       0.662222 0.181037
raw_timeline_composite     S4 0.228889 0.131111       0.657778       0.560000 0.219842
 raw_timeline_fp_guard     Q1 0.137778 0.166667       0.466667       0.495556 0.202468
 raw_timeline_fp_guard     Q2 0.313333 0.160000       0.715556       0.562222 0.246953
 raw_timeline_fp_guard     Q3 0.346667 0.055556       0.891111       0.600000 0.235201
 raw_timeline_fp_guard     S1 0.186667 0.060000       0.808889       0.682222 0.160563
 raw_timeline_fp_guard     S2 0.220000 0.060000       0.811111       0.651111 0.188890
 raw_timeline_fp_guard     S3 0.244444 0.046667       0.860000       0.662222 0.181037
 raw_timeline_fp_guard     S4 0.228889 0.131111       0.657778       0.560000 0.219842
     raw_timeline_full     Q1 0.137778 0.166667       0.466667       0.495556 0.202468
     raw_timeline_full     Q2 0.346667 0.113333       0.795556       0.562222 0.245760
     raw_timeline_full     Q3 0.346667 0.055556       0.891111       0.600000 0.235201
     raw_timeline_full     S1 0.186667 0.060000       0.808889       0.682222 0.160563
     raw_timeline_full     S2 0.226667 0.048889       0.828889       0.651111 0.188945
     raw_timeline_full     S3 0.253333 0.013333       0.902222       0.662222 0.180786
     raw_timeline_full     S4 0.228889 0.131111       0.657778       0.560000 0.219842
     raw_timeline_last     Q1 0.137778 0.166667       0.466667       0.495556 0.202468
     raw_timeline_last     Q2 0.315556 0.148889       0.728889       0.562222 0.249609
     raw_timeline_last     Q3 0.346667 0.055556       0.891111       0.600000 0.235201
     raw_timeline_last     S1 0.202222 0.044444       0.840000       0.682222 0.162266
     raw_timeline_last     S2 0.220000 0.060000       0.811111       0.651111 0.188890
     raw_timeline_last     S3 0.244444 0.046667       0.860000       0.662222 0.181037
     raw_timeline_last     S4 0.228889 0.131111       0.657778       0.560000 0.219842

## Target Choices

             candidate target  scope profile weight_profile  top_k blend_mode  model_weight  intercept  temperature  full_logloss  last_logloss  fp_rate  fn_rate  pred_pos_rate  rank_score
raw_timeline_composite     Q1 target compact        uniform     20      logit           0.0       0.00          1.0      0.590425      0.618506 0.137778 0.166667       0.466667    0.598668
raw_timeline_composite     Q2 target compact        uniform     20       prob           0.5      -0.12          1.0      0.686789      0.667438 0.313333 0.160000       0.715556    0.694477
raw_timeline_composite     Q3 target compact        uniform     20      logit           0.0       0.00          1.0      0.662681      0.634692 0.346667 0.055556       0.891111    0.678925
raw_timeline_composite     S1 target compact        uniform     20      logit           0.0      -0.12          1.0      0.489137      0.452916 0.186667 0.060000       0.808889    0.494490
raw_timeline_composite     S2 target compact        uniform     20      logit           0.0      -0.12          1.0      0.560209      0.611058 0.220000 0.060000       0.811111    0.581111
raw_timeline_composite     S3 target compact        uniform     20       prob           0.0      -0.12          1.0      0.542788      0.597632 0.244444 0.046667       0.860000    0.567291
raw_timeline_composite     S4 target compact        uniform     20      logit           0.0       0.00          1.0      0.630799      0.588654 0.228889 0.131111       0.657778    0.634444
 raw_timeline_fp_guard     Q1 target compact        uniform     20      logit           0.0       0.00          1.0      0.590425      0.618506 0.137778 0.166667       0.466667    0.598668
 raw_timeline_fp_guard     Q2 target compact        uniform     20       prob           0.5      -0.12          1.0      0.686789      0.667438 0.313333 0.160000       0.715556    0.694477
 raw_timeline_fp_guard     Q3 target compact        uniform     20      logit           0.0       0.00          1.0      0.662681      0.634692 0.346667 0.055556       0.891111    0.678925
 raw_timeline_fp_guard     S1 target compact        uniform     20      logit           0.0      -0.12          1.0      0.489137      0.452916 0.186667 0.060000       0.808889    0.494490
 raw_timeline_fp_guard     S2 target compact        uniform     20      logit           0.0      -0.12          1.0      0.560209      0.611058 0.220000 0.060000       0.811111    0.581111
 raw_timeline_fp_guard     S3 target compact        uniform     20      logit           0.0      -0.12          1.0      0.542788      0.597632 0.244444 0.046667       0.860000    0.567291
 raw_timeline_fp_guard     S4 target compact        uniform     20      logit           0.0       0.00          1.0      0.630799      0.588654 0.228889 0.131111       0.657778    0.634444
     raw_timeline_full     Q1 target compact        uniform     20      logit           0.0       0.00          1.0      0.590425      0.618506 0.137778 0.166667       0.466667    0.598668
     raw_timeline_full     Q2 target compact        uniform     20       prob           0.5       0.00          1.0      0.684344      0.648581 0.346667 0.113333       0.795556    0.696280
     raw_timeline_full     Q3 target compact        uniform     20      logit           0.0       0.00          1.0      0.662681      0.634692 0.346667 0.055556       0.891111    0.678925
     raw_timeline_full     S1 target compact        uniform     20      logit           0.0      -0.12          1.0      0.489137      0.452916 0.186667 0.060000       0.808889    0.494490
     raw_timeline_full     S2 target compact        uniform     20      logit           0.0       0.00          1.0      0.560120      0.622815 0.226667 0.048889       0.828889    0.585527
     raw_timeline_full     S3 target compact        uniform     20      logit           0.0       0.00          1.0      0.541551      0.600679 0.253333 0.013333       0.902222    0.569183
     raw_timeline_full     S4 target compact        uniform     20      logit           0.0       0.00          1.0      0.630799      0.588654 0.228889 0.131111       0.657778    0.634444
     raw_timeline_last     Q1 target compact        uniform     20      logit           0.0       0.00          1.0      0.590425      0.618506 0.137778 0.166667       0.466667    0.598668
     raw_timeline_last     Q2 target compact        uniform     20      logit           0.0       0.00          1.0      0.692880      0.638084 0.315556 0.148889       0.728889    0.703825
     raw_timeline_last     Q3 target compact        uniform     20      logit           0.0       0.00          1.0      0.662681      0.634692 0.346667 0.055556       0.891111    0.678925
     raw_timeline_last     S1 target compact        uniform     20      logit           0.0       0.00          1.0      0.492352      0.449544 0.202222 0.044444       0.840000    0.499598
     raw_timeline_last     S2 target compact        uniform     20      logit           0.0      -0.12          1.0      0.560209      0.611058 0.220000 0.060000       0.811111    0.581111
     raw_timeline_last     S3 target compact        uniform     20       prob           0.0      -0.12          1.0      0.542788      0.597632 0.244444 0.046667       0.860000    0.567291
     raw_timeline_last     S4 target compact        uniform     20      logit           0.0       0.00          1.0      0.630799      0.588654 0.228889 0.131111       0.657778    0.634444

## Early Stopping

- `fit_diagnostics.csv` records `best_iteration_mean`, `stop_policy`, and `fallback_logic`.