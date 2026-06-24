# Raw Timeline Guarded Blend

## Candidate Scores

                  candidate  full_logloss  last_logloss  full_delta_vs_base  last_delta_vs_base  fold_std  tail3_worst
         manual_public_tail      0.588923      0.562776           -0.003725           -0.007831  0.017876     0.617031
global_raw_last_logit_a0p50      0.590916      0.565922           -0.001733           -0.004685  0.018366     0.621326
           base_public_best      0.592648      0.570607            0.000000            0.000000  0.016101     0.618609
       selector_public_tail      0.588111      0.571046           -0.004538            0.000439  0.014600     0.614466
     reference_public_tight      0.591908      0.574343           -0.000741            0.003736  0.015205     0.618728
global_raw_comp_logit_a0p65      0.584443      0.576208           -0.008205            0.005602  0.011759     0.606542
global_raw_full_logit_a0p80      0.583519      0.578422           -0.009129            0.007816  0.008809     0.599214
          selector_balanced      0.585871      0.583791           -0.006778            0.013184  0.010032     0.605045
       selector_fp_suppress      0.586567      0.584912           -0.006081            0.014305  0.010116     0.605970
      selector_private_full      0.586346      0.584936           -0.006302            0.014330  0.009959     0.605327

## Target Choices

                  candidate target   source  mode  alpha  full_logloss  last_logloss  fp_rate  fn_rate  pred_pos_rate  test_pos_rate  public_tail_score  private_full_score  balanced_score
           base_public_best     Q1     base  prob   0.00      0.584556      0.603240 0.135556 0.168889       0.462222          0.516           0.603240            0.584556        0.916338
           base_public_best     Q2     base  prob   0.00      0.692666      0.616951 0.337778 0.115556       0.784444          0.840           1.283617            1.025999        1.498656
           base_public_best     Q3     base  prob   0.00      0.665483      0.622419 0.351111 0.051111       0.900000          0.916           1.402419            1.055483        1.553813
           base_public_best     S1     base  prob   0.00      0.488844      0.430973 0.200000 0.044444       0.837778          0.800           0.555418            0.551067        0.812991
           base_public_best     S2     base  prob   0.00      0.562622      0.568421 0.195556 0.086667       0.760000          0.788           0.655532            0.606178        0.936231
           base_public_best     S3     base  prob   0.00      0.529442      0.577938 0.213333 0.031111       0.844444          0.916           1.015272            0.748109        1.153442
           base_public_best     S4     base  prob   0.00      0.624925      0.574304 0.242222 0.115556       0.686667          0.648           0.700970            0.688259        1.029459
       selector_public_tail     Q1     base  prob   0.00      0.584556      0.603240 0.135556 0.168889       0.462222          0.516           0.603240            0.584556        0.916338
       selector_public_tail     Q2     base  prob   0.00      0.692666      0.616951 0.337778 0.115556       0.784444          0.840           1.283617            1.025999        1.498656
       selector_public_tail     Q3 raw_comp logit   1.00      0.654098      0.649363 0.242222 0.157778       0.684444          0.736           0.912545            0.783806        1.194020
       selector_public_tail     S1 raw_comp logit   0.70      0.483080      0.440918 0.157778 0.071111       0.768889          0.732           0.518462            0.522473        0.779592
       selector_public_tail     S2 raw_last logit   0.80      0.549469      0.526986 0.146667 0.155556       0.642222          0.564           0.570380            0.564550        0.869122
       selector_public_tail     S3     base  prob   0.00      0.529442      0.577938 0.213333 0.031111       0.844444          0.916           1.015272            0.748109        1.153442
       selector_public_tail     S4 raw_comp logit   1.00      0.623463      0.581924 0.204444 0.153333       0.611111          0.544           0.648820            0.655804        0.989822
      selector_private_full     Q1     base  prob   0.00      0.584556      0.603240 0.135556 0.168889       0.462222          0.516           0.603240            0.584556        0.916338
      selector_private_full     Q2 raw_comp logit   1.00      0.675647      0.679731 0.297778 0.193333       0.666667          0.764           1.040522            0.860689        1.300470
      selector_private_full     Q3 raw_comp logit   1.00      0.654098      0.649363 0.242222 0.157778       0.684444          0.736           0.912545            0.783806        1.194020
      selector_private_full     S1 raw_comp logit   0.90      0.484168      0.451858 0.146667 0.075556       0.753333          0.720           0.519278            0.520300        0.779533
      selector_private_full     S2 raw_last logit   0.70      0.547923      0.527161 0.146667 0.137778       0.660000          0.572           0.571438            0.564144        0.868011
      selector_private_full     S3 raw_comp logit   1.00      0.534570      0.601279 0.168889 0.102222       0.728889          0.716           0.798199            0.632840        1.003939
      selector_private_full     S4 raw_comp logit   1.00      0.623463      0.581924 0.204444 0.153333       0.611111          0.544           0.648820            0.655804        0.989822
          selector_balanced     Q1     base  prob   0.00      0.584556      0.603240 0.135556 0.168889       0.462222          0.516           0.603240            0.584556        0.916338
          selector_balanced     Q2 raw_comp logit   1.00      0.675647      0.679731 0.297778 0.193333       0.666667          0.764           1.040522            0.860689        1.300470
          selector_balanced     Q3 raw_comp logit   1.00      0.654098      0.649363 0.242222 0.157778       0.684444          0.736           0.912545            0.783806        1.194020
          selector_balanced     S1 raw_comp logit   0.90      0.484168      0.451858 0.146667 0.075556       0.753333          0.720           0.519278            0.520300        0.779533
          selector_balanced     S2 raw_last logit   0.70      0.547923      0.527161 0.146667 0.137778       0.660000          0.572           0.571438            0.564144        0.868011
          selector_balanced     S3 raw_comp logit   0.75      0.531239      0.593260 0.186667 0.088889       0.760000          0.768           0.854418            0.661539        1.040991
          selector_balanced     S4 raw_comp logit   1.00      0.623463      0.581924 0.204444 0.153333       0.611111          0.544           0.648820            0.655804        0.989822
       selector_fp_suppress     Q1     base  prob   0.00      0.584556      0.603240 0.135556 0.168889       0.462222          0.516           0.603240            0.584556        0.916338
       selector_fp_suppress     Q2 raw_comp logit   1.00      0.675647      0.679731 0.297778 0.193333       0.666667          0.764           1.040522            0.860689        1.300470
       selector_fp_suppress     Q3 raw_comp logit   1.00      0.654098      0.649363 0.242222 0.157778       0.684444          0.736           0.912545            0.783806        1.194020
       selector_fp_suppress     S1 raw_comp logit   0.90      0.484168      0.451858 0.146667 0.075556       0.753333          0.720           0.519278            0.520300        0.779533
       selector_fp_suppress     S2 raw_last logit   0.80      0.549469      0.526986 0.146667 0.155556       0.642222          0.564           0.570380            0.564550        0.869122
       selector_fp_suppress     S3 raw_comp logit   1.00      0.534570      0.601279 0.168889 0.102222       0.728889          0.716           0.798199            0.632840        1.003939
       selector_fp_suppress     S4 raw_comp logit   1.00      0.623463      0.581924 0.204444 0.153333       0.611111          0.544           0.648820            0.655804        0.989822
         manual_public_tail     Q1     base  prob   0.00      0.584556      0.603240 0.135556 0.168889       0.462222          0.516           0.603240            0.584556        0.916338
         manual_public_tail     Q2     base  prob   0.00      0.692666      0.616951 0.337778 0.115556       0.784444          0.840           1.283617            1.025999        1.498656
         manual_public_tail     Q3 raw_full logit   0.50      0.653246      0.615549 0.324444 0.055556       0.868889          0.908           1.327701            1.007148        1.489869
         manual_public_tail     S1 raw_last logit   0.60      0.489805      0.428376 0.200000 0.042222       0.840000          0.800           0.558394            0.554493        0.816743
         manual_public_tail     S2 raw_last logit   0.75      0.548580      0.526886 0.148889 0.142222       0.657778          0.564           0.572437            0.565089        0.869536
         manual_public_tail     S3     base  prob   0.00      0.529442      0.577938 0.213333 0.031111       0.844444          0.916           1.015272            0.748109        1.153442
         manual_public_tail     S4 raw_last logit   0.55      0.624169      0.570490 0.248889 0.115556       0.693333          0.660           0.727495            0.693126        1.042572
global_raw_full_logit_a0p80     Q1 raw_full logit   0.80      0.588624      0.614955 0.137778 0.166667       0.466667          0.532           0.625385            0.592594        0.932802
global_raw_full_logit_a0p80     Q2 raw_full logit   0.80      0.672307      0.650201 0.366667 0.073333       0.855556          0.960           1.868898            1.131856        1.791715
global_raw_full_logit_a0p80     Q3 raw_full logit   0.80      0.649352      0.618444 0.317778 0.071111       0.846667          0.888           1.280746            0.977009        1.452409
global_raw_full_logit_a0p80     S1 raw_full logit   0.80      0.483252      0.441735 0.160000 0.066667       0.775556          0.732           0.524596            0.525469        0.783935
global_raw_full_logit_a0p80     S2 raw_full logit   0.80      0.539558      0.550780 0.193333 0.073333       0.771111          0.704           0.674866            0.596920        0.928411
global_raw_full_logit_a0p80     S3 raw_full logit   0.80      0.529077      0.595178 0.208889 0.048889       0.822222          0.836           0.995887            0.731301        1.138088
global_raw_full_logit_a0p80     S4 raw_full logit   0.80      0.622465      0.577663 0.220000 0.137778       0.642222          0.600           0.668480            0.667113        1.003465
global_raw_last_logit_a0p50     Q1 raw_last logit   0.50      0.586478      0.606894 0.146667 0.164444       0.477778          0.532           0.614408            0.587942        0.924009
global_raw_last_logit_a0p50     Q2 raw_last logit   0.50      0.694486      0.621899 0.328889 0.128889       0.762222          0.836           1.233438            1.000379        1.465733
global_raw_last_logit_a0p50     Q3 raw_last logit   0.50      0.656355      0.616198 0.333333 0.060000       0.873333          0.916           1.333975            1.014059        1.497471
global_raw_last_logit_a0p50     S1 raw_last logit   0.50      0.489518      0.428629 0.200000 0.042222       0.840000          0.800           0.557939            0.553874        0.815986
global_raw_last_logit_a0p50     S2 raw_last logit   0.50      0.547587      0.531780 0.153333 0.111111       0.693333          0.632           0.591552            0.573141        0.881040
global_raw_last_logit_a0p50     S3 raw_last logit   0.50      0.537952      0.585554 0.235556 0.022222       0.875556          0.916           1.114100            0.800171        1.229071
global_raw_last_logit_a0p50     S4 raw_last logit   0.50      0.624033      0.570499 0.251111 0.117778       0.693333          0.664           0.732478            0.692781        1.044264
global_raw_comp_logit_a0p65     Q1 raw_comp logit   0.65      0.587499      0.608358 0.148889 0.160000       0.484444          0.536           0.618476            0.589520        0.927142
global_raw_comp_logit_a0p65     Q2 raw_comp logit   0.65      0.677884      0.653320 0.333333 0.131111       0.764444          0.840           1.290251            0.998579        1.482052
global_raw_comp_logit_a0p65     Q3 raw_comp logit   0.65      0.650763      0.627611 0.304444 0.088889       0.815556          0.828           1.216092            0.941369        1.406951
global_raw_comp_logit_a0p65     S1 raw_comp logit   0.65      0.483176      0.439690 0.157778 0.066667       0.773333          0.732           0.520206            0.523907        0.781113
global_raw_comp_logit_a0p65     S2 raw_comp logit   0.65      0.540279      0.537188 0.180000 0.095556       0.735556          0.684           0.633037            0.583488        0.901884
global_raw_comp_logit_a0p65     S3 raw_comp logit   0.65      0.530281      0.590462 0.186667 0.077778       0.771111          0.792           0.874597            0.671930        1.054422
global_raw_comp_logit_a0p65     S4 raw_comp logit   0.65      0.621220      0.576830 0.215556 0.146667       0.628889          0.580           0.655950            0.659581        0.993520

## Notes

- `base_public_best` is the previously submitted public-aware family baseline.
- Direct raw timeline submissions are not selected; raw sources are used target-wise with FP/test-movement penalties.
- Review candidate_scores and candidate_choices before any submission decision.