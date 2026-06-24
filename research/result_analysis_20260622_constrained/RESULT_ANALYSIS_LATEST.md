# Result Analysis Report, 2026-06-22

Generated: 2026-06-22 16:33:41

Scope: completed OOF/result artifacts only. No model training is performed by this report.

## Executive Findings

- Best loaded local candidate is `blend_ridge_logit_newton / ridge_knn_blend_full`: full `0.587732`, last `0.588717`.
- No-TE Ridge full reference: full `0.587810`, last `0.590089`.
- Old blend full reference: full `0.592591`, last `0.590445`.
- `bins_te` did not improve the Ridge residual family. It improves some last-block targets, but worsens full stability.
- The no-TE logit blend remains the best full-logloss base; constrained target portfolios trade tiny full cost for better last-block OOF.
- OOF source-correlation output shows many exact anchor clones from earlier source banks, so future blend search must deduplicate sources before greedy/stacking.

## Key Candidate Comparison

```
                     run                   candidate full_logloss last_logloss full_delta_vs_anchor last_delta_vs_anchor rank_score fold_std
 constrained_logit_blend                        full     0.587732     0.588717            -0.008098            -0.004565   0.593041 0.015432
blend_ridge_logit_newton        ridge_knn_blend_full     0.587732     0.588717            -0.008098            -0.004565   0.593041 0.015432
 constrained_logit_blend            last_guard_0p008     0.587743     0.588383            -0.008087            -0.004899   0.592730 0.015408
 constrained_logit_blend            last_guard_0p006     0.587774     0.588102            -0.008056            -0.005180   0.592470 0.015387
      ridge_logit_newton         ridge_residual_full     0.587810     0.590089            -0.008020            -0.003193   0.594322 0.015406
 constrained_logit_blend positive_last_penalty_a0p25     0.587837     0.587745            -0.007993            -0.005537   0.592116 0.015354
 constrained_logit_blend          tradeoff_cap_a0p15     0.587898     0.586630            -0.007932            -0.006652   0.591064 0.015405
 constrained_logit_blend  positive_last_penalty_a0p5     0.587911     0.587499            -0.007919            -0.005783   0.591869 0.015321
 constrained_logit_blend          tradeoff_cap_a0p25     0.588271     0.584942            -0.007559            -0.008341   0.589465 0.015421
 constrained_logit_blend          tradeoff_cap_a0p35     0.588345     0.584695            -0.007485            -0.008587   0.589218 0.015409
 constrained_logit_blend           tradeoff_cap_a0p5     0.588696     0.583872            -0.007134            -0.009410   0.588454 0.015569
         blend_ridge_knn        ridge_knn_blend_full     0.592591     0.590445            -0.003239            -0.002837   0.595263 0.015744
     blend_logit_te_full        ridge_knn_blend_full     0.593723     0.591800            -0.002107            -0.001483   0.595653 0.013422
     ridge_logit_te_full         ridge_residual_full     0.593726     0.591578            -0.002104            -0.001704   0.595475 0.013512
blend_ridge_logit_newton   ridge_knn_blend_composite     0.594314     0.581319            -0.001516            -0.011964   0.586236 0.016412
      ridge_logit_newton    ridge_residual_composite     0.594519     0.581374            -0.001311            -0.011909   0.586294 0.016529
         blend_ridge_knn   ridge_knn_blend_composite     0.595536     0.584969            -0.000294            -0.008314   0.589775 0.014847
                  anchor                      anchor     0.595830     0.593282             0.000000             0.000000                    
     blend_logit_te_full   ridge_knn_blend_composite     0.596128     0.588013             0.000299            -0.005269   0.592618 0.014256
      ridge_logit_newton         ridge_residual_last     0.596299     0.580663             0.000469            -0.012619   0.585777 0.016987
     ridge_logit_te_full    ridge_residual_composite     0.597277     0.587759             0.001447            -0.005523   0.593442 0.014529
```

## Top by Full Logloss

```
                     run                   candidate full_logloss last_logloss full_delta_vs_anchor last_delta_vs_anchor
blend_ridge_logit_newton        ridge_knn_blend_full     0.587732     0.588717            -0.008098            -0.004565
 constrained_logit_blend                        full     0.587732     0.588717            -0.008098            -0.004565
 constrained_logit_blend            last_guard_0p008     0.587743     0.588383            -0.008087            -0.004899
 constrained_logit_blend            last_guard_0p006     0.587774     0.588102            -0.008056            -0.005180
      ridge_logit_newton         ridge_residual_full     0.587810     0.590089            -0.008020            -0.003193
 constrained_logit_blend positive_last_penalty_a0p25     0.587837     0.587745            -0.007993            -0.005537
 constrained_logit_blend          tradeoff_cap_a0p15     0.587898     0.586630            -0.007932            -0.006652
 constrained_logit_blend  positive_last_penalty_a0p5     0.587911     0.587499            -0.007919            -0.005783
 constrained_logit_blend          tradeoff_cap_a0p25     0.588271     0.584942            -0.007559            -0.008341
 constrained_logit_blend          tradeoff_cap_a0p35     0.588345     0.584695            -0.007485            -0.008587
 constrained_logit_blend            last_guard_0p004     0.588346     0.587181            -0.007484            -0.006101
 constrained_logit_blend            last_guard_0p002     0.588411     0.586975            -0.007418            -0.006307
```

## Top by Last Logloss

```
                     run                             candidate full_logloss last_logloss full_delta_vs_anchor last_delta_vs_anchor
 constrained_logit_blend                                  last     0.595795     0.580503            -0.000035            -0.012779
      ridge_logit_newton                   ridge_residual_last     0.596299     0.580663             0.000469            -0.012619
 constrained_logit_blend                             composite     0.594399     0.581028            -0.001431            -0.012254
blend_ridge_logit_newton                  ridge_knn_blend_last     0.594384     0.581261            -0.001446            -0.012021
blend_ridge_logit_newton             ridge_knn_blend_composite     0.594314     0.581319            -0.001516            -0.011964
      ridge_logit_newton              ridge_residual_composite     0.594519     0.581374            -0.001311            -0.011909
      anchor_bank_sparse              targetwise_sparse_greedy     0.599491     0.582418             0.003661            -0.010864
        kaggle_last_mile                     oof_sparse_greedy     0.599491     0.582418             0.003661            -0.010864
        kaggle_last_mile rankpatch_sparse_greedy_k8_f0p1_d0p35     0.599334     0.582890             0.003505            -0.010392
 constrained_logit_blend                     tradeoff_cap_a0p5     0.588696     0.583872            -0.007134            -0.009410
        hist_gb_residual                 hist_gb_residual_last     0.611207     0.584409             0.015377            -0.008873
 constrained_logit_blend                    tradeoff_cap_a0p35     0.588345     0.584695            -0.007485            -0.008587
```

## TE Feature Bank Read

- Smoke `bins_te`: `ridge_residual_full` full `0.594343`, last `0.591565`; worse than no-TE logit Ridge full candidate.
- Full `bins_te`: `ridge_residual_full` full `0.593726`, last `0.591578`; still worse than no-TE logit Ridge full candidate.
- Full `bins_te` composite: full `0.597277`, last `0.587759`; last improves but full is worse than anchor.
- Interpretation: current fold-safe TE/bin bank is too noisy for this 450-row dataset. Keep the implementation, but do not use it globally in the next submit path.

## Blend Read

- No-TE logit blend full candidate: full `0.587732`, last `0.588717`.
- Constrained `last_guard_0p008`: full `0.587743`, last `0.588383`; almost no full cost versus full base, with better last.
- Constrained `positive_last_penalty_a0p25`: full `0.587837`, last `0.587745`; stronger last gain with still small full cost.
- Constrained `tradeoff_cap_a0p15`: full `0.587898`, last `0.586630`; best balanced attack candidate before full cost starts rising.
- Old blend full candidate: full `0.592591`, last `0.590445`.
- New TE blend full candidate: full `0.593723`, last `0.591800`.
- New TE blend composite: full `0.596128`, last `0.588013`.
- Interpretation: no-TE logit residual is useful as a blend source; TE source is not adding useful diversity.

## Correlation Summary

```
target  source_count  anchor_clone_count  pairs_corr_ge_0p999  pairs_corr_lt_0p98                                                   best_full_source best_full_logloss                                                   best_last_source best_last_logloss
    Q1           297                 218                24656               15921  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.587968                        constrained_target_blend_logit_newton__last          0.596240
    Q2           297                 217                24784               15620                                                  model_lgbm_anchor          0.675684                                                   anchor_temp_1p15          0.634842
    Q3           297                 232                27981               14203                                                  model_lgbm_anchor          0.651124                                                       seq_Q3_0p5_1          0.624186
    S1           297                 218                24636               13999 residual_submission_blend_ridge_logit_newton__ridge_knn_blend_full          0.489767 residual_submission_blend_ridge_logit_newton__ridge_knn_blend_last          0.443303
    S2           297                 217                24427               18682  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.552330                        constrained_target_blend_logit_newton__last          0.584161
    S3           297                 218                25282               17404  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.507728                        constrained_target_blend_logit_newton__last          0.593313
    S4           297                 217                24426               19027  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.622643                                                    seq_S4_1p5_0p45          0.551026
```

## Figures

- `candidate_frontier.png`: full/last scatter over all loaded candidates.
- `submission_frontier.png`: clean submit-candidate-only full/last scatter.
- `constrained_frontier.png`: full-cost/last-gain tradeoff among constrained portfolios.
- `key_candidate_deltas.png`: full/last delta against anchor for major candidates.
- `key_fold_curves.png`: fold stability curves.
- `target_last_delta_heatmap.png`: target-wise last-block gain/loss.
- `target_full_delta_heatmap.png`: target-wise full-period gain/loss.

## Next Research Step

1. Primary submit-candidate path: `constrained_logit_blend / last_guard_0p008` when public-risk control matters, or `constrained_logit_blend / full` when pure full OOF is preferred.
2. Attack candidate: `constrained_logit_blend / tradeoff_cap_a0p15`; it buys much better last OOF for a still small full cost.
3. Keep `ridge_logit_newton / ridge_residual_full` as the no-blend fallback candidate.
4. Freeze `bins_te` as an analysis-only branch for now; do not use it in the next submit candidate.
5. Next modeling work should be source deduplication plus target-specific constraints for Q1/S2/S3, not more global TE.
