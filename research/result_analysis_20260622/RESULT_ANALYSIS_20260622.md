# Result Analysis Report, 2026-06-22

Generated: 2026-06-22 15:27:44

Scope: completed OOF/result artifacts only. No model training is performed by this report.

## Executive Findings

- Best loaded local candidate is `blend_ridge_logit_newton / ridge_knn_blend_full`: full `0.587732`, last `0.588717`.
- No-TE Ridge full reference: full `0.587810`, last `0.590089`.
- Old blend full reference: full `0.592591`, last `0.590445`.
- `bins_te` did not improve the Ridge residual family. It improves some last-block targets, but worsens full stability.
- The no-TE logit blend is the only newly generated result that improves both full and last versus the no-TE Ridge full candidate.
- OOF source-correlation output shows many exact anchor clones from earlier source banks, so future blend search must deduplicate sources before greedy/stacking.

## Key Candidate Comparison

```
                     run                 candidate full_logloss last_logloss full_delta_vs_anchor last_delta_vs_anchor rank_score fold_std
blend_ridge_logit_newton      ridge_knn_blend_full     0.587732     0.588717            -0.008098            -0.004565   0.593041 0.015432
      ridge_logit_newton       ridge_residual_full     0.587810     0.590089            -0.008020            -0.003193   0.594322 0.015406
         blend_ridge_knn      ridge_knn_blend_full     0.592591     0.590445            -0.003239            -0.002837   0.595263 0.015744
     blend_logit_te_full      ridge_knn_blend_full     0.593723     0.591800            -0.002107            -0.001483   0.595653 0.013422
     ridge_logit_te_full       ridge_residual_full     0.593726     0.591578            -0.002104            -0.001704   0.595475 0.013512
blend_ridge_logit_newton ridge_knn_blend_composite     0.594314     0.581319            -0.001516            -0.011964   0.586236 0.016412
      ridge_logit_newton  ridge_residual_composite     0.594519     0.581374            -0.001311            -0.011909   0.586294 0.016529
         blend_ridge_knn ridge_knn_blend_composite     0.595536     0.584969            -0.000294            -0.008314   0.589775 0.014847
                  anchor                    anchor     0.595830     0.593282             0.000000             0.000000                    
     blend_logit_te_full ridge_knn_blend_composite     0.596128     0.588013             0.000299            -0.005269   0.592618 0.014256
      ridge_logit_newton       ridge_residual_last     0.596299     0.580663             0.000469            -0.012619   0.585777 0.016987
     ridge_logit_te_full  ridge_residual_composite     0.597277     0.587759             0.001447            -0.005523   0.593442 0.014529
```

## Top by Full Logloss

```
                     run                 candidate full_logloss last_logloss full_delta_vs_anchor last_delta_vs_anchor
blend_ridge_logit_newton      ridge_knn_blend_full     0.587732     0.588717            -0.008098            -0.004565
      ridge_logit_newton       ridge_residual_full     0.587810     0.590089            -0.008020            -0.003193
         blend_ridge_knn      ridge_knn_blend_full     0.592591     0.590445            -0.003239            -0.002837
              ridge_prob       ridge_residual_full     0.592740     0.592791            -0.003090            -0.000491
        kaggle_last_mile           meta_extratrees     0.593542     0.605017            -0.002288             0.011735
     blend_logit_te_full      ridge_knn_blend_full     0.593723     0.591800            -0.002107            -0.001483
     ridge_logit_te_full       ridge_residual_full     0.593726     0.591578            -0.002104            -0.001704
blend_ridge_logit_newton ridge_knn_blend_composite     0.594314     0.581319            -0.001516            -0.011964
    blend_logit_te_smoke      ridge_knn_blend_full     0.594334     0.591403            -0.001496            -0.001879
    ridge_logit_te_smoke       ridge_residual_full     0.594343     0.591565            -0.001487            -0.001717
blend_ridge_logit_newton      ridge_knn_blend_last     0.594384     0.581261            -0.001446            -0.012021
      ridge_logit_newton  ridge_residual_composite     0.594519     0.581374            -0.001311            -0.011909
```

## Top by Last Logloss

```
                     run                             candidate full_logloss last_logloss full_delta_vs_anchor last_delta_vs_anchor
      ridge_logit_newton                   ridge_residual_last     0.596299     0.580663             0.000469            -0.012619
blend_ridge_logit_newton                  ridge_knn_blend_last     0.594384     0.581261            -0.001446            -0.012021
blend_ridge_logit_newton             ridge_knn_blend_composite     0.594314     0.581319            -0.001516            -0.011964
      ridge_logit_newton              ridge_residual_composite     0.594519     0.581374            -0.001311            -0.011909
      anchor_bank_sparse              targetwise_sparse_greedy     0.599491     0.582418             0.003661            -0.010864
        kaggle_last_mile                     oof_sparse_greedy     0.599491     0.582418             0.003661            -0.010864
        kaggle_last_mile rankpatch_sparse_greedy_k8_f0p1_d0p35     0.599334     0.582890             0.003505            -0.010392
        hist_gb_residual                 hist_gb_residual_last     0.611207     0.584409             0.015377            -0.008873
              ridge_prob                   ridge_residual_last     0.616519     0.584758             0.020690            -0.008525
     ridge_logit_te_full                   ridge_residual_last     0.607673     0.584839             0.011843            -0.008444
         blend_ridge_knn             ridge_knn_blend_composite     0.595536     0.584969            -0.000294            -0.008314
         blend_ridge_knn                  ridge_knn_blend_last     0.595536     0.584969            -0.000294            -0.008314
```

## TE Feature Bank Read

- Smoke `bins_te`: `ridge_residual_full` full `0.594343`, last `0.591565`; worse than no-TE logit Ridge full candidate.
- Full `bins_te`: `ridge_residual_full` full `0.593726`, last `0.591578`; still worse than no-TE logit Ridge full candidate.
- Full `bins_te` composite: full `0.597277`, last `0.587759`; last improves but full is worse than anchor.
- Interpretation: current fold-safe TE/bin bank is too noisy for this 450-row dataset. Keep the implementation, but do not use it globally in the next submit path.

## Blend Read

- No-TE logit blend full candidate: full `0.587732`, last `0.588717`.
- Old blend full candidate: full `0.592591`, last `0.590445`.
- New TE blend full candidate: full `0.593723`, last `0.591800`.
- New TE blend composite: full `0.596128`, last `0.588013`.
- Interpretation: no-TE logit residual is useful as a blend source; TE source is not adding useful diversity.

## Correlation Summary

```
target  source_count  anchor_clone_count  pairs_corr_ge_0p999  pairs_corr_lt_0p98                                                   best_full_source best_full_logloss                                                       best_last_source best_last_logloss
    Q1           281                 218                24566               13957  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.587968      residual_single_model_opt_ridge_logit_newton__ridge_residual_last          0.596395
    Q2           281                 217                24320               14253                                                  model_lgbm_anchor          0.675684                                                       anchor_temp_1p15          0.634842
    Q3           281                 219                24764               13116                                                  model_lgbm_anchor          0.651124                                                           seq_Q3_0p5_1          0.624186
    S1           281                 218                24546               13151 residual_submission_blend_ridge_logit_newton__ridge_knn_blend_full          0.489767 residual_single_model_opt_ridge_logit_newton__ridge_residual_composite          0.443303
    S2           281                 217                24336               14218  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.552330 residual_single_model_opt_ridge_logit_newton__ridge_residual_composite          0.584161
    S3           281                 218                25212               12916  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.507728     residual_submission_blend_ridge_logit_newton__ridge_knn_blend_last          0.593313
    S4           281                 217                24351               14627  residual_single_model_opt_ridge_logit_newton__ridge_residual_full          0.622643                                                        seq_S4_1p5_0p45          0.551026
```

## Figures

- `candidate_frontier.png`: full/last scatter over all loaded candidates.
- `submission_frontier.png`: clean submit-candidate-only full/last scatter.
- `key_candidate_deltas.png`: full/last delta against anchor for major candidates.
- `key_fold_curves.png`: fold stability curves.
- `target_last_delta_heatmap.png`: target-wise last-block gain/loss.
- `target_full_delta_heatmap.png`: target-wise full-period gain/loss.

## Next Research Step

1. Treat `blend_ridge_logit_newton / ridge_knn_blend_full` as the next primary candidate.
2. Keep `ridge_logit_newton / ridge_residual_full` as the no-blend fallback candidate.
3. Freeze `bins_te` as an analysis-only branch for now; do not use it in the next submit candidate.
4. Add deduplicated source selection before broader stacking/blending: remove sources with identical predictions or corr >= 0.999 against anchor/another better source.
5. Next modeling work should be target-specific constraints for Q1 and S2, not more global TE.
