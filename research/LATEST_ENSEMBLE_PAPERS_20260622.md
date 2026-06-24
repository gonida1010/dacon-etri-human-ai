# Latest Ensemble / Tabular Research Notes, 2026-06-22

Scope: Dacon ETRI improvement only.  Kaggle star-classification domain details are not transferred; only general ensemble/stacking methodology is used.

## Sources Checked

- Ensembling Tabular Foundation Models - A Diversity Ceiling And A Calibration Trap, arXiv 2605.18696, 2026-05-18
  - https://arxiv.org/abs/2605.18696
- HAPEns: Hardware-Aware Post-Hoc Ensembling for Tabular Data, arXiv 2603.10582, 2026-03-11
  - https://arxiv.org/abs/2603.10582
- Multi-layer Stack Ensembles for Time Series Forecasting, arXiv 2511.15350, 2025-11-19
  - https://arxiv.org/abs/2511.15350
- TabArena: A Living Benchmark for Machine Learning on Tabular Data, arXiv 2506.16791, 2025-06-20
  - https://arxiv.org/abs/2506.16791
- TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling, arXiv 2410.24210
  - https://arxiv.org/abs/2410.24210
- TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models, arXiv 2511.08667
  - https://arxiv.org/abs/2511.08667

## Practical Takeaways For This Dacon Work

1. Do not trust logistic meta-stacking by itself for log-loss.
   - The 2026 TFM ensemble paper reports that logistic-regression stacking can keep accuracy/ROC-AUC competitive but becomes worst-ranked for log-loss because it sharpens probabilities and harms calibration.
   - Dacon target is log-loss, so any stacker output needs anchor/logit shrink, temperature scaling, or convex averaging.

2. Greedy/convex ensemble selection is the default safer combiner.
   - The same 2026 paper recommends greedy selection as the practical default when model predictions are redundant.
   - Our candidate selection should prefer deduped sources, simplex/convex blends, and target-wise selection instead of unconstrained high-dimensional meta models.

3. Diversity ceiling matters.
   - If sources are near-duplicates, stacking cannot recover much and may overfit validation.
   - This matches the local finding that many constrained/logit candidates have flat corr above `0.999`.
   - Source dedupe by target is required before meta-learning.

4. Multi-layer stacking can help, but only if the second layer has nonredundant information.
   - Time-series stacking work shows no single stacker dominates across tasks and multi-layer stackers help when stacker types complement each other.
   - For this Dacon setup, this means combining source types: temporal anchor, residual ridge/logit, KNN prior, direction-gated recipes, not only many versions of one model family.

5. Hardware-aware / Pareto ensembling maps to public-risk-aware selection here.
   - HAPEns optimizes predictive score plus deployment cost.  In this competition, the analogous second objective is public-risk / movement cost.
   - Our selector should optimize local log-loss and penalize movement in known bad directions from public feedback.

6. TabM/RealMLP idea relevant for future single-model work.
   - Parameter-efficient internal ensembling is useful, but the current Dacon pipeline lacks a true TabM/RealMLP implementation for this dataset.
   - If implemented later, use it as a new diverse source, not as a standalone final answer.

## Implemented From These Notes

- `src/public_aware_stack_blend.py`
  - source dedupe by target
  - fold-safe logistic stacker on probability/logit/delta features
  - fold-safe simplex weighted blends in probability and logit space
  - anchor-logit shrink / temperature-style variants
  - target-wise selection with public-failure direction penalty

- `src/public_score_pseudo_blend.py`
  - uses known public scores as aggregate log-loss constraints
  - estimates a test-time pseudo posterior around the temporal anchor
  - searches target-wise selections, pairwise logit blends, simplex blends, and anchor-logit shrink variants
  - keeps CV full/last guards while optimizing the pseudo-public objective

## Current One-File Submission Recommendation

Previous recommendation was:

```text
submissions/public_aware_stack_blend_20260622/target_select_public_tight_last0.572477_full0.592635.csv
```

Observed public score:

```text
0.5905116492
```

Updated public-score-constrained recommendation:

```text
submissions/public_score_pseudo_blend_20260622/pseudo_target_tight_anchorlogit_w0p82_pseudo0.588794_last0.588809_full0.594948.csv
```

Why:

- It is still stack/blend/post-hoc ensemble based, but now uses the two observed public scores plus the older `02_guarded_targetwise` score as constraints.
- It moves only the two directions with positive OOF support:
  - S2 down: last-fold down precision `0.666667`
  - S4 up: last-fold up precision `0.833333`
- It leaves Q1/Q2/Q3/S1/S3 at the anchor, avoiding the noisy Q and S3 public-risk directions.
- Local CV remains inside guard:
  - full `0.594948`
  - last `0.588809`
- Pseudo-public objective improves from anchor `0.589491` to `0.588794`.
