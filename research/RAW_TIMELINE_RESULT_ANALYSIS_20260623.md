# Raw Timeline Result Analysis 2026-06-23

## Files Reviewed

- `research/raw_timeline_target_model_20260623_full/candidate_scores.csv`
- `research/raw_timeline_target_model_20260623_full/fp_fn_summary.csv`
- `research/raw_timeline_target_model_20260623_full/target_choices_all.csv`
- `research/raw_timeline_target_model_20260623_full/fit_diagnostics.csv`
- `submissions/raw_timeline_target_model_20260623_full/*.csv`
- Comparison source: `research/guarded_lgbm_integration_20260623_v2/public_aware_stack_blend_20260622_target_select_public_balanced__balanced_Q2_only_5_oof.csv`

## Main Scores

| candidate | full | last | fold_std | tail3_worst | note |
|---|---:|---:|---:|---:|---|
| public_best submitted family | 0.592648 | 0.570607 | 0.016101 | 0.618609 | public score 0.590284128 before q2 probe |
| q2_big_probe submitted | local not regenerated here | local near public_best | - | - | public score 0.5902817192, only tiny gain |
| raw_timeline_full | 0.584434 | 0.586415 | 0.007583 | 0.597139 | best full/stability, weak public-tail fit |
| raw_timeline_composite | 0.586764 | 0.590669 | 0.010546 | 0.605283 | safer positivity than raw_full but last worse |
| raw_timeline_last | 0.595367 | 0.572793 | 0.018730 | 0.628044 | best raw last, unstable and weak full |

## What Changed

- Raw timeline features are a real new signal source. `raw_timeline_full` is the best full-CV candidate produced so far: `0.584434`, beating the previous full-CV best around `0.5877`.
- This is not automatically a public submission candidate. The public-oriented family still has much better last-CV (`0.570607`) than `raw_timeline_full` (`0.586415`).
- The new source improves fold stability. `raw_timeline_full` tail3_worst is `0.597139`, much better than public_best `0.618609`.

## FP/FN Pattern

`raw_timeline_full` still overpredicts positives on major problem targets:

| target | true_pos_rate | raw_full pred_pos_rate | raw_full FP | raw_full FN |
|---|---:|---:|---:|---:|
| Q2 | 0.562222 | 0.911111 | 0.377778 | 0.028889 |
| Q3 | 0.600000 | 0.822222 | 0.300000 | 0.077778 |
| S3 | 0.662222 | 0.815556 | 0.208889 | 0.055556 |

`raw_timeline_composite` suppresses this better:

| target | composite pred_pos_rate | composite FP | composite FN |
|---|---:|---:|---:|
| Q2 | 0.666667 | 0.297778 | 0.193333 |
| Q3 | 0.684444 | 0.242222 | 0.157778 |
| S3 | 0.728889 | 0.168889 | 0.102222 |

So the raw model's useful role is not direct submission; it should be used as a feature/source in a guarded stack that controls Q2/Q3/S3 FP.

## Blend Diagnostics With Current Public-Best OOF

Global blend of current public_best and `raw_timeline_full`:

- Best full blend: logit alpha `0.80`
- OOF: full `0.583519`, last `0.578422`, tail3_worst `0.599214`
- This is excellent for private/full-CV but sacrifices public-tail proxy.

Global blend of current public_best and `raw_timeline_last`:

- Best last blend: logit alpha around `0.50-0.55`
- OOF: full around `0.590916-0.591048`, last `0.565922`, tail3_worst `0.621326-0.621813`
- This is a public-tail probe candidate class, but fold2/tail risk is high.

Target-wise blend signals:

- Q1: raw adds no value. Best alpha is `0`.
- Q2: raw_full improves full but worsens last; no guarded last improvement from raw.
- Q3: raw_full gives useful full/last blend around logit alpha `0.50-0.55`.
- S1: raw_last gives strong last improvement but full worsens slightly.
- S2: raw_last improves last strongly, with acceptable full if guarded.
- S3: raw source does not improve last; keep current public_best for S3 in public-tail mode.
- S4: raw_last improves last around logit alpha `0.50-0.60`.

## Test Distribution Risks

Compared with current public_best:

- `raw_full` makes Q2 extremely positive on test: `>0.5` rate `0.984` vs public_best `0.840`. This is too aggressive for direct submission.
- `raw_comp` lowers Q2/Q3/S3 positivity substantially: Q2 `0.764`, Q3 `0.736`, S3 `0.716`. This may be useful for FP suppression, but local last is weak.
- `raw_last` cuts S2 positivity hard: `0.548` vs public_best `0.788`; this matches its strong last gain on S2 but can be dangerous if public S2 prevalence is high.

## Early Stopping / Fallback Check

- All `1260` fold/config rows used `validation_logloss_early_stopping`.
- All rows record fallback as `best_iteration or current_iteration`.
- `best_iteration_mean` ranges from `1.0` to `555.5`, mean `104.37`.
- Several Q2 fold4 rows stop at iteration `1-3`, which means Q2 raw features are weak/unstable for the last fold. This explains why direct Q2 raw replacement is not a good public-tail move.

## Decision

Do not submit any raw timeline CSV directly.

Use raw timeline outputs as new stack/blend sources:

1. Private/full improvement source: `raw_timeline_full`
2. Public-tail source only under guard: `raw_timeline_last`
3. FP suppressor source for Q2/Q3/S3 exploration: `raw_timeline_composite`

The next implementation should run a guarded target-wise blend over current public_best plus these three raw sources, with:

- Q1 fixed to public_best
- Q2 heavily guarded, mostly public_best or composite only
- Q3 allowed raw_full logit alpha around `0.35-0.65`
- S1/S2/S4 allowed raw_last alpha around `0.35-0.75`
- S3 fixed or very small raw alpha
- selection must penalize tail3_worst above `0.620` and Q2/Q3/S3 positive-rate inflation
