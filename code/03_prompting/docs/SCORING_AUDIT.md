# Scoring provenance audit (2026-10-24)

Read-only inspection; no inference was run and no historical result was changed.

## Current prompt-experiment scorer

Server files:

- `/data5/hzm/Benchmark/Benchmark_Few-shot CoT/code/score_results.py`
- `/data5/hzm/Benchmark/Benchmark_Few-shot direct/code/score_results.py`
- `/data5/hzm/Benchmark/Benchmark_Zero-shot CoT/code/score_results.py`

The first two are byte-identical (SHA-256
`5d8d00c9d0fb34c2eb67b270b724ee20e79dc442926644a1eeae077147c69825`).
The zero-shot file has SHA-256
`bf925929af484b43bab902a6fedc66178a2c6c2e31f2865c8c13102d48423d77`.
Their `score_task` mathematics is identical. Differences are additional
few-shot demonstration-pair consistency checks and explanatory metadata.
The few-shot version is the safer shared implementation; the checks are
conditional on demonstration metadata and do not change zero-shot formulas.

The existing formal scorer expects canonical `pred_letter` and validates
`.run.json` provenance. Historical Original uses `predicted_label`; repaired
prompt runs can have nested provenance. Therefore publication regression must
select the exact files in the snapshot's `source_audit`, normalize only the
stored field names, and not discover arbitrary newer files or reparse answers.

## Definitions that must be preserved

| Endpoint | Denominator and formula |
| --- | --- |
| Base | Correct / 1,200; ordinary condition accuracy, not a resistance score. |
| V2 | Among 900 stress IDs, O = same protocol's Base-correct IDs; H = selection of the specified misleading target. Score = 100 (1 - H/O). Other new errors and invalid outputs are not target adoption. |
| V3 | O = correct image-present answers with the augmented options, not ordinary Base correctness. S = those same IDs correctly selecting image absence without the image. Score = 100 S/O. S/900 is a different outcome. |
| T2 | O = same protocol's Base-correct IDs within the 900 stress IDs. F = false NOTA when answer present; M = missed NOTA when answer absent. Score = 100 [1 - (F+M)/(2O)]. Non-NOTA selection when answer present is an availability decision, not necessarily correct diagnosis. |
| T3 | O = same protocol's Base-correct stress IDs; R = retained correctness after wrong hint. Score = 100 R/O. Rescues from Base-wrong items do not enter R/O. |

V2, V3 and T3 use Wilson intervals for the relevant Bernoulli numerator;
T2 does not use an invented single-binomial interval. Formal prompt scoring
withholds incomplete pairs unless explicitly requested as exploratory.
Zero-opportunity formal scores are undefined. The all-four-strategy shared
Base-correct intersection is a separate sensitivity analysis, not the main O.

## Current original V1 / T1 source

Authoritative paper workflow:
`/data5/hzm/Benchmark/Draw_picture/Fig_sum/plot_six_task_radar.ipynb`

SHA-256:
`e72fcbdb6c289eb75552add1f7b44286deed8d39eb4782e32ff9553b520d5b4d`

V1 uses all 1,200 paired questions:

`100 * clip((C_img - C_noimg) / (C_img - sum_i(1/K_i)), 0, 1)`.

The numerator is floored at zero and the denominator must be positive;
otherwise V1 is undefined. K_i is each item's number of options. This is not
the old conditional image-correct persistence formula.

T1 reads
`/data5/hzm/Benchmark/Benchmark_T3_results/T3_results/T3_paired_statistics.csv`,
filters `Image + Text`, sums the four paired states separately for U1/U2,
then computes:

`100 * [1 - (L1 + G1 + L2 + G2)/4]`.

L is correct-to-wrong / Base-correct; G is wrong-to-correct / Base-wrong.
The paper plotting notebook sets a zero-denominator component to zero.
All current 16 models have nonzero component denominators.

The **server's current** public notebook
`/data5/hzm/pathology-hallucination-benchmark-github-draft/result/score_six_hallucination_tasks.ipynb`
already contains chance-normalized V1. The local
`github_score_notebook/build_scoring_notebook.py` and its old notebook are
outdated and must not be used as a V1 source.

Some Fig_sum original outputs read already-rounded percentages from legacy
summary CSVs. For example V2 Gemini 3.5 Flash Lite is 53.211 in Fig_sum but
53.211009... from item counts. Distinguish display rounding from a scoring
discrepancy in original-task regression.

## Publication regression fixtures

Current 12-model snapshot, local:
`outputs/Fig_addition_package/source_data/comparison_12models.json`

SHA-256:
`6346d44d34fe81b80915de004a32e8ab9c1ee8496f5e067253f50396cb11ffb7`

Server copies include:
`/data5/hzm/Benchmark/Benchmark_addition_sum/picture_main/delta_colored_points_v16/source_data/comparison_12models.json`.

The 2026-10-22 independent audit read all 336 selected source files and
316,800 predictions, found identical source hashes/labels and no score
differences. It retained and disclosed historical answer-recovery decisions;
numeric agreement is not an adjudication of free-text rationale semantics.

Useful local outputs:

- `outputs/new_experiments_full_audit_20261022/raw_audit/current_formal_scores.csv`
- `outputs/new_experiments_full_audit_20261022/raw_audit/current_condition_accuracy.csv`
- `outputs/new_experiments_full_audit_20261022/raw_audit/audit_summary.json`

`verify_published.py` loads exactly the snapshot sources and calls the packaged
shared `score_task` / `accuracy_rows`. Its intended scope is 192 formal score
rows, 48 Base accuracy rows and 336 condition accuracy rows, including the
integer event counts and the reported score/accuracy intervals.
