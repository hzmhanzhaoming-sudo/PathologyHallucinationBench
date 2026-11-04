# Independent task scoring

Six scripts score V1, V2, V3, T1, T2 and T3 independently. A seventh,
`score_axes.py`, combines their saved reports into three separate scores and
Pareto layers. Scoring uses Python's standard library only; no model, API,
checkpoint or image files are needed. Keep the sibling `evaluation/` directory,
which supplies the shared JSON validation code.

## Files needed by each scorer

`--data-root` is the released `data` directory. `--runs` names one or more model
result directories. Only the following reference/prediction pairs are read:

| Script | Reference JSONs under `data-root` | Prediction files in each run |
|---|---|---|
| `score_v1.py` | `V1/v1_with_image.json`, `V1/v1_without_image.json` | `V1_with_image.jsonl`, `V1_without_image.jsonl` |
| `score_v2.py` | `V1/v1_with_image.json`, `V2/v2_image_text_conflict_with_image.json` | `V1_with_image.jsonl`, `V2.jsonl` |
| `score_v3.py` | `V3/v3_with_image.json`, `V3/v3_without_image.json` | `V3_with_image.jsonl`, `V3_without_image.jsonl` |
| `score_t1.py` | Base plus the three JSONs under `T1/4-choice`, `T1/6-choice`, `T1/7-choice` | `V1_with_image.jsonl`, `T1_4choice.jsonl`, `T1_6choice.jsonl`, `T1_7choice.jsonl` |
| `score_t2.py` | Base, `T2/t2_answer_present_with_image.json`, `T2/t2_answer_absent_with_image.json` | `V1_with_image.jsonl`, `T2_present.jsonl`, `T2_absent.jsonl` |
| `score_t3.py` | Base, `T3/t3_wrong_answer_hint_with_image.json` | `V1_with_image.jsonl`, `T3.jsonl` |

Base is `V1/v1_with_image.json`. V3 does not require the ordinary Base output.
V1 can be scored as soon as both of its conditions finish, without waiting for
V2–T3. A full-run manifest is accepted while other conditions are still running
only if the selected task's inputs and outputs have been registered and verified.

## Run one task

From the repository root (named `code` in the desktop copy), score the
Qwen3.5-9B run created by the evaluation example:

```bash
python scoring/score_v1.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/V1
```

To score the remaining tasks, run each command separately:

```bash
python scoring/score_v2.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/V2
python scoring/score_v3.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/V3
python scoring/score_t1.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/T1
python scoring/score_t2.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/T2
python scoring/score_t3.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/T3
```

Only run the tasks you need. For multiple models, supply all run directories to
each scorer, for example `--runs result/main/qwen35_9b result/main/gpt_model`,
and use a new output tree such as `result/scores/comparison/V1` through `T3`.
Scripts also work by absolute path from any working directory. Existing output
directories are never overwritten. `--allow-custom-size` permits small custom
datasets while preserving identity, option and pairing validation.

Each task writes:

- `<TASK>_scores.csv`: the task's 0–100 score for each model.
- `task_event_counts.csv`: denominators, response-event counts and defined status.
- `condition_accuracy.csv`: condition-level correct/total/invalid counts.
- `provenance.json`: input, prediction and output hashes, model IDs and cohort.
- V1 additionally writes `base_dataset_counts.csv` for weighted accuracy.

Predictions are JSONL records with a unique `num` and a parsed `predicted_label`
(`pred_letter` is also accepted). Legal letters must match the task's options;
invalid responses must be explicit `null`. A missing item or recorded inference
error stops scoring; it is not silently converted to an incorrect answer.
Stored correctness flags are ignored and recomputed from the reference answers.

## Combine all six tasks

After scoring the same models in all six tasks:

```bash
python scoring/score_axes.py \
  --task-results result/scores/qwen35_9b/V1 result/scores/qwen35_9b/V2 \
                 result/scores/qwen35_9b/V3 result/scores/qwen35_9b/T1 \
                 result/scores/qwen35_9b/T2 result/scores/qwen35_9b/T3 \
  --weights scoring/weights.csv --output result/three_axes/qwen35_9b
```

The summary checks that reports have matching model inventories, core question
sets, reference hashes and shared Base predictions. It rejects mixed batches
and changed report files. Saved task-report directories can be moved without
requiring the original prediction or reference paths.

Outputs include `three_axis_scores.csv`, merged `six_task_scores.csv`, Base
dataset counts and provenance. The three axes are weighted accuracy, visual
resistance and textual resistance; there is no combined overall score. Pareto
dominance maximizes all three, and ties do not dominate each other. Layers are
relative to the supplied models; one model alone necessarily occupies layer 1.
Undefined scores stay blank in task reports. If any supplied task score is
undefined, aggregation stops rather than inventing an axis value or Pareto layer.

For independently prepared CSVs, the explicit lower-level interface is:

```bash
python scoring/score_axes.py --scores six_task_scores.csv \
  --base-counts base_dataset_counts.csv --weights scoring/weights.csv \
  --output /path/to/scores/three_axes
```

The CSV-only interface validates values but cannot verify their source batches.
Prefer `--task-results` when using the six supplied scorers. See
[SCORING.md](SCORING.md) for formulas and [weights.csv](weights.csv) for explicit
weights. No weights are needed for individual task scores.

