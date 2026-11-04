# Evaluation

Only model testing lives here. No benchmark data, checkpoints, API keys, published
predictions, server-specific launchers or embedded result tables are included.
Python 3.10+ is required. Planning and validation use only the standard library.
GPU inference additionally requires the appropriate model environment and weights.

## Entry point

Run from the repository root (named `code` in the desktop copy). The example
uses the released data in `benchmark_data/` and a Qwen3.5-9B checkpoint at
`checkpoints/Qwen3.5-9B`; replace only the checkpoint path if yours differs.

```bash
python evaluation/run_benchmark.py --runner eval_qwen35.py \
  --model-path checkpoints/Qwen3.5-9B \
  --data-root benchmark_data/data --image-root benchmark_data/images \
  --output-dir result/main/qwen35_9b --max-tokens 16 --validate-inputs
```

This validates inputs and prints the plan, **without inference or output writes**.
To execute the evaluation:

```bash
python evaluation/run_benchmark.py --runner eval_qwen35.py \
  --model-path checkpoints/Qwen3.5-9B \
  --data-root benchmark_data/data --image-root benchmark_data/images \
  --output-dir result/main/qwen35_9b --max-tokens 16 --execute
```

Examples use Bash line continuations. Windows paths containing spaces must be quoted;
in PowerShell, enter each command on one line or use PowerShell's continuation syntax.
The output directory must be new and separate from inputs, checkpoints and code.
One full run contains 11 conditions / 9,600 initial item evaluations; retries can
increase the number of inference requests. Each condition starts a backend process.

For API evaluation, set `OPENAI_API_KEY` and `OPENAI_MODEL_ID` in the environment.
`OPENAI_MODEL_ID` must be the actual provider model ID, not a paper display label:

```bash
python evaluation/run_benchmark.py --runner eval_gpt.py \
  --model "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID to your provider model ID}" \
  --data-root benchmark_data/data --image-root benchmark_data/images \
  --output-dir result/main/gpt_model --max-tokens 512 --execute
```

Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in the environment. The planner never records
API keys. API calls can incur costs; previewing does not make any calls.

## Model adapters and settings

| Adapter | Additional requirements |
|---|---|
| `eval_gpt.py`, `eval_gemini.py` | Explicit `--model`; corresponding API key |
| `eval_qwen35.py`, `eval_glm46v.py` | Local checkpoint and compatible Torch/Transformers |
| `eval_patho_r1.py` | Checkpoint, Torch/Transformers and `qwen_vl_utils` |
| `eval_quilt_llava.py`, `eval_pathgen_llava.py` | Checkpoint and matching LLaVA source checkout |
| `eval_huatuogpt_vision.py`, `eval_unimedvl.py`, `eval_healthgpt_pro.py` | Checkpoint and corresponding model source checkout |

Use `--backend-config /path/to/settings.json` for model-specific options, for example:

```json
{"repo-root": "/path/to/model-source", "temperature": 0.0}
```

Keys are adapter CLI option names without `--`. Boolean values are only for
switches such as `load-4bit`; numeric switches such as `do-sample` use `0` or `1`.
Paths in this JSON resolve from the caller's working directory. Required source
checkouts must be supplied explicitly; the package does not download them.
Do not put credentials in this file. Dataset/output/model switches cannot be
overridden through it. Native adapter `--help` lists additional options (local
adapters require their model dependencies even for `--help`).

`--max-tokens N` overrides the output-token limit. When omitted, defaults remain:
512 for GPT, Gemini and Patho-R1; 64 for UniMedVL; 16 for the other adapters.
Most local adapters use greedy generation; API sampling defaults are adapter/provider
specific. Keep settings identical across paired conditions. The adapters retain
their own prompt, parser and invalid-response retry logic; Patho-R1 requests
`<think>...` / `<answer>...`, whereas the other primary adapters request a letter.
Record the model/version and generation settings for each run.

## Data contract

The 11 input paths and expected sizes are defined once in `schema.py`:

| Condition | Records | Images |
|---|---:|---|
| V1 with / without image | 1,200 each | yes / no |
| V2 | 900 | yes |
| V3 with / without image | 900 each | yes / no |
| T1 4-choice / 6-choice / 7-choice | 1,000 / 400 / 400 | yes |
| T2 present / absent | 900 each | yes |
| T3 | 900 | yes |

Each JSON file contains an array. Each item requires `num`, `question`, `options`,
`answer` and `img`. Options are labelled `A)` to `H)` (also `.` or `:` separators).
The answer is a legal letter or an exact full option string. `img` is relative to
the image root, or empty for an image-absent condition. Absolute/traversing image
paths and ambiguous basename searches are not accepted.

Additional fields:

- V2: `target_wrong_option`, a legal **wrong** letter.
- T1: `t1_original_num` and integer `t1_unknown_count` (1 or 2).
- Dataset identity: explicit `dataset` / `source_dataset`, or the defined source-ID
  prefix (`100`, `200`, `400`, `500`).
- V3 abstention option: `Cannot see the image`.
- T2 NOTA option: `The correct answer is not among the listed options.`

The shared core and both T1 levels must match baseline IDs exactly. V1 paired
questions/options may differ only in image availability. `--allow-custom-size`
permits smaller complete paired inputs for development; it does not permit missing
conditions, unpaired items, or silently label a small run as the full benchmark.

## Outputs

Each new run directory contains:

- `run_manifest.json`: model, commands, status, source/input/output hashes.
- Eleven `CONDITION.jsonl` files: `num`, `predicted_label`, `gold_label`, `correct`.
- `raw/`: native backend records, prompts, raw responses and summaries for auditing.

An invalid-format response has `predicted_label: null`. Execution/network/image
errors instead stop finalization; the manifest is marked `failed`, not `complete`.
Existing output directories are never resumed or overwritten. Investigate a failed
run and start a new directory; no automatic mixing of attempts is performed.

Feed each completed task into its `scoring/score_v1.py`–`score_t3.py` entry point;
combine all six reports using `scoring/score_axes.py`. This directory contains
primary benchmark evaluation; prompting interventions are a separate workflow.

