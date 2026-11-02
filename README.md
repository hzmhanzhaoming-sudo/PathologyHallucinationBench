# PathologyHallucinationBench

Code for six paired pathology stress tests and three-dimensional scoring.

```text
evaluation/
  eval_*.py                         Model-family evaluators
  run_benchmark.py                   Run planner for the Hugging Face layout
  manuscript/                       Task-specific 16-model evaluation runners
  tests/                            Evaluation-entry tests
scoring/
  score.py                          Command-line scoring
  score_six_hallucination_tasks.ipynb Six-task scoring notebook
  score_stress_run.py                Scoring for manuscript-layout runs
  score_axes.py                      Three dimensions and Pareto layers
  core/                             Shared scoring and input-validation functions
  tests/                            Scoring tests
```

## Install and download

```bash
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hanzhaoming/triplet-benchmark', repo_type='dataset', local_dir='.', allow_patterns=['data/**', 'images/**'])"
```

The [dataset](https://huggingface.co/datasets/hanzhaoming/triplet-benchmark)
uses `data/` and `images/`. Use an authorized Hugging Face login or `HF_TOKEN`.
Local models require their checkpoints and compatible model environments;
API credentials use `OPENAI_API_KEY` or `GEMINI_API_KEY`.

## Evaluate

```bash
# Preview the 11 conditions; does not run inference.
python evaluation/run_benchmark.py --runner eval_qwen35.py \
  --model-path /path/to/Qwen3.5-9B --output-dir result/qwen_new --validate-inputs

# Add --execute to run the displayed plan.
# For API models, use --runner eval_gpt.py --model YOUR_MODEL_ID instead.
```

The complete benchmark has 9,600 initial requests per model. The public-layout
planner uses the model-family evaluators. Task-specific manuscript runners and
their server-layout commands are in [`evaluation/manuscript`](evaluation/manuscript).

## Score

```bash
python scoring/score.py --repo-root . --output result/scored_qwen
```

Place prediction JSONL files in `result/<model>/`. The scorer reports:

- Dataset-weighted diagnostic accuracy;
- Visual resistance;
- Textual resistance;
- Six task-specific scores, event counts and input coverage.

There is no combined overall score. Existing output directories are not overwritten.
The notebook rejects malformed, duplicate or incomplete inputs by default.

To compute three-axis scores and Pareto layers from explicit score/count/weight CSVs:

```bash
python scoring/score_axes.py --scores six_task_scores.csv \
  --base-counts base_dataset_counts.csv --weights clinician_weights.csv \
  --output result/three_axes
```

CSV columns are documented in `scoring/score_axes.py`.

## Test

```bash
python -m unittest discover -s evaluation/tests -v
python evaluation/manuscript/test_offline.py
python -m unittest discover -s scoring/tests -v
```
