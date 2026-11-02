# Six paired stress tests

Task-specific evaluation runners for the 16-model benchmark, using the original
server input layout. `vendor/` contains the original runners; `registry.py` maps
the six paper tasks to their 11 input conditions.

| Task | Conditions | Requests per model |
| --- | --- | ---: |
| Base | Image-present control | 1,200 |
| V1 | Image removed | 1,200 |
| V2 | Misleading image description | 900 |
| V3 | Image present / absent with abstention option | 1,800 |
| T1 | One / two distractors replaced with unknown | 1,800 |
| T2 | Correct answer present / absent | 1,800 |
| T3 | Wrong-answer hint | 900 |

From the repository root:

```bash
# Preview all conditions; add --validate-inputs to check the server-layout inputs.
python evaluation/manuscript/run_local.py --model qwen35-4b --task all
python evaluation/manuscript/run_api.py --model gemini-3.6-flash --task all

# Execute into a new run directory after configuring the model/data environment.
python evaluation/manuscript/run_local.py --model qwen35-4b --task all --execute --run-id run1

# Score the saved answers.
python scoring/score_stress_run.py \
  --run evaluation/manuscript/results/run1/qwen35-4b \
  --output result/scored_run1
```

Use `--data-root`, `--python` and `--model-path` to select installed resources.
API credentials use `OPENAI_API_KEY` or `GEMINI_API_KEY`. Input and original-source
hashes are checked against `provenance/original_source_manifest.json`.
For the Hugging Face file layout, use `evaluation/run_benchmark.py` instead.
