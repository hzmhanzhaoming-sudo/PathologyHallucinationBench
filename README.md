<div align="center">

# PathologyHallucinationBench

### A paired pathology benchmark for separating visual and textual hallucination resistance

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Baseline items](https://img.shields.io/badge/Baseline_items-1%2C200-1F77B4)](#benchmark-composition)
[![Stress tests](https://img.shields.io/badge/Stress_tests-6-E76F51)](#six-paired-stress-tests)
[![Evaluated models](https://img.shields.io/badge/Evaluated_models-16-6A4C93)](#model-runners)
[![Scoring smoke test](https://img.shields.io/badge/Scoring_smoke_test-passed-2A9D8F)](#reproducibility-check)

**Standard accuracy can look strong even when a model ignores visual evidence, follows a misleading statement, or fails to recognize that the answer is absent. PathologyHallucinationBench makes these failure routes measurable.**

[Overview](#study-overview) · [Dataset](#benchmark-composition) · [Stress tests](#six-paired-stress-tests) · [Scoring](#scoring) · [Results](#main-findings) · [Run](#quick-start)

</div>

---

<table align="center">
  <tr>
    <td align="center" width="25%"><strong>1,200</strong><br><sub>baseline pathology items</sub></td>
    <td align="center" width="25%"><strong>6</strong><br><sub>paired stress tests</sub></td>
    <td align="center" width="25%"><strong>16</strong><br><sub>evaluated models</sub></td>
    <td align="center" width="25%"><strong>4</strong><br><sub>model families</sub></td>
  </tr>
</table>

## Why this benchmark?

Pathology visual question answering is usually reported as a single accuracy value. That value does not reveal *why* a model succeeds. A correct answer may arise from image-grounded interpretation, textual shortcuts, answer-option priors, or compliance with an external hint.

PathologyHallucinationBench uses paired interventions on the same source questions to separate three capabilities:

1. **Dataset-weighted diagnostic accuracy** — performance above the chance level of each source dataset.
2. **Visual hallucination resistance** — robustness to missing images, image–text conflict, and image-availability switching.
3. **Textual hallucination resistance** — robustness to option replacement, answer absence, and wrong-answer hints.

The benchmark evaluates **16 models** from four families: closed-source general, open-source general, pathology-specialized, and general medical systems.

## Benchmark composition

The baseline contains **1,200 pathology VQA questions** from four source datasets.

| Dataset | Items | Main answer format | Clinician-derived weight |
|---|---:|---|---:|
| QuiltVQA | 100 | Binary | 0.2227 |
| PathVQA | 200 | Binary | 0.2312 |
| PathMMU | 500 | Four-choice | 0.2850 |
| PatchVQA | 400 | Six- or seven-choice | 0.2611 |
| **Total** | **1,200** | — | **1.0000** |

The five later interventions use the shared **900-question non-binary subset**: 500 four-choice, 200 six-choice, and 200 seven-choice questions. V1 uses all 1,200 baseline questions.

Every stress-test record retains the identifier of its baseline item. This paired design makes it possible to measure response transitions on the same question—for example, correct-to-incorrect flips after image removal or adoption of a prespecified wrong hint—rather than comparing aggregate accuracy across unrelated samples.

## Study overview

<p align="center">
  <img src="assets/figure1_ab_problem.png" width="100%" alt="Figure 1 panels a and b: standard evaluation and the two hallucination routes">
</p>

Panels **a–b** establish the motivating problem. Models from all four families can reach the correct endpoint under standard evaluation, yet paired stress tests reveal two distinct failure routes: answering without usable visual evidence and following misleading text over the image. The benchmark therefore treats endpoint accuracy, visual grounding, and textual robustness as separate properties.

<p align="center">
  <img src="assets/figure1_c_benchmark_design.png" width="68%" alt="Figure 1 panel c: pathology benchmark sources, curation, and paired stress-test construction">
</p>

Panel **c** summarizes benchmark construction: four complementary pathology sources are filtered through rule-based eligibility checks and multi-LLM-assisted semantic review, then mapped to three visual and three textual paired interventions while preserving the baseline item identity.

## Six paired stress tests

| Task | Route | Paired intervention | Targeted failure mode |
|---|---|---|---|
| **V1** | Visual | Remove the image from the 1,200 baseline questions | Correct answers sustained by text-only shortcuts |
| **V2** | Visual | Add a plausible but image-inconsistent description | Adoption of a prespecified misleading visual target |
| **V3** | Visual | Add `Cannot see the image` and compare image/no-image conditions | Failure to switch between diagnosis and abstention |
| **T1** | Textual | Replace one or two distractors with `unknown` | Candidate-space and option-content instability |
| **T2** | Textual | Replace a distractor or the correct answer with NOTA | Failure to detect whether a valid answer is available |
| **T3** | Textual | Add an explicit suggestion pointing to a wrong option | Retention of an externally supplied incorrect hint |

Each item preserves a stable source identifier, enabling paired item-level analysis rather than comparison between unrelated question sets.

## Scoring

All six task scores range from 0 to 100, and **higher is better**.

| Score | Definition |
|---|---|
| V1 | `100 × (1 − no-image-correct / image-correct)` |
| V2 | `100 × (1 − misleading-target adoption / original-correct)` |
| V3 | `100 × correct image/no-image switches / image-correct` |
| T1 | `100 × [1 − (L1 + G1 + L2 + G2) / 4]` |
| T2 | `100 × [1 − (false-NOTA + missed-NOTA) / 2]` |
| T3 | `100 × retained-correct after wrong hint / original-correct` |

The task weights come from five-dimensional ratings by three clinicians and are normalized within each route:

```text
Visual resistance = 0.32974684×V1 + 0.31455696×V2 + 0.35569620×V3
Text resistance   = 0.29059829×T1 + 0.36231884×T2 + 0.34708287×T3
```

The exported overall hallucination-resistance score normalizes the same six clinician-derived task weights jointly:

```text
Overall resistance = 0.16487342×V1 + 0.15727848×V2 + 0.17784810×V3
                   + 0.14529915×T1 + 0.18115942×T2 + 0.17354144×T3
```

For dataset `d`, chance-adjusted accuracy is:

```text
AdjustedAccuracy_d = 100 × clip[(Accuracy_d − Chance_d) / (100 − Chance_d), 0, 1]
```

The final accuracy coordinate is the clinician-weighted sum of the four adjusted dataset accuracies.

## Main findings

### Visual stress-test results

<p align="center">
  <img src="assets/visual_stress_test_results.png" width="100%" alt="Visual hallucination stress-test results for image removal, image-text conflict, and image availability">
</p>

The visual route measures three distinct behaviors. **V1** tests whether correct answers survive after the image is removed; **V2** measures adoption of a prespecified misleading image description; and **V3** tests whether a model diagnoses when the image is present but correctly abstains when it is absent. The panels report both model-level and family-level response transitions.

### Textual stress-test results

<p align="center">
  <img src="assets/text_stress_test_results.png" width="100%" alt="Textual hallucination stress-test results for unknown options, answer absence, and wrong-answer hints">
</p>

The textual route separates option-content instability, answer-absence detection, and resistance to explicit wrong-answer hints. **T1** traces accuracy as distractors are replaced by `unknown`; **T2** contrasts false NOTA acceptance with correct NOTA detection; and **T3** follows baseline-correct answers after an incorrect external suggestion.

Together, the two result figures show that apparently similar endpoint accuracy can arise from very different response-transition profiles. Visual grounding and textual robustness therefore remain separate reported coordinates rather than being collapsed into one unqualified hallucination rate.

### Interactive three-axis robustness landscape

<p align="center">
  <a href="assets/Fig3D_interactive_pareto.html?raw=1">
    <img src="assets/three_axis_pareto.png" width="100%" alt="Interactive three-axis Pareto landscape of visual resistance, textual resistance, and adjusted accuracy">
  </a><br>
  <sub>Visual resistance × textual resistance × clinician-weighted chance-adjusted accuracy. Highlighted surfaces denote the first two Pareto layers.</sub>
</p>

<p align="center">
  <a href="assets/Fig3D_interactive_pareto.html?raw=1"><strong>Download the interactive 3D explorer</strong></a>
  &nbsp;·&nbsp; drag to rotate &nbsp;·&nbsp; scroll to zoom &nbsp;·&nbsp; hover for model values
</p>

GitHub README pages cannot execute embedded JavaScript. Download the standalone HTML above and open it in a browser, or serve the cloned repository locally:

```bash
python3 -m http.server 8000
# open http://localhost:8000/assets/Fig3D_interactive_pareto.html
```

The three-axis view preserves the three coordinates instead of compressing them into one headline score. The Pareto layers identify competitive trade-offs and make visible whether an apparently strong model is limited by perception, textual robustness, or baseline diagnostic performance.

### Family-level numerical summary

The table below provides the exact family means corresponding to the three reported coordinates; the complete model-level values can be reproduced with the scoring notebook.

| Model family | Visual resistance | Text resistance | Adjusted accuracy |
|---|---:|---:|---:|
| Closed-source general | 64.1 | 76.5 | 54.3 |
| Open-source general | 45.2 | 76.8 | 46.7 |
| Pathology-specialized | 26.6 | 70.3 | 34.1 |
| General medical | 39.4 | 72.5 | 42.6 |

## Repository layout

```text
PathologyHallucinationBench/
├── assets/                     # README figures
├── data/                       # V1–V3 and T1–T3 JSON inputs
├── evaluation/                 # model-family evaluation runners
├── example/                    # one question represented across six tasks
├── result/
│   └── score_six_hallucination_tasks.ipynb
├── README.md
└── requirements.txt
```

Task JSON files contain image filenames rather than machine-specific absolute paths. Source pathology images are stored separately and supplied at runtime through `--image-root`.

## Quick start

```bash
git clone https://github.com/hzmhanzhaoming-sudo/PathologyHallucinationBench.git
cd PathologyHallucinationBench

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The base installation supports dataset inspection, API runners, and the scoring notebook. Local multimodal runners additionally require their original model repositories, checkpoints, and compatible `torch`/`transformers` environments.

Run a model-family evaluator by changing only the model path, input file, output file, and image condition:

```bash
python3 evaluation/eval_qwen35.py \
  --model-path /path/to/Qwen3.5-9B \
  --input data/V1/v1_with_image.json \
  --output result/qwen35/v1_with_image.jsonl \
  --use-image 1 \
  --max-new-tokens 16 \
  --temperature 0
```

For API models, provide credentials through environment variables:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
python3 evaluation/eval_gpt.py \
  --input data/V1/v1_with_image.json \
  --output result/gpt/v1_with_image.jsonl \
  --use-image 1 \
  --model YOUR_MODEL_ID
```

Place each model's JSONL outputs under `result/<model_name>/`, then open and run:

```text
result/score_six_hallucination_tasks.ipynb
```

The notebook exports six task scores, event counts, coverage audits, visual and textual clinician-weighted scores, and the clinician-weighted six-task summary.

It can also be executed non-interactively:

```bash
jupyter nbconvert --to notebook --execute \
  result/score_six_hallucination_tasks.ipynb \
  --output score_six_hallucination_tasks.executed.ipynb \
  --ExecutePreprocessor.timeout=600
```

## Reproducibility check

The public package was checked at three levels before release:

| Check | Scope | Status |
|---|---|---|
| Dataset audit | all V1–V3 and T1–T3 JSON files, row counts, identifiers, and paired conditions | **Passed** |
| Runner validation | Python syntax compilation for all ten evaluator scripts | **Passed** |
| Scoring smoke test | synthetic outputs spanning all six tasks, executed end-to-end through the notebook | **Passed** |

The smoke test recognized all 11 required result conditions, produced coverage `1.0` for every task, and exported `six_task_robustness_scores.csv`, `six_task_event_counts.csv`, and `scoring_input_audit.csv`. Model inference itself remains environment-dependent because checkpoints and upstream model implementations are not redistributed in this repository.

## Model runners

| Runner | Model variants |
|---|---|
| `eval_gpt.py` | GPT-5.6 SOL, GPT-5.6 Luna |
| `eval_gemini.py` | Gemini 3.5 Flash Lite, Gemini 3.6 Flash |
| `eval_qwen35.py` | Qwen3.5-2B, Qwen3.5-4B, Qwen3.5-9B |
| `eval_glm46v.py` | GLM-4.6V Flash |
| `eval_quilt_llava.py` | Quilt-LLaVA |
| `eval_pathgen_llava.py` | PathGen-LLaVA |
| `eval_patho_r1.py` | Patho-R1-3B, Patho-R1-7B |
| `eval_huatuogpt_vision.py` | HuatuoGPT-Vision |
| `eval_unimedvl.py` | UniMedVL |
| `eval_healthgpt_pro.py` | HealthGPT-Pro-4B, HealthGPT-Pro-8B |

## Output schema

Evaluation outputs are append-safe JSONL records. The scoring notebook supports both common schemas:

- `predicted_label`, `gold_label`, `correct`, `source_file`;
- `pred_letter`, `gold_letter`, `correct`.

Only complete paired items enter each score denominator, and the exported coverage fields record the exact evaluated sample size.

## Scope

PathologyHallucinationBench is a research benchmark for controlled model evaluation. It does not provide clinical diagnoses or replace pathologist review.
