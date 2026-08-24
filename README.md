<div align="center">

# PathologyHallucinationBench

### A paired pathology benchmark for separating visual and textual hallucination resistance

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#quick-start)
[![Baseline items](https://img.shields.io/badge/Baseline_items-1%2C200-1F77B4)](#benchmark-composition)
[![Stress tests](https://img.shields.io/badge/Stress_tests-6-E76F51)](#six-paired-stress-tests)
[![Evaluated models](https://img.shields.io/badge/Evaluated_models-16-6A4C93)](#model-runners)

**Standard accuracy can look strong even when a model ignores visual evidence, follows a misleading statement, or fails to recognize that the answer is absent. PathologyHallucinationBench makes these failure routes measurable.**

</div>

---

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

```mermaid
flowchart LR
    A[1,200 baseline pathology VQA items] --> B[Visual route]
    A --> C[Textual route]
    B --> V1[V1 · Image removal]
    B --> V2[V2 · Image–text conflict]
    B --> V3[V3 · Image availability]
    C --> T1[T1 · Unknown option]
    C --> T2[T2 · Answer absence]
    C --> T3[T3 · Wrong-answer hint]
    V1 --> X[Visual resistance]
    V2 --> X
    V3 --> X
    T1 --> Y[Text resistance]
    T2 --> Y
    T3 --> Y
    A --> Z[Dataset-weighted accuracy]
    X --> P[Three-axis robustness landscape]
    Y --> P
    Z --> P
```

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

<p align="center">
  <img src="assets/six_task_family_profiles.png" width="62%" alt="Six-task robustness profiles by model family">
</p>

The six-task profiles reveal model-family-specific failure patterns that are hidden by overall accuracy. In particular, strong textual resistance does not imply strong visual resistance.

<p align="center">
  <img src="assets/visual_text_correlation.png" width="40%" alt="Visual-text cross-route Spearman correlation matrix">
</p>

The cross-route correlation matrix contains positive, near-zero, and negative associations. Visual and textual hallucination resistance therefore represent related but non-interchangeable properties and should be reported separately.

<p align="center">
  <img src="assets/three_axis_pareto.png" width="68%" alt="Three-axis Pareto landscape">
</p>

The three-axis landscape jointly represents visual resistance, textual resistance, and dataset-weighted accuracy. Pareto layers identify models that achieve competitive trade-offs without collapsing the three measurements into a single headline number.

<p align="center">
  <img src="assets/family_three_metrics.png" width="48%" alt="Mean three-axis scores for four model families">
</p>

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
