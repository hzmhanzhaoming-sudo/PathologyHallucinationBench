<div align="center">

# PathologyHallucinationBench

### A paired pathology benchmark for separating visual and textual hallucination resistance

**Standard accuracy can look strong even when a model ignores visual evidence, follows a misleading statement, or fails to recognize that the answer is absent. PathologyHallucinationBench makes these failure routes measurable.**

[Overview](#study-overview) · [Dataset](#benchmark-composition) · [Stress tests](#six-paired-stress-tests) · [Scoring](#scoring) · [Models](#evaluated-model-resources) · [Results](#main-findings) · [Run](#quick-start)

</div>

---

## Why this benchmark?

Pathology visual question answering is usually reported as a single accuracy value. That value does not reveal *why* a model succeeds. A correct answer may arise from image-grounded interpretation, textual shortcuts, answer-option priors, or compliance with an external hint.

PathologyHallucinationBench uses paired interventions on the same source questions to separate three capabilities:

1. **Dataset-weighted diagnostic accuracy** — the clinician-weighted proportion of correctly answered items across the four source datasets.
2. **Visual hallucination resistance** — robustness to missing images, image–text conflict, and image-availability switching.
3. **Textual hallucination resistance** — robustness to option replacement, answer absence, and wrong-answer hints.

The benchmark evaluates **16 models** from four families: closed-source general, open-source general, pathology-specialized, and general medical systems.

## Benchmark composition

The baseline contains **1,200 pathology VQA questions** from four source datasets.

| Dataset | Questions | Main answer format | Clinician-derived weight |
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

**Three complementary dimensions · 0–100 · Higher is better**

- **Accuracy** — correct answers, weighted by source dataset.
- **Visual resistance** — responses to altered or missing image evidence (V1–V3).
- **Textual resistance** — responses to altered options and misleading hints (T1–T3).

Dataset and task weights are clinician-derived, with task weights normalized within each resistance dimension.

The three dimensions are reported **separately, without a combined overall score**.
Accuracy uses weighted item counts, with no chance correction or min–max scaling.

[Task-level formulas and exact weights →](scoring/SCORING.md) · [Scoring code →](scoring/)

## Dataset and model source links

### Benchmark data

The curated evaluation inputs and images are available from the [PathologyHallucinationBench dataset on Hugging Face](https://huggingface.co/datasets/hanzhaoming/triplet-benchmark). This release is distinct from the original source datasets below.

### Original source datasets

| Source | Official project / data access |
|---|---|
| QuiltVQA | [Quilt-LLaVA project](https://github.com/aldraus/quilt-llava) · [Quilt-VQA data instructions](https://github.com/aldraus/quilt-llava/blob/main/docs/Data.md) |
| PathVQA | [PathVQA repository](https://github.com/UCSD-AI4H/PathVQA) · [Official challenge and data information](https://pathvqachallenge.grand-challenge.org/Data_Info/) |
| PathMMU | [PathMMU repository](https://github.com/PathMMU-Benchmark/PathMMU) · [Dataset](https://huggingface.co/datasets/jamessyx/PathMMU) |
| PatchVQA | [PathBench repository and PatchVQA access instructions](https://github.com/superjamessyx/PathBench) |

Please cite the original source works and follow their respective licenses and access conditions when using their data.

### Evaluated model resources

The benchmark evaluates **16 models across four families**, with four models per family. Names follow the experiment configurations. Provider catalogs are general documentation, not immutable snapshots of the evaluated API versions. Model-family runners are in [`evaluation/`](evaluation/).

- **Closed-source general:** [GPT model documentation](https://platform.openai.com/docs/models) (GPT-5.6 SOL / Luna configuration labels); [Gemini model documentation](https://ai.google.dev/gemini-api/docs/models) (Gemini 3.5 Flash Lite / 3.6 Flash configuration labels).
- **Open-source general:** [GLM-4.6V Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash), [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
- **Pathology-specialized:** [Quilt-LLaVA](https://github.com/aldraus/quilt-llava), [PathGen-LLaVA](https://huggingface.co/jamessyx/PathGen-LLaVA), [Patho-R1-3B](https://huggingface.co/WenchuanZhang/Patho-R1-3B), [Patho-R1-7B](https://huggingface.co/WenchuanZhang/Patho-R1-7B).
- **General medical:** [HuatuoGPT-Vision](https://github.com/FreedomIntelligence/HuatuoGPT-Vision), [UniMedVL](https://huggingface.co/General-Medical-AI/UniMedVL), [HealthGPT-Pro-4B / Pro-8B](https://github.com/ZJU4HealthCare/HealthGPT/tree/main/HealthGPT-Pro).

## Main findings

### Visual stress-test results

<p align="center">
  <img src="assets/visual_stress_test_results.png" width="100%" alt="Visual hallucination stress-test results for image removal, image-text conflict, and image availability">
</p>

The visual route measures three distinct behaviors. **V1** quantifies the chance-adjusted loss of clean-image capability after image removal, so a higher score indicates weaker retention of no-image shortcuts; **V2** measures adoption of a prespecified misleading image description; and **V3** tests whether a model diagnoses when the image is present but correctly abstains when it is absent. The panels report both model-level and family-level response transitions.

### Textual stress-test results

<p align="center">
  <img src="assets/text_stress_test_results.png" width="100%" alt="Textual hallucination stress-test results for unknown options, answer absence, and wrong-answer hints">
</p>

The textual route separates option-content instability, answer-absence detection, and resistance to explicit wrong-answer hints. **T1** traces accuracy as distractors are replaced by `unknown`; **T2** contrasts false NOTA acceptance with correct NOTA detection; and **T3** follows baseline-correct answers after an incorrect external suggestion.

Together, the two result figures show that apparently similar endpoint accuracy can arise from very different response-transition profiles. Visual grounding and textual robustness therefore remain separate reported coordinates rather than being collapsed into one unqualified hallucination rate.

### Three-axis robustness landscape

<p align="center">
  <img src="assets/three_axis_pareto.png" width="100%" alt="Three-axis Pareto landscape of visual resistance, textual resistance, and dataset-weighted accuracy"><br>
  <sub>Visual resistance × textual resistance × clinician-weighted micro-accuracy across the four source datasets. Highlighted surfaces denote the first two Pareto layers.</sub>
</p>

The three-axis view preserves the three coordinates instead of compressing them into one headline score. The Pareto layers identify competitive trade-offs and make visible whether an apparently strong model is limited by perception, textual robustness, or baseline diagnostic performance.

### Family-level numerical summary

<p align="center">
  <img src="assets/family_three_metrics.png" width="88%" alt="Family means of visual resistance, text resistance and dataset-weighted accuracy">
</p>

### Six-task profiles and cross-task relationships

<p align="center">
  <img src="assets/six_task_family_profiles.png" width="49%" alt="Six-task resistance profiles for the four model families">
  <img src="assets/visual_text_correlation.png" width="49%" alt="Spearman correlations between visual and textual task scores across models">
</p>

The family-level profiles retain all six task scores, while the correlation matrix summarizes relationships between visual and textual task performance across models.

## Repository layout

```text
PathologyHallucinationBench/
├── assets/       # Overview and result figures displayed on this page
├── evaluation/   # Model adapters, input validation and the 11-condition run planner
├── scoring/      # Six-task scoring, three separate dimensions and Pareto analysis
├── README.md
└── requirements.txt
```

## Quick start

### Installation and data

The commands below use Bash and are run from the repository root (the desktop
copy is named `code`). Python 3.10+ is required.

```bash
git clone https://github.com/hzmhanzhaoming-sudo/PathologyHallucinationBench.git
cd PathologyHallucinationBench
python -m pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hanzhaoming/triplet-benchmark', repo_type='dataset', local_dir='benchmark_data', allow_patterns=['data/**', 'images/**'])"
```

If the repository is already present, skip the first two commands and open its
root directory. Authenticate with Hugging Face if the dataset requires access.
The download is kept in `benchmark_data/`, separate from the source code:

```text
benchmark_data/
  data/V1/ ... data/T3/      # Released reference JSONs
  images/                   # Image paths referenced by those JSONs
```

For local inference, install the selected model's Torch/Transformers environment
and obtain its checkpoint separately. `requirements.txt` provides the common
utilities, not every model's GPU environment. Scoring alone needs only Python's
standard library and the reference/prediction files.

### 1. Evaluate Qwen3.5-9B

Place the Qwen3.5-9B checkpoint in `checkpoints/Qwen3.5-9B`, or replace
`--model-path` with its actual location. First check all inputs without inference:

```bash
python evaluation/run_benchmark.py --runner eval_qwen35.py \
  --model-path checkpoints/Qwen3.5-9B \
  --data-root benchmark_data/data --image-root benchmark_data/images \
  --output-dir result/main/qwen35_9b --max-tokens 16 --validate-inputs
```

Run the same plan with inference enabled:

```bash
python evaluation/run_benchmark.py --runner eval_qwen35.py \
  --model-path checkpoints/Qwen3.5-9B \
  --data-root benchmark_data/data --image-root benchmark_data/images \
  --output-dir result/main/qwen35_9b --max-tokens 16 --execute
```

This evaluates 11 conditions (9,600 initial item evaluations) and writes
normalized prediction files plus `run_manifest.json` into `result/main/qwen35_9b`.
Previewing creates no output directory. An actual run requires a new output
directory; use a different name for each model or repeat run.

For an API model, set `OPENAI_API_KEY` and `OPENAI_MODEL_ID` in your environment
and use the provider's actual model ID. The following command makes real API calls:

```bash
python evaluation/run_benchmark.py --runner eval_gpt.py \
  --model "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID to your provider model ID}" \
  --data-root benchmark_data/data --image-root benchmark_data/images \
  --output-dir result/main/gpt_model --max-tokens 512 --execute
```

For Gemini use `eval_gemini.py`, `GEMINI_API_KEY` and the corresponding provider
model ID, with a separate output directory. See
[`evaluation/README.md`](evaluation/README.md) for the other model adapters,
source-checkout configuration and generation settings.

### 2. Score the six tasks independently

The following commands score the saved Qwen3.5-9B predictions; they do not load
the model or call an API. Each task has its own output directory.

```bash
python scoring/score_v1.py --data-root benchmark_data/data \
  --runs result/main/qwen35_9b --output result/scores/qwen35_9b/V1
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

**To score V1 only, run just the first command.** V1 needs its image-present and
image-absent outputs; V3 needs only its own two outputs. Other tasks can remain
unfinished. Each scorer reads only its required files and exports the task score,
event counts, condition accuracies and provenance. The exact file requirements
are listed in [`scoring/README.md`](scoring/README.md).

### 3. Combine the six reports into three scores and Pareto layers

```bash
python scoring/score_axes.py \
  --task-results result/scores/qwen35_9b/V1 result/scores/qwen35_9b/V2 \
                 result/scores/qwen35_9b/V3 result/scores/qwen35_9b/T1 \
                 result/scores/qwen35_9b/T2 result/scores/qwen35_9b/T3 \
  --weights scoring/weights.csv --output result/three_axes/qwen35_9b
```

The main output is `result/three_axes/qwen35_9b/three_axis_scores.csv`, containing
weighted accuracy, visual resistance, textual resistance and Pareto layer.
There is no combined overall score. With only one model, its Pareto layer is
necessarily 1; meaningful comparisons require multiple models.

To compare models, pass the same model directories to **each** task scorer,
for example `--runs result/main/qwen35_9b result/main/gpt_model`, and save the
six reports under a new tree such as `result/scores/comparison/V1` through `T3`.
Then pass those six comparison directories to `score_axes.py`. All reports must
contain the same models and paired reference data. Existing outputs are not
overwritten, and missing records or inference errors stop scoring.
