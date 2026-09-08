# Protocol aligned with Figure 1c

This document distinguishes the **intended protocol**, the **frozen evaluation cohort**, and the **review work actually recorded**. It does not certify clinical correctness or report a newly selected cohort.

## Two-stage curation and audit status

### Stage 1 — Hard Eligibility

| Figure criterion | Operational check | Evidence required |
|---|---|---|
| Traceable source | Link the question to its original dataset, source record, split, and image | Source identifiers and a provenance mapping |
| Valid gold answer | Check that the supplied answer resolves to a valid option; distinguish label validity from clinical validity | Original annotation; image–text adjudication for clinical validity |
| Complete Q–A record | Require a usable question and answer, and complete options for MCQs | Parsed record and missing-field checks |
| Readable image | Decode the file and inspect whether the relevant content is legible | File validation plus actual visual review; successful decoding alone is insufficient |
| Pathology-relevant | Check that the image and question address pathology | Joint image–text review |
| No duplicate image | Group shared source-image identifiers and select representatives; assess exact and near-duplicate images separately | Source mapping, byte/pixel checks, and documented near-duplicate review where performed |

Eligibility is not a guarantee of suitability for a visual-grounding stress test. Questions with valid annotations may still be answerable from text alone.

### Stage 2 — Semantic Filter Within the Eligible Pool

| Figure criterion | Operational interpretation |
|---|---|
| Clear image–question evidence | Identify the visible evidence supporting the question and gold answer; flag text-sufficient, ambiguous, or unsupported items |
| Fill under-covered pathology content | Assess marginal coverage against the retained cohort, not against an item in isolation |
| Retain clinically discriminative cases | Prefer cases for which the available image supports a meaningful distinction; require qualified review for clinical claims |
| Preserve gold-label stability | Flag ambiguity or contradictions without silently rewriting the source answer |
| Unique image | Recheck the final cohort after merging sources; do not treat repeated paired variants of one baseline item as unwanted duplicates |

Stage 1 removes ineligible records; Stage 2 assesses semantic suitability and cohort composition. The repeated image-uniqueness criterion is a final cross-source check, not a reason to delete the paired interventions required for scoring.

### Recorded evidence as of 8 September 2026

| Artifact or operation | Recorded scope | What it does not establish |
|---|---|---|
| Retrospective text/JSON audit | 1,200 fixed items at each stage; unresolved and rejected statuses retained | Image review or clinical certification |
| Retrospective candidate reconstruction | 9,121 source-image-identifier representatives; includes the fixed 1,200 | An independently selected, fully multimodally screened candidate pool |
| Subsequent image–text pilot | Four items reviewed at Stage 1 and the same four at Stage 2 | Full review of 1,200 or 9,121 items; population pass rates |
| Fixed-cohort preservation | Original questions, options, answers, and order retained | Every preserved item passed the review criteria |

The image–text pilot deliberately included problematic cases. Stage 2 pilot judgments included diagnostic checks of Stage 1 failures; these are not admissions through a completed prospective eligibility gate. No population pass rate should be extrapolated from this pilot.

The labels **Terra** and **Sol** in the schematic are reviewer-role labels. They must not be substituted for logged model identifiers or interpreted as evidence that these two named models reviewed the full cohort. A completed release should disclose actual model identifiers, exact prompts, input hashes, image-exposure logs, per-item dispositions, and adjudication records. Pixel-level and near-duplicate uniqueness must not be claimed from source identifiers alone.

These status statements summarize local audit records (`07_codex_native_review/summary.json`, `08_reconstructed_two_stage/summary.json`, and `09_image_text_review/run_status.json`). The full audit package is not bundled here. Subsequent image downloads improve availability but do not constitute additional semantic review.

## Source scope and cohort size

The frozen cohort contains 100 QuiltVQA, 200 PathVQA, 500 PathMMU, and 400 PatchVQA questions. The 400 PatchVQA questions comprise 200 six-choice and 200 seven-choice items. Source-record tracing identifies 1,180 `test` items and 20 PatchVQA `test_tiny` items; these are not training or validation questions.

The four source roles in the figure are illustrative. In particular, PatchVQA and PathMMU share source material, and PathMMU is not restricted to PubMed or teaching resources. Source-domain overlap is distinct from duplicate questions, duplicate images, and possible overlap with a model's training data.

The benchmark's 1,200 selected questions, the 900 non-binary baseline questions, the 9,121 reconstructed candidate representatives, and the complete upstream datasets are different populations. Do not substitute one count for another or describe the frozen target size as a natural pass count from the proposed protocol.

## Task numbering map

| Current figure/release | Intervention | Historical experiment/folder |
|---|---|---|
| V1 | Image removal | Benchmark_T1 |
| V2 | Image–text conflict | Benchmark_T4 |
| V3 | Cannot-see option | Benchmark_T6 |
| T1 | Unknown substitution | Benchmark_T3 |
| T2 | NOTA intervention | Benchmark_T5 |
| T3 | Wrong-answer hint | Benchmark_T7 |
| Auxiliary; outside the six-score aggregate | Option-order shuffling | Benchmark_T2 |

Do not infer current task semantics from an old `Benchmark_Tn` directory name. The public input filenames below use the current numbering.

## Released inputs and pairing

Download the JSON files from the dataset repository into `data/`. The expected layout, also checked against the local release copy, is:

| Input relative to `data/` | Rows | Image condition |
|---|---:|---|
| `V1/v1_with_image.json` | 1,200 | With image |
| `V1/v1_without_image.json` | 1,200 | No image |
| `V2/v2_image_text_conflict_with_image.json` | 900 | With true image and false description |
| `V3/v3_with_image.json` | 900 | With image and cannot-see option |
| `V3/v3_without_image.json` | 900 | No image and cannot-see option |
| `T1/4-choice/t1_4choice_unknown1_2_{with_image,without_image}.json` | 1,000 each | Both; 500 baseline items × 2 unknown levels |
| `T1/6-choice/t1_6choice_unknown1_2_{with_image,without_image}.json` | 400 each | Both; 200 baseline items × 2 unknown levels |
| `T1/7-choice/t1_7choice_unknown1_2_{with_image,without_image}.json` | 400 each | Both; 200 baseline items × 2 unknown levels |
| `T2/t2_answer_present_with_image.json` | 900 | With image; correct option retained |
| `T2/t2_answer_absent_with_image.json` | 900 | With image; correct option replaced |
| `T3/t3_wrong_answer_hint_with_image.json` | 900 | With image and qualified wrong hint |

Braces in the T1 rows denote two filenames, not literal filenames. The 14 input files contain 11,400 variant records, **not 11,400 independent cases**. The primary T1 score uses only the with-image U1/U2 conditions; no-image variants support additional analysis.

- Preserve the supplied item identifiers. T1 uses `t1_original_num` to pair with the clean baseline; some six-/seven-choice variant IDs begin with `200-`, even though their baseline source is PatchVQA (`400-`). Do not classify their dataset from the variant prefix alone.
- Keep each T1 choice-group output in a separate file. The scorer combines the groups by original item and unknown level.
- For V2, preserve `target_wrong_option`; target adoption cannot be reconstructed reliably from a generic accuracy flag alone.
- V3 adds one option, yielding **5, 7, or 8 choices**. The no-image gold answer is the per-item cannot-see option, not a fixed E. The with-image gold remains the original diagnostic answer after any label shift.
- In T2, NOTA means `The correct answer is not among the listed options.` It is an answer-availability response, not a statement about whether an image was supplied.
- The mixed file in `example/` illustrates one source question across conditions; it is not a single-mode evaluation input. Its V2 example includes the target label.

## Evaluation prompts and controls

The common first-attempt instruction used by most runners is:

```text
Answer this pathology multiple-choice question using the question and options.
Return only the option letter, such as A or B. Do not explain.

Question: {question}
Options:
{options}

Valid answer letters: {valid_letters}.
```

Task-specific descriptions and hints are already embedded in the input question/options. Do not append them again. Chat templates, image tokens, API wrappers, and invalid-format retry instructions remain runner-specific.

**Patho-R1 is an existing exception.** Its with-image instruction is `Answer this pathology visual multiple-choice question using the image, question, and options.` Its no-image instruction is `Answer this pathology multiple-choice question` (the code also contains surrounding whitespace). Both request brief reasoning followed by `<answer>A</answer>`. This release alignment does not change those prompts or claim that Patho-R1 was tested under the identical common prompt.

For paired comparisons, retain the same model checkpoint/API identifier, decoding settings, token budget, retry policy, and image preprocessing. Different output budgets for different model families should be disclosed, especially for reasoning models. Runtime failures, unparsed outputs, and missing pairs must be reported alongside accuracy. Human-readable model labels in the README are not guaranteed provider API identifiers; use the exact identifier recorded for each run.

## Interpretation and reporting limits

1. **V1 measures dependence, not correctness of visual reasoning.** Correct text-only answers can reflect knowledge, textual cues, option priors, or memorization. No single explanation is established by the accuracy gap.
2. **V2 is target-specific.** A model that avoids the misleading target but gives another wrong answer is not counted as target adoption. Report those other errors too.
3. **V3 measures operational modality awareness.** A high conditional switching score requires accompanying with-image accuracy and opportunity counts; it is not a general clinical abstention guarantee.
4. **T1 measures bidirectional instability.** Both losses and gains enter the defined score. Removing informative distractors can legitimately increase accuracy; gains alone are not proof of hallucination.
5. **T2 concerns answer-set validity.** A false NOTA selection and a failure to select NOTA when the correct answer is absent are different errors. Neither is synonymous with image unavailability.
6. **T3 measures correctness retention under a wrong suggestion.** The existing score counts every new error, not only explicit hint following. It probes susceptibility under the constructed condition, not its prevalence in routine use.

Scores are conditional operational proxies and require baseline accuracy, paired coverage, denominators, and uncertainty estimates. Task variants share baseline questions and must not be treated as independent samples for uncertainty estimation. Source-image overlap and possible training-data contamination require separate checks.

This synchronization changes documentation, runner path/metadata defaults, and label handling for eight-option V3 scoring. It does not rerun models, alter the six score formulas, change clinician weights, or regenerate the historical result figures. After any scoring compatibility correction, recompute results from the original predictions before claiming that a historical plotted value is unchanged.
