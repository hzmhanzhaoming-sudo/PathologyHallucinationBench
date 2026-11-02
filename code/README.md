# Reproducibility modules

These four cooperating modules are a curated export of the desktop code package.
Keep them together. They do **not** replace `evaluation/` or silently convert the
public Hugging Face files into the historical server layout.

| Module | Included | Required separately |
| --- | --- | --- |
| `01_stress_tests` | 16-model routing, 11 conditions, frozen task-specific runners and input/source hashes | Audited server-layout inputs, model environments/checkpoints |
| `02_scoring` | Six-task scoring, three separate axes/Pareto, prompt comparisons, regression tools | Predictions; historical source snapshot for full server regression |
| `03_prompting` | Zero-shot CoT, few-shot direct, few-shot CoT; fixed demo text, standard and historical API profiles | Authorized demo images, source inputs, model environments |
| `04_screening` | Classification, eligibility and semantic-comparison prompts/adapters | Caller-supplied schema and source records/review findings |

## Safe entry points

From the repository root:

```bash
# Preview only; no model/API execution.
python code/01_stress_tests/run_local.py --model qwen35-4b --task all

# Offline synthetic tests, not a re-run of manuscript results.
python code/02_scoring/verify_all.py --offline-only --output /path/to/NEW_report

# Score existing public-layout outputs with the current notebook, into a NEW folder.
python code/02_scoring/score_original.py --repo-root . --output /path/to/NEW_scores

# Server-layout prompt experiment: always use --dry-run when checking a command.
python code/03_prompting/run_local.py --model qwen35-4b --task all --strategy zero_shot_cot --dry-run

# Byte-exact checks of the supplied screening prompt text.
python code/04_screening/verify_prompt_source.py
```

`01_stress_tests` and `04_screening/run_api.py` require `--execute` for inference.
`03_prompting` retains its desktop interface: **omitting `--dry-run` runs inference**.
It has seven conditions (6,600 initial requests/model/strategy), not all six stress
tests. Its published comparison cohort was 12 models; inspect
`03_prompting/docs/HISTORICAL_PROTOCOLS.md` before reproducing historical settings.
Native reasoning settings and the historical Luna prompt are not interchangeable.

For the public Hugging Face layout, use `tools/run_benchmark.py` and the root README.
The frozen `01_stress_tests` source hashes deliberately reject modified inputs;
do not disable them to make public-release files pass as the historical snapshot.
Public `t1_original_num`/`t1_unknown_count` and historical
`t3_original_num`/`t3_unknown_count` are different schema names for the same pairing
concept; the public notebook accepts either, but this is not a full dataset adapter.

## Scores and provenance

Keep diagnostic accuracy, visual resistance and textual resistance separate.
There is no overall composite score. Six task scores and their event counts remain
available. `score_axes.py` takes explicit clinician weights; it does not fit weights
to performance. A zero opportunity denominator is undefined, not zero resistance.
Prompt comparisons use each strategy's own opportunities for the primary analysis;
common-opportunity comparisons are separate sensitivity analyses.

Historical regression scripts require the original server files and, where noted,
an external `comparison_12models.json` supplied through `--snapshot`. This export
does not include old results, patient/test images, repeated historical source trees,
plotting notebooks with cleanup actions, or the frozen duplicate scoring notebook.
`package_manifest.json` describes this export, not the full desktop archive.

The fixed few-shot source and reference texts are included for inspection, but the
three source images must be obtained under the applicable upstream authorization.
The original absolute-path metadata are retained for traceability; they are not
portable defaults or an instruction to download/release those images. In particular,
few-shot inference needs the exact reviewed source images and compatible suite
hashes, not arbitrary substitute examples. Offline tests use synthetic images.

## Explicit remaining limits

- Screening is **not** the complete 4,500-to-1,200 curation pipeline: sampling,
  deduplication, independent image review, neighbour retrieval and final selection
  are not implemented by the three prompt adapters. No missing prompt is invented.
- Screening provenance records the supplied slide/Note numbering at extraction
  time; it is not a claim that the current manuscript still uses those numbers.
- Server model environments and historical result/image paths remain external.
  Passing offline tests is not evidence of successful GPU/API inference.
- Figure-specific bootstrap confidence intervals, correlation/permutation analyses,
  and all figure-generation scripts are not included in this code export. Existing
  paired tests in the prompt scoring module do not replace those analyses.
- Upstream dataset/image licenses and the release of the private Hugging Face
  repository must be resolved independently; this update changes neither.
