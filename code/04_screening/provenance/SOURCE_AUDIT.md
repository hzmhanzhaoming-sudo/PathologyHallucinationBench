# Manuscript-supplied screening prompts: literal-source audit

Source checked: current `npjdm_for_all/supply_绘图.pptx`, slides 16–18; the current SI Note 6 explicitly sends readers to the separately supplied Figures for the full prompts. Native text is preferable to OCR or historical operational scripts.

The corresponding supplied figures are:

- `Note_06_01_Stage1_content_classification.PNG`
- `Note_06_02_Stage1_eligibility_screening.PNG`
- `Note_06_03_Stage2_semantic_comparison.PNG`

All source PPTX/XML/image SHA-256 hashes and extracted-text hashes are in `prompt_source_manifest.json`. Each slide's prompt is one native shape (zero-based shape 5). Text runs were concatenated within each native paragraph, and paragraph boundaries were serialized as LF. Original punctuation, spaces, spelling and blank paragraphs were not repaired.

## Exactly supplied content

1. Stage 1 content classification: fixed instruction only, 3,430 characters. Three semantic axes, confidence, a contiguous source excerpt (at most 160 characters), one row per unit_id. It refers to a supplied JSON schema, but no actual JSON schema or user-template wording appears on the supplied slide.
2. Stage 1 eligibility screening: fixed instruction, 1,339 characters, plus one exact user template: `Review every record and return only schema-valid JSON. records={records_json}`. The five criteria are traceable source; complete Q-A record; answer-key alignment; textual pathology relevance; clarity/clinical discrimination. This is text-only and explicitly forbids assuming image content. The full JSON schema is not displayed.
3. Stage 2 semantic comparison: fixed instruction only, 1,813 characters. Requires original texts, existing image-review findings, content labels and cohort summaries. Compares only supplied neighbours; does not inspect images; advisory only, not final cohort selection. It names output fields and value sets but does not provide a formal JSON schema/container or user-message template.

The source label is “Fixed instruction,” not “system message.” Mapping that block to an API system role is an implementation choice and should not be advertised as an explicitly published role mapping. Likewise, absent schema/user-template text must not be silently reconstructed from an old execution script and called “exactly the manuscript prompt.”

## Source imperfections that must not be silently repaired

The current slide 17 includes these exact joined strings:

- `not_assessable.Return eligible only if `
- `all five criteria pass; return reject for a clear failure Return only the `
- `requested JSON and do not reveal private chain-of-thought.`

The current SI Note 6 prose specifies `needs_review` otherwise, whereas the displayed eligibility instruction does not explicitly state that branch. This is an existing source discrepancy, not a reason to change the supplied prompt without permission. Keep literal text and flag it in documentation. No medical assessment or screening run has been performed.

## Scope boundary

These three supplied prompts do not implement the complete historical 4,500-to-1,200 selection workflow. In particular, the supplied Stage 2 comparison consumes image-review findings but explicitly does not re-inspect images. There is no separately supplied full image-review prompt in these three figures. Do not invent a fourth “verbatim” instruction by paraphrasing Methods, and do not claim that running these prompts alone exactly recreates the original image review, quotas, selection order, deduplication or cohort membership.

Current main Methods describes Stage 2 in future tense, whereas SI Note 6 describes a completed workflow. This audit records that wording difference only; it does not adjudicate which historical process was run and does not alter manuscript text.
