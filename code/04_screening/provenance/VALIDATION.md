# Delivery validation — 2026-10-27

- Current supplied PPTX slides 16–18 and the three Note 6 PNG files were read without alteration.
- Source verification: **65 / 65 checks passed**, including all prompt hashes, native paragraph extraction, template separation, original PPTX/XML/PNG fingerprints and unchanged source files.
- Combined local offline suite: **46 / 46 tests passed** (35 adapter/transport tests plus 11 source-integrity tests).
- Tests use synthetic records, temporary files, a locally isolated `jsonschema` 4.23.0 dependency, and mocked API responses. No screening model was called and no API credential was read by tests.
- Transport tests cover the default no-network/no-write preview, required explicit execution, refusal/incomplete/non-JSON failures, no automatic retry, no credential leakage in errors, HTTPS-only endpoints and redirect refusal.
- These checks do not constitute online provider compatibility testing, medical validation, an executed screening experiment or replication of the historical cohort.
- Source fixed prompts are literal; API role mapping, input envelope, caller-provided response schema and validation are implementation settings, not additional published prompt text.

Fresh standalone source check: `python -B verify_prompt_source.py`.

Fresh combined tests after installing dependencies in an isolated environment: `python -B -m unittest discover -p 'test_*.py' -v`.

The package is deliberately additive. Deployment checks compare all pre-existing CODE files and the original 400-file three-package manifest without changing them. Deployment reports are retained separately in the task workspace, not mixed with experiment results.
