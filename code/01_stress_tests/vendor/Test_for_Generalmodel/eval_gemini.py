#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Google Gemini on the T1 pathology MCQ benchmark.

This entry point reuses ``eval_T2_gemini.py`` so T1 and T2 have identical API
calling, prompt, parsing, retry, resume and summary behavior. Both image and
no-image modes use exactly the same question/options prompt.
"""

from __future__ import annotations

import sys

import eval_T2_gemini as core


DATA_DIR = "/data5/hzm/Benchmark/Benchmark_T1_data"
RESULT_DIR = (
    "/data5/hzm/Benchmark/Benchmark_T1_results/gemini3.6_flash_T1_results"
)

core.DEFAULT_DATA_DIR = DATA_DIR
core.DEFAULT_RESULT_DIR = RESULT_DIR
core.DEFAULT_WITH_IMAGE_JSON = (
    DATA_DIR + "/pathology_vqa_benchmark_1200_with_image.json"
)
core.DEFAULT_WITHOUT_IMAGE_JSON = (
    DATA_DIR + "/pathology_vqa_benchmark_1200_empty_image.json"
)
core.DEFAULT_WITH_IMAGE_OUTPUT = (
    RESULT_DIR + "/gemini3.6_flash_T1_with_image.jsonl"
)
core.DEFAULT_WITHOUT_IMAGE_OUTPUT = (
    RESULT_DIR + "/gemini3.6_flash_T1_no_image.jsonl"
)


def has_option(option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:])


if __name__ == "__main__":
    if not has_option("--task-name"):
        sys.argv.extend(["--task-name", "T1"])
    if not any(has_option(option) for option in ("--mode", "--input", "--output")):
        sys.argv.extend(["--mode", "both"])
    core.main()
