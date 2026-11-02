#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemini evaluator for the merged T3 seven-choice unknown-option variants."""

from __future__ import annotations

import sys

import eval_T2_gemini as core


DATA_DIR = "/data5/hzm/Benchmark/Benchmark_T3_data/7-choice-test"
RESULT_DIR = "/data5/hzm/Benchmark/Benchmark_T3_results/7-choice/gemini3.6_flash_T3_results"

core.DEFAULT_DATA_DIR = DATA_DIR
core.DEFAULT_RESULT_DIR = RESULT_DIR
core.DEFAULT_WITH_IMAGE_JSON = DATA_DIR + "/pathology_vqa_benchmark_1200_with_image.json"
core.DEFAULT_WITHOUT_IMAGE_JSON = DATA_DIR + "/pathology_vqa_benchmark_1200_empty_image.json"
core.DEFAULT_WITH_IMAGE_OUTPUT = RESULT_DIR + "/gemini3.6_flash_T3_7choice_with_image_full.jsonl"
core.DEFAULT_WITHOUT_IMAGE_OUTPUT = RESULT_DIR + "/gemini3.6_flash_T3_7choice_no_image_full.jsonl"


def has_option(option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:])


if __name__ == "__main__":
    if not has_option("--task-name"):
        sys.argv.extend(["--task-name", "T3_7choice"])
    if not any(has_option(option) for option in ("--mode", "--input", "--output")):
        sys.argv.extend(["--mode", "both"])
    core.main()
