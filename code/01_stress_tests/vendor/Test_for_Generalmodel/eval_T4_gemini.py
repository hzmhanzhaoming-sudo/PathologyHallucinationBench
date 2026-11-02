#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Google Gemini on the T4 wrong-image-description benchmark.

T4 is an image-only stress test.  This entry point deliberately reuses the
tested API, prompt, parsing, resume, retry, usage and summary implementation in
``eval_T2_gemini.py`` so that T1/T2/T4 differ only in benchmark data and
result locations.

The unified prompt remains:

    Answer this pathology multiple-choice question using the question and options.
    Return only the option letter, such as A or B. Do not explain.
"""

from __future__ import annotations

import sys

import eval_T2_gemini as core


core.DEFAULT_DATA_DIR = "/data5/hzm/Benchmark/Benchmark_T4_data"
core.DEFAULT_RESULT_DIR = (
    "/data5/hzm/Benchmark/Benchmark_T4_results/gemini3.6_flash_T4_results"
)
core.DEFAULT_WITH_IMAGE_JSON = (
    core.DEFAULT_DATA_DIR
    + "/pathology_vqa_benchmark_900_wrong_image_description_with_image.json"
)
core.DEFAULT_WITHOUT_IMAGE_JSON = ""
core.DEFAULT_WITH_IMAGE_OUTPUT = (
    core.DEFAULT_RESULT_DIR + "/gemini3.6_flash_T4_with_image_full.jsonl"
)
core.DEFAULT_WITHOUT_IMAGE_OUTPUT = ""


def option_value(argv: list[str], option: str) -> str | None:
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None


def has_option(argv: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in argv)


def main() -> None:
    argv = sys.argv[1:]
    requested_mode = option_value(argv, "--mode")
    requested_use_image = option_value(argv, "--use-image")

    if requested_mode in {"without_image", "both"} or requested_use_image == "0":
        raise SystemExit(
            "T4 is an image-only benchmark. Use --mode with_image or --use-image 1."
        )

    if not has_option(argv, "--task-name"):
        sys.argv.extend(["--task-name", "T4"])

    # With no explicit single-run input/output and no mode, use the validated
    # T4 image-only defaults.
    if (
        not has_option(argv, "--mode")
        and not has_option(argv, "--input")
        and not has_option(argv, "--output")
    ):
        sys.argv.extend(["--mode", "with_image"])

    core.main()


if __name__ == "__main__":
    main()
