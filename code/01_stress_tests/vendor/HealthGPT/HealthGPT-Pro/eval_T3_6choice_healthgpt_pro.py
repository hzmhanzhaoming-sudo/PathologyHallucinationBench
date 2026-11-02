#!/usr/bin/env python3
"""Evaluate HealthGPT-Pro on the T1 pathology MCQ benchmark.

Intended server location:
    /data5/hzm/HealthGPT/HealthGPT-Pro/eval_healthgpt_pro_pathology_mcq.py

This is a single-file evaluator for both image and no-image modes. The prompt
is intentionally identical for both modes:

    Answer this pathology multiple-choice question using the question and options.
    Return only the option letter, such as A or B. Do not explain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_REPO_ROOT = Path("/data5/hzm/HealthGPT/HealthGPT-Pro")
DEFAULT_MODEL_PATH = Path("/data5/hzm/Model/HealthGPT-Pro-8B")
DEFAULT_INPUT_WITH_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_data/6-choice-test/pathology_vqa_benchmark_1000_with_image.json"
)
DEFAULT_INPUT_NO_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_data/6-choice-test/pathology_vqa_benchmark_1000_empty_image.json"
)
DEFAULT_OUTPUT_WITH_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_results/6-choice/healthgpt_pro_T3_results/healthgpt_pro_T3_6choice_with_image.jsonl"
)
DEFAULT_OUTPUT_NO_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_results/6-choice/healthgpt_pro_T3_results/healthgpt_pro_T3_6choice_no_image.jsonl"
)

LABEL_RE = re.compile(r"^\s*([A-Z])\s*[\)\].:\-]?", re.I)
NUM_PREFIX_TO_DATASET = {
    "100": "QuiltVQA",
    "200": "PathVQA",
    "400": "PatchVQA",
    "500": "PathMMU",
}
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
    ".BMP",
    ".TIF",
    ".TIFF",
)


def import_healthgpt(repo_root: Path) -> Any:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from inference.utils import load_model, decode_output  # type: ignore

    return load_model, decode_output


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_healthgpt_model(
    load_model: Any,
    model_path: Path,
    dtype: torch.dtype,
    attn_implementation: str,
) -> tuple[Any, Any, str]:
    try:
        model, processor = load_model(
            model_id=str(model_path),
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map="auto",
        )
        return model, processor, attn_implementation
    except Exception as exc:
        if attn_implementation != "flash_attention_2":
            raise
        print(
            "HealthGPT-Pro load with flash_attention_2 failed; retrying with eager.\n"
            f"Original error: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, processor = load_model(
            model_id=str(model_path),
            dtype=dtype,
            attn_implementation="eager",
            device_map="auto",
        )
        return model, processor, "eager"


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def option_label(text: str) -> str | None:
    match = LABEL_RE.match(str(text))
    return match.group(1).upper() if match else None


def valid_letters(options: list[Any]) -> list[str]:
    labels = []
    for idx, option in enumerate(options):
        label = option_label(str(option))
        labels.append(label or chr(ord("A") + idx))
    return labels


def gold_letter(answer: Any) -> str | None:
    return option_label(str(answer))


def normalize_option_body(option: Any) -> str:
    return LABEL_RE.sub("", str(option)).strip().lower().rstrip(". ")


def parse_prediction(text: str, options: list[Any]) -> str | None:
    legal = set(valid_letters(options))
    raw = str(text or "").strip()
    upper = raw.upper()

    patterns = [
        r"^\s*\(?([A-Z])\)?[\s\.\):\-]*$",
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])[\)\.:\-\s]",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match and match.group(1) in legal:
            return match.group(1)

    standalone = [x for x in re.findall(r"\b([A-Z])\b", upper) if x in legal]
    if len(set(standalone)) == 1:
        return standalone[0]

    normalized = raw.lower().rstrip(". ")
    matches = []
    for option in options:
        label = option_label(str(option))
        body = normalize_option_body(option)
        if label and body and (normalized == body or body in normalized):
            matches.append(label)
    return matches[0] if len(set(matches)) == 1 else None


def build_prompt(row: dict[str, Any], strict_retry: bool = False) -> str:
    options = row.get("options", [])
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"Invalid options for sample {row.get('num')}: {options!r}")
    letters = valid_letters(options)
    question = str(row.get("question", "")).strip()
    option_text = "\n".join(str(option).strip() for option in options)

    instruction = (
        "Answer this pathology multiple-choice question using the question and options."
    )
    if strict_retry:
        instruction += " Your previous response did not follow the required format."

    return (
        f"{instruction}\n"
        "Return only the option letter, such as A or B. Do not explain.\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_text}\n\n"
        f"Valid answer letters: {', '.join(letters)}."
    )


def get_dataset_name(row: dict[str, Any]) -> str:
    explicit = row.get("category") or row.get("dataset") or row.get("source")
    if explicit:
        return str(explicit)
    prefix = str(row.get("num", "")).split("-", 1)[0]
    return NUM_PREFIX_TO_DATASET.get(prefix, prefix or "unknown")


def resolve_image(raw: Any, input_path: Path, image_root: Path | None) -> Path | None:
    raw_text = str(raw or "").strip()
    if not raw_text:
        return None
    original = Path(raw_text)
    candidates = []
    if original.is_absolute():
        candidates.append(original)
    else:
        candidates.append((input_path.parent / original).resolve())
        if image_root is not None:
            candidates.append((image_root / original).resolve())
    if image_root is not None:
        candidates.extend([image_root / original.name, image_root / raw_text.lstrip("/\\")])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if not candidate.suffix:
            for suffix in IMAGE_EXTENSIONS:
                with_suffix = candidate.with_suffix(suffix)
                if with_suffix.is_file():
                    return with_suffix
    return None


def build_messages(prompt: str, image_path: Path | None) -> list[dict[str, Any]]:
    if image_path is None:
        content = [{"type": "text", "text": prompt}]
    else:
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    return [{"role": "user", "content": content}]


def infer_one(
    row: dict[str, Any],
    image_path: Path | None,
    model: Any,
    processor: Any,
    decode_output: Any,
    max_new_tokens: int,
    temperature: float,
    strict_retry: bool = False,
) -> tuple[str, str]:
    prompt = build_prompt(row, strict_retry=strict_retry)
    messages = build_messages(prompt, image_path=image_path)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    raw_output = decode_output(generated_ids, inputs.input_ids, processor).strip()
    return raw_output, prompt


def add_stat(stats: dict[str, dict[str, int]], key: str, correct: bool) -> None:
    stats[key]["total"] += 1
    stats[key]["correct"] += int(correct)


def finalize_stats(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    return {
        key: {**value, "accuracy": value["correct"] / value["total"]}
        for key, value in sorted(stats.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--use-image", type=int, choices=(0, 1), default=1)
    parser.add_argument("--image-root", type=Path, default=Path("/data5/hzm/Benchmark"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-invalid", type=int, default=1)
    parser.add_argument("--print-raw-output", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "eager"),
        default="flash_attention_2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "with_image" if args.use_image else "no_image"
    input_path = args.input or (
        DEFAULT_INPUT_WITH_IMAGE if args.use_image else DEFAULT_INPUT_NO_IMAGE
    )
    output_path = args.output or (
        DEFAULT_OUTPUT_WITH_IMAGE if args.use_image else DEFAULT_OUTPUT_NO_IMAGE
    )

    for path in [args.repo_root, args.model_path, input_path]:
        if not path.exists():
            raise FileNotFoundError(str(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    load_model, decode_output = import_healthgpt(args.repo_root)
    model, processor, actual_attn = load_healthgpt_model(
        load_model=load_model,
        model_path=args.model_path,
        dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )

    rows = read_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    stats: dict[str, dict[str, dict[str, int]]] = {
        name: defaultdict(lambda: {"total": 0, "correct": 0})
        for name in ("overall", "unknown_level")
    }

    with output_path.open("w", encoding="utf-8", buffering=1) as out:
        for row in tqdm(rows, desc=f"HealthGPT-Pro/{mode}"):
            started = time.perf_counter()
            options = row.get("options", [])
            image_path = None
            error = None
            raw_outputs: list[str] = []
            prompts: list[str] = []
            pred = None
            raw = ""
            prompt = ""
            try:
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError(f"Invalid options: {options!r}")
                if args.use_image:
                    image_path = resolve_image(row.get("img"), input_path, args.image_root)
                    if image_path is None:
                        raise FileNotFoundError(f"Image not found: {row.get('img')}")

                for attempt in range(args.retry_invalid + 1):
                    raw, prompt = infer_one(
                        row=row,
                        image_path=image_path,
                        model=model,
                        processor=processor,
                        decode_output=decode_output,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        strict_retry=(attempt > 0),
                    )
                    raw_outputs.append(raw)
                    prompts.append(prompt)
                    pred = parse_prediction(raw, options)
                    if pred is not None:
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if args.print_raw_output:
                    traceback.print_exc()

            gold = gold_letter(row.get("answer"))
            correct = pred is not None and pred == gold
            source_dataset = get_dataset_name(row)
            num_prefix = str(row.get("num", "")).split("-", 1)[0] or "unknown"
            unknown_count = row.get("t3_unknown_count")
            if unknown_count is None:
                match = re.search(r"-unknown(\d+)$", str(row.get("num", "")))
                unknown_count = int(match.group(1)) if match else None
            else:
                unknown_count = int(unknown_count)
            unknown_level = (
                f"unknown{unknown_count}" if unknown_count is not None else "unknown"
            )

            record = {
                "dataset": mode,
                "source_file": str(input_path),
                "num": row.get("num"),
                "category": source_dataset,
                "num_prefix": num_prefix,
                "option_count": len(options) if isinstance(options, list) else 0,
                "t3_original_num": row.get("t3_original_num"),
                "t3_unknown_count": unknown_count,
                "t3_variant": row.get("t3_variant", unknown_level),
                "img": row.get("img") if args.use_image else "",
                "resolved_image": str(image_path) if image_path else None,
                "question": row.get("question"),
                "options": options,
                "gold_answer": row.get("answer"),
                "gold_label": gold,
                "raw_output": raw,
                "raw_outputs": raw_outputs,
                "predicted_label": pred,
                "correct": correct,
                "prompt": prompt,
                "prompts": prompts,
                "model_id": str(args.model_path),
                "attn_implementation": actual_attn,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "error": error,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.print_raw_output:
                print(
                    f"{row.get('num')}\tgold={gold}\tpred={pred}\tcorrect={correct}\traw={raw!r}\terror={error}",
                    flush=True,
                )

            add_stat(stats["overall"], "all", correct)
            add_stat(stats["unknown_level"], unknown_level, correct)

    summary: dict[str, Any] = {
        "overall": finalize_stats(stats["overall"])["all"],
        "by_unknown_level": finalize_stats(stats["unknown_level"]),
    }

    summary_json = output_path.with_name(output_path.name + ".summary.json")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
