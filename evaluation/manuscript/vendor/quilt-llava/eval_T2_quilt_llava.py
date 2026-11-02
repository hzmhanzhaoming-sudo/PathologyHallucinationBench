#!/usr/bin/env python3
"""Evaluate Quilt-LLaVA on the T1 pathology MCQ benchmark.

Intended server location:
    /data5/hzm/quilt-llava/eval_quilt_llava_pathology_mcq.py

The prompt is intentionally identical for image and no-image modes:
    Answer this pathology multiple-choice question using the question and options.
    Return only the option letter, such as A or B. Do not explain.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


DEFAULT_REPO_ROOT = Path("/data5/hzm/quilt-llava")
DEFAULT_MODEL_PATH = Path("/data5/hzm/Model/Quilt-Llava")
DEFAULT_INPUT_WITH_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T2_data/pathology_vqa_benchmark_900_with_image.json"
)
DEFAULT_INPUT_NO_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T2_data/pathology_vqa_benchmark_900_empty_image.json"
)
DEFAULT_OUTPUT_WITH_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T2_results/quilt_llava_T2_results/quilt_llava_T2_with_image.jsonl"
)
DEFAULT_OUTPUT_NO_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T2_results/quilt_llava_T2_results/quilt_llava_T2_no_image.jsonl"
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


def import_quilt_llava(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from llava.constants import (  # type: ignore
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates  # type: ignore
    from llava.mm_utils import (  # type: ignore
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model  # type: ignore
    from llava.utils import disable_torch_init  # type: ignore

    return {
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "SeparatorStyle": SeparatorStyle,
        "conv_templates": conv_templates,
        "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
        "get_model_name_from_path": get_model_name_from_path,
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


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
    if standalone:
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
    bases = []
    if original.is_absolute():
        bases.append(original)
    else:
        bases.append((input_path.parent / original).resolve())
        if image_root is not None:
            bases.append((image_root / original).resolve())
    if image_root is not None:
        bases.extend([image_root / original.name, image_root / raw_text.lstrip("/\\")])

    for candidate in bases:
        if candidate.is_file():
            return candidate
        if not candidate.suffix:
            for suffix in IMAGE_EXTENSIONS:
                with_suffix = candidate.with_suffix(suffix)
                if with_suffix.is_file():
                    return with_suffix
    return None


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def prepare_image_tensor(
    image_path: Path,
    image_processor: Any,
    model: Any,
    image_aspect_ratio: str,
    quilt: dict[str, Any],
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    args = argparse.Namespace(image_aspect_ratio=image_aspect_ratio)
    try:
        image_tensor = quilt["process_images"]([image], image_processor, args)
    except Exception:
        if image_aspect_ratio == "pad":
            mean = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, mean)
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ]

    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.to(model.device, dtype=torch.float16)


def infer_one(
    row: dict[str, Any],
    image_path: Path | None,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    conv_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None,
    num_beams: int,
    image_aspect_ratio: str,
    quilt: dict[str, Any],
    strict_retry: bool = False,
) -> tuple[str, str]:
    question = build_prompt(row, strict_retry=strict_retry)
    image_tensor = None

    if image_path is not None:
        if getattr(model.config, "mm_use_im_start_end", False):
            image_token = (
                quilt["DEFAULT_IM_START_TOKEN"]
                + quilt["DEFAULT_IMAGE_TOKEN"]
                + quilt["DEFAULT_IM_END_TOKEN"]
            )
        else:
            image_token = quilt["DEFAULT_IMAGE_TOKEN"]
        question = image_token + "\n" + question
        image_tensor = prepare_image_tensor(
            image_path, image_processor, model, image_aspect_ratio, quilt
        )

    conv = quilt["conv_templates"][conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image_path is not None:
        input_ids = quilt["tokenizer_image_token"](
            prompt,
            tokenizer,
            quilt["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to(model.device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(
            model.device
        )

    stop_str = conv.sep if conv.sep_style != quilt["SeparatorStyle"].TWO else conv.sep2
    stopping_criteria = quilt["KeywordsStoppingCriteria"]([stop_str], tokenizer, input_ids)
    generate_kwargs = {
        "input_ids": input_ids,
        "do_sample": temperature > 0,
        "temperature": temperature,
        "top_p": top_p,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "stopping_criteria": [stopping_criteria],
    }
    if image_tensor is not None:
        generate_kwargs["images"] = image_tensor

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)

    input_token_len = input_ids.shape[1]
    output_text = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0].strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)].strip()
    return output_text, prompt


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
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--use-image", type=int, choices=(0, 1), default=1)
    parser.add_argument("--image-root", type=Path, default=Path("/data5/hzm/Benchmark"))
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-invalid", type=int, default=1)
    parser.add_argument("--print-raw-output", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-aspect-ratio", default="pad")
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

    quilt = import_quilt_llava(args.repo_root)
    quilt["disable_torch_init"]()
    model_name = quilt["get_model_name_from_path"](str(args.model_path))
    print(f"model_name={model_name}, conv_mode={args.conv_mode}", flush=True)
    tokenizer, model, image_processor, _ = quilt["load_pretrained_model"](
        str(args.model_path),
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    model.eval()

    rows = read_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    stats: dict[str, dict[str, dict[str, int]]] = {
        name: defaultdict(lambda: {"total": 0, "correct": 0})
        for name in ("overall", "source_dataset", "option_count")
    }

    with output_path.open("w", encoding="utf-8", buffering=1) as out:
        for row in tqdm(rows, desc=f"Quilt-LLaVA/{mode}"):
            started = time.perf_counter()
            options = row.get("options", [])
            image_path = None
            error = None
            raw_outputs: list[str] = []
            prompts: list[str] = []
            pred = None
            prompt = ""
            raw = ""
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
                        tokenizer=tokenizer,
                        model=model,
                        image_processor=image_processor,
                        conv_mode=args.conv_mode,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        image_aspect_ratio=args.image_aspect_ratio,
                        quilt=quilt,
                        strict_retry=(attempt > 0),
                    )
                    raw_outputs.append(raw)
                    prompts.append(prompt)
                    pred = parse_prediction(raw, options)
                    if pred is not None:
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raw = ""
                prompt = prompt or ""
                raw_outputs = raw_outputs or []
                if args.print_raw_output:
                    traceback.print_exc()

            gold = gold_letter(row.get("answer"))
            correct = pred is not None and pred == gold
            source_dataset = get_dataset_name(row)
            num_prefix = str(row.get("num", "")).split("-", 1)[0] or "unknown"

            record = {
                "dataset": mode,
                "source_file": str(input_path),
                "num": row.get("num"),
                "category": source_dataset,
                "num_prefix": num_prefix,
                "option_count": len(options) if isinstance(options, list) else 0,
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
                "model_id": model_name,
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
            add_stat(stats["source_dataset"], source_dataset, correct)
            add_stat(
                stats["option_count"],
                f"{len(options) if isinstance(options, list) else 0}-choice",
                correct,
            )

    summary: dict[str, Any] = {
        "overall": finalize_stats(stats["overall"])["all"],
        "by_dataset": finalize_stats(stats["source_dataset"]),
        "by_question_type": finalize_stats(stats["option_count"]),
    }

    summary_json = output_path.with_name(output_path.name + ".summary.json")
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_csv = output_path.with_name(output_path.name + ".summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "category", "total", "correct", "accuracy"])
        overall = summary["overall"]
        writer.writerow(
            ["overall", "all", overall["total"], overall["correct"], overall["accuracy"]]
        )
        for group in ("by_dataset", "by_question_type"):
            for key, stat in summary[group].items():
                writer.writerow(
                    [group, key, stat["total"], stat["correct"], stat["accuracy"]]
                )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
