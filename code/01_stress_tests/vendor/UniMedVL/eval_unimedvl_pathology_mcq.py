#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate UniMedVL on the unified pathology MCQ benchmark.

Only this evaluator file is modified. It calls UniMedVL's own eval utilities from:
  /data5/hzm/UniMedVL/codes/eval/vlm/utils.py

Default output layout mimics Benchmark_T1_results/healthgpt_pro_8B_T1_results:
  /data5/hzm/Benchmark/Benchmark_T1_results/unimedvl_T1_results/
    unimedvl_with_image_debug.jsonl
    unimedvl_with_image_debug.jsonl.summary.json
    unimedvl_text_force_debug.jsonl
    unimedvl_text_force_debug.jsonl.summary.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_REPO_ROOT = "/data5/hzm/UniMedVL"
DEFAULT_MODEL_PATH = "/data5/hzm/Model/UniMedVL"
DEFAULT_DATA_DIR = "/data5/hzm/Benchmark/Benchmark_T1_data"
DEFAULT_RESULT_DIR = "/data5/hzm/Benchmark/Benchmark_T1_results/unimedvl_T1_results"
DEFAULT_WITH_IMAGE_JSON = DEFAULT_DATA_DIR + "/pathology_vqa_benchmark_1200_with_image.json"
DEFAULT_WITHOUT_IMAGE_JSON = DEFAULT_DATA_DIR + "/pathology_vqa_benchmark_1200_empty_image.json"
DEFAULT_WITH_IMAGE_OUTPUT = DEFAULT_RESULT_DIR + "/unimedvl_with_image_debug.jsonl"
DEFAULT_WITHOUT_IMAGE_OUTPUT = DEFAULT_RESULT_DIR + "/unimedvl_text_force_debug.jsonl"


DATASET_PREFIX = {
    "100": "QuiltVQA",
    "200": "PathVQA",
    "400": "PatchVQA",
    "500": "PathMMU",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate UniMedVL on pathology multiple-choice VQA."
    )
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--checkpoint-weight-path",
        default=None,
        help="Optional fine-tuned checkpoint directory containing ema_bf16.safetensors or ema.safetensors.",
    )

    parser.add_argument(
        "--mode",
        choices=["with_image", "without_image", "both"],
        default=None,
        help=(
            "Convenience mode. If set, default benchmark inputs and HealthGPT-like "
            "output names are used unless --input/--output override them."
        ),
    )
    parser.add_argument("--output-root", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--with-image-input", default=DEFAULT_WITH_IMAGE_JSON)
    parser.add_argument("--without-image-input", default=DEFAULT_WITHOUT_IMAGE_JSON)
    parser.add_argument("--with-image-output", default=DEFAULT_WITH_IMAGE_OUTPUT)
    parser.add_argument("--without-image-output", default=DEFAULT_WITHOUT_IMAGE_OUTPUT)

    parser.add_argument("--input", default=None, help="Input benchmark JSON array.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--image-root", default=None)
    parser.add_argument(
        "--use-image",
        type=int,
        choices=[0, 1],
        default=None,
        help="Single-run mode only: 1=image + text; 0=text-only.",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--do-sample",
        type=int,
        choices=[0, 1],
        default=0,
        help="0=greedy decoding, recommended for MCQ benchmark.",
    )
    parser.add_argument(
        "--retry-invalid",
        type=int,
        default=1,
        help="Retry with stricter prompt if the first output has no legal letter.",
    )
    parser.add_argument(
        "--cpu-optimized-loading",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use UniMedVL eval utility CPU-optimized loading path.",
    )
    parser.add_argument(
        "--use-fast-chat-api",
        type=int,
        choices=[0, 1],
        default=1,
        help="Together with cpu_optimized_loading selects optimized VQA loading path.",
    )
    parser.add_argument(
        "--loader",
        choices=["auto", "simple", "multi_gpu_dispatch"],
        default="auto",
        help=(
            "Model loading backend. auto/simple use UniMedVL's public "
            "load_model_and_tokenizer(); multi_gpu_dispatch uses UniMedVL's "
            "internal multi-GPU dispatch helper without modifying source code."
        ),
    )
    parser.add_argument("--max-mem-per-gpu", default="40GiB")
    parser.add_argument("--print-raw-output", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input benchmark must be a JSON array: {path}")
    return data


def load_finished_ids(output_path: Path):
    latest = {}
    if not output_path.is_file():
        return set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                latest[str(row["num"])] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return {sample_id for sample_id, row in latest.items() if row.get("error") is None}


def get_dataset_name(row):
    prefix = str(row.get("num", "unknown")).split("-", 1)[0]
    return DATASET_PREFIX.get(prefix, f"Unknown_{prefix}")


def resolve_image(image_value, input_json_path: Path, image_root):
    if image_value is None or str(image_value).strip() == "":
        return None
    raw = str(image_value).strip()
    original = Path(raw)
    candidates = []
    if original.is_absolute():
        candidates.append(original)
    else:
        base = Path(image_root) if image_root else input_json_path.parent
        candidates.append((base / original).resolve())
    if image_root:
        root = Path(image_root)
        candidates.append(root / original.name)
        candidates.append(root / raw.lstrip("/"))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if candidate.is_file():
            return candidate
        if candidate.suffix == "":
            for suffix in (
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
            ):
                extended = candidate.with_suffix(suffix)
                if extended.is_file():
                    return extended
    return None


def valid_letters(options):
    labels = []
    for option in options or []:
        match = re.match(r"\s*([A-Za-z])\s*[\)\.:\-]", str(option))
        if match:
            label = match.group(1).upper()
            if label not in labels:
                labels.append(label)
    if not labels:
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options or [])])
    return labels


def gold_letter(answer):
    match = re.match(r"\s*([A-Za-z])", str(answer))
    return match.group(1).upper() if match else None


def build_prompt(row, use_image: bool, strict_retry: bool = False):
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


def extract_letter(raw_output, options):
    legal = set(valid_letters(options))
    text = str(raw_output).strip().upper()
    if not text:
        return None

    patterns = [
        r"^\s*\(?([A-Z])\)?\s*[\.\):\-]?\s*$",
        r"<ANSWER>\s*([A-Z])\s*</ANSWER>",
        r"(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])\)?\s*[\).:\-]",
    ]
    for pattern in patterns:
        found = [
            m.upper()
            for m in re.findall(pattern, text, flags=re.IGNORECASE)
            if m.upper() in legal
        ]
        if len(set(found)) == 1:
            return found[-1]

    standalone = [label for label in re.findall(r"\b([A-Z])\b", text) if label in legal]
    if len(set(standalone)) == 1:
        return standalone[0]
    return None


def import_unimedvl_eval(repo_root: Path):
    codes_dir = repo_root / "codes"
    utils_path = codes_dir / "eval" / "vlm" / "utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(f"Cannot find UniMedVL eval utility: {utils_path}")
    if not (codes_dir / "data" / "default.yaml").is_file():
        raise FileNotFoundError(
            f"Cannot find UniMedVL data config: {codes_dir / 'data' / 'default.yaml'}"
        )
    sys.path.insert(0, str(codes_dir))
    from eval.vlm import utils as unimedvl_eval_utils
    from eval.vlm.utils import build_transform, load_model_and_tokenizer, process_conversation

    return unimedvl_eval_utils, load_model_and_tokenizer, build_transform, process_conversation


def prepare_runtime_model_path(repo_root: Path, model_path: Path) -> Path:
    """Create a temporary symlinked model dir when config and weights are split.

    Some local installs keep llm_config/vit_config/tokenizer files under
    /data5/hzm/Model/UniMedVL, while ema.safetensors is under /data5/hzm/UniMedVL.
    UniMedVL's loader expects ema.safetensors inside args.model_path. This keeps
    source code and model files untouched by creating a small /tmp symlink view.
    """
    if (model_path / "ema.safetensors").is_file():
        return model_path

    ema_candidates = [
        model_path / "ema_bf16.safetensors",
        model_path / "ema.safetensors",
        repo_root / "ema_bf16.safetensors",
        repo_root / "ema.safetensors",
    ]
    source_ema = next((candidate for candidate in ema_candidates if candidate.is_file()), None)
    if source_ema is None:
        return model_path

    runtime_dir = Path("/tmp") / f"unimedvl_runtime_model_{os.getuid()}"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for source in model_path.iterdir():
        target = runtime_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source)

    ema_target = runtime_dir / "ema.safetensors"
    if ema_target.exists() or ema_target.is_symlink():
        try:
            if ema_target.resolve() != source_ema.resolve():
                ema_target.unlink()
                ema_target.symlink_to(source_ema)
        except FileNotFoundError:
            ema_target.unlink()
            ema_target.symlink_to(source_ema)
    else:
        ema_target.symlink_to(source_ema)

    print(f"Using runtime model path with symlinked ema.safetensors: {runtime_dir}")
    print(f"Runtime ema.safetensors -> {source_ema}")
    return runtime_dir


def load_unimedvl(args):
    repo_root = Path(args.repo_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"UniMedVL repo root not found: {repo_root}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"UniMedVL model path not found: {model_path}")
    model_path = prepare_runtime_model_path(repo_root, model_path)

    unimedvl_eval_utils, load_model_and_tokenizer, build_transform, process_conversation = import_unimedvl_eval(
        repo_root
    )

    os.chdir(str(repo_root))
    utility_args = SimpleNamespace(
        model_path=str(model_path),
        checkpoint_weight_path=args.checkpoint_weight_path,
        cpu_optimized_loading=bool(args.cpu_optimized_loading),
        use_fast_chat_api=bool(args.use_fast_chat_api),
        max_mem_per_gpu=args.max_mem_per_gpu,
    )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices or '<not set>'}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    if args.loader == "multi_gpu_dispatch":
        if not hasattr(unimedvl_eval_utils, "_load_model_cpu_optimized0"):
            raise RuntimeError("UniMedVL eval utils has no _load_model_cpu_optimized0 helper.")
        print("Loading UniMedVL via eval.vlm.utils._load_model_cpu_optimized0() ...")
        model, tokenizer, new_token_ids = unimedvl_eval_utils._load_model_cpu_optimized0(utility_args)
    else:
        print("Loading UniMedVL via eval.vlm.utils.load_model_and_tokenizer() ...")
        model, tokenizer, new_token_ids = load_model_and_tokenizer(utility_args)
    print("Building image transform via eval.vlm.utils.build_transform() ...")
    image_transform = build_transform()
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "new_token_ids": new_token_ids,
        "image_transform": image_transform,
        "process_conversation": process_conversation,
    }


def run_unimedvl_inference(bundle, prompt, image_path, max_new_tokens, do_sample, temperature):
    images = []
    if image_path is not None:
        with Image.open(image_path) as source:
            images = [source.convert("RGB").copy()]

    processed_images, conversation = bundle["process_conversation"](images, prompt)
    with torch.inference_mode():
        response = bundle["model"].chat(
            bundle["tokenizer"],
            bundle["new_token_ids"],
            bundle["image_transform"],
            images=processed_images,
            prompt=conversation,
            max_length=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
    return str(response).strip()


def metrics(rows):
    parsed = [row for row in rows if row.get("pred_letter") is not None]
    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "completed": len(rows),
        "parsed": len(parsed),
        "invalid": len(rows) - len(parsed),
        "correct": correct,
        "accuracy": correct / len(parsed) if parsed else None,
        "strict_accuracy": correct / len(rows) if rows else None,
    }


def write_summary(output_path: Path):
    latest = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "num" in row and row.get("error") is None:
                latest[str(row["num"])] = row

    rows = list(latest.values())
    by_dataset = defaultdict(list)
    by_option_count = defaultdict(list)
    prediction_counts = Counter()
    input_mode_counts = Counter()

    for row in rows:
        by_dataset[row.get("dataset", get_dataset_name(row))].append(row)
        option_count = row.get("option_count") or len(row.get("options") or [])
        by_option_count[f"{option_count}-choice"].append(row)
        input_mode_counts[row.get("input_mode", "unknown")] += 1
        if row.get("pred_letter") is not None:
            prediction_counts[row["pred_letter"]] += 1

    summary = {
        "overall": metrics(rows),
        "answer_source_counts": dict(sorted(input_mode_counts.items())),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "by_dataset": {name: metrics(group) for name, group in sorted(by_dataset.items())},
        "by_option_count": {
            name: metrics(group) for name, group in sorted(by_option_count.items())
        },
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, summary_path


def evaluate_file(args, bundle, input_path: Path, output_path: Path, use_image: bool):
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input benchmark JSON not found: {input_path}")

    rows = load_json(input_path)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative.")
        rows = rows[: args.limit]

    finished = load_finished_ids(output_path)
    pending = [row for row in rows if str(row.get("num")) not in finished]
    input_mode = "image_text" if use_image else "text_only"
    print(
        f"\n[{input_mode}] input={input_path}\n"
        f"[{input_mode}] output={output_path}\n"
        f"[{input_mode}] total={len(rows)}, finished={len(rows) - len(pending)}, "
        f"pending={len(pending)}, max_new_tokens={args.max_new_tokens}, "
        f"do_sample={bool(args.do_sample)}"
    )

    with output_path.open("a", encoding="utf-8", buffering=1) as out:
        for row in tqdm(pending, desc=f"UniMedVL {input_mode}"):
            sample_id = str(row.get("num"))
            options = row.get("options", [])
            started = time.perf_counter()
            result = {
                "num": sample_id,
                "img": row.get("img") if use_image else "",
                "input_mode": input_mode,
                "dataset": get_dataset_name(row),
                "option_count": len(options) if isinstance(options, list) else 0,
                "question": row.get("question"),
                "options": options,
                "answer": row.get("answer"),
            }
            try:
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError(f"Expected at least two options, got {options!r}")

                if use_image:
                    image_path = resolve_image(row.get("img"), input_path, args.image_root)
                    if image_path is None:
                        raise FileNotFoundError(f"Image not found: {row.get('img')}")
                else:
                    image_path = None

                raw_outputs = []
                pred = None
                for attempt in range(args.retry_invalid + 1):
                    prompt = build_prompt(row, use_image=use_image, strict_retry=(attempt > 0))
                    raw = run_unimedvl_inference(
                        bundle=bundle,
                        prompt=prompt,
                        image_path=image_path,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=bool(args.do_sample),
                        temperature=args.temperature,
                    )
                    raw_outputs.append(raw)
                    pred = extract_letter(raw, options)
                    if pred is not None:
                        break

                prediction = raw_outputs[-1] if raw_outputs else ""
                gold = gold_letter(row.get("answer"))
                result.update(
                    image_path=str(image_path) if image_path else None,
                    prediction=prediction,
                    raw_outputs=raw_outputs,
                    pred_letter=pred,
                    gold_letter=gold,
                    correct=(pred == gold if pred and gold else False),
                    latency_seconds=round(time.perf_counter() - started, 4),
                    error=None,
                )
                if args.print_raw_output:
                    print(
                        f"\n[{sample_id}] pred={pred} gold={gold} "
                        f"correct={result['correct']} raw={prediction!r}\n"
                    )
            except Exception as error:
                result.update(
                    image_path=None,
                    prediction=None,
                    raw_outputs=[],
                    pred_letter=None,
                    gold_letter=gold_letter(row.get("answer")),
                    correct=False,
                    latency_seconds=round(time.perf_counter() - started, 4),
                    error=f"{type(error).__name__}: {error}",
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary, summary_path = write_summary(output_path)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    return summary, summary_path


def planned_runs(args):
    if args.mode is None:
        if args.input is None or args.output is None:
            raise ValueError(
                "Either use --mode {with_image,without_image,both}, or provide both --input and --output."
            )
        use_image = bool(args.use_image) if args.use_image is not None else True
        return [(Path(args.input), Path(args.output), use_image)]

    output_root = Path(args.output_root)
    with_output = Path(args.output or args.with_image_output)
    without_output = Path(args.output or args.without_image_output)
    if args.output is None and args.output_root != DEFAULT_RESULT_DIR:
        with_output = output_root / "unimedvl_with_image_debug.jsonl"
        without_output = output_root / "unimedvl_text_force_debug.jsonl"

    if args.mode == "with_image":
        return [(Path(args.input or args.with_image_input), with_output, True)]
    if args.mode == "without_image":
        return [(Path(args.input or args.without_image_input), without_output, False)]
    return [
        (Path(args.with_image_input), with_output, True),
        (Path(args.without_image_input), without_output, False),
    ]


def main():
    args = parse_args()
    runs = planned_runs(args)
    bundle = load_unimedvl(args)
    for input_path, output_path, use_image in runs:
        evaluate_file(args, bundle, input_path, output_path, use_image)


if __name__ == "__main__":
    main()
