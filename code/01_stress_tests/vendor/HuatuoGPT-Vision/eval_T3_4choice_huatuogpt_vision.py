#!/usr/bin/env python3
"""Evaluate HuatuoGPT-Vision on the T1 pathology MCQ benchmark.

Intended server location:
    /data5/hzm/HuatuoGPT-Vision/eval_huatuogpt_vision_pathology_mcq.py

The prompt is intentionally identical for image and no-image modes:
    Answer this pathology multiple-choice question using the question and options.
    Return only the option letter, such as A or B. Do not explain.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


DEFAULT_REPO_ROOT = Path("/data5/hzm/HuatuoGPT-Vision")
DEFAULT_MODEL_PATH = Path("/data5/hzm/Model/HuatuoGPT-Vision-7B")
DEFAULT_INPUT_WITH_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_data/4-choice-test/pathology_vqa_benchmark_1500_with_image.json"
)
DEFAULT_INPUT_NO_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_data/4-choice-test/pathology_vqa_benchmark_1500_empty_image.json"
)
DEFAULT_OUTPUT_WITH_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_results/4-choice/huatuogpt_vision_T3_results/huatuogpt_vision_T3_4choice_with_image.jsonl"
)
DEFAULT_OUTPUT_NO_IMAGE = Path(
    "/data5/hzm/Benchmark/Benchmark_T3_results/4-choice/huatuogpt_vision_T3_results/huatuogpt_vision_T3_4choice_no_image.jsonl"
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


def import_huatuo(repo_root: Path) -> Any:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from cli import HuatuoChatbot  # type: ignore

    return HuatuoChatbot


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


def extract_new_token_ids(output_ids: torch.Tensor, input_ids: torch.Tensor, tokenizer: Any) -> list[int]:
    output_list = [int(token_id) for token_id in output_ids[0].tolist()]
    input_list = [int(token_id) for token_id in input_ids[0].tolist()]

    if len(output_list) >= len(input_list) and output_list[: len(input_list)] == input_list:
        generated = output_list[len(input_list) :]
    else:
        generated = list(output_list)
        if (
            generated
            and tokenizer.bos_token_id is not None
            and generated[0] == tokenizer.bos_token_id
        ):
            generated = generated[1:]

    special_ids = {
        token_id
        for token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
        if token_id is not None
    }
    return [token_id for token_id in generated if token_id not in special_ids]


def decode_new_text(output_ids: torch.Tensor, input_ids: torch.Tensor, tokenizer: Any) -> tuple[str, list[int]]:
    new_token_ids = extract_new_token_ids(output_ids, input_ids, tokenizer)
    raw_output = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return raw_output, new_token_ids


def patch_huatuo_forward_cache_position(model: Any) -> None:
    """Make old HuatuoGPT-Vision forward compatible with newer Transformers.

    Newer transformers.generate() may pass cache_position into model.forward().
    This HuatuoGPT-Vision fork's LlavaQwen2ForCausalLM.forward() does not accept
    that argument, causing:
      TypeError: ... got an unexpected keyword argument 'cache_position'

    Dropping only cache_position preserves the repository's original generation
    path used by eval_huatuogpt_vision_simple.py.
    """
    original_forward = model.forward

    def forward_without_cache_position(
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.Tensor | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("cache_position", None)
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )

    model.forward = forward_without_cache_position


def patch_huatuo_generate_attention_mask(model: Any) -> None:
    """Patch generate() so attention_mask is not overwritten with None."""

    def generate_with_attention_mask(
        self: Any,
        inputs: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if attention_mask is None and inputs is not None:
            attention_mask = torch.ones_like(inputs, dtype=torch.long, device=inputs.device)

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal_new(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super(type(self), self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    model.generate = types.MethodType(torch.no_grad()(generate_with_attention_mask), model)


def infer_one(
    row: dict[str, Any],
    image_path: Path | None,
    bot: Any,
    max_new_tokens: int,
    temperature: float,
    strict_retry: bool = False,
) -> tuple[str, str, list[int]]:
    prompt = build_prompt(row, strict_retry=strict_retry)
    text = bot.input_moderation(prompt)
    if image_path is not None:
        text = bot.insert_image_placeholder(text, num_images=1)

    conversation = bot.get_conv_without_history(text)
    input_ids = bot.preprocess(conversation, return_tensors="pt").unsqueeze(0).to(bot.device)

    image_tensors = None
    if image_path is not None:
        image_tensor_list = bot.get_image_tensors([str(image_path)])
        image_tensors = torch.stack(image_tensor_list).to(dtype=torch.bfloat16, device=bot.device)

    generation_kwargs = {
        "do_sample": temperature > 0,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0,
        "use_cache": True,
        "eos_token_id": bot.tokenizer.eos_token_id,
        "pad_token_id": bot.tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = bot.model.generate(
            input_ids,
            images=image_tensors,
            **generation_kwargs,
        )

    raw_output, generated_token_ids = decode_new_text(output_ids, input_ids, bot.tokenizer)
    return raw_output, text, generated_token_ids


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
    parser.add_argument("--device", default="cuda")
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

    HuatuoChatbot = import_huatuo(args.repo_root)
    bot = HuatuoChatbot(model_dir=str(args.model_path), device=args.device)
    bot.model.eval()
    patch_huatuo_forward_cache_position(bot.model)
    patch_huatuo_generate_attention_mask(bot.model)
    if bot.tokenizer.pad_token_id is None:
        bot.tokenizer.pad_token_id = bot.tokenizer.eos_token_id

    rows = read_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    stats: dict[str, dict[str, dict[str, int]]] = {
        name: defaultdict(lambda: {"total": 0, "correct": 0})
        for name in ("overall", "unknown_level")
    }

    with output_path.open("w", encoding="utf-8", buffering=1) as out:
        for row in tqdm(rows, desc=f"HuatuoGPT-Vision/{mode}"):
            started = time.perf_counter()
            options = row.get("options", [])
            image_path = None
            error = None
            raw_outputs: list[str] = []
            generated_token_ids_all: list[list[int]] = []
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
                    raw, prompt, generated_token_ids = infer_one(
                        row=row,
                        image_path=image_path,
                        bot=bot,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        strict_retry=(attempt > 0),
                    )
                    raw_outputs.append(raw)
                    generated_token_ids_all.append(generated_token_ids)
                    prompts.append(prompt)
                    pred = parse_prediction(raw, options)
                    if pred is not None:
                        break
            except torch.cuda.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                error = f"CUDA OOM: {exc!r}"
                if args.print_raw_output:
                    traceback.print_exc()
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
                "generated_token_ids": generated_token_ids_all[-1] if generated_token_ids_all else [],
                "generated_token_ids_all": generated_token_ids_all,
                "predicted_label": pred,
                "correct": correct,
                "prompt": prompt,
                "prompts": prompts,
                "model_id": str(args.model_path),
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
