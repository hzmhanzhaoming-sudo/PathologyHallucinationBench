#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model adapter: gemini. Explicit input, output and model configuration.

Use run_benchmark.py to validate and evaluate all benchmark conditions.
"""

from __future__ import annotations

from backend_io import validate_backend_args, resolve_input_image

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable, **kwargs):
        return iterable


DATASET_ORDER = ("QuiltVQA", "PathVQA", "PatchVQA", "PathMMU")

DATASET_PREFIX = {
    "100": "QuiltVQA",
    "200": "PathVQA",
    "400": "PatchVQA",
    "500": "PathMMU",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--use-image', type=int, choices=(0, 1), required=True)
    parser.add_argument('--image-root', required=True)

    parser.add_argument("--api-key", default=None, help="Default: env GEMINI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ),
        help="Gemini API base URL; defaults to GEMINI_BASE_URL.",
    )
    parser.add_argument('--model', required=True)
    parser.add_argument(
        "--thinking-level",
        "--reasoning-effort",
        dest="thinking_level",
        choices=["default", "minimal", "low", "medium", "high"],
        default="medium",
        help=(
            "Gemini thinking level. --reasoning-effort is accepted as an alias. "
            "Keep this identical across image/no-image comparisons."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=(
            "Optional numeric thinking budget for Gemini 2.5 models. "
            "When set, it takes precedence over --thinking-level."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Maximum Gemini output tokens, including any billed thinking tokens.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto", "original"],
        default="auto",
        help="Compatibility option; Gemini determines image tokenization automatically.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--retry-invalid",
        type=int,
        default=1,
        help="Retry when the response cannot be parsed as one valid option.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-raw-output", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array.")
    return data


def dataset_name(row):
    prefix = str(row.get("num", "unknown")).split("-", 1)[0]
    return DATASET_PREFIX.get(prefix, f"Unknown_{prefix}")


def resolve_image(image_value, json_path, image_root):
    return resolve_input_image(image_value, image_root)


def image_to_inline_data(image_path):
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return {"mime_type": mime, "data": encoded}


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


def extract_letter(text, options):
    valid = set(valid_letters(options))
    cleaned = str(text).strip().upper()
    patterns = [
        r"^\s*\(?([A-Z])\)?[\s\.\):\-]*$",
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])[\)\.:\-\s]",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match and match.group(1) in valid:
            return match.group(1)
    standalone = [x for x in re.findall(r"\b([A-Z])\b", cleaned) if x in valid]
    return standalone[0] if len(set(standalone)) == 1 else None


def gold_letter(answer):
    match = re.match(r"\s*([A-Za-z])", str(answer))
    return match.group(1).upper() if match else None


def make_prompt(row, strict_retry=False):
    options = row.get("options", [])
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"Invalid options for sample {row.get('num')}: {options!r}")
    option_text = "\n".join(str(option) for option in options)
    letters = valid_letters(options)
    question = str(row.get("question", "")).strip()
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


def build_gemini_contents(
    row,
    use_image,
    image_path=None,
    send_image=True,
    image_detail="auto",
    strict_retry=False,
):
    image_sent = bool(use_image and send_image)
    prompt = make_prompt(row, strict_retry=strict_retry)
    parts = [{"text": prompt}]
    if image_sent:
        if image_path is None or not Path(image_path).is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        parts.append({"inline_data": image_to_inline_data(image_path)})
    return [{"role": "user", "parts": parts}], prompt


def extract_output_text(response):
    texts = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "\n".join(texts).strip()


def gemini_generate_content_call(args, contents):
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    model_name = urllib.parse.quote(args.model, safe="")
    url = args.base_url.rstrip("/") + f"/models/{model_name}:generateContent"
    generation_config = {"maxOutputTokens": args.max_output_tokens}
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.thinking_budget is not None:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": args.thinking_budget
        }
    elif args.model.startswith("gemini-2.5"):
        # Gemini 2.5 supports thinkingBudget rather than thinkingLevel. Leave
        # the model default unchanged unless --thinking-budget is explicit.
        pass
    elif args.thinking_level != "default":
        if args.thinking_level == "minimal" and "pro" in args.model.lower():
            raise ValueError(
                "Gemini Pro does not support thinking-level=minimal; "
                "use low, medium, high, or default."
            )
        generation_config["thinkingConfig"] = {
            "thinkingLevel": args.thinking_level.upper()
        }
    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (
                extract_output_text(data),
                data.get("usageMetadata"),
                data.get("responseId"),
            )
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTPError {exc.code}: {error_text}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.retries:
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError(last_error)


def metrics(rows):
    total = len(rows)
    parsed = [
        row
        for row in rows
        if (row.get("predicted_label") or row.get("pred_letter")) is not None
    ]
    correct = sum(bool(row.get("correct")) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    return {
        "total": total,
        "parsed": len(parsed),
        "invalid": total - len(parsed),
        "errors": errors,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
        "accuracy_percent": round(correct / total * 100, 4) if total else None,
        "parsed_accuracy": correct / len(parsed) if parsed else None,
    }


def write_summary(
    output_path,
    input_path=None,
    model_id=None,
    use_image=None,
    task_name=None,
    thinking_level=None,
    thinking_budget=None,
):
    latest = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Keep the latest attempt for every sample. API/runtime errors must
            # remain in the denominator and are counted as incorrect, otherwise
            # the reported benchmark accuracy would be artificially inflated.
            if "num" in row:
                latest[str(row["num"])] = row

    rows = list(latest.values())
    prediction_counts = Counter()
    by_dataset = {name: [] for name in DATASET_ORDER}
    unknown_datasets = defaultdict(list)
    by_option_count = defaultdict(list)
    by_unknown_level = defaultdict(list)
    input_mode_counts = Counter()
    total_usage = Counter()

    for row in rows:
        predicted_label = row.get("predicted_label") or row.get("pred_letter")
        if predicted_label:
            prediction_counts[predicted_label] += 1
        category = row.get("category") or row.get("dataset") or "Unknown"
        if category in by_dataset:
            by_dataset[category].append(row)
        else:
            unknown_datasets[category].append(row)
        by_option_count[f"{len(row.get('options') or [])}-choice"].append(row)
        unknown_count = row.get("t3_unknown_count")
        if unknown_count is None:
            match = re.search(r"-unknown(\d+)$", str(row.get("num", "")))
            unknown_count = int(match.group(1)) if match else None
        if unknown_count is not None:
            by_unknown_level[f"unknown{int(unknown_count)}"].append(row)
        input_mode_counts[row.get("input_mode", "unknown")] += 1
        usage = row.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total_usage[key] += value

    summary = {
        "task": task_name,
        "model_id": model_id or (rows[0].get("model_id") if rows else None),
        "reasoning_effort": thinking_level,
        "thinking_level": thinking_level,
        "thinking_budget": thinking_budget,
        "prediction_file": str(output_path),
        "input_file": str(input_path) if input_path is not None else None,
        "use_image": use_image,
        "overall": metrics(rows),
        "by_dataset": {name: metrics(by_dataset[name]) for name in DATASET_ORDER},
        "by_option_count": {
            name: metrics(group) for name, group in sorted(by_option_count.items())
        },
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "input_mode_counts": dict(sorted(input_mode_counts.items())),
        "usage_total": dict(total_usage),
    }
    if unknown_datasets:
        summary["unknown_datasets"] = {
            name: metrics(group) for name, group in sorted(unknown_datasets.items())
        }
    if by_unknown_level:
        summary["by_unknown_level"] = {
            name: metrics(group)
            for name, group in sorted(
                by_unknown_level.items(),
                key=lambda item: int(
                    item[0][len("unknown") :]
                    if item[0].startswith("unknown")
                    else item[0]
                ),
            )
        }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, summary_path


def planned_runs(args):
    return [(Path(args.input), Path(args.output), bool(args.use_image))]


def evaluate_file(args, input_path, output_path, use_image):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    finished = set()
    pending = [row for row in rows if str(row["num"]) not in finished]
    send_image = bool(use_image)
    mode = "with_image" if use_image else "no_image"
    input_mode = (
        "with_image"
        if send_image
        else "with_image_text_only"
        if use_image
        else "no_image"
    )
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(
        f"mode={input_mode}, model={args.model}, total={len(rows)}, "
        f"finished={len(rows)-len(pending)}, pending={len(pending)}"
    )

    with open(output_path, "a", encoding="utf-8", buffering=1) as out:
        for row in tqdm(pending, desc=f"Gemini API {input_mode}"):
            sample_id = str(row["num"])
            started = time.perf_counter()
            options = row.get("options", [])
            category = dataset_name(row)
            num_prefix = sample_id.split("-", 1)[0] or "unknown"
            image_path = None
            raw_output = ""
            raw_outputs = []
            prompt = ""
            prompts = []
            usage = None
            usage_history = []
            response_id = None
            response_ids = []
            predicted_label = None
            error = None
            error_traceback = None

            result = {
                "dataset": mode,
                "source_file": str(input_path),
                "num": sample_id,
                "category": category,
                "num_prefix": num_prefix,
                "option_count": len(options) if isinstance(options, list) else 0,
                "img": row.get("img") if use_image else "",
                "input_mode": input_mode,
                "question": row.get("question"),
                "options": options,
                "gold_answer": row.get("answer"),
                "gold_label": gold_letter(row.get("answer")),
                "model_id": args.model,
                "task_name": input_path.stem,
                "reasoning_effort": args.thinking_level,
                "thinking_level": args.thinking_level,
                "thinking_budget": args.thinking_budget,
                "t3_unknown_count": row.get("t3_unknown_count"),
                "t3_variant": row.get("t3_variant"),
            }

            try:
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError(f"Invalid options: {options!r}")

                if use_image:
                    image_path = resolve_image(row.get("img"), input_path, args.image_root)
                    if send_image and (
                        image_path is None or not Path(image_path).is_file()
                    ):
                        raise FileNotFoundError(f"Image not found: {image_path}")

                for attempt in range(args.retry_invalid + 1):
                    contents, prompt = build_gemini_contents(
                        row,
                        use_image=use_image,
                        image_path=image_path,
                        send_image=send_image,
                        image_detail=args.image_detail,
                        strict_retry=attempt > 0,
                    )
                    raw_output, usage, response_id = gemini_generate_content_call(
                        args, contents
                    )
                    raw_outputs.append(raw_output)
                    prompts.append(prompt)
                    usage_history.append(usage)
                    response_ids.append(response_id)
                    predicted_label = extract_letter(raw_output, options)
                    if predicted_label is not None:
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                error_traceback = traceback.format_exc(limit=2)

            gold = gold_letter(row.get("answer"))
            correct = predicted_label is not None and predicted_label == gold
            result.update(
                resolved_image=str(image_path) if image_path else None,
                raw_output=raw_output,
                raw_outputs=raw_outputs,
                predicted_label=predicted_label,
                correct=correct,
                prompt=prompt,
                prompts=prompts,
                usage=usage,
                usage_history=usage_history,
                response_id=response_id,
                response_ids=response_ids,
                latency_seconds=round(time.perf_counter() - started, 4),
                error=error,
            )
            if error_traceback:
                result["traceback"] = error_traceback

            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if args.print_raw_output:
                print(
                    f"\n[{sample_id}] dataset={category} gold={gold} "
                    f"pred={predicted_label} correct={correct} "
                    f"raw={raw_output!r} error={error}\n",
                    flush=True,
                )

    summary, summary_path = write_summary(
        output_path,
        input_path=input_path,
        model_id=args.model,
        use_image=bool(use_image),
        task_name=input_path.stem,
        thinking_level=args.thinking_level,
        thinking_budget=args.thinking_budget,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")


def main():
    args = parse_args()
    validate_backend_args(args)
    for input_path, output_path, use_image in planned_runs(args):
        evaluate_file(args, input_path, output_path, use_image)


if __name__ == "__main__":
    main()
