#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate OpenAI GPT on the pathology MCQ benchmark (V1–V3 and T1–T3).

The evaluator follows the same conventions as the other benchmark scripts:
  - one file supports both image and no-image modes;
  - the prompt is identical in both modes;
  - JSONL output records the complete question, raw answer and parsed label;
  - summary reports overall accuracy and all four source datasets;
  - invalid-format responses can be retried;
  - interrupted runs resume from completed sample IDs.

Unified prompt:

    Answer this pathology multiple-choice question using the question and options.
    Return only the option letter, such as A or B. Do not explain.

The script calls the OpenAI Responses API through urllib and therefore does not
require the openai Python package. Images are sent as Base64 data URLs.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable, **kwargs):
        return iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPOSITORY_ROOT / "data" / "V1"
DEFAULT_RESULT_DIR = str(REPOSITORY_ROOT / "result" / "gpt")
DEFAULT_WITH_IMAGE_JSON = str(
    DEFAULT_DATA_DIR / "v1_with_image.json"
)
DEFAULT_WITHOUT_IMAGE_JSON = str(
    DEFAULT_DATA_DIR / "v1_without_image.json"
)
DEFAULT_WITH_IMAGE_OUTPUT = str(
    Path(DEFAULT_RESULT_DIR) / "v1_with_image.jsonl"
)
DEFAULT_WITHOUT_IMAGE_OUTPUT = str(
    Path(DEFAULT_RESULT_DIR) / "v1_without_image.jsonl"
)

DATASET_ORDER = ("QuiltVQA", "PathVQA", "PatchVQA", "PathMMU")

DATASET_PREFIX = {
    "100": "QuiltVQA",
    "200": "PathVQA",
    "400": "PatchVQA",
    "500": "PathMMU",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["with_image", "without_image", "both"],
        default=None,
        help="Convenience mode using default V1 benchmark paths.",
    )
    parser.add_argument("--input", default=None, help="Single-run input JSON array.")
    parser.add_argument("--output", default=None, help="Single-run output JSONL.")
    parser.add_argument(
        "--use-image",
        type=int,
        choices=(0, 1),
        default=None,
        help="Single-run mode: 1 sends images; 0 uses question/options only.",
    )
    parser.add_argument("--with-image-input", default=DEFAULT_WITH_IMAGE_JSON)
    parser.add_argument("--without-image-input", default=DEFAULT_WITHOUT_IMAGE_JSON)
    parser.add_argument("--with-image-output", default=DEFAULT_WITH_IMAGE_OUTPUT)
    parser.add_argument("--without-image-output", default=DEFAULT_WITHOUT_IMAGE_OUTPUT)
    parser.add_argument("--output-root", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--image-root", default=None)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Single-run mode only: ignore images even if --input is with-image.",
    )
    parser.add_argument(
        "--no-send-image",
        action="store_true",
        help="For with_image mode, do not send image pixels; run blind text baseline.",
    )

    parser.add_argument("--api-key", default=None, help="Default: env OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible /v1 base URL; defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        help="Vision-capable GPT model. Override if your account uses another model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="GPT reasoning tokens share this budget; very small values may yield empty output.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto", "original"],
        default="auto",
        help="OpenAI image detail level.",
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


def load_finished_ids(path):
    if not path.exists():
        return set()
    latest = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "num" in row:
                latest[str(row["num"])] = row
    # Resume from the latest attempt, matching write_summary().  The previous
    # implementation permanently marked an ID finished if any older attempt
    # was valid, even when a later retry was invalid.
    return {
        sample_id
        for sample_id, row in latest.items()
        if not row.get("error")
        and (row.get("predicted_label") or row.get("pred_letter")) is not None
    }


def dataset_name(row):
    # T1 unknown variants may use a legacy 200- prefix for PatchVQA.
    # Prefer the preserved baseline identifier without changing the input ID.
    source_num = row.get("t1_original_num") or row.get("source_num") or row.get("num", "unknown")
    prefix = str(source_num).split("-", 1)[0]
    return DATASET_PREFIX.get(prefix, f"Unknown_{prefix}")


def resolve_image(image_value, json_path, image_root):
    if image_value is None or str(image_value).strip() == "":
        return None
    image_path = Path(str(image_value))
    if not image_path.is_absolute():
        base = Path(image_root) if image_root else json_path.parent
        image_path = (base / image_path).resolve()
    if image_path.is_file():
        return image_path
    if not image_path.suffix:
        for suffix in (
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".TIF",
            ".TIFF",
            ".WEBP",
        ):
            candidate = image_path.with_suffix(suffix)
            if candidate.is_file():
                return candidate
    return image_path


def image_to_data_url(image_path):
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def build_responses_input(
    row,
    use_image,
    image_path=None,
    send_image=True,
    image_detail="auto",
    strict_retry=False,
):
    image_sent = bool(use_image and send_image)
    prompt = make_prompt(row, strict_retry=strict_retry)
    content = [{"type": "input_text", "text": prompt}]
    if image_sent:
        if image_path is None or not Path(image_path).is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_part = {
            "type": "input_image",
            "image_url": image_to_data_url(image_path),
        }
        if image_detail:
            image_part["detail"] = image_detail
        content.append(image_part)
    return [{"role": "user", "content": content}], prompt


def extract_output_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"].strip()
    texts = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                texts.append(content["text"])
    return "\n".join(texts).strip()


def openai_responses_call(args, responses_input):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    url = args.base_url.rstrip("/") + "/responses"
    payload = {
        "model": args.model,
        "input": responses_input,
        "max_output_tokens": args.max_output_tokens,
        "store": False,
    }
    if args.temperature is not None:
        payload["temperature"] = args.temperature

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    last_error = None
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return extract_output_text(data), data.get("usage"), data.get("id")
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


def infer_task_name(input_path):
    """Read current public V1–V3/T1–T3 names; do not guess legacy numbering."""
    parts = str(input_path or "").replace("\\", "/").split("/")
    for part in reversed(parts):
        match = re.search(r"(?:^|[_.-])([VT][123])(?=[_.-]|$)", part, re.I)
        if match:
            return match.group(1).upper()
    return None


def write_summary(output_path, input_path=None, model_id=None, use_image=None):
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
        input_mode_counts[row.get("input_mode", "unknown")] += 1
        usage = row.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total_usage[key] += value

    summary = {
        "task": infer_task_name(input_path),
        "model_id": model_id or (rows[0].get("model_id") if rows else None),
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
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, summary_path


def planned_runs(args):
    if args.mode is None:
        if not args.input or not args.output:
            raise ValueError("Use --mode {with_image,without_image,both}, or pass --input and --output.")
        if args.use_image is not None:
            use_image = bool(args.use_image)
        else:
            use_image = not args.text_only
        return [(Path(args.input), Path(args.output), use_image)]

    output_root = Path(args.output_root)
    with_output = Path(args.output or args.with_image_output)
    without_output = Path(args.output or args.without_image_output)
    if args.output is None and args.output_root != DEFAULT_RESULT_DIR:
        with_output = output_root / "v1_with_image.jsonl"
        without_output = output_root / "v1_without_image.jsonl"

    if args.mode == "with_image":
        return [(Path(args.input or args.with_image_input), with_output, True)]
    if args.mode == "without_image":
        return [(Path(args.input or args.without_image_input), without_output, False)]
    return [
        (Path(args.with_image_input), with_output, True),
        (Path(args.without_image_input), without_output, False),
    ]


def evaluate_file(args, input_path, output_path, use_image):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_json(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    finished = load_finished_ids(output_path)
    pending = [row for row in rows if str(row["num"]) not in finished]
    send_image = bool(use_image and not args.no_send_image)
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
        for row in tqdm(pending, desc=f"OpenAI GPT API {input_mode}"):
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
                    responses_input, prompt = build_responses_input(
                        row,
                        use_image=use_image,
                        image_path=image_path,
                        send_image=send_image,
                        image_detail=args.image_detail,
                        strict_retry=attempt > 0,
                    )
                    raw_output, usage, response_id = openai_responses_call(
                        args, responses_input
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
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")


def main():
    args = parse_args()
    for input_path, output_path, use_image in planned_runs(args):
        evaluate_file(args, input_path, output_path, use_image)


if __name__ == "__main__":
    main()
