"""CPU-only tests: no checkpoint imports, network requests, or prediction writes."""

from __future__ import annotations

import ast
import json
import math
import re
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = sorted((ROOT / "evaluation").glob("eval_*.py"))
NOTEBOOK = ROOT / "result" / "score_six_hallucination_tasks.ipynb"


def runner_metadata(path):
    """Execute only path/constants and pure metadata helpers, never model code."""
    namespace = {"Path": Path, "Any": Any, "re": re, "__file__": str(path)}
    module = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name.startswith("DEFAULT_") or name in {
                "PROJECT_ROOT", "REPOSITORY_ROOT", "DATASET_PREFIX", "NUM_PREFIX_TO_DATASET"
            } for name in names):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "get_dataset_name", "dataset_name", "infer_task_name"
        }:
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def scoring_functions():
    """Load scoring definitions without running notebook discovery or exports."""
    namespace = {
        "Path": Path, "Any": Any, "re": re, "json": json, "math": math,
        "defaultdict": defaultdict, "INPUT_INDEX": {}, "INPUT_CATALOG": {},
    }
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    selected = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"notebook-cell-{i}", "exec")
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef):
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {
                    "DATASET_WEIGHTS", "NUM_PREFIX_TO_DATASET", "LABEL_RE", "NOTA_TEXT"
                } for target in node.targets
            ):
                selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class ReleaseConsistencyTests(unittest.TestCase):
    def test_ten_runners_compile_and_use_public_v1_defaults(self):
        self.assertEqual(len(RUNNERS), 10)
        for path in RUNNERS:
            with self.subTest(runner=path.name):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
                values = runner_metadata(path)
                self.assertEqual(Path(values["DEFAULT_DATA_DIR"]), ROOT / "data" / "V1")
                image_input = values.get("DEFAULT_INPUT_WITH_IMAGE", values.get("DEFAULT_WITH_IMAGE_JSON"))
                text_input = values.get("DEFAULT_INPUT_NO_IMAGE", values.get("DEFAULT_WITHOUT_IMAGE_JSON"))
                self.assertEqual(Path(image_input).name, "v1_with_image.json")
                self.assertEqual(Path(text_input).name, "v1_without_image.json")
                output = values.get("DEFAULT_OUTPUT_DIR", values.get("DEFAULT_RESULT_DIR"))
                self.assertEqual(Path(output).parent, ROOT / "result")
                image_output = values.get("DEFAULT_OUTPUT_WITH_IMAGE", values.get("DEFAULT_WITH_IMAGE_OUTPUT"))
                text_output = values.get("DEFAULT_OUTPUT_NO_IMAGE", values.get("DEFAULT_WITHOUT_IMAGE_OUTPUT"))
                self.assertEqual(Path(image_output).name, "v1_with_image.jsonl")
                self.assertEqual(Path(text_output).name, "v1_without_image.jsonl")

    def test_t1_variants_use_original_dataset_not_legacy_200_prefix(self):
        for path in RUNNERS:
            with self.subTest(runner=path.name):
                namespace = runner_metadata(path)
                classify = namespace.get("get_dataset_name", namespace.get("dataset_name"))
                self.assertEqual(classify({"num": "200-4-unknown1", "t1_original_num": "400-4"}), "PatchVQA")
                self.assertEqual(classify({"num": "200-4"}), "PathVQA")
                self.assertEqual(classify({"num": "500-1-unknown2", "t1_original_num": "500-1"}), "PathMMU")

    def test_api_summary_task_inference(self):
        for filename in ("eval_gpt.py", "eval_gemini.py"):
            classify = runner_metadata(ROOT / "evaluation" / filename)["infer_task_name"]
            for task in ("V1", "V2", "V3", "T1", "T2", "T3"):
                self.assertEqual(classify(f"data/{task}/{task.lower()}_with_image.json"), task)
                self.assertEqual(classify(f"C:\\bench\\{task}\\input.json"), task)
            self.assertIsNone(classify("unlabelled/input.json"))

    def test_v3_h_label_and_missing_prediction(self):
        scoring = scoring_functions()
        for label in ("H", "H) Cannot see the image", "<answer>H</answer>"):
            self.assertEqual(scoring["norm_label"](label), "H")
        baseline = {"400-1": {"correct": True}}
        no_image = {"400-1": {"predicted_label": "H", "v3_abstain_label": "H"}}
        bundle = {"V3_with_image": baseline, "V3_without_image": no_image}
        self.assertEqual(scoring["score_v3"]("test", bundle)["score"], 100.0)
        no_image["400-1"]["predicted_label"] = None
        self.assertEqual(scoring["score_v3"]("test", bundle)["score"], 0.0)
        no_image["400-1"].pop("v3_abstain_label")
        with self.assertRaisesRegex(ValueError, "cannot-see label"):
            scoring["score_v3"]("test", bundle)

    def test_six_score_formulas_on_small_paired_fixture(self):
        scoring = scoring_functions()
        options = ["A) gold", "B) wrong", "C) other", "D) another"]
        baseline = {
            num: {"num": num, "options": options, "correct": i < 2}
            for i, num in enumerate(("500-1", "500-2", "500-3", "500-4"))
        }
        no_image = {num: {"correct": False} for num in baseline}
        bundle = {"V1_with_image": baseline, "V1_without_image": no_image}
        self.assertEqual(scoring["score_v1"]("test", bundle)["score"], 100.0)
        scoring["V2_TARGETS"] = {num: "B" for num in baseline}
        bundle["V2"] = {num: {"predicted_label": "B", "correct": False} for num in baseline}
        self.assertEqual(scoring["score_v2"]("test", bundle)["score"], 0.0)
        bundle["V3_with_image"] = baseline
        bundle["V3_without_image"] = {
            num: {"predicted_label": "E", "v3_abstain_label": "E"} for num in baseline
        }
        self.assertEqual(scoring["score_v3"]("test", bundle)["score"], 100.0)
        bundle["T1_with_image"] = {
            f"{num}-unknown{u}": {**row, "num": f"{num}-unknown{u}", "t1_original_num": num, "t1_unknown_count": u}
            for num, row in baseline.items() for u in (1, 2)
        }
        self.assertEqual(scoring["score_t1"]("test", bundle)["score"], 100.0)
        nota_options = ["A) gold", "B) The correct answer is not among the listed options."]
        bundle["T2_present"] = {num: {"predicted_label": "A", "options": nota_options} for num in baseline}
        bundle["T2_absent"] = {num: {"predicted_label": "B", "options": nota_options} for num in baseline}
        self.assertEqual(scoring["score_t2"]("test", bundle)["score"], 100.0)
        bundle["T3"] = baseline
        self.assertEqual(scoring["score_t3"]("test", bundle)["score"], 100.0)

    def test_public_paths_classify_as_current_tasks(self):
        classify = scoring_functions()["classify_task"]
        expected = {
            "data/V1/v1_with_image.json": "V1_with_image",
            "data/V1/v1_without_image.json": "V1_without_image",
            "data/V2/v2_image_text_conflict_with_image.json": "V2",
            "data/V3/v3_with_image.json": "V3_with_image",
            "data/V3/v3_without_image.json": "V3_without_image",
            "data/T1/7-choice/t1_7choice_unknown1_2_with_image.json": "T1_with_image",
            "data/T2/t2_answer_present_with_image.json": "T2_present",
            "data/T2/t2_answer_absent_with_image.json": "T2_absent",
            "data/T3/t3_wrong_answer_hint_with_image.json": "T3",
        }
        for path, condition in expected.items():
            self.assertEqual(classify(path), condition)

    def test_example_has_v2_target_and_current_task_labels(self):
        rows = json.loads((ROOT / "example" / "V1-V3_T1-T3_single_question.json").read_text(encoding="utf-8"))
        self.assertEqual({row["task"] for row in rows}, {"V1", "V2", "V3", "T1", "T2", "T3"})
        scoring = scoring_functions()
        for row in rows:
            if row["task"] == "V2":
                target = scoring["norm_label"](row["target_wrong_option"])
                self.assertIn(target, [scoring["norm_label"](option) for option in row["options"]])
                self.assertNotEqual(target, scoring["norm_label"](row["answer"]))


if __name__ == "__main__":
    unittest.main()
