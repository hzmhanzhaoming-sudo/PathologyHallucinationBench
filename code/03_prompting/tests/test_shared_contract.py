"""Offline contract tests for the frozen shared prompting and scoring modules.

Only the Python standard library is required; no model or provider is called.
TEST_CORE_DIR may point at an archived code snapshot to regression-check it.
The default is ../core relative to this file.  Fixtures are deliberately
small synthetic cases, not claims about the benchmark's observed results.
"""
import base64
import copy
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'02_scoring/core'))
import tempfile
import unittest


CORE = Path(os.environ.get(
    "TEST_CORE_DIR", str(Path(__file__).resolve().parents[1] / "core")
)).resolve()
if not (CORE / "common.py").is_file():
    raise RuntimeError(f"Shared core not found at {CORE}; set TEST_CORE_DIR")
sys.path.insert(0, str(CORE))

import common
import score_results
import task_demos


OPTIONS = ["A) correct diagnosis", "B) another diagnosis", "C) target error", "D) other error"]
TASKS = ("baseline", "V2", "T3", "V3_with", "V3_without", "T2_present", "T2_absent")
STRATEGIES = ("zero_shot_cot", "few_shot_direct", "few_shot_cot")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def score_fixture(n=6):
    """All present-condition gold=A; missing image/answer gold=B."""
    refs = {task: {} for task in TASKS}
    bundle = {task: {} for task in TASKS}
    for task in TASKS:
        for i in range(n):
            key = str(i)
            options = OPTIONS[:]
            if task.startswith("V3"):
                options[1] = "B) Cannot see the image"
            if task.startswith("T2"):
                options[1] = "B) The correct answer is not among the listed options."
            answer = "B" if task in ("V3_without", "T2_absent") else "A"
            refs[task][key] = dict(num=key, question="Synthetic question?", options=options, answer=answer)
            bundle[task][key] = dict(pred_letter="A")
    targets = {task: {str(i): "C" for i in range(n)} for task in ("V2", "T3")}
    return refs, bundle, targets


def predictions(bundle, task, letters):
    for i, letter in enumerate(letters):
        bundle[task][str(i)]["pred_letter"] = letter


def demonstration_fixture(root):
    """Match the frozen source IDs/order and intervention prerequisites."""
    texts = [
        ["Solid nests with peripheral palisading", "Sheets of undifferentiated cells",
         "Disorganized with infiltrative growth pattern", "Papillary with fibrovascular cores"],
        ["Subepidermal blister", "Intraepidermal blister", "Suprabasal blister", "Intradermal blister"],
        ["Fungal infection", "Acute bacterial infection", "Cytomegalovirus (CMV) infection",
         "Tuberculous granulomatous disease"],
    ]
    rows, references = [], []
    for i, (opts, answer) in enumerate(zip(texts, "DAC"), 1):
        image = root / f"source-{i}.png"
        image.write_bytes(PNG)
        row = dict(num=f"FS-{i:03d}", img=str(image), question=f"Source question {i}?",
                   options=[f"{chr(65+j)}) {text}" for j, text in enumerate(opts)], answer=answer)
        rows.append(row)
        references.append(dict(num=row["num"], original_record={
            **row, "explanation": f"Synthetic annotated explanation for source {i}."}))
    path = root / "fewshot_direct.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    reference_dir = root / "reference_only"
    reference_dir.mkdir()
    (reference_dir / "source_records_with_explanations.json").write_text(
        json.dumps(references), encoding="utf-8")
    suite = task_demos.make_suite(path)
    suite_path = root / "task_specific_v3" / "demo_suite.json"
    suite_path.parent.mkdir()
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    return path, suite_path, suite


class ScoringContract(unittest.TestCase):
    def test_v2_is_anti_target_resistance_not_retained_accuracy(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "baseline", ["A", "A", "A", "A", "D", "D"])
        predictions(bundle, "V2", ["A", "C", "D", None, "A", "D"])
        result = score_results.score_task("V2", bundle, refs, targets)
        self.assertEqual((result["N"], result["O"], result["R"], result["H"], result["E"]), (6, 4, 1, 1, 2))
        self.assertEqual(result["score_percent"], 75.0)
        self.assertEqual(result["retention_percent"], 25.0)
        self.assertEqual(result["rescued_original_wrong"], 1)
        self.assertEqual(result["invalid_on_opportunities"], 1)

    def test_t3_is_retention_not_anti_target_resistance(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "baseline", ["A", "A", "A", "A", "D", "D"])
        predictions(bundle, "T3", ["A", "C", "D", None, "A", "D"])
        result = score_results.score_task("T3", bundle, refs, targets)
        self.assertEqual(result["score_percent"], 25.0)
        self.assertEqual(result["O"], result["R"] + result["H"] + result["E"])
        self.assertEqual(result["rescued_original_wrong"], 1)

    def test_v3_uses_augmented_image_present_not_base(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "baseline", ["D"] * 6)
        predictions(bundle, "V3_with", ["A", "A", "D", "D", "A", "B"])
        predictions(bundle, "V3_without", ["B", "A", "B", "B", "B", "A"])
        result = score_results.score_task("V3", bundle, refs, targets)
        self.assertEqual((result["O"], result["S"]), (3, 2))
        self.assertAlmostEqual(result["score_percent"], 200 / 3)
        self.assertAlmostEqual(result["paired_switch_success_percent"], 100 / 3)
        self.assertEqual(result["without_image_correct"], 4)
        self.assertEqual(result["false_refusal_with_image"], 1)

    def test_t2_formula_and_non_nota_errors_are_distinct(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "T2_present", ["A", "B", "C", None, "B", "A"])
        predictions(bundle, "T2_absent", ["B", "B", "B", None, "A", "C"])
        result = score_results.score_task("T2", bundle, refs, targets)
        self.assertEqual((result["O"], result["F"], result["M"]), (6, 2, 3))
        self.assertAlmostEqual(result["score_percent"], 100 * (1 - 5 / 12))
        # A wrong substantive choice (C) is not false NOTA; invalid absent output is missed NOTA.
        self.assertEqual(result["both_availability_decisions_correct"], 2)
        self.assertEqual(result["both_availability_errors"], 1)
        self.assertEqual(result["false_NOTA_only"], 1)
        self.assertEqual(result["missed_NOTA_only"], 2)
        self.assertIsNone(result["score_ci95_low"])
        self.assertIsNone(result["score_ci95_high"])

    def test_zero_opportunities_are_undefined_not_zero(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "baseline", ["D"] * 6)
        predictions(bundle, "V3_with", ["D"] * 6)
        for task in ("V2", "V3", "T2", "T3"):
            with self.subTest(task=task):
                result = score_results.score_task(task, bundle, refs, targets)
                self.assertEqual(result["O"], 0)
                self.assertFalse(result["defined"])
                self.assertIsNone(result["score_percent"])
                self.assertIsNone(result["score_ci95_low"])
                self.assertIsNone(result["score_ci95_high"])

    def test_only_paired_ids_enter_denominator(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "V2", ["C", "A", "A", "A", "A", "A"])
        del bundle["baseline"]["0"]
        del bundle["V2"]["1"]
        result = score_results.score_task("V2", bundle, refs, targets)
        self.assertEqual((result["N"], result["O"], result["H"]), (4, 4, 0))
        self.assertEqual(result["expected_N"], 6)
        self.assertAlmostEqual(result["coverage"], 4 / 6)
        self.assertEqual(result["score_percent"], 100.0)

    def test_restriction_intersects_current_protocol_correct_ids(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "baseline", ["A", "A", "A", "A", "D", "D"])
        predictions(bundle, "V2", ["A", "C", "D", None, "A", "D"])
        result = score_results.score_task("V2", bundle, refs, targets, {"0", "2", "4", "unknown"})
        self.assertEqual((result["N"], result["O"], result["R"], result["H"], result["E"]), (6, 2, 1, 0, 1))
        self.assertEqual(result["score_percent"], 100.0)

    def test_v3_restriction_still_requires_image_present_correct(self):
        refs, bundle, targets = score_fixture()
        predictions(bundle, "V3_with", ["A", "D", "A", "A", "A", "A"])
        predictions(bundle, "V3_without", ["B"] * 6)
        result = score_results.score_task("V3", bundle, refs, targets, {"0", "1"})
        self.assertEqual((result["O"], result["S"]), (1, 1))

    def test_empty_restriction_leaves_coverage_but_undefined_score(self):
        refs, bundle, targets = score_fixture()
        result = score_results.score_task("T3", bundle, refs, targets, set())
        self.assertEqual((result["N"], result["O"]), (6, 0))
        self.assertIsNone(result["score_percent"])

    def test_missing_condition_returns_no_task_result(self):
        refs, bundle, targets = score_fixture()
        for task, required in score_results.NEEDED.items():
            for condition in required:
                with self.subTest(task=task, condition=condition):
                    shortened = {key: value for key, value in bundle.items() if key != condition}
                    self.assertIsNone(score_results.score_task(task, shortened, refs, targets))

    def test_no_common_ids_returns_no_task_result(self):
        refs, bundle, targets = score_fixture()
        bundle["V2"] = {}
        self.assertIsNone(score_results.score_task("V2", bundle, refs, targets))

    def test_scoring_does_not_mutate_inputs(self):
        refs, bundle, targets = score_fixture()
        before = copy.deepcopy((refs, bundle, targets))
        for task in score_results.NEEDED:
            score_results.score_task(task, bundle, refs, targets)
        self.assertEqual((refs, bundle, targets), before)


class ParsingContract(unittest.TestCase):
    def test_explicit_and_exact_answers_parse(self):
        cases = {"A": "A", "(B).": "B", OPTIONS[2]: "C",
                 "Reasoning mentions A and C.\n<answer>B</answer>": "B",
                 "Reasoning\nFinal answer: D": "D", "<answer>a</answer>": "A"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(common.parse_answer(raw, OPTIONS), expected)

    def test_incidental_letters_and_ambiguous_prose_are_invalid(self):
        for raw in ("A or B", "Consider A, but B could fit.", "The answer is probably A.", "", None):
            with self.subTest(raw=raw):
                self.assertIsNone(common.parse_answer(raw, OPTIONS))

    def test_conflicting_tagged_finals_are_invalid(self):
        self.assertIsNone(common.parse_answer("<answer>A</answer><answer>B</answer>", OPTIONS))
        self.assertEqual(common.parse_answer("<answer>B</answer><answer>B</answer>", OPTIONS), "B")

    @unittest.expectedFailure
    def test_known_legacy_boundary_multiline_final_conflict(self):
        # Known historical boundary: the untagged answer regex is not multiline,
        # so only the final line is recognized.  Do not change the frozen parser
        # silently: doing so would change historical experiment parsing semantics.
        self.assertIsNone(common.parse_answer("Answer: A\nFinal answer: B", OPTIONS))

    def test_legacy_multiline_final_precedence_is_frozen(self):
        # All three prompting strategies must reuse this exact same behavior;
        # this test documents compatibility, not a claim that every conflict is rejected.
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                self.assertEqual(common.parse_answer("Answer: A\nFinal answer: B", OPTIONS), "B")

    def test_illegal_option_is_invalid(self):
        for raw in ("H", "<answer>H</answer>", "Final answer: H"):
            with self.subTest(raw=raw):
                self.assertIsNone(common.parse_answer(raw, OPTIONS))

    def test_hidden_thinking_does_not_supply_the_answer(self):
        self.assertEqual(common.parse_answer("<think><answer>A</answer></think><answer>D</answer>", OPTIONS), "D")
        self.assertIsNone(common.parse_answer("<think>Still considering A", OPTIONS))
        self.assertIsNone(common.parse_answer("<think><answer>B</answer></think>", OPTIONS))


class PromptContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.suite_path, self.suite = demonstration_fixture(self.root)
        target_image = self.root / "target.png"
        target_image.write_bytes(PNG)
        self.row = dict(num="target", img=str(target_image), question="Target diagnosis?",
                        options=OPTIONS[:], answer="A", explanation="SECRET_TARGET_EXPLANATION",
                        target_wrong_option="SECRET_TARGET_LABEL", metadata="SECRET_METADATA")

    def test_three_strategies_preserve_same_target_input(self):
        signatures = []
        for strategy in STRATEGIES:
            demos, _ = task_demos.load_task_demos(strategy, self.path, "V2")
            messages = common.make_messages(self.row, self.path, strategy, demos)
            self.assertEqual(len(messages), 1 if strategy.startswith("zero") else 7)
            target = messages[-1]
            self.assertEqual(target["role"], "user")
            self.assertEqual(target["content"][0], {"type": "text", "text": "Target question"})
            input_text = target["content"][-1]["text"].split("\nQuestion: ", 1)[1]
            images = [part for part in target["content"] if part["type"] == "image"]
            signatures.append((input_text, images))
            if strategy.endswith("cot"):
                self.assertIn("Think step by step", target["content"][-1]["text"])
            else:
                self.assertIn("Do not explain", target["content"][-1]["text"])
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_gold_and_private_target_annotations_never_leak(self):
        for strategy in STRATEGIES:
            for task in task_demos.FAMILY:
                with self.subTest(strategy=strategy, task=task):
                    demos, _ = task_demos.load_task_demos(strategy, self.path, task)
                    original = copy.deepcopy(self.row)
                    first = common.make_messages(self.row, self.path, strategy, demos)
                    changed = {**self.row, "answer": "D", "explanation": "OTHER_SECRET"}
                    second = common.make_messages(changed, self.path, strategy, demos)
                    self.assertEqual(first, second)
                    self.assertNotIn("SECRET", json.dumps(first))
                    self.assertEqual(self.row, original)

    def test_target_without_image_does_not_reuse_example_image(self):
        row = {**self.row, "img": ""}
        for strategy in STRATEGIES:
            demos, _ = task_demos.load_task_demos(strategy, self.path, "V3_without")
            messages = common.make_messages(row, self.path, strategy, demos)
            self.assertFalse(any(part["type"] == "image" for part in messages[-1]["content"]))

    def test_invalid_strategy_rejected(self):
        with self.assertRaises(ValueError):
            common.make_messages(self.row, self.path, "not_a_strategy", [])

    def test_v3_paired_conditions_share_identical_demonstrations(self):
        for strategy in ("few_shot_direct", "few_shot_cot"):
            present, pp = task_demos.load_task_demos(strategy, self.path, "V3_with")
            absent, pa = task_demos.load_task_demos(strategy, self.path, "V3_without")
            self.assertEqual(present, absent)
            self.assertEqual(pp["content"], pa["content"])
            self.assertEqual(pp["pair_count"], 1)
            self.assertEqual(len(present), 2)
            self.assertIsNotNone(present[0]["image"])
            self.assertIsNone(present[1]["image"])
            self.assertEqual(present[0]["options"], present[1]["options"])
            self.assertNotEqual(common.parse_answer(present[0]["assistant"], present[0]["options"]),
                                common.parse_answer(present[1]["assistant"], present[1]["options"]))

    def test_t2_paired_conditions_share_inputs_with_correct_nota_semantics(self):
        for strategy in ("few_shot_direct", "few_shot_cot"):
            present, pp = task_demos.load_task_demos(strategy, self.path, "T2_present")
            absent, pa = task_demos.load_task_demos(strategy, self.path, "T2_absent")
            self.assertEqual(present, absent)
            self.assertEqual(pp["content"], pa["content"])
            self.assertEqual(pp["variants"], ["answer_present", "answer_absent"])
            for demo, is_absent in zip(present, (False, True)):
                letter = common.parse_answer(demo["assistant"], demo["options"])
                answer_text = next(option for option in demo["options"] if option.startswith(letter + ")"))
                self.assertEqual(task_demos.NOTA in answer_text, is_absent)
                self.assertIsNotNone(demo["image"])

    def test_fewshot_direct_and_cot_share_demonstration_questions_and_answers(self):
        for task in task_demos.FAMILY:
            direct, pd = task_demos.load_task_demos("few_shot_direct", self.path, task)
            cot, pc = task_demos.load_task_demos("few_shot_cot", self.path, task)
            self.assertEqual(pd["input_content"], pc["input_content"])
            for left, right in zip(direct, cot):
                self.assertEqual({k: v for k, v in left.items() if k != "assistant"},
                                 {k: v for k, v in right.items() if k != "assistant"})
                self.assertEqual(left["assistant"], common.parse_answer(right["assistant"], right["options"]))

    def test_zero_shot_needs_no_demonstration_files(self):
        for task in task_demos.FAMILY:
            self.assertEqual(task_demos.load_task_demos("zero_shot_cot", self.root / "missing.json", task), ([], {}))

    def test_modified_demonstration_suite_is_rejected(self):
        self.suite["banks"]["V3_pairs"][1]["answer"] = "A"
        self.suite_path.write_text(json.dumps(self.suite), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "suite/source mismatch"):
            task_demos.load_task_demos("few_shot_cot", self.path, "V3_without")


if __name__ == "__main__":
    unittest.main(verbosity=2)
