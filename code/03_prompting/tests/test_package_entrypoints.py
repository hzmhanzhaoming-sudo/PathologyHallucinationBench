"""Offline package-boundary tests: no model loads, API calls or real-result writes.

All writable fixtures are temporary. Dispatch is mocked; runner integration only
uses --dry-run or paths that fail/return before importing a backend. These tests
do not validate provider service behavior or numerical model reproducibility.
"""
import argparse
import builtins
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


PACKAGE = Path(__file__).resolve().parents[1]
SCORING = PACKAGE.parent / "02_scoring"
sys.path.insert(0, str(SCORING / "core"))
CORE = PACKAGE / "core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(PACKAGE))
sys.dont_write_bytecode = True

import common
import launcher
import registry
import runner
import score_results


def load_entry(name):
    spec = importlib.util.spec_from_file_location("package_test_" + name, (SCORING if name == "score_original" else PACKAGE) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_entry = load_entry("export_prompts")
original_entry = load_entry("score_original")


@contextlib.contextmanager
def no_backend_imports():
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "local_backends", "api_backends", "api_execution"}:
            raise AssertionError("Offline test tried importing backend: " + name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded):
        yield


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class LauncherTests(TemporaryTest):
    def dispatch(self, api, arguments, historical=False):
        with patch.object(sys, "argv", ["run_api.py" if api else "run_local.py", *arguments]), \
                patch.object(launcher.subprocess, "call", return_value=0) as call, \
                contextlib.redirect_stdout(io.StringIO()), no_backend_imports():
            with self.assertRaises(SystemExit) as caught:
                launcher.main(api=api, historical=historical)
            self.assertEqual(caught.exception.code, 0)
            return call.call_args

    def test_local_dry_run_uses_current_python_and_no_backend(self):
        result = self.dispatch(False, ["--model", "qwen35-4b", "--strategy", "zero_shot_cot", "--dry-run"])
        command = result.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn(str(PACKAGE / "execute.py"), command)
        self.assertIn("local", command)
        self.assertEqual(result.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")

    def test_new_api_default_is_shared_standard(self):
        result = self.dispatch(True, ["--model", "gpt-5.6-luna", "--strategy", "zero_shot_cot", "--dry-run"])
        command = result.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("api", command)
        self.assertNotIn("api_visible_explanation", command)
        parsed = runner.parser(True).parse_args([
            "--model", "gpt-5.6-luna", "--strategy", "zero_shot_cot"])
        self.assertEqual(parsed.prompt_profile, "standard")

    def test_new_entry_rejects_legacy_visible_profile(self):
        for api, model in ((False, "qwen35-4b"), (True, "gpt-5.6-luna")):
            with self.subTest(api=api), \
                    patch.object(sys, "argv", ["run_api.py", "--model", model, "--strategy", "zero_shot_cot",
                                                "--prompt-profile", "api_visible_explanation", "--dry-run"]), \
                    patch.object(launcher.subprocess, "call") as call, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((SystemExit, ValueError)):
                    launcher.main(api=api)
                call.assert_not_called()

    def test_historical_luna_cot_uses_recorded_visible_profile_and_separate_root(self):
        result = self.dispatch(True, ["--model", "gpt-5.6-luna", "--strategy", "zero_shot_cot", "--dry-run"], historical=True)
        command = result.args[0]
        self.assertEqual(command[command.index("--prompt-profile") + 1], "api_visible_explanation")
        self.assertEqual(command[command.index("--reasoning-effort") + 1], "none")
        self.assertEqual(command[command.index("--output-root") + 1], str(PACKAGE / "historical_reproduction_results"))
        self.assertEqual(command[command.index("--max-new-tokens") + 1], "512")
        self.assertEqual(command[command.index("--force-answer-max-tokens") + 1], "32")

    def test_other_recorded_api_protocols_remain_standard(self):
        for model, strategy, native_flag, native_value in (
            ("gpt-5.6-luna", "few_shot_cot", "--reasoning-effort", "none"),
            ("gemini-3.6-flash", "few_shot_direct", "--thinking-level", "minimal"),
            ("gemini-3.5-flash-lite", "zero_shot_cot", None, None),
        ):
            with self.subTest(model=model, strategy=strategy):
                command = self.dispatch(True, ["--model", model, "--strategy", strategy, "--dry-run"], historical=True).args[0]
                self.assertEqual(command[command.index("--prompt-profile") + 1], "standard")
                if native_flag:
                    self.assertEqual(command[command.index(native_flag) + 1], native_value)

    def test_historical_incompatible_overrides_never_dispatch(self):
        for extra in (["--prompt-profile", "standard"], ["--max-new-tokens", "1024"],
                      ["--reasoning-effort", "high"], ["--temperature", "0"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", ["run_historical_api.py", "--model", "gpt-5.6-luna",
                    "--strategy", "zero_shot_cot", "--dry-run", *extra]), patch.object(launcher.subprocess, "call") as call:
                with self.assertRaises(SystemExit):
                    launcher.main(api=True, historical=True)
                call.assert_not_called()

    def test_historical_wrapper_is_explicit_and_not_uniform_entry(self):
        with patch.object(launcher, "main") as dispatch:
            runpy.run_path(str(PACKAGE / "run_historical_api.py"), run_name="__main__")
        dispatch.assert_called_once_with(api=True, historical=True)


class RunnerBoundaryTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.input_path = self.root / "target.json"
        self.input_path.write_text(json.dumps([{
            "num": "100-test", "question": "Synthetic diagnosis?", "options": ["A) correct", "B) incorrect"],
            "answer": "A", "img": "", "explanation": "PRIVATE_TARGET_EXPLANATION"
        }]), encoding="utf-8")
        self.legacy = self.root / "legacy_source.py"
        self.legacy.write_text("# offline fixture\n", encoding="utf-8")
        self.entry = {**registry.MODELS["qwen35-4b"], "legacy": str(self.legacy)}
        self.output = self.root / "new-results"

    def invoke(self, extra=(), api=False):
        arguments = ["execute.py", "--model", "gpt-5.6-luna" if api else "qwen35-4b",
                     "--strategy", "zero_shot_cot", "--task", "baseline", "--input", str(self.input_path),
                     "--output-root", str(self.output), *extra]
        captured = io.StringIO()
        with patch.object(sys, "argv", arguments), patch.dict(runner.MODELS, {"qwen35-4b": self.entry}), \
                contextlib.redirect_stdout(captured), no_backend_imports():
            runner.main(api=api)
        return captured.getvalue()

    def preview_plan(self, api=False):
        return json.loads(self.invoke(["--dry-run"], api=api).splitlines()[0])

    def test_defaults_are_package_local_not_historical(self):
        self.assertEqual(runner.RESULTS_DIR.resolve(), (PACKAGE / "results").resolve())
        self.assertEqual(runner.DEMO_DEFAULT.resolve(), (PACKAGE / "prompts/fewshot_data/fewshot_direct.json").resolve())

    def test_local_dry_run_writes_nothing_and_is_pilot_isolated(self):
        plan = self.preview_plan()
        result = Path(plan["output"])
        self.assertTrue(result == self.output or self.output in result.parents)
        self.assertIn("zero_shot_cot", result.parts)
        self.assertIn("standard", result.parts)
        self.assertIn("pilot", result.parts)
        self.assertFalse(self.output.exists())
        self.assertEqual(plan["max_new_tokens"], 512)
        self.assertEqual(plan["n"], 1)

    def test_api_dry_run_calls_no_provider_and_does_not_leak_gold(self):
        output = self.invoke(["--dry-run", "--prompt-profile", "standard"], api=True)
        self.assertNotIn("PRIVATE_TARGET_EXPLANATION", output)
        self.assertIn("No model/API called; no result written", output)
        self.assertFalse(self.output.exists())

    def test_historical_roots_and_descendants_rejected_before_reads(self):
        fake_benchmark = self.root / "Benchmark"
        for experiment in ("Benchmark_Zero-shot CoT", "Benchmark_Few-shot direct", "Benchmark_Few-shot CoT"):
            for suffix in (Path(), Path("nested/results")):
                with self.subTest(experiment=experiment, suffix=str(suffix)):
                    self.output = fake_benchmark / experiment / "results" / suffix
                    with patch.object(runner, "ROOT", fake_benchmark), patch.object(runner, "load_rows") as load:
                        with self.assertRaisesRegex(ValueError, "Historical result"):
                            self.invoke(["--dry-run"])
                        load.assert_not_called()
                    self.assertFalse(self.output.exists())

    def test_existing_result_requires_explicit_resume(self):
        plan = self.preview_plan()
        path = Path(plan["output"])
        path.parent.mkdir(parents=True)
        sentinel = "untouched existing data\n"
        path.write_text(sentinel, encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(path.read_text(encoding="utf-8"), sentinel)

    def test_resume_rejects_mismatched_fingerprint(self):
        plan = self.preview_plan()
        path = Path(plan["output"])
        path.parent.mkdir(parents=True)
        metadata = path.with_suffix(".run.json")
        metadata.write_text(json.dumps({"fingerprint": "incompatible"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incompatible configuration"):
            self.invoke(["--resume"])
        self.assertFalse(path.exists())

    def test_complete_resume_returns_without_backend_load(self):
        plan = self.preview_plan()
        path = Path(plan["output"])
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"num": "100-test", "run_fingerprint": plan["fingerprint"]}) + "\n", encoding="utf-8")
        path.with_suffix(".run.json").write_text(json.dumps({"fingerprint": plan["fingerprint"]}), encoding="utf-8")
        self.assertIn("already completed", self.invoke(["--resume"]))


class ScoreEntryTests(TemporaryTest):
    def run_score_entry(self, arguments):
        with patch.object(sys, "argv", ["score.py", *arguments]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runpy.run_path(str(SCORING / "score.py"), run_name="__main__")

    def test_new_score_requires_explicit_inputs_and_destination(self):
        with patch.object(score_results, "main") as scorer:
            with self.assertRaises(SystemExit):
                self.run_score_entry([])
            scorer.assert_not_called()

    def test_new_score_refuses_existing_output(self):
        results = self.root / "input"
        results.mkdir()
        output = self.root / "old-statistics"
        output.mkdir()
        sentinel = output / "scores.csv"
        sentinel.write_text("old score must remain", encoding="utf-8")
        with patch.object(score_results, "main") as scorer:
            with self.assertRaises((SystemExit, ValueError, FileExistsError)):
                self.run_score_entry(["--results", str(results), "--output", str(output)])
            scorer.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "old score must remain")

    def test_new_score_accepts_equals_form_with_new_output(self):
        results = self.root / "input"
        results.mkdir()
        output = self.root / "new-statistics"
        with patch.object(score_results, "main") as scorer:
            self.run_score_entry(["--results=" + str(results), "--output=" + str(output)])
            scorer.assert_called_once()

    def test_new_score_checks_and_forwards_same_expanded_output(self):
        results = self.root / "input"
        results.mkdir()
        home = self.root / "home"
        home.mkdir()
        observed = {}

        def capture_core_arguments():
            parser = argparse.ArgumentParser()
            parser.add_argument("--output", type=Path)
            observed["output"] = parser.parse_known_args()[0].output

        with patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}), \
                patch.object(score_results, "main", side_effect=capture_core_arguments):
            self.run_score_entry(["--results", str(results), "--output", "~/future-statistics"])
        self.assertEqual(observed["output"], (home / "future-statistics").resolve())

    def test_new_score_rejects_destination_under_historical_results(self):
        results = self.root / "input"
        results.mkdir()
        for name in ("Benchmark_Zero-shot CoT", "Benchmark_Few-shot direct", "Benchmark_Few-shot CoT"):
            output = registry.ROOT / name / "results" / "statistics/test_package_entrypoints_never_create"
            with self.subTest(name=name), patch.object(score_results, "main") as scorer:
                with self.assertRaises((SystemExit, ValueError, FileExistsError)):
                    self.run_score_entry(["--results", str(results), "--output", str(output)])
                scorer.assert_not_called()

    def test_original_score_refuses_existing_output(self):
        repo = self.root / "repo"
        (repo / "data").mkdir(parents=True)
        (repo / "result").mkdir()
        output = self.root / "existing"
        output.mkdir()
        with patch.object(sys, "argv", ["score_original.py", "--repo-root", str(repo), "--output", str(output)]):
            with self.assertRaisesRegex(SystemExit, "existing outputs"):
                original_entry.main()

    def test_original_notebook_outputs_redirected_and_cwd_restored(self):
        repo = self.root / "repo"
        (repo / "data").mkdir(parents=True)
        (repo / "result").mkdir()
        old = repo / "result/six_task_robustness_scores.csv"
        old.write_text("original", encoding="utf-8")
        output = self.root / "new-original-statistics"
        fake_notebook = {"cells": [
            {"cell_type": "code", "source": ["from pathlib import Path\nSCORE_OUTPUT=Path('result/six_task_robustness_scores.csv')\nEVENT_OUTPUT=Path('result/events.csv')\nAUDIT_OUTPUT=Path('result/audit.csv')\n"]},
            {"cell_type": "code", "source": ["score_rows=[{'fixture': True}]\nSCORE_OUTPUT.write_text('new scores')\nEVENT_OUTPUT.write_text('new events')\nAUDIT_OUTPUT.write_text('new audit')\n"]},
        ]}
        before = Path.cwd()
        with patch.object(sys, "argv", ["score_original.py", "--repo-root", str(repo), "--output", str(output)]), \
                patch.object(original_entry.json, "loads", return_value=fake_notebook), \
                contextlib.redirect_stdout(io.StringIO()), no_backend_imports():
            original_entry.main()
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(old.read_text(encoding="utf-8"), "original")
        self.assertEqual((output / "six_task_robustness_scores.csv").read_text(), "new scores")
        self.assertTrue((output / "six_task_event_counts.csv").is_file())
        self.assertTrue((output / "scoring_input_audit.csv").is_file())
        report = json.loads((output / "provenance.json").read_text())
        self.assertEqual((report["models"], report["model_calls"], report["api_calls"]), (1, 0, 0))


class ExportEntryTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.input = self.root / registry.TASK_FILES["baseline"]
        self.input.parent.mkdir(parents=True)
        self.row = dict(num="100-test", question="Synthetic target?", options=["A) x", "B) y"],
                        answer="A", img="", explanation="PRIVATE_ANNOTATION")
        self.input.write_text(json.dumps([self.row]), encoding="utf-8")
        self.output = self.root / "preview"

    def invoke(self, extra=()):
        args = ["export_prompts.py", "--strategy", "zero_shot_cot", "--task", "baseline",
                "--data-root", str(self.root), "--output", str(self.output), *extra]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()), no_backend_imports():
            export_entry.main()

    def test_export_is_exact_standard_prompt_and_no_target_annotations(self):
        self.invoke()
        data = json.loads((self.output / "baseline.json").read_text())
        expected = common.make_messages(self.row, self.input, "zero_shot_cot", [])
        self.assertEqual(data["messages"], expected)
        self.assertEqual(data["message_sha256"], common.digest(expected))
        self.assertNotIn("PRIVATE_ANNOTATION", json.dumps(data))
        self.assertNotIn("gold_letter", data)
        self.assertNotIn("answer", data)

    def test_export_refuses_existing_directory(self):
        self.output.mkdir()
        with self.assertRaisesRegex(SystemExit, "not overwritten"):
            self.invoke()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_absent_requested_item_leaves_no_partial_export(self):
        with self.assertRaisesRegex(SystemExit, "Item absent"):
            self.invoke(["--item-id", "unknown"])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
