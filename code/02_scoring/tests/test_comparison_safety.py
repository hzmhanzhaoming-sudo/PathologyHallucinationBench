"""Synthetic, offline path-guard tests; no model/API calls or real result writes."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PACKAGE = Path(__file__).resolve().parents[1]
CODE = PACKAGE.parent
sys.path.insert(0, str(CODE / '03_prompting' / 'core'))
sys.path.insert(0, str(PACKAGE / 'core'))
spec = importlib.util.spec_from_file_location('safe_comparison_entry', PACKAGE / 'compare_strategies.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
import registry
import runner


class ComparisonSafety(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.results = [self.root / 'input-a', self.root / 'input-b']
        for folder in self.results:
            folder.mkdir()
        self.output = self.root / 'new-report'
        self.data_root = self.root / 'custom-benchmark'
        self.default_root = self.root / 'default-benchmark'

    def invoke(self, extra=(), roots=None, output=True):
        args = ['compare_strategies.py', '--roots', *map(str, roots or self.results),
                '--data-root', str(self.data_root)]
        if output:
            args.extend(['--output', str(self.output)])
        args.extend(extra)
        self.captured = {}
        def fake_main():
            self.captured['argv'] = list(sys.argv)
        core = SimpleNamespace(main=Mock(side_effect=fake_main))
        with patch.object(sys, 'argv', args), patch.object(registry, 'ROOT', self.default_root), \
                patch.object(entry, '_load_comparison', return_value=core) as loader, \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                entry.main()
            except BaseException:
                loader.assert_not_called()
                raise
            self.assertIs(sys.argv, args)
        return core

    def test_explicit_new_output_is_required(self):
        with self.assertRaises(SystemExit):
            self.invoke(output=False)
        self.assertFalse(self.output.exists())

    def test_two_distinct_input_directories_required(self):
        for roots in ([self.results[0]], [self.results[0], self.results[0]],
                      [self.results[0], self.root / 'absent-input']):
            with self.subTest(roots=roots), self.assertRaises(SystemExit):
                self.invoke(roots=roots)

    def test_existing_output_unchanged(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('unchanged', encoding='utf8')
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(sentinel.read_text(encoding='utf8'), 'unchanged')

    def test_output_cannot_be_inside_input(self):
        self.output = self.results[0] / 'new-comparison'
        with self.assertRaisesRegex(SystemExit, 'read-only'):
            self.invoke()
        self.assertFalse(self.output.exists())

    def test_all_old_result_roots_protected_before_core_import(self):
        for benchmark in (self.default_root, self.data_root):
            for subpath in [f'Benchmark_T{i}_results' for i in range(1, 8)] + [
                    'Benchmark_Zero-shot CoT/results', 'Benchmark_Few-shot direct/results',
                    'Benchmark_Few-shot CoT/results']:
                for suffix in ('', 'nested/new-report'):
                    with self.subTest(benchmark=benchmark, subpath=subpath, suffix=suffix):
                        self.output = benchmark / subpath / suffix
                        with self.assertRaisesRegex(SystemExit, 'read-only'):
                            self.invoke()
                        self.assertFalse(self.output.exists())

    def test_safe_arguments_forwarded_without_changing_protocol(self):
        core = self.invoke(['--allow-task-demo-comparison'])
        core.main.assert_called_once_with()
        args = self.captured['argv']
        self.assertEqual(args[args.index('--output') + 1], str(self.output.resolve()))
        self.assertEqual(args[args.index('--data-root') + 1], str(self.data_root.resolve()))
        self.assertIn('--allow-task-demo-comparison', args)
        self.assertFalse(self.output.exists())

    def test_help_does_not_load_comparison_core(self):
        with patch.object(sys, 'argv', ['compare_strategies.py', '--help']), \
                patch.object(entry, '_load_comparison') as loader, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                entry.main()
            self.assertEqual(caught.exception.code, 0)
            loader.assert_not_called()


class PromptingResultProtection(unittest.TestCase):
    def test_original_result_roots_under_default_and_custom_data_roots_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = root / 'default-benchmark'
            custom = root / 'custom-benchmark'
            for benchmark in (default, custom):
                for task in range(1, 8):
                    for suffix in ('', 'nested/new-run'):
                        output = benchmark / f'Benchmark_T{task}_results' / suffix
                        args = ['execute.py', '--model', 'qwen35-4b', '--strategy', 'zero_shot_cot',
                                '--task', 'baseline', '--data-root', str(custom),
                                '--output-root', str(output), '--dry-run']
                        with self.subTest(benchmark=benchmark, task=task, suffix=suffix), \
                                patch.object(sys, 'argv', args), patch.object(runner, 'ROOT', default), \
                                patch.object(runner, 'validate_suite') as validate, \
                                patch.object(runner, 'load_rows') as load:
                            with self.assertRaisesRegex(ValueError, 'Historical result'):
                                runner.main(api=False)
                            validate.assert_not_called()
                            load.assert_not_called()
                            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
