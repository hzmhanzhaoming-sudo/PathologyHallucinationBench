import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'run_benchmark.py'
spec = importlib.util.spec_from_file_location('public_plan', PATH)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


class PlanTests(unittest.TestCase):
    def test_eleven_conditions_9600_requests(self):
        self.assertEqual(len(plan.CONDITIONS), 11)
        self.assertEqual(sum(v[1] for v in plan.CONDITIONS.values()), 9600)
        self.assertEqual(len(plan.RUNNERS), 10)

    def test_preview_never_executes_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'new'
            args = ['run_benchmark.py', '--runner', 'eval_gpt.py', '--model', 'offline-placeholder',
                    '--output-dir', str(output)]
            stream = io.StringIO()
            with patch.object(plan.sys, 'argv', args), patch.object(plan.subprocess, 'run') as run, contextlib.redirect_stdout(stream):
                plan.main()
            run.assert_not_called()
            self.assertFalse(output.exists())
            data = json.loads(stream.getvalue())
            for entry in data['plan']:
                self.assertIn('--image-root', entry['command'])
                self.assertIn('--max-output-tokens', entry['command'])
            self.assertEqual(data['mode'], 'preview')

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ['run_benchmark.py', '--runner', 'eval_gpt.py', '--model', 'offline-placeholder', '--output-dir', tmp]
            with patch.object(plan.sys, 'argv', args), patch.object(plan.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    plan.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
