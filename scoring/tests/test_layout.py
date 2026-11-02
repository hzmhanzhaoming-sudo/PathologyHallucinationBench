"""Check relocated entry points without model calls or external data."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_public_scoring import synthetic_run

ROOT = Path(__file__).resolve().parents[2]


class LayoutTests(unittest.TestCase):
    def test_public_cli_scores_to_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)/'fixture';fixture.mkdir()
            synthetic_run(fixture)
            output = Path(tmp)/'scores'
            command = [sys.executable,'-B',str(ROOT/'scoring/score.py'),
                       '--repo-root',str(fixture),'--output',str(output)]
            result = subprocess.run(command,capture_output=True,text=True,encoding='utf8')
            self.assertEqual(result.returncode,0,result.stderr)
            with (output/'six_task_robustness_scores.csv').open(encoding='utf-8-sig') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row['V3']),100)
            self.assertNotIn('overall_clinician_weighted',row)
            result = subprocess.run(command,capture_output=True,text=True,encoding='utf8')
            self.assertNotEqual(result.returncode,0)

    def test_manuscript_scoring_has_no_prompt_package_dependency(self):
        program = '''
import sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'scoring'))
from score_stress_run import load_shared
common,score=load_shared(root/'scoring/core')
ref={'num':'synthetic','question':'Synthetic?','options':['A) gold','B) wrong'],'answer':'A'}
refs={'V2':{'synthetic':ref},'baseline':{'synthetic':ref}}
bundle={'baseline':{'synthetic':{'pred_letter':'A'}},'V2':{'synthetic':{'pred_letter':'B'}}}
assert score.score_task('V2',bundle,refs,{'V2':{'synthetic':'B'}})['score_percent']==0
assert common.gold(ref)=='A'
assert Path(common.__file__).resolve().parent == (root/'scoring/core').resolve()
assert not any('prompt' in name for name in sys.modules)
'''
        result = subprocess.run([sys.executable,'-B','-c',program,str(ROOT)],
                                capture_output=True,text=True,encoding='utf8')
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
