"""Synthetic-only checks; never import a model, contact a provider or read real results."""
import ast
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unittest
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/'scoring/score_six_hallucination_tasks.ipynb'


def functions():
    cells = json.loads(NOTEBOOK.read_text(encoding='utf8'))['cells']
    ns = dict(re=re, json=json, Path=Path, math=math, defaultdict=defaultdict,
              NOTA_TEXT='the correct answer is not among the listed options', ALLOW_PARTIAL=False)
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            compile(source, str(NOTEBOOK), 'exec')
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
    return ns


def synthetic_run(root):
    ids = ['100-fixture', '200-fixture', '500-fixture', '400-fixture']
    options = ['A) correct', 'B) other', 'C) target', 'D) alternative']
    files = {
        'V1_with_image': 'V1/v1_with_image', 'V1_without_image': 'V1/v1_without_image',
        'V2': 'V2/v2_image_text_conflict_with_image',
        'V3_with_image': 'V3/v3_with_image', 'V3_without_image': 'V3/v3_without_image',
        'T1_with_image': 'T1/t1_unknown1_2_with_image',
        'T2_present': 'T2/t2_answer_present_with_image',
        'T2_absent': 'T2/t2_answer_absent_with_image',
        'T3': 'T3/t3_wrong_answer_hint_with_image',
    }
    for condition, relative in files.items():
        refs, predictions = [], []
        for i, num in enumerate(ids):
            for level in ((1, 2) if condition.startswith('T1') else (None,)):
                row = dict(num=num, question='Synthetic?', options=list(options), answer='A', img='synthetic.png')
                pred = 'A' if i % 2 == 0 else 'B'
                if condition == 'V1_without_image':
                    row['img'] = ''; pred = 'B'
                elif condition == 'V2':
                    row['target_wrong_option'] = 'C'; pred = 'C'
                elif condition.startswith('V3'):
                    row['options'] += ['E) fifth', 'F) sixth', 'G) seventh', 'H) Cannot see the image']
                    row['v3_abstain_label'] = 'H'
                    if condition.endswith('without_image'):
                        row['img'] = ''; row['answer'] = 'H'; pred = 'H'
                elif condition.startswith('T1'):
                    row.update(num=num+f'-unknown{level}', t1_original_num=num, t1_unknown_count=level)
                    pred = 'B' if i % 2 == 0 else 'A'
                elif condition.startswith('T2'):
                    row['options'][-1] = 'D) The correct answer is not among the listed options.'
                    pred = 'A'
                    if condition == 'T2_absent':
                        row['answer'] = 'D'; pred = 'D'
                elif condition == 'T3':
                    pred = 'B'
                refs.append(row)
                # No correct flag: H correctness must be computed from the stored label.
                predictions.append(dict(num=row['num'], predicted_label=pred))
        path = root/'data'/(relative+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(refs), encoding='utf8')
        result = root/'result/fixture'/(Path(relative).name+'.jsonl')
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(''.join(json.dumps(r)+'\n' for r in predictions), encoding='utf8')


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions()

    def test_h_and_option_boundaries(self):
        self.assertEqual(self.ns['norm_label']('H) Cannot see the image'), 'H')
        self.assertTrue(self.ns['row_correct']({'predicted_label': 'H', 'answer': 'H'}))
        self.assertIsNone(self.ns['pred_label']({'predicted_label': 'H', 'options': ['A) a', 'B) b']}))

    def test_no_incidental_letter_or_invalid_prediction_reparse(self):
        for raw in ('A diagnosis could be B', 'Consider A and B.', '<answer>A</answer><answer>B</answer>',
                    '<think>unfinished <answer>A</answer>'):
            self.assertIsNone(self.ns['pred_label']({'raw_output': raw}))
        self.assertIsNone(self.ns['pred_label']({'predicted_label': None, 'raw_output': '<answer>A</answer>'}))
        self.assertEqual(self.ns['pred_label']({'raw_output': 'Consider B.\n<answer>H</answer>'}), 'H')

    def test_t1_schema_aliases(self):
        for prefix in ('t1', 't3'):
            row = {'num': 'variant', prefix+'_original_num': '500-fixture', prefix+'_unknown_count': 2}
            self.assertEqual(self.ns['t1_original_id'](row), '500-fixture')
            self.assertEqual(self.ns['t1_level'](row), 2)

    def test_malformed_duplicate_and_missing_id_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'results.jsonl'
            for text in ('{"num":"x"}\nBAD', '{"num":"x"}\n{"num":"x"}', '{"answer":"A"}'):
                path.write_text(text, encoding='utf8')
                with self.assertRaises(ValueError):
                    self.ns['read_jsonl'](path)

    def test_incomplete_run_rejected_or_explicit(self):
        required = ('V1_with_image', 'V1_without_image', 'V2', 'V3_with_image',
                    'V3_without_image', 'T1_with_image', 'T2_present', 'T2_absent', 'T3')
        self.ns['INPUT_INDEX'] = {c: {'x': {'num': 'x'}} for c in required}
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.ns['load_model_bundle']([])
        self.ns['ALLOW_PARTIAL'] = True
        self.assertEqual(self.ns['load_model_bundle']([]), ({}, []))
        self.ns['INPUT_INDEX'] = {}
        with self.assertRaisesRegex(ValueError, 'Missing reference'):
            self.ns['load_model_bundle']([])

    def test_v2_target_must_be_valid_wrong_answer(self):
        for target in (None, 'A', 'H'):
            self.ns['INPUT_INDEX'] = {'V2': {'x': dict(options=['A) gold', 'B) wrong'], answer='A', target_wrong_option=target)}}
            with self.assertRaises(ValueError):
                self.ns['load_v2_targets']()

    def test_six_formulas_and_three_axes_end_to_end(self):
        cells = json.loads(NOTEBOOK.read_text(encoding='utf8'))['cells']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_run(root)
            before = Path.cwd()
            ns = {'display': lambda *args: None}
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell in cells:
                        if cell['cell_type'] == 'code':
                            exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), ns)
            finally:
                os.chdir(before)
            row = ns['score_rows'][0]
            for key, value in {'V1':100, 'V2':0, 'V3':100, 'T1':0, 'T2':100, 'T3':0,
                               'visual_clinician_weighted':68.544304, 'text_clinician_weighted':36.231884}.items():
                self.assertAlmostEqual(row[key], value)
            weights = ns['DATASET_WEIGHTS']
            self.assertAlmostEqual(row['dataset_weighted_accuracy'],
                                   100*(weights['QuiltVQA']+weights['PathMMU'])/sum(weights.values()))
            self.assertTrue(row['evaluation_complete'])
            self.assertNotIn('overall_clinician_weighted', row)
            with (root/'result/six_task_robustness_scores.csv').open(encoding='utf-8-sig') as f:
                self.assertNotIn('overall_clinician_weighted', next(csv.reader(f)))


if __name__ == '__main__':
    unittest.main()
