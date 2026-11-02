import ast
import hashlib
import json
from pathlib import Path
import unittest
from launcher import parser,make_plan
from registry import HERE,MODELS,CONDITIONS,expand_task,script_relative

class OfflineContract(unittest.TestCase):
    def test_six_tasks_not_six_conditions(self):
        self.assertEqual(len(expand_task('all')),11)
        self.assertEqual(sum(CONDITIONS[c][2] for c in expand_task('all')),9600)
        self.assertEqual(expand_task('V1'),['Base','V1_without'])
    def test_v1_uses_full_1200(self):
        for c in ('Base','V1_without'):
            self.assertIn('Benchmark_T1_data/',CONDITIONS[c][0])
            self.assertEqual(CONDITIONS[c][2],1200)
    def test_t1_only_two_replacements(self):
        for c in ('T1_4choice','T1_6choice','T1_7choice'):
            self.assertIn('unknown1_2',CONDITIONS[c][0])
    def test_all_16_model_routes_no_import_or_calls(self):
        for model,m in MODELS.items():
            kind='api' if m['api'] else 'local'
            args=parser(kind).parse_args(['--model',model])
            root,plan=make_plan(args,kind)
            self.assertFalse(args.execute)
            self.assertEqual(len(plan),11)
            self.assertTrue(all('--output' in p['command'] for p in plan))
            self.assertFalse(root.exists())
    def test_original_sources_hash_and_syntax(self):
        manifest=json.loads((HERE/'provenance'/'original_source_manifest.json').read_text(encoding='utf8'))
        for rel,rec in manifest['sources'].items():
            b=(HERE/'vendor'/rel).read_bytes()
            self.assertEqual(hashlib.sha256(b).hexdigest(),rec['sha256'])
            ast.parse(b.decode('utf8'))
    def test_refuse_unsafe_run_ids(self):
        for value in ('../old','a/b','a\\b','..'):
            a=parser('local').parse_args(['--model','qwen35-4b','--run-id',value])
            with self.assertRaises(ValueError): make_plan(a,'local')
    def test_no_implicit_execution(self):
        a=parser('local').parse_args(['--model','qwen35-4b','--execute'])
        with self.assertRaises(ValueError): make_plan(a,'local')
    def test_gpt_rejects_gemini_flags(self):
        a=parser('api').parse_args(['--model','gpt-5.6-sol','--thinking-level','minimal'])
        with self.assertRaises(ValueError): make_plan(a,'api')

if __name__=='__main__': unittest.main()
