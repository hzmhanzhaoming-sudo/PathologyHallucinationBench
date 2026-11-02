"""No models, providers or real evaluation output are used by these tests."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/'02_scoring/core'))
import base64
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import gold, load_demos, make_messages, parse_answer, read_json
from task_demos import FAMILY, POLICY, NOTA, UNSEEN, make_suite, load_task_demos


def fixture(root):
    rows=[]
    refs=[]
    options=[['Solid nests with peripheral palisading','Sheets of undifferentiated cells',
              'Disorganized with infiltrative growth pattern','Papillary with fibrovascular cores'],
             ['Subepidermal blister','Intraepidermal blister','Suprabasal blister','Intradermal blister'],
             ['Fungal infection','Acute bacterial infection','Cytomegalovirus (CMV) infection','Tuberculous granulomatous disease']]
    questions=['Architecture?','Blister location?','Diagnosis?']
    explanations=['Papillary projections with cores are present.',
                  'The image shows separation beneath an intact epidermis.',
                  'Inclusion bodies support the source CMV diagnosis.']
    for i,(question,opts,answer) in enumerate(zip(questions,options,'DAC'),1):
        image=root/f'image{i}.png'
        image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')+bytes([i]))
        row=dict(num=f'FS-{i:03d}',img=str(image),question=question,
                 options=[f'{chr(65+j)}) {t}' for j,t in enumerate(opts)],answer=answer)
        rows.append(row)
        refs.append(dict(num=row['num'],original_record={**row,'explanation':explanations[i-1]}))
    path=root/'fewshot_direct.json'
    path.write_text(json.dumps(rows),encoding='utf8')
    (root/'reference_only').mkdir()
    (root/'reference_only/source_records_with_explanations.json').write_text(json.dumps(refs),encoding='utf8')
    suite=make_suite(path)
    suite_path=root/'task_specific_v3/demo_suite.json'
    suite_path.parent.mkdir()
    suite_path.write_text(json.dumps(suite),encoding='utf8')
    return path,suite_path,suite


class TaskDemosTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.path,self.suite_path,self.suite=fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_original_three_shots_unchanged(self):
        for strategy in ('few_shot_direct','few_shot_cot'):
            before,_=load_demos(strategy,self.path)
            after,_=load_task_demos(strategy,self.path,'baseline')
            self.assertEqual(before,after)
            subset,_=load_task_demos(strategy,self.path,'baseline900')
            self.assertEqual(after,subset)

    def test_v2_t3_targets_wrong_but_demonstration_answers_correct(self):
        original=self.suite['banks']['baseline']
        for task,field in [('V2','target_wrong_option'),('T3','hinted_wrong_option')]:
            for row,base in zip(self.suite['banks'][task],original):
                self.assertEqual(row['options'],base['options'])
                self.assertEqual(row['answer'],base['answer'])
                self.assertEqual(row['img'],base['img'])
                self.assertNotEqual(row['adaptation'][field],gold(row))
                self.assertIn('not a verbatim',row['rationale_source'])
                self.assertNotEqual(row['explanation'],base['explanation'])

    def test_v3_one_complete_pair_shared_between_conditions(self):
        for strategy in ('few_shot_direct','few_shot_cot'):
            left,pl=load_task_demos(strategy,self.path,'V3_with')
            right,pr=load_task_demos(strategy,self.path,'V3_without')
            self.assertEqual(left,right)
            self.assertEqual(pl['suite_sha256'],pr['suite_sha256'])
            self.assertEqual(pl['visible_images'],1)
            self.assertEqual(pr['visible_images'],1)
            self.assertEqual(pl['pair_count'],1)
            self.assertEqual(pl['source_question_count'],1)
            self.assertEqual([parse_answer(d['assistant'],d['options']) for d in left],list('EB'))
            self.assertTrue(pl['shared_paired_bank'])
            for present,absent in zip(left[::2],left[1::2]):
                self.assertEqual(present['question'],absent['question'])
                self.assertEqual(present['options'],absent['options'])
                self.assertIsNotNone(present['image'])
                self.assertIsNone(absent['image'])
                self.assertNotIn('The image shows',absent['assistant'])
                letter=parse_answer(absent['assistant'],absent['options'])
                self.assertIn(letter+') '+UNSEEN,absent['options'])

    def test_t2_answer_semantics_change_even_if_letter_is_same(self):
        left,pl=load_task_demos('few_shot_cot',self.path,'T2_present')
        right,pr=load_task_demos('few_shot_cot',self.path,'T2_absent')
        self.assertEqual(left,right)
        self.assertEqual(pl['suite_sha256'],pr['suite_sha256'])
        self.assertEqual(pl['variants'],['answer_present','answer_absent'])
        self.assertEqual(pr['variants'],pl['variants'])
        self.assertEqual([parse_answer(d['assistant'],d['options']) for d in left],list('DD'))
        self.assertEqual(pl['pair_count'],1)
        for present,absent in zip(left[::2],left[1::2]):
            self.assertEqual(present['image'],absent['image'])
            self.assertIsNotNone(absent['image'])
            letter=parse_answer(absent['assistant'],absent['options'])
            self.assertIn(letter+') '+NOTA,absent['options'])
            self.assertNotIn(letter+') '+NOTA,present['options'])
            self.assertIn('not among the listed',absent['assistant'])

    def test_direct_and_cot_use_identical_inputs_and_answers(self):
        for task in FAMILY:
            direct,pd=load_task_demos('few_shot_direct',self.path,task)
            cot,pc=load_task_demos('few_shot_cot',self.path,task)
            self.assertEqual(pd['input_content'],pc['input_content'])
            self.assertEqual(pd['suite_sha256'],pc['suite_sha256'])
            for a,b in zip(direct,cot):
                self.assertEqual({k:v for k,v in a.items() if k!='assistant'},
                                 {k:v for k,v in b.items() if k!='assistant'})
                self.assertEqual(a['assistant'],parse_answer(b['assistant'],b['options']))
                self.assertEqual(len(a['assistant']),1)

    def test_no_target_gold_or_explanation_leaks(self):
        row=dict(num='target',img='',question='Current question?',options=['A) x','B) y'],
                 answer='A',explanation='SECRET_TARGET_EXPLANATION',target_wrong_option='SECRET_WRONG_TARGET')
        for task in FAMILY:
            demos,_=load_task_demos('few_shot_cot',self.path,task)
            first=make_messages(row,self.path,'few_shot_cot',demos)
            row['answer']='B'
            second=make_messages(row,self.path,'few_shot_cot',demos)
            self.assertEqual(first,second)
            self.assertNotIn('SECRET_',json.dumps(first))

    def test_image_counts_and_answer_positions(self):
        row=dict(num='target',img=str(self.root/'image1.png'),question='Question?',options=['A) x','B) y'])
        for task in FAMILY:
            demos,_=load_task_demos('few_shot_direct',self.path,task)
            row['img']='' if task=='V3_without' else str(self.root/'image1.png')
            messages=make_messages(row,self.path,'few_shot_direct',demos)
            expected={'V3_with':2,'V3_without':1,'T2_present':3,'T2_absent':3}.get(task,4)
            self.assertEqual(sum(c['type']=='image' for m in messages for c in m['content']),expected)
            self.assertEqual(len(messages),2*len(demos)+1)
            self.assertEqual(messages[-1]['role'],'user')

    def test_suite_tampering_is_rejected(self):
        self.suite['banks']['V3_pairs'][1]['answer']='A'
        self.suite_path.write_text(json.dumps(self.suite),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'suite/source mismatch'):
            load_task_demos('few_shot_cot',self.path,'V3_without')

    def test_zero_shot_never_reads_any_demonstration_file(self):
        for strategy in ('zero_shot_cot','zero_shot_direct'):
            for task in FAMILY:
                self.assertEqual(load_task_demos(strategy,self.root/'nonexistent',task),([],{}))

    def test_runner_real_message_flow_with_fake_backend(self):
        import runner
        seen=[]
        class Fake:
            info={'backend':'TEST_ONLY_no_model_or_network'}
            def __init__(self,*args):
                pass
            def __call__(self,messages):
                seen.append(messages)
                return dict(raw_output='<answer>B</answer>',truncated=False)
        here=self.root/'code'
        here.mkdir()
        (here/'strategy.json').write_text(json.dumps({'strategy':'few_shot_cot'}))
        target=self.root/'targets.json'
        target.write_text(json.dumps([dict(num='target',question='Current image?',img='',
                                         options=['A) lesion','B) Cannot see the image'],answer='B')]))
        argv=['eval_api.py','--strategy','few_shot_cot','--model','gpt-5.6-sol','--task','V3_without','--input',str(target),
              '--demos',str(self.path),'--data-root',str(self.root),'--output-root',str(self.root/'out')]
        with patch.object(runner,'HERE',here),patch('api_backends.APIBackend',Fake),\
             patch('sys.argv',argv),patch.dict('os.environ',{'OPENAI_BASE_URL':'https://api.openai.com/v1'}),\
             contextlib.redirect_stdout(io.StringIO()):
            runner.main(api=True)
        self.assertEqual(len(seen),1)
        self.assertEqual(sum(c['type']=='image' for m in seen[0] for c in m['content']),1)
        path=next((self.root/'out').rglob('V3_without.jsonl'))
        self.assertTrue(read_json(path)['correct'])
        config=read_json(path.with_suffix('.run.json'))['config']
        self.assertEqual(config['arguments']['demonstration_policy'],POLICY)
        self.assertEqual(config['demonstrations']['variants'],['image_present','image_unavailable'])
        self.assertEqual(config['demonstrations']['pair_count'],1)

    def test_optional_two_pairs_and_other_tasks_still_three(self):
        suite=make_suite(self.path,2)
        alternate=self.suite_path.parent/'demo_suite_2pairs.json'
        alternate.write_text(json.dumps(suite),encoding='utf8')
        for task in FAMILY:
            demos,p=load_task_demos('few_shot_cot',self.path,task,alternate)
            paired=task.startswith(('V3','T2'))
            self.assertEqual(len(demos),4 if paired else 3)
            self.assertEqual(p['pair_count'],2 if paired else 0)
            if paired:
                self.assertEqual(p['source_question_count'],2)
                self.assertEqual(len({d['num'] for d in demos}),4)
                for a,b in zip(demos[::2],demos[1::2]):
                    self.assertEqual(a['question'],b['question'])
            if task.startswith('V3'):
                self.assertEqual(p['visible_images'],2)
            elif task.startswith('T2'):
                self.assertEqual(p['visible_images'],4)
        for count in (0,3,True,None):
            with self.assertRaises(ValueError):
                make_suite(self.path,count)

    def test_comparison_override_checks_reviewed_code_and_all_decoding_controls(self):
        import compare_strategies as compare
        manifest={'reviewed_code_sha256':[{'runner.py':'old'},{'runner.py':'new','task_demos.py':'new'}]}
        a=dict(strategy='zero_shot_cot',code_sha256=manifest['reviewed_code_sha256'][0],max_new_tokens=512,temperature=0)
        b=dict(strategy='few_shot_cot',code_sha256=manifest['reviewed_code_sha256'][1],max_new_tokens=512,temperature=0,
               demonstration_policy=POLICY,demo_suite='example_suite',demo_suite_sha256='abc')
        with patch.object(compare,'read_json',return_value=manifest):
            self.assertEqual(compare.comparison_controls(a,True),compare.comparison_controls(b,True))
            self.assertNotEqual(compare.comparison_controls(a),compare.comparison_controls(b))
            b['max_new_tokens']=1024
            self.assertNotEqual(compare.comparison_controls(a,True),compare.comparison_controls(b,True))
            b['code_sha256']={'local_backends.py':'unknown'}
            with self.assertRaisesRegex(ValueError,'Unreviewed'):
                compare.comparison_controls(b,True)


if __name__=='__main__':
    unittest.main()
