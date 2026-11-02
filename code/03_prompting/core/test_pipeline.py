
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/'02_scoring/core'))
import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import contextlib
import io
from unittest.mock import patch

from common import parse_answer, make_messages, gold, index_rows, load_demos, digest
from api_backends import openai_payload, gemini_payload, extract_response
from score_results import score_task, mcnemar, holm
from local_backends import LocalBackend
from finalization import generate_with_final_answer, final_answer_messages, FinalAnswerError

OPTIONS=['A) one','B) two','C) three','D) four']

class Tests(unittest.TestCase):
    def test_grouped_result_layout(self):
        from layout import result_path,locate_result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for task,group in [('baseline','baseline'),('baseline900','baseline'),('V2','V2'),
                               ('V3_with','V3'),('V3_without','V3'),('T2_present','T2'),('T2_absent','T2'),('T3','T3')]:
                self.assertEqual(result_path(root,'model',task),root/group/'model'/(task+'.jsonl'))
            new=result_path(root,'model','V2')
            old=root/'model/V2.jsonl'
            old.parent.mkdir(parents=True)
            old.write_text('old')
            self.assertEqual(locate_result(root,'model','V2'),old)
            new.parent.mkdir(parents=True)
            new.write_text('new')
            with self.assertRaises(ValueError):
                locate_result(root,'model','V2')

    def test_cot_token_budget_and_payload(self):
        from runner import token_budget
        for api in (False,True):
            for strategy in ('zero_shot_cot','few_shot_cot'):
                self.assertEqual(token_budget(api,strategy),512)
        self.assertEqual(token_budget(False,'few_shot_direct'),512)
        self.assertEqual(token_budget(True,'few_shot_direct'),512)
        self.assertEqual(token_budget(True,'few_shot_cot',256),256)
        a=SimpleNamespace(max_new_tokens=token_budget(True,'zero_shot_cot'),temperature=None,
            reasoning_effort=None,image_detail='auto',thinking_budget=None,thinking_level='default')
        message=[{'role':'user','content':[{'type':'text','text':'Question'}]}]
        self.assertEqual(openai_payload(message,'gpt-test',a)['max_output_tokens'],512)
        self.assertEqual(gemini_payload(message,a)['generationConfig']['maxOutputTokens'],512)

    def test_parser(self):
        cases={'A':'A','(B).':'B','B) two':'B','Consider A, then B.\n<answer>C</answer>':'C',
               '<think><answer>A</answer></think>\n<answer>D</answer>':'D',
               'A is unlikely, B may be true':None,'Consider A.':None,
               '<answer>A</answer><answer>B</answer>':None,'<answer>H</answer>':None,
               '<think>Still considering A':None,'Analysis\nFinal answer: B':'B'}
        for text,answer in cases.items():
            self.assertEqual(parse_answer(text,OPTIONS),answer,text)

    def test_gold_and_duplicates(self):
        self.assertEqual(gold(dict(num='1',options=OPTIONS,answer='D) four')),'D')
        with self.assertRaises(ValueError):
            gold(dict(num='1',options=OPTIONS,answer='D) fabricated'))
        with self.assertRaises(ValueError):
            index_rows([dict(num='1'),dict(num='1')])

    def test_prompts_and_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'image.png'
            image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='))
            row=dict(num='target',question='Test question',options=OPTIONS,img=str(image),answer='C) three',
                     explanation='SECRET_TARGET_EXPLANATION')
            demos=[dict(num=str(i),question='Example question',options=OPTIONS,image=str(image),assistant='A') for i in range(3)]
            messages=make_messages(row,Path(tmp)/'data.json','few_shot_direct',demos)
            self.assertEqual(len(messages),7)
            self.assertEqual(sum(c['type']=='image' for m in messages for c in m['content']),4)
            self.assertNotIn('SECRET_TARGET_EXPLANATION',json.dumps(messages))
            row['img']=''
            none=make_messages(row,Path(tmp)/'data.json','few_shot_cot',demos)
            self.assertEqual(sum(c['type']=='image' for m in none for c in m['content']),3)
            zero=make_messages(row,Path(tmp)/'data.json','zero_shot_cot',[])
            self.assertEqual(len(zero),1)
            args=SimpleNamespace(max_new_tokens=512,temperature=None,reasoning_effort=None,
                image_detail='auto',thinking_level='default',thinking_budget=None)
            oa=openai_payload(messages,'gpt-test',args)
            ge=gemini_payload(messages,args)
            self.assertEqual(sum(c['type']=='input_image' for m in oa['input'] if m['role']=='user' for c in m['content']),4)
            self.assertEqual(sum('inline_data' in part for m in ge['contents'] for part in m['parts']),4)
            self.assertNotIn('temperature',oa)
            self.assertNotIn('temperature',ge['generationConfig'])
            self.assertEqual([m['role'] for m in ge['contents']],['user','model','user','model','user','model','user'])

    def test_api_hidden_reasoning_and_truncation(self):
        gemini={'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[
            {'thought':True,'text':'<answer>A</answer>'},{'text':'<answer>B</answer>'}]}}]}
        r=extract_response(gemini,'gemini')
        self.assertEqual(r['raw_output'],'<answer>B</answer>')
        self.assertTrue(r['truncated'])
        response={'status':'incomplete','output':[{'type':'reasoning','summary':[{'text':'A'}]},
            {'type':'message','content':[{'type':'output_text','text':'C'}]}]}
        self.assertEqual(extract_response(response,'openai')['raw_output'],'C')

    def test_finalization_only_after_cot_limit_without_final(self):
        class Fake:
            def __init__(self, first):
                self.first, self.calls = first, []
            def __call__(self, messages):
                self.calls.append(('initial',messages))
                return self.first
            def final_answer(self, messages, budget):
                self.calls.append(('final',messages,budget))
                return dict(raw_output='B',generated_tokens=2,truncated=False)
        messages=[{'role':'user','content':[{'type':'text','text':'Target question'},
                   {'type':'image','image':'target.png'},{'type':'text','text':'Question and options'}]}]
        original=json.dumps(messages)
        first=dict(raw_output='<think>Considering A and B',generated_tokens=512,truncated=True)
        for strategy in ('zero_shot_cot','few_shot_cot'):
            b=Fake(first)
            r=generate_with_final_answer(b,messages,OPTIONS,strategy)
            self.assertEqual(r['raw_output'],'B')
            self.assertEqual(r['initial_response'],first)
            self.assertEqual(r['generation_calls'],2)
            self.assertEqual(r['total_output_tokens'],514)
            self.assertTrue(r['finalization_succeeded'])
            self.assertTrue(r['truncated'])
            self.assertEqual(b.calls[1][2],32)
            self.assertEqual(b.calls[1][1][0],messages[0])
            self.assertTrue(b.calls[1][1][-2]['content'][0]['text'].endswith('</think>'))
            self.assertIn('A, B, C, D',b.calls[1][1][-1]['content'][0]['text'])
        self.assertEqual(json.dumps(messages),original)
        # No gold is supplied to the protocol; even a wrong final is not retried.
        for raw,cutoff,strategy,enabled in [
            ('<answer>D</answer>',True,'zero_shot_cot',True),
            ('Incomplete analysis',False,'zero_shot_cot',True),
            ('Incomplete analysis',True,'few_shot_direct',True),
            ('Incomplete analysis',True,'zero_shot_direct',True),
            ('Incomplete analysis',True,'zero_shot_cot',False),
        ]:
            b=Fake(dict(raw_output=raw,truncated=cutoff))
            r=generate_with_final_answer(b,messages,OPTIONS,strategy,enabled)
            self.assertFalse(r['finalization_attempted'])
            self.assertEqual(len(b.calls),1)

    def test_finalization_preserves_fewshot_images_and_empty_initial(self):
        messages=[]
        for n in range(3):
            messages.extend([{'role':'user','content':[{'type':'image','image':f'{n}.png'}]},
                             {'role':'assistant','content':[{'type':'text','text':'Example answer'}]}])
        messages.append({'role':'user','content':[{'type':'text','text':'Target question'}]})
        for raw in ('','unfinished analysis'):
            followup=final_answer_messages(messages,raw,OPTIONS)
            self.assertEqual(followup[:6],messages[:6])
            self.assertEqual(sum(c['type']=='image' for m in followup for c in m['content']),3)
            self.assertEqual(followup[-1]['role'],'user')
            self.assertTrue(all(m['content'] for m in followup))

    def test_finalization_not_for_api_filter_or_other_incompletion(self):
        for reason in ('content_filter',None):
            b=SimpleNamespace()
            class Fake:
                def __call__(self,messages):
                    return extract_response(dict(status='incomplete',output=[],
                        incomplete_details={'reason':reason}),'openai')
                def final_answer(self,*args):
                    raise AssertionError('Should never retry this response')
            r=generate_with_final_answer(Fake(),[],OPTIONS,'zero_shot_cot')
            self.assertFalse(r['finalization_attempted'])
        r=extract_response(dict(status='incomplete',output=[],
            incomplete_details={'reason':'max_output_tokens'}),'openai')
        self.assertTrue(r['token_limit_reached'])

    def test_finalization_failed_output_is_not_fabricated(self):
        class Fake:
            def __call__(self,messages):
                return dict(raw_output='unfinished',truncated=True)
            def final_answer(self,messages,budget):
                return dict(raw_output='Still considering A or B',truncated=True)
        messages=[{'role':'user','content':[{'type':'text','text':'Question'}]}]
        r=generate_with_final_answer(Fake(),messages,OPTIONS,'zero_shot_cot')
        self.assertFalse(r['finalization_succeeded'])
        self.assertIsNone(parse_answer(r['raw_output'],OPTIONS))
        self.assertEqual(r['generation_calls'],2)
        class Broken(Fake):
            def final_answer(self,*args):
                raise RuntimeError('test failure')
        with self.assertRaises(FinalAnswerError) as caught:
            generate_with_final_answer(Broken(),messages,OPTIONS,'zero_shot_cot')
        self.assertEqual(caught.exception.initial_response['raw_output'],'unfinished')

    def test_finalization_backend_budget_is_temporary(self):
        from api_backends import APIBackend
        for base in (LocalBackend,APIBackend):
            class Fake(base):
                def __init__(self):
                    self.args=SimpleNamespace(max_new_tokens=512,enable_thinking=1)
                def __call__(self,messages):
                    return self._send(messages,self.args)
                def _send(self,messages,args):
                    self.called_budget=args.max_new_tokens
                    self.called_thinking=args.enable_thinking
                    if messages=='raise':
                        raise RuntimeError('test')
                    return {'raw_output':'A'}
            b=Fake()
            original=b.args
            b.final_answer([],32)
            self.assertEqual(b.called_budget,32)
            self.assertEqual(b.called_thinking,0 if base is LocalBackend else 1)
            self.assertIs(b.args,original)
            self.assertEqual(b.args.max_new_tokens,512)
            with self.assertRaises(RuntimeError):
                b.final_answer('raise',32)
            self.assertIs(b.args,original)

    def test_finalization_api_payload_budget_without_network(self):
        from api_backends import APIBackend
        captured=[]
        class Response:
            def __enter__(self):
                return io.StringIO(json.dumps({'status':'completed','output':[
                    {'type':'message','content':[{'type':'output_text','text':'B'}]}]}))
            def __exit__(self,*args):
                pass
        def send(request,**kwargs):
            captured.append(json.loads(request.data))
            return Response()
        a=SimpleNamespace(model='gpt-test',base_url=None,max_new_tokens=512,
            temperature=None,reasoning_effort=None,image_detail='auto',timeout=1,retries=0)
        with patch.dict('os.environ',{'OPENAI_API_KEY':'offline-test-only',
                                     'OPENAI_BASE_URL':'https://api.openai.com/v1'}),\
             patch('urllib.request.urlopen',send):
            b=APIBackend({'backend':'openai'},a)
            msgs=[{'role':'user','content':[{'type':'text','text':'Return B'}]}]
            b(msgs)
            b.final_answer(msgs,32)
        self.assertEqual([p['max_output_tokens'] for p in captured],[512,32])
        self.assertIs(b.args,a)

    def fixtures(self,n=900):
        tasks=['baseline','V2','T3','V3_with','V3_without','T2_present','T2_absent']
        refs={t:{} for t in tasks}
        bundle={t:{} for t in tasks}
        for t in tasks:
            for i in range(n):
                options=OPTIONS[:]
                if t.startswith('V3'):
                    options[1]='B) Cannot see the image'
                if t.startswith('T2'):
                    options[1]='B) The correct answer is not among the listed options.'
                ans=options[1] if t in ('V3_without','T2_absent') else options[0]
                refs[t][str(i)]=dict(num=str(i),options=options,answer=ans)
                bundle[t][str(i)]=dict(pred_letter='A')
        target={t:{str(i):'C' for i in range(n)} for t in ('V2','T3')}
        return refs,bundle,target

    def test_published_count_examples(self):
        refs,b,target=self.fixtures()
        for i in range(545,900):
            b['baseline'][str(i)]['pred_letter']='D'
        for i in range(255):
            b['V2'][str(i)]['pred_letter']='C'
        for i in range(255,276):
            b['V2'][str(i)]['pred_letter']=None
        s=score_task('V2',b,refs,target)
        self.assertEqual((s['O'],s['H'],s['R'],s['E']),(545,255,269,21))
        self.assertAlmostEqual(s['score_percent'],53.2110091743,places=8)
        for i in range(534,900):
            b['V3_with'][str(i)]['pred_letter']='D'
        for i in range(382):
            b['V3_without'][str(i)]['pred_letter']='B'
        v=score_task('V3',b,refs,target)
        self.assertEqual((v['O'],v['S']),(534,382))
        self.assertAlmostEqual(v['score_percent'],71.5355805243,places=8)
        self.assertNotEqual(v['score_percent'],v['paired_switch_success_percent'])
        for i in range(42):
            b['T2_present'][str(i)]['pred_letter']='B'
        for i in range(255,545):
            b['T2_absent'][str(i)]['pred_letter']='B'
        t2=score_task('T2',b,refs,target)
        self.assertEqual((t2['O'],t2['F'],t2['M']),(545,42,255))
        self.assertAlmostEqual(t2['score_percent'],72.752293578,places=8)
        for i in range(55):
            b['T3'][str(i)]['pred_letter']='C'
        for i in range(55,82):
            b['T3'][str(i)]['pred_letter']=None
        t3=score_task('T3',b,refs,target)
        self.assertEqual((t3['O'],t3['R'],t3['H'],t3['E']),(545,463,55,27))
        self.assertAlmostEqual(t3['score_percent'],84.9541284404,places=8)

    def test_zero_opportunities_and_common_set(self):
        refs,b,t=self.fixtures(4)
        for task in ('V2','V3','T2','T3'):
            r=score_task(task,b,refs,t,restrict_opportunities=set())
            self.assertIsNone(r['score_percent'])
            self.assertFalse(r['defined'])

    def test_statistics(self):
        s=mcnemar([True]*10,[False]*10)
        self.assertAlmostEqual(s['mcnemar_p_two_sided'],2/1024)
        self.assertEqual(s['delta_accuracy_pp'],-100)
        self.assertEqual(mcnemar([True,False],[True,False])['mcnemar_p_two_sided'],1)
        r=[{'mcnemar_p_two_sided':x} for x in (.01,.04,.03)]
        holm(r)
        self.assertEqual([x['mcnemar_p_two_sided_holm'] for x in r],[.03,.06,.06])

    def test_guard_never_truncates(self):
        backend=LocalBackend.__new__(LocalBackend)
        backend.info={}
        config=SimpleNamespace(tokenizer_model_max_length=2048,max_position_embeddings=4096)
        called=[]
        def prepare():
            called.append(config.tokenizer_model_max_length)
            return None,None,None,None,SimpleNamespace(shape=(1,3700,8)),None
        backend.model=SimpleNamespace(config=config,prepare_inputs_labels_for_multimodal=prepare)
        backend.args=SimpleNamespace(max_context_tokens=0,max_new_tokens=512)
        backend.install_multimodal_guard('prepare_inputs_labels_for_multimodal')
        with self.assertRaises(ValueError):
            backend.model.prepare_inputs_labels_for_multimodal()
        self.assertEqual(called,[None])
        self.assertEqual(config.tokenizer_model_max_length,2048)

    def test_runner_resume_without_network(self):
        import runner
        class FakeBackend:
            info={'backend':'mock_no_network'}
            calls=0
            def __init__(self,*args):
                pass
            def __call__(self,messages):
                self.__class__.calls+=1
                if self.__class__.calls==2:
                    raise RuntimeError('simulated infrastructure interruption')
                return dict(raw_output='<answer>A</answer>',truncated=False)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'strategy.json').write_text(json.dumps({'strategy':'zero_shot_cot'}))
            data=[dict(num=str(i),img='',question='Question?',options=OPTIONS,answer=OPTIONS[0]) for i in range(2)]
            input_path=folder/'input.json'
            input_path.write_text(json.dumps(data))
            argv=['eval_api.py','--strategy','zero_shot_cot','--model','gpt-5.6-sol','--input',str(input_path),
                  '--output-root',str(folder/'results'),'--data-root',str(folder)]
            with patch.object(runner,'HERE',folder),patch('api_backends.APIBackend',FakeBackend),\
                 patch.dict('os.environ',{'OPENAI_BASE_URL':'https://api.openai.com/v1'}),contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv',argv),self.assertRaises(RuntimeError):
                    runner.main(api=True)
                files=list((folder/'results').rglob('baseline.jsonl'))
                self.assertEqual(len(files),1)
                self.assertEqual(len(files[0].read_text().splitlines()),1)
                with patch('sys.argv',argv+['--resume']):
                    runner.main(api=True)
                self.assertEqual(len(files[0].read_text().splitlines()),2)
                self.assertEqual(FakeBackend.calls,3)
                with patch('sys.argv',argv+['--resume']):
                    runner.main(api=True)
                self.assertEqual(FakeBackend.calls,3)
                with patch('sys.argv',argv),self.assertRaises(FileExistsError):
                    runner.main(api=True)
                records=[json.loads(x) for x in files[0].read_text().splitlines()]
                self.assertTrue(all(x['correct'] for x in records))
                self.assertEqual(len({x['num'] for x in records}),2)

    def test_runner_finalization_persistence_and_summary(self):
        import runner
        class FakeBackend:
            info={'backend':'mock_no_network'}
            def __init__(self,*args):
                pass
            def __call__(self,messages):
                return dict(raw_output='unfinished analysis',truncated=True,generated_tokens=512)
            def final_answer(self,messages,budget):
                self_budget=budget
                assert self_budget==32
                return dict(raw_output='B',truncated=False,generated_tokens=2)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'strategy.json').write_text(json.dumps({'strategy':'zero_shot_cot'}))
            row=dict(num='1',img='',question='Question?',options=OPTIONS,answer=OPTIONS[1])
            source=folder/'input.json'
            source.write_text(json.dumps([row]))
            argv=['eval_api.py','--strategy','zero_shot_cot','--model','gpt-5.6-sol','--input',str(source),
                  '--output-root',str(folder/'results'),'--data-root',str(folder)]
            with patch.object(runner,'HERE',folder),patch('api_backends.APIBackend',FakeBackend),\
                 patch.dict('os.environ',{'OPENAI_BASE_URL':'https://api.openai.com/v1'}),\
                 patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                runner.main(api=True)
            path=next((folder/'results').rglob('baseline.jsonl'))
            record=json.loads(path.read_text())
            self.assertTrue(record['correct'])
            self.assertEqual(record['pred_letter'],'B')
            self.assertEqual(record['initial_response']['raw_output'],'unfinished analysis')
            self.assertEqual(record['total_output_tokens'],514)
            stats=json.loads(path.with_suffix('.summary.json').read_text())['groups']['overall']
            self.assertEqual(stats['finalization_attempted'],1)
            self.assertEqual(stats['finalization_succeeded'],1)
            self.assertEqual(stats['generation_calls'],2)
            from score_results import accuracy_rows
            accuracy=accuracy_rows('mock','zero_shot_cot',{'baseline':{'1':record}},
                                   {'baseline':{'1':row}})[0]
            self.assertEqual(accuracy['accuracy_percent'],100)
            self.assertEqual(accuracy['finalization_attempted'],1)

if __name__=='__main__':
    unittest.main()
