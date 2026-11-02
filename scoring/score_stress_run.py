"""Score manuscript evaluation runs from stored labels, without calling models.

V2/V3/T2/T3 call the shared score_task implementation in scoring/core.
V1 and T1 use original six-task formulas; zero opportunities remain undefined.
No free-text response is re-parsed. No published value is used as an input.
"""
from collections import Counter
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TASK_CONDITIONS={'Base':('Base',),'V1':('Base','V1_without'),
 'V2':('Base','V2'),'V3':('V3_with','V3_without'),
 'T1':('Base','T1_4choice','T1_6choice','T1_7choice'),
 'T2':('Base','T2_present','T2_absent'),'T3':('Base','T3')}

def validate_report_target(root,out,run):
    if out.exists():raise ValueError('Output must be new; no report overwrite')
    if out==run or run in out.parents or out in run.parents:raise ValueError('Reports must not overlap the inference run')
    historical=[root/'Draw_picture',root/'Conclusion']+[root/f'Benchmark_T{i}_{suffix}' for i in range(1,8) for suffix in ('data','results')]
    historical += [path.resolve() for path in root.glob('Benchmark_*') if path.is_dir()]
    if out==root or any(out==path or path in out.parents for path in historical):
        raise ValueError('Reports cannot be written inside historical benchmark data, results, experiments or figures')

def v1_score(base,noimage,reference):
    ids=set(base)&set(noimage)&set(reference)
    ci=sum(base[i]['correct'] for i in ids); cn=sum(noimage[i]['correct'] for i in ids)
    chance=sum(1/len(reference[i]['options']) for i in sorted(ids))
    defined=bool(ids) and ci>chance
    return dict(task='V1',N=len(ids),expected_N=len(reference),coverage=len(ids)/len(reference),
                C_image=ci,C_no_image=cn,C_chance=chance,denominator=ci-chance,
                score_percent=min(100.,max(0.,100*(ci-cn)/(ci-chance))) if defined else None,
                defined=defined,score_formula='100*clip((C_image-C_no_image)/(C_image-sum(1/K_i)),0,1)',
                undefined_reason=None if defined else 'No paired items or image-correct count does not exceed chance')

def t1_score(base,variants,references,expected_ids):
    counts={1:Counter(),2:Counter()};seen={1:set(),2:set()}
    for condition,rows in variants.items():
        for num,row in rows.items():
            ref=references[condition][num]
            level=int(ref['t3_unknown_count']); original=str(ref['t3_original_num'])
            if level not in (1,2): raise ValueError('T1 accepts unknown levels 1 and 2 only')
            if original not in expected_ids: raise ValueError('T1 maps outside the 900-question core')
            if original not in base: continue
            if original in seen[level]: raise ValueError('Duplicate T1 original item within unknown level')
            seen[level].add(original)
            b=base[original]['correct']; v=row['correct']
            counts[level][('R' if v else 'L') if b else ('G' if v else 'W')]+=1
    rates=[];levels=[];defined=True
    for level in (1,2):
        c=counts[level]; n_correct=c['R']+c['L'];n_wrong=c['G']+c['W']
        loss=c['L']/n_correct if n_correct else None
        gain=c['G']/n_wrong if n_wrong else None
        defined=defined and loss is not None and gain is not None
        if loss is not None and gain is not None: rates.extend([loss,gain])
        levels.append(dict(level=level,N=len(seen[level]),expected_N=len(expected_ids),
                           C_base=n_correct,W_base=n_wrong,R=c['R'],L=c['L'],G=c['G'],W=c['W'],
                           loss_fraction=loss,gain_fraction=gain))
    n=sum(len(v) for v in seen.values());expected=2*len(expected_ids)
    return dict(task='T1',N=n,expected_N=expected,coverage=n/expected,
                score_percent=100*(1-sum(rates)/4) if defined else None,defined=defined,
                score_formula='100*(1-(L1/C1+G1/W1+L2/C2+G2/W2)/4)',levels=levels,
                zero_denominator_policy='undefined; historical plotting notebook used zero for an empty component, which does not affect the audited 16-model cohort',
                undefined_reason=None if defined else 'At least one baseline-correct or baseline-wrong opportunity set is empty')

def load_shared(scoring_core):
    # A fresh CLI process prevents accidental use of the 01 registry module.
    for name in ('common','registry','score_results'):
        if name in sys.modules: raise ValueError('Use a fresh process; shared module already loaded: '+name)
    sys.path.insert(0,str(scoring_core.resolve()))
    common=importlib.import_module('common');score=importlib.import_module('score_results')
    if Path(score.__file__).resolve()!=scoring_core.resolve()/'score_results.py':
        raise ValueError('Unexpected shared scorer location')
    return common,score

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Single model directory containing run_manifest.json')
    p.add_argument('--output',type=Path,required=True,help='New report directory outside the run directory')
    p.add_argument('--data-root',type=Path,help='Only needed for manifests created before data_root was recorded')
    p.add_argument('--allow-partial',action='store_true',help='Exploratory paired subsets, never marked as a complete formal score')
    p.add_argument('--scoring-core',type=Path,default=HERE/'core')
    a=p.parse_args();run=a.run.resolve();out=a.output.resolve()
    if out.exists():raise ValueError('Output must be new; no report overwrite')
    if out==run or run in out.parents or out in run.parents:raise ValueError('Reports must not overlap the inference run')
    sources={}
    def hashed(path):
        path=Path(path).resolve();raw=path.read_bytes();sources[str(path)]=hashlib.sha256(raw).hexdigest();return raw
    manifest=json.loads(hashed(run/'run_manifest.json'))
    root=a.data_root or (Path(manifest['data_root']) if manifest.get('data_root') else None)
    if root is None:raise ValueError('Run has no data_root; pass --data-root explicitly')
    root=root.resolve()
    validate_report_target(root,out,run)
    common,shared=load_shared(a.scoring_core)
    refs=common.validate_suite(root);target_maps={task:common.targets(task,root) for task in ('V2','T3')}
    for module in ('common','score_results','registry'):hashed(Path(sys.modules[module].__file__))
    registry=sys.modules['registry']
    for rel in set(registry.TASK_FILES.values())|set(registry.TARGET_FILES.values()):hashed(root/rel)
    entries={}
    for e in manifest['plan']:
        condition=e['condition']
        if condition in entries:raise ValueError('Duplicate planned condition')
        if condition not in {c for cs in TASK_CONDITIONS.values() for c in cs}:raise ValueError('Unknown condition')
        entries[condition]=e
    if 'Base' not in entries:raise ValueError('Missing planned Base control')
    requested=manifest.get('task','all')
    tasks=list(TASK_CONDITIONS) if requested=='all' else list(dict.fromkeys(['Base',requested]))
    if any(t not in TASK_CONDITIONS for t in tasks):raise ValueError('Unknown planned task')
    needed=set(c for t in tasks for c in TASK_CONDITIONS[t])
    if set(entries)!=needed:raise ValueError('Manifest task/condition inventory mismatch')
    normalized={};condition_refs={};counts=[]
    for condition,e in entries.items():
        ip=Path(e['input']).resolve()
        if root not in ip.parents:raise ValueError('Planned input outside selected data root')
        raw_input=hashed(ip);h=sources[str(ip)]
        if not e.get('input_sha256') or h!=e['input_sha256']:
            raise ValueError('Input changed or no execution-time input hash: '+condition)
        rr=json.loads(raw_input);ref={str(r['num']):r for r in rr}
        if len(ref)!=len(rr) or len(ref)!=e['input_expected_rows']:raise ValueError('Input IDs/count mismatch')
        shared_key='baseline' if condition=='Base' else condition
        if shared_key in refs and ref!=refs[shared_key]:raise ValueError('Planned input differs from the canonical task input: '+condition)
        if condition=='V1_without':
            if set(ref)!=set(refs['baseline']):raise ValueError('V1 input is not paired with full Base')
            for num,row in ref.items():
                b=refs['baseline'][num]
                if any(row.get(k)!=b.get(k) for k in ('question','options','answer')) or row.get('img'):
                    raise ValueError('V1 no-image input does not preserve the clean question/options/gold')
        condition_refs[condition]=ref
        path=run/(condition+'.jsonl')
        if Path(e['output']).name!=path.name:raise ValueError('Unexpected output basename')
        if not path.is_file():
            if not a.allow_partial:raise ValueError('Missing result file: '+condition)
            normalized[condition]={};counts.append(dict(condition=condition,rows=0,expected=len(ref),complete=False));continue
        rows={}
        for text in hashed(path).decode('utf-8-sig').splitlines():
            if not text.strip():continue
            row=json.loads(text);num=str(row['num'])
            if num not in ref or num in rows:raise ValueError('Unknown/duplicate result ID: '+condition+'/'+num)
            r=ref[num]
            if 'pred_letter' in row and 'predicted_label' in row and row['pred_letter']!=row['predicted_label']:
                raise ValueError('Conflicting stored predictions')
            if 'pred_letter' not in row and 'predicted_label' not in row:raise ValueError('Missing stored prediction field')
            pred=row.get('pred_letter',row.get('predicted_label'))
            if pred is not None and pred not in common.labels(r['options']):raise ValueError('Stored prediction not legal or null')
            gold=row.get('gold_letter',row.get('gold_label'))
            if gold!=common.gold(r):raise ValueError('Stored gold differs from input')
            correct=pred is not None and pred==gold
            if type(row.get('correct')) is not bool or row['correct']!=correct:raise ValueError('Correct flag differs from stored prediction')
            if 'options' in row and row['options']!=r['options']:raise ValueError('Stored options differ from input')
            if 'question' in row and row['question']!=r.get('question'):raise ValueError('Stored question differs from input')
            if row.get('source_file') and Path(row['source_file']).resolve()!=ip:raise ValueError('Stored source file differs from planned input')
            rows[num]=dict(pred_letter=pred,correct=correct)
        complete=set(rows)==set(ref)
        if not complete and not a.allow_partial:raise ValueError('Incomplete result condition: '+condition)
        normalized[condition]=rows
        counts.append(dict(condition=condition,rows=len(rows),expected=len(ref),complete=complete,
                           invalid=sum(r['pred_letter'] is None for r in rows.values())))
    base=normalized['Base'];n=len(base);c=sum(r['correct'] for r in base.values())
    result=[dict(task='Base',N=n,C=c,expected_N=len(refs['baseline']),coverage=n/len(refs['baseline']),
                 score_percent=100*c/n if n else None,defined=bool(n),score_formula='100*C/N',unit='accuracy_percent')]
    bundle={('baseline' if k=='Base' else k):v for k,v in normalized.items()}
    for task in tasks:
        if task=='Base':continue
        if task=='V1':s=v1_score(base,normalized['V1_without'],refs['baseline'])
        elif task=='T1':
            variants={k:normalized[k] for k in TASK_CONDITIONS['T1'] if k!='Base'}
            s=t1_score(base,variants,condition_refs,set(refs['V2']))
        else:
            s=shared.score_task(task,bundle,refs,target_maps)
            if s is None:s=dict(task=task,N=0,expected_N=900,coverage=0.,score_percent=None,defined=False,undefined_reason='No shared paired items')
        s['unit']='resistance_score_points';result.append(s)
    complete=all(x['complete'] for x in counts)
    for s in result:s['formal_complete']=complete and s['coverage']==1 and s['defined']
    report=dict(model=manifest['model'],task_scope=requested,run=str(run),complete=complete,
                all_scores_defined=all(s['defined'] for s in result),all_scores_formal_complete=all(s['formal_complete'] for s in result),
                exploratory=a.allow_partial and not complete,conditions=counts,scores=result,
                sources=sources,model_calls=0,api_calls=0,raw_output_reparsed=False,
                source_results_modified=False,
                note='Full-precision computed values. Historical V2/T3 figure CSVs may retain three-decimal source rounding; no such rounding is applied here.')
    for path,h in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise ValueError('Source changed during scoring')
    if any(out==Path(s) or out in Path(s).parents for s in sources):raise ValueError('Report overlaps an input source')
    out.mkdir(parents=True,exist_ok=False)
    (out/'scores.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['model','task','score_percent','unit','N','expected_N','coverage','defined','formal_complete']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result:writer.writerow({key:manifest['model'] if key=='model' else row.get(key) for key in fields})
    # Export counts, not percentages, for the explicitly weighted accuracy axis.
    dataset_counts={}
    for num,row in base.items():
        name=common.dataset(refs['baseline'][num])
        record=dataset_counts.setdefault(name,dict(model_key=manifest['model'],dataset=name,correct=0,total=0))
        record['correct']+=int(row['correct']);record['total']+=1
    if counts and next(r for r in counts if r['condition']=='Base')['complete']:
        with (out/'base_dataset_counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['model_key','dataset','correct','total']);writer.writeheader()
            writer.writerows(dataset_counts[k] for k in sorted(dataset_counts))
    if requested=='all' and all(s['formal_complete'] for s in result):
        wide={'model_key':manifest['model'],**{s['task']:s['score_percent'] for s in result if s['task']!='Base'}}
        with (out/'six_task_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['model_key','V1','V2','V3','T1','T2','T3']);writer.writeheader();writer.writerow(wide)
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
