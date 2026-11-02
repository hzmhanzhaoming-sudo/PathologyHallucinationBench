"""Read-only original 16-model six-task recheck from stored JSONL predictions.

No inference, raw-output reparsing, CSV-derived event counts, or source writes.
Golden CSV is used only after independently calculating all six scores.
Legacy V2/T3 figures preserve three-decimal percentage rounding; both exact
values and the declared precision comparison are reported explicitly.
"""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, importlib, json, math, re, sys
from datetime import datetime, timezone


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/data5/hzm/Benchmark'))
    p.add_argument('--core',type=Path,default=Path(__file__).resolve().parent/'core')
    p.add_argument('--output',type=Path,help='Optional NEW local report; default prints JSON only')
    a=p.parse_args(); root=a.data_root.resolve(); core=a.core.resolve()
    prompt_core=Path(__file__).resolve().parents[1]/'03_prompting/core'
    sys.path.insert(0,str(prompt_core));sys.path.insert(0,str(core)); common=importlib.import_module('common'); scorer=importlib.import_module('score_results'); registry=importlib.import_module('registry')
    assert Path(scorer.__file__).resolve()==core/'score_results.py'
    checks=Counter(); failures=[]; files={}; file_rows=[]; events=[]; scores=[]
    def check(name,ok,context=None):
        checks[name]+=1
        if not ok: failures.append({'check':name,'context':context})
    def hashed(path):
        path=Path(path); data=path.read_bytes(); files[str(path)]=hashlib.sha256(data).hexdigest(); return data
    def readcsv(path):
        return list(csv.DictReader(hashed(path).decode('utf-8-sig').splitlines()))
    golden=readcsv(root/'Draw_picture/Fig_sum/six_task_robustness_scores.csv')
    refs=common.validate_suite(root); target_maps={t:common.targets(t,root) for t in ('V2','T3')}
    for task,path in registry.TASK_FILES.items(): hashed(root/path)
    for path in (core/'score_results.py',prompt_core/'common.py',prompt_core/'registry.py'): hashed(path)
    noimage_path=root/'Benchmark_T1_data/pathology_vqa_benchmark_1200_empty_image.json'
    noimage={str(r['num']):r for r in json.loads(hashed(noimage_path))}
    check('V1_input_paired',set(noimage)==set(refs['baseline']))
    t1refs={}
    for choice,n in ((4,1500),(6,1000),(7,1200)):
        path=root/f'Benchmark_T3_data/{choice}-choice-test/pathology_vqa_benchmark_{n}_with_image.json'
        rs=json.loads(hashed(path)); t1refs[choice]={str(r['num']):r for r in rs}
        check('T1_input_unique_ids',len(rs)==len(t1refs[choice]),str(path))
    def unique(folder,pattern):
        found=sorted(folder.glob(pattern))
        if len(found)!=1: raise ValueError(f'Expected exactly one {folder}/{pattern}; found {found}')
        return found[0]
    def load(path,reference,context,filter_levels=False):
        raw=hashed(path); records={}; count=0; selected=0
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip():continue
            row=json.loads(line); count+=1; key=str(row['num'])
            if filter_levels:
                ref=reference.get(key)
                if ref is None: raise ValueError(f'T1 unknown input ID {path}/{key}')
                level=int(ref.get('t3_unknown_count') or re.search(r'unknown(\d+)$',key).group(1))
                if level not in (1,2): continue
            check('raw_id_known',key in reference,[context,key])
            if key not in reference:continue
            ref=reference[key]; selected+=1
            check('raw_id_unique',key not in records,[context,key])
            pred=row.get('pred_letter',row.get('predicted_label'))
            valid=isinstance(pred,str) and pred in common.labels(ref['options'])
            correct=valid and pred==common.gold(ref)
            check('raw_pred_legal_or_null',valid or pred is None,[context,key,pred])
            stored_gold=row.get('gold_letter',row.get('gold_label'))
            check('raw_gold_matches_input',stored_gold==common.gold(ref),[context,key,stored_gold,common.gold(ref)])
            check('raw_correct_flag_matches_prediction',row.get('correct')==correct,[context,key])
            if 'options' in row:check('raw_options_match_input',row['options']==ref['options'],[context,key])
            records[key]={'pred_letter':pred,'correct':correct}
        file_rows.append({'file':str(path),'rows_read':count,'rows_scored':selected,'sha256':files[str(path)]})
        return records
    columns={'V1':'V1_visual_shortcut_resistance','V2':'V2_conflict_target_resistance','V3':'V3_modality_switch_resistance','T1':'T1_unknown_shortcut_resistance','T2':'T2_answer_absence_resistance','T3':'T3_wrong_hint_correctness_retention'}
    for meta in golden:
        key=meta['model_key']; bundle={}
        specs={'baseline':(1,None,'*_with_image.jsonl'),'V2':(4,None,'*_with_image.jsonl'),'V3_with':(6,None,'*_with_image.jsonl'),'V3_without':(6,None,'*_without_image.jsonl'),'T2_present':(5,'V1','*_with_image.jsonl'),'T2_absent':(5,'V2','*_with_image.jsonl'),'T3':(7,None,'*_with_image.jsonl')}
        for task,(number,sub,pattern) in specs.items():
            folder=root/f'Benchmark_T{number}_results'
            if sub:folder/=sub
            path=unique(folder/f'{key}_T{number}_results',pattern)
            bundle[task]=load(path,refs[task],[key,task])
            check('condition_complete',set(bundle[task])==set(refs[task]),[key,task,len(bundle[task])])
        path=unique(root/f'Benchmark_T1_results/{key}_T1_results','*_without_image.jsonl')
        no=load(path,noimage,[key,'V1_without'])
        check('condition_complete',set(no)==set(noimage),[key,'V1_without',len(no)])
        c1=sum(x['correct'] for x in bundle['baseline'].values());c0=sum(x['correct'] for x in no.values())
        chance=sum(1/len(refs['baseline'][i]['options']) for i in sorted(refs['baseline']))
        if c1<=chance:raise ValueError('Undefined V1 denominator '+key)
        task_scores={'V1':min(100.,max(0.,100*(c1-c0)/(c1-chance)))}
        detail={'model_key':key,'model':meta['model'],'family':meta['family_plot'],'V1_N':len(no),'V1_C_img':c1,'V1_C_noimg':c0,'V1_C_random':chance}
        for task in ('V2','V3','T2','T3'):
            s=scorer.score_task(task,bundle,refs,target_maps)
            check('formal_task_defined_complete',s is not None and s['defined'] and s['coverage']==1,[key,task])
            task_scores[task]=s['score_percent']; detail[task]=s
        transitions={1:Counter(),2:Counter()}; seen={1:set(),2:set()}
        for choice in (4,6,7):
            path=unique(root/f'Benchmark_T3_results/{choice}-choice/{key}_T3_results','*_with_image.jsonl')
            rows=load(path,t1refs[choice],[key,'T1',choice],filter_levels=True)
            for id,row in rows.items():
                ref=t1refs[choice][id]; level=int(ref.get('t3_unknown_count') or re.search(r'unknown(\d+)$',id).group(1))
                original=str(ref['t3_original_num'])
                check('T1_mapping_unique',original not in seen[level],[key,level,original]);seen[level].add(original)
                b=bundle['baseline'][original]['correct'];v=row['correct'];transitions[level][('R' if v else 'L') if b else ('G' if v else 'W')]+=1
        rates=[]
        for level,counts in transitions.items():
            check('T1_complete_ids',seen[level]==set(refs['V2']),[key,level,len(seen[level])])
            c,w=counts['R']+counts['L'],counts['G']+counts['W']
            if not c or not w:raise ValueError('Undefined T1 denominator '+key)
            rates.extend([counts['L']/c,counts['G']/w]);detail[f'T1_U{level}']=dict(counts)
        task_scores['T1']=100*(1-sum(rates)/4)
        for task,actual in task_scores.items():
            saved=float(meta[columns[task]]);delta=actual-saved
            # Fixed legacy source precision known before inspecting values, not fitted per cell.
            tol=0.0005000001 if task in ('V2','T3') else 1e-9
            check('six_score_published_precision',abs(delta)<=tol,[key,task,actual,saved,delta,tol])
            check('six_score_one_decimal_display',f'{actual:.1f}'==f'{saved:.1f}',[key,task,actual,saved])
            scores.append({'model_key':key,'model':meta['model'],'family':meta['family_plot'],'task':task,'raw_score':actual,'figure_score':saved,'exact_difference':delta,'comparison_tolerance':tol,'one_decimal':f'{actual:.1f}'})
        events.append(detail)
    check('six_score_inventory',len(golden)==16 and len(scores)==96)
    for path,h in files.items():check('source_unchanged',hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path)
    report={'passed':not failures,'finished_utc':datetime.now(timezone.utc).isoformat(),'scope':'Fresh original stored JSONL predictions, all 16 models; no raw-output reparsing or model inference','checks':dict(checks),'failures':failures,'sources':files,'result_files':file_rows,'prediction_rows_read':sum(r['rows_read'] for r in file_rows),'prediction_rows_scored':sum(r['rows_scored'] for r in file_rows),'scores':scores,'events':events,'model_calls':0,'api_calls':0,'source_results_modified':False,'notes':['V2/T3 historical published summary percentages use three decimal places, tested with 0.0005 score-point bound and independently with manuscript one-decimal display. Exact differences remain visible.','T1 uses authoritative t3_original_num metadata, not numeric-prefix guessing; only U1/U2 are scored.','Checks stored prediction labels and correct flags against input references. Does not adjudicate rationale semantics or silently repeat historical parsing repairs.']}
    if a.output:
        out=a.output.resolve()
        if out.exists():raise ValueError('Output must be new')
        protected=[root/'Draw_picture',root/'Conclusion']+list(root.glob('Benchmark_T*_results'))
        protected += [root/n/'results' for n in ('Benchmark_Zero-shot CoT','Benchmark_Few-shot direct','Benchmark_Few-shot CoT')]
        if any(q==out or q in out.parents for q in protected):raise ValueError('Historical directories are read-only')
        if str(out) in files:raise ValueError('Cannot overwrite an input')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
