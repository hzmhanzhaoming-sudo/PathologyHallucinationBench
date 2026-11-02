"""Read-only original six-task scoring using the existing server data layout.

Matches the frozen Fig_sum notebook's data sources, precision and edge policies.
Unlike executing the plotting notebook, this never deletes figures or writes to
its original output folder. No pandas/numpy installation is needed.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

COLUMNS={'V1':'V1_visual_shortcut_resistance','V2':'V2_conflict_target_resistance',
         'V3':'V3_modality_switch_resistance','T1':'T1_unknown_shortcut_resistance',
         'T2':'T2_answer_absence_resistance','T3':'T3_wrong_hint_correctness_retention'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/data5/hzm/Benchmark'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.data_root.resolve();out=a.output.expanduser().resolve()
    if out.exists():raise SystemExit('Choose a NEW output directory; existing files are never overwritten.')
    forbidden=[root/'Draw_picture',root/'Conclusion']+list(root.glob('Benchmark_T*_results'))
    if any(x==out or x in out.parents for x in forbidden):raise SystemExit('Output cannot be inside historical results/figures.')
    sources={}
    def record(path):
        h=hashlib.sha256(path.read_bytes()).hexdigest();sources[str(path)]=h
    def rows(relative):
        path=root/relative;record(path)
        with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    def keyed(data,key='model_key'):
        result={}
        for r in data:
            if r[key] in result:raise ValueError('Duplicate key '+r[key])
            result[r[key]]=r
        return result
    def jsonl(path):
        record(path);result={}
        with path.open(encoding='utf8') as f:
            for line in f:
                if line.strip():
                    r=json.loads(line);result[str(r['num'])]=r
        return result  # same latest-row convention as the frozen notebook
    def yes(v):return str(v).strip().lower() in ('true','1','yes')
    def norm(v):return re.sub('[^a-z0-9]+','',str(v).lower())
    def clip(v):return min(100.0,max(0.0,v))
    meta=keyed(rows('Conclusion/Conclusion_model_scores.csv'))
    v2=keyed([r for r in rows('Benchmark_T4_results/T4_results/T4_overall_comparison.csv') if r['scope'].lower()=='overall'])
    v3=keyed(rows('Benchmark_T6_results/T6_results/T6_overall_by_model.csv'))
    t3=keyed(rows('Benchmark_T7_results/T7_results/T7_overall_by_model.csv'))
    paired=[r for r in rows('Benchmark_T3_results/T3_results/T3_paired_statistics.csv') if r['Modality'].lower()=='image + text']
    t2=defaultdict(list)
    for r in rows('Benchmark_T5_results/T5_results/T5_item_level.csv'):t2[r['model_key']].append(r)
    result=[];events=[]
    for key,m in sorted(meta.items(),key=lambda kv:float(kv[1]['model_order'])):
        folder=root/'Benchmark_T1_results'/(key+'_T1_results')
        with_files=list(folder.glob('*_with_image.jsonl'));without_files=list(folder.glob('*_without_image.jsonl'))
        if len(with_files)!=1 or len(without_files)!=1:raise ValueError('Expected exactly one V1 pair: '+key)
        wi,wo=jsonl(with_files[0]),jsonl(without_files[0])
        if set(wi)!=set(wo) or len(wi)!=1200:raise ValueError('V1 IDs/count mismatch: '+key)
        counts=[int(wi[i].get('option_count') or len(wi[i].get('options',[]))) for i in sorted(wi)]
        if min(counts)<2:raise ValueError('V1 invalid option counts')
        c1=sum(bool(r.get('correct',False)) for r in wi.values());c0=sum(bool(r.get('correct',False)) for r in wo.values())
        chance=sum(1.0/n for n in counts)
        if c1<=chance:raise ValueError('Undefined V1 denominator: '+key)
        s={'V1':clip(100.0*max(0,c1-c0)/(c1-chance)),
           'V2':clip(100-float(v2[key]['induced_target_given_t1_correct_percent'])),
           'T3':clip(float(t3[key]['correct_retention_percent']))}
        denom=float(v3[key]['t6_with_image_correct']);succ=float(v3[key]['t6_both_modalities_correct_count'])
        if denom<=0:raise ValueError('Undefined V3 denominator: '+key)
        s['V3']=clip(100*succ/denom)
        levels=[]
        for level in (1,2):
            subset=[r for r in paired if norm(r['Model'])==norm(m['model']) and float(r['Unknown Level'])==level]
            if not subset:raise ValueError('Missing T1 unknown level: '+key)
            retained,lost,gained,wrong=[sum(float(r[k]) for r in subset) for k in ('Both Correct','Lost Correct','Gained Correct','Both Wrong')]
            regression=100*lost/(retained+lost) if retained+lost else 0.0
            gain=100*gained/(gained+wrong) if gained+wrong else 0.0
            levels.append(100-0.5*(regression+gain))
        s['T1']=clip(sum(levels)/2)
        opportunities=[r for r in t2[key] if yes(r['t1_correct'])]
        n=len(opportunities)
        if not n:raise ValueError('No T2 opportunities: '+key)
        f=sum(yes(r['v1_selected_nota']) for r in opportunities)
        missed=sum(not yes(r['v2_selected_nota']) for r in opportunities)
        s['T2']=clip(100-0.5*(100*f/n+100*missed/n))
        result.append(dict(model_key=key,model=m['model'],family=m['family'],**s))
        events.append(dict(model_key=key,V1_N=len(wi),V1_C_img=c1,V1_C_noimg=c0,V1_random_expected=chance,
                           V3_O=denom,V3_S=succ,T2_O=n,T2_F=f,T2_M=missed))
    golden=keyed(rows('Draw_picture/Fig_sum/six_task_robustness_scores.csv'))
    errors=[];checked=0
    for r in result:
        for task,column in COLUMNS.items():
            checked+=1
            if not math.isclose(r[task],float(golden[r['model_key']][column]),rel_tol=0,abs_tol=1e-9):
                errors.append({'model':r['model_key'],'task':task,'recomputed':r[task],'saved':golden[r['model_key']][column]})
    if len(result)!=16 or checked!=96:raise ValueError('Expected current 16-model original cohort')
    # Ensure all read-only sources stayed unchanged during computation.
    for name,h in sources.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Source changed: '+name)
    out.mkdir(parents=True,exist_ok=False)
    def write(name,data):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write('six_task_scores.csv',result);write('event_counts.csv',events)
    report={'passed':not errors,'models':len(result),'score_cells':checked,'issues':errors,'sources':sources,
            'model_calls':0,'api_calls':0,'method':'Frozen Fig_sum notebook sources and precision; V2/T3 retain historical CSV rounding; T1 zero-component denominator retains historical 0 rule.'}
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'}))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
