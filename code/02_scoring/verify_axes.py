"""Read-only published-axis check using explicit weights and original Base JSONL."""
from pathlib import Path
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from score_axes import compute_axes,AXES,guard_new_output


def verify(root):
    root=Path(root).resolve();sources={};issues=[]
    def read(path):
        data=path.read_bytes();sources[str(path)]=hashlib.sha256(data).hexdigest();return data
    def rows(path):return list(csv.DictReader(read(path).decode('utf-8-sig').splitlines()))
    weights=rows(root/'Draw_picture/Fig_3D/Fig3D_weight_design.csv')
    golden=rows(root/'Draw_picture/Fig_3D/Fig3D_model_metrics.csv')
    six=rows(root/'Draw_picture/Fig_sum/six_task_robustness_scores.csv')
    cols={'V1':'V1_visual_shortcut_resistance','V2':'V2_conflict_target_resistance','V3':'V3_modality_switch_resistance','T1':'T1_unknown_shortcut_resistance','T2':'T2_answer_absence_resistance','T3':'T3_wrong_hint_correctness_retention'}
    score_inputs=[{'model_key':r['model_key'],'model':r['model'],'family':r['family_plot'],**{k:float(r[col]) for k,col in cols.items()}} for r in six]
    refpath=root/'Benchmark_T1_data/pathology_vqa_benchmark_1200_with_image.json'
    refrows=json.loads(read(refpath));refs={str(r['num']):r for r in refrows}
    if len(refrows)!=1200 or len(refs)!=1200:raise ValueError('Incomplete/duplicate Base reference IDs')
    dataset_map={2:None,4:'PathMMU',6:'PatchVQA',7:'PatchVQA'}
    counts=[];nrows=0
    for model in score_inputs:
        key=model['model_key'];paths=list((root/f'Benchmark_T1_results/{key}_T1_results').glob('*_with_image.jsonl'))
        if len(paths)!=1:raise ValueError('Ambiguous Base path '+key)
        data=[json.loads(line) for line in read(paths[0]).decode('utf8').splitlines() if line.strip()]
        if len(data)!=1200 or {str(r['num']) for r in data}!=set(refs):raise ValueError('Incomplete Base data '+key)
        ns={d:[0,0] for d in ('QuiltVQA','PathVQA','PathMMU','PatchVQA')}
        for r in data:
            nrows+=1;ref=refs[str(r['num'])];pred=r.get('pred_letter',r.get('predicted_label'));gold=ref['answer'].strip()[0]
            correct=pred==gold
            if r.get('correct')!=correct:issues.append({'flag':key,'id':r['num']})
            # Original Base's 100/200 IDs identify the two binary source datasets;
            # non-binary membership comes from explicit option counts.
            options=len(ref['options']);dataset=dataset_map[options]
            if options==2:dataset={'100':'QuiltVQA','200':'PathVQA'}[str(ref['num']).split('-')[0]]
            ns[dataset][0]+=int(correct);ns[dataset][1]+=1
        for d,(c,n) in ns.items():counts.append({'model_key':key,'dataset':d,'correct':c,'total':n})
    actual=compute_axes(score_inputs,counts,weights);saved={r['model_key']:r for r in golden}
    if len(actual)!=16 or len(saved)!=16 or set(saved)!={r['model_key'] for r in actual}:raise ValueError('Expected same complete 16-model cohort')
    checked=0;maxdiff=0.
    for r in actual:
        g=saved[r['model_key']]
        for field in AXES+('visual_hallucination_rate','text_hallucination_rate','pareto_3d_layer'):
            checked+=1;delta=abs(r[field]-float(g[field]));maxdiff=max(maxdiff,delta)
            if delta>1e-9:issues.append({'model':r['model_key'],'field':field,'computed':r[field],'saved':g[field],'difference':delta})
    for name,h in sources.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:issues.append({'source_changed':name})
    return {'passed':not issues,'finished_utc':datetime.now(timezone.utc).isoformat(),'data_root':str(root),'models':len(actual),'base_rows_read':nrows,'numeric_cells_checked':checked,'maximum_difference':maxdiff,'issues':issues,'axes':actual,'weights':weights,'base_counts':counts,'score_inputs':score_inputs,'sources':sources,'model_calls':0,'api_calls':0,'source_results_modified':False,'scope':'Fresh Base counts + published six-task source scores + explicit published clinician weights; task scores separately verified from raw JSONL by original_raw_check','limitations':['Does not independently reconstruct clinician weights from all original individual ratings.','Uses the six-task publication CSV precision, preserving historical V2/T3 rounding already separately checked.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/data5/hzm/Benchmark'))
    p.add_argument('--output',type=Path,help='Optional NEW report file, outside historical inputs/figures')
    a=p.parse_args()
    out=guard_new_output(a.output) if a.output else None
    report=verify(a.data_root)
    if out:
        guard_new_output(out,report['sources'])
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
    return int(not report['passed'])

if __name__=='__main__':raise SystemExit(main())
