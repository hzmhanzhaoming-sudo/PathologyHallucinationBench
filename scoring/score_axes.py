"""Compute clinician-weighted visual/textual axes, accuracy, and Pareto layers.

Explicit inputs only; no inference or plotting. Weights are supplied, not
estimated from performance or silently tuned to match a publication.
Six-score CSV: model_key, model (optional), family (optional), V1,V2,V3,T1,T2,T3.
Base-count CSV: model_key,dataset,correct,total (one row per model/dataset).
Weight CSV: domain,component,weight; domains dataset, visual, text.
"""
import argparse,csv,hashlib,json,math,re
from pathlib import Path

DATASETS=('QuiltVQA','PathVQA','PathMMU','PatchVQA')
DOMAINS={'visual':('V1','V2','V3'),'text':('T1','T2','T3'),'dataset':DATASETS}
AXES=('visual_resistance','text_resistance','weighted_accuracy')


def guard_new_output(path,source_paths=()):
    out=Path(path).expanduser().resolve()
    if out.exists():raise ValueError('Choose a new output path; existing outputs are never overwritten')
    protected_names={'draw_picture','conclusion','historical','historical_reproduction_results',
                     'benchmark_zero-shot cot','benchmark_few-shot direct','benchmark_few-shot cot'}
    if any(part.casefold() in protected_names or re.fullmatch(r'benchmark_t\d+_(data|results)',part.casefold()) for part in out.parts):
        raise ValueError('Reports cannot be written inside historical result/data/figure directories')
    if out in {Path(p).resolve() for p in source_paths}:raise ValueError('Output cannot overwrite a source')
    return out


def pareto_layers(rows):
    remaining=list(range(len(rows)));layers={};level=1
    while remaining:
        front=[]
        for i in remaining:
            dominated=any(j!=i and all(rows[j][k]>=rows[i][k] for k in AXES)
                          and any(rows[j][k]>rows[i][k] for k in AXES) for j in remaining)
            if not dominated:front.append(i)
        if not front:raise ValueError('No Pareto frontier; check finite numeric inputs')
        for i in front:layers[i]=level
        remaining=[i for i in remaining if i not in front];level+=1
    return [layers[i] for i in range(len(rows))]


def compute_axes(score_rows,base_rows,weight_rows):
    weights={domain:{} for domain in DOMAINS}
    for row in weight_rows:
        d=row['domain'];k=row['component'];v=float(row['weight'])
        if d not in weights or k not in DOMAINS[d]:raise ValueError(f'Unknown weight {d}/{k}')
        if k in weights[d] or not math.isfinite(v) or v<0:raise ValueError(f'Invalid/duplicate weight {d}/{k}')
        weights[d][k]=v
    for d,components in DOMAINS.items():
        if set(weights[d])!=set(components):raise ValueError('Incomplete weights '+d)
        if not math.isclose(sum(weights[d].values()),1,rel_tol=0,abs_tol=1e-7):raise ValueError('Weights not normalized '+d)
    counts={}
    for r in base_rows:
        key=(r['model_key'],r['dataset']);c=float(r['correct']);n=float(r['total'])
        if key in counts or key[1] not in DATASETS or not (c.is_integer() and n.is_integer() and n>0 and 0<=c<=n):raise ValueError('Invalid/duplicate Base counts '+str(key))
        counts[key]=(int(c),int(n))
    out=[];seen=set()
    for r in score_rows:
        model=r['model_key']
        if model in seen:raise ValueError('Duplicate model '+model)
        seen.add(model);s={k:float(r[k]) for d in ('visual','text') for k in DOMAINS[d]}
        if not all(math.isfinite(v) and 0<=v<=100 for v in s.values()):raise ValueError('Invalid/undefined task score '+model)
        if any((model,d) not in counts for d in DATASETS):raise ValueError('Incomplete dataset counts '+model)
        x=sum(weights['visual'][k]*s[k] for k in DOMAINS['visual'])
        y=sum(weights['text'][k]*s[k] for k in DOMAINS['text'])
        numerator=sum(weights['dataset'][d]*counts[model,d][0] for d in DATASETS)
        denominator=sum(weights['dataset'][d]*counts[model,d][1] for d in DATASETS)
        z=100*numerator/denominator
        out.append({'model_key':model,'model':r.get('model',model),'family':r.get('family',r.get('family_plot','')),
                    'visual_resistance':x,'text_resistance':y,'weighted_accuracy':z,
                    'visual_hallucination_rate':100-x,'text_hallucination_rate':100-y})
    if set(m for m,d in counts)!=seen:raise ValueError('Base and score model inventories differ')
    for row,layer in zip(out,pareto_layers(out)):row['pareto_3d_layer']=layer
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--base-counts',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sources={}
    def read(path):
        data=path.read_bytes();sources[str(path.resolve())]=hashlib.sha256(data).hexdigest()
        return list(csv.DictReader(data.decode('utf-8-sig').splitlines()))
    rows=compute_axes(read(a.scores),read(a.base_counts),read(a.weights))
    if not rows:raise ValueError('No models supplied')
    for name,h in sources.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed during scoring: '+name)
    out=guard_new_output(a.output,sources)
    out.mkdir(parents=True)
    with (out/'three_axis_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'models':len(rows),'sources':sources,'notes':['Weights are explicit published inputs, not re-estimated from clinician raw records by this program.','Weighted accuracy is a ratio of weighted counts, not a mean of per-dataset percentages.','Axes are separate; no across-axis overall score is constructed.','Floating input weights are kept at supplied precision; not silently normalized again.'],'model_calls':0,'api_calls':0}
    (out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
