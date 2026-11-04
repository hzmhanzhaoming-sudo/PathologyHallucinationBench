"""Compute clinician-weighted visual/textual axes, accuracy, and Pareto layers.

Explicit inputs only; no inference or plotting. Weights are supplied, not
estimated from performance or silently tuned to match a publication.
Six-score CSV: model_key, model (optional), family (optional), V1,V2,V3,T1,T2,T3.
Base-count CSV: model_key,dataset,correct,total (one row per model/dataset).
Weight CSV: domain,component,weight; domains dataset, visual, text.
"""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.schema import new_output, sha256, write_json
from scoring.metrics import TASK_CONDITIONS
from scoring.task_entry import write_csv

DATASETS=('QuiltVQA','PathVQA','PathMMU','PatchVQA')
DOMAINS={'visual':('V1','V2','V3'),'text':('T1','T2','T3'),'dataset':DATASETS}
AXES=('visual_resistance','text_resistance','weighted_accuracy')


def guard_new_output(path,source_paths=()):
    return new_output(path, source_paths)


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
        seen.add(model)
        if any(r.get(k) in (None, '') for d in ('visual','text') for k in DOMAINS[d]):
            raise ValueError('Undefined task score: cannot construct axes for '+model)
        s={k:float(r[k]) for d in ('visual','text') for k in DOMAINS[d]}
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


def collect_task_reports(folders):
    """Join six reports by model ID, checking shared data and baseline provenance."""
    reports, merged, sources, counts = {}, {}, {}, None
    reference_hashes, baseline_hashes, cohorts, custom_sizes = {}, {}, set(), set()
    expected_models = None
    for folder in folders:
        folder = Path(folder).expanduser().resolve()
        meta_path = folder / 'provenance.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        sources[str(meta_path)] = sha256(meta_path)
        task = meta.get('task')
        if meta.get('schema_version') != 2 or task not in TASK_CONDITIONS or task in reports:
            raise ValueError(f'{folder}: expected one distinct task report for each of V1,V2,V3,T1,T2,T3')
        reports[task] = meta
        custom_sizes.add(meta['custom_size'])
        if task != 'V1':
            cohorts.add(meta['cohort_sha256'])
        for path, digest in meta['input_sha256'].items():
            if path in reference_hashes and reference_hashes[path] != digest:
                raise ValueError('Reference data differ between task reports: ' + path)
            reference_hashes[path] = digest
        def read_report(name):
            path = folder / name
            digest = sha256(path)
            if digest != meta.get('output_sha256', {}).get(name):
                raise ValueError(f'{path}: task report was modified or not registered')
            sources[str(path)] = digest
            with path.open(encoding='utf-8-sig', newline='') as stream:
                return list(csv.DictReader(stream))
        rows = read_report(task + '_scores.csv')
        keys = [r['model_key'] for r in rows]
        if not keys or len(set(keys)) != len(keys) or set(keys) != set(meta['model_metadata']):
            raise ValueError(f'{task}: invalid or duplicate model inventory')
        if expected_models is None:
            expected_models = set(keys)
        if set(keys) != expected_models:
            raise ValueError('All six task reports must contain exactly the same models')
        for row in rows:
            model = row['model_key']
            merged.setdefault(model, dict(model_key=model, model=row.get('model', model)))
            if merged[model]['model'] != row.get('model', model):
                raise ValueError('Model labels differ between tasks: ' + model)
            merged[model][task] = row[task]
            baseline = meta['model_metadata'][model]['prediction_sha256'].get('V1_with_image')
            if baseline:
                if model in baseline_hashes and baseline_hashes[model] != baseline:
                    raise ValueError('Baseline predictions differ between tasks: ' + model)
                baseline_hashes[model] = baseline
        if task == 'V1':
            counts = read_report('base_dataset_counts.csv')
    if set(reports) != set(TASK_CONDITIONS):
        raise ValueError('Six task reports are required; missing: ' + ', '.join(sorted(set(TASK_CONDITIONS) - set(reports))))
    if len(cohorts) != 1 or len(custom_sizes) != 1:
        raise ValueError('Task cohorts or custom-size settings differ between reports')
    return list(merged.values()), counts, sources


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--task-results',type=Path,nargs='+',help='Six independently scored task directories, in any order')
    mode.add_argument('--scores',type=Path,help='Alternatively, one explicit six-score CSV')
    p.add_argument('--base-counts',type=Path,help='Required with --scores; not used with --task-results')
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);sources={}
    if bool(a.scores) != bool(a.base_counts):
        raise ValueError('--base-counts is required only with --scores')
    protected=[*(a.task_results or []), *([a.scores,a.base_counts] if a.scores else []), a.weights, Path(__file__).parent]
    out=guard_new_output(a.output,protected)
    def read(path):
        data=path.read_bytes();sources[str(path.resolve())]=hashlib.sha256(data).hexdigest()
        return list(csv.DictReader(data.decode('utf-8-sig').splitlines()))
    if a.task_results:
        scores,counts,sources=collect_task_reports(a.task_results)
    else:
        scores,counts=read(a.scores),read(a.base_counts)
    rows=compute_axes(scores,counts,read(a.weights))
    if not rows:raise ValueError('No models supplied')
    for name,h in sources.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed during scoring: '+name)
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'six_task_scores.csv',scores)
    write_csv(out/'base_dataset_counts.csv',counts)
    with (out/'three_axis_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'models':len(rows),'sources':sources,'notes':['Weights are explicit published inputs, not re-estimated from clinician raw records by this program.','Weighted accuracy is a ratio of weighted counts, not a mean of per-dataset percentages.','Axes are separate; no across-axis overall score is constructed.','Floating input weights are kept at supplied precision; not silently normalized again.'],'model_calls':0,'api_calls':0}
    (out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if len(rows)==1:
        report['notes'].append('A single model is necessarily Pareto layer 1; comparative layers require multiple models.')
        write_json(out/'provenance.json',report)
    print(json.dumps(report))
    return report

if __name__=='__main__':main()
