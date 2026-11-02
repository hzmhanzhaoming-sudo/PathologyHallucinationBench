"""V2/V3/T2/T3: standard-library-only, paired IDs, auditable denominators.

Formulas match Supplementary Note 2: V2=1-H/O; V3=S/O;
T2=1-(F/O+M/O)/2; T3=R/O. All displayed scores are percentages.
"""
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from common import (read_json, index_rows, gold, digest, dataset, option_containing,
                    targets, validate_suite, file_hash)
from registry import ROOT, MODELS
from layout import RESULTS_DIR,locate_result

NEEDED = {'V2': ('baseline','V2'), 'V3': ('V3_with','V3_without'),
          'T2': ('baseline','T2_present','T2_absent'), 'T3': ('baseline','T3')}

def pct(x, n):
    return 100*x/n if n else None

def wilson(x, n):
    if not n:
        return [None,None]
    z = 1.959963984540054
    p = x/n
    c = (p+z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*max(0,c-h),100*min(1,c+h)]

def mcnemar(a, b):
    n11 = sum(x and y for x,y in zip(a,b))
    n10 = sum(x and not y for x,y in zip(a,b))
    n01 = sum(not x and y for x,y in zip(a,b))
    n00 = len(a)-n11-n10-n01
    n, k = n10+n01, min(n10,n01)
    p = 1.0 if not n else min(1.0, 2*sum(math.exp(math.lgamma(n+1)-math.lgamma(i+1)-
             math.lgamma(n-i+1)-n*math.log(2)) for i in range(k+1)))
    return dict(n11=n11,n10=n10,n01=n01,n00=n00,n=len(a),
                delta_accuracy_pp=pct(n01-n10,len(a)),mcnemar_p_two_sided=p)

def holm(rows, field='mcnemar_p_two_sided'):
    valid = sorted((r[field],i) for i,r in enumerate(rows) if r.get(field) is not None)
    running = 0.0
    for rank,(p,i) in enumerate(valid):
        running = max(running,min(1,p*(len(valid)-rank)))
        rows[i][field+'_holm'] = running
        rows[i]['holm_family_size'] = len(valid)

def correct(row, reference):
    return row.get('pred_letter') == gold(reference)

def score_task(task, bundle, refs, target_maps, restrict_opportunities=None):
    required = NEEDED[task]
    if any(key not in bundle for key in required):
        return None
    ids = set(refs['V2'])
    for key in required:
        ids &= set(bundle[key])
    if not ids:
        return None
    ids = sorted(ids)
    base_key = 'V3_with' if task=='V3' else 'baseline'
    opp = [i for i in ids if correct(bundle[base_key][i],refs[base_key][i])]
    if restrict_opportunities is not None:
        opp = [i for i in opp if i in restrict_opportunities]
    n, o = len(ids),len(opp)
    result = dict(task=task,N=n,O=o,expected_N=len(refs['V2']),coverage=n/len(refs['V2']),
                  score_percent=None,score_ci95_low=None,score_ci95_high=None)
    if task in ('V2','T3'):
        r = sum(correct(bundle[task][i],refs[task][i]) for i in opp)
        h = sum(bundle[task][i].get('pred_letter')==target_maps[task][i] for i in opp)
        e = o-r-h
        result.update(R=r,H=h,E=e,retention_percent=pct(r,o),target_adoption_percent=pct(h,o),
                      other_new_error_percent=pct(e,o),invalid_on_opportunities=sum(bundle[task][i].get('pred_letter') is None for i in opp))
        numerator = o-h if task=='V2' else r
        result['score_percent'] = pct(numerator,o)
        result['score_ci95_low'],result['score_ci95_high'] = wilson(numerator,o)
        result['score_formula'] = '100*(1-H/O)' if task=='V2' else '100*R/O'
        result['rescued_original_wrong'] = sum(not correct(bundle['baseline'][i],refs['baseline'][i]) and
                                                 correct(bundle[task][i],refs[task][i]) for i in ids)
    elif task=='V3':
        s = sum(correct(bundle['V3_without'][i],refs['V3_without'][i]) for i in opp)
        result.update(S=s,score_percent=pct(s,o),paired_switch_success_percent=pct(s,n),
            with_image_correct=sum(correct(bundle['V3_with'][i],refs['V3_with'][i]) for i in ids),
            without_image_correct=sum(correct(bundle['V3_without'][i],refs['V3_without'][i]) for i in ids),
            false_refusal_with_image=sum(bundle['V3_with'][i].get('pred_letter')==gold(refs['V3_without'][i]) for i in ids),
            score_formula='100*S/O; O=correct in augmented-option image-present condition')
        result['score_ci95_low'],result['score_ci95_high'] = wilson(s,o)
    else:
        f = [bundle['T2_present'][i].get('pred_letter')==option_containing(refs['T2_present'][i], 'correct answer is not among') for i in opp]
        missed = [bundle['T2_absent'][i].get('pred_letter')!=gold(refs['T2_absent'][i]) for i in opp]
        result.update(F=sum(f),M=sum(missed),false_NOTA_percent=pct(sum(f),o),missed_NOTA_percent=pct(sum(missed),o),
            score_percent=100*(1-(sum(f)+sum(missed))/(2*o)) if o else None,
            both_availability_decisions_correct=sum(not x and not y for x,y in zip(f,missed)),
            both_availability_errors=sum(x and y for x,y in zip(f,missed)),
            false_NOTA_only=sum(x and not y for x,y in zip(f,missed)),
            missed_NOTA_only=sum(not x and y for x,y in zip(f,missed)),
            score_formula='100*(1-(F/O+M/O)/2)',
            score_ci_note='No binomial Wilson interval for this paired two-component score')
    result['defined'] = o>0
    return result

def read_result(path, task, refs, model):
    metadata = read_json(path.with_suffix('.run.json'))
    config = metadata['config']
    if metadata['fingerprint'] != digest(config):
        raise ValueError(f'Run metadata fingerprint mismatch: {path}')
    if config['model']!=model or config['task']!=task:
        raise ValueError(f'Run metadata mismatch: {path}')
    if config.get('input_sha256')!=file_hash(config['input']):
        raise ValueError(f'Input changed after inference: {path}')
    records = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    records = index_rows(records)
    if not set(records) <= set(refs[task]):
        raise ValueError(f'Unexpected IDs: {path}')
    for key,r in records.items():
        source = refs[task][key]
        if (r.get('task')!=task or r.get('model')!=model or r.get('strategy')!=config['strategy'] or
            r.get('run_fingerprint')!=metadata['fingerprint'] or r.get('gold_letter')!=gold(source) or
            r.get('item_sha256')!=digest(source)):
            raise ValueError(f'Invalid item provenance/target: {path}/{key}')
    return records,config

def load_bundle(result_root, refs, model):
    bundle,configs = {},{}
    for task in refs:
        path = locate_result(result_root,model,task)
        if path.is_file():
            bundle[task],configs[task] = read_result(path,task,refs,model)
    if 'baseline900' in bundle:
        if 'baseline' in bundle:
            for i,r in bundle['baseline900'].items():
                if i not in bundle['baseline'] or r['pred_letter']!=bundle['baseline'][i]['pred_letter']:
                    raise ValueError('baseline and baseline900 disagree: select one baseline result file')
        else:
            bundle['baseline']=bundle['baseline900']
    if configs:
        signatures = {digest({k:v for k,v in c['arguments'].items() if k!='input'}) for c in configs.values()}
        if len(signatures)!=1:
            raise ValueError(f'Decoding settings/code differ between paired conditions: {result_root}/{model}')
        for left,right in (('V3_with','V3_without'),('T2_present','T2_absent')):
            if left in configs and right in configs:
                dl,dr = (configs[t].get('demonstrations',{}) for t in (left,right))
                if dl.get('protocol')=='task_specific_mixed_v1' or dr.get('protocol')=='task_specific_mixed_v1':
                    if dl.get('protocol')!=dr.get('protocol') or dl.get('content')!=dr.get('content'):
                        raise ValueError(f'Demonstrations differ within a paired test: {left}/{right}')
                if dl.get('protocol')=='task_specific_fixed_pairs_v3' or dr.get('protocol')=='task_specific_fixed_pairs_v3':
                    expected_bank='V3_pairs' if left.startswith('V3') else 'T2_pairs'
                    if (dl.get('protocol')!=dr.get('protocol') or dl.get('content')!=dr.get('content')
                            or dl.get('suite_sha256')!=dr.get('suite_sha256')
                            or dl.get('bank')!=expected_bank or dr.get('bank')!=expected_bank
                            or dl.get('pair_count') not in (1,2) or dl.get('pair_count')!=dr.get('pair_count')):
                        raise ValueError(f'Inconsistent fixed demonstration pairs: {left}/{right}')
                if dl.get('protocol')=='task_specific_condition_groups_v2' or dr.get('protocol')=='task_specific_condition_groups_v2':
                    if (dl.get('protocol')!=dr.get('protocol') or dl.get('suite_sha256')!=dr.get('suite_sha256')
                            or dl.get('bank')!=left or dr.get('bank')!=right):
                        raise ValueError(f'Incorrect condition-matched demonstration bank: {left}/{right}')
    return bundle,configs

def accuracy_rows(model,strategy,bundle,refs):
    out=[]
    for task,records in bundle.items():
        if task=='baseline900':
            continue
        groups=defaultdict(list)
        for key,r in records.items():
            groups['overall'].append(key)
            groups['dataset:'+dataset(refs[task][key])].append(key)
            groups['choices:'+str(len(refs[task][key]['options']))].append(key)
        for group,ids in groups.items():
            c=sum(correct(records[i],refs[task][i]) for i in ids)
            low,high=wilson(c,len(ids))
            out.append(dict(model=model,strategy=strategy,task=task,group=group,N=len(ids),correct=c,
                accuracy_percent=pct(c,len(ids)),ci95_low=low,ci95_high=high,
                invalid=sum(records[i].get('pred_letter') is None for i in ids),
                truncated=sum(bool(records[i].get('truncated')) for i in ids),
                finalization_attempted=sum(bool(records[i].get('finalization_attempted')) for i in ids),
                finalization_succeeded=sum(records[i].get('finalization_succeeded') is True for i in ids),
                finalization_failed=sum(records[i].get('finalization_succeeded') is False for i in ids),
                generation_calls=sum(records[i].get('generation_calls',1) for i in ids),
                expected_task_N=len(refs[task]),task_complete=set(records)==set(refs[task])))
    return out

def transition_rows(model,strategy,bundle,refs):
    out=[]
    pairs=[('baseline','V2'),('baseline','T3'),('baseline','T2_present'),
           ('T2_present','T2_absent'),('V3_with','V3_without')]
    for a,b in pairs:
        if a not in bundle or b not in bundle:
            continue
        ids=sorted(set(bundle[a])&set(bundle[b]))
        out.append(dict(model=model,strategy=strategy,condition_a=a,condition_b=b,
            endpoints_differ=(a,b) in [('T2_present','T2_absent'),('V3_with','V3_without')],
            **mcnemar([correct(bundle[a][i],refs[a][i]) for i in ids],
                      [correct(bundle[b][i],refs[b][i]) for i in ids])))
    return out

def write_csv(path, rows):
    path=Path(path)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(rows)

def audit_root(result_root, data_root=ROOT, allow_partial=False):
    refs=validate_suite(data_root)
    target_maps={t:targets(t,data_root) for t in ('V2','T3')}
    scores,acc,transitions,audit=[],[],[],[]
    bundles={}
    for model in sorted(MODELS):
        bundle,configs=load_bundle(result_root,refs,model)
        if not bundle:
            continue
        strategy=next(iter(configs.values()))['strategy']
        bundles[model]=(strategy,bundle)
        acc.extend(accuracy_rows(model,strategy,bundle,refs))
        transitions.extend(transition_rows(model,strategy,bundle,refs))
        for task in NEEDED:
            result=score_task(task,bundle,refs,target_maps)
            complete=result is not None and result['N']==len(refs['V2'])
            audit.append(dict(model=model,strategy=strategy,task=task,complete=complete,
                paired_N=result['N'] if result else 0,expected_N=len(refs['V2']),
                missing_conditions=','.join(x for x in NEEDED[task] if x not in bundle)))
            if result is not None and (complete or allow_partial):
                protocol=next(iter(configs.values())).get('demonstrations',{}).get('protocol','generic_or_zero_shot')
                result['demonstration_policy']=protocol
                if protocol=='task_specific_mixed_v1':
                    result['prompt_comparison_note']=(
                        'V3/T2 condition pairs share fixed mixed examples. For V2/T2/T3, the original '
                        'baseline uses generic examples and the stress task uses adapted examples; '
                        'the score is under task-adapted prompting, not an isolated input-only intervention.')
                elif protocol=='task_specific_condition_groups_v2':
                    result['prompt_comparison_note']=(
                        'Each V3/T2 target condition receives its matching full three-example bank; '
                        'demonstrations change across conditions. These are condition-adapted prompting '
                        'scores, not effects of changing only the target input.')
                elif protocol=='task_specific_fixed_pairs_v3':
                    demo=next(c['demonstrations'] for c in configs.values() if c.get('demonstrations',{}).get('bank')==('V3_pairs' if task=='V3' else 'T2_pairs')) if task in ('V3','T2') else None
                    result['demonstration_pairs']=demo['pair_count'] if demo else 0
                    result['demonstration_examples']=2*demo['pair_count'] if demo else 3
                    result['prompt_comparison_note']=(
                        'V3/T2 use identical complete demonstration pairs on both target conditions. '
                        'V2/T3 retain three task-specific examples. The generic original baseline differs '
                        'from stress-task prefixes, so baseline-to-stress changes are not isolated input-only effects.')
                scores.append(dict(model=model,strategy=strategy,family=MODELS[model]['family'],
                                   exploratory_partial=not complete,**result))
    # Correction families are one condition comparison across all available models.
    for pair in {(r['condition_a'],r['condition_b']) for r in transitions}:
        holm([r for r in transitions if (r['condition_a'],r['condition_b'])==pair])
    return dict(scores=scores,accuracy=acc,transitions=transitions,coverage=audit),bundles,refs,target_maps

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=RESULTS_DIR)
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--allow-partial',action='store_true',help='Exploratory only; never treat as formal 900-item scores')
    a=p.parse_args()
    if not a.results.is_dir():
        raise SystemExit('No results yet. Run eval_local.py / eval_api.py first.')
    report,_,_,_=audit_root(a.results,a.data_root,a.allow_partial)
    if not report['accuracy']:
        raise SystemExit('No model result files found; no empty score tables written.')
    out=a.output or a.results/'statistics/output'
    out.mkdir(parents=True,exist_ok=True)
    for key,rows in report.items():
        write_csv(out/(key+'.csv'),rows)
    family=defaultdict(list)
    for r in report['scores']:
        if r['score_percent'] is not None:
            family[(r['strategy'],r['family'],r['task'])].append(r['score_percent'])
    write_csv(out/'family_scores.csv',[dict(strategy=s,family=f,task=t,n_models=len(v),
        complete_four_models=len(v)==4,mean_score_percent=sum(v)/len(v)) for (s,f,t),v in family.items()])
    report['notes'] = [
        'V3 primary denominator is O3 (correct with augmented options), NOT all N.',
        'V2 avoids a specified wrong target; invalid answers cannot count as target selection and are reported separately.',
        'T2 invalid answer-absent outputs count as missed NOTA; no interval is inferred from two independent binomials.',
        'McNemar is two-sided exact; Holm correction within each condition pair across the models in this report.',
        'V3 and T2-present/absent comparisons have different correctness targets; they are not same-label accuracy drops.',
        'No score is assigned to zero opportunities or incomplete pairs unless explicitly requested as exploratory.',
        'This script does not aggregate four tasks into the original six-task visual/text axes.'
    ]
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(f"Wrote {out}; {len(report['scores'])} task scores, {len(report['coverage'])} coverage audits")

if __name__=='__main__':
    main()
