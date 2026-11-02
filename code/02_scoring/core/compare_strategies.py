"""Paired strategy comparison, with both own-baseline and common-opportunity scores."""
import argparse
import itertools
from pathlib import Path
from score_results import audit_root, correct, mcnemar, holm, score_task, write_csv, load_bundle
from common import digest, read_json
from registry import ROOT
from layout import RESULTS_DIR


def comparison_controls(args, allow_task_demos=False):
    result={k:v for k,v in args.items() if k!='strategy'}
    if not allow_task_demos:
        return result
    compatibility=read_json(Path(__file__).resolve().parent/'task_demo_compatibility.json')
    actual=result.pop('code_sha256',None)
    allowed=compatibility['reviewed_code_sha256']
    if actual is None or actual not in allowed:
        raise ValueError('Unreviewed code version; cannot compare across task-demonstration protocols')
    policy=result.pop('demonstration_policy',None)
    if policy not in (None,'task_specific_fixed_pairs_v3'):
        raise ValueError('Unsupported demonstration protocol')
    for key in ('demo_suite','demo_suite_sha256'):
        result.pop(key,None)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',type=Path,required=True,help='Two or more result directories')
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=RESULTS_DIR/'statistics/strategy_comparison')
    p.add_argument('--allow-task-demo-comparison',action='store_true',help=(
        'Explicitly compare reviewed old zero-shot code and new task-specific few-shot code; '
        'this is a protocol comparison, not an isolated CoT effect. All decoding controls remain checked.'))
    a=p.parse_args()
    reports=[]
    for root in a.roots:
        report,bundles,refs,targets=audit_root(root,a.data_root)
        reports.append((root,report,bundles))
    paired,score_deltas,own=[],[],[]
    for root,report,_ in reports:
        own.extend(dict(result_root=str(root),**r) for r in report['scores'])
    for (root_a,report_a,models_a),(root_b,report_b,models_b) in itertools.combinations(reports,2):
        for model in sorted(set(models_a)&set(models_b)):
            _,configs_a=load_bundle(root_a,refs,model)
            _,configs_b=load_bundle(root_b,refs,model)
            args_a=next(iter(configs_a.values()))['arguments']
            args_b=next(iter(configs_b.values()))['arguments']
            if digest(comparison_controls(args_a,a.allow_task_demo_comparison))!=digest(comparison_controls(args_b,a.allow_task_demo_comparison)):
                raise ValueError(f'Non-strategy settings differ for {model}; for reviewed task-demo protocols '
                                 'use --allow-task-demo-comparison. Decoding differences still require separate analysis.')
            strategy_a,ba=models_a[model]
            strategy_b,bb=models_b[model]
            identity=dict(model=model,strategy_a=strategy_a,strategy_b=strategy_b,
                          result_root_a=str(root_a),result_root_b=str(root_b),
                          demonstration_policy_a=args_a.get('demonstration_policy','generic_or_zero_shot'),
                          demonstration_policy_b=args_b.get('demonstration_policy','generic_or_zero_shot'),
                          comparison_scope='task-adapted prompting protocol comparison' if a.allow_task_demo_comparison else 'matched implementation')
            for task in sorted(set(configs_a)&set(configs_b)):
                da,db=(c[task].get('demonstrations',{}) for c in (configs_a,configs_b))
                if da.get('protocol')==db.get('protocol') and da.get('protocol') in ('task_specific_mixed_v1','task_specific_condition_groups_v2','task_specific_fixed_pairs_v3'):
                    if da.get('suite_sha256')!=db.get('suite_sha256') or da.get('input_content')!=db.get('input_content'):
                        raise ValueError('Different example inputs/suites across few-shot strategies')
            for condition in sorted(set(ba)&set(bb)):
                if condition=='baseline900':
                    continue
                ids=sorted(set(ba[condition])&set(bb[condition]))
                expected=set(refs[condition])
                # baseline900 is an explicitly eligible stress-test baseline, not full accuracy.
                if condition=='baseline' and set(ids)==set(refs['V2']):
                    label='baseline900'
                elif set(ids)!=expected:
                    continue
                else:
                    label=condition
                paired.append(dict(**identity,condition=label,**mcnemar(
                    [correct(ba[condition][i],refs[condition][i]) for i in ids],
                    [correct(bb[condition][i],refs[condition][i]) for i in ids])))
            for task in ('V2','V3','T2','T3'):
                sa=score_task(task,ba,refs,targets)
                sb=score_task(task,bb,refs,targets)
                if sa is None or sb is None or sa['coverage']!=1 or sb['coverage']!=1:
                    continue
                base='V3_with' if task=='V3' else 'baseline'
                common={i for i in refs['V2'] if correct(ba[base][i],refs[base][i]) and correct(bb[base][i],refs[base][i])}
                ca=score_task(task,ba,refs,targets,common)
                cb=score_task(task,bb,refs,targets,common)
                score_deltas.append(dict(**identity,task=task,opportunities_a=sa['O'],opportunities_b=sb['O'],
                    own_score_a=sa['score_percent'],own_score_b=sb['score_percent'],
                    common_opportunities=len(common),common_score_a=ca['score_percent'],common_score_b=cb['score_percent'],
                    common_delta_pp=(cb['score_percent']-ca['score_percent']) if common else None))
    # Explicit multiple-testing family: all model x strategy-pair comparisons per condition.
    for condition in {r['condition'] for r in paired}:
        holm([r for r in paired if r['condition']==condition])
    if not paired:
        raise SystemExit('No complete matched strategy pairs; no comparison table written')
    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'paired_accuracy_comparisons.csv',paired)
    write_csv(a.output/'scores_with_common_opportunities.csv',score_deltas)
    write_csv(a.output/'original_scores_each_strategy.csv',own)
    print(f'Wrote {a.output}; {len(paired)} matched comparisons')

if __name__=='__main__':
    main()
