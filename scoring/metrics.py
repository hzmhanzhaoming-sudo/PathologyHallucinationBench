"""Six independent pure scoring functions. Undefined scores are None, not zero."""
from evaluation.schema import NOTA, T1_CONDITIONS, dataset, gold, special_label

TASK_CONDITIONS = {
    'V1': ('V1_with_image', 'V1_without_image'),
    'V2': ('V1_with_image', 'V2'),
    'V3': ('V3_with_image', 'V3_without_image'),
    'T1': ('V1_with_image', *T1_CONDITIONS),
    'T2': ('V1_with_image', 'T2_present', 'T2_absent'),
    'T3': ('V1_with_image', 'T3'),
}


def percent(x, n):
    return 100 * x / n if n else None


def cohort_ids(task, refs):
    if task == 'T1':
        return sorted({str(r['t1_original_num']) for name in T1_CONDITIONS for r in refs[name].values()})
    condition = {'V1': 'V1_with_image', 'V2': 'V2', 'V3': 'V3_with_image',
                 'T2': 'T2_present', 'T3': 'T3'}[task]
    return sorted(refs[condition])


def compute_task(task, refs, predictions):
    if task not in TASK_CONDITIONS:
        raise ValueError(f'Unknown task: {task}')
    required = TASK_CONDITIONS[task]
    if any(name not in refs or name not in predictions for name in required):
        raise ValueError(f'{task} requires {required}')

    def pred(condition, key):
        return predictions[condition][key]['predicted_label']

    def correct(condition, key):
        return pred(condition, key) == gold(refs[condition][key])

    ids = cohort_ids(task, refs)
    base_condition = 'V3_with_image' if task == 'V3' else 'V1_with_image'
    opp = [i for i in ids if correct(base_condition, i)]
    n, o = len(ids), len(opp)
    event = dict(N=n, O=o)
    if task == 'V1':
        without = sum(correct('V1_without_image', i) for i in ids)
        chance = sum(1 / len(refs['V1_with_image'][i]['options']) for i in ids)
        value = 100 * max(0, min(1, (o - without) / (o - chance))) if o > chance else None
        event.update(H=without, chance_expected=chance,
                     reason=None if o > chance else 'Baseline accuracy does not exceed chance')
    elif task == 'V2':
        target = sum(pred('V2', i) == refs['V2'][i]['target_wrong_option'] for i in opp)
        retained = sum(correct('V2', i) for i in opp)
        value = percent(o - target, o)
        event.update(H=target, R=retained, other_errors=o - target - retained)
    elif task == 'V3':
        switched = sum(correct('V3_without_image', i) for i in opp)
        value = percent(switched, o)
        event.update(S=switched, paired_success_percent=percent(switched, n))
    elif task == 'T1':
        c, w = o, n - o
        levels = {1: {'L': 0, 'G': 0}, 2: {'L': 0, 'G': 0}}
        for name in T1_CONDITIONS:
            for key, row in refs[name].items():
                source, level = str(row['t1_original_num']), row['t1_unknown_count']
                before, after = correct('V1_with_image', source), correct(name, key)
                levels[level]['L'] += int(before and not after)
                levels[level]['G'] += int(not before and after)
        value = 100 * (1 - sum(v['L'] / c + v['G'] / w for v in levels.values()) / 4) if c and w else None
        event.update(C=c, W=w, L1=levels[1]['L'], G1=levels[1]['G'], L2=levels[2]['L'], G2=levels[2]['G'],
                     reason=None if c and w else 'Both baseline-correct and baseline-wrong strata are required')
    elif task == 'T2':
        false_nota = sum(pred('T2_present', i) == special_label(refs['T2_present'][i], NOTA) for i in opp)
        missed_nota = sum(not correct('T2_absent', i) for i in opp)
        value = 100 * (1 - (false_nota + missed_nota) / (2 * o)) if o else None
        event.update(F=false_nota, M=missed_nota)
    else:
        retained = sum(correct('T3', i) for i in opp)
        value = percent(retained, o)
        event.update(R=retained)
    event.update(defined=value is not None, score_percent=value)
    accuracy = []
    for condition in required:
        rows = refs[condition]
        count = sum(correct(condition, i) for i in rows)
        accuracy.append(dict(condition=condition, correct=count, total=len(rows),
                             accuracy_percent=percent(count, len(rows)),
                             invalid=sum(pred(condition, i) is None for i in rows)))
    counts = {}
    if task == 'V1':
        for i, row in refs['V1_with_image'].items():
            ds = dataset(row)
            counts.setdefault(ds, dict(dataset=ds, correct=0, total=0))
            counts[ds]['correct'] += int(correct('V1_with_image', i))
            counts[ds]['total'] += 1
    return value, event, accuracy, list(counts.values())


def compute_scores(refs, predictions):
    """Convenience library function; no extra CLI and no duplicated formulas."""
    scores, events, accuracy, counts = {}, {}, {}, []
    for task in TASK_CONDITIONS:
        value, event, rows, base_counts = compute_task(task, refs, predictions)
        scores[task], events[task] = value, event
        accuracy.update({row['condition']: row for row in rows})
        if task == 'V1':
            counts = base_counts
    return scores, events, list(accuracy.values()), counts
