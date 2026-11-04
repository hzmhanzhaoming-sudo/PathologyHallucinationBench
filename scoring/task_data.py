"""Task-scoped validation: unrelated tasks are never required or read."""
from evaluation.schema import ABSTAIN, NOTA, T1_CONDITIONS, dataset, gold, load_conditions, options, special_label
from scoring.metrics import TASK_CONDITIONS, cohort_ids


def load_task_inputs(task, data_root, custom_size=False):
    refs, hashes = load_conditions(data_root, TASK_CONDITIONS[task], custom_size=custom_size)
    validate_task_pairs(task, refs, custom_size)
    return refs, hashes


def validate_task_pairs(task, g, custom_size=False):
    if task == 'T1' and any('t1_original_num' not in row for name in T1_CONDITIONS for row in g[name].values()):
        raise ValueError('T1: missing t1_original_num mapping')
    ids = set(cohort_ids(task, g))
    if not ids:
        raise ValueError(f'{task}: no paired items')
    if task != 'V3' and not ids <= set(g['V1_with_image']):
        raise ValueError(f'{task}: task IDs map outside baseline')
    if task == 'V1':
        if ids != set(g['V1_without_image']):
            raise ValueError('V1 paired IDs differ')
        for i in ids:
            a, b = g['V1_with_image'][i], g['V1_without_image'][i]
            if a['question'] != b['question'] or a['options'] != b['options'] or gold(a) != gold(b):
                raise ValueError(f'V1/{i}: only image availability may differ')
            dataset(a)
    elif task in ('V2', 'T3'):
        for i in ids:
            a, b = g['V1_with_image'][i], g[task][i]
            if a['options'] != b['options'] or gold(a) != gold(b):
                raise ValueError(f'{task}/{i}: baseline answer/options changed')
            if task == 'V2' and (b.get('target_wrong_option') not in options(b) or b['target_wrong_option'] == gold(b)):
                raise ValueError(f'V2/{i}: invalid target_wrong_option')
    elif task == 'V3':
        if ids != set(g['V3_without_image']):
            raise ValueError('V3 paired IDs differ')
        for i in ids:
            a, b = g['V3_with_image'][i], g['V3_without_image'][i]
            if a['options'] != b['options'] or a['question'] != b['question']:
                raise ValueError(f'V3/{i}: paired question/options differ')
            if gold(b) != special_label(b, ABSTAIN) or gold(a) == gold(b):
                raise ValueError(f'V3/{i}: incorrect image-availability reference')
    elif task == 'T2':
        if ids != set(g['T2_absent']):
            raise ValueError('T2 paired IDs differ')
        for i in ids:
            a, b = g['T2_present'][i], g['T2_absent'][i]
            if gold(a) == special_label(a, NOTA) or gold(b) != special_label(b, NOTA):
                raise ValueError(f'T2/{i}: incorrect answer-availability reference')
    else:
        seen = {1: set(), 2: set()}
        for name in T1_CONDITIONS:
            for row in g[name].values():
                source, level = str(row.get('t1_original_num', '')), row.get('t1_unknown_count')
                if type(level) is not int or level not in seen or source not in g['V1_with_image'] or source in seen[level]:
                    raise ValueError(f'T1: invalid/duplicate mapping {row["num"]}')
                if gold(row) != gold(g['V1_with_image'][source]):
                    raise ValueError(f'T1/{source}: correct answer label changed')
                seen[level].add(source)
        if seen[1] != seen[2] or not seen[1] or not custom_size and len(seen[1]) != 900:
            raise ValueError('T1: levels must cover the same complete core')
