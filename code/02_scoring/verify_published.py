"""Read-only regression of the selected published runs against shared scoring.

The snapshot's source_audit is authoritative: do not glob for newer reruns or
reparse stored model answers. Only --output is written, never source results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def equal(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)
    return left == right


def verify(args):
    core = args.core.resolve()
    for name in ('score_results.py',):
        if not (core / name).is_file():
            raise ValueError(f'Missing shared module: {core / name}')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '03_prompting/core'))
    sys.path.insert(0, str(core))
    scorer = importlib.import_module('score_results')
    common = importlib.import_module('common')
    registry = importlib.import_module('registry')
    if Path(scorer.__file__).resolve() != core / 'score_results.py':
        raise RuntimeError('Unexpected scoring module imported')

    snapshot_path = args.snapshot.resolve()
    snapshot_sha = file_sha(snapshot_path)
    snapshot = common.read_json(snapshot_path)
    data_root = args.data_root or registry.ROOT
    refs = common.validate_suite(data_root)
    targets = {task: common.targets(task, data_root) for task in ('V2', 'T3')}
    task_hashes = {task: file_sha(Path(data_root) / registry.TASK_FILES[task]) for task in refs}
    checked = Counter()
    issues = Counter()
    examples = []

    def check(category, ok, context=None):
        checked[category] += 1
        if not ok:
            issues[category] += 1
            if len(examples) < 100:
                examples.append({'category': category, 'context': context})

    models = snapshot['models']
    strategies = snapshot['strategies']
    expected_conditions = ('baseline', 'V2', 'V3_with', 'V3_without',
                           'T2_present', 'T2_absent', 'T3')
    expected_keys = {(m, s, t) for m in models for s in strategies for t in expected_conditions}
    audit_keys = [(r['model'], r['strategy'], r['condition']) for r in snapshot['source_audit']]
    check('source_inventory_unique_complete', len(audit_keys) == len(set(audit_keys))
          and set(audit_keys) == expected_keys,
          {'expected': len(expected_keys), 'actual': len(audit_keys)})
    bundles = defaultdict(dict)
    file_records = []
    prediction_rows = 0
    protected = {snapshot_path}

    for source in snapshot['source_audit']:
        model, strategy, task = source['model'], source['strategy'], source['condition']
        context = [model, strategy, task]
        path = Path(source['source_file']).resolve()
        protected.add(path)
        check('source_exists', path.is_file(), context)
        if not path.is_file():
            continue
        before = path.stat()
        hasher = hashlib.sha256()
        rows = {}
        n = 0
        local_issues = Counter()
        with path.open('rb') as stream:
            for line_number, raw in enumerate(stream, 1):
                hasher.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw.decode('utf-8-sig'))
                n += 1
                key = str(row.get('num'))
                if key in rows:
                    local_issues['duplicate_id'] += 1
                ref = refs[task].get(key)
                if ref is None:
                    local_issues['unexpected_id'] += 1
                    continue
                # Historical Original uses predicted_label/gold_label; prompt
                # results use pred_letter/gold_letter. Never guess or reparse.
                pred = row.get('pred_letter', row.get('predicted_label'))
                stored_gold = row.get('gold_letter', row.get('gold_label'))
                gold = common.gold(ref)
                valid = isinstance(pred, str) and pred in common.labels(ref['options'])
                is_correct = valid and pred == gold
                local_issues['invalid_prediction'] += int(not valid)
                local_issues['gold_mismatch'] += int(stored_gold != gold)
                local_issues['correct_flag_mismatch'] += int(row.get('correct') != is_correct)
                if strategy != 'original':
                    local_issues['item_hash_mismatch'] += int(row.get('item_sha256') != common.digest(ref))
                    local_issues['identity_mismatch'] += int(
                        row.get('model') != model or row.get('task') != task
                        or not str(row.get('strategy', '')).startswith(strategy))
                rows[key] = {k: row[k] for k in ('truncated', 'finalization_attempted',
                            'finalization_succeeded', 'generation_calls') if k in row}
                rows[key]['pred_letter'] = pred
        after = path.stat()
        actual_sha = hasher.hexdigest()
        prediction_rows += n
        check('source_unchanged_during_read',
              (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), context)
        check('source_hash_matches_snapshot', actual_sha == source['sha256'], context)
        check('input_hash_matches_snapshot', task_hashes[task] == source['input_sha256'], context)
        check('source_expected_count', n == source['rows'], context)
        check('source_complete_ids', set(rows) == set(refs[task]), context)
        for issue, count in local_issues.items():
            check('row_' + issue, count == 0, {'key': context, 'count': count})
        bundles[model, strategy][task] = rows
        file_records.append({'model': model, 'strategy': strategy, 'condition': task,
                             'source_file': str(path), 'rows': n, 'sha256': actual_sha,
                             'expected_sha256': source['sha256']})
        if len(file_records) % 24 == 0:
            print(f'Read {len(file_records)}/{len(audit_keys)} selected result files', file=sys.stderr)

    expected_accuracy = {(r['model'], r['strategy'], r['task']): r
                         for r in snapshot['accuracy'] if r['group'] == 'overall'}
    expected_scores = {(r['model'], r['strategy'], r['task']): r for r in snapshot['scores']}
    check('accuracy_inventory', set(expected_accuracy) == expected_keys)
    check('formal_inventory', set(expected_scores) == {
        (m, s, t) for m in models for s in strategies for t in ('V2', 'V3', 'T2', 'T3')})
    accuracy_results, score_results = [], []
    accuracy_fields = ('N', 'correct', 'accuracy_percent', 'ci95_low', 'ci95_high', 'invalid')
    score_fields = ('N', 'O', 'expected_N', 'coverage', 'score_percent',
                    'score_ci95_low', 'score_ci95_high', 'R', 'H', 'E', 'S', 'F', 'M',
                    'retention_percent', 'target_adoption_percent', 'other_new_error_percent',
                    'invalid_on_opportunities', 'rescued_original_wrong',
                    'paired_switch_success_percent', 'with_image_correct', 'without_image_correct',
                    'false_refusal_with_image', 'false_NOTA_percent', 'missed_NOTA_percent',
                    'both_availability_decisions_correct', 'both_availability_errors',
                    'false_NOTA_only', 'missed_NOTA_only', 'defined')
    for model in models:
        for strategy in strategies:
            bundle = bundles[model, strategy]
            complete = set(bundle) == set(expected_conditions) and all(
                set(bundle[t]) == set(refs[t]) for t in expected_conditions if t in bundle)
            check('bundle_complete', complete, [model, strategy])
            if not complete:
                continue
            for actual in scorer.accuracy_rows(model, strategy, bundle, refs):
                if actual['group'] != 'overall':
                    continue
                key = (model, strategy, actual['task'])
                expected = expected_accuracy.get(key)
                if expected is None:
                    continue
                for field in accuracy_fields:
                    check('accuracy_' + field, equal(actual[field], expected[field]),
                          {'key': key, 'actual': actual[field], 'expected': expected[field]})
                accuracy_results.append({k: actual[k] for k in
                                         ('model', 'strategy', 'task') + accuracy_fields})
            for task in ('V2', 'V3', 'T2', 'T3'):
                actual = scorer.score_task(task, bundle, refs, targets)
                key = (model, strategy, task)
                expected = expected_scores.get(key)
                check('formal_score_defined', actual is not None and actual.get('defined'), key)
                if actual is None or expected is None:
                    continue
                for field in score_fields:
                    if field in expected:
                        check('formal_' + field, field in actual and equal(actual[field], expected[field]),
                              {'key': key, 'field': field, 'actual': actual.get(field),
                               'expected': expected[field]})
                score_results.append({'model': model, 'strategy': strategy, **actual})
    check('snapshot_unchanged', file_sha(snapshot_path) == snapshot_sha)
    check('prediction_row_total', prediction_rows == snapshot['total_predictions'])
    check('accuracy_row_total', len(accuracy_results) == len(expected_keys))
    check('formal_row_total', len(score_results) == len(models) * len(strategies) * 4)
    output = args.output.resolve()
    if output in protected or any(output == (Path(data_root) / p).resolve()
                                  for p in registry.TASK_FILES.values()):
        raise ValueError('Output would overwrite an input; choose a new report path')
    report = {
        'passed': not bool(issues), 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Selected stored predictions, not a new inference run or reparsing of free text',
        'snapshot': str(snapshot_path), 'snapshot_sha256': snapshot_sha,
        'core': str(core), 'scorer_sha256': file_sha(core / 'score_results.py'),
        'data_root': str(data_root), 'models': len(models), 'strategies': len(strategies),
        'source_files': len(file_records), 'prediction_rows': prediction_rows,
        'formal_score_rows': len(score_results),
        'base_accuracy_rows': sum(r['task'] == 'baseline' for r in accuracy_results),
        'condition_accuracy_rows': len(accuracy_results), 'checks': dict(checked),
        'issue_counts': dict(issues), 'issue_examples': examples,
        'source_results_modified': False, 'sources': file_records,
        'scores': score_results, 'condition_accuracy': accuracy_results,
        'limitations': [
            'Retains the selected historical repair/recovery decisions recorded in source files.',
            'Does not adjudicate rationale semantics or declare historical parsing choices correct.',
            'Formal scores use each strategy\'s own opportunities; fixed-intersection sensitivity '
            'analyses are distinct and are not silently substituted.',
            'V1/T1 were not rerun in these prompt experiments and are outside this regression.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('passed', 'source_files', 'prediction_rows',
                     'formal_score_rows', 'base_accuracy_rows', 'condition_accuracy_rows', 'issue_counts')},
                     ensure_ascii=False))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', required=True, type=Path, help='Directory containing shared score_results.py')
    parser.add_argument('--snapshot', required=True, type=Path, help='Published comparison_12models.json')
    parser.add_argument('--output', required=True, type=Path, help='New JSON verification report (only written file)')
    parser.add_argument('--data-root', type=Path, help='Original Benchmark input-data root')
    return verify(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
