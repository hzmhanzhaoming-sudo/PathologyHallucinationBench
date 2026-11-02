"""Verify the new 01-to-02 result layout using isolated historical-label fixtures.

No inference, model import, API request, answer reparsing or source writes.
The fixture is NOT a new experiment: every original row is preserved under
source_record, while its stored labels/correct flag are copied to the canonical
new-layout envelope. T1 selects only the canonical U1/U2 subset IDs.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


MODEL_KEY = 'qwen3.5_4B'
PACKAGE_MODEL = 'qwen35-4b'
COLUMNS = {'V1': 'V1_visual_shortcut_resistance', 'V2': 'V2_conflict_target_resistance',
           'V3': 'V3_modality_switch_resistance', 'T1': 'T1_unknown_shortcut_resistance',
           'T2': 'T2_answer_absence_resistance', 'T3': 'T3_wrong_hint_correctness_retention'}


def default_code_root():
    parent = Path(__file__).resolve().parent.parent
    for candidate in (parent / 'CODE', parent):
        if (candidate / '01_stress_tests/registry.py').is_file():
            return candidate
    return None


def source_result(data_root, condition):
    specifications = {
        'Base': (1, None, '*_with_image.jsonl'),
        'V1_without': (1, None, '*_without_image.jsonl'),
        'V2': (4, None, '*_with_image.jsonl'),
        'V3_with': (6, None, '*_with_image.jsonl'),
        'V3_without': (6, None, '*_without_image.jsonl'),
        'T2_present': (5, 'V1', '*_with_image.jsonl'),
        'T2_absent': (5, 'V2', '*_with_image.jsonl'),
        'T3': (7, None, '*_with_image.jsonl'),
    }
    if condition.startswith('T1_'):
        choices = condition.split('_')[1].replace('choice', '')
        folder = data_root / f'Benchmark_T3_results/{choices}-choice/{MODEL_KEY}_T3_results'
        pattern = '*_with_image.jsonl'
    else:
        task, subfolder, pattern = specifications[condition]
        folder = data_root / f'Benchmark_T{task}_results'
        if subfolder:
            folder /= subfolder
        folder /= f'{MODEL_KEY}_T{task}_results'
    files = sorted(folder.glob(pattern))
    if len(files) != 1:
        raise ValueError(f'Expected one result source for {condition}, found {files}')
    return files[0].resolve()


def ensure_safe_output(output, data_root):
    expanded = output.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise ValueError('Output must be a NEW directory; no fixtures or reports are overwritten.')
    out = expanded.resolve()
    if out == data_root or out in data_root.parents:
        raise ValueError('Output cannot be the data root or its ancestor.')
    protected = [data_root / f'Benchmark_T{i}_{suffix}'
                 for i in range(1, 8) for suffix in ('results', 'data')]
    protected.extend(data_root.glob('Benchmark_T*_results'))
    protected.extend(data_root.glob('Benchmark_T*_data'))
    protected.extend(data_root / name for name in (
        'Benchmark_Zero-shot CoT', 'Benchmark_Few-shot direct', 'Benchmark_Few-shot CoT',
        'Draw_picture'))
    if any(folder.resolve() == out or folder.resolve() in out.parents for folder in protected):
        raise ValueError('Historical data, predictions and figures are read-only.')
    return out


def gold_letter(reference):
    labels = [re.match(r'^\s*([A-Z])\s*[).:\-]', value, re.I).group(1).upper()
              for value in reference['options']]
    answer = str(reference['answer']).strip()
    if answer.upper() in labels:
        return answer.upper()
    if answer in reference['options']:
        return labels[reference['options'].index(answer)]
    raise ValueError('Reference gold is not a legal option.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=default_code_root())
    parser.add_argument('--raw-audit', type=Path,
                        help='Optional independently recomputed original_raw_check.json for exact-value comparison.')
    args = parser.parse_args()
    if args.code_root is None:
        parser.error('--code-root is required when the CODE layout cannot be found beside this verifier.')
    root = args.data_root.expanduser().resolve()
    code = args.code_root.expanduser().resolve()
    output = ensure_safe_output(args.output, root)
    registry_path = code / '01_stress_tests/registry.py'
    scorer_path = code / '02_scoring/score_stress_run.py'
    if not registry_path.is_file() or not scorer_path.is_file():
        raise ValueError('The complete 01/02/03 CODE layout is required.')
    sources = {}
    def read_hashed(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return data
    def load_json(path):
        return json.loads(read_hashed(path).decode('utf-8-sig'))
    read_hashed(registry_path)
    read_hashed(scorer_path)
    # Use a unique module name, never the generic 'registry' used by package 02.
    spec = importlib.util.spec_from_file_location('_fixture_stress_registry', registry_path)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    conditions = registry.expand_task('all')
    if len(conditions) != 11 or PACKAGE_MODEL not in registry.MODELS:
        raise ValueError('Unexpected first-package inventory.')
    audit = load_json(args.raw_audit) if args.raw_audit else None
    if audit is not None and not audit.get('passed'):
        raise ValueError('Supplied raw audit did not pass.')
    buffers, plans, selections = {}, [], []
    fixture = output / 'verification_fixture' / PACKAGE_MODEL
    for condition in conditions:
        relative, _, expected, _ = registry.CONDITIONS[condition]
        input_path = (root / relative).resolve()
        references = load_json(input_path)
        by_id = {str(row['num']): row for row in references}
        if len(references) != expected or len(by_id) != expected:
            raise ValueError('Canonical input count or uniqueness mismatch: ' + condition)
        if condition.startswith('T1_') and any(int(r['t3_unknown_count']) not in (1, 2) for r in references):
            raise ValueError('Canonical T1 fixture may only contain U1/U2.')
        source_path = source_result(root, condition)
        raw = read_hashed(source_path)
        original_rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        seen = set()
        selected = []
        for line_number, row in enumerate(original_rows, 1):
            key = str(row['num'])
            if key in seen:
                raise ValueError('Duplicate historical result ID: ' + condition + '/' + key)
            seen.add(key)
            if key not in by_id:
                if condition.startswith('T1_'):
                    continue
                raise ValueError('Historical result ID outside canonical condition.')
            ref = by_id[key]
            if 'pred_letter' in row and 'predicted_label' in row and row['pred_letter'] != row['predicted_label']:
                raise ValueError('Conflicting stored prediction aliases.')
            if 'gold_letter' in row and 'gold_label' in row and row['gold_letter'] != row['gold_label']:
                raise ValueError('Conflicting stored gold aliases.')
            if not ('pred_letter' in row or 'predicted_label' in row):
                raise ValueError('No stored prediction label.')
            prediction = row.get('pred_letter', row.get('predicted_label'))
            gold = row.get('gold_letter', row.get('gold_label'))
            if gold != gold_letter(ref):
                raise ValueError('Historical stored gold differs from canonical input.')
            if type(row.get('correct')) is not bool or row['correct'] != (prediction is not None and prediction == gold):
                raise ValueError('Historical correctness flag is inconsistent.')
            if 'options' in row and row['options'] != ref['options']:
                raise ValueError('Historical options differ from canonical subset.')
            if 'question' in row and row['question'] != ref['question']:
                raise ValueError('Historical question differs from canonical subset.')
            selected.append(dict(
                num=key, pred_letter=prediction, gold_letter=gold, correct=row['correct'],
                source_file=str(input_path), source_record=row,
                verification_fixture=True,
                provenance=dict(source_result_file=str(source_path), source_result_sha256=sources[str(source_path)],
                                original_record_index=line_number,
                                original_source_file=row.get('source_file'),
                                canonical_input=str(input_path), canonical_input_sha256=sources[str(input_path)],
                                transformation='Stored labels/correct flag copied unchanged; source_file normalized only in fixture envelope; complete original record retained.')))
        if {row['num'] for row in selected} != set(by_id):
            raise ValueError('Incomplete canonical selected predictions: ' + condition)
        if audit is not None and audit['sources'].get(str(source_path)) != sources[str(source_path)]:
            raise ValueError('Raw audit and current source hash differ: ' + str(source_path))
        buffers[condition] = selected
        selections.append(dict(condition=condition, source_result=str(source_path), rows_read=len(original_rows),
                               rows_selected=len(selected), source_sha256=sources[str(source_path)]))
        plans.append(dict(condition=condition, input=str(input_path), input_expected_rows=expected,
                          input_sha256=sources[str(input_path)], output=str(fixture / (condition + '.jsonl')),
                          source_result=str(source_path), source_result_sha256=sources[str(source_path)],
                          verification_fixture=True))
    golden_path = root / 'Draw_picture/Fig_sum/six_task_robustness_scores.csv'
    golden_rows = list(csv.DictReader(read_hashed(golden_path).decode('utf-8-sig').splitlines()))
    golden_matches = [row for row in golden_rows if row['model_key'] == MODEL_KEY]
    if len(golden_matches) != 1:
        raise ValueError('Expected one golden row for ' + MODEL_KEY)
    golden = golden_matches[0]
    base_correct = sum(row['correct'] for row in buffers['Base'])
    base_expected = 100 * base_correct / len(buffers['Base'])
    manifest = dict(mode='verification_fixture', model=PACKAGE_MODEL, original_model_key=MODEL_KEY,
                    task='all', data_root=str(root), model_calls=0, api_calls=0, plan=plans,
                    note='Historical predictions normalized solely to test new layout/scorer integration. NOT a new model run.')
    # All source material is validated in memory before the first output write.
    for path, expected_hash in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Source changed during fixture preparation: ' + path)
    output.mkdir(parents=True, exist_ok=False)
    fixture.mkdir(parents=True, exist_ok=False)
    for condition, records in buffers.items():
        with (fixture / (condition + '.jsonl')).open('x', encoding='utf8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    (fixture / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    command = [sys.executable, '-B', str(scorer_path), '--run', str(fixture),
               '--output', str(output / 'scored'), '--data-root', str(root)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf8', check=False)
    (output / 'scorer_stdout.txt').write_text(result.stdout, encoding='utf8')
    (output / 'scorer_stderr.txt').write_text(result.stderr, encoding='utf8')
    checks = []
    if result.returncode == 0:
        score_report = json.loads((output / 'scored/scores.json').read_text(encoding='utf8'))
        calculated = {row['task']: row for row in score_report['scores']}
        if set(calculated) != {'Base', *COLUMNS}:
            raise ValueError('Unexpected scorer inventory.')
        for task, row in calculated.items():
            actual = row['score_percent']
            reference = base_expected if task == 'Base' else float(golden[COLUMNS[task]])
            tolerance = 0.0005000001 if task in ('V2', 'T3') else 1e-9
            check = dict(task=task, actual=actual, reference=reference, difference=actual-reference,
                         tolerance=tolerance, published_precision_matches=abs(actual-reference) <= tolerance,
                         one_decimal_matches=f'{actual:.1f}' == f'{reference:.1f}',
                         formal_complete=row['formal_complete'])
            if audit is not None and task != 'Base':
                candidates = [item for item in audit['scores'] if item['model_key'] == MODEL_KEY and item['task'] == task]
                if len(candidates) != 1:
                    raise ValueError('Raw audit model/task inventory mismatch.')
                exact = candidates[0]['raw_score']
                check.update(independent_raw_score=exact, independent_raw_difference=actual-exact,
                             independent_raw_matches=abs(actual-exact) <= 1e-9)
            checks.append(check)
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in sources.items())
    passed = result.returncode == 0 and len(checks) == 7 and unchanged and all(
        c['published_precision_matches'] and c['one_decimal_matches'] and c['formal_complete']
        and c.get('independent_raw_matches', True) for c in checks)
    report = dict(passed=passed, type='historical-label new-layout integration verification',
                  original_model_key=MODEL_KEY, package_model=PACKAGE_MODEL, model_calls=0, api_calls=0,
                  source_results_modified=False, source_hashes_unchanged=unchanged, raw_output_reparsed=False,
                  fixture_is_new_experiment=False, selected_rows=sum(len(rows) for rows in buffers.values()),
                  selection=selections, sources=sources, scorer_command=command, scorer_returncode=result.returncode,
                  base_raw_correct=base_correct, base_raw_total=len(buffers['Base']), checks=checks,
                  precision_note='V2/T3 golden CSV retains historical three-decimal percentage source rounding; all other exact tolerances 1e-9. Full differences are reported.')
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('sources', 'selection', 'scorer_command')}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
