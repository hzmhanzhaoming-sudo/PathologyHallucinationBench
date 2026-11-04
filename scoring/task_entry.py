"""Shared implementation for six independent task entry points."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluation.schema import CONDITIONS, load_predictions, new_output, sha256, write_json
from scoring.metrics import TASK_CONDITIONS, cohort_ids, compute_task
from scoring.task_data import load_task_inputs


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export an empty report')
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def selected_manifest(path, required):
    """A task can be scored once its own outputs are registered, even mid-run."""
    if not path.is_file():
        return None
    m = json.loads(path.read_text(encoding='utf-8'))
    if m.get('schema_version') != 1 or m.get('status') not in ('complete', 'running', 'failed'):
        raise ValueError(f'{path}: unsupported run manifest/status')
    inputs = {CONDITIONS[n][0]: m.get('input_sha256', {}).get(CONDITIONS[n][0]) for n in required}
    predictions = {n + '.jsonl': m.get('prediction_sha256', {}).get(n + '.jsonl') for n in required}
    if not all(inputs.values()) or not all(predictions.values()):
        raise ValueError(f'{path}: this task is not completed/registered yet')
    return dict(model_key=m.get('model_key', path.parent.name), model=m.get('model', path.parent.name),
                input_sha256=inputs, prediction_sha256=predictions)


def main(task, argv=None):
    p = argparse.ArgumentParser(description=f'Score {task} only; no unrelated task files or model calls are required.')
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--runs', required=True, type=Path, nargs='+')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--allow-custom-size', action='store_true')
    a = p.parse_args(argv)
    runs = [r.expanduser().resolve() for r in a.runs]
    out = new_output(a.output, [a.data_root, *runs, Path(__file__).parent, Path(__file__).parents[1] / 'evaluation'])
    required = TASK_CONDITIONS[task]
    refs, hashes = load_task_inputs(task, a.data_root, a.allow_custom_size)
    sources = {str((a.data_root / path).resolve()): digest for path, digest in hashes.items()}
    scores, events, accuracy, counts, metadata = [], [], [], [], {}
    manifest_checks = []
    for run in runs:
        mpath = run / 'run_manifest.json'
        m = selected_manifest(mpath, required)
        if m is not None and m['input_sha256'] != hashes:
            raise ValueError(f'{run}: input hashes differ from reference data')
        model = m['model_key'] if m else run.name
        if model in metadata:
            raise ValueError(f'Duplicate model key: {model}')
        label = m['model'] if m else model
        predictions, prediction_hashes = {}, {}
        for condition in required:
            path = run / (condition + '.jsonl')
            digest = sha256(path)
            sources[str(path)] = digest
            prediction_hashes[condition] = digest
            if m and m['prediction_sha256'][path.name] != digest:
                raise ValueError(f'{path}: prediction file changed after registration')
            predictions[condition] = load_predictions(path, refs[condition])
        value, event, rows, base_counts = compute_task(task, refs, predictions)
        scores.append(dict(model_key=model, model=label, **{task: value}))
        events.append(dict(model_key=model, task=task, **event))
        accuracy.extend(dict(model_key=model, **row) for row in rows)
        counts.extend(dict(model_key=model, **row) for row in base_counts)
        metadata[model] = dict(model=label, run_directory=str(run), prediction_sha256=prediction_hashes,
                               manifest_verified=m is not None)
        manifest_checks.append((mpath, m))
    if any(sha256(path) != digest for path, digest in sources.items()):
        raise ValueError('Input changed during scoring')
    for path, initial in manifest_checks:
        if selected_manifest(path, required) != initial:
            raise ValueError('Selected task manifest changed during scoring')
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / f'{task}_scores.csv', scores)
    write_csv(out / 'task_event_counts.csv', events)
    write_csv(out / 'condition_accuracy.csv', accuracy)
    if task == 'V1':
        write_csv(out / 'base_dataset_counts.csv', counts)
    report = dict(schema_version=2, task=task, models=len(scores), custom_size=a.allow_custom_size,
                  input_sha256=hashes, cohort_sha256=signature(cohort_ids(task, refs)),
                  model_metadata=metadata, sources=sources, model_calls=0, api_calls=0,
                  undefined_tasks=[r['model_key'] for r in events if not r['defined']],
                  output_sha256={path.name: sha256(path) for path in out.glob('*.csv')})
    write_json(out / 'provenance.json', report)
    print(json.dumps(dict(task=task, models=len(scores), output=str(out), undefined=report['undefined_tasks']), ensure_ascii=False))
    return report


def cli(task):
    try:
        main(task)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
