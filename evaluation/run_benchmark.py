"""Plan all 11 public-layout conditions; inference requires explicit --execute.

This dispatches the existing public evaluation/ runners. It does not reconstruct
the manuscript-era task-specific prompts in evaluation/manuscript/vendor/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    'V1_with_image': ('V1/v1_with_image.json', 1200, 1),
    'V1_without_image': ('V1/v1_without_image.json', 1200, 0),
    'V2': ('V2/v2_image_text_conflict_with_image.json', 900, 1),
    'V3_with_image': ('V3/v3_with_image.json', 900, 1),
    'V3_without_image': ('V3/v3_without_image.json', 900, 0),
    'T1_4choice': ('T1/4-choice/t1_4choice_unknown1_2_with_image.json', 1000, 1),
    'T1_6choice': ('T1/6-choice/t1_6choice_unknown1_2_with_image.json', 400, 1),
    'T1_7choice': ('T1/7-choice/t1_7choice_unknown1_2_with_image.json', 400, 1),
    'T2_present': ('T2/t2_answer_present_with_image.json', 900, 1),
    'T2_absent': ('T2/t2_answer_absent_with_image.json', 900, 1),
    'T3': ('T3/t3_wrong_answer_hint_with_image.json', 900, 1),
}
RUNNERS = tuple(p.name for p in sorted((ROOT/'evaluation').glob('eval_*.py')))


def read_records(path, expected):
    raw = Path(path).read_bytes()
    records = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError(f'{path}: expected {expected} records')
    ids = [str(r.get('num')) for r in records]
    if len(set(ids)) != len(ids) or any(r.get('num') in (None, '') for r in records):
        raise ValueError(f'{path}: missing or duplicate IDs')
    for row in records:
        options = row.get('options', [])
        labels = [re.match(r'^([A-H])[).:]\s*', str(o)) for o in options]
        if len(labels) < 2 or not all(labels):
            raise ValueError(f'{path}/{row["num"]}: invalid options')
        legal = {m.group(1) for m in labels}
        if len(legal) != len(options):
            raise ValueError(f'{path}/{row["num"]}: duplicate option labels')
        answer = str(row.get('answer', '')).strip()
        if answer not in legal and answer not in options:
            raise ValueError(f'{path}/{row["num"]}: answer is not an option')
        if not str(row.get('question', '')).strip():
            raise ValueError(f'{path}/{row["num"]}: empty question')
    return records, hashlib.sha256(raw).hexdigest()


def validate_inputs(data_root, image_root=None):
    """Read-only structure, pairing, wrong-target and optional image checks."""
    data_root = Path(data_root).resolve()
    groups, hashes = {}, {}
    for name, (relative, count, with_image) in CONDITIONS.items():
        rows, digest = read_records(data_root/relative, count)
        hashes[relative] = digest
        groups[name] = {str(row['num']): row for row in rows}
        for row in rows:
            image = row.get('img', '')
            if bool(image) != bool(with_image):
                raise ValueError(f'{name}/{row["num"]}: wrong image-availability condition')
            if image:
                if Path(image).name != image or '/' in image or '\\' in image:
                    raise ValueError('Public image references must be basenames, not absolute/traversing paths')
                if image_root is not None and not (Path(image_root)/image).is_file():
                    raise FileNotFoundError(Path(image_root)/image)
    baseline = set(groups['V1_with_image'])
    if baseline != set(groups['V1_without_image']):
        raise ValueError('V1 paired IDs differ')
    core = set(groups['V2'])
    if not core <= baseline:
        raise ValueError('V2 maps outside baseline')
    for name in ('V3_with_image', 'V3_without_image', 'T2_present', 'T2_absent', 'T3'):
        if set(groups[name]) != core:
            raise ValueError(f'{name}: 900-question core IDs differ')
    seen = {1: set(), 2: set()}
    for name in ('T1_4choice', 'T1_6choice', 'T1_7choice'):
        for row in groups[name].values():
            num = str(row.get('t1_original_num', row.get('t3_original_num', '')))
            level = row.get('t1_unknown_count', row.get('t3_unknown_count'))
            if level not in seen or num not in core or num in seen[level]:
                raise ValueError(f'{name}: missing/duplicate/out-of-core T1 mapping')
            seen[level].add(num)
    if any(ids != core for ids in seen.values()):
        raise ValueError('T1 levels do not each cover the complete core')
    for row in groups['V2'].values():
        target = row.get('target_wrong_option')
        gold = str(row['answer']).strip()[0]
        if target not in {option[0] for option in row['options']} or target == gold:
            raise ValueError(f'Invalid V2 target_wrong_option: {row["num"]}')
    return {'files': len(groups), 'records': sum(len(g) for g in groups.values()),
            'images_checked': image_root is not None, 'sha256': hashes}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', choices=RUNNERS, required=True)
    p.add_argument('--model', help='API model ID; not guessed from display names')
    p.add_argument('--model-path', type=Path, help='Local checkpoint directory')
    p.add_argument('--python', default=sys.executable, help='Python in the installed model environment')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--image-root', type=Path, default=ROOT/'images')
    p.add_argument('--output-dir', type=Path, required=True, help='New directory, e.g. result/qwen_new')
    p.add_argument('--validate-inputs', action='store_true')
    p.add_argument('--execute', action='store_true', help='Actually invoke models/APIs; may incur costs')
    p.add_argument('--max-tokens', type=int, default=16)
    return p


def main():
    a = parser().parse_args()
    api = a.runner in ('eval_gpt.py', 'eval_gemini.py')
    if (api and not a.model) or (not api and a.model_path is None):
        raise SystemExit('API runners require --model; local runners require --model-path')
    if a.max_tokens <= 0:
        raise SystemExit('--max-tokens must be positive')
    output = a.output_dir.expanduser().resolve()
    data, images = a.data_root.resolve(), a.image_root.resolve()
    if output.exists() or any(p == output or p in output.parents or output in p.parents for p in (data, images)):
        raise SystemExit('Choose a NEW output directory outside data and images')
    audit = validate_inputs(data, images) if a.validate_inputs or a.execute else None
    plan = []
    for name, (relative, count, with_image) in CONDITIONS.items():
        command = [a.python, str(ROOT/'evaluation'/a.runner), '--input', str(data/relative),
                   '--output', str(output/(Path(relative).stem+'.jsonl')), '--use-image', str(with_image),
                   '--image-root', str(images), '--max-output-tokens' if api else '--max-new-tokens', str(a.max_tokens)]
        command += ['--model', a.model] if api else ['--model-path', str(a.model_path.resolve())]
        plan.append({'condition': name, 'requests': count, 'command': command})
    manifest = {'mode': 'execute' if a.execute else 'preview', 'initial_requests': 9600,
                'protocol': 'public_generic_runners_not_historical_task_specific', 'inputs': audit, 'plan': plan}
    print(json.dumps(manifest, indent=2))
    if not a.execute:
        return
    output.mkdir(parents=True, exist_ok=False)
    (output/'public_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    for entry in plan:
        subprocess.run(entry['command'], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
