"""Plan or run all 11 benchmark conditions. Inference requires --execute."""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.schema import CONDITIONS, load_inputs, load_predictions, new_output, sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNNERS = tuple(p.name for p in sorted(HERE.glob('eval_*.py')))
API = ('eval_gpt.py', 'eval_gemini.py')
RESERVED = {'input', 'output', 'use-image', 'image-root', 'model', 'model-path',
            'max-new-tokens', 'max-output-tokens', 'api-key', 'mode', 'limit',
            'with-image-input', 'without-image-input', 'with-image-output',
            'without-image-output', 'output-root', 'text-only', 'no-send-image'}


def backend_options(path, config):
    """Check option names without importing a GPU backend. No shell expansion."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    available = {node.args[0].value[2:]: node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'add_argument' and node.args
                 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                 and node.args[0].value.startswith('--')}
    if not isinstance(config, dict):
        raise ValueError('Backend config must be a JSON object of CLI option names to values')
    for name, node in available.items():
        if name not in RESERVED and any(k.arg == 'required' and isinstance(k.value, ast.Constant) and k.value.value for k in node.keywords) and name not in config:
            raise ValueError(f'This backend requires {name!r} in --backend-config')
    result = []
    for key, value in config.items():
        if key in RESERVED or key not in available:
            raise ValueError(f'Unsupported/reserved backend setting: {key}')
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(f'Backend setting must be a scalar: {key}')
        if isinstance(value, bool):
            action = next((k.value.value for k in available[key].keywords if k.arg == 'action' and isinstance(k.value, ast.Constant)), None)
            if action != 'store_true':
                raise ValueError(f'{key}: use a string or number, not a Boolean')
            if value:
                result.append('--' + key)
        else:
            if key in {'repo-root', 'model-base', 'checkpoint-weight-path'}:
                value = str(Path(value).expanduser().resolve())
            result.extend(['--' + key, str(value)])
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', required=True, choices=RUNNERS)
    p.add_argument('--model', help='Explicit provider model ID for API inference')
    p.add_argument('--model-path', type=Path, help='Local checkpoint path')
    p.add_argument('--python', default=sys.executable, help='Python in the backend environment')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--image-root', type=Path, default=ROOT / 'images')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--backend-config', type=Path, help='JSON of optional model-loader/generation settings')
    p.add_argument('--max-tokens', type=int, help='Override backend token budget; omitted preserves its default')
    p.add_argument('--allow-custom-size', action='store_true', help='Allow complete paired toy/custom datasets')
    p.add_argument('--validate-inputs', action='store_true', help='Check JSON, pairing and images without inference')
    p.add_argument('--execute', action='store_true', help='Invoke model/API; may incur costs')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    api = a.runner in API
    if (api and (not a.model or a.model_path)) or (not api and (not a.model_path or a.model)):
        raise ValueError('API: specify only --model; local: specify only --model-path')
    if a.max_tokens is not None and a.max_tokens <= 0:
        raise ValueError('--max-tokens must be positive')
    data, images = a.data_root.expanduser().resolve(), a.image_root.expanduser().resolve()
    protected = [data, images, HERE, ROOT / 'scoring']
    if a.model_path:
        protected.append(a.model_path)
    if a.backend_config:
        protected.append(a.backend_config)
    out = new_output(a.output_dir, protected)
    backend = HERE / a.runner
    config = json.loads(a.backend_config.read_text(encoding='utf-8-sig')) if a.backend_config else {}
    extra = backend_options(backend, config)
    refs, hashes = load_inputs(data, images, a.allow_custom_size) if a.validate_inputs or a.execute else ({}, {})
    plan = []
    for name, (relative, count, with_image) in CONDITIONS.items():
        command = [a.python, str(backend), '--input', str(data / relative),
                   '--output', str(out / 'raw' / (name + '.jsonl')),
                   '--use-image', str(int(with_image)), '--image-root', str(images)]
        command += ['--model', a.model] if api else ['--model-path', str(a.model_path.expanduser().resolve())]
        if a.max_tokens is not None:
            command += ['--max-output-tokens' if api else '--max-new-tokens', str(a.max_tokens)]
        command += extra
        plan.append(dict(condition=name, records=len(refs[name]) if refs else count, command=command))
    manifest = dict(schema_version=1, status='planned',
                    model_key=out.name, model=a.model or a.model_path.name,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    custom_size=a.allow_custom_size, input_sha256=hashes,
                    initial_requests=sum(p['records'] for p in plan),
                    source_sha256={p.name: sha256(p) for p in (backend, Path(__file__), HERE / 'schema.py', HERE / 'backend_io.py')},
                    plan=plan, prediction_sha256={})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not a.execute:
        return manifest
    if not api and not a.model_path.expanduser().resolve().is_dir():
        raise FileNotFoundError(a.model_path)
    key = 'OPENAI_API_KEY' if a.runner == 'eval_gpt.py' else 'GEMINI_API_KEY'
    if api and not os.environ.get(key):
        raise ValueError(f'Set {key} in the environment; credentials are never written to the manifest')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'raw').mkdir()
    manifest['status'] = 'running'
    write_json(out / 'run_manifest.json', manifest)
    try:
        for entry in plan:
            name = entry['condition']
            # Fail before dispatch if inputs changed after validation.
            if any(sha256(data / relative) != digest for relative, digest in hashes.items()):
                raise ValueError('Input files changed during evaluation')
            subprocess.run(entry['command'], check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            clean = load_predictions(out / 'raw' / (name + '.jsonl'), refs[name])
            target = out / (name + '.jsonl')
            with target.open('x', encoding='utf-8') as stream:
                for row in clean.values():
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            manifest['prediction_sha256'][target.name] = sha256(target)
            write_json(out / 'run_manifest.json', manifest)
        if any(sha256(data / relative) != digest for relative, digest in hashes.items()):
            raise ValueError('Input files changed during evaluation')
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['failure_type'] = type(error).__name__
        raise
    finally:
        write_json(out / 'run_manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
