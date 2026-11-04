"""Single-condition backend checks, also used when an adapter is run directly."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.schema import gold, index_records, new_output, relative_file


def resolve_input_image(value, image_root):
    if value is None or not str(value).strip():
        return None
    path = relative_file(image_root, value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_backend_args(args):
    source = Path(args.input).expanduser().resolve()
    images = Path(args.image_root).expanduser().resolve()
    for name, path in (('input', source), ('output', Path(args.output).expanduser().resolve()), ('image_root', images)):
        setattr(args, name, path if isinstance(getattr(args, name), Path) else str(path))
    protected = [source, images, Path(__file__).parent, Path(__file__).parents[1] / 'scoring']
    for name in ('model_path', 'repo_root'):
        if getattr(args, name, None):
            path = Path(getattr(args, name)).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(path)
            protected.append(path)
            setattr(args, name, path if isinstance(getattr(args, name), Path) else str(path))
    new_output(args.output, protected)
    records = index_records(json.loads(source.read_text(encoding='utf-8-sig')), str(source))
    if not records:
        raise ValueError('Empty evaluation input')
    for row in records.values():
        gold(row)
        if not row.get('question'):
            raise ValueError('Empty question')
        if bool(row.get('img')) != bool(args.use_image):
            raise ValueError('Input image availability does not match --use-image')
        if args.use_image and not relative_file(images, row['img']).is_file():
            raise FileNotFoundError(row['img'])
    for name in ('max_new_tokens', 'max_output_tokens'):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('retry_invalid', 'retries'):
        if hasattr(args, name) and getattr(args, name) < 0:
            raise ValueError(f'{name} must be nonnegative')
