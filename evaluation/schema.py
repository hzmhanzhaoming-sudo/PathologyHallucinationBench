"""Shared input/output contract. Standard library only; no model imports."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

CONDITIONS = {
    'V1_with_image': ('V1/v1_with_image.json', 1200, True),
    'V1_without_image': ('V1/v1_without_image.json', 1200, False),
    'V2': ('V2/v2_image_text_conflict_with_image.json', 900, True),
    'V3_with_image': ('V3/v3_with_image.json', 900, True),
    'V3_without_image': ('V3/v3_without_image.json', 900, False),
    'T1_4choice': ('T1/4-choice/t1_4choice_unknown1_2_with_image.json', 1000, True),
    'T1_6choice': ('T1/6-choice/t1_6choice_unknown1_2_with_image.json', 400, True),
    'T1_7choice': ('T1/7-choice/t1_7choice_unknown1_2_with_image.json', 400, True),
    'T2_present': ('T2/t2_answer_present_with_image.json', 900, True),
    'T2_absent': ('T2/t2_answer_absent_with_image.json', 900, True),
    'T3': ('T3/t3_wrong_answer_hint_with_image.json', 900, True),
}
T1_CONDITIONS = ('T1_4choice', 'T1_6choice', 'T1_7choice')
DATASETS = {'100': 'QuiltVQA', '200': 'PathVQA', '400': 'PatchVQA', '500': 'PathMMU'}
NOTA = 'The correct answer is not among the listed options.'
ABSTAIN = 'Cannot see the image'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def new_output(path, sources=()):
    out = Path(path).expanduser().resolve()
    if out.exists():
        raise ValueError('Choose a new output directory; existing outputs are never overwritten')
    for source in sources:
        src = Path(source).expanduser().resolve()
        if out == src or src in out.parents or out in src.parents:
            raise ValueError('Output must be separate from all input paths')
    return out


def relative_file(root, value):
    """Accept portable relative paths only, including on a different host OS."""
    text = str(value).replace('\\', '/')
    part = Path(text)
    if not text or part.is_absolute() or PureWindowsPath(text).drive or '..' in part.parts:
        raise ValueError(f'Expected a relative path without traversal: {value!r}')
    root = Path(root).resolve()
    target = (root / part).resolve()
    if root not in target.parents:
        raise ValueError(f'Path escapes root: {value!r}')
    return target


def options(row):
    values = row.get('options')
    if not isinstance(values, list) or not 2 <= len(values) <= 8:
        raise ValueError('Each item requires 2 to 8 labelled options')
    result = {}
    for value in values:
        match = re.fullmatch(r'([A-H])[).:]\s*(\S.*)', str(value).strip())
        if not match or match[1] in result:
            raise ValueError(f'Invalid/duplicate option: {value!r}')
        result[match[1]] = match[2].strip()
    return result


def gold(row):
    labels = options(row)
    answer = str(row.get('answer', '')).strip()
    if answer in labels:
        return answer
    if answer in row['options']:
        return answer[0]
    raise ValueError(f'Answer is not an option: {row.get("num")}')


def special_label(row, text):
    matches = [k for k, v in options(row).items() if v.casefold().rstrip('.') == text.casefold().rstrip('.')]
    if len(matches) != 1:
        raise ValueError(f'{row.get("num")}: expected one {text!r} option')
    return matches[0]


def dataset(row):
    value = row.get('dataset') or row.get('source_dataset')
    if value in DATASETS.values():
        return value
    prefix = str(row['num']).split('-', 1)[0]
    if prefix in DATASETS:
        return DATASETS[prefix]
    if value:
        return str(value)
    raise ValueError(f'{row["num"]}: provide an explicit dataset field')


def index_records(rows, context):
    if not isinstance(rows, list):
        raise ValueError(f'{context}: expected a JSON array')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('num') in (None, ''):
            raise ValueError(f'{context}: missing item num')
        key = str(row['num'])
        if key in indexed:
            raise ValueError(f'{context}: duplicate ID {key}')
        indexed[key] = row
    return indexed


def load_conditions(data_root, names, image_root=None, custom_size=False):
    """Read only named conditions; callers validate the relevant paired design."""
    groups, hashes = {}, {}
    for name in names:
        if name not in CONDITIONS or name in groups:
            raise ValueError(f'Unknown/duplicate condition: {name}')
        relative, expected, with_image = CONDITIONS[name]
        path = relative_file(data_root, relative)
        raw = path.read_bytes()
        rows = json.loads(raw.decode('utf-8-sig'))
        groups[name] = index_records(rows, name)
        if not rows:
            raise ValueError(f'{name}: empty condition input')
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        if not custom_size and len(rows) != expected:
            raise ValueError(f'{name}: expected {expected} records, got {len(rows)}')
        for row in rows:
            gold(row)
            if not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError(f'{name}/{row["num"]}: empty question')
            image = row.get('img', '')
            if bool(image) != with_image:
                raise ValueError(f'{name}/{row["num"]}: wrong image availability')
            if image:
                image_path = relative_file(image_root or data_root, image)
                if image_root is not None and not image_path.is_file():
                    raise FileNotFoundError(image_path)
    return groups, hashes


def load_inputs(data_root, image_root=None, custom_size=False):
    groups, hashes = load_conditions(data_root, CONDITIONS, image_root, custom_size)
    validate_pairs(groups)
    return groups, hashes


def validate_pairs(g):
    base, core = set(g['V1_with_image']), set(g['V2'])
    if not base or not core or not core <= base:
        raise ValueError('Core must be a nonempty subset of baseline IDs')
    if base != set(g['V1_without_image']):
        raise ValueError('V1 paired IDs differ')
    for i in base:
        a, b = g['V1_with_image'][i], g['V1_without_image'][i]
        if any(a[k] != b[k] for k in ('question', 'options')) or gold(a) != gold(b):
            raise ValueError(f'V1/{i}: only image availability may differ')
        dataset(a)
    for name in ('V3_with_image', 'V3_without_image', 'T2_present', 'T2_absent', 'T3'):
        if set(g[name]) != core:
            raise ValueError(f'{name}: core IDs differ')
    for i in core:
        baseline = g['V1_with_image'][i]
        for name in ('V2', 'T3'):
            row = g[name][i]
            if gold(row) != gold(baseline) or row['options'] != baseline['options']:
                raise ValueError(f'{name}/{i}: baseline answer/options changed')
        target = g['V2'][i].get('target_wrong_option')
        if target not in options(g['V2'][i]) or target == gold(baseline):
            raise ValueError(f'V2/{i}: invalid target_wrong_option')
        present, absent = g['V3_with_image'][i], g['V3_without_image'][i]
        if present['options'] != absent['options'] or present['question'] != absent['question']:
            raise ValueError(f'V3/{i}: paired question/options differ')
        if gold(absent) != special_label(absent, ABSTAIN) or gold(present) == gold(absent):
            raise ValueError(f'V3/{i}: incorrect image-availability reference')
        present, absent = g['T2_present'][i], g['T2_absent'][i]
        if gold(present) == special_label(present, NOTA) or gold(absent) != special_label(absent, NOTA):
            raise ValueError(f'T2/{i}: incorrect answer-availability reference')
    seen = {1: set(), 2: set()}
    for name in T1_CONDITIONS:
        for row in g[name].values():
            source, level = str(row.get('t1_original_num', '')), row.get('t1_unknown_count')
            if type(level) is not int or level not in seen or source not in core or source in seen[level]:
                raise ValueError(f'{name}: invalid/duplicate t1_original_num or t1_unknown_count')
            if gold(row) != gold(g['V1_with_image'][source]):
                raise ValueError(f'{name}/{source}: correct answer label changed')
            seen[level].add(source)
    if any(ids != core for ids in seen.values()):
        raise ValueError('Each T1 level must cover the complete core')


def load_predictions(path, refs):
    """Use stored parsed labels; never infer predictions from explanations/gold."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    indexed = index_records(rows, str(path))
    if set(indexed) != set(refs):
        raise ValueError(f'{path}: prediction IDs must exactly match input IDs')
    clean = {}
    for key, row in indexed.items():
        ref = refs[key]
        fields = [row[k] for k in ('predicted_label', 'pred_letter') if k in row]
        if not fields or any(p != fields[0] for p in fields):
            raise ValueError(f'{path}/{key}: missing/conflicting prediction fields')
        pred = fields[0]
        if pred is not None and pred not in options(ref):
            raise ValueError(f'{path}/{key}: prediction must be a legal label or null')
        for field in ('gold_label', 'gold_letter'):
            if field in row and row[field] != gold(ref):
                raise ValueError(f'{path}/{key}: stored gold conflicts with reference')
        if row.get('error'):
            raise ValueError(f'{path}/{key}: inference error; fix/re-run before scoring')
        clean[key] = dict(num=key, predicted_label=pred, gold_label=gold(ref),
                          correct=pred == gold(ref))
    return clean
