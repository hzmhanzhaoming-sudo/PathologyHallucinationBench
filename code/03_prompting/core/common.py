"""Dependency-free input validation, prompts, conservative parsing and provenance."""
import hashlib
import json
import re
from pathlib import Path
from registry import ROOT, TASK_FILES, TARGET_FILES

LABEL = re.compile(r'^\s*([A-Z])\s*[).:\-]\s*', re.I)
STRATEGIES = ('few_shot_cot', 'few_shot_direct', 'zero_shot_cot', 'zero_shot_direct')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def labels(options):
    out = []
    for x in options:
        m = LABEL.match(str(x))
        if not m:
            raise ValueError(f'Unlabelled option: {x!r}')
        out.append(m.group(1).upper())
    if len(set(out)) != len(out) or len(out) < 2:
        raise ValueError('Duplicate labels or fewer than two choices')
    return out

def gold(row):
    answer = str(row['answer']).strip()
    if answer.upper() in labels(row['options']):
        return answer.upper()
    m = LABEL.match(answer)
    if not m or answer not in row['options']:
        raise ValueError(f"Gold is not an option: {row.get('num')}")
    return m.group(1).upper()

def parse_answer(raw, options):
    """Never pick an incidental A/B from a rationale; conflicting finals are invalid."""
    s = re.sub(r'<think>.*?</think>', '', raw or '', flags=re.I | re.S).strip()
    if re.search(r'<think>', s, re.I):  # incomplete hidden/thinking block
        return None
    legal = set(labels(options))
    matches = re.findall(r'<answer>\s*([A-Z])\s*</answer>', s, re.I)
    if not matches:
        matches = re.findall(r'(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*([A-Z])\s*[.)]?\s*$', s, re.I)
    if matches:
        matches = {x.upper() for x in matches}
        return next(iter(matches)) if len(matches) == 1 and matches <= legal else None
    m = re.fullmatch(r'\s*\(?([A-Z])\)?\s*[.)]?\s*', s, re.I)
    if m:
        return m.group(1).upper() if m.group(1).upper() in legal else None
    # Exact full-option output is safe, unlike searching inside free-form prose.
    for option in options:
        if s.casefold() == option.strip().casefold():
            return LABEL.match(option).group(1).upper()
    return None

def index_rows(rows):
    result = {}
    for row in rows:
        key = str(row['num'])
        if key in result:
            raise ValueError(f'Duplicate item ID: {key}')
        result[key] = row
    return result

def dataset(row):
    return {'100': 'QuiltVQA', '200': 'PathVQA', '400': 'PatchVQA', '500': 'PathMMU'}.get(
        str(row['num']).split('-')[0], row.get('category', 'custom'))

def load_rows(task, root=ROOT, custom=None):
    root = Path(root)
    path = Path(custom) if custom else root / TASK_FILES[task]
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Expected nonempty JSON list: {path}')
    index_rows(rows)
    if task == 'baseline900' and not custom:
        wanted = set(index_rows(read_json(root / TASK_FILES['V2'])))
        rows = [r for r in rows if str(r['num']) in wanted]
        if len(rows) != len(wanted):
            raise ValueError('Missing original baseline IDs')
    for r in rows:
        gold(r)
        if not str(r['question']).strip():
            raise ValueError('Empty question')
    return rows, path.resolve()

def image_path(row, source):
    raw = row.get('img', '')
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path(source).parent / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image missing for {row['num']}: {p}")
    return str(p)

def load_demos(strategy, demo_path):
    if strategy.startswith('zero'):
        return [], {}
    path = Path(demo_path).resolve()
    rows = read_json(path)
    if len(rows) != 3:
        raise ValueError('This experiment fixes exactly three demonstrations')
    refs = {}
    if strategy.endswith('cot'):
        ref_path = path.parent / 'reference_only/source_records_with_explanations.json'
        refs = {str(r['num']): r['original_record']['explanation'] for r in read_json(ref_path)}
    demos = []
    for r in rows:
        g = gold(r)
        img = image_path(r, path)
        if not img:
            raise ValueError('Every demonstration must include its original image')
        response = g
        if strategy.endswith('cot'):
            explanation = refs.get(str(r['num']), '').strip()
            if not explanation:
                raise ValueError('Missing original PathMMU explanation')
            response = explanation + f'\n<answer>{g}</answer>'
        # Deliberately whitelist prompt fields; source metadata/gold never enters target prompts.
        demos.append(dict(num=str(r['num']), question=r['question'], options=r['options'],
                          image=img, assistant=response))
    provenance = dict(path=str(path), sha256=file_hash(path),
                      images={d['num']: file_hash(d['image']) for d in demos}, content=digest(demos))
    return demos, provenance

def make_messages(row, source, strategy, demos):
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    instruction = ('Answer the pathology multiple-choice question. Think step by step, then return '
                   'the final option letter within <answer> and </answer> tags.'
                   if strategy.endswith('cot') else
                   'Answer the pathology multiple-choice question. Return only the selected option letter. Do not explain.')
    messages = []
    def user(question, options, img, title):
        content = [{'type': 'text', 'text': title}]
        if img:
            content.append({'type': 'image', 'image': img})
        content.append({'type': 'text', 'text': instruction+'\nQuestion: '+question+
                        '\nOptions:\n'+'\n'.join(options)})
        return {'role': 'user', 'content': content}
    for i, d in enumerate(demos, 1):
        messages.append(user(d['question'], d['options'], d['image'], f'Example {i}'))
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': d['assistant']}]})
    messages.append(user(row['question'], row['options'], image_path(row, source), 'Target question'))
    return messages

def targets(task, root=ROOT):
    data = read_json(Path(root) / TARGET_FILES[task])
    if task == 'T3':
        data = data['items']
    field = 'target_wrong_option' if task == 'V2' else 'hinted_wrong_option'
    return {str(r['num']): r[field].upper() for r in data}

def validate_suite(root=ROOT):
    """Fail before scoring/inference when source interventions no longer match."""
    data = {k: index_rows(load_rows(k, root)[0]) for k in TASK_FILES}
    core = set(data['V2'])
    if len(data['baseline']) != 1200 or len(core) != 900:
        raise ValueError('Expected 1,200 baseline and 900 stress-test items')
    for key in ('baseline900', 'T3', 'V3_with', 'V3_without', 'T2_present', 'T2_absent'):
        if set(data[key]) != core:
            raise ValueError(f'Mismatched core IDs: {key}')
    for task in ('V2', 'T3'):
        target = targets(task, root)
        if set(target) != core:
            raise ValueError('Incomplete target manifest')
        for k in core:
            b, p = data['baseline'][k], data[task][k]
            if b['options'] != p['options'] or gold(b) != gold(p) or b['img'] != p['img']:
                raise ValueError(f'Baseline/options mismatch: {task}/{k}')
            if target[k] not in labels(p['options']) or target[k] == gold(p):
                raise ValueError(f'Invalid wrong target: {task}/{k}')
    for k in core:
        w, n = data['V3_with'][k], data['V3_without'][k]
        if w['options'] != n['options'] or w['question'] != n['question'] or n['img']:
            raise ValueError(f'Invalid V3 pair: {k}')
        abstain = option_containing(n, 'Cannot see the image')
        if gold(n) != abstain or gold(w) == abstain:
            raise ValueError(f'Invalid V3 targets: {k}')
        p, a = data['T2_present'][k], data['T2_absent'][k]
        if gold(p) == option_containing(p, 'correct answer is not among') or gold(a) != option_containing(a, 'correct answer is not among'):
            raise ValueError(f'Invalid T2 targets: {k}')
    return data

def option_containing(row, text):
    found = [LABEL.match(o).group(1).upper() for o in row['options'] if text.casefold() in o.casefold()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one special option: {row['num']}/{text}")
    return found[0]

