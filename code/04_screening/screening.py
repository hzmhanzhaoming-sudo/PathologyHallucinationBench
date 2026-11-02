"""Manuscript-prompt adapter, not a reconstruction of historical cohort selection.

Python 3.8+. No network on import or request preparation. Prompts are immutable
UTF-8 resources. API role, JSON envelopes and validation are implementation
choices, explicitly separated from manuscript wording in README.md.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    'classify': 'stage1_content_classification',
    'eligibility': 'stage1_eligibility_screening',
    'compare': 'stage2_semantic_comparison',
}
PAPER_MODELS = {'classify': 'GPT-5.6 Terra', 'eligibility': 'GPT-5.6 Terra',
                'compare': 'GPT-5.6 SOL'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def _invalid_constant(value):
    raise ValueError('Non-JSON numeric constant: ' + value)


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_invalid_constant)


def read_json(path):
    return loads(Path(path).read_bytes().decode('utf-8-sig'))


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def write_new(path, data):
    with Path(path).open('xb') as f:
        f.write(data)


def literal_prompts(stage, root=ROOT):
    from verify_prompt_source import verify
    if not verify(package_root=root)['passed']:
        raise ValueError('Published prompt source verification failed; refusing to run')
    manifest = read_json(root / 'provenance' / 'prompt_source_manifest.json')
    entry = next(e for e in manifest['prompts'] if e['name'] == STAGES[stage])
    fixed = (root / 'prompts' / entry['fixed_file']).read_bytes()
    if sha(fixed) != entry['fixed_sha256']:
        raise ValueError('Published fixed instruction has changed; refusing to run')
    user = None
    if entry.get('has_explicit_user_template'):
        user = (root / 'prompts' / entry['user_template_file']).read_bytes()
        if sha(user) != entry['user_template_sha256']:
            raise ValueError('Published user template has changed; refusing to run')
    return fixed.decode('utf8'), user.decode('utf8') if user is not None else None, entry


def record_ids(records, id_key):
    if not isinstance(records, list) or not records:
        raise ValueError('Input records/targets must be a nonempty JSON array')
    ids = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Each input record must be an object')
        value = row.get(id_key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Each record needs a nonempty string ' + id_key)
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input IDs')
    return ids


def input_contract(stage, data, id_key):
    if stage != 'compare':
        return record_ids(data, id_key), None
    if not isinstance(data, dict):
        raise ValueError('Comparison input must be an object')
    if not data.get('cohort_summaries'):
        raise ValueError('Supply actual cohort_summaries; none are inferred')
    targets = data.get('targets')
    ids = record_ids(targets, id_key)
    neighbours = {}
    for row in targets:
        for key in ('original_texts', 'image_review_findings', 'content_labels'):
            if not row.get(key):
                raise ValueError('Comparison target needs supplied ' + key)
        near = row.get('neighbours')
        if not isinstance(near, list):
            raise ValueError('neighbours must be a supplied list (may be empty)')
        near_ids = record_ids(near, id_key) if near else []
        if row[id_key] in near_ids:
            raise ValueError('A target must not be its own neighbour')
        neighbours[row[id_key]] = near_ids
    return ids, neighbours


def check_schema(schema):
    """Never retrieve external schemas. Validation requires an explicit dependency."""
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('$ref', '$dynamicRef', '$recursiveRef'):
                    if not isinstance(child, str) or not child.startswith('#'):
                        raise ValueError('Only local JSON schema references are allowed')
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ValueError('Provide an explicit object-root response JSON schema')
    visit(schema)
    try:
        from jsonschema.validators import validator_for
    except ImportError:
        raise RuntimeError('Install requirements.txt before prepare/validate; no API was called')
    validator = validator_for(schema)
    validator.check_schema(schema)
    return validator(schema)


def build_request(stage, data, schema, model, max_output_tokens,
                  instruction_role='system', result_key='results', id_key=None):
    if stage not in STAGES:
        raise ValueError('Unknown stage')
    id_key = id_key or ('review_id' if stage == 'compare' else 'unit_id')
    if stage in ('classify', 'compare') and id_key != ('review_id' if stage == 'compare' else 'unit_id'):
        raise ValueError('ID name must match the explicit manuscript prompt')
    if instruction_role not in ('system', 'developer'):
        raise ValueError('Unsupported fixed-instruction role')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('Supply the actual provider model ID; no model substitution')
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError('max_output_tokens must be a positive integer')
    if not isinstance(result_key, str) or not result_key:
        raise ValueError('Specify the top-level array key in the external result schema')
    check_schema(schema)
    ids, neighbours = input_contract(stage, data, id_key)
    fixed, template, source = literal_prompts(stage)
    # No invented prose, hidden prefix, few-shot example or response-repair prompt.
    serialized = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    user = template.replace('{records_json}', serialized) if template is not None else serialized
    body = {
        'model': model,
        'input': [{'role': instruction_role, 'content': fixed},
                  {'role': 'user', 'content': user}],
        'text': {'format': {'type': 'json_schema', 'name': STAGES[stage],
                            'strict': True, 'schema': schema}},
        'max_output_tokens': max_output_tokens,
        'store': False,
    }
    contract = {
        'stage': stage, 'paper_model_label': PAPER_MODELS[stage],
        'requested_api_model': model, 'expected_ids': ids, 'id_key': id_key,
        'result_key': result_key, 'allowed_neighbour_ids': neighbours,
        'fixed_prompt_sha256': source['fixed_sha256'],
        'instruction_role_is_implementation_setting': True,
        'response_schema_is_caller_supplied_not_published': True,
        'extra_user_prose_added': False,
        'does_not_reproduce_historical_cohort_membership': True,
    }
    return body, contract


def load_prepared(folder):
    folder = Path(folder)
    manifest = read_json(folder / 'prepared_manifest.json')
    expected = {'request.json', 'input.json', 'schema.json', 'contract.json'}
    if set(manifest.get('files', {})) != expected:
        raise ValueError('Unexpected prepared-file manifest')
    contents = {}
    for name in sorted(expected):
        raw = (folder / name).read_bytes()
        if sha(raw) != manifest['files'][name]:
            raise ValueError('Prepared file changed: ' + name)
        contents[name] = loads(raw.decode('utf8'))
    request = contents['request.json']
    contract = contents['contract.json']
    rebuilt, contract_expected = build_request(
        contract['stage'], contents['input.json'], contents['schema.json'],
        request['model'], request['max_output_tokens'], request['input'][0]['role'],
        contract['result_key'], contract['id_key'])
    if request != rebuilt or contract != contract_expected:
        raise ValueError('Prepared request is not exactly the published-prompt adapter output')
    return request, contents['schema.json'], contract


def validate_result(result, schema, contract):
    validator = check_schema(schema)
    # Do not print error.instance or a model-generated payload on failure.
    if next(validator.iter_errors(result), None) is not None:
        raise ValueError('Result does not match the supplied response schema')
    rows = result.get(contract['result_key']) if isinstance(result, dict) else None
    ids = record_ids(rows, contract['id_key'])
    if set(ids) != set(contract['expected_ids']):
        raise ValueError('Missing, extra or changed result IDs')
    if contract['stage'] == 'compare':
        for row in rows:
            values = row.get('related_ids')
            allowed = contract['allowed_neighbour_ids'][row['review_id']]
            if not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values):
                raise ValueError('related_ids must use only this target\'s supplied neighbours')
    return {'valid': True, 'stage': contract['stage'], 'rows': len(rows),
            'checks': ['caller_supplied_schema', 'exact_ID_coverage', 'unique_IDs'],
            'medical_correctness_verified': False, 'cohort_membership_decided': False}
