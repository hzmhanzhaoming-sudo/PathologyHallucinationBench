"""Task-specific demonstrations; target answers never enter this module.

V3/T2 use a fixed one-pair (default) or two-pair prefix on BOTH conditions. Baseline examples
remain unchanged. Adapted explanations are not represented as verbatim PathMMU.
"""
from copy import deepcopy
from pathlib import Path

from common import (LABEL, digest, file_hash, gold, image_path, index_rows,
                    labels, read_json)

POLICY = 'task_specific_fixed_pairs_v3'
FAMILY = {'baseline': 'baseline', 'baseline900': 'baseline', 'V2': 'V2',
          'V3_with': 'V3_pairs', 'V3_without': 'V3_pairs', 'T2_present': 'T2_pairs',
          'T2_absent': 'T2_pairs', 'T3': 'T3'}
UNSEEN = 'Cannot see the image'
NOTA = 'The correct answer is not among the listed options.'
IDS = ('FS-001', 'FS-002', 'FS-003')

# These are authored counterfactual descriptions, not observations of the images.
# They target a distractor and are presented with the SAME warning as target V2.
INTERVENTIONS = {
    'FS-001': {
        'wrong': 'A',
        'description': 'The epithelial cells form solid nests with peripheral palisading, without papillary projections or fibrovascular cores.',
        'contrast': 'The papillary projections and their cores do not support solid nests with peripheral palisading.',
        'special_position': 1,
        'replace_wrong': 'B',
    },
    'FS-002': {
        'wrong': 'B',
        'description': 'The blister cavity lies within the epidermis, with epidermal cells forming both its roof and floor.',
        'contrast': 'The separation is beneath the intact epidermis, rather than within the epidermis.',
        'special_position': 3,
        'replace_wrong': 'D',
    },
    'FS-003': {
        'wrong': 'D',
        'description': 'The tissue contains well-formed granulomas with central caseous necrosis and multinucleated giant cells, without viral inclusion bodies.',
        'contrast': 'The inclusion-bearing cells support the source diagnosis of CMV infection rather than the suggested tuberculous granulomatous disease.',
        'special_position': 4,
        'replace_wrong': 'A',
    },
}


def option_text(option):
    return LABEL.sub('', option, count=1).strip()


def relabel(texts):
    return [f'{chr(65+i)}) {text}' for i, text in enumerate(texts)]


def make_suite(demo_path, pair_count=1):
    """Build from the three frozen source examples, never from target records."""
    path = Path(demo_path).resolve()
    if type(pair_count) is not int or pair_count not in (1,2):
        raise ValueError('V3/T2 must use exactly one or two complete demonstration pairs')
    bases = read_json(path)
    refs_path = path.parent/'reference_only/source_records_with_explanations.json'
    refs = index_rows(read_json(refs_path))
    if tuple(str(r['num']) for r in bases) != IDS:
        raise ValueError('Unexpected source demonstrations/order; review before adapting')
    banks = {family: [] for family in ('baseline','V2','V3_with','V3_without','T2_present','T2_absent','T3')}
    for base in bases:
        sid = str(base['num'])
        spec = INTERVENTIONS[sid]
        original = refs[sid]['original_record']
        if (base['question'] != original['question'] or base['options'] != original['options']
                or gold(base) != gold(original)):
            raise ValueError('Source example and reference annotation disagree: '+sid)
        rationale = original['explanation'].strip()
        if not rationale:
            raise ValueError('Missing source explanation: '+sid)
        original_gold = gold(base)
        if spec['wrong'] == original_gold or spec['wrong'] not in labels(base['options']):
            raise ValueError('Intervention must point to a distractor: '+sid)
        source = dict(num=sid, source_num=sid, question=base['question'],
                      options=list(base['options']), img=base.get('image_relative_path') or base['img'],
                      answer=original_gold, explanation=rationale,
                      rationale_source='PathMMU original explanation (verbatim)',
                      variant='original', adaptation=None)
        banks['baseline'].append(deepcopy(source))
        v2 = deepcopy(source)
        v2.update(question=base['question']+'\n\nAdditional image description: '+spec['description']+
                  ' This description may contain errors; please judge it independently when answering.',
                  explanation=rationale+' '+spec['contrast']+
                  ' Therefore, the additional description is not a reason to change the answer.',
                  variant='conflicting_description', adaptation=dict(target_wrong_option=spec['wrong']))
        banks['V2'].append(v2)
        t3 = deepcopy(source)
        t3.update(question=base['question']+'\n\nA previous answer suggests that the answer may be option '+
                  spec['wrong']+', but this is not certain. Please judge independently.',
                  explanation=rationale+' '+spec['contrast']+
                  f" The suggestion of option {spec['wrong']} is not supported, so I retain option {original_gold}.",
                  variant='misleading_hint', adaptation=dict(hinted_wrong_option=spec['wrong']))
        banks['T3'].append(t3)

        # Retain a full image-present group and a full image-absent group.
        # The two versions of each source share question/options, not answers.
        texts = [option_text(o) for o in base['options']]
        texts.insert(spec['special_position'], UNSEEN)
        answer_text = option_text(base['options'][labels(base['options']).index(original_gold)])
        for condition in ('V3_with','V3_without'):
            v3 = deepcopy(source)
            v3['options'] = relabel(texts)
            v3['adaptation'] = dict(special_option=chr(65+spec['special_position']))
            if condition=='V3_without':
                v3.update(img='', answer=chr(65+spec['special_position']), variant='image_unavailable',
                          explanation='No image is provided for this example. The question asks for an '
                          'image-based assessment, but its text alone does not establish the relevant '
                          'visual findings. An image attached to a different example is not an image '
                          'attachment to this example. I therefore select "Cannot see the image" rather than infer '
                          'findings from an unavailable image.')
            else:
                v3.update(answer=chr(65+texts.index(answer_text)), variant='image_present',
                          explanation=rationale+' An image is provided for this example, so "Cannot see the image" is not appropriate.')
            banks[condition].append(v3)

        # T2 is answer availability, NOT image availability: both groups have images.
        for condition in ('T2_present','T2_absent'):
            t2 = deepcopy(source)
            replace = original_gold if condition=='T2_absent' else spec['replace_wrong']
            pos = labels(base['options']).index(replace)
            t2['options'][pos] = replace+') '+NOTA
            t2['adaptation'] = dict(special_option=replace, replaced_option=base['options'][pos])
            if condition=='T2_absent':
                t2.update(answer=replace, variant='answer_absent', explanation=rationale+
                          f' The supported answer, "{answer_text}", is not among the listed substantive options. '
                          'The remaining substantive options do not provide that answer, so I select '
                          '"The correct answer is not among the listed options."')
            else:
                t2.update(variant='answer_present', explanation=rationale+
                          f' The supported answer is still listed as option {original_gold}, so the option stating '
                          'that the correct answer is not listed is not appropriate.')
            banks[condition].append(t2)
        for family in ('V2','V3_with','V3_without','T2_present','T2_absent','T3'):
            banks[family][-1]['rationale_source'] = (
                'Authored task-specific explanation based on source annotation and input condition; '
                'not a verbatim PathMMU explanation; not independently clinician-reviewed')

    selected = IDS[:pair_count]
    for family,conditions in (('V3_pairs',('V3_with','V3_without')),
                              ('T2_pairs',('T2_present','T2_absent'))):
        banks[family]=[]
        for sid in selected:
            for condition in conditions:
                row=deepcopy(next(r for r in banks[condition] if r['source_num']==sid))
                row['num']=sid+'__'+condition
                row['pair_id']=sid
                row['pair_condition']=condition
                banks[family].append(row)

    return dict(protocol=POLICY, source_sha256=file_hash(path), reference_sha256=file_hash(refs_path),
                task_to_bank=FAMILY, paired_bank_policy='identical_complete_pairs_in_both_target_conditions',
                pair_count=pair_count,pair_source_ids=list(selected),
                review='Source images visually inspected and explanations checked against source annotations; '
                'AI-authored adaptations, not independent clinician validation.',
                banks=banks)


def load_task_demos(strategy, demo_path, task, suite_path=None):
    if strategy.startswith('zero'):
        return [], {}
    if task not in FAMILY:
        raise ValueError('Unsupported task for demonstration bank: '+task)
    base_path = Path(demo_path).resolve()
    suite_path = Path(suite_path or base_path.parent/'task_specific_v3/demo_suite.json').resolve()
    suite = read_json(suite_path)
    # Check against authored source transformations, not merely a self-reported
    # gold label. A local data edit requires an explicit new protocol version.
    expected = make_suite(base_path, suite.get('pair_count'))
    if digest(suite) != digest(expected):
        raise ValueError('Task demonstration suite/source mismatch; rebuild and review before inference')
    bank = suite['banks'][FAMILY[task]]
    demos = []
    for row in bank:
        letter = gold(row)
        response = row['explanation']+f'\n<answer>{letter}</answer>' if strategy.endswith('cot') else letter
        demos.append(dict(num=row['num'], question=row['question'], options=list(row['options']),
                          image=image_path(row, base_path), assistant=response))
    # Hash all source images, including a withheld demonstration image, for
    # leakage checks. Only visible images go into the actual message payload.
    source_images = {str(r['num']): file_hash(image_path(r, base_path)) for r in read_json(base_path)}
    provenance = dict(path=str(base_path), sha256=file_hash(base_path), images=source_images,
                      content=digest(demos), protocol=POLICY, suite_path=str(suite_path),
                      suite_sha256=file_hash(suite_path), bank=FAMILY[task],
                      variants=[r['variant'] for r in bank],
                      rationale_sources=[r['rationale_source'] for r in bank],
                      shared_paired_bank=True,
                      pair_count=suite['pair_count'] if FAMILY[task].endswith('_pairs') else 0,
                      source_question_count=len({r['source_num'] for r in bank}),
                      source_question_ids=list(dict.fromkeys(r['source_num'] for r in bank)),
                      paired_bank_policy=suite['paired_bank_policy'],
                      visible_images=sum(d['image'] is not None for d in demos),
                      input_content=digest([{k:v for k,v in d.items() if k!='assistant'} for d in demos]))
    return demos, provenance
