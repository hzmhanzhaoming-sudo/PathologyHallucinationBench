"""One bounded, gold-blind final-answer request after CoT exhausts its budget.

This is an explicit two-stage protocol, not a retry of an incorrect answer.
It does not guarantee that every model will comply with the final instruction.
"""
from copy import deepcopy

from common import digest, labels, parse_answer

POLICY = 'cot_limit_then_answer_v1'


class FinalAnswerError(RuntimeError):
    def __init__(self, initial_response, cause):
        super().__init__(f'Final-answer stage failed: {type(cause).__name__}: {cause}')
        self.initial_response = initial_response


def final_answer_messages(messages, raw, options):
    """Keep all original images/questions/options/demonstrations, without gold."""
    out = deepcopy(messages)
    instruction = (
        'The reasoning budget is exhausted. Stop reasoning now. '
        'For the target question above (not the demonstration questions), '
        'choose exactly one of the listed options using the information already available. '
        'Return only one uppercase option letter from: ' + ', '.join(labels(options)) + '. '
        'Do not continue the analysis. Do not provide an explanation or any other text.'
    )
    if raw.strip():
        # Close an interrupted think block in the FOLLOW-UP context only. The
        # original generation is preserved byte-for-byte in initial_response.
        context = raw
        if raw.lower().count('<think>') > raw.lower().count('</think>'):
            context += '\n</think>'
        out.append({'role': 'assistant', 'content': [{'type': 'text', 'text': context}]})
        out.append({'role': 'user', 'content': [{'type': 'text', 'text': instruction}]})
    else:
        # Native API reasoning can use the entire budget without visible text.
        # Avoid fabricating an assistant turn or an empty assistant API message.
        out[-1]['content'].append({'type': 'text', 'text': instruction})
    return out


def output_tokens(result):
    if isinstance(result.get('generated_tokens'), int):
        return result['generated_tokens']
    usage = result.get('usage') or {}
    if isinstance(usage.get('output_tokens'), int):
        return usage['output_tokens']
    if isinstance(usage.get('candidatesTokenCount'), int):
        return usage['candidatesTokenCount'] + (usage.get('thoughtsTokenCount') or 0)
    return None


def generate_with_final_answer(backend, messages, options, strategy,
                               enabled=True, answer_tokens=32):
    initial = backend(messages)
    exhausted = bool(initial.get('token_limit_reached', initial.get('truncated', False)))
    pred = parse_answer(initial['raw_output'], options)
    result = dict(initial, finalization_policy=POLICY,
                  finalization_attempted=False, finalization_succeeded=None,
                  initial_token_limit_reached=exhausted, generation_calls=1,
                  total_output_tokens=output_tokens(initial))
    # In particular, a valid but WRONG answer never triggers a second attempt.
    if not (enabled and strategy.endswith('cot') and exhausted and pred is None):
        return result
    followup = final_answer_messages(messages, initial['raw_output'], options)
    try:
        final = backend.final_answer(followup, answer_tokens)
    except Exception as error:
        raise FinalAnswerError(initial, error) from error
    final_pred = parse_answer(final['raw_output'], options)
    counts = [output_tokens(initial), output_tokens(final)]
    # Top-level response metadata refers to the scored final response. Both
    # complete stage records and summed output usage remain available below.
    result.update(final)
    result.update(raw_output=final['raw_output'],
                  truncated=bool(initial.get('truncated') or final.get('truncated')),
                  initial_truncated=bool(initial.get('truncated')),
                  final_answer_truncated=bool(final.get('truncated')),
                  finalization_attempted=True, finalization_succeeded=final_pred is not None,
                  finalization_max_tokens=answer_tokens, generation_calls=2,
                  initial_response=initial, finalization_response=final,
                  finalization_prompt_sha256=digest(followup),
                  total_output_tokens=sum(counts) if all(x is not None for x in counts) else None)
    return result
