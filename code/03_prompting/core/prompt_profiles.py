"""Explicit historical prompt profiles; never silently change an experiment."""
from common import STRATEGIES, image_path, make_messages

def make_messages_api_visible(row, source, strategy, demos):
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    instruction = ('Answer the pathology multiple-choice question. Think step by step, then return '
                   'the final option letter within <answer> and </answer> tags.'
                   if strategy.endswith('cot') else
                   'Answer the pathology multiple-choice question. Return only the selected option letter. Do not explain.')
    if strategy == 'zero_shot_cot':
        instruction = (
            'Answer the pathology multiple-choice question. First, write a brief explanation '
            'in 2-4 sentences supporting your selected option, using only the information '
            'provided. The explanation must be visible in your response. Then, on a new line, '
            'return the final option letter within <answer> and </answer> tags. '
            'Do not return the answer letter alone. Keep the entire response concise so that '
            'both the explanation and final answer fit within 512 output tokens.'
        )
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

def make_messages_profile(row, source, strategy, demos, profile='standard'):
    if profile == 'standard':
        return make_messages(row, source, strategy, demos)
    if profile == 'api_visible_explanation':
        return make_messages_api_visible(row, source, strategy, demos)
    raise ValueError('Unknown prompt profile: '+profile)
