"""Native Responses/Gemini REST; no SDK installation, no hard-coded credentials."""
import base64
from contextlib import nullcontext
from copy import copy
import json
import mimetypes
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from api_concurrency import (ModelGate, APIStopped, APIQuotaError,
                             classify_http_error, retry_delay)

def encoded_image(path):
    mime = mimetypes.guess_type(path)[0]
    if mime not in ('image/png', 'image/jpeg', 'image/webp', 'image/gif'):
        raise ValueError(f'Unsupported API image format (no silent conversion): {path}')
    return mime, base64.b64encode(Path(path).read_bytes()).decode('ascii')

def openai_payload(messages, model, args):
    inputs = []
    for msg in messages:
        if msg['role'] == 'assistant':
            inputs.append({'role': 'assistant', 'content': '\n'.join(c['text'] for c in msg['content'])})
            continue
        content = []
        for c in msg['content']:
            if c['type'] == 'text':
                content.append({'type': 'input_text', 'text': c['text']})
            else:
                mime, b64 = encoded_image(c['image'])
                content.append({'type': 'input_image', 'image_url': f'data:{mime};base64,{b64}',
                                'detail': args.image_detail})
        inputs.append({'role': 'user', 'content': content})
    payload = dict(model=model, input=inputs, max_output_tokens=args.max_new_tokens, store=False)
    if args.temperature is not None:
        payload['temperature'] = args.temperature
    if args.reasoning_effort:
        payload['reasoning'] = {'effort': args.reasoning_effort}
    return payload

def gemini_payload(messages, args):
    contents = []
    for msg in messages:
        parts = []
        for c in msg['content']:
            if c['type'] == 'text':
                parts.append({'text': c['text']})
            else:
                mime, b64 = encoded_image(c['image'])
                parts.append({'inline_data': {'mime_type': mime, 'data': b64}})
        contents.append({'role': 'model' if msg['role'] == 'assistant' else 'user', 'parts': parts})
    config = dict(maxOutputTokens=args.max_new_tokens)
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.thinking_budget is not None:
        config['thinkingConfig'] = {'thinkingBudget': args.thinking_budget}
    elif args.thinking_level != 'default':
        config['thinkingConfig'] = {'thinkingLevel': args.thinking_level.upper()}
    return dict(contents=contents, generationConfig=config)

def extract_response(data, kind):
    if kind == 'openai':
        blocks = [c['text'] for item in data.get('output', []) if item.get('type') == 'message'
                  for c in item.get('content', []) if c.get('type') == 'output_text']
        status = data.get('status')
        exhausted = status == 'incomplete' and (data.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'
        return dict(raw_output='\n'.join(blocks), usage=data.get('usage'), response_id=data.get('id'),
                    finish_reason=status, truncated=(status == 'incomplete'), token_limit_reached=exhausted,
                    incomplete_details=data.get('incomplete_details'), returned_model=data.get('model'))
    candidates = data.get('candidates', [])
    first = candidates[0] if candidates else {}
    blocks = [p['text'] for p in first.get('content', {}).get('parts', [])
              if 'text' in p and not p.get('thought')]
    return dict(raw_output='\n'.join(blocks), usage=data.get('usageMetadata'),
                response_id=data.get('responseId'), returned_model=data.get('modelVersion'),
                finish_reason=first.get('finishReason'),
                truncated=first.get('finishReason') == 'MAX_TOKENS',
                token_limit_reached=first.get('finishReason') == 'MAX_TOKENS',
                prompt_feedback=data.get('promptFeedback'))

class APIBackend:
    def __init__(self, entry, args):
        self.kind, self.args = entry['backend'], args
        if self.kind == 'openai':
            self.base = args.base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            self.key = os.getenv('OPENAI_API_KEY')
        else:
            self.base = args.base_url or os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta')
            self.key = os.getenv('GEMINI_API_KEY')
        parsed = urllib.parse.urlsplit(self.base)
        if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use an HTTPS base URL without credentials/query parameters')
        if not self.key:
            raise RuntimeError('Set OPENAI_API_KEY or GEMINI_API_KEY in the environment; no API request was sent')
        self.cancelled = threading.Event()
        state_dir = getattr(args, 'api_state_dir', None)
        self.gate = (ModelGate(state_dir, self.kind, self.base, args.model,
                              getattr(args, 'api_concurrency', 3),
                              getattr(args, 'api_serial_interval', 2.0)) if state_dir else None)
        if self.gate and getattr(args, 'api_clear_stop', False):
            self.gate.clear_stop()
        self.info = dict(backend=self.kind, base_url=self.base,
                         api_concurrency=getattr(args, 'api_concurrency', 3),
                         throttle_policy='shared_model_3_to_1_v1',
                         scheduler_state=str(self.gate.state_path) if self.gate else None)

    def cancel_pending(self):
        self.cancelled.set()

    def final_answer(self, messages, max_new_tokens):
        # Do not invent unsupported native-reasoning settings for API models.
        # The final instruction is answer-only, but provider-internal reasoning
        # may still consume this budget; failed finalizations remain unparsed.
        args = copy(self.args)
        args.max_new_tokens = max_new_tokens
        return self._send(messages, args)

    def __call__(self, messages):
        return self._send(messages, self.args)

    def _send(self, messages, a):
        headers = {'Content-Type': 'application/json'}
        if self.kind == 'openai':
            url = self.base.rstrip('/') + '/responses'
            headers['Authorization'] = 'Bearer ' + self.key
            payload = openai_payload(messages, a.model, a)
        else:
            name = urllib.parse.quote(a.model, safe='')
            url = self.base.rstrip('/') + f'/models/{name}:generateContent'
            headers['x-goog-api-key'] = self.key
            payload = gemini_payload(messages, a)
        body = json.dumps(payload, ensure_ascii=False).encode()
        attempts = 0
        transport_failures = 0
        rate_failures = 0
        rate_deadline = None
        max_wait = getattr(a, 'api_rate_limit_wait', 900.0)
        base_wait = getattr(a, 'api_rate_limit_backoff', 60.0)
        while True:
            if self.cancelled.is_set():
                raise APIStopped('API worker stopped; no new request sent')
            if rate_deadline is not None and time.monotonic() >= rate_deadline:
                raise APIStopped('API rate-limit retry window exhausted; progress saved, resume later')
            request = urllib.request.Request(url, body, headers=headers, method='POST')
            wait_budget = max_wait if rate_deadline is None else max(0.01, rate_deadline-time.monotonic())
            slot = self.gate.slot(self.cancelled, wait_budget) if self.gate else nullcontext(1)
            with slot as active_limit:
                attempts += 1
                try:
                    with urllib.request.urlopen(request, timeout=a.timeout) as response:
                        data = json.load(response)
                    if data.get('error') is not None:
                        raise RuntimeError('API returned an error object; inspect provider logs (body not logged)')
                    result = extract_response(data, self.kind)
                    result.update(transport_attempts=attempts, rate_limit_retries=rate_failures,
                                  api_concurrency_at_request=active_limit)
                    return result
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    try:
                        # Parse only to classify; NEVER log/raise the body or URL.
                        try:
                            error_data = json.loads(exc.read(65536))
                        except (ValueError, TypeError, UnicodeError):
                            error_data = {}
                        category = classify_http_error(status, error_data)
                        suggested = retry_delay(exc.headers, error_data)
                    finally:
                        exc.close()
                    if category in ('billing_or_quota', 'daily_quota'):
                        if self.gate:
                            self.gate.block(category)
                        self.cancel_pending()
                        raise APIQuotaError('API HTTP %s: %s; no automatic retry; '
                                            'fix quota/billing, then --resume --api-clear-stop' %
                                            (status, category)) from None
                    if category == 'rate_limit':
                        if rate_deadline is None:
                            rate_deadline = time.monotonic()+max_wait
                        delay = max(suggested, min(base_wait*(2**min(rate_failures, 4)), 300.0))
                        delay += random.uniform(0, min(1.0, delay/10))
                        rate_failures += 1
                        if self.gate:
                            # Set shared state BEFORE releasing the occupied slot.
                            self.gate.throttle(delay)
                        if time.monotonic()+delay >= rate_deadline:
                            raise APIStopped('API rate-limit wait exceeds retry window; resume later') from None
                        if not self.gate and self.cancelled.wait(delay):
                            raise APIStopped('API worker stopped') from None
                        continue
                    if category != 'transient' or transport_failures >= a.retries:
                        raise RuntimeError('API HTTP %s: %s; request not altered' %
                                           (status, category)) from None
                    delay = max(suggested, min(2**transport_failures, 30))
                except (urllib.error.URLError, TimeoutError):
                    if transport_failures >= a.retries:
                        raise RuntimeError('API transport failed; request not altered') from None
                    delay = min(2**transport_failures, 30)
                transport_failures += 1
                if delay > max_wait:
                    raise APIStopped('Provider retry delay exceeds retry window; resume later')
                if self.cancelled.wait(delay):
                    raise APIStopped('API worker stopped')
