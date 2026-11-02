"""Offline tests only: HTTP is mocked; no keys or paid calls required."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/'02_scoring/core'))
from concurrent.futures import ThreadPoolExecutor
import contextlib
from email.utils import formatdate
import io
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from api_backends import APIBackend
from api_concurrency import ModelGate, APIStopped, APIQuotaError, classify_http_error, retry_delay


MESSAGES = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Return A'}]}]
OPTIONS = ['A) one', 'B) two']


def args_for(root, **changes):
    values = dict(model='gpt-test', base_url=None, max_new_tokens=512, temperature=None,
                  reasoning_effort=None, image_detail='auto', timeout=1, retries=1,
                  thinking_level='default', thinking_budget=None,
                  api_state_dir=Path(root), api_concurrency=3, api_rate_limit_wait=2,
                  api_rate_limit_backoff=0.01, api_serial_interval=0, api_clear_stop=False)
    values.update(changes)
    return SimpleNamespace(**values)


def response(kind='openai'):
    if kind == 'openai':
        data = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': 'A'}]}], 'usage': {'output_tokens': 1}}
    else:
        data = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': 'A'}]}}]}
    return io.StringIO(json.dumps(data))


def http_error(payload, headers=None, status=429):
    return urllib.error.HTTPError('https://not-called.invalid', status, 'mock', headers or {},
                                  io.BytesIO(json.dumps(payload).encode()))


def process_slot(root, barrier, active, peak, lock):
    gate = ModelGate(root, 'test', 'https://not-called.invalid', 'test', 3, 0)
    barrier.wait(timeout=20)
    for _ in range(3):
        with gate.slot(threading.Event()):
            with lock:
                active.value += 1
                peak.value = max(peak.value, active.value)
            time.sleep(0.02)
            with lock:
                active.value -= 1


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.keys = patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-test-only',
                                           'GEMINI_API_KEY': 'offline-test-only',
                                           'OPENAI_BASE_URL': 'https://api.openai.com/v1',
                                           'GEMINI_BASE_URL': 'https://generativelanguage.googleapis.com/v1beta'})
        self.keys.start()
        self.addCleanup(self.keys.stop)

    def test_classifier_and_retry_after(self):
        self.assertEqual(classify_http_error(429, {'error': {'code': 'insufficient_quota'}}), 'billing_or_quota')
        self.assertEqual(classify_http_error(402, {}), 'billing_or_quota')
        for quota in ('GenerateRequestsPerDayPerProjectPerModel', 'requests_per_day'):
            payload = {'error': {'status': 'RESOURCE_EXHAUSTED', 'details': [{'quotaId': quota}]}}
            self.assertEqual(classify_http_error(429, payload), 'daily_quota')
        self.assertEqual(classify_http_error(429, {'error': {'code': 'rate_limit_exceeded'}}), 'rate_limit')
        self.assertEqual(classify_http_error(503, {'error': {'code': 'slow_down'}}), 'rate_limit')
        self.assertEqual(classify_http_error(503, {}), 'transient')
        self.assertEqual(classify_http_error(401, {}), 'request_error')
        self.assertEqual(retry_delay({'Retry-After': '12'}, {}), 12)
        self.assertEqual(retry_delay({}, {'error': {'details': [{'retryDelay': '3.5s'}]}}), 3.5)
        self.assertGreater(retry_delay({'Retry-After': formatdate(time.time()+20, usegmt=True)}, {}), 18)

    def test_default_and_local_settings(self):
        import runner
        a = runner.parser(True).parse_args(['--strategy', 'zero_shot_cot', '--model', 'gpt-5.6-luna'])
        self.assertEqual(a.api_concurrency, 3)
        for strategy in ('few_shot_direct', 'few_shot_cot', 'zero_shot_cot'):
            self.assertEqual(runner.token_budget(True, strategy), 512)
        local = runner.parser(False).parse_args(['--strategy', 'zero_shot_cot', '--model', 'qwen35-2b'])
        self.assertFalse(hasattr(local, 'api_concurrency'))

    def test_shared_cap_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context('spawn')
            active, peak, lock = ctx.Value('i', 0), ctx.Value('i', 0), ctx.Lock()
            barrier = ctx.Barrier(5)
            processes = [ctx.Process(target=process_slot, args=(tmp, barrier, active, peak, lock)) for _ in range(5)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            self.assertLessEqual(peak.value, 3)
            self.assertGreaterEqual(peak.value, 2)
            self.assertEqual(active.value, 0)

    def test_downgrade_waits_for_existing_calls_to_drain(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            first = ModelGate(tmp, 'test', 'https://test.invalid', 'one', 3, 0)
            second = ModelGate(tmp, 'test', 'https://test.invalid', 'one', 3, 0)
            entered = threading.Event()
            slot1 = first.slot(threading.Event())
            slot2 = first.slot(threading.Event())
            slot1.__enter__(); slot2.__enter__()
            first.throttle(0)
            def worker():
                with second.slot(threading.Event()):
                    entered.set()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(worker)
                self.assertFalse(entered.wait(0.08))
                slot1.__exit__(None, None, None)
                self.assertFalse(entered.wait(0.08))
                slot2.__exit__(None, None, None)
                future.result(timeout=2)
            self.assertTrue(entered.is_set())
            self.assertTrue(second.snapshot()['serial'])
            other = ModelGate(tmp, 'test', 'https://test.invalid', 'other-model', 3, 0)
            self.assertFalse(other.snapshot()['serial'])

    def test_budget_is_not_shared_between_requests(self):
        for kind in ('openai', 'gemini'):
            with tempfile.TemporaryDirectory() as tmp:
                backend = APIBackend({'backend': kind}, args_for(tmp))
                barrier = threading.Barrier(3)
                budgets = []
                def send(request, **kwargs):
                    payload = json.loads(request.data)
                    budgets.append(payload['max_output_tokens'] if kind == 'openai' else
                                   payload['generationConfig']['maxOutputTokens'])
                    barrier.wait(timeout=3)
                    return response(kind)
                with patch('urllib.request.urlopen', send), ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(backend, MESSAGES), pool.submit(backend, MESSAGES),
                               pool.submit(backend.final_answer, MESSAGES, 32)]
                    for future in futures:
                        self.assertEqual(future.result(timeout=4)['raw_output'], 'A')
                self.assertEqual(sorted(budgets), [32, 512, 512])
                self.assertEqual(backend.args.max_new_tokens, 512)

    def test_rate_limit_retries_same_payload_in_serial(self):
        for kind in ('openai', 'gemini'):
            with tempfile.TemporaryDirectory() as tmp:
                backend = APIBackend({'backend': kind}, args_for(tmp))
                bodies, started = [], []
                def send(request, **kwargs):
                    bodies.append(request.data)
                    started.append(time.monotonic())
                    if len(bodies) == 1:
                        raise http_error({'error': {'code': 'rate_limit_exceeded',
                                         'details': [{'retryDelay': '0.06s'}]}}, {'Retry-After': '0.04'})
                    return response(kind)
                with patch('urllib.request.urlopen', send):
                    result = backend(MESSAGES)
                    followup = backend(MESSAGES)
                self.assertEqual(bodies[0], bodies[1])
                self.assertGreaterEqual(started[1]-started[0], 0.06)
                self.assertEqual(result['transport_attempts'], 2)
                self.assertEqual(result['api_concurrency_at_request'], 1)
                self.assertEqual(followup['api_concurrency_at_request'], 1)
                self.assertTrue(backend.gate.snapshot()['serial'])

    def test_billing_failure_blocks_other_workers_without_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = APIBackend({'backend': 'openai'}, args_for(tmp))
            error = http_error({'error': {'code': 'credit_balance_exhausted', 'message': 'SECRET_NOT_LOGGED'}})
            with patch('urllib.request.urlopen', side_effect=error) as send:
                with self.assertRaises(APIQuotaError) as caught:
                    backend(MESSAGES)
                self.assertEqual(send.call_count, 1)
                self.assertNotIn('SECRET_NOT_LOGGED', str(caught.exception))
                other = APIBackend({'backend': 'openai'}, args_for(tmp))
                with self.assertRaises(APIQuotaError):
                    other(MESSAGES)
                self.assertEqual(send.call_count, 1)
            restored = APIBackend({'backend': 'openai'}, args_for(tmp, api_clear_stop=True))
            with patch('urllib.request.urlopen', return_value=response()):
                self.assertEqual(restored(MESSAGES)['raw_output'], 'A')

    def test_permanent_daily_quota_stops_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = APIBackend({'backend': 'gemini'}, args_for(tmp))
            error = http_error({'error': {'details': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel'}]}})
            with patch('urllib.request.urlopen', side_effect=error) as send:
                with self.assertRaises(APIQuotaError):
                    backend(MESSAGES)
                self.assertEqual(send.call_count, 1)

    def test_wait_budget_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = APIBackend({'backend': 'openai'}, args_for(tmp, api_rate_limit_wait=0.02))
            error = http_error({'error': {'code': 'rate_limit_exceeded'}}, {'Retry-After': '30'})
            with patch('urllib.request.urlopen', side_effect=error) as send:
                with self.assertRaises(APIStopped):
                    backend(MESSAGES)
                self.assertEqual(send.call_count, 1)
                self.assertTrue(backend.gate.snapshot()['serial'])

    def test_error_drains_successes_and_resume_skips_them(self):
        import runner
        barrier = threading.Barrier(3)
        class Fake:
            info = {'backend': 'offline-mock'}
            calls = []
            fail = True
            def __init__(self, *args):
                pass
            def __call__(self, messages):
                text = json.dumps(messages)
                index = next(i for i in range(5) if 'Question %d?' % i in text)
                Fake.calls.append(index)
                if Fake.fail:
                    barrier.wait(timeout=3)
                    if index == 1:
                        raise RuntimeError('simulated failure')
                    time.sleep(0.05)
                return {'raw_output': 'A', 'truncated': False}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'strategy.json').write_text(json.dumps({'strategy': 'zero_shot_cot'}))
            source = folder/'input.json'
            source.write_text(json.dumps([dict(num=str(i), img='', question='Question %d?' % i,
                                              options=OPTIONS, answer=OPTIONS[0]) for i in range(5)]))
            argv = ['eval_api.py', '--strategy', 'zero_shot_cot', '--model', 'gpt-5.6-luna', '--input', str(source),
                    '--data-root', str(folder), '--output-root', str(folder/'results')]
            with patch.object(runner, 'HERE', folder), patch('api_backends.APIBackend', Fake),\
                 contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', argv), self.assertRaises(RuntimeError):
                    runner.main(api=True)
                path = next((folder/'results').rglob('baseline.jsonl'))
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual({r['num'] for r in records}, {'0', '2'})
                self.assertTrue(all(r['pred_letter'] == 'A' for r in records))
                self.assertFalse(path.with_suffix('.lock').exists())
                self.assertEqual(sorted(Fake.calls), [0, 1, 2])
                Fake.fail = False
                with patch('sys.argv', argv+['--resume', '--api-concurrency', '1']):
                    runner.main(api=True)
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(records), 5)
                self.assertEqual({r['num'] for r in records}, {'0', '1', '2', '3', '4'})
                self.assertEqual(Fake.calls.count(0), 1)
                self.assertEqual(Fake.calls.count(2), 1)
                self.assertEqual(json.loads(path.with_suffix('.summary.json').read_text())['completed'], 5)
                self.assertTrue(json.loads(path.with_suffix('.failure.json').read_text())['resolved'])
                calls = len(Fake.calls)
                with patch('sys.argv', argv+['--resume']):
                    runner.main(api=True)
                self.assertEqual(len(Fake.calls), calls)


if __name__ == '__main__':
    unittest.main()
