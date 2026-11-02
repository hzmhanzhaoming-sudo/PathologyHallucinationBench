"""Cross-process per-model request slots and persistent 3 -> 1 throttling.

Only scheduler metadata is stored here, never credentials, prompts or answers.
The three experiments must use the same --api-state-dir (default: data root).
OS locks release automatically if a process exits; state writes are atomic.
"""
from contextlib import contextmanager
from datetime import timezone
from email.utils import parsedate_to_datetime
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


class APIStopped(RuntimeError):
    pass


class APIQuotaError(APIStopped):
    pass


def retry_delay(headers, payload):
    delays = []
    value = headers.get('Retry-After') if headers else None
    if value:
        try:
            delays.append(float(value))
        except (TypeError, ValueError):
            try:
                dt = parsedate_to_datetime(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                delays.append(dt.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    err = payload.get('error', {}) if isinstance(payload, dict) else {}
    if isinstance(err, dict):
        for detail in err.get('details', []) or []:
            if not isinstance(detail, dict):
                continue
            value = detail.get('retryDelay')
            try:
                if isinstance(value, str) and value.endswith('s'):
                    delays.append(float(value[:-1]))
                elif isinstance(value, dict):
                    delays.append(float(value.get('seconds', 0)) + float(value.get('nanos', 0))/1e9)
            except (TypeError, ValueError):
                pass
    return max([0.0] + [x for x in delays if math.isfinite(x)])


def classify_http_error(status, payload):
    """Return a safe category, never echo an HTTP body (it can contain secrets)."""
    err = payload.get('error', {}) if isinstance(payload, dict) else {}
    if not isinstance(err, dict):
        err = {}
    code = str(err.get('code', '')).lower()
    kind = str(err.get('type', '')).lower()
    description = json.dumps(err, ensure_ascii=True).lower()
    permanent = {'insufficient_quota', 'credit_balance_exhausted',
                 'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
                 'organization_usage_limit_exceeded', 'usage_limit_exceeded'}
    if status == 402 or code in permanent or kind in permanent:
        return 'billing_or_quota'
    if status == 429 and any(x in description for x in (
            'perday', 'per_day', 'per day', 'daily quota', 'requests/day', 'tokens/day')):
        return 'daily_quota'
    if status == 429 and any(x in description for x in (
            'insufficient_quota', 'credit balance', 'prepay credit', 'billing account disabled',
            'billing is disabled', 'monthly spend', 'limit: 0,', 'limit: 0\\n')):
        return 'billing_or_quota'
    if status == 429 or (status == 503 and code == 'slow_down'):
        return 'rate_limit'
    if status in (408, 500, 502, 503, 504):
        return 'transient'
    return 'request_error'


def _try_lock(path):
    handle = open(path, 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return None
        raise


def _unlock(handle):
    if handle is None:
        return
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class ModelGate:
    def __init__(self, root, provider, base_url, model, limit=3, serial_interval=2.0):
        if limit not in (1, 2, 3) or not math.isfinite(serial_interval) or serial_interval < 0:
            raise ValueError('API concurrency must be 1..3; interval must be nonnegative')
        self.limit, self.serial_interval = limit, serial_interval
        key = hashlib.sha256((provider+'|'+base_url.rstrip('/')+'|'+model).encode()).hexdigest()[:24]
        self.folder = Path(root)/key
        self.folder.mkdir(parents=True, exist_ok=True)
        self.state_path = self.folder/'state.json'
        self.model = model

    @contextmanager
    def _state_lock(self):
        handle = None
        while handle is None:
            handle = _try_lock(self.folder/'state.lock')
            if handle is None:
                time.sleep(0.02)
        try:
            yield
        finally:
            _unlock(handle)

    def _read(self):
        if not self.state_path.exists():
            return {'version': 1, 'serial': False, 'cooldown_until': 0.0,
                    'next_start': 0.0, 'rate_limit_events': 0, 'stop_reason': None}
        return json.loads(self.state_path.read_text(encoding='utf8'))

    def _write(self, state):
        tmp = self.folder/'state.json.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
        os.replace(tmp, self.state_path)

    def snapshot(self):
        with self._state_lock():
            return self._read()

    def throttle(self, delay):
        with self._state_lock():
            state = self._read()
            first = not state['serial']
            state.update(serial=True, cooldown_until=max(state['cooldown_until'], time.time()+delay),
                         rate_limit_events=state['rate_limit_events']+1)
            self._write(state)
        print(json.dumps({'event': 'api_rate_limit', 'model': self.model,
                          'concurrency': 1, 'first_downgrade': first,
                          'cooldown_seconds': round(delay, 2)}, ensure_ascii=False), flush=True)

    def block(self, reason):
        with self._state_lock():
            state = self._read()
            state['stop_reason'] = reason
            self._write(state)

    def clear_stop(self):
        """Explicit recovery after fixing billing/daily quota; retain serial fallback."""
        with self._state_lock():
            state = self._read()
            state['stop_reason'] = None
            self._write(state)

    @contextmanager
    def slot(self, cancelled, wait_timeout=900):
        owned = None
        actual_limit = self.limit
        deadline = time.monotonic()+wait_timeout
        while owned is None:
            if cancelled.is_set():
                raise APIStopped('API worker stopped; saved answers are retained')
            if time.monotonic() >= deadline:
                raise APIStopped('Timed out waiting for API slots/cooldown; resume later')
            with self._state_lock():
                state = self._read()
                if state['stop_reason']:
                    raise APIQuotaError('API stopped: '+state['stop_reason']+
                                        '; fix account/quota, then use --resume --api-clear-stop')
                now = time.time()
                actual_limit = 1 if state['serial'] else self.limit
                ready_at = max(state['cooldown_until'], state['next_start'])
                if now >= ready_at:
                    free = []
                    try:
                        # Probe all 3 slots: after downgrade, wait for requests
                        # already in flight to drain before admitting ONE new call.
                        for index in range(3):
                            handle = _try_lock(self.folder/('slot%d.lock' % index))
                            if handle is not None:
                                free.append(handle)
                        active = 3-len(free)
                        if active < actual_limit:
                            owned = free.pop(0)
                            if actual_limit == 1:
                                state['next_start'] = now+self.serial_interval
                                self._write(state)
                    finally:
                        for handle in free:
                            _unlock(handle)
            if owned is None:
                cancelled.wait(0.05)
        try:
            yield actual_limit
        finally:
            _unlock(owned)


def validate_api_args(args):
    if args.api_concurrency not in (1, 2, 3):
        raise ValueError('--api-concurrency must be 1, 2 or 3')
    for name in ('api_rate_limit_wait', 'api_rate_limit_backoff', 'api_serial_interval'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid '+name)
    if args.api_rate_limit_wait <= 0 or args.api_rate_limit_backoff <= 0:
        raise ValueError('API wait and backoff must be positive')
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 0:
        raise ValueError('Invalid API timeout/retry limit')
