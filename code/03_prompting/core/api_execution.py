"""Bounded API-only workers; one coordinator writes each result file.

Requests may complete out of order. Item IDs, not line order, define pairing.
On errors, stop scheduling, drain in-flight workers, preserve successful answers,
then report failure. Failed requests are NOT recorded as incorrect/null answers.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
import time

from common import dataset, digest, gold, parse_answer
from finalization import generate_with_final_answer


EXECUTION_KEYS = ('api_concurrency', 'api_state_dir', 'api_rate_limit_wait',
                  'api_rate_limit_backoff', 'api_serial_interval', 'api_clear_stop')


def iter_api_results(backend, prepared, done_ids, args):
    items = iter((row, messages) for row, messages in prepared if str(row['num']) not in done_ids)

    def work(row, messages):
        started = time.perf_counter()
        try:
            result = generate_with_final_answer(backend, messages, row['options'], args.strategy,
                                                enabled=bool(args.force_answer_on_limit),
                                                answer_tokens=args.force_answer_max_tokens)
            error = None
        except Exception as exc:
            result, error = None, exc
        return row, messages, result, round(time.perf_counter()-started, 4), error

    with ThreadPoolExecutor(max_workers=args.api_concurrency, thread_name_prefix='api-item') as pool:
        pending = set()
        stopping = False
        while True:
            while not stopping and len(pending) < args.api_concurrency:
                item = next(items, None)
                if item is None:
                    break
                pending.add(pool.submit(work, *item))
            if not pending:
                return
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            outcomes = [future.result() for future in completed]
            if any(outcome[-1] is not None for outcome in outcomes):
                stopping = True
                if hasattr(backend, 'cancel_pending'):
                    backend.cancel_pending()
            for outcome in outcomes:
                yield outcome


def run_api_plans(plans, args, entry):
    # Late imports also let tests replace the backend without network access.
    from api_backends import APIBackend
    from runner import existing_results, write_json, summary

    backend = APIBackend(entry, args)
    settings = {key: str(getattr(args, key)) if key == 'api_state_dir' else getattr(args, key)
                for key in EXECUTION_KEYS}
    print(json.dumps({'event': 'api_scheduler', 'model': args.model,
                      'concurrency': args.api_concurrency, 'on_rate_limit': 'serial_until_manual_change',
                      'shared_state': str(args.api_state_dir)}, ensure_ascii=False), flush=True)
    for task, prepared, config, fingerprint, path in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix('.lock')
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError('Run lock exists: %s; check running processes before removing a stale lock' % lock)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        observed = []
        errors = []
        try:
            meta = path.with_suffix('.run.json')
            if not args.resume and (path.exists() or meta.exists()):
                raise FileExistsError(str(path))
            if args.resume and meta.exists():
                if json.loads(meta.read_text(encoding='utf8')).get('fingerprint') != fingerprint:
                    raise ValueError('Run metadata changed before acquiring output lock')
            observed = existing_results(path, fingerprint)
            write_json(meta, dict(fingerprint=fingerprint, config=config, backend=backend.info,
                                  execution_settings=settings))
            done = {str(row['num']) for row in observed}
            with path.open('a', encoding='utf8', buffering=1) as stream:
                for row, messages, result, elapsed, error in iter_api_results(backend, prepared, done, args):
                    if error is not None:
                        errors.append(dict(num=str(row['num']), error='%s: %s' % (type(error).__name__, error),
                                           initial_response=getattr(error, 'initial_response', None)))
                        write_json(path.with_suffix('.failure.json'),
                                   dict(fingerprint=fingerprint, completed=len(observed), failures=errors))
                        continue
                    pred = parse_answer(result['raw_output'], row['options'])
                    rec = dict(num=str(row['num']), dataset=dataset(row), task=task, model=args.model,
                               strategy=args.strategy, option_count=len(row['options']), gold_letter=gold(row),
                               pred_letter=pred, correct=pred == gold(row),
                               status='unparsed' if pred is None else 'ok', run_fingerprint=fingerprint,
                               item_sha256=digest(row), prompt_sha256=digest(messages), sampling_seed=None,
                               latency_seconds=elapsed, **result)
                    stream.write(json.dumps(rec, ensure_ascii=False)+'\n')
                    stream.flush()
                    observed.append(rec)
                    if len(observed) % 25 == 0:
                        print('%s: %d/%d' % (task, len(observed), len(prepared)), flush=True)
                    if args.print_raw_output:
                        print('%s: %s' % (rec['num'], result['raw_output']), flush=True)
            if errors:
                write_json(path.with_suffix('.failure.json'),
                           dict(fingerprint=fingerprint, completed=len(observed), failures=errors))
                raise RuntimeError('API run stopped with %d failed/unfinished items; %d answers saved. '
                                   'See %s; fix the cause and use --resume.' %
                                   (len(errors), len(observed), path.with_suffix('.failure.json')))
            failure_path = path.with_suffix('.failure.json')
            if failure_path.exists():
                old_failure = json.loads(failure_path.read_text(encoding='utf8'))
                old_failure.update(resolved=True, resolved_completed=len(observed))
                write_json(failure_path, old_failure)
        finally:
            write_json(path.with_suffix('.summary.json'),
                       summary(observed, config['expected_items'], task, args.strategy, args.model))
            lock.unlink(missing_ok=True)
        print(json.dumps(summary(observed, config['expected_items'], task, args.strategy, args.model),
                         ensure_ascii=False), flush=True)
