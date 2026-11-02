import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time
from common import (STRATEGIES, digest, file_hash, read_json, load_rows, load_demos,
                    make_messages, parse_answer, gold, dataset, validate_suite, index_rows)
from registry import ROOT, MODELS, TASK_FILES, GROUPS
from layout import RESULTS_DIR,result_path
from finalization import generate_with_final_answer
from task_demos import load_task_demos, POLICY as DEMO_POLICY
from prompt_profiles import make_messages_profile

HERE = Path(__file__).resolve().parent
DEMO_DEFAULT = HERE.parent / 'prompts/fewshot_data/fewshot_direct.json'

def parser(api):
    p = argparse.ArgumentParser(description='Paired pathology benchmark with fixed demonstration/prompt strategy')
    p.add_argument('--model', required=True, choices=[k for k,v in MODELS.items() if (v['backend'] in ('openai','gemini')) == api])
    p.add_argument('--task', default='baseline', choices=list(TASK_FILES) + list(GROUPS))
    p.add_argument('--strategy', choices=STRATEGIES, required=True)
    p.add_argument('--prompt-profile', choices=('standard', 'api_visible_explanation'), default='standard')
    p.add_argument('--data-root', type=Path, default=ROOT)
    p.add_argument('--input', type=Path, help='Custom list; only for a single task, reported separately')
    p.add_argument('--demos', type=Path, default=DEMO_DEFAULT)
    p.add_argument('--demo-suite', type=Path, help='Reviewed task-specific demo suite; defaults beside --demos')
    p.add_argument('--output-root', type=Path, default=RESULTS_DIR)
    p.add_argument('--limit', type=int)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--dry-run', action='store_true', help='No model import, no network, no result writes')
    p.add_argument('--resume', action='store_true', help='Only resumes exactly matching run fingerprints')
    p.add_argument('--max-new-tokens', type=int, default=None,
                   help='Default: 512 per independent request, including API direct')
    p.add_argument('--force-answer-on-limit', type=int, choices=(0,1), default=1,
                   help='CoT only: request a final letter once after token-limit truncation without an answer')
    p.add_argument('--force-answer-max-tokens', type=int, default=32,
                   help='Additional output budget for the final-letter request, separate from --max-new-tokens')
    p.add_argument('--temperature', type=float, default=None if api else 0.0)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--print-raw-output', action='store_true')
    if api:
        p.add_argument('--api-concurrency', type=int, choices=(1,2,3), default=3,
                       help='Per-model shared request cap across the three experiments; 429 downgrades to 1')
        p.add_argument('--api-state-dir', type=Path, help='Shared scheduler state; default: DATA_ROOT/.api_runtime')
        p.add_argument('--api-rate-limit-wait', type=float, default=900,
                       help='Bounded rate-limit retry window in seconds; saves progress on exhaustion')
        p.add_argument('--api-rate-limit-backoff', type=float, default=60,
                       help='Initial rate-limit cooldown; Retry-After may extend it')
        p.add_argument('--api-serial-interval', type=float, default=2,
                       help='Minimum seconds between starts after serial fallback')
        p.add_argument('--api-clear-stop', action='store_true',
                       help='Explicitly clear saved billing/daily-quota stop AFTER fixing the cause')
        p.add_argument('--base-url')
        p.add_argument('--timeout', type=float, default=180)
        p.add_argument('--retries', type=int, default=3, help='Transport retries only; may cause duplicate billing after timeouts')
        p.add_argument('--image-detail', choices=('auto','low','high'), default='auto')
        p.add_argument('--reasoning-effort', help='Optional OpenAI native setting; keep identical across strategies')
        p.add_argument('--thinking-level', default='default', choices=('default','minimal','low','medium','high'))
        p.add_argument('--thinking-budget', type=int)
    else:
        p.add_argument('--model-path', type=Path)
        p.add_argument('--repo-root', type=Path)
        p.add_argument('--torch-dtype', default='auto')
        p.add_argument('--health-dtype', default='bfloat16', choices=('bfloat16','float16','float32'))
        p.add_argument('--device-map', default='auto')
        p.add_argument('--device', default='cuda')
        p.add_argument('--conv-mode', default='llava_v1')
        p.add_argument('--attn-implementation', default='flash_attention_2', choices=('flash_attention_2','eager'))
        p.add_argument('--enable-thinking', type=int, default=0, choices=(0,1))
        p.add_argument('--max-context-tokens', type=int, default=0, help='Optional stricter context cap, never an expansion')
        p.add_argument('--max-mem-per-gpu', default='40GiB')
    return p

def token_budget(api, strategy, explicit=None):
    if explicit is not None:
        return explicit
    return 512

def summary(rows, expected, task, strategy, model):
    groups = defaultdict(list)
    for r in rows:
        groups['overall'].append(r)
        groups['dataset:'+r['dataset']].append(r)
        groups['choices:'+str(r['option_count'])].append(r)
    result = dict(task=task, strategy=strategy, model=model, expected=expected,
                  completed=len(rows), complete=len(rows)==expected, groups={})
    for key, values in groups.items():
        count = len(values)
        result['groups'][key] = dict(n=count, correct=sum(r['correct'] for r in values),
            accuracy_percent=100*sum(r['correct'] for r in values)/count,
            invalid=sum(r['pred_letter'] is None for r in values),
            truncated=sum(bool(r.get('truncated')) for r in values),
            finalization_attempted=sum(bool(r.get('finalization_attempted')) for r in values),
            finalization_succeeded=sum(r.get('finalization_succeeded') is True for r in values),
            finalization_failed=sum(r.get('finalization_succeeded') is False for r in values),
            generation_calls=sum(r.get('generation_calls',1) for r in values))
    return result

def write_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(temporary, path)

def existing_results(path, fingerprint):
    rows = []
    if path.exists():
        for line in path.read_text(encoding='utf8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)  # partial lines fail explicitly; never silently discard
            if row.get('run_fingerprint') != fingerprint:
                raise ValueError('Mixed run fingerprints in result file')
            rows.append(row)
        index_rows(rows)
    return rows

def main(api=False):
    a = parser(api).parse_args()
    if a.prompt_profile == 'api_visible_explanation' and not api:
        raise ValueError('Historical visible-explanation profile belongs to API runs only')
    a.max_new_tokens = token_budget(api, a.strategy, a.max_new_tokens)
    if a.max_new_tokens < 1 or a.force_answer_max_tokens < 1 or a.offset < 0 or (a.limit is not None and a.limit < 1):
        raise ValueError('Invalid token/item limits')
    if a.temperature is not None and a.temperature < 0:
        raise ValueError('Temperature must be nonnegative')
    a.output_root = a.output_root.expanduser().resolve()
    protected = []
    # Packaging-only safety guard: never put new predictions in historical
    # prompting or original six-test result roots, including a custom data root.
    for root in {ROOT.expanduser().resolve(), a.data_root.expanduser().resolve()}:
        protected.extend(root / name / 'results' for name in (
            'Benchmark_Zero-shot CoT', 'Benchmark_Few-shot direct', 'Benchmark_Few-shot CoT'))
        protected.extend(root / f'Benchmark_T{i}_results' for i in range(1, 8))
        protected.extend(root.glob('Benchmark_T*_results'))
    if any(p.resolve() == a.output_root or p.resolve() in a.output_root.parents for p in protected):
        raise ValueError('Historical result directories are read-only for this packaged runner; choose a new output root')
    a.data_root, a.demos = a.data_root.resolve(), a.demos.resolve()
    if api:
        from api_concurrency import validate_api_args
        validate_api_args(a)
        a.api_state_dir = (a.api_state_dir or a.data_root/'.api_runtime').expanduser().resolve()
    tasks = GROUPS.get(a.task, [a.task])
    if a.input and len(tasks) != 1:
        raise ValueError('--input requires a single task')
    if not a.input:
        validate_suite(a.data_root)
    entry = MODELS[a.model]
    plans = []
    semantic_args = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()
                     if k not in ('resume','dry_run','print_raw_output','task','output_root','limit','offset',
                                  'api_concurrency','api_state_dir','api_rate_limit_wait',
                                  'api_rate_limit_backoff','api_serial_interval','api_clear_stop')}
    if a.strategy.startswith('few'):
        suite = (a.demo_suite or a.demos.parent/'task_specific_v3/demo_suite.json').resolve()
        semantic_args.update(demonstration_policy=DEMO_POLICY, demo_suite=str(suite),
                             demo_suite_sha256=file_hash(suite))
    semantic_args['code_sha256'] = {p.name: file_hash(p) for p in sorted(HERE.glob('*.py'))}
    if not api:
        semantic_args['legacy_sha256'] = file_hash(entry['legacy'])
    else:
        env_base = 'OPENAI_BASE_URL' if entry['backend'] == 'openai' else 'GEMINI_BASE_URL'
        semantic_args['effective_base_url'] = a.base_url or os.getenv(env_base) or ('https://api.openai.com/v1' if entry['backend'] == 'openai' else 'https://generativelanguage.googleapis.com/v1beta')
        # URL is metadata; credentials must only be provided via the key environment variables.
        from urllib.parse import urlsplit
        u = urlsplit(semantic_args['effective_base_url'])
        if u.username or u.password or u.query or u.fragment or u.scheme != 'https':
            raise ValueError('Invalid base URL; credentials/query strings are not accepted')
    for task in tasks:
        demos, demo_provenance = load_task_demos(a.strategy, a.demos, task, a.demo_suite)
        rows, source = load_rows(task, a.data_root, a.input)
        all_count = len(rows)
        rows = rows[a.offset:(a.offset+a.limit) if a.limit is not None else None]
        if not rows:
            raise ValueError('Empty selected slice')
        prepared = []
        for r in rows:
            msgs = make_messages_profile(r, source, a.strategy, demos, a.prompt_profile)
            for d in demos:
                if d['num'] == str(r['num']) or (r.get('img') and d['image'] and
                                                Path(r['img']).resolve() == Path(d['image']).resolve()):
                    raise ValueError('Demonstration-target overlap')
            prepared.append((r,msgs))
        image_paths = sorted({c['image'] for _,ms in prepared for m in ms for c in m['content'] if c['type']=='image'})
        config = dict(arguments=semantic_args, task=task, strategy=a.strategy, model=a.model, model_config=entry,
            input=str(source), input_sha256=file_hash(source), demonstrations=demo_provenance,
            image_sha256={p:file_hash(p) for p in image_paths}, selected_ids=[str(r['num']) for r in rows],
            expected_items=all_count)
        demo_hashes=set(demo_provenance.get('images',{}).values())
        for row,messages in prepared:
            target_images=[c['image'] for c in messages[-1]['content'] if c['type']=='image']
            if any(config['image_sha256'][p] in demo_hashes for p in target_images):
                raise ValueError(f"Exact demonstration-target image overlap: {row['num']}")
        fingerprint = digest(config)
        # Smoke tests, custom inputs and alternate strategy controls cannot contaminate full-run files.
        outdir = a.output_root
        outdir = outdir/a.strategy/a.prompt_profile
        if a.input or a.limit is not None or a.offset:
            outdir = outdir/'pilot'/fingerprint[:12]
        path = result_path(outdir,a.model,task)
        plans.append((task,prepared,config,fingerprint,path))
        print(json.dumps(dict(task=task, model=a.model, strategy=a.strategy, n=len(rows),
            expected=all_count, max_new_tokens=a.max_new_tokens,
            api_concurrency=a.api_concurrency if api else None,
            force_answer_on_limit=bool(a.force_answer_on_limit and a.strategy.endswith('cot')),
            force_answer_max_tokens=a.force_answer_max_tokens,
            examples=len(demos), first_item_images=sum(c['type']=='image' for m in prepared[0][1] for c in m['content']),
            demonstration_policy=demo_provenance.get('protocol', 'no_demonstrations'),
            demonstration_bank=demo_provenance.get('bank'),
            demonstration_variants=demo_provenance.get('variants', []),
            demonstration_pairs=demo_provenance.get('pair_count',0),
            output=str(path), fingerprint=fingerprint), ensure_ascii=False), flush=True)
    if a.dry_run:
        print(json.dumps({'first_prompt': plans[0][1][0][1], 'note': 'No model/API called; no result written'}, ensure_ascii=False, indent=2))
        return
    # Validate all destination files BEFORE any model load or paid request.
    for task, prepared, config, fingerprint, path in plans:
        meta = path.with_suffix('.run.json')
        if path.exists() or meta.exists():
            if not a.resume:
                raise FileExistsError(f'{path}: choose --resume or a new --output-root')
            if not meta.exists() or read_json(meta).get('fingerprint') != fingerprint:
                raise ValueError(f'Cannot resume incompatible configuration: {path}')
            existing_results(path, fingerprint)
    if all(len(existing_results(path,fp))==len(prepared) for _,prepared,_,fp,path in plans):
        print('All requested items already completed; no model loaded or provider called.')
        return
    if api:
        from api_execution import run_api_plans
        run_api_plans(plans, a, entry)
        return
    random.seed(a.seed)
    if api:
        from api_backends import APIBackend
        backend = APIBackend(entry,a)
    else:
        import torch
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        from local_backends import LocalBackend
        backend = LocalBackend(entry,a)
    for task, prepared, config, fingerprint, path in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix('.lock')
        # Exclusive lock prevents concurrent processes appending to the same results.
        try:
            lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f'Run lock exists: {lock}; check the process before removing a stale lock')
        os.write(lock_fd, str(os.getpid()).encode()); os.close(lock_fd)
        observed = []
        try:
            if not a.resume and (path.exists() or path.with_suffix('.run.json').exists()):
                raise FileExistsError(str(path))
            if a.resume and path.with_suffix('.run.json').exists():
                if read_json(path.with_suffix('.run.json')).get('fingerprint')!=fingerprint:
                    raise ValueError('Run metadata changed while waiting for model load/lock')
            write_json(path.with_suffix('.run.json'), dict(fingerprint=fingerprint, config=config, backend=backend.info))
            observed = existing_results(path,fingerprint)
            done = {r['num'] for r in observed}
            with path.open('a',encoding='utf8',buffering=1) as f:
                for row,messages in prepared:
                    if str(row['num']) in done:
                        continue
                    started = time.perf_counter()
                    sampling_seed=(a.seed+int(digest(str(row['num']))[:8],16))%(2**32)
                    if not api:
                        torch.manual_seed(sampling_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(sampling_seed)
                    try:
                        result = generate_with_final_answer(backend, messages, row['options'], a.strategy,
                            enabled=bool(a.force_answer_on_limit), answer_tokens=a.force_answer_max_tokens)
                    except Exception as e:
                        # Infrastructure/context errors stop, so they cannot silently become 900 model mistakes.
                        write_json(path.with_suffix('.failure.json'), dict(num=str(row['num']),
                            error=f'{type(e).__name__}: {e}', completed=len(observed), fingerprint=fingerprint,
                            initial_response=getattr(e,'initial_response',None)))
                        raise
                    pred = parse_answer(result['raw_output'],row['options'])
                    rec = dict(num=str(row['num']),dataset=dataset(row),task=task,model=a.model,
                        strategy=a.strategy,option_count=len(row['options']),gold_letter=gold(row),
                        pred_letter=pred,correct=pred==gold(row),status='unparsed' if pred is None else 'ok',
                        run_fingerprint=fingerprint,item_sha256=digest(row),prompt_sha256=digest(messages),
                        sampling_seed=sampling_seed if not api else None,
                        latency_seconds=round(time.perf_counter()-started,4),**result)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    f.flush()
                    observed.append(rec)
                    if len(observed)%25 == 0:
                        print(f'{task}: {len(observed)}/{len(prepared)}',flush=True)
                    if a.print_raw_output:
                        print(f"{rec['num']}: {result['raw_output']}",flush=True)
        finally:
            write_json(path.with_suffix('.summary.json'), summary(observed,config['expected_items'],task,a.strategy,a.model))
            lock.unlink(missing_ok=True)
        print(json.dumps(summary(observed,config['expected_items'],task,a.strategy,a.model),ensure_ascii=False),flush=True)
