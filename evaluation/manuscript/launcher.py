"""Safe routing only: never reconstruct prompts, predictions, or scores."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlparse

from registry import HERE, DATA_ROOT, CONDITIONS, GROUPS, MODELS, expand_task, script_relative

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parser(kind):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=[k for k,v in MODELS.items() if v['api']==(kind=='api')])
    p.add_argument('--task', choices=list(GROUPS), default='all')
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--run-id', default='preview')
    g=p.add_mutually_exclusive_group(); g.add_argument('--execute',action='store_true'); g.add_argument('--dry-run',action='store_true')
    p.add_argument('--validate-inputs',action='store_true',help='Read/hash inputs, with no model calls in dry-run')
    p.add_argument('--python', default=None)
    p.add_argument('--limit',type=int)
    p.add_argument('--max-tokens',type=int,default=None)
    p.add_argument('--temperature',type=float,default=None)
    p.add_argument('--retry-invalid',type=int,default=None)
    if kind=='local': p.add_argument('--model-path',type=Path)
    if kind=='api': p.add_argument('--thinking-level',choices=['default','minimal','low','medium','high'])
    return p

def source_dependencies(path,seen=None):
    """Follow only bundled sibling eval modules; never import their top-level code."""
    seen=set() if seen is None else seen
    if path in seen:return []
    seen.add(path);result=[]
    tree=ast.parse(path.read_text(encoding='utf8'))
    for node in ast.walk(tree):
        names=[]
        if isinstance(node,ast.Import):names=[a.name for a in node.names]
        elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
        for name in names:
            if name.startswith('eval') and '.' not in name:
                sibling=path.parent/(name+'.py')
                if sibling.is_file():result.extend(source_dependencies(sibling,seen))
    result.append(path)
    return result

def literal_defaults(path):
    """Read defaults without importing a GPU or API module."""
    out={}
    for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            names=[a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str)]
            if not names: continue
            d=next((k.value for k in node.keywords if k.arg=='default'),None)
            if isinstance(d,ast.Constant): out[names[0]]=d.value
    return out

def make_plan(args,kind):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.run_id):
        raise ValueError('--run-id must be an ASCII identifier, not a path')
    if args.execute and args.run_id=='preview': raise ValueError('--execute requires a new explicit --run-id')
    if args.limit is not None and args.limit<=0: raise ValueError('--limit must be positive')
    if args.max_tokens is not None and args.max_tokens<=0: raise ValueError('--max-tokens must be positive')
    m=MODELS[args.model]
    if kind=='api' and args.thinking_level and m['stem']!='gemini':
        raise ValueError('--thinking-level is only supported by the original Gemini runner')
    manifest=json.loads((HERE/'provenance'/'original_source_manifest.json').read_text(encoding='utf8'))
    root=(HERE/('pilot_results' if args.limit else 'results')/args.run_id/args.model).resolve()
    expected_base=(HERE/('pilot_results' if args.limit else 'results')).resolve()
    if root!=expected_base and expected_base not in root.parents: raise ValueError('Output escaped the dedicated results directory')
    if args.execute and root.exists(): raise ValueError('Output exists; choose a new run-id. No implicit resume/overwrite.')
    python=args.python or m['python']
    if args.python and ('/' in args.python or '\\' in args.python):
        python=str(Path(args.python).expanduser().resolve())
    plan=[]
    for condition in expand_task(args.task):
        relative=script_relative(args.model,condition)
        script=HERE/'vendor'/relative
        key=relative.as_posix()
        expected=manifest['sources'][key]['sha256']
        dependencies=source_dependencies(script);defaults={};source_hashes={}
        for dependency in dependencies:
            dk=dependency.relative_to(HERE/'vendor').as_posix()
            dh=manifest['sources'][dk]['sha256']
            if sha(dependency)!=dh: raise ValueError('Original source modified: '+dk)
            source_hashes[dk]=dh;defaults.update(literal_defaults(dependency))
        rel_input, use_image, count, _=CONDITIONS[condition]
        inp=(args.data_root/rel_input).resolve(); out=root/(condition+'.jsonl')
        cmd=[python,str(script),'--input',str(inp),'--output',str(out),'--use-image',str(use_image)]
        if m['api']: cmd += ['--model',args.model]
        else: cmd += ['--model-path',str(args.model_path.expanduser().resolve()) if args.model_path else m['model_path']]
        if args.limit: cmd += ['--limit',str(args.limit)]
        if args.max_tokens: cmd += ['--max-output-tokens' if m['api'] else '--max-new-tokens',str(args.max_tokens)]
        if args.temperature is not None: cmd += ['--temperature',str(args.temperature)]
        if args.retry_invalid is not None: cmd += ['--retry-invalid',str(args.retry_invalid)]
        if kind=='api' and args.thinking_level: cmd += ['--thinking-level',args.thinking_level]
        entry=dict(condition=condition, input=str(inp), input_expected_rows=count, output=str(out),
                   source_sha256=expected, command=cmd,
                   source_dependency_hashes=source_hashes,source_defaults=defaults, input_status='not_checked')
        if args.validate_inputs or args.execute:
            if not inp.is_file(): raise FileNotFoundError(inp)
            rows=json.loads(inp.read_text(encoding='utf8')); h=sha(inp)
            record=manifest['data_metadata'][rel_input]
            if not isinstance(rows,list) or len(rows)!=count: raise ValueError('Wrong input row count: '+rel_input)
            if len({str(r['num']) for r in rows})!=count: raise ValueError('Duplicate input IDs: '+rel_input)
            if h!=record['sha256']: raise ValueError('Input differs from audited manuscript-era snapshot: '+rel_input)
            entry.update(input_status='hash_and_rows_match',input_sha256=h)
        plan.append(entry)
    return root,plan

def validate_api_environment(stem):
    prefix='OPENAI' if stem=='gpt' else 'GEMINI'
    if not os.environ.get(prefix+'_API_KEY'): raise ValueError('Missing '+prefix+'_API_KEY')
    url=os.environ.get(prefix+'_BASE_URL')
    if url:
        u=urlparse(url)
        if u.scheme!='https' or not u.netloc or u.username or u.password or u.query or u.fragment:
            raise ValueError('API endpoint must be HTTPS without credentials/query/fragment')

def main(kind):
    args=parser(kind).parse_args()
    try:
        root,plan=make_plan(args,kind)
        payload=dict(schema_version=1,mode='execute' if args.execute else 'dry-run',model=args.model,task=args.task,
                     data_root=str(args.data_root.expanduser().resolve()),
                     initial_requests=sum(min(e['input_expected_rows'],args.limit) if args.limit else e['input_expected_rows'] for e in plan),
                     score_package='../../scoring',plan=plan)
        print(json.dumps(payload,ensure_ascii=False,indent=2))
        if not args.execute: return
        m=MODELS[args.model]
        executable=plan[0]['command'][0]
        if not (Path(executable).is_file() or shutil.which(executable)): raise ValueError('Python executable unavailable: '+executable)
        if kind=='api': validate_api_environment(m['stem'])
        else:
            weight=Path(plan[0]['command'][plan[0]['command'].index('--model-path')+1])
            if not weight.is_dir(): raise ValueError('Model directory missing: '+str(weight))
        # The entire job is checked before any result directory is created.
        root.mkdir(parents=True,exist_ok=False)
        (root/'run_manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        for entry in plan:
            subprocess.run(entry['command'],check=True,cwd=HERE)
    except (ValueError,FileNotFoundError) as e:
        raise SystemExit(str(e))
