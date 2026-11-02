"""Read-only regression orchestration; no inference, API calls or source writes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
PROMPT=ROOT/'03_prompting'
SCORE=ROOT/'02_scoring'

def verify_manifest():
    manifest=json.loads((ROOT/'package_manifest.json').read_text(encoding='utf8'))
    bad=[]
    for name,digest in manifest.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            bad.append(name)
    if bad:raise ValueError('Package changed: '+repr(bad))
    return len(manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='New report directory; never overwritten')
    p.add_argument('--data-root',type=Path,default=Path('/data5/hzm/Benchmark'))
    p.add_argument('--offline-only',action='store_true',help='Synthetic tests only; does NOT verify manuscript numbers')
    p.add_argument('--snapshot',type=Path,help='External historical comparison_12models.json; required for server regression')
    a=p.parse_args();out=a.output.expanduser().resolve();data=a.data_root.resolve()
    protected=[data/'Draw_picture',data/'Conclusion']+list(data.glob('Benchmark_T*_results'))
    protected += [data/n/'results' for n in ('Benchmark_Zero-shot CoT','Benchmark_Few-shot direct','Benchmark_Few-shot CoT')]
    if out.exists():raise SystemExit('Choose a NEW output directory.')
    if any(q==out or q in out.parents for q in protected):raise SystemExit('Historical output directories are read-only.')
    if not a.offline_only and (a.snapshot is None or not a.snapshot.is_file()):
        raise SystemExit('Full server regression requires --snapshot and the original server data/results; these are not distributed in Git.')
    count=verify_manifest()
    out.mkdir(parents=True,exist_ok=False)
    temp=out/'test_temp';temp.mkdir()
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1',
         'BENCHMARK_DATA_ROOT':str(data),
         'PYTHONPATH':os.pathsep.join([str(PROMPT/'core'),str(SCORE/'core')]),
         'TMP':str(temp),'TEMP':str(temp),'TMPDIR':str(temp)}
    report={'finished_utc':None,'package_files':count,'offline_only':a.offline_only,
            'model_calls':0,'api_calls':0,'source_results_modified':False,'commands':[],'passed':False}
    commands=[('prompt_unit_tests',[sys.executable,'-B','-m','unittest','discover','-s','core','-p','test_*.py'],PROMPT),
              ('shared_contract',[sys.executable,'-B','tests/test_shared_contract.py'],PROMPT),
              ('package_boundary',[sys.executable,'-B','tests/test_package_entrypoints.py'],PROMPT)]
    for filename in sorted((SCORE/'tests').glob('test_*.py')):
        commands.append((filename.stem,[sys.executable,'-B',str(filename)],SCORE))
    first_tests=ROOT/'01_stress_tests/test_offline.py'
    if first_tests.is_file():commands.append(('six_tests_contract',[sys.executable,'-B',str(first_tests)],first_tests.parent))
    if not a.offline_only:
        commands += [
            ('published_prediction_regression',[sys.executable,'-B',str(SCORE/'verify_published.py'),
                '--core',str(SCORE/'core'),'--snapshot',str(a.snapshot),
                '--data-root',str(data),'--output',str(out/'published_regression.json')],SCORE),
            ('original_six_scores',[sys.executable,'-B',str(SCORE/'score_original_server.py'),
                '--data-root',str(data),'--output',str(out/'original_six_scores')],SCORE),
            ]
        raw=SCORE/'verify_original_raw.py'
        if raw.is_file():
            commands.append(('original_raw_predictions',[sys.executable,'-B',str(raw),'--data-root',str(data),
                             '--output',str(out/'original_raw.json')],SCORE))
        axes=SCORE/'verify_axes.py'
        if axes.is_file():
            commands.append(('three_axes',[sys.executable,'-B',str(axes),'--data-root',str(data),
                             '--output',str(out/'three_axes.json')],SCORE))
        integration=SCORE/'verify_new_layout.py'
        if integration.is_file():
            commands.append(('new_layout_integration',[sys.executable,'-B',str(integration),'--data-root',str(data),
                             '--code-root',str(ROOT),'--raw-audit',str(out/'original_raw.json'),
                             '--output',str(out/'new_layout_fixture')],SCORE))
    try:
        for label,command,cwd in commands:
            run=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8')
            (out/(label+'.log')).write_text(run.stdout+'\n'+run.stderr,encoding='utf8')
            report['commands'].append({'name':label,'returncode':run.returncode})
            print(json.dumps({'name':label,'returncode':run.returncode}),flush=True)
            if run.returncode:raise RuntimeError(label+' failed; see log')
        verify_manifest()
        report['passed']=True
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
