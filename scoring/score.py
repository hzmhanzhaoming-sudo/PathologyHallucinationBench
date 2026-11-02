"""Run the public six-task notebook; only write a NEW output directory."""
from pathlib import Path
import argparse
import hashlib
import json
import os

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=a.repo_root.resolve(); out=a.output.resolve()
    if not (root/'data').is_dir() or not (root/'result').is_dir():
        raise SystemExit('Original notebook requires the existing public-layout data/ and result/ folders.')
    if out.exists(): raise SystemExit('Choose a new output directory; existing outputs are never overwritten.')
    notebook=Path(__file__).resolve().parent/'score_six_hallucination_tasks.ipynb'
    cells=json.loads(notebook.read_text(encoding='utf8'))['cells']
    namespace={'__name__':'six_task_scorer','display':lambda *values: None}
    before=Path.cwd()
    out.mkdir(parents=True)
    try:
        os.chdir(root)
        first=True
        for index,cell in enumerate(cells):
            if cell['cell_type']!='code': continue
            exec(compile(''.join(cell['source']),str(notebook)+'#cell'+str(index),'exec'),namespace)
            if first:
                for variable,name in [('SCORE_OUTPUT','six_task_robustness_scores.csv'),
                                      ('EVENT_OUTPUT','six_task_event_counts.csv'),('AUDIT_OUTPUT','scoring_input_audit.csv')]:
                    namespace[variable]=out/name
                first=False
    finally:
        os.chdir(before)
    count=len(namespace.get('score_rows',[]))
    report={'notebook_sha256':hashlib.sha256(notebook.read_bytes()).hexdigest(),
            'source_repo':str(root),'models':count,'model_calls':0,'api_calls':0,
            'note':'Six-task scoring with three separate performance dimensions.'}
    (out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if not count: raise SystemExit('No original result bundles found; see output provenance.')
    print(json.dumps(report))

if __name__=='__main__': main()
