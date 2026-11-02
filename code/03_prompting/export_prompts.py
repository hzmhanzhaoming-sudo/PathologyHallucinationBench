"""Export the exact text/image structure without model calls or answer leakage."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'02_scoring/core'))
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from common import STRATEGIES, load_rows, file_hash, digest
from registry import ROOT, ALL_TASKS
from task_demos import load_task_demos
from prompt_profiles import make_messages_profile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategy',required=True,choices=STRATEGIES)
    p.add_argument('--prompt-profile',choices=('standard','api_visible_explanation'),default='standard')
    p.add_argument('--task',choices=ALL_TASKS+['all'],default='all')
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--item-id')
    a=p.parse_args()
    if a.output.exists(): raise SystemExit('Choose a new output directory; existing previews are not overwritten.')
    demos_path=Path(__file__).resolve().parent/'prompts/fewshot_data/fewshot_direct.json'
    plans=[]
    for task in ALL_TASKS if a.task=='all' else [a.task]:
        rows,source=load_rows(task,a.data_root)
        row=next((r for r in rows if str(r['num'])==a.item_id),None) if a.item_id else rows[0]
        if row is None: raise SystemExit('Item absent from '+task)
        demos,provenance=load_task_demos(a.strategy,demos_path,task)
        messages=make_messages_profile(row,source,a.strategy,demos,a.prompt_profile)
        plans.append((task,dict(task=task,strategy=a.strategy,profile=a.prompt_profile,num=str(row['num']),
                               messages=messages,message_sha256=digest(messages),demonstrations=provenance)))
    a.output.mkdir(parents=True)
    for task,data in plans:
        (a.output/(task+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'output':str(a.output),'tasks':len(plans),'model_calls':0,'api_calls':0}))

if __name__=='__main__': main()
