"""All executable sources live in code/; outputs live in sibling results/."""
from pathlib import Path

CODE_DIR=Path(__file__).resolve().parent
EXPERIMENT_DIR=CODE_DIR.parent
RESULTS_DIR=EXPERIMENT_DIR/'results'
TASK_GROUP={
    'baseline':'baseline','baseline900':'baseline',
    'V2':'V2','V3_with':'V3','V3_without':'V3',
    'T2_present':'T2','T2_absent':'T2','T3':'T3',
}
GROUP_NAMES=('baseline','V2','V3','T2','T3')

def result_path(root,model,task):
    return Path(root)/TASK_GROUP[task]/model/(task+'.jsonl')

def locate_result(root,model,task):
    """Read the new layout, while accepting an unmigrated legacy result root."""
    new=result_path(root,model,task)
    old=Path(root)/model/(task+'.jsonl')
    if new.is_file() and old.is_file():
        raise ValueError(f'Duplicate old/new result locations: {new} and {old}')
    return new if new.is_file() else old

