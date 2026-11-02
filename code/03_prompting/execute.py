"""Internal runner after selecting the correct Python environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'02_scoring/core'))
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode not in ('api','local'): raise SystemExit('Expected api or local')
    from runner import main
    main(api=mode=='api')
