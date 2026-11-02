"""Shared five-task scorer for NEW runs; read-only inputs, explicit output folder."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'03_prompting/core'))
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))

def main():
    from registry import ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args()
    output=a.output.expanduser().resolve()
    protected=[ROOT/name/'results' for name in ('Benchmark_Zero-shot CoT','Benchmark_Few-shot direct','Benchmark_Few-shot CoT')]
    protected += [p for p in ROOT.glob('Benchmark_T*_results') if p.is_dir()]
    if output.exists():
        raise SystemExit('Choose a NEW output directory; existing statistics are never overwritten.')
    if any(folder.resolve()==output or folder.resolve() in output.parents for folder in protected):
        raise SystemExit('Historical result directories are read-only; choose an output folder under CODE/reports.')
    from score_results import main as score_main
    original_argv=sys.argv
    sys.argv=[original_argv[0],'--results',str(a.results.expanduser().resolve()),
              '--output',str(output),'--data-root',str(a.data_root.expanduser().resolve())]
    if a.allow_partial:sys.argv.append('--allow-partial')
    try:
        score_main()
    finally:
        sys.argv=original_argv

if __name__=='__main__':main()
