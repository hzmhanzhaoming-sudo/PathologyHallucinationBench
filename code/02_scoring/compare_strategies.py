"""Read-only strategy comparison with an explicit, fresh report destination.

The implementation in core/compare_strategies.py is internal; use this entry
instead of executing the core script, whose archived CLI defaults are retained.
This entry never runs models, calls providers or changes result predictions.
"""
from pathlib import Path
import argparse
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '03_prompting' / 'core'))
sys.path.insert(0, str(HERE / 'core'))


def _load_comparison():
    path = HERE / 'core' / 'compare_strategies.py'
    spec = importlib.util.spec_from_file_location('_benchmark_comparison_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    from registry import ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True,
                        help='At least two distinct result directories (read-only).')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True,
                        help='A new directory outside all result directories.')
    parser.add_argument('--allow-task-demo-comparison', action='store_true',
                        help='Only reviewed historical code hashes are allowed; does not waive decoding checks.')
    args = parser.parse_args()
    roots = [path.expanduser().resolve() for path in args.roots]
    output = args.output.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise SystemExit('Provide at least two distinct result roots.')
    if any(not root.is_dir() for root in roots):
        raise SystemExit('All comparison result roots must be existing directories.')
    if output.exists() or args.output.expanduser().is_symlink():
        raise SystemExit('Choose a NEW output directory; existing comparisons are never overwritten.')
    protected = list(roots)
    for root in {ROOT.expanduser().resolve(), data_root}:
        protected.extend(root / name / 'results' for name in (
            'Benchmark_Zero-shot CoT', 'Benchmark_Few-shot direct', 'Benchmark_Few-shot CoT'))
        # Protect known original roots even when not present on this machine.
        protected.extend(root / f'Benchmark_T{i}_results' for i in range(1, 8))
        protected.extend(root.glob('Benchmark_T*_results'))
    if any(folder.resolve() == output or folder.resolve() in output.parents for folder in protected):
        raise SystemExit('Result directories are read-only; choose a new report folder outside them.')
    implementation = _load_comparison()
    previous_argv = sys.argv
    sys.argv = [str(HERE / 'core' / 'compare_strategies.py'), '--roots',
                *map(str, roots), '--data-root', str(data_root), '--output', str(output)]
    if args.allow_task_demo_comparison:
        sys.argv.append('--allow-task-demo-comparison')
    try:
        implementation.main()
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    main()
