"""Lightweight dispatcher to installed model environments; no implicit inference."""
from pathlib import Path
import os
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'02_scoring/core'))

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))

def main(api=False, historical=False):
    from runner import parser
    from registry import MODELS
    # Parse the exact underlying options, without loading any model or API SDK.
    args=parser(api).parse_args()
    extra=list(sys.argv[1:])
    def supplied(flag):
        return any(value==flag or value.startswith(flag+'=') for value in extra)
    if not historical and args.prompt_profile!='standard':
        raise SystemExit('New runs use one shared standard prompt only. Use run_historical_api.py for historical API profiles.')
    if historical:
        if not api: raise SystemExit('Historical dispatcher is API-only.')
        if args.model not in ('gpt-5.6-luna','gemini-3.5-flash-lite','gemini-3.6-flash'):
            raise SystemExit('No verified historical prompt-run configuration for this model.')
        if args.strategy=='zero_shot_direct':
            raise SystemExit('Historical Original uses separate legacy inference, not this prompt-experiment runner.')
        expected='api_visible_explanation' if args.model=='gpt-5.6-luna' and args.strategy=='zero_shot_cot' else 'standard'
        if supplied('--prompt-profile') and args.prompt_profile!=expected:
            raise SystemExit('Profile conflicts with verified historical configuration.')
        if not supplied('--prompt-profile'): extra+=['--prompt-profile',expected]
        defaults={'--max-new-tokens':'512','--force-answer-max-tokens':'32','--force-answer-on-limit':'1'}
        if args.model=='gpt-5.6-luna': defaults['--reasoning-effort']='none'
        if args.model=='gemini-3.6-flash': defaults['--thinking-level']='minimal'
        for flag,value in defaults.items():
            if supplied(flag) and str(getattr(args,flag[2:].replace('-','_')))!=value:
                raise SystemExit('Historical configuration mismatch: '+flag)
            if not supplied(flag): extra+=[flag,value]
        if args.temperature is not None or args.thinking_budget is not None:
            raise SystemExit('Historical API configuration omitted temperature/thinking-budget.')
        if not supplied('--output-root'):
            extra+=['--output-root',str(ROOT/'historical_reproduction_results')]
    executable=sys.executable if api or args.dry_run else MODELS[args.model]['python']
    if not Path(executable).is_file():
        raise SystemExit('Installed model Python not found: '+str(executable))
    command=[executable,'-B',str(ROOT/'execute.py'),'api' if api else 'local']+extra
    print(' '.join(shlex.quote(s) for s in command),flush=True)
    raise SystemExit(subprocess.call(command,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}))
