"""Paper task names, kept separate from historical server T1--T7 names."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = Path('/data5/hzm')
DATA_ROOT = SERVER/'Benchmark'
MODELS = {}

def add(key, stem, folder, weight, env):
    MODELS[key] = dict(stem=stem, folder=folder, model_path=str(SERVER/'Model'/weight),
                       python=str(SERVER/'miniconda3'/'envs'/env/'bin'/'python'), api=False)

for n in (2,4,9): add(f'qwen35-{n}b','qwen35','Test_for_Generalmodel',f'Qwen3.5-{n}B','qwen35')
add('glm46v','glm46v','Test_for_Generalmodel','GLM-4.6V-Flash','qwen35')
for n in (3,7): add(f'patho-r1-{n}b','patho_r1','Patho-R1',f'Patho-R1-{n}B','patho-r1')
add('quilt-llava','quilt_llava','quilt-llava','Quilt-Llava','qllava')
add('pathgen-llava','pathgen_llava','LLaVA','PathGen-LLaVA','pathgen-llava')
add('huatuogpt-vision','huatuogpt_vision','HuatuoGPT-Vision','HuatuoGPT-Vision-7B','healthgpt-pro')
for n in (4,8): add(f'healthgpt-pro-{n}b','healthgpt_pro','HealthGPT/HealthGPT-Pro',f'HealthGPT-Pro-{n}B','healthgpt-pro')
add('unimedvl','unimedvl','UniMedVL','UniMedVL','unimedvl')
for key in ('gpt-5.6-sol','gpt-5.6-luna','gemini-3.5-flash-lite','gemini-3.6-flash'):
    MODELS[key] = dict(stem='gpt' if key.startswith('gpt') else 'gemini',
                       folder='Test_for_Generalmodel', python='python3', api=True)

CONDITIONS = {
 'Base': ('Benchmark_T1_data/pathology_vqa_benchmark_1200_with_image.json',1,1200,'base'),
 'V1_without': ('Benchmark_T1_data/pathology_vqa_benchmark_1200_empty_image.json',0,1200,'base'),
 'V2': ('Benchmark_T4_data/pathology_vqa_benchmark_900_wrong_image_description_with_image.json',1,900,'T4'),
 'V3_with': ('Benchmark_T6_data/pathology_vqa_benchmark_900_with_image.json',1,900,'T2'),
 'V3_without': ('Benchmark_T6_data/pathology_vqa_benchmark_900_empty_image.json',0,900,'T2'),
 'T1_4choice': ('Benchmark_T3_data/4-choice-test/pathology_vqa_benchmark_1000_unknown1_2_with_image.json',1,1000,'T3_4choice'),
 'T1_6choice': ('Benchmark_T3_data/6-choice-test/pathology_vqa_benchmark_400_unknown1_2_with_image.json',1,400,'T3_6choice'),
 'T1_7choice': ('Benchmark_T3_data/7-choice-test/pathology_vqa_benchmark_400_unknown1_2_with_image.json',1,400,'T3_7choice'),
 'T2_present': ('Benchmark_T5_data/pathology_vqa_benchmark_900_T5_V1_answer_present_with_image.json',1,900,'T2'),
 'T2_absent': ('Benchmark_T5_data/pathology_vqa_benchmark_900_T5_V2_answer_absent_with_image.json',1,900,'T2'),
 'T3': ('Benchmark_T7_data/pathology_vqa_benchmark_900_wrong_answer_hint_with_image.json',1,900,'T2'),
}
GROUPS = {'Base':['Base'],'V1':['V1_without'],'V2':['V2'],'V3':['V3_with','V3_without'],
          'T1':['T1_4choice','T1_6choice','T1_7choice'],'T2':['T2_present','T2_absent'],'T3':['T3']}
GROUPS['all'] = list(CONDITIONS)

def expand_task(task):
    # A single shared clean Base is needed for each task's paired analysis.
    return list(dict.fromkeys(['Base'] + GROUPS[task]))

def script_relative(model, condition):
    m=MODELS[model]; suffix=CONDITIONS[condition][3]
    if suffix == 'base':
        name='eval_'+m['stem']+('' if m['api'] else '_pathology_mcq')+'.py'
    else:
        name=f"eval_{suffix}_{m['stem']}.py"
    return Path(m['folder'])/name
