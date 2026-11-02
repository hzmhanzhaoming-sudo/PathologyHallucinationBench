"""Manuscript task names; never confuse them with legacy T1--T7 folders."""
from pathlib import Path
import os

ROOT = Path(os.environ.get('BENCHMARK_DATA_ROOT', '/data5/hzm/Benchmark'))
TASK_FILES = {
    'baseline': 'Benchmark_T1_data/pathology_vqa_benchmark_1200_with_image.json',
    'baseline900': 'Benchmark_T1_data/pathology_vqa_benchmark_1200_with_image.json',
    'V2': 'Benchmark_T4_data/pathology_vqa_benchmark_900_wrong_image_description_with_image.json',
    'V3_with': 'Benchmark_T6_data/pathology_vqa_benchmark_900_with_image.json',
    'V3_without': 'Benchmark_T6_data/pathology_vqa_benchmark_900_empty_image.json',
    'T2_present': 'Benchmark_T5_data/pathology_vqa_benchmark_900_T5_V1_answer_present_with_image.json',
    'T2_absent': 'Benchmark_T5_data/pathology_vqa_benchmark_900_T5_V2_answer_absent_with_image.json',
    'T3': 'Benchmark_T7_data/pathology_vqa_benchmark_900_wrong_answer_hint_with_image.json',
}
ALL_TASKS = ['baseline', 'V2', 'V3_with', 'V3_without', 'T2_present', 'T2_absent', 'T3']
GROUPS = {'V3': ['V3_with', 'V3_without'], 'T2': ['T2_present', 'T2_absent'], 'all': ALL_TASKS}
TARGET_FILES = {
    'V2': 'Benchmark_T4_data/pathology_vqa_benchmark_900_wrong_image_description_manifest.json',
    'T3': 'Benchmark_T7_data/pathology_vqa_benchmark_900_wrong_answer_hint_manifest.json',
}
MODELS = {}
def register(key, backend, weight, env, repo, legacy, family):
    MODELS[key] = dict(backend=backend, model_path='/data5/hzm/Model/'+weight,
                       python='/data5/hzm/miniconda3/envs/'+env+'/bin/python',
                       repo_root='/data5/hzm/'+repo, legacy='/data5/hzm/'+legacy, family=family)
for size in (2, 4, 9):
    register(f'qwen35-{size}b', 'qwen', f'Qwen3.5-{size}B', 'qwen35', 'Test_for_Generalmodel',
             'Test_for_Generalmodel/eval_T2_qwen35.py', 'Open-source general')
register('glm46v', 'glm', 'GLM-4.6V-Flash', 'qwen35', 'Test_for_Generalmodel',
         'Test_for_Generalmodel/eval_T2_glm46v.py', 'Open-source general')
for size in (3, 7):
    register(f'patho-r1-{size}b', 'patho', f'Patho-R1-{size}B', 'patho-r1', 'Patho-R1',
             'Patho-R1/eval_T2_patho_r1.py', 'Pathology-specialized')
register('quilt-llava', 'quilt', 'Quilt-Llava', 'qllava', 'quilt-llava',
         'quilt-llava/eval_T2_quilt_llava.py', 'Pathology-specialized')
register('pathgen-llava', 'pathgen', 'PathGen-LLaVA', 'pathgen-llava', 'LLaVA',
         'LLaVA/eval_T2_pathgen_llava.py', 'Pathology-specialized')
register('huatuogpt-vision', 'huatuo', 'HuatuoGPT-Vision-7B', 'healthgpt-pro', 'HuatuoGPT-Vision',
         'HuatuoGPT-Vision/eval_T2_huatuogpt_vision.py', 'General medical')
for size in (4, 8):
    register(f'healthgpt-pro-{size}b', 'health', f'HealthGPT-Pro-{size}B', 'healthgpt-pro',
             'HealthGPT/HealthGPT-Pro', 'HealthGPT/HealthGPT-Pro/eval_T2_healthgpt_pro.py', 'General medical')
register('unimedvl', 'unimed', 'UniMedVL', 'unimedvl', 'UniMedVL',
         'UniMedVL/eval_T2_unimedvl.py', 'General medical')
for key in ('gpt-5.6-sol', 'gpt-5.6-luna', 'gemini-3.5-flash-lite', 'gemini-3.6-flash'):
    MODELS[key] = dict(backend='openai' if key.startswith('gpt') else 'gemini',
                       family='Closed-source general', python='python3')


# Bundled inference loaders, unchanged from the currently installed scripts.
_ADAPTER_ROOT = Path(__file__).resolve().parents[1] / 'model_adapters'
for _key, _entry in MODELS.items():
    if _entry['backend'] not in ('openai', 'gemini'):
        _entry['legacy'] = str(_ADAPTER_ROOT / (_entry['backend'] + '.py'))
