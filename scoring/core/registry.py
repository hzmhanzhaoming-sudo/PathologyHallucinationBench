"""Manuscript input-file registry for scoring saved six-task runs."""
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

TARGET_FILES = {
    'V2': 'Benchmark_T4_data/pathology_vqa_benchmark_900_wrong_image_description_manifest.json',
    'T3': 'Benchmark_T7_data/pathology_vqa_benchmark_900_wrong_answer_hint_manifest.json',
}
