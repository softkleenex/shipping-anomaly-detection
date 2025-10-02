"""
긴급 제출 - 기존 최고 모델 재제출
"""

from dacon_submit_api import dacon_submit_api

TOKEN = 'debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466'
COMPETITION_ID = '236590'
FILE_PATH = '../submissions/final_ensemble_gpu_20250928_154041.csv'

print("="*70)
print("EMERGENCY SUBMISSION")
print("="*70)
print(f"Time: 9:05 AM")
print(f"Deadline: 10:00 AM")
print(f"File: {FILE_PATH}")
print("="*70)

result = dacon_submit_api.post_submission_file(
    FILE_PATH,
    TOKEN,
    COMPETITION_ID,
    'emergency_team',
    'Final submission - GPU ensemble (LightGBM+XGBoost+CatBoost) - Emergency resubmit before deadline'
)

print("\nResult:", result)
