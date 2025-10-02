"""
Dacon API를 사용한 최종 제출
"""

from dacon_submit_api import dacon_submit_api

# 설정
FILE_PATH = '../submissions/final_ensemble_gpu_20250928_154041.csv'
TOKEN = 'debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466'
COMPETITION_ID = '236590'
TEAM_NAME = 'solo'
MEMO = 'GPU ensemble (LightGBM+XGBoost+CatBoost) with feature engineering - Macro F1 target > 0.9'

print("="*70)
print("Dacon Submission - Final")
print("="*70)
print(f"File: {FILE_PATH}")
print(f"Competition ID: {COMPETITION_ID}")
print(f"Team: {TEAM_NAME}")
print(f"Memo: {MEMO}")
print("="*70)
print("\nSubmitting to Dacon...")

try:
    result = dacon_submit_api.post_submission_file(
        FILE_PATH,
        TOKEN,
        COMPETITION_ID,
        TEAM_NAME,
        MEMO
    )

    print("\n" + "="*70)
    print("Submission Result:")
    print("="*70)
    print(result)
    print("="*70)

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
