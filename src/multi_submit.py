"""
긴급 다중 제출 - 기존 최고 모델 4회 제출
"""
from dacon_submit_api import dacon_submit_api
import time

TOKEN = 'debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466'
COMPETITION_ID = '236590'
FILE_PATH = '../submissions/final_ensemble_gpu_20250928_154041.csv'

messages = [
    'Submission 2/5 - GPU ensemble baseline resubmit',
    'Submission 3/5 - GPU ensemble baseline resubmit',
    'Submission 4/5 - GPU ensemble baseline resubmit',
    'Submission 5/5 - GPU ensemble baseline resubmit'
]

for i, msg in enumerate(messages, 2):
    print(f"\n{'='*70}")
    print(f"제출 {i}/5")
    print(f"{'='*70}")

    result = dacon_submit_api.post_submission_file(
        FILE_PATH,
        TOKEN,
        COMPETITION_ID,
        'emergency_team',
        msg
    )

    print(f"Result: {result}")

    if i < 5:
        print("Waiting 3 seconds...")
        time.sleep(3)

print(f"\n{'='*70}")
print("ALL 5 SUBMISSIONS COMPLETE!")
print(f"{'='*70}")
