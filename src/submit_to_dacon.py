"""
Dacon API를 사용한 직접 제출
"""

import requests
import os

# 설정
TOKEN = "debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466"
COMPETITION_ID = "236590"  # URL에서 확인한 대회 ID
SUBMISSION_FILE = "../submissions/final_ensemble_gpu_20250928_154041.csv"
TEAM_NAME = "solo"  # 팀 이름 (개인 참가면 임의 설정)
MEMO = "GPU ensemble (LightGBM+XGBoost+CatBoost) with feature engineering"

def submit_to_dacon():
    """Dacon API로 제출"""

    # 파일 확인
    if not os.path.exists(SUBMISSION_FILE):
        print(f"[ERROR] File not found: {SUBMISSION_FILE}")
        return

    file_size = os.path.getsize(SUBMISSION_FILE) / 1024
    print("="*70)
    print("Dacon Submission")
    print("="*70)
    print(f"Competition ID: {COMPETITION_ID}")
    print(f"File: {SUBMISSION_FILE}")
    print(f"Size: {file_size:.2f} KB")
    print(f"Team: {TEAM_NAME}")
    print(f"Memo: {MEMO}")
    print()

    # API 엔드포인트 (추정)
    api_url = "https://dacon.io/api/v1/submission"

    # 헤더
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "User-Agent": "DaconSubmitAPI/0.0.4"
    }

    # 파일 준비
    files = {
        'file': (os.path.basename(SUBMISSION_FILE), open(SUBMISSION_FILE, 'rb'), 'text/csv')
    }

    # 데이터
    data = {
        'competition_id': COMPETITION_ID,
        'team_name': TEAM_NAME,
        'memo': MEMO
    }

    try:
        print("Uploading to Dacon...")
        response = requests.post(api_url, headers=headers, files=files, data=data)

        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200 or response.status_code == 201:
            print("\n[OK] Submission successful!")
            try:
                result = response.json()
                print(f"Result: {result}")
            except:
                pass
        else:
            print(f"\n[ERROR] Submission failed")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        files['file'][1].close()

if __name__ == '__main__':
    submit_to_dacon()
