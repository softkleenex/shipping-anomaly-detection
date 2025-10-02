"""
Dacon 리더보드 점수 확인
"""

import requests
import json

TOKEN = 'debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466'
COMPETITION_ID = '236590'

def check_my_submissions():
    """내 제출 기록 확인"""

    # 가능한 API 엔드포인트들
    endpoints = [
        f"https://dacon.io/api/v1/competitions/{COMPETITION_ID}/submissions",
        f"https://dacon.io/api/competitions/{COMPETITION_ID}/submissions",
        f"https://dacon.io/api/v1/my-submissions/{COMPETITION_ID}",
        f"https://dacon.io/api/my/submissions/{COMPETITION_ID}",
    ]

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    print("="*70)
    print("Checking Dacon Submissions & Scores")
    print("="*70)

    for endpoint in endpoints:
        print(f"\nTrying: {endpoint}")

        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print("\n[SUCCESS] Found submission data!")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    return data
                except:
                    print(f"Response (text): {response.text[:500]}")
                    return response.text
            elif response.status_code == 401:
                print("[ERROR] Unauthorized - Check your token")
            elif response.status_code == 404:
                print("[SKIP] Not found")
            else:
                print(f"[INFO] Response: {response.text[:200]}")

        except Exception as e:
            print(f"[ERROR] {e}")

    print("\n" + "="*70)
    print("Could not access submission data via API")
    print("="*70)
    print("\nPlease check manually:")
    print(f"https://dacon.io/competitions/official/{COMPETITION_ID}/leaderboard")
    print(f"https://dacon.io/competitions/official/{COMPETITION_ID}/mysubmit")

    return None

def check_leaderboard():
    """리더보드 확인"""

    url = f"https://dacon.io/api/v1/competitions/{COMPETITION_ID}/leaderboard"

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    print("\n" + "="*70)
    print("Checking Public Leaderboard")
    print("="*70)

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"\nTop 10 Scores:")

            if isinstance(data, list):
                for i, entry in enumerate(data[:10], 1):
                    score = entry.get('score', 'N/A')
                    team = entry.get('team_name', 'N/A')
                    print(f"{i}. {team}: {score}")
            elif isinstance(data, dict) and 'data' in data:
                for i, entry in enumerate(data['data'][:10], 1):
                    score = entry.get('score', 'N/A')
                    team = entry.get('team_name', 'N/A')
                    print(f"{i}. {team}: {score}")
            else:
                print(json.dumps(data, indent=2, ensure_ascii=False))

        else:
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:300]}")

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == '__main__':
    # 내 제출 기록 확인
    my_submissions = check_my_submissions()

    # 리더보드 확인
    check_leaderboard()

    print("\n" + "="*70)
    print("Please visit the website to see your exact score:")
    print(f"https://dacon.io/competitions/official/{COMPETITION_ID}/mysubmit")
    print("="*70)
