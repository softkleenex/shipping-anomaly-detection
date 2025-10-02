"""
Dacon Submission API 직접 구현
Dacon의 제출 API를 역공학하여 구현
"""

import requests
import os
import json

class DaconSubmitAPI:
    def __init__(self):
        self.base_url = "https://dacon.io"

    def post_submission_file(self, file_path, token, competition_id, team_name="", memo=""):
        """
        Dacon에 제출 파일 업로드

        Args:
            file_path: 제출할 파일 경로
            token: Dacon API 토큰
            competition_id: 대회 ID
            team_name: 팀 이름 (선택)
            memo: 제출 메모 (선택)

        Returns:
            API 응답 결과
        """

        # 파일 존재 확인
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}

        file_size = os.path.getsize(file_path) / 1024
        file_name = os.path.basename(file_path)

        print("="*70)
        print("Dacon Submission API")
        print("="*70)
        print(f"Competition ID: {competition_id}")
        print(f"File: {file_name}")
        print(f"Size: {file_size:.2f} KB")
        print(f"Team: {team_name if team_name else 'N/A'}")
        print(f"Memo: {memo if memo else 'N/A'}")
        print("="*70)

        # 가능한 엔드포인트들 시도
        endpoints = [
            f"/api/v1/competitions/{competition_id}/submissions",
            f"/api/competitions/{competition_id}/submit",
            f"/competitions/official/{competition_id}/submit",
        ]

        # 헤더 설정
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "DaconSubmitAPI/0.0.4",
            "Accept": "application/json",
        }

        # 파일 및 데이터 준비
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_name, f, 'text/csv')
            }

            data = {
                'team_name': team_name,
                'memo': memo
            }

            # 여러 엔드포인트 시도
            for endpoint in endpoints:
                url = self.base_url + endpoint
                print(f"\nTrying endpoint: {endpoint}")

                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        files={'file': (file_name, open(file_path, 'rb'), 'text/csv')},
                        data=data,
                        timeout=30
                    )

                    print(f"Status: {response.status_code}")

                    if response.status_code in [200, 201]:
                        print("[OK] Submission successful!")
                        try:
                            result = response.json()
                            print(f"Response: {json.dumps(result, indent=2)}")
                            return {"success": True, "data": result}
                        except:
                            print(f"Response: {response.text[:500]}")
                            return {"success": True, "data": response.text}
                    elif response.status_code == 404:
                        print("[SKIP] Endpoint not found, trying next...")
                        continue
                    else:
                        print(f"[ERROR] Status {response.status_code}")
                        print(f"Response: {response.text[:500]}")

                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Request failed: {e}")
                    continue

        # 모든 엔드포인트 실패
        print("\n" + "="*70)
        print("[FAILED] All endpoints failed")
        print("="*70)
        print("\nPossible solutions:")
        print("1. Check if token is valid")
        print("2. Download file and upload manually:")
        print(f"   - File location: {file_path}")
        print(f"   - Or download from: https://0x0.st/KMZy.csv")
        print(f"   - Upload at: https://dacon.io/competitions/official/{competition_id}/codeshare")

        return {"success": False, "error": "All API endpoints failed"}


def main():
    """메인 실행"""

    # 설정
    TOKEN = "debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466"
    COMPETITION_ID = "236590"
    FILE_PATH = "../submissions/final_ensemble_gpu_20250928_154041.csv"
    TEAM_NAME = ""  # 선택사항
    MEMO = "GPU ensemble: LightGBM+XGBoost+CatBoost with feature engineering"

    # API 인스턴스 생성
    api = DaconSubmitAPI()

    # 제출
    result = api.post_submission_file(
        file_path=FILE_PATH,
        token=TOKEN,
        competition_id=COMPETITION_ID,
        team_name=TEAM_NAME,
        memo=MEMO
    )

    return result


if __name__ == "__main__":
    result = main()
    print(f"\nFinal result: {result}")
