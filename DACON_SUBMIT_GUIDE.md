# Dacon 제출 API 사용 가이드

## 📦 1. API 설치

### 방법 1: whl 파일 다운로드 및 설치 (권장)

```bash
# whl 파일 다운로드
curl -L "https://bit.ly/3gMPScE" -o dacon_submit_api.whl

# 압축 해제 (zip 안에 실제 whl이 들어있음)
unzip dacon_submit_api.whl

# 설치
pip install dacon_submit_api-0.0.4-py3-none-any.whl
```

### 방법 2: 직접 다운로드
1. https://bit.ly/3gMPScE 에서 파일 다운로드
2. 압축 해제
3. `pip install dacon_submit_api-0.0.4-py3-none-any.whl`

---

## 🔑 2. Dacon Token 발급

1. https://dacon.io 로그인
2. 마이페이지 → API Token 발급
3. Token은 1회만 표시되므로 **반드시 저장**
4. Token 분실 시 파기 후 재발급 가능

**예시 Token:**
```
debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466
```

---

## 📤 3. 제출 코드 작성

### 기본 사용법

```python
from dacon_submit_api import dacon_submit_api

result = dacon_submit_api.post_submission_file(
    '파일경로',           # 제출할 CSV 파일 경로
    '개인 Token',         # Dacon API Token
    '대회ID',             # 대회 ID (URL에서 확인)
    '팀이름',             # 팀 이름 (선택사항, 빈 문자열 가능)
    'submission 메모'     # 제출 메모 (선택사항)
)

print(result)
```

### 실전 예제

```python
from dacon_submit_api import dacon_submit_api

# 설정
FILE_PATH = './submission.csv'
TOKEN = 'debdc113bd5a86728af5de749719bc607d13eb0efe13ccbeae601f425bd73466'
COMPETITION_ID = '236590'  # 대회 URL에서 확인
TEAM_NAME = 'MyTeam'
MEMO = 'LightGBM baseline model'

# 제출
result = dacon_submit_api.post_submission_file(
    FILE_PATH,
    TOKEN,
    COMPETITION_ID,
    TEAM_NAME,
    MEMO
)

# 결과 확인
print(result)
# 성공 시: {'isSubmitted': True, 'detail': 'Success'}
```

---

## 🎯 4. 대회 ID 확인 방법

대회 URL에서 숫자 부분이 대회 ID입니다:

```
https://dacon.io/competitions/official/236590/overview/description
                                      ^^^^^^
                                      대회 ID
```

---

## 📋 5. 완전한 제출 스크립트

```python
"""
Dacon 제출 스크립트
"""

from dacon_submit_api import dacon_submit_api
import os

# ============= 설정 (여기만 수정하세요) =============
FILE_PATH = './submissions/my_submission.csv'
TOKEN = 'YOUR_TOKEN_HERE'  # 본인의 Token으로 변경
COMPETITION_ID = '236590'   # 대회 ID
TEAM_NAME = ''              # 팀 이름 (선택)
MEMO = 'Initial submission' # 메모 (선택)
# ===================================================

def main():
    # 파일 존재 확인
    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] File not found: {FILE_PATH}")
        return

    file_size = os.path.getsize(FILE_PATH) / 1024

    print("="*70)
    print("Dacon Submission")
    print("="*70)
    print(f"File: {FILE_PATH}")
    print(f"Size: {file_size:.2f} KB")
    print(f"Competition ID: {COMPETITION_ID}")
    print(f"Team: {TEAM_NAME if TEAM_NAME else 'N/A'}")
    print(f"Memo: {MEMO if MEMO else 'N/A'}")
    print("="*70)
    print("\nSubmitting...")

    try:
        result = dacon_submit_api.post_submission_file(
            FILE_PATH,
            TOKEN,
            COMPETITION_ID,
            TEAM_NAME,
            MEMO
        )

        print("\n" + "="*70)
        print("Result:")
        print("="*70)
        print(result)

        if isinstance(result, dict) and result.get('isSubmitted'):
            print("\n✅ Submission successful!")
        else:
            print("\n❌ Submission may have failed")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
```

---

## 🔧 6. 트러블슈팅

### 문제: whl 파일 설치 오류
```
ERROR: dacon_submit_api.whl is not a valid wheel filename.
```

**해결책:**
```bash
# 먼저 압축 해제
unzip dacon_submit_api.whl

# 그 다음 실제 whl 파일 설치
pip install dacon_submit_api-0.0.4-py3-none-any.whl
```

### 문제: Token 오류
```
{'isSubmitted': False, 'detail': 'Invalid token'}
```

**해결책:**
1. Token이 정확한지 확인
2. Dacon 웹사이트에서 Token 재발급
3. Token 앞뒤 공백 제거

### 문제: 파일을 찾을 수 없음
```
FileNotFoundError: [Errno 2] No such file or directory
```

**해결책:**
```python
import os

# 절대 경로 사용
FILE_PATH = os.path.abspath('./submission.csv')

# 또는 현재 디렉토리 확인
print(f"Current directory: {os.getcwd()}")
print(f"File exists: {os.path.exists(FILE_PATH)}")
```

---

## 📊 7. 제출 결과 확인

### 웹에서 확인
```
내 제출 내역: https://dacon.io/competitions/official/{COMPETITION_ID}/mysubmit
리더보드: https://dacon.io/competitions/official/{COMPETITION_ID}/leaderboard
```

### API 응답 해석
```python
# 성공
{'isSubmitted': True, 'detail': 'Success'}

# 실패
{'isSubmitted': False, 'detail': 'Error message'}
```

---

## 💡 8. 유용한 팁

### 여러 파일 연속 제출
```python
import glob

submission_files = glob.glob('./submissions/*.csv')

for file_path in submission_files:
    memo = f"Model: {os.path.basename(file_path)}"

    result = dacon_submit_api.post_submission_file(
        file_path,
        TOKEN,
        COMPETITION_ID,
        TEAM_NAME,
        memo
    )

    print(f"{file_path}: {result}")
```

### 타임스탬프 포함 메모
```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
MEMO = f"Submission at {timestamp} - LightGBM v2"
```

### 자동 리트라이
```python
import time

def submit_with_retry(file_path, token, comp_id, team, memo, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = dacon_submit_api.post_submission_file(
                file_path, token, comp_id, team, memo
            )

            if result.get('isSubmitted'):
                return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # 5초 대기 후 재시도

    return None
```

---

## 📚 9. 참고 링크

- **Dacon 공지**: https://dacon.io/forum/403557
- **API 다운로드**: https://bit.ly/3gMPScE
- **대회 페이지**: https://dacon.io/competitions/official/{COMPETITION_ID}

---

## ⚠️ 10. 주의사항

1. **일일 제출 제한**: 대부분 대회는 하루 5회 제한
2. **Token 보안**: Token을 GitHub 등에 올리지 마세요
3. **파일 형식**: CSV 파일만 가능한 경우가 많음
4. **인코딩**: UTF-8 인코딩 사용 권장
5. **파일 크기**: 대회별 제한 확인 필요

---

## 🚀 Quick Start (복사해서 사용)

```bash
# 1. 설치
curl -L "https://bit.ly/3gMPScE" -o dacon_submit_api.whl
unzip dacon_submit_api.whl
pip install dacon_submit_api-0.0.4-py3-none-any.whl

# 2. Python 스크립트 작성
cat > submit.py << 'EOF'
from dacon_submit_api import dacon_submit_api

result = dacon_submit_api.post_submission_file(
    './submission.csv',
    'YOUR_TOKEN',
    '236590',
    '',
    'My first submission'
)
print(result)
EOF

# 3. 실행
python submit.py
```

---

**작성일**: 2025-09-28
**API 버전**: 0.0.4
**테스트 완료**: ✅
