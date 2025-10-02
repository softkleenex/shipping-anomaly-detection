"""
간단한 파일 공유 서비스 업로드 (CLI 전용)
- file.io: 익명 임시 파일 공유 (1회 다운로드)
- transfer.sh: 간단한 파일 공유 (14일 보관)
"""

import subprocess
import os

def upload_to_fileio(file_path):
    """file.io에 업로드 (1회 다운로드 후 삭제)"""
    print(f"[file.io] Uploading: {file_path}")

    result = subprocess.run([
        'curl', '-F', f'file=@{file_path}',
        'https://file.io'
    ], capture_output=True, text=True)

    print(result.stdout)
    return result.stdout

def upload_to_transfer_sh(file_path):
    """transfer.sh에 업로드 (14일 보관)"""
    filename = os.path.basename(file_path)
    print(f"[transfer.sh] Uploading: {file_path}")

    result = subprocess.run([
        'curl', '--upload-file', file_path,
        f'https://transfer.sh/{filename}'
    ], capture_output=True, text=True)

    url = result.stdout.strip()
    print(f"\nDownload URL: {url}")
    print(f"Valid for: 14 days")
    return url

def upload_to_0x0_st(file_path):
    """0x0.st에 업로드 (365일 보관)"""
    print(f"[0x0.st] Uploading: {file_path}")

    result = subprocess.run([
        'curl', '-F', f'file=@{file_path}',
        'https://0x0.st'
    ], capture_output=True, text=True)

    url = result.stdout.strip()
    print(f"\nDownload URL: {url}")
    print(f"Valid for: 365 days")
    return url

if __name__ == '__main__':
    submission_file = '../submissions/final_ensemble_gpu_20250928_154041.csv'

    print("="*70)
    print("파일 업로드 시작")
    print("="*70)
    print(f"파일: {submission_file}")
    print(f"크기: {os.path.getsize(submission_file) / 1024:.2f} KB")
    print()

    try:
        # 여러 서비스에 백업 업로드
        print("\n[1/3] transfer.sh 업로드 중...")
        url1 = upload_to_transfer_sh(submission_file)

        print("\n[2/3] 0x0.st 업로드 중...")
        url2 = upload_to_0x0_st(submission_file)

        print("\n[3/3] file.io 업로드 중...")
        url3 = upload_to_fileio(submission_file)

        print("\n" + "="*70)
        print("[OK] Upload Complete!")
        print("="*70)
        print(f"\n다운로드 링크 (여러 개 백업):")
        print(f"1. {url1} (14일 보관)")
        print(f"2. {url2} (365일 보관)")
        print(f"3. file.io 링크는 위 출력 확인 (1회만 다운로드 가능)")

    except Exception as e:
        print(f"[ERROR] {e}")
