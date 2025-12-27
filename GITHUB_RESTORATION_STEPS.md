# GitHub 저장소 복원 단계별 가이드

## 현재 상태 확인
✅ `_pages/_chat_simulator.py` - 이미 존재
✅ `_pages/_content.py` - 이미 존재
✅ 관련 파일들 대부분 존재

하지만 GitHub의 최신 버전과 동기화가 필요할 수 있습니다.

## 방법 1: Git으로 직접 가져오기 (권장)

### 단계 1: 현재 프로젝트 위치 확인
```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
```

### 단계 2: Git 저장소 초기화 (아직 안 했다면)
```bash
git init
git remote add origin https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits.git
```

### 단계 3: 원격 저장소에서 특정 파일만 가져오기
```bash
# 원격 저장소 정보 가져오기
git fetch origin

# 특정 파일만 가져오기 (sparse checkout 사용)
git config core.sparseCheckout true
echo "_pages/_chat_simulator.py" >> .git/info/sparse-checkout
echo "_pages/_chat_*.py" >> .git/info/sparse-checkout
echo "_pages/_content*.py" >> .git/info/sparse-checkout
echo "_pages/chat_modules/*" >> .git/info/sparse-checkout
echo "simulation_handler.py" >> .git/info/sparse-checkout
echo "utils/*.py" >> .git/info/sparse-checkout

# 병합 (충돌 시 수동 해결 필요)
git pull origin main --allow-unrelated-histories
```

## 방법 2: GitHub 웹사이트에서 직접 다운로드 (더 간단)

### 단계 1: GitHub 저장소 접속
https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits

### 단계 2: 필요한 파일 직접 다운로드
1. 각 파일을 클릭하여 Raw 버튼 클릭
2. 내용 복사하여 로컬 파일에 저장

### 다운로드해야 할 주요 파일 목록:

#### 채팅 시뮬레이터 핵심 파일
1. `_pages/_chat_simulator.py` (메인)
2. `_pages/_chat_agent_turn.py` (상담원 입력)
3. `_pages/_chat_customer_turn.py` (고객 반응)
4. `_pages/_chat_closing.py` (종료 처리)
5. `_pages/_chat_history.py` (이력 관리)
6. `_pages/_chat_initial_query.py` (초기 문의)
7. `_pages/_chat_messages.py` (메시지 표시)
8. `_pages/_chat_role_selection.py` (역할 선택)
9. `_pages/_chat_transfer.py` (이관)
10. `_pages/_chat_customer_message.py` (고객 메시지)
11. `_pages/_chat_styles.py` (스타일)

#### 맞춤형 콘텐츠 생성 파일
1. `_pages/_content.py`
2. `_pages/_content_generator.py` (있다면)

#### 관련 유틸리티 파일
1. `simulation_handler.py`
2. `utils/history_handler.py`
3. `utils/customer_analysis.py`
4. `utils/customer_verification.py`
5. `utils/audio_handler.py`
6. `simulation_perspective_logic.py`

#### chat_modules 폴더 전체
- `_pages/chat_modules/agent_turn.py`
- `_pages/chat_modules/closing_confirmation.py`
- `_pages/chat_modules/customer_closing_response.py`
- `_pages/chat_modules/customer_turn.py`
- `_pages/chat_modules/guideline_draft_customer.py`
- 기타 모든 파일

## 방법 3: ZIP 다운로드 후 파일 복사 (가장 간단)

### 단계 1: ZIP 다운로드
1. https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits 접속
2. Code → Download ZIP 클릭
3. ZIP 파일 다운로드 (예: `Utility-convenience-for-streamlits-main.zip`)

### 단계 2: 파일 복사
```bash
# PowerShell 사용
$zipPath = "C:\Users\Admin\Downloads\Utility-convenience-for-streamlits-main.zip"
$extractPath = "C:\Users\Admin\Downloads\temp_github"
$targetPath = "C:\Users\Admin\Downloads\Updated_streamlit_app_files"

# ZIP 압축 해제
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

# 필요한 파일 복사
Copy-Item "$extractPath\Utility-convenience-for-streamlits-main\_pages\_chat_*.py" "$targetPath\_pages\" -Force
Copy-Item "$extractPath\Utility-convenience-for-streamlits-main\_pages\_content*.py" "$targetPath\_pages\" -Force
Copy-Item "$extractPath\Utility-convenience-for-streamlits-main\_pages\chat_modules\*" "$targetPath\_pages\chat_modules\" -Recurse -Force
Copy-Item "$extractPath\Utility-convenience-for-streamlits-main\simulation_handler.py" "$targetPath\" -Force
Copy-Item "$extractPath\Utility-convenience-for-streamlits-main\utils\*.py" "$targetPath\utils\" -Force

# 임시 폴더 삭제
Remove-Item $extractPath -Recurse -Force
```

## 방법 4: Python 스크립트로 자동 복원

아래 Python 스크립트를 실행하면 자동으로 GitHub에서 필요한 파일을 가져옵니다:

```python
# restore_from_github.py
import os
import requests
import zipfile
import shutil
from pathlib import Path

GITHUB_REPO = "Salmonyeonwoo/Utility-convenience-for-streamlits"
GITHUB_BRANCH = "main"
TARGET_DIR = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files"
TEMP_DIR = r"C:\Users\Admin\Downloads\temp_github_restore"

def download_github_zip():
    """GitHub 저장소를 ZIP으로 다운로드"""
    url = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
    
    print(f"다운로드 중: {url}")
    response = requests.get(url, stream=True)
    zip_path = os.path.join(TEMP_DIR, "repo.zip")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"다운로드 완료: {zip_path}")
    return zip_path

def extract_and_copy_files(zip_path):
    """ZIP 압축 해제 및 필요한 파일 복사"""
    extract_path = os.path.join(TEMP_DIR, "extracted")
    
    # 압축 해제
    print("압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    repo_dir = os.path.join(extract_path, f"Utility-convenience-for-streamlits-{GITHUB_BRANCH}")
    
    # 복사할 파일 목록
    files_to_copy = [
        ("_pages/_chat_simulator.py", "_pages"),
        ("_pages/_chat_agent_turn.py", "_pages"),
        ("_pages/_chat_customer_turn.py", "_pages"),
        ("_pages/_chat_closing.py", "_pages"),
        ("_pages/_chat_history.py", "_pages"),
        ("_pages/_chat_initial_query.py", "_pages"),
        ("_pages/_chat_messages.py", "_pages"),
        ("_pages/_chat_role_selection.py", "_pages"),
        ("_pages/_chat_transfer.py", "_pages"),
        ("_pages/_chat_customer_message.py", "_pages"),
        ("_pages/_chat_styles.py", "_pages"),
        ("_pages/_content.py", "_pages"),
        ("_pages/_content_generator.py", "_pages"),
        ("simulation_handler.py", ""),
        ("simulation_perspective_logic.py", ""),
    ]
    
    # 파일 복사
    print("파일 복사 중...")
    for file_path, target_subdir in files_to_copy:
        source = os.path.join(repo_dir, file_path)
        if os.path.exists(source):
            target_dir = os.path.join(TARGET_DIR, target_subdir) if target_subdir else TARGET_DIR
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, os.path.basename(file_path))
            shutil.copy2(source, target)
            print(f"✓ {file_path}")
    
    # chat_modules 폴더 전체 복사
    chat_modules_source = os.path.join(repo_dir, "_pages", "chat_modules")
    chat_modules_target = os.path.join(TARGET_DIR, "_pages", "chat_modules")
    if os.path.exists(chat_modules_source):
        if os.path.exists(chat_modules_target):
            shutil.rmtree(chat_modules_target)
        shutil.copytree(chat_modules_source, chat_modules_target)
        print("✓ _pages/chat_modules/ (전체 폴더)")
    
    # utils 폴더의 파일들 복사
    utils_source = os.path.join(repo_dir, "utils")
    utils_target = os.path.join(TARGET_DIR, "utils")
    if os.path.exists(utils_source):
        os.makedirs(utils_target, exist_ok=True)
        for file in os.listdir(utils_source):
            if file.endswith('.py'):
                shutil.copy2(
                    os.path.join(utils_source, file),
                    os.path.join(utils_target, file)
                )
                print(f"✓ utils/{file}")

def cleanup():
    """임시 파일 정리"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"임시 파일 삭제 완료: {TEMP_DIR}")

if __name__ == "__main__":
    try:
        zip_path = download_github_zip()
        extract_and_copy_files(zip_path)
        cleanup()
        print("\n✅ 복원 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
```

## 복원 후 확인 사항

### 1. Import 오류 확인
```bash
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"
python -m py_compile _pages/_chat_simulator.py
python -m py_compile _pages/_content.py
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 테스트 실행
```bash
streamlit run streamlit_app.py
```

## 문제 해결

### Import 오류 발생 시
1. 파일 경로 확인
2. `__init__.py` 파일 존재 확인
3. 상대 import 경로 확인 및 수정

### 세션 상태 오류 시
- `config.py`의 `init_session_state()` 함수에 필요한 변수 추가

### UI 렌더링 오류 시
- CSS 파일 확인 (`_pages/_chat_styles.py`)
- Streamlit 버전 확인 및 업데이트


