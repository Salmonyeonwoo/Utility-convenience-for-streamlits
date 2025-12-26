# GitHub 저장소에서 기능 복원 가이드

## 저장소 정보
- **URL**: https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits
- **복원 대상**: 채팅 시뮬레이터 기능 + 맞춤형 콘텐츠 생성 탭

## 복원 방법

### 방법 1: Git Clone (권장)

```bash
# 1. 현재 프로젝트 디렉토리로 이동
cd "C:\Users\Admin\Downloads\Updated_streamlit_app_files"

# 2. GitHub 저장소 클론 (임시 디렉토리)
cd ..
git clone https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits.git temp_github_repo

# 3. 필요한 파일 복사
# 채팅 시뮬레이터 관련 파일
copy temp_github_repo\_pages\_chat_simulator.py "_pages\_chat_simulator.py"
copy temp_github_repo\_pages\_chat_*.py "_pages\" 
copy temp_github_repo\_pages\chat_modules\*.py "_pages\chat_modules\"

# 맞춤형 콘텐츠 생성 관련 파일
copy temp_github_repo\_pages\_content*.py "_pages\"

# 관련 유틸리티 파일
copy temp_github_repo\simulation_handler.py .
copy temp_github_repo\utils\*.py "utils\"

# 4. 임시 디렉토리 삭제
cd ..
rmdir /s /q temp_github_repo
```

### 방법 2: GitHub 웹사이트에서 직접 다운로드

1. 저장소 접속: https://github.com/Salmonyeonwoo/Utility-convenience-for-streamlits
2. Code → Download ZIP 클릭
3. ZIP 파일 압축 해제
4. 다음 파일들을 현재 프로젝트에 복사:

#### 채팅 시뮬레이터 관련 파일
```
_pages/_chat_simulator.py
_pages/_chat_agent_turn.py
_pages/_chat_customer_turn.py
_pages/_chat_closing.py
_pages/_chat_history.py
_pages/_chat_initial_query.py
_pages/_chat_messages.py
_pages/_chat_role_selection.py
_pages/_chat_transfer.py
_pages/_chat_customer_message.py
_pages/_chat_styles.py
_pages/chat_modules/ (전체 폴더)
```

#### 맞춤형 콘텐츠 생성 관련 파일
```
_pages/_content.py
_pages/_content_generator.py (있다면)
```

#### 관련 유틸리티 파일
```
simulation_handler.py
utils/history_handler.py (업데이트 필요)
utils/customer_analysis.py
utils/customer_verification.py
utils/audio_handler.py
simulation_perspective_logic.py
```

## 복원 후 확인 사항

### 1. 채팅 시뮬레이터 기능 확인
- `_pages/_chat_simulator.py` 파일 존재 확인
- 카카오톡 스타일 말풍선 UI 작동 확인
- 고객/상담원 채팅 기능 확인
- AI 응대 가이드라인 기능 확인
- 이관 기능 확인
- 힌트 기능 확인

### 2. 맞춤형 콘텐츠 생성 기능 확인
- `_pages/_content.py` 파일 존재 확인
- 콘텐츠 생성 UI 확인
- 요약/퀴즈/실용 예제 생성 기능 확인

### 3. 의존성 확인
```bash
# requirements.txt 확인 및 설치
pip install -r requirements.txt
```

### 4. streamlit_app.py 업데이트 확인
- 채팅 시뮬레이터 import 및 렌더링 코드 확인
- 맞춤형 콘텐츠 생성 탭 연결 확인

## 복원 순서

1. **백업 생성** (현재 파일 보존)
   ```bash
   # 현재 _pages 폴더 백업
   xcopy "_pages" "_pages_backup" /E /I
   ```

2. **GitHub 저장소에서 파일 다운로드**
   - 방법 1 또는 방법 2 선택

3. **파일 복사 및 덮어쓰기**
   - 위의 파일 목록 참고

4. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

5. **import 경로 확인**
   - 각 파일의 import 문이 현재 프로젝트 구조와 맞는지 확인

6. **테스트 실행**
   ```bash
   streamlit run streamlit_app.py
   ```

7. **오류 수정**
   - import 오류 수정
   - 경로 오류 수정
   - 세션 상태 초기화 확인

## 주요 확인 포인트

### Import 경로
현재 프로젝트와 GitHub 저장소의 구조가 다를 수 있으므로 import 경로 확인:

```python
# 예시: 현재 프로젝트 구조에 맞게 수정 필요
from _pages._chat_simulator import render_chat_simulator
from utils.history_handler import save_simulation_history_local
```

### 세션 상태 초기화
`config.py`의 `init_session_state()` 함수에 필요한 세션 상태 변수 추가 확인

### 설정 파일
`config.py`, `lang_pack.py` 등에 필요한 설정 추가 확인

## 문제 해결

### ImportError 발생 시
1. 파일 경로 확인
2. `__init__.py` 파일 존재 확인 (필요한 경우 생성)
3. Python 경로 확인

### 세션 상태 오류 시
- `config.py`의 `init_session_state()` 함수 확인
- 필요한 세션 상태 변수 추가

### UI 렌더링 오류 시
- CSS 파일 확인 (`_pages/_chat_styles.py`)
- Streamlit 버전 확인

