# Streamlit 앱 모듈화 가이드

## 📁 프로젝트 구조

```
/Updated_streamlit_app_files/
├── streamlit_app.py              # 메인 앱 (간소화됨)
├── requirements.txt
├── utils/                        # 유틸리티 모듈
│   ├── __init__.py
│   ├── config.py                 # 경로 설정, 상수
│   ├── data_helpers.py           # 데이터 입출력 ✅ 생성 완료
│   ├── llm_clients.py            # LLM 클라이언트 관리
│   ├── tts_whisper.py            # TTS/Whisper 함수
│   ├── rag_helpers.py            # RAG 관련 함수
│   └── prompt_generator.py       # 프롬프트 생성 함수
└── pages/                        # Streamlit 다중 페이지 (선택사항)
    ├── 1_chat_simulator.py
    ├── 2_phone_simulator.py
    └── 3_rag_chatbot.py
```

## 🔧 모듈화 진행 단계

### 1단계: utils/data_helpers.py ✅ 완료
- `_load_json`, `_save_json`
- `load_voice_records`, `save_voice_records`
- `load_simulation_histories_local`, `save_simulation_history_local`
- `export_history_to_json/text/excel`

### 2단계: utils/llm_clients.py (다음 작업)
다음 함수들을 포함해야 합니다:
- `SUPPORTED_APIS` 딕셔너리
- `get_api_key(api)`
- `get_llm_client()`
- `run_llm(prompt: str)`
- `init_openai_audio_client()`
- `init_llm_clients_lazy()`

### 3단계: utils/tts_whisper.py
- `transcribe_bytes_with_whisper()`
- `transcribe_audio()`
- `synthesize_tts()`
- `render_tts_button()`

### 4단계: utils/rag_helpers.py
- `load_documents()`
- `split_documents()`
- `get_embedding_model()`
- `get_embedding_function()`
- `build_rag_index()`
- `load_rag_index()`
- `rag_answer()`

### 5단계: utils/prompt_generator.py
- `generate_customer_reaction()`
- `generate_agent_response_draft()`
- `summarize_history_with_ai()`
- `generate_customer_reaction_for_call()`
- `generate_agent_first_greeting()`
- 기타 프롬프트 생성 함수들

### 6단계: 메인 streamlit_app.py 재구성
- 모든 import를 utils 모듈에서 가져오기
- UI 로직만 남기고 비즈니스 로직은 함수 호출로 대체

## 📝 사용 예시

### 메인 앱에서 사용하기

```python
# streamlit_app.py
import streamlit as st
from utils.data_helpers import load_simulation_histories_local, save_simulation_history_local
from utils.llm_clients import run_llm, get_api_key, init_llm_clients_lazy
from utils.tts_whisper import synthesize_tts, transcribe_bytes_with_whisper
from utils.rag_helpers import build_rag_index, rag_answer
from utils.prompt_generator import generate_customer_reaction

# LLM 초기화 (렌더링 이후)
init_llm_clients_lazy()

# 함수 사용
response = run_llm("안녕하세요")
histories = load_simulation_histories_local("ko")
```

## ⚠️ 주의사항

1. **순환 import 방지**: 모듈 간 의존성을 최소화하세요
2. **Streamlit session_state**: `st.session_state`는 메인 앱에서만 직접 접근
3. **경로 설정**: `utils/config.py`의 경로 설정이 올바른지 확인
4. **다국어 설정**: LANG 딕셔너리는 별도 파일(`utils/i18n.py`)로 분리 권장

## 🚀 다음 단계

1. 각 모듈 파일 생성 완료
2. 메인 streamlit_app.py에서 기존 코드를 import로 대체
3. 테스트 및 디버깅
4. pages/ 폴더로 UI 분리 (선택사항)



































