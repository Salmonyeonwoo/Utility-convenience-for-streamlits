"""
LLM 클라이언트 관리 모듈
API 키 관리, LLM 클라이언트 초기화, LLM 실행 등을 포함합니다.
"""
import os
import hashlib
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic

# google.generativeai는 지연 로딩 (필요할 때만 import)
GENAI_AVAILABLE = False
genai = None

def _ensure_genai():
    """genai 모듈을 필요할 때만 로드"""
    global genai, GENAI_AVAILABLE
    if not GENAI_AVAILABLE:
        try:
            import google.generativeai as genai
            GENAI_AVAILABLE = True
        except (ImportError, AttributeError):
            GENAI_AVAILABLE = False
            genai = None
    return GENAI_AVAILABLE

# 지원하는 API 목록 정의
SUPPORTED_APIS = {
    "openai": {
        "label": "OpenAI API Key",
        "secret_key": "OPENAI_API_KEY",
        "session_key": "user_openai_key",
        "placeholder": "sk-proj-**************************",
    },
    "gemini": {
        "label": "Google Gemini API Key",
        "secret_key": "GEMINI_API_KEY",
        "session_key": "user_gemini_key",
        "placeholder": "AIza***********************************",
    },
    "nvidia": {
        "label": "NVIDIA NIM API Key",
        "secret_key": "NVIDIA_API_KEY",
        "session_key": "user_nvidia_key",
        "placeholder": "nvapi-**************************",
    },
    "claude": {
        "label": "Anthropic Claude API Key",
        "secret_key": "CLAUDE_API_KEY",
        "session_key": "user_claude_key",
        "placeholder": "sk-ant-api-**************************",
    },
    "groq": {
        "label": "Groq API Key",
        "secret_key": "GROQ_API_KEY",
        "session_key": "user_groq_key",
        "placeholder": "gsk_**************************",
    },
}


def get_api_key(api: str) -> str:
    """API 키를 가져옵니다 (Secrets > Env Var > Session State 순서)"""
    cfg = SUPPORTED_APIS[api]

    # 1. Streamlit Secrets (최우선)
    try:
        if hasattr(st, "secrets") and cfg["secret_key"] in st.secrets:
            return st.secrets[cfg["secret_key"]]
    except Exception:
        pass

    # 2. Environment Variable
    env_key = os.environ.get(cfg["secret_key"])
    if env_key:
        return env_key

    # 3. Session State
    user_key = st.session_state.get(cfg["session_key"], "")
    if user_key:
        return user_key

    return ""


def get_llm_client():
    """선택된 모델에 맞는 클라이언트 + 모델코드 반환"""
    model_key = st.session_state.get("selected_llm", "openai_gpt4")

    # --- OpenAI ---
    if model_key.startswith("openai"):
        key = get_api_key("openai")
        if not key:
            return None, None
        try:
            client = OpenAI(api_key=key)
            model_name = "gpt-4o" if model_key == "openai_gpt4" else "gpt-3.5-turbo"
            return client, ("openai", model_name)
        except Exception:
            return None, None

    # --- Gemini ---
    if model_key.startswith("gemini"):
        if not _ensure_genai():
            return None, None
        key = get_api_key("gemini")
        if not key:
            return None, None
        try:
            genai.configure(api_key=key)
            model_name = "gemini-2.5-pro" if model_key == "gemini_pro" else "gemini-2.5-flash"
            return genai, ("gemini", model_name)
        except Exception:
            return None, None

    # --- Claude ---
    if model_key.startswith("claude"):
        key = get_api_key("claude")
        if not key:
            return None, None
        try:
            client = Anthropic(api_key=key)
            model_name = "claude-3-5-sonnet-latest"
            return client, ("claude", model_name)
        except Exception:
            return None, None

    # --- Groq ---
    if model_key.startswith("groq"):
        try:
            from groq import Groq
        except ImportError:
            return None, None
        key = get_api_key("groq")
        if not key:
            return None, None
        try:
            client = Groq(api_key=key)
            model_name = (
                "llama3-70b-8192"
                if "llama3" in model_key
                else "mixtral-8x7b-32768"
            )
            return client, ("groq", model_name)
        except Exception:
            return None, None

    return None, None


def run_llm(prompt: str, max_tokens: int = 2000) -> str:
    """
    선택된 LLM으로 프롬프트 실행 (Fallback 지원)
    
    Args:
        prompt: LLM에 전달할 프롬프트
        max_tokens: 최대 토큰 수 (기본값: 2000, 채팅 응답에 적합)
                    전화 응답 등 짧은 응답이 필요한 경우 200 등으로 조정 가능
    """
    client, info = get_llm_client()

    # None 값 언팩 방지
    if info and isinstance(info, tuple) and len(info) == 2:
        provider, model_name = info
    else:
        provider, model_name = None, None

    # Fallback 순서 정의 (Gemini 우선)
    llm_attempts = []

    # 1. Gemini
    gemini_key = get_api_key("gemini")
    if gemini_key and _ensure_genai():
        llm_attempts.append(("gemini", gemini_key, "gemini-2.5-pro" if "pro" in str(model_name) else "gemini-2.5-flash"))

    # 2. OpenAI
    openai_key = get_api_key("openai")
    if openai_key:
        llm_attempts.append(("openai", openai_key, "gpt-4o" if "4" in str(model_name) else "gpt-3.5-turbo"))

    # 3. Claude
    claude_key = get_api_key("claude")
    if claude_key:
        llm_attempts.append(("claude", claude_key, "claude-3-5-sonnet-latest"))

    # 4. Groq
    groq_key = get_api_key("groq")
    if groq_key:
        groq_model = "llama3-70b-8192" if "llama3" in str(model_name) else "mixtral-8x7b-32768"
        llm_attempts.append(("groq", groq_key, groq_model))

    # 주력 모델을 첫 순서로
    if provider and provider in [attempt[0] for attempt in llm_attempts]:
        primary_attempt = next((attempt for attempt in llm_attempts if attempt[0] == provider), None)
        if primary_attempt:
            llm_attempts.remove(primary_attempt)
            llm_attempts.insert(0, primary_attempt)

    # Fallback 루프 무한 대기 방지
    if not llm_attempts:
        return "❌ 사용 가능한 LLM API 키가 없습니다. API 키를 설정해주세요."

    # LLM 순차 실행
    for provider, key, model in llm_attempts:
        if not key:
            continue

        try:
            if provider == "gemini" and _ensure_genai():
                genai.configure(api_key=key)
                gen_model = genai.GenerativeModel(model)
                # 채팅 응답을 위한 충분한 토큰 수 설정
                generation_config = {
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                }
                resp = gen_model.generate_content(prompt, generation_config=generation_config)
                return resp.text

            elif provider == "openai":
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,  # 채팅 응답을 위한 충분한 토큰 수
                    temperature=0.7,
                )
                return resp.choices[0].message.content

            elif provider == "claude":
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,  # 채팅 응답을 위한 충분한 토큰 수
                    temperature=0.7,
                )
                return resp.content[0].text

            elif provider == "groq":
                from groq import Groq
                g_client = Groq(api_key=key)
                resp = g_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,  # 채팅 응답을 위한 충분한 토큰 수
                    temperature=0.7,
                )
                return resp.choices[0].message.content

        except Exception as e:
            print(f"LLM {provider} ({model}) failed: {e}")
            continue

    return "❌ 모든 LLM API 키가 작동하지 않거나 할당량이 소진되었습니다."


def init_openai_audio_client():
    """OpenAI TTS/Whisper용 클라이언트 초기화"""
    key = get_api_key("openai")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except:
        return None


def init_llm_clients_lazy():
    """LLM 클라이언트를 지연 로딩으로 초기화 (앱 렌더링 이후에만 호출)"""
    # OpenAI 클라이언트 캐싱
    if "openai_client" not in st.session_state or st.session_state.openai_client is None:
        try:
            st.session_state.openai_client = init_openai_audio_client()
        except Exception:
            st.session_state.openai_client = None

    # LLM 준비 상태 캐싱
    if "is_llm_ready" not in st.session_state or "llm_ready_checked" not in st.session_state:
        try:
            probe_client, _ = get_llm_client()
            st.session_state.is_llm_ready = probe_client is not None
        except Exception:
            st.session_state.is_llm_ready = False
        st.session_state.llm_ready_checked = True

    # API 키 변경 감지
    current_api_keys_hash = hashlib.md5(
        f"{get_api_key('openai')}{get_api_key('gemini')}{get_api_key('claude')}{get_api_key('groq')}".encode()
    ).hexdigest()

    if "api_keys_hash" not in st.session_state:
        st.session_state.api_keys_hash = current_api_keys_hash
    elif st.session_state.api_keys_hash != current_api_keys_hash:
        try:
            probe_client, _ = get_llm_client()
            st.session_state.is_llm_ready = probe_client is not None
        except Exception:
            st.session_state.is_llm_ready = False
        st.session_state.api_keys_hash = current_api_keys_hash
        
        try:
            st.session_state.openai_client = init_openai_audio_client()
        except Exception:
            st.session_state.openai_client = None
