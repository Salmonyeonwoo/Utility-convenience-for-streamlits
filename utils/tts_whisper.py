"""
TTS 및 Whisper 관련 함수 모듈
음성 전사, TTS 생성, TTS 버튼 렌더링 등을 포함합니다.
"""
import os
import tempfile
import hashlib
import streamlit as st
from utils.llm_clients import get_api_key
from utils.i18n import LANG

# 역할별 TTS 음성 스타일 설정
TTS_VOICES = {
    "customer": {
        "gender": "male",
        "voice": "alloy"  # Distinct Male, Generic/Customer
    },
    "agent": {
        "gender": "female",
        "voice": "shimmer"  # Distinct Female, Professional/Agent
    },
    "supervisor": {
        "gender": "female",
        "voice": "nova"  # Another Distinct Female, Informative/Supervisor
    }
}


def transcribe_bytes_with_whisper(audio_bytes: bytes, mime_type: str = "audio/webm", lang_code: str = "ko") -> str:
    """
    OpenAI Whisper API를 사용하여 오디오 바이트를 텍스트로 전사합니다.
    """
    L = LANG[st.session_state.language]
    client = st.session_state.openai_client
    if client is None:
        return f"❌ {L['openai_missing']}"

    whisper_lang = {"ko": "ko", "en": "en", "ja": "ja"}.get(lang_code, "en")

    # 임시 파일 저장 (Whisper API 호환성)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()

    try:
        with open(tmp.name, "rb") as f:
            res = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
                language=whisper_lang,
            )
        # res.text 속성이 있는지 확인하고 없으면 res 자체를 문자열로 변환
        return res.text.strip() if hasattr(res, 'text') else str(res).strip()
    except Exception as e:
        # 파일 형식 오류 등 상세 오류 처리
        return f"❌ {L['error']} Whisper: {e}"
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def transcribe_audio(audio_bytes, filename="audio.wav"):
    """오디오를 텍스트로 전사 (OpenAI Whisper 우선, Gemini Fallback)"""
    import io
    client = st.session_state.openai_client

    # 1️⃣ OpenAI Whisper 시도
    if client:
        try:
            bio = io.BytesIO(audio_bytes)
            bio.name = filename
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
            )
            return resp.text
        except Exception as e:
            print("Whisper OpenAI failed:", e)

    # 2️⃣ Gemini STT fallback
    try:
        # google.generativeai는 지연 로딩
        try:
            import google.generativeai as genai
            genai.configure(api_key=get_api_key("gemini"))
            model = genai.GenerativeModel("gemini-2.5-flash")
            text = model.generate_content("Transcribe this audio:").text
            return text or ""
        except (ImportError, Exception) as e:
            print("Gemini STT failed:", e)
    except Exception as e:
        print("Gemini STT failed:", e)

    return "❌ STT not available"


def synthesize_tts(text: str, lang_key: str, role: str = "agent"):
    """TTS로 텍스트를 음성으로 변환"""
    L = LANG[lang_key]
    client = st.session_state.openai_client
    if client is None:
        return None, L["openai_missing"]

    if role not in TTS_VOICES:
        role = "agent"

    voice_name = TTS_VOICES[role]["voice"]

    try:
        # tts-1 모델 사용 (안정성)
        resp = client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=text
            # format="mp3"은 기본값입니다.
        )
        return resp.read(), L["tts_status_success"]

    except Exception as e:
        return None, f"{L['tts_status_error']}: {e}"


def render_tts_button(text, lang_key, role="customer", prefix="", index: int = -1):
    """TTS 재생 버튼을 렌더링"""
    L = LANG[lang_key]

    # ⭐ 수정: index=-1인 경우, UUID를 사용하여 safe_key 생성
    if index == -1:
        # 이관 요약처럼 인덱스가 고정되지 않는 경우, 텍스트 해시와 세션 인스턴스 ID를 조합
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        session_id_part = st.session_state.get('sim_instance_id', 'default_session')
        safe_key = f"{prefix}_tts_{session_id_part}_{content_hash}"
    else:
        safe_key = f"{prefix}_tts_{index}"

    if st.button(f"🔊 {L['button_listen_audio']}", key=safe_key):
        with st.spinner(L["tts_status_generating"]):
            audio_bytes, status_msg = synthesize_tts(text, lang_key, role)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                st.success(status_msg)
            else:
                st.error(status_msg)
