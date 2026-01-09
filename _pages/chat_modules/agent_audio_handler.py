# ========================================
# chat_modules/agent_audio_handler.py
# 에이전트 오디오 전사 처리 모듈
# ========================================

import streamlit as st
from utils.audio_handler import transcribe_bytes_with_whisper
from llm_client import get_api_key


def handle_audio_transcription(L):
    """오디오 전사 처리"""
    if "bytes_to_process" not in st.session_state or st.session_state.bytes_to_process is None:
        return False
    
    has_openai = st.session_state.openai_client is not None
    has_gemini = bool(get_api_key("gemini"))
    
    if not has_openai and not has_gemini:
        st.error(
            L.get("whisper_client_error", "Whisper 클라이언트 오류") +
            " (OpenAI 또는 Gemini API Key 필요)")
        st.session_state.bytes_to_process = None
        return False
    
    agent_response_transcript = None
    audio_bytes_backup = st.session_state.bytes_to_process
    st.session_state.bytes_to_process = None
    
    with st.spinner(L.get("whisper_processing", "전사 중...")):
        try:
            agent_response_transcript = transcribe_bytes_with_whisper(
                audio_bytes_backup, "audio/wav", lang_code=None, auto_detect=True)
        except Exception as e:
            agent_response_transcript = L.get(
                "transcription_error_with_error",
                "❌ 전사 오류: {error}").format(error=str(e))
    
    if not agent_response_transcript or agent_response_transcript.startswith("❌"):
        error_msg = agent_response_transcript if agent_response_transcript else L.get(
            "transcription_no_result", "전사 결과가 없습니다.")
        st.error(error_msg)
        _reset_transcription_state()
        return False
    
    if not agent_response_transcript.strip():
        st.warning(L.get("transcription_empty_warning", "전사 결과가 비어 있습니다."))
        _reset_transcription_state()
        return False
    
    # 전사 성공 처리
    agent_response_transcript = agent_response_transcript.strip()
    st.session_state.last_transcript = agent_response_transcript
    
    if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
        st.session_state.agent_response_area_text = agent_response_transcript
    else:
        st.session_state.current_agent_audio_text = agent_response_transcript
        if "agent_response_input_box_widget_call" in st.session_state:
            st.session_state.agent_response_input_box_widget_call = agent_response_transcript
    
    snippet = agent_response_transcript[:50].replace("\n", " ")
    if len(agent_response_transcript) > 50:
        snippet += "..."
    st.success(
        L.get("whisper_success", "전사 완료") +
        f" **{L.get('recognized_content', '인식 내용')}:** *{snippet}*")
    st.info(L.get("transcription_auto_filled",
                  "💡 전사된 텍스트가 CC 자막 및 입력창에 자동으로 입력되었습니다."))
    
    return True


def _reset_transcription_state():
    """전사 상태 초기화"""
    if st.session_state.get("feature_selection") == st.session_state.get("L", {}).get("sim_tab_chat_email"):
        st.session_state.agent_response_area_text = ""
    else:
        st.session_state.current_agent_audio_text = ""
        if "agent_response_input_box_widget_call" in st.session_state:
            st.session_state.agent_response_input_box_widget_call = ""
    st.session_state.last_transcript = ""

