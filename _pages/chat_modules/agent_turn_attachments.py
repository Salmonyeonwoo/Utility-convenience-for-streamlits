# ========================================
# _pages/chat_modules/agent_turn_attachments.py
# 에이전트 턴 - 첨부 파일 및 오디오 처리
# ========================================

import streamlit as st
from utils.audio_handler import transcribe_bytes_with_whisper
from llm_client import get_api_key

def render_agent_attachments(L):
    """에이전트 첨부 파일 업로더 렌더링"""
    agent_attachment_files = None
    if st.session_state.get("show_agent_file_uploader", False):
        agent_attachment_files = st.file_uploader(
            L["agent_attachment_label"],
            type=["png", "jpg", "jpeg", "pdf"],
            key="agent_attachment_file_uploader",
            help=L["agent_attachment_placeholder"],
            accept_multiple_files=True
        )
        if agent_attachment_files:
            st.session_state.agent_attachment_file = [
                {"name": f.name, "type": f.type, "size": f.size} for f in agent_attachment_files
            ]
            file_names = ", ".join(
                [f["name"] for f in st.session_state.agent_attachment_file])
            st.info(
                L.get(
                    "agent_attachment_files_ready",
                    "✅ {count}개 에이전트 첨부 파일 준비 완료: {files}").format(
                    count=len(agent_attachment_files),
                    files=file_names))
            st.session_state.show_agent_file_uploader = False
        else:
            st.session_state.agent_attachment_file = []
    else:
        st.session_state.agent_attachment_file = []

def process_audio_transcription(L):
    """마이크 녹음 처리"""
    if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process is not None:
        has_openai = st.session_state.openai_client is not None
        has_gemini = bool(get_api_key("gemini"))

        if not has_openai and not has_gemini:
            st.error(
                L.get(
                    "whisper_client_error",
                    "Whisper 클라이언트 오류") +
                " (OpenAI 또는 Gemini API Key 필요)")
            st.session_state.bytes_to_process = None
        else:
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
                        "❌ 전사 오류: {error}").format(
                        error=str(e))

            if not agent_response_transcript or agent_response_transcript.startswith("❌"):
                error_msg = agent_response_transcript if agent_response_transcript else L.get(
                    "transcription_no_result", "전사 결과가 없습니다.")
                st.error(error_msg)

                if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                    st.session_state.agent_response_area_text = ""
                    st.session_state.last_transcript = ""
                else:
                    st.session_state.current_agent_audio_text = L.get(
                        "transcription_error", "전사 오류")
                    if "agent_response_input_box_widget_call" in st.session_state:
                        st.session_state.agent_response_input_box_widget_call = ""
                    st.session_state.last_transcript = ""

            elif not agent_response_transcript.strip():
                st.warning(
                    L.get(
                        "transcription_empty_warning",
                        "전사 결과가 비어 있습니다."))
                if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                    st.session_state.agent_response_area_text = ""
                else:
                    st.session_state.current_agent_audio_text = ""
                    if "agent_response_input_box_widget_call" in st.session_state:
                        st.session_state.agent_response_input_box_widget_call = ""
                st.session_state.last_transcript = ""

            elif agent_response_transcript.strip():
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
                st.info(
                    L.get(
                        "transcription_auto_filled",
                        "💡 전사된 텍스트가 CC 자막 및 입력창에 자동으로 입력되었습니다."))

