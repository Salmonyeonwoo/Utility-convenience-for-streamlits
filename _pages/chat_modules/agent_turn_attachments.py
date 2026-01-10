# ========================================
# _pages/chat_modules/agent_turn_attachments.py
# 에이전트 턴 - 첨부 파일 및 오디오 처리
# ========================================

import streamlit as st
import base64
import io
from datetime import datetime
from utils.audio_handler import transcribe_bytes_with_whisper
from llm_client import get_api_key

def render_agent_attachments(L):
    """에이전트 첨부 파일 업로더 렌더링 (스크린샷/동영상 지원)"""
    # ⭐ 클립보드 처리: JavaScript에서 자동으로 처리되므로 Python 레벨에서 rerun 불필요
    # 파일 업로더가 열려 있을 때 JavaScript에서 자동으로 LocalStorage에서 데이터를 읽어서 처리
    
    agent_attachment_files = None
    if st.session_state.get("show_agent_file_uploader", False):
        agent_attachment_files = st.file_uploader(
            L["agent_attachment_label"],
            type=["png", "jpg", "jpeg", "pdf", "mp4", "webm", "mov", "avi"],  # 동영상 지원 추가
            key="agent_attachment_file_uploader",
            help=L.get("agent_attachment_placeholder", "스크린샷 또는 동영상 파일을 업로드하세요. (Ctrl+V로 클립보드 이미지/동영상 붙여넣기 가능)"),
            accept_multiple_files=True
        )
        # ⭐ 클립보드에서 붙여넣은 파일이 이미 agent_attachment_file에 추가되었는지 확인
        if "agent_attachment_file" not in st.session_state:
            st.session_state.agent_attachment_file = []
        
        # 파일 업로더에서 선택한 파일 처리
        if agent_attachment_files:
            # 파일 정보 저장
            attachment_list = st.session_state.agent_attachment_file.copy() if st.session_state.agent_attachment_file else []
            
            for f in agent_attachment_files:
                # 이미 추가된 파일인지 확인 (중복 방지)
                already_added = any(
                    existing.get("name") == f.name and 
                    existing.get("size") == f.size 
                    for existing in attachment_list
                )
                
                if not already_added:
                    file_info = {
                        "name": f.name,
                        "type": f.type,
                        "size": f.size,
                        "data": None  # base64 데이터 (필요 시)
                    }
                    
                    # 이미지/동영상 파일인 경우 base64로 인코딩하여 저장
                    if f.type and (f.type.startswith("image/") or f.type.startswith("video/")):
                        file_bytes = f.read()
                        base64_data = base64.b64encode(file_bytes).decode('utf-8')
                        file_info["data"] = base64_data
                        file_info["data_url"] = f"data:{f.type};base64,{base64_data}"
                        # 파일 포인터 재설정
                        f.seek(0)
                    
                    attachment_list.append(file_info)
            
            st.session_state.agent_attachment_file = attachment_list
            
            file_names = ", ".join([f["name"] for f in st.session_state.agent_attachment_file])
            st.success(
                L.get(
                    "agent_attachment_files_ready",
                    "✅ {count}개 에이전트 첨부 파일 준비 완료: {files}").format(
                    count=len(agent_attachment_files),
                    files=file_names))
            
            # 첨부된 이미지/동영상 미리보기 표시
            for idx, file_info in enumerate(st.session_state.agent_attachment_file):
                if file_info.get("data_url"):
                    if file_info["type"].startswith("image/"):
                        st.image(file_info["data_url"], caption=file_info["name"], use_column_width=True)
                    elif file_info["type"].startswith("video/"):
                        st.video(file_info["data_url"])
            
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

