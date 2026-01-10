# ========================================
# _pages/chat_modules/agent_turn_chat_input_ui.py
# 에이전트 턴 - 채팅 입력 UI (카카오톡 스타일)
# ========================================

import streamlit as st
from _pages.chat_modules._chat_input_styles import get_chat_input_styles
from _pages.chat_modules._chat_input_js import render_draft_fill_script, render_attach_button_script
from _pages.chat_modules._clipboard_handler import render_clipboard_paste_handler, handle_clipboard_processing

def render_chat_input_ui(L, current_lang, perspective):
    """채팅 입력 UI 렌더링 (카카오톡 스타일)"""
    # 응대 초안이 있으면 자동으로 입력창에 표시
    initial_value = ""
    if st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        initial_value = st.session_state.auto_generated_draft_text
    elif st.session_state.get("agent_response_area_text") and st.session_state.agent_response_area_text:
        initial_value = st.session_state.agent_response_area_text
    
    placeholder_text = L.get("agent_response_placeholder", "고객에게 응답하세요...")
    
    # ⭐ 상담원 모드일 때만 카카오톡 스타일 채팅 입력창 및 파일 첨부 버튼 표시
    if perspective == "AGENT":
        st.markdown(get_chat_input_styles(), unsafe_allow_html=True)
        
        # ⭐ Ctrl+V로 이미지/동영상 붙여넣기 기능 추가 (먼저 추가하여 파일 업로더가 열리기 전에 감지)
        st.markdown(render_clipboard_paste_handler(L, current_lang), unsafe_allow_html=True)
    
    # Streamlit의 chat_input 사용 (자동 업데이트 지원)
    agent_response_input = None
    if perspective == "AGENT":
        agent_response_input = st.chat_input(placeholder_text, key="agent_chat_input_main")
        
        # 클립보드 처리 (JavaScript에서 자동 처리되므로 rerun 없음)
        # handle_clipboard_processing(L)  # 불필요한 rerun 방지
    
    # ⭐ 응대 초안이 있으면 입력창에 자동으로 채우기
    if perspective == "AGENT" and agent_response_input is not None and st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        draft_text = st.session_state.auto_generated_draft_text
        st.markdown(render_draft_fill_script(draft_text, current_lang), unsafe_allow_html=True)

    # ⭐ 파일 첨부 버튼을 입력창 안쪽에 배치 (카카오톡 스타일)
    if perspective == "AGENT":
        st.markdown(render_attach_button_script(), unsafe_allow_html=True)
        
        if st.button(
                "➕",
                key="btn_add_attachment_unified_hidden",
                help=L.get("button_add_attachment", "➕ 파일 첨부"),
                use_container_width=False,
                type="secondary"):
            st.session_state.show_agent_file_uploader = True
    
    return agent_response_input
