# ========================================
# chat_modules/agent_input_ui.py
# 에이전트 입력 UI 모듈
# ========================================

import streamlit as st
from chat_modules.agent_input_js import render_draft_auto_fill_js, render_attachment_button_js


def render_chat_input(L, current_lang):
    """채팅 입력 UI 렌더링"""
    perspective = st.session_state.get("sim_perspective", "AGENT")
    
    if perspective == "AGENT":
        _render_kakao_style_input()
    
    # 입력창 초기값 설정
    initial_value = ""
    if st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        initial_value = st.session_state.auto_generated_draft_text
    elif st.session_state.get("agent_response_area_text") and st.session_state.agent_response_area_text:
        initial_value = st.session_state.agent_response_area_text
    
    placeholder_text = L.get("agent_response_placeholder", "고객에게 응답하세요...")
    
    # Streamlit의 chat_input 사용
    agent_response_input = None
    if perspective == "AGENT":
        agent_response_input = st.chat_input(placeholder_text, key="agent_chat_input_main")
    
    # 응대 초안 자동 채우기
    if perspective == "AGENT" and agent_response_input is not None and st.session_state.get("auto_generated_draft_text"):
        render_draft_auto_fill_js(st.session_state.auto_generated_draft_text, current_lang)
    
    # 파일 첨부 버튼 렌더링
    render_attachment_button_js(L)
    _render_attachment_button(L)
    
    return agent_response_input


def _render_kakao_style_input():
    """카카오톡 스타일 입력창 스타일"""
    st.markdown("""
    <style>
    .kakao-chat-input {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 24px;
        padding: 12px 20px;
        font-size: 15px;
        min-height: 50px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kakao-chat-input:focus {
        outline: none;
        border-color: #FEE500;
        box-shadow: 0 2px 8px rgba(254, 229, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


def _render_attachment_button(L):
    """파일 첨부 버튼 렌더링"""
    if st.button(
            "➕",
            key="btn_add_attachment_unified_hidden",
            help=L.get("button_add_attachment", "➕ 파일 첨부"),
            use_container_width=False,
            type="secondary"):
        st.session_state.show_agent_file_uploader = True

