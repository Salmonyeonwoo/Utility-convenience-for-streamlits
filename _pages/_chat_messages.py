# ========================================
# _pages/_chat_messages.py
# 채팅 시뮬레이터 - 대화 로그 표시 모듈 (간소화 버전)
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.audio_handler import render_tts_button
from _pages._chat_styles import get_chat_styles
from _pages._chat_customer_message import render_customer_message_with_icons
from _pages._chat_transfer import render_transfer_summary


def render_chat_messages(L, current_lang):
    """대화 로그 표시 및 메시지 렌더링 (카카오톡 스타일)"""
    # 피드백 저장 콜백 함수
    def save_feedback(index):
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{index}"
        if feedback_key in st.session_state:
            feedback_value = st.session_state[feedback_key]
            if index < len(st.session_state.simulator_messages):
                st.session_state.simulator_messages[index]["feedback"] = feedback_value

    # CSS 스타일 적용
    st.markdown(get_chat_styles(), unsafe_allow_html=True)

    # 메시지 표시
    if st.session_state.simulator_messages:
        for idx, msg in enumerate(st.session_state.simulator_messages):
            role = msg["role"]
            content = msg["content"]

            # 시스템 메시지는 제외
            if role in ["system_end", "system_transfer"]:
                continue

            # 메시지 타입별 렌더링
            if role == "customer" or role == "customer_rebuttal" or role == "initial_query":
                render_customer_message_with_icons(L, idx, content, current_lang)

            elif role == "agent_response":
                _render_agent_message(L, idx, content, save_feedback)

            elif role == "supervisor":
                _render_supervisor_message(content)

            # 고객 첨부 파일 표시
            if idx == 0 and role == "customer" and st.session_state.customer_attachment_b64:
                _render_attachment(L)
    else:
        st.info(L.get("no_messages", "아직 메시지가 없습니다."))

    # 이관 요약 표시
    actual_current_lang = st.session_state.get("language", current_lang)
    if actual_current_lang not in ["ko", "en", "ja"]:
        actual_current_lang = "ko"
    actual_L = LANG.get(actual_current_lang, LANG["ko"])
    
    show_guideline_ui = st.session_state.get("show_draft_ui", False) or st.session_state.get("show_customer_data_ui", False)
    should_show_transfer_summary = (
        (st.session_state.transfer_summary_text or (st.session_state.language != st.session_state.language_at_transfer_start and st.session_state.language_at_transfer_start)) and
        st.session_state.sim_stage != "AGENT_TURN" and not show_guideline_ui
    )
    if should_show_transfer_summary:
        render_transfer_summary(actual_L, actual_current_lang)


def _render_agent_message(L, idx, content, save_feedback):
    """에이전트 메시지 렌더링"""
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-start; margin: 8px 0; animation: slideInLeft 0.4s ease-out;">
        <div class="message-bubble message-agent">
            <div style="line-height: 1.5;">{content.replace(chr(10), '<br>')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 에이전트 응답 아이콘 버튼들
    col_icons = st.columns([1, 1, 1, 1, 1])
    with col_icons[0]:
        render_tts_button(
            content,
            st.session_state.language,
            role="agent",
            prefix="agent_response_",
            index=idx)
    
    with col_icons[1]:
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{idx}"
        existing_feedback = st.session_state.simulator_messages[idx].get("feedback", None)
        if existing_feedback is not None:
            st.session_state[feedback_key] = existing_feedback
        st.feedback(
            "thumbs",
            key=feedback_key,
            disabled=existing_feedback is not None,
            on_change=save_feedback,
            args=[idx],
        )


def _render_supervisor_message(content):
    """Supervisor 메시지 렌더링"""
    st.markdown(f"""
    <div style="display: flex; justify-content: center; margin: 10px 0;">
        <div class="message-bubble message-supervisor">
            <div>{content.replace(chr(10), '<br>')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_attachment(L):
    """첨부 파일 렌더링"""
    mime = st.session_state.customer_attachment_mime or "image/png"
    data_url = f"data:{mime};base64,{st.session_state.customer_attachment_b64}"

    if mime.startswith("image/"):
        caption_text = L.get("attachment_evidence_caption", "첨부된 증거물").format(
            filename=st.session_state.customer_attachment_file.name)
        st.image(data_url, caption=caption_text, use_column_width=True)
    elif mime == "application/pdf":
        warning_text = L.get(
            "attachment_pdf_warning",
            "첨부된 PDF 파일 ({filename})은 현재 인라인 미리보기가 지원되지 않습니다.").format(
            filename=st.session_state.customer_attachment_file.name)
        st.warning(warning_text)


