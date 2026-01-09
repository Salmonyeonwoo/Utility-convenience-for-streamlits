# ========================================
# _pages/_chat_agent_turn.py
# 채팅 시뮬레이터 - 에이전트 입력 단계 처리 모듈 (메인)
# ========================================

import streamlit as st
from lang_pack import LANG

# 모듈화된 컴포넌트들 import
from _pages.chat_modules.agent_turn_customer_mode import handle_customer_mode_auto_response
from _pages.chat_modules.agent_turn_customer_reaction import handle_pending_customer_reaction
from _pages.chat_modules.agent_turn_guidelines import render_guidelines_and_info
from _pages.chat_modules.agent_turn_verification import process_customer_verification
from _pages.chat_modules.agent_turn_attachments import render_agent_attachments, process_audio_transcription
from _pages.chat_modules.agent_turn_draft_generation import handle_auto_draft_generation, handle_transcript_auto_send
from _pages.chat_modules.agent_turn_chat_input_ui import render_chat_input_ui
from _pages.chat_modules.agent_turn_manual_input import handle_manual_agent_input
from _pages.chat_modules.agent_turn_language_transfer import render_language_transfer
from _pages.chat_modules.agent_turn_verification_ui import render_verification_debug_info, render_verification_ui

def render_agent_turn(L, current_lang):
    """에이전트 입력 단계 UI 렌더링"""
    perspective = st.session_state.get("sim_perspective", "AGENT")
    
    # ⭐ 고객 체험 모드일 때 AI가 자동으로 응답 생성
    if handle_customer_mode_auto_response(L, current_lang):
        return  # 고객 모드일 때는 기존 상담원 입력 UI를 표시하지 않음
    
    # ⭐ 자연스러운 대화 흐름: 에이전트 응답 후 고객 반응 생성
    handle_pending_customer_reaction(L)
    
    show_verification_from_button = st.session_state.get("show_verification_ui", False)
    show_draft_ui = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui = st.session_state.get("show_customer_data_ui", False)

    if show_verification_from_button:
        pass
    elif show_draft_ui:
        pass
    elif show_customer_data_ui:
        pass
    else:
        st.markdown(f"### {L['agent_response_header']}")

    # 고객 성향 기반 가이드라인 추천 및 정보 표시
    render_guidelines_and_info(L)

    # 고객 검증 프로세스
    is_login_inquiry, customer_provided_info, customer_has_attachment, all_customer_texts, all_roles, customer_messages = process_customer_verification(L)

    # 고객 검증 UI 표시
    show_draft_ui_check = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui_check = st.session_state.get("show_customer_data_ui", False)
    if show_verification_from_button and not show_draft_ui_check and not show_customer_data_ui_check:
        st.markdown("---")
        st.markdown(f"### {L.get('verification_header', '고객 검증')}")
        st.warning(L.get('verification_warning', '고객 검증이 필요합니다.'))

    # 검증 UI 렌더링
    if is_login_inquiry and show_verification_from_button:
        render_verification_debug_info(L, is_login_inquiry, customer_provided_info, 
                                        customer_has_attachment, all_customer_texts, all_roles, customer_messages)

    show_draft_ui_check2 = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui_check2 = st.session_state.get("show_customer_data_ui", False)
    if is_login_inquiry and customer_provided_info and not st.session_state.is_customer_verified and show_verification_from_button and not show_draft_ui_check2 and not show_customer_data_ui_check2:
        render_verification_ui(L, customer_has_attachment)

    elif is_login_inquiry and st.session_state.is_customer_verified:
        st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))

    # 에이전트 첨부 파일 업로더
    render_agent_attachments(L)

    # 마이크 녹음 처리
    process_audio_transcription(L)

    # 솔루션 체크박스
    if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
        st.session_state.is_solution_provided = st.checkbox(
            L["solution_check_label"],
            value=st.session_state.is_solution_provided,
            key="solution_checkbox_widget",
        )

    # 메시지 입력 칸 초기화 처리
    if st.session_state.get("reset_agent_response_area", False):
        if not st.session_state.get("last_transcript") or not st.session_state.last_transcript:
            st.session_state.agent_response_area_text = ""
        st.session_state.reset_agent_response_area = False

    # ⭐ 응대 초안 자동 생성
    handle_auto_draft_generation(L)

    # 전사 결과 반영 및 자동 전송
    handle_transcript_auto_send(L)

    # 채팅 입력 UI (카카오톡 스타일)
    agent_response_input = render_chat_input_ui(L, current_lang, perspective)

    # ⭐ 수동 입력 처리
    handle_manual_agent_input(L, agent_response_input)

    # 언어 이관 버튼
    render_language_transfer(L, current_lang)
