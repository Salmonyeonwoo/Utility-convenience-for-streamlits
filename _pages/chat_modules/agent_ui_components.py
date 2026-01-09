# ========================================
# chat_modules/agent_ui_components.py
# 에이전트 UI 컴포넌트 모듈
# ========================================

import streamlit as st
from utils.history_handler import (
    generate_chat_summary, load_simulation_histories_local,
    recommend_guideline_for_customer
)


def render_ui_header(L):
    """UI 헤더 렌더링"""
    show_verification_from_button = st.session_state.get("show_verification_ui", False)
    show_draft_ui = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui = st.session_state.get("show_customer_data_ui", False)
    
    if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
        st.markdown(f"### {L['agent_response_header']}")


def render_customer_guideline(L):
    """고객 성향 기반 가이드라인 추천"""
    if st.session_state.simulator_messages and len(st.session_state.simulator_messages) >= 2:
        try:
            temp_summary = generate_chat_summary(
                st.session_state.simulator_messages,
                st.session_state.customer_query_text_area,
                st.session_state.get("customer_type_sim_select", ""),
                st.session_state.language
            )
            
            if temp_summary and temp_summary.get("customer_sentiment_score"):
                all_histories = load_simulation_histories_local(st.session_state.language)
                recommended_guideline = recommend_guideline_for_customer(
                    temp_summary, all_histories, st.session_state.language
                )
                
                if recommended_guideline:
                    with st.expander("💡 고객 성향 기반 응대 가이드라인 추천", expanded=False):
                        st.markdown(recommended_guideline)
                        st.caption("💡 이 가이드는 유사한 과거 고객 사례를 분석하여 자동 생성되었습니다.")
        except Exception:
            pass


def render_language_transfer_warning(L):
    """언어 이관 요청 경고 표시"""
    if st.session_state.language_transfer_requested:
        st.error(L.get("language_transfer_requested_msg",
                      "🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。"))


def render_customer_attachment_info():
    """고객 첨부 파일 정보 표시"""
    if st.session_state.sim_attachment_context_for_llm:
        st.info(f"📎 최초 문의 시 첨부된 파일 정보:\n\n"
                f"{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")


def render_agent_file_uploader(L):
    """에이전트 파일 업로더 렌더링"""
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
            file_names = ", ".join([f["name"] for f in st.session_state.agent_attachment_file])
            st.info(L.get("agent_attachment_files_ready",
                         "✅ {count}개 에이전트 첨부 파일 준비 완료: {files}").format(
                count=len(agent_attachment_files), files=file_names))
            st.session_state.show_agent_file_uploader = False
        else:
            st.session_state.agent_attachment_file = []
    else:
        st.session_state.agent_attachment_file = []


def render_solution_checkbox(L):
    """솔루션 체크박스 렌더링"""
    show_verification_from_button = st.session_state.get("show_verification_ui", False)
    show_draft_ui = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui = st.session_state.get("show_customer_data_ui", False)
    
    if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
        st.session_state.is_solution_provided = st.checkbox(
            L["solution_check_label"],
            value=st.session_state.is_solution_provided,
            key="solution_checkbox_widget",
        )


def handle_transcript_auto_send(L):
    """전사 결과 자동 전송 처리"""
    if st.session_state.get("last_transcript") and st.session_state.last_transcript:
        agent_response_auto = st.session_state.last_transcript.strip()
        if agent_response_auto:
            new_message = {"role": "agent_response", "content": agent_response_auto}
            st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
            # ⭐ 메시지 추가 후 즉시 화면 업데이트
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
            
            st.session_state.last_transcript = ""
            st.session_state.agent_response_area_text = ""
            st.session_state.auto_draft_generated = False
            
            if st.session_state.is_llm_ready:
                st.session_state.pending_customer_reaction = True
                st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1
            return True
    return False


def render_language_transfer_buttons(L, current_lang):
    """언어 이관 버튼 렌더링"""
    from chat_modules.language_transfer import handle_language_transfer
    from lang_pack import LANG
    
    st.markdown("---")
    st.markdown(f"**{L['transfer_header']}**")
    transfer_cols = st.columns(len(LANG) - 1)
    
    languages = list(LANG.keys())
    languages.remove(current_lang)
    
    for idx, lang_code in enumerate(languages):
        lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang_code, lang_code)
        transfer_label = L.get(f"transfer_to_{lang_code}", f"Transfer to {lang_name} Team")
        
        with transfer_cols[idx]:
            if st.button(transfer_label, key=f"btn_transfer_{lang_code}_{st.session_state.sim_instance_id}", use_container_width=True):
                handle_language_transfer(lang_code, st.session_state.simulator_messages, L, current_lang)

