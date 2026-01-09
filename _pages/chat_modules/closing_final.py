# ========================================
# chat_modules/closing_final.py
# 최종 종료 행동 단계 모듈
# ========================================

import streamlit as st
from utils.history_handler import save_simulation_history_local


def render_final_closing_action(L, current_lang):
    """최종 종료 행동 단계 렌더링 (백업 파일의 원본 기능 유지)"""
    st.markdown("---")
    st.success(L["no_more_inquiries_confirmed"])
    st.markdown(f"### {L['consultation_end_header']}")
    st.info(L["click_survey_button_to_end"])
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        end_chat_button = st.button(L["sim_end_chat_button"], key="btn_final_end_chat", 
                                   use_container_width=True, type="primary")

        if end_chat_button:
            st.session_state.start_time = None
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append({"role": "system_end", "content": end_msg})
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"
            customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            st.session_state.realtime_hint_text = ""
