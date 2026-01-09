# ========================================
# chat_modules/customer_mode_handler.py
# 고객 모드 입력 처리 모듈
# ========================================

import streamlit as st
from chat_modules.customer_language_detection import detect_and_handle_language_change


def handle_customer_mode_input(L, current_lang, detect_closing_intent_func, determine_customer_turn_stage_func):
    """고객 모드 입력 처리"""
    st.info(L.get("customer_mode_info", "👤 고객 입장에서 AI 상담원에게 응답을 입력하세요."))
    user_customer_input = st.chat_input(
        L.get("customer_inquiry_input_placeholder", "문의 사항을 입력하세요 (고객 입장)..."))
    
    if user_customer_input:
        customer_response = user_customer_input
        current_lang = detect_and_handle_language_change(customer_response, current_lang)
        
        # 메시지 추가
        new_msg = {"role": "customer", "content": user_customer_input}
        st.session_state.simulator_messages.append(new_msg)
        # ⭐ 메시지 추가 후 즉시 화면 업데이트
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
        
        # 종료 의도 감지 및 단계 결정
        closing_intent = detect_closing_intent_func(customer_response, L)
        has_agent_response = any(
            msg.get("role") == "agent_response" 
            for msg in st.session_state.simulator_messages
        )
        is_solution_provided = st.session_state.get("is_solution_provided", False) or has_agent_response
        
        if closing_intent["has_additional_inquiry_intent"]:
            st.session_state.sim_stage = "AGENT_TURN"
            st.session_state.ai_agent_response_generated = False
        elif closing_intent["should_close"]:
            if is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                # ⭐ 메시지 추가 후 즉시 화면 업데이트
                st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
                # ⭐ rerun 제거: 메시지 업데이트 트리거로 자동 반영됨
            else:
                st.session_state.sim_stage = "AGENT_TURN"
                st.session_state.ai_agent_response_generated = False
        elif customer_response.startswith(L["customer_escalation_start"]):
            st.session_state.sim_stage = "ESCALATION_REQUIRED"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
            st.session_state.ai_agent_response_generated = False

