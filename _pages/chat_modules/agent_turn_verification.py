# ========================================
# _pages/chat_modules/agent_turn_verification.py
# 에이전트 턴 - 고객 검증 프로세스
# ========================================

import streamlit as st
import re
from utils.customer_verification import (
    check_if_login_related_inquiry, check_if_customer_provided_verification_info
)

def process_customer_verification(L):
    """고객 검증 프로세스 처리"""
    customer_has_attachment = (
        st.session_state.customer_attachment_file is not None or
        (st.session_state.sim_attachment_context_for_llm and
         st.session_state.sim_attachment_context_for_llm.strip())
    )

    # 고객 검증 프로세스
    initial_query = st.session_state.get('customer_query_text_area', '')
    all_customer_texts = []
    if initial_query:
        all_customer_texts.append(initial_query)

    if st.session_state.simulator_messages:
        all_roles = [msg.get("role")
                     for msg in st.session_state.simulator_messages]
        customer_messages = [
            msg for msg in st.session_state.simulator_messages if msg.get("role") in [
                "customer", "customer_rebuttal", "initial_query"]]

        for msg in customer_messages:
            content = msg.get("content", "")
            if content and content not in all_customer_texts:
                all_customer_texts.append(content)

        combined_customer_text = " ".join(all_customer_texts)
        is_login_inquiry = check_if_login_related_inquiry(
            combined_customer_text)

        customer_provided_info = check_if_customer_provided_verification_info(
            st.session_state.simulator_messages)

        if customer_has_attachment and is_login_inquiry:
            customer_provided_info = True
            st.session_state.debug_attachment_detected = True

        if not customer_provided_info and is_login_inquiry:
            verification_keywords = [
                "영수증", "receipt", "예약번호", "reservation", "결제", "payment",
                "카드", "card", "계좌", "account", "이메일", "email", "전화", "phone",
                "성함", "이름", "name", "주문번호", "order", "주문", "결제내역",
                "스크린샷", "screenshot", "사진", "photo", "첨부", "attachment", "파일", "file"]
            combined_text_lower = combined_customer_text.lower()
            manual_check = any(
                keyword.lower() in combined_text_lower for keyword in verification_keywords)

            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b'
            has_email = bool(re.search(email_pattern, combined_customer_text))
            has_phone = bool(re.search(phone_pattern, combined_customer_text))

            if customer_has_attachment:
                customer_provided_info = True
                st.session_state.debug_manual_verification_detected = True
                st.session_state.debug_attachment_detected = True
            elif manual_check or has_email or has_phone:
                customer_provided_info = True
                st.session_state.debug_manual_verification_detected = True
                st.session_state.debug_attachment_detected = False
            else:
                st.session_state.debug_manual_verification_detected = False
                st.session_state.debug_attachment_detected = False

            if is_login_inquiry:
                st.session_state.debug_verification_info = customer_provided_info
                st.session_state.debug_all_roles = all_roles
                st.session_state.debug_customer_messages_count = len(
                    customer_messages)
                st.session_state.debug_combined_customer_text = combined_customer_text[:200]
    else:
        is_login_inquiry = check_if_login_related_inquiry(initial_query)
        customer_provided_info = False
        all_roles = []
        customer_messages = []

    return is_login_inquiry, customer_provided_info, customer_has_attachment, all_customer_texts, all_roles, customer_messages

