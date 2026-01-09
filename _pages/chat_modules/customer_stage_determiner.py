# ========================================
# chat_modules/customer_stage_determiner.py
# 고객 응답 후 단계 결정 모듈
# ========================================

import streamlit as st
import re


def determine_stage_after_customer_response(customer_response, L, current_lang):
    """고객 응답 후 다음 단계 결정"""
    # 종료 조건 검토
    escaped_no_more = re.escape(L["customer_no_more_inquiries"])
    no_more_pattern = escaped_no_more.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
    escaped_positive = re.escape(L["customer_positive_response"])
    positive_pattern = escaped_positive.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    positive_regex = re.compile(positive_pattern, re.IGNORECASE)
    is_positive_closing = no_more_regex.search(customer_response) is not None or positive_regex.search(customer_response) is not None

    # 메일 응대 종료 문구 확인
    is_email_closing = st.session_state.get("has_email_closing", False)

    if not is_email_closing:
        last_agent_response = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "agent_response" and msg.get("content"):
                last_agent_response = msg.get("content", "")
                break

        email_closing_patterns = [
            "추가 문의사항이 있으면 언제든지 연락",
            "추가 문의 사항이 있으면 언제든지 연락",
            "additional inquiries", "any additional questions",
            "feel free to contact", "please feel free to contact",
            "追加のご質問", "追加のお問い合わせ"]
        if last_agent_response:
            is_email_closing = any(pattern.lower() in last_agent_response.lower() for pattern in email_closing_patterns)
            if is_email_closing:
                st.session_state.has_email_closing = True

    # 메일 끝인사 처리
    if is_email_closing:
        return _handle_email_closing_stage(customer_response, L, current_lang, is_positive_closing, no_more_regex)
    else:
        return _handle_normal_stage(customer_response, L, is_positive_closing, no_more_regex)


def _handle_email_closing_stage(customer_response, L, current_lang, is_positive_closing, no_more_regex):
    """메일 종료 단계 처리"""
    no_more_keywords = [
        L['customer_no_more_inquiries'],
        "No, that will be all", "no more", "없습니다", "감사합니다",
        "Thank you", "ありがとう", "추가 문의 사항 없습니다",
        "no additional", "追加の質問はありません", "알겠습니다", "ok", "네", "yes"]
    has_no_more_inquiry = False
    for keyword in no_more_keywords:
        escaped = re.escape(keyword)
        pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        regex = re.compile(pattern, re.IGNORECASE)
        if regex.search(customer_response):
            has_no_more_inquiry = True
            break
    if "없습니다" in customer_response and "감사합니다" in customer_response:
        has_no_more_inquiry = True

    positive_keywords = [
        "알겠습니다", "알겠어요", "네", "yes", "ok", "okay",
        "감사합니다", "thank you", "ありがとう", "좋습니다", "good", "fine", "괜찮습니다"]
    is_positive_response = any(keyword.lower() in customer_response.lower() for keyword in positive_keywords)

    escaped_check = re.escape(L['customer_no_more_inquiries'])
    no_more_pattern_check = escaped_check.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)
    
    if is_positive_closing or has_no_more_inquiry or no_more_regex_check.search(customer_response) or is_positive_response:
        _add_agent_closing_if_needed(current_lang)
        return "WAIT_CUSTOMER_CLOSING_RESPONSE"
    else:
        return "AGENT_TURN"


def _handle_normal_stage(customer_response, L, is_positive_closing, no_more_regex):
    """일반 단계 처리 (백업 파일의 원본 로직 유지)"""
    if L["customer_positive_response"] in customer_response or ("알겠습니다" in customer_response and "감사합니다" in customer_response):
        if st.session_state.is_solution_provided:
            return "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            return "AGENT_TURN"
    elif is_positive_closing:
        escaped = re.escape(L['customer_no_more_inquiries'])
        no_more_pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
        if no_more_regex.search(customer_response):
            return "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            if st.session_state.is_solution_provided:
                return "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                return "AGENT_TURN"
    elif customer_response.startswith(L.get("customer_escalation_start", "")):
        return "ESCALATION_REQUIRED"
    else:
        return "AGENT_TURN"


def _add_agent_closing_if_needed(current_lang):
    """에이전트 감사 인사 추가 (필요한 경우)"""
    agent_closing_added = False
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "agent_response":
            agent_msg_content = msg.get("content", "")
            if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                agent_closing_added = True
            break

    if not agent_closing_added:
        agent_name = st.session_state.get("agent_name", "000")
        if current_lang == "ko":
            agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
        elif current_lang == "en":
            agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
        else:
            agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

        st.session_state.simulator_messages.append(
            {"role": "agent_response", "content": agent_closing_msg}
        )
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)

