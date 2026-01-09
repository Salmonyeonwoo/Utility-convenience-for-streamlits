# ========================================
# chat_modules/customer_email_closing.py
# 고객 이메일 종료 처리 모듈
# ========================================

import streamlit as st
import re


def handle_email_closing(customer_response, L, current_lang):
    """메일 응대 종료 처리"""
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
            is_email_closing = any(pattern.lower() in last_agent_response.lower() 
                                  for pattern in email_closing_patterns)
            if is_email_closing:
                st.session_state.has_email_closing = True
    
    if is_email_closing:
        if _check_no_more_inquiry(customer_response, L):
            _add_agent_closing_if_needed(L, current_lang)
            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"


def _check_no_more_inquiry(customer_response, L):
    """추가 문의 없음 확인"""
    no_more_keywords = [
        L['customer_no_more_inquiries'],
        "No, that will be all", "no more", "없습니다", "감사합니다",
        "Thank you", "ありがとうございます", "추가 문의 사항 없습니다",
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
        "감사합니다", "thank you", "ありがとうございます", "좋습니다", "good", "fine", "괜찮습니다"]
    is_positive_response = any(
        keyword.lower() in customer_response.lower() for keyword in positive_keywords)
    
    escaped_check = re.escape(L['customer_no_more_inquiries'])
    no_more_pattern_check = escaped_check.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)
    
    return (has_no_more_inquiry or no_more_regex_check.search(customer_response) or 
            is_positive_response)


def _add_agent_closing_if_needed(L, current_lang):
    """에이전트 감사 인사 추가 (필요한 경우)"""
    agent_closing_added = False
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "agent_response"):
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
        # ⭐ 메시지 추가 후 즉시 화면 업데이트
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)

