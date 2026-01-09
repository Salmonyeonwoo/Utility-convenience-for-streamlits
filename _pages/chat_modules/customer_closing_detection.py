# ========================================
# chat_modules/customer_closing_detection.py
# 고객 종료 응답 감지 모듈
# ========================================

import re
import streamlit as st


def detect_closing_intent(customer_response, L):
    """고객 종료 의도 감지"""
    # 추가 문의 의도 키워드 확인
    additional_inquiry_keywords = [
        "추가", "더", "또", "그런데", "그리고", "또한", "문의", "질문", "궁금",
        "additional", "more", "also", "but", "and", "question", "inquiry", "wonder",
        "追加", "もっと", "また", "でも", "そして", "質問", "問い合わせ", "疑問"
    ]
    has_additional_inquiry_intent = any(
        keyword in customer_response for keyword in additional_inquiry_keywords
    )
    
    # 긍정적 종료 응답 감지
    positive_response_keywords = [
        L["customer_positive_response"],
        "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", 
        "承知致しました", "承知いたしました", "了解しました"
    ]
    has_positive_response = any(
        keyword.lower() in customer_response.lower() 
        for keyword in positive_response_keywords
    )
    
    # "알겠습니다" + "감사합니다" 조합 감지
    has_positive_combination = (
        not has_additional_inquiry_intent and
        (("알겠습니다" in customer_response or "알겠어요" in customer_response or 
         "承知致しました" in customer_response or "承知いたしました" in customer_response or
         "了解しました" in customer_response or
         "yes" in customer_response.lower() or "ok" in customer_response.lower() or "okay" in customer_response.lower() or
         "承知" in customer_response or "了解" in customer_response) and
        ("감사합니다" in customer_response or "ありがとうございます" in customer_response or 
         "ありがとう" in customer_response or
         "thank you" in customer_response.lower() or "thanks" in customer_response.lower() or "thank" in customer_response.lower()))
    )
    
    # 종료 조건 검토
    escaped_no_more = re.escape(L["customer_no_more_inquiries"])
    no_more_pattern = escaped_no_more.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
    
    escaped_positive = re.escape(L["customer_positive_response"])
    positive_pattern = escaped_positive.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    positive_regex = re.compile(positive_pattern, re.IGNORECASE)
    
    is_positive_closing = (no_more_regex.search(customer_response) is not None or 
                          positive_regex.search(customer_response) is not None)
    
    return {
        "has_additional_inquiry_intent": has_additional_inquiry_intent,
        "has_positive_response": has_positive_response,
        "has_positive_combination": has_positive_combination,
        "is_positive_closing": is_positive_closing,
        "should_close": (L["customer_positive_response"] in customer_response or 
                        has_positive_response or has_positive_combination or is_positive_closing)
    }


def determine_customer_turn_stage(customer_response, L, closing_intent):
    """고객 턴 단계 결정"""
    has_agent_response = any(
        msg.get("role") == "agent_response" 
        for msg in st.session_state.simulator_messages
    )
    is_solution_provided = st.session_state.get("is_solution_provided", False) or has_agent_response
    
    # 추가 문의 의도가 있으면 엔딩으로 가지 않음
    if closing_intent["has_additional_inquiry_intent"]:
        return "AGENT_TURN"
    
    # 종료 의도가 있는 경우
    if closing_intent["should_close"]:
        if is_solution_provided:
            return "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            return "AGENT_TURN"
    
    # 에스컬레이션 요청
    if customer_response.startswith(L["customer_escalation_start"]):
        return "ESCALATION_REQUIRED"
    
    return "AGENT_TURN"

