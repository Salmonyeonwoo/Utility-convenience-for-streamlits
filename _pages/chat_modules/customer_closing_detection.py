# ========================================
# chat_modules/customer_closing_detection.py
# 고객 종료 응답 감지 모듈
# ========================================

import re
import streamlit as st


def is_agent_actively_asking():
    """상담원이 질문을 하거나 사유/서류 등 추가 정보를 요청 중인지 감지"""
    last_agent_msg = ""
    for msg in reversed(st.session_state.get("simulator_messages", [])):
        if msg.get("role") == "agent_response" and msg.get("content"):
            last_agent_msg = msg.get("content", "")
            break
    if not last_agent_msg:
        return False
    if "?" in last_agent_msg:
        return True
    asking_keywords = [
        "어떤 사정", "사유", "이유", "사정인가요", "증빙", "서류",
        "회신 부탁", "알려주", "말씀해 주", "확인 부탁", "어떻게 되시",
        "무엇인가요", "남겨주", "기종", "예약 번호", "되실까요", "있으실까요"
    ]
    return any(w in last_agent_msg.lower() for w in asking_keywords)


def detect_closing_intent(customer_response, L):
    """고객 종료 의도 감지 (상담원 질문 중일 때는 종료 불가, 오직 솔루션 후 '알겠습니다 감사합니다' 승인 시에만 종료)"""
    # 상담원이 질문/정보 요청 중이면 절대 종료하지 않음
    if is_agent_actively_asking():
        return {
            "has_additional_inquiry_intent": False,
            "has_positive_response": False,
            "has_positive_combination": False,
            "is_positive_closing": False,
            "should_close": False
        }

    # 추가 문의 의도 키워드 확인
    additional_inquiry_keywords = [
        "추가", "더", "또", "그런데", "그리고", "또한", "문의", "질문", "궁금",
        "additional", "more", "also", "but", "and", "question", "inquiry", "wonder",
        "追加", "もっと", "また", "でも", "そして", "質問", "問い合わせ", "疑問"
    ]
    has_additional_inquiry_intent = any(
        keyword in customer_response for keyword in additional_inquiry_keywords
    )
    
    # 명시적 "알겠습니다 + 감사합니다" 승인 및 감사 조합 감지 (단순 "네"는 배제)
    has_positive_combination = (
        not has_additional_inquiry_intent and
        (("알겠습니다" in customer_response or "이해했습니다" in customer_response or 
          "확인했습니다" in customer_response or "承知致しました" in customer_response or 
          "承知いたしました" in customer_response or "了解しました" in customer_response or
          "承知" in customer_response or "了解" in customer_response) and
         ("감사합니다" in customer_response or "ありがとうございます" in customer_response or 
          "ありがとう" in customer_response or
          "thank you" in customer_response.lower() or "thanks" in customer_response.lower()))
    )
    
    # 종료 조건 검토
    escaped_no_more = re.escape(L.get("customer_no_more_inquiries", "다른 문의 사항은 없습니다"))
    no_more_pattern = escaped_no_more.replace(r'\.', r'[.\s]*').replace(r'\ ', r'[.\s]*')
    no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
    
    is_positive_closing = (no_more_regex.search(customer_response) is not None or
                           "다른 문의 사항은 없습니다" in customer_response or
                           "더 궁금한 점은 없습니다" in customer_response)
    
    should_close = (has_positive_combination or is_positive_closing) and not has_additional_inquiry_intent
    
    return {
        "has_additional_inquiry_intent": has_additional_inquiry_intent,
        "has_positive_response": False,
        "has_positive_combination": has_positive_combination,
        "is_positive_closing": is_positive_closing,
        "should_close": should_close
    }


def determine_customer_turn_stage(customer_response, L, closing_intent):
    """고객 턴 단계 결정: 상담원이 질문 중이거나 아직 대화 중이면 무조건 AGENT_TURN 유지"""
    if is_agent_actively_asking():
        st.session_state.is_solution_provided = False
        return "AGENT_TURN"

    is_solution_provided = st.session_state.get("is_solution_provided", False)
    
    # 추가 문의 의도가 있으면 상담 계속
    if closing_intent.get("has_additional_inquiry_intent"):
        return "AGENT_TURN"
    
    # 오직 솔루션/환불 완료 후 고객이 '알겠습니다 감사합니다' 승인할 때만 종료 확인 단계로 전이
    if closing_intent.get("should_close"):
        if is_solution_provided:
            return "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            return "AGENT_TURN"
    
    # 에스컬레이션 요청
    if customer_response.startswith(L.get("customer_escalation_start", "")):
        return "ESCALATION_REQUIRED"
    
    return "AGENT_TURN"
