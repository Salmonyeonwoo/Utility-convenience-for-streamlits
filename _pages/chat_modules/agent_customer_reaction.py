# ========================================
# chat_modules/agent_customer_reaction.py
# 에이전트 응답 후 고객 반응 생성 모듈
# ========================================

import streamlit as st
from simulation_handler import generate_customer_reaction


def handle_pending_customer_reaction(L):
    """대기 중인 고객 반응 처리"""
    if not st.session_state.get("pending_customer_reaction", False):
        return False
    
    pending_msg_idx = st.session_state.get("pending_customer_reaction_for_msg_idx", -1)
    if pending_msg_idx < 0 or not st.session_state.is_llm_ready:
        return False
    
    try:
        with st.spinner(L.get("generating_customer_response", "고객 응답 생성 중...")):
            customer_response = generate_customer_reaction(
                st.session_state.language, is_call=False)
            customer_message = {"role": "customer", "content": customer_response}
            st.session_state.simulator_messages = st.session_state.simulator_messages + [customer_message]
            
            # ⭐ 메시지 추가 후 즉시 화면 업데이트
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
            
            # 고객 반응 분석 및 상태 업데이트
            _analyze_customer_response(customer_response)
            
            # 다음 단계 결정
            _determine_next_stage_after_reaction(customer_response)
            
            # 플래그 초기화
            st.session_state.pending_customer_reaction = False
            st.session_state.pending_customer_reaction_for_msg_idx = -1
            
            return True
    except Exception as e:
        print(f"고객 반응 생성 오류: {e}")
        st.session_state.pending_customer_reaction = False
        st.session_state.pending_customer_reaction_for_msg_idx = -1
        return False


def _analyze_customer_response(customer_response):
    """고객 응답 분석 및 자동 응답 비활성화 처리"""
    # 고객 불만/직접 응대 요청 감지
    complaint_keywords = [
        "불만", "불만족", "해결 안 됨", "도와주세요", "에이전트", "상담원", "직접",
        "상담원 연결", "직접 상담", "사람과", "complaint", "dissatisfied",
        "not resolved", "help me", "agent", "representative", "directly",
        "speak to", "talk to", "connect me"
    ]
    has_complaint = any(keyword in customer_response.lower() for keyword in complaint_keywords)
    
    # 취소/환불 키워드 확인
    cancellation_keywords = [
        "취소", "환불", "환불해주세요", "취소해주세요", "cancel", "refund",
        "cancel please", "refund please", "キャンセル", "返金"
    ]
    has_cancellation_request = any(keyword in customer_response.lower() for keyword in cancellation_keywords)
    
    # 예외 사유 키워드
    exception_keywords = [
        "비행기 결항", "비행기 지연", "항공편 결항", "항공편 지연", "항공사", "airline",
        "flight cancelled", "flight delayed", "cancelled flight", "delayed flight",
        "날씨", "태풍", "폭설", "weather", "typhoon", "snowstorm",
        "병가", "병원", "입원", "수술", "응급", "긴급", "sick", "hospital", "emergency",
        "medical", "surgery", "urgent", "critical",
        "기기 결함", "제품 결함", "불량품", "오작동", "고장", "작동 안 함", "안 됨", "안돼",
        "defect", "malfunction", "faulty", "broken", "not working", "doesn't work",
        "배송 지연", "배송 오류", "배송 누락", "배송 안 됨", "배송 못 받음", "배송 안 옴",
        "delivery delay", "delivery error", "delivery missing", "late delivery", 
        "wrong delivery", "not delivered", "missing delivery",
        "품질 문제", "품질 불량", "불량", "quality issue", "quality problem", "poor quality",
        "포장 파손", "박스 파손", "상품 파손", "포장 뜯김", "damaged", "broken package",
        "교환", "반품", "exchange", "return", "교환 요청", "반품 요청",
        "불가피", "예외", "특별한 사정", "특수한 경우", "unavoidable", "exceptional",
        "special circumstances", "unforeseen", "unexpected",
        "법적", "정책", "규정", "legal", "policy", "regulation"
    ]
    has_exception = any(keyword in customer_response.lower() for keyword in exception_keywords)
    
    # 만족/해결 키워드 확인
    satisfaction_keywords = [
        "감사합니다", "감사해요", "해결됐어요", "해결되었습니다", "알겠습니다", "좋아요",
        "thank you", "thanks", "resolved", "solved", "ok", "okay", "good",
        "ありがとう", "解決しました", "了解しました"
    ]
    is_satisfied = any(keyword in customer_response.lower() for keyword in satisfaction_keywords)
    
    # 불만이 있거나 직접 응대를 원하는 경우 자동 응답 비활성화
    if has_complaint and not is_satisfied:
        st.session_state.auto_response_disabled = True
        st.session_state.requires_agent_response = True
        print(f"⚠️ 고객 불만/직접 응대 요청 감지: 자동 응답 비활성화")
    elif has_cancellation_request:
        if has_exception:
            print(f"ℹ️ 취소/환불 요청이지만 예외 사유 확인: 자동 응답 유지")
        else:
            st.session_state.auto_response_disabled = True
            st.session_state.requires_agent_response = True
            print(f"⚠️ 취소/환불 요청 감지: 자동 응답 비활성화 (에이전트 직접 응대 필요)")


def _determine_next_stage_after_reaction(customer_response):
    """고객 반응 후 다음 단계 결정"""
    if st.session_state.get("has_email_closing", False):
        positive_keywords = [
            "No, that will be all", "no more", "없습니다", "감사합니다",
            "Thank you", "ありがとうございます", "추가 문의 사항 없습니다",
            "no additional", "追加の質問はありません", "알겠습니다", "ok", "네", "yes"]
        is_positive = any(
            keyword.lower() in customer_response.lower() for keyword in positive_keywords)
        
        if is_positive:
            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
    else:
        st.session_state.sim_stage = "AGENT_TURN"

