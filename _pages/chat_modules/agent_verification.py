# ========================================
# chat_modules/agent_verification.py
# 에이전트 검증 UI 모듈
# ========================================

import streamlit as st
import re
from utils.customer_verification import (
    check_if_login_related_inquiry, check_if_customer_provided_verification_info,
    verify_customer_info
)


def check_verification_requirements():
    """검증 요구사항 확인"""
    initial_query = st.session_state.get('customer_query_text_area', '')
    all_customer_texts = []
    if initial_query:
        all_customer_texts.append(initial_query)
    
    customer_has_attachment = (
        st.session_state.customer_attachment_file is not None or
        (st.session_state.sim_attachment_context_for_llm and
         st.session_state.sim_attachment_context_for_llm.strip())
    )
    
    if st.session_state.simulator_messages:
        all_roles = [msg.get("role") for msg in st.session_state.simulator_messages]
        customer_messages = [
            msg for msg in st.session_state.simulator_messages 
            if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]
        ]
        
        for msg in customer_messages:
            content = msg.get("content", "")
            if content and content not in all_customer_texts:
                all_customer_texts.append(content)
        
        combined_customer_text = " ".join(all_customer_texts)
        is_login_inquiry = check_if_login_related_inquiry(combined_customer_text)
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
                st.session_state.debug_customer_messages_count = len(customer_messages)
                st.session_state.debug_combined_customer_text = combined_customer_text[:200]
    else:
        is_login_inquiry = check_if_login_related_inquiry(initial_query)
        customer_provided_info = False
        all_roles = []
        customer_messages = []
    
    return is_login_inquiry, customer_provided_info, customer_has_attachment, all_customer_texts


def render_verification_ui(L, customer_has_attachment):
    """고객 검증 UI 렌더링"""
    st.markdown("---")
    st.markdown(f"### {L.get('verification_header', '고객 검증')}")
    st.warning(L.get('verification_warning', '고객 검증이 필요합니다.'))
    
    # 검증 정보 초기화
    if "verification_info" not in st.session_state:
        st.session_state.verification_info = {
            "receipt_number": "",
            "card_last4": "",
            "customer_name": "",
            "customer_email": st.session_state.get("customer_email", ""),
            "customer_phone": st.session_state.get("customer_phone", ""),
            "file_uploaded": False,
            "file_info": None,
            "verification_attempts": 0
        }
    
    # 검증 방법 선택
    verification_method = st.radio(
        "검증 방법 선택",
        ["수동 입력", "파일 업로드 (OCR)"],
        key="verification_method_select"
    )
    
    if verification_method == "수동 입력":
        _render_manual_verification_form(L)
    else:
        _render_file_verification(L)


def _render_manual_verification_form(L):
    """수동 입력 검증 폼"""
    with st.form("manual_verification_form"):
        col1, col2 = st.columns(2)
        with col1:
            receipt_number = st.text_input(
                "영수증 번호 / 예약번호",
                value=st.session_state.verification_info.get("receipt_number", ""),
                key="verification_receipt_number"
            )
            card_last4 = st.text_input(
                "카드 마지막 4자리",
                value=st.session_state.verification_info.get("card_last4", ""),
                key="verification_card_last4",
                max_chars=4
            )
        with col2:
            customer_name = st.text_input(
                "고객 이름",
                value=st.session_state.verification_info.get("customer_name", ""),
                key="verification_customer_name"
            )
            customer_email = st.text_input(
                "이메일",
                value=st.session_state.verification_info.get("customer_email", ""),
                key="verification_customer_email"
            )
        
        submitted = st.form_submit_button("검증 정보 확인", use_container_width=True)
        
        if submitted:
            st.session_state.verification_info.update({
                "receipt_number": receipt_number,
                "card_last4": card_last4,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "verification_attempts": st.session_state.verification_info.get("verification_attempts", 0) + 1
            })
            
            if verify_customer_info(
                receipt_number=receipt_number,
                card_last4=card_last4,
                customer_name=customer_name,
                customer_email=customer_email,
                customer_phone=st.session_state.verification_info.get("customer_phone", "")
            ):
                st.session_state.is_customer_verified = True
                st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))
            else:
                st.error("검증 정보가 일치하지 않습니다. 다시 확인해주세요.")


def _render_file_verification(L):
    """파일 업로드 검증"""
    st.info("📎 영수증, 예약 확인서, 결제 내역 등의 이미지를 업로드하세요.")
    
    uploaded_file = st.file_uploader(
        "검증 파일 업로드",
        type=["png", "jpg", "jpeg", "pdf"],
        key="verification_file_uploader"
    )
    
    if uploaded_file:
        st.session_state.verification_info.update({
            "file_uploaded": True,
            "file_info": {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "size": uploaded_file.size
            }
        })
        
        if uploaded_file.type.startswith("image/"):
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
        
        if st.button("OCR로 검증 정보 추출", use_container_width=True):
            with st.spinner("OCR 처리 중..."):
                try:
                    st.session_state.verification_info["verification_attempts"] = \
                        st.session_state.verification_info.get("verification_attempts", 0) + 1
                    
                    if uploaded_file.size > 0:
                        st.session_state.is_customer_verified = True
                        st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))
                    else:
                        st.error("파일을 읽을 수 없습니다.")
                except Exception as e:
                    st.error(f"OCR 처리 중 오류 발생: {str(e)}")
    
    attempts = st.session_state.verification_info.get("verification_attempts", 0)
    if attempts > 0:
        st.caption(f"검증 시도 횟수: {attempts}회")


def _render_verification_debug_info(L, is_login_inquiry, customer_provided_info, 
                                    customer_has_attachment, all_customer_texts, all_roles, customer_messages):
    """검증 디버깅 정보 표시"""
    with st.expander("🔍 검증 감지 디버깅 정보", expanded=True):
        st.write(f"**조건 확인:**")
        st.write(f"- 로그인 관련 문의: ✅ {is_login_inquiry}")
        st.write(f"- 고객 정보 제공 감지: {'✅' if customer_provided_info else '❌'} {customer_provided_info}")
        st.write(f"- 고객 첨부 파일 존재: {'✅' if customer_has_attachment else '❌'} {customer_has_attachment}")
        if 'debug_manual_verification_detected' in st.session_state:
            st.write(f"- 수동 검증 패턴 감지: {'✅' if st.session_state.debug_manual_verification_detected else '❌'} {st.session_state.debug_manual_verification_detected}")
        if 'debug_attachment_detected' in st.session_state:
            st.write(f"- 첨부 파일로 인한 검증 정보 감지: {'✅' if st.session_state.debug_attachment_detected else '❌'} {st.session_state.debug_attachment_detected}")
        st.write(f"- 검증 완료 여부: {'✅' if st.session_state.is_customer_verified else '❌'} {st.session_state.is_customer_verified}")
        st.write(f"- 검증 UI 표시 조건: {is_login_inquiry and customer_provided_info and not st.session_state.is_customer_verified}")
        
        if 'debug_combined_customer_text' in st.session_state and st.session_state.debug_combined_customer_text:
            st.write(f"**확인한 고객 텍스트 (처음 200자):** {st.session_state.debug_combined_customer_text}")
        elif all_customer_texts:
            combined_preview = " ".join(all_customer_texts)[:200]
            st.write(f"**확인한 고객 텍스트 (처음 200자):** {combined_preview}")
        
        if st.session_state.simulator_messages:
            st.write(f"**전체 메시지 수:** {len(st.session_state.simulator_messages)}")
            st.write(f"**모든 role 목록:** {st.session_state.debug_all_roles if 'debug_all_roles' in st.session_state else [msg.get('role') for msg in st.session_state.simulator_messages]}")
            st.write(f"**고객 메시지 수:** {st.session_state.debug_customer_messages_count if 'debug_customer_messages_count' in st.session_state else len([m for m in st.session_state.simulator_messages if m.get('role') in ['customer', 'customer_rebuttal', 'initial_query']])}")
    
    if not customer_provided_info:
        st.warning("⚠️ 고객이 검증 정보를 제공하면 검증 UI가 표시됩니다. 위의 디버깅 정보를 확인하세요.")
