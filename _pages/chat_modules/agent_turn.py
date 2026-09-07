# 이 모듈은 _chat_simulator.py에서 분리된 부분입니다
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
from simulation_handler import *
from visualization import *
from audio_handler import *
from llm_client import get_api_key
from typing import List, Dict, Any
import uuid
import time
import os

    # 5. 에이전트 입력 단계 (AGENT_TURN) - ⭐ 수정: 원위치 복원 - 항상 입력 칸 표시
    # =========================
    # ⭐ 수정: AGENT_TURN 단계에서 항상 에이전트 응답 입력 UI를 표시 (원위치 복원)
    # app.py 스타일: AGENT_TURN 단계에서 항상 입력 칸이 보이도록 함
    # 단, 검증 UI나 응대 초안 UI가 표시될 때는 에이전트 응답 UI를 숨김
    if st.session_state.sim_stage == "AGENT_TURN":
        # ⭐ 수정: app.py 스타일 - 플래그 기반 처리 제거, 단순한 흐름 유지
        # 메시지 전송은 위의 agent_response 처리 부분에서 직접 처리됨
        show_verification_from_button = st.session_state.get("show_verification_ui", False)
        show_draft_ui = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui = st.session_state.get("show_customer_data_ui", False)
        
        # 각 기능이 표시될 때는 해당 기능의 헤더만 표시
        if show_verification_from_button:
            # 고객 검증 헤더는 아래에서 표시됨
            pass
        elif show_draft_ui:
            # 응대 초안은 메시지로 표시되므로 헤더 불필요
            pass
        elif show_customer_data_ui:
            # 데이터 가져오기는 메시지로 표시되므로 헤더 불필요
            pass
        else:
            # 기본 에이전트 응답 헤더는 메시지 입력 칸 바로 위에 표시 (아래로 이동)
            pass  # 헤더는 입력 칸 바로 위로 이동

        # ⭐ 실시간 응대 힌트 영역 제거 (메시지 말풍선에 버튼으로 이동)
        # 힌트는 에이전트 응답 메시지 말풍선의 '응대 힌트' 버튼을 통해 사용할 수 있습니다.
        
        # ⭐ 추가: 고객 성향 기반 가이드라인 추천 (신규 고객 문의 시)
        if st.session_state.simulator_messages and len(st.session_state.simulator_messages) >= 2:
            # 고객 메시지가 있고 요약이 생성 가능한 경우
            try:
                # 현재 대화를 임시 요약하여 고객 성향 분석
                temp_summary = generate_chat_summary(
                    st.session_state.simulator_messages,
                    st.session_state.customer_query_text_area,
                    st.session_state.get("customer_type_sim_select", ""),
                    st.session_state.language
                )
                
                if temp_summary and temp_summary.get("customer_sentiment_score"):
                    # 과거 이력 로드
                    all_histories = load_simulation_histories_local(st.session_state.language)
                    
                    # 가이드라인 추천 생성
                    recommended_guideline = recommend_guideline_for_customer(
                        temp_summary,
                        all_histories,
                        st.session_state.language
                    )
                    
                    if recommended_guideline:
                        with st.expander("💡 고객 성향 기반 응대 가이드라인 추천", expanded=False):
                            st.markdown(recommended_guideline)
                            st.caption("💡 이 가이드는 유사한 과거 고객 사례를 분석하여 자동 생성되었습니다.")
            except Exception as e:
                # 가이드라인 추천 실패 시 무시 (비차단)
                pass

        # --- 언어 이관 요청 강조 표시 ---
        if st.session_state.language_transfer_requested:
            st.error(L.get("language_transfer_requested_msg", "🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。"))

        # --- 고객 첨부 파일 정보 재표시 ---
        if st.session_state.sim_attachment_context_for_llm:
            st.info(
                f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")
        
        # 고객 첨부 파일이 있는지 확인 (검증 프로세스에서 사용)
        customer_has_attachment = (
            st.session_state.customer_attachment_file is not None or 
            (st.session_state.sim_attachment_context_for_llm and 
             st.session_state.sim_attachment_context_for_llm.strip())
        )

        # --- 고객 검증 프로세스 (로그인/계정 관련 문의이고 고객이 정보를 제공한 경우) ---
        # 개선: 초기 쿼리뿐만 아니라 모든 고객 메시지에서 로그인 관련 문의 확인
        initial_query = st.session_state.get('customer_query_text_area', '')
        
        # 모든 고객 메시지 수집 (초기 쿼리 포함)
        all_customer_texts = []
        if initial_query:
            all_customer_texts.append(initial_query)
        
        if st.session_state.simulator_messages:
            # 디버깅: 메시지 확인
            all_roles = [msg.get("role") for msg in st.session_state.simulator_messages]
            customer_messages = [msg for msg in st.session_state.simulator_messages if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]]
            
            # 모든 고객 메시지의 내용 수집
            for msg in customer_messages:
                content = msg.get("content", "")
                if content and content not in all_customer_texts:
                    all_customer_texts.append(content)
            
            # 모든 고객 메시지를 합쳐서 로그인 관련 문의 확인
            combined_customer_text = " ".join(all_customer_texts)
            is_login_inquiry = check_if_login_related_inquiry(combined_customer_text)
            
            # 고객이 검증 정보를 제공했는지 확인
            customer_provided_info = check_if_customer_provided_verification_info(st.session_state.simulator_messages)
            
            # 고객이 첨부 파일을 제공한 경우 검증 정보 제공으로 간주
            if customer_has_attachment and is_login_inquiry:
                customer_provided_info = True
                st.session_state.debug_attachment_detected = True
            
            # 보조 검증: 함수 결과가 False인 경우에도 직접 패턴 확인 (디버깅 및 보완)
            if not customer_provided_info and is_login_inquiry:
                # 고객 메시지에서 검증 정보 패턴 직접 확인
                verification_keywords = [
                    "영수증", "receipt", "예약번호", "reservation", "결제", "payment",
                    "카드", "card", "계좌", "account", "이메일", "email", "전화", "phone",
                    "성함", "이름", "name", "주문번호", "order", "주문", "결제내역",
                    "스크린샷", "screenshot", "사진", "photo", "첨부", "attachment", "파일", "file"
                ]
                combined_text_lower = combined_customer_text.lower()
                manual_check = any(keyword.lower() in combined_text_lower for keyword in verification_keywords)
                
                # 이메일이나 전화번호 패턴 확인
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                phone_pattern = r'\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b'
                has_email = bool(re.search(email_pattern, combined_customer_text))
                has_phone = bool(re.search(phone_pattern, combined_customer_text))
                
                # 고객이 첨부 파일을 제공한 경우도 검증 정보 제공으로 간주
                if customer_has_attachment:
                    customer_provided_info = True
                    st.session_state.debug_manual_verification_detected = True
                    st.session_state.debug_attachment_detected = True
                # 수동 확인 결과도 고려 (더 관대한 검증)
                elif manual_check or has_email or has_phone:
                    customer_provided_info = True
                    st.session_state.debug_manual_verification_detected = True
                    st.session_state.debug_attachment_detected = False
                else:
                    st.session_state.debug_manual_verification_detected = False
                    st.session_state.debug_attachment_detected = False
            
            # 디버깅용: 정보 제공 여부 확인
            if is_login_inquiry:
                st.session_state.debug_verification_info = customer_provided_info
                st.session_state.debug_all_roles = all_roles
                st.session_state.debug_customer_messages_count = len(customer_messages)
                st.session_state.debug_combined_customer_text = combined_customer_text[:200]  # 처음 200자만 저장
        else:
            # 메시지가 없는 경우 초기 쿼리만 확인
            is_login_inquiry = check_if_login_related_inquiry(initial_query)
            customer_provided_info = False
            all_roles = []
            customer_messages = []
        
        # ⭐ 수정: 검증 UI는 고객 메시지 버튼 클릭 시에만 표시 (기존 자동 표시 제거)
        # 로그인 관련 문의이고, 고객이 정보를 제공했으며, 아직 검증되지 않은 경우
        # 그리고 고객 메시지에서 검증 버튼을 클릭한 경우에만 검증 UI 표시
        # show_verification_from_button은 위에서 이미 정의됨
        
        # ⭐ 고객 검증 UI 표시 (버튼 클릭 시에만, 다른 기능이 표시되지 않을 때만)
        show_draft_ui_check = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui_check = st.session_state.get("show_customer_data_ui", False)
        if show_verification_from_button and not show_draft_ui_check and not show_customer_data_ui_check:
            st.markdown("---")
            st.markdown(f"### {L.get('verification_header', '고객 검증')}")
            st.warning(L.get('verification_warning', '고객 검증이 필요합니다.'))
        
        # 디버깅: 조건 확인 (기존 유지하되, 자동 표시는 제거)
        if is_login_inquiry and show_verification_from_button:
            # 디버깅 정보 표시 (항상 표시)
            with st.expander(L.get("verification_debug_header", "🔍 검증 감지 디버깅 정보"), expanded=True):
                st.write(f"**{L.get('verification_debug_condition_check', '조건 확인')}:**")
                st.write(f"- {L.get('verification_debug_login_inquiry', '로그인 관련 문의')}: ✅ {is_login_inquiry}")
                st.write(f"- {L.get('verification_debug_customer_info_provided', '고객 정보 제공 감지')}: {'✅' if customer_provided_info else '❌'} {customer_provided_info}")
                st.write(f"- {L.get('verification_debug_customer_attachment_exists', '고객 첨부 파일 존재')}: {'✅' if customer_has_attachment else '❌'} {customer_has_attachment}")
                if 'debug_manual_verification_detected' in st.session_state:
                    st.write(f"- {L.get('verification_debug_manual_pattern_detected', '수동 검증 패턴 감지')}: {'✅' if st.session_state.debug_manual_verification_detected else '❌'} {st.session_state.debug_manual_verification_detected}")
                if 'debug_attachment_detected' in st.session_state:
                    st.write(f"- {L.get('verification_debug_attachment_detected', '첨부 파일로 인한 검증 정보 감지')}: {'✅' if st.session_state.debug_attachment_detected else '❌'} {st.session_state.debug_attachment_detected}")
                st.write(f"- {L.get('verification_debug_verification_completed', '검증 완료 여부')}: {'✅' if st.session_state.is_customer_verified else '❌'} {st.session_state.is_customer_verified}")
                st.write(f"- {L.get('verification_debug_ui_display_condition', '검증 UI 표시 조건')}: {is_login_inquiry and customer_provided_info and not st.session_state.is_customer_verified}")
                
                # 확인한 텍스트 정보 표시
                if 'debug_combined_customer_text' in st.session_state and st.session_state.debug_combined_customer_text:
                    st.write(f"**{L.get('verification_debug_customer_text_preview', '확인한 고객 텍스트 (처음 200자)')}:** {st.session_state.debug_combined_customer_text}")
                elif all_customer_texts:
                    combined_preview = " ".join(all_customer_texts)[:200]
                    st.write(f"**{L.get('verification_debug_customer_text_preview', '확인한 고객 텍스트 (처음 200자)')}:** {combined_preview}")
                
                if st.session_state.simulator_messages:
                    st.write(f"**{L.get('verification_debug_total_messages', '전체 메시지 수')}:** {len(st.session_state.simulator_messages)}")
                    st.write(f"**{L.get('verification_debug_all_roles', '모든 role 목록')}:** {st.session_state.debug_all_roles if 'debug_all_roles' in st.session_state else [msg.get('role') for msg in st.session_state.simulator_messages]}")
                    st.write(f"**{L.get('verification_debug_customer_messages_count', '고객 메시지 수')}:** {st.session_state.debug_customer_messages_count if 'debug_customer_messages_count' in st.session_state else len([m for m in st.session_state.simulator_messages if m.get('role') in ['customer', 'customer_rebuttal', 'initial_query']])}")
                    
                    # ⭐ 추가: 고객 데이터 정보 표시 (app.py 스타일)
                    if st.session_state.customer_data:
                        customer_info = st.session_state.customer_data.get("data", {})
                        st.write(f"**{L.get('customer_data_label', '고객 데이터')}:** ✅ {L.get('loaded', '불러옴')}")
                        st.write(f"- {L.get('name_label', '이름')}: {customer_info.get('name', 'N/A')}")
                        st.write(f"- {L.get('email_label', '이메일')}: {customer_info.get('email', 'N/A')}")
                        st.write(f"- {L.get('phone_label', '전화번호')}: {customer_info.get('phone', 'N/A')}")
                        if customer_info.get('purchase_history'):
                            st.write(f"- {L.get('purchase_history_label', '구매 이력')}: {len(customer_info.get('purchase_history', []))}{L.get('cases_label', '건')}")
                    else:
                        st.write(f"**{L.get('customer_data_label', '고객 데이터')}:** ❌ {L.get('none', '없음')}")
                    
                    # ⭐ 추가: 누적 데이터 수 자동 확인 (고객 데이터 매니저에서)
                    try:
                        all_customers = st.session_state.customer_data_manager.list_all_customers()
                        st.write(f"**{L.get('accumulated_customer_data_label', '누적 고객 데이터 수')}:** {len(all_customers)}{L.get('cases_label', '건')}")
                    except Exception:
                        st.write(f"**{L.get('accumulated_customer_data_label', '누적 고객 데이터 수')}:** {L.get('unavailable', '확인 불가')}")
                    
                    # 모든 메시지 표시 (최근 10개)
                    st.write(f"**{L.get('verification_debug_recent_messages', '최근 모든 메시지 (최근 10개)')}:**")
                    for i, msg in enumerate(st.session_state.simulator_messages[-10:], 1):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")[:300]
                        st.write(f"{i}. [{role}] {content}")
                    
                    # 고객 메시지만 필터링하여 표시
                    customer_messages = [
                        {"role": msg.get("role"), "content": msg.get("content", "")[:300]} 
                        for msg in st.session_state.simulator_messages[-10:] 
                        if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]
                    ]
                    st.write(f"**{L.get('verification_debug_customer_messages_only', '고객 메시지만 (최근 10개)')}:**")
                    if customer_messages:
                        for i, msg in enumerate(customer_messages, 1):
                            st.write(f"{i}. [{msg['role']}] {msg['content']}")
                    else:
                        st.write(L.get("verification_debug_no_customer_messages", "고객 메시지 없음"))
                else:
                    st.write(f"**{L.get('verification_debug_no_messages', '메시지 없음')}**")
            
            if not customer_provided_info:
                # 정보가 아직 제공되지 않은 경우 안내 메시지 표시
                st.warning(L.get("verification_info_provided_warning", "⚠️ 고객이 검증 정보를 제공하면 검증 UI가 표시됩니다. 위의 디버깅 정보를 확인하세요."))
        
        # ⭐ 수정: 검증 UI는 고객 메시지 버튼 클릭 시에만 표시
        # 고객 데이터 정보를 디버깅 정보에 포함
        # 다른 기능이 표시되지 않을 때만 검증 UI 표시
        # ⭐ 개선: 버튼 클릭 시 항상 검증 UI 표시 (customer_provided_info 조건 완화)
        show_draft_ui_check2 = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui_check2 = st.session_state.get("show_customer_data_ui", False)
        # 검증 버튼을 클릭했고, 아직 검증되지 않았으며, 다른 UI가 표시되지 않을 때 검증 UI 표시
        if show_verification_from_button and not st.session_state.is_customer_verified and not show_draft_ui_check2 and not show_customer_data_ui_check2:
            # 헤더는 위에서 이미 표시했으므로 중복 제거
            
            # 고객 데이터 표시 (있는 경우)
            if st.session_state.customer_data:
                customer_info = st.session_state.customer_data.get("data", {})
                with st.expander(L.get("customer_data_info_expander", "📋 고객 데이터 정보"), expanded=False):
                    st.json(customer_info)
                    # 누적 데이터 수 표시
                    try:
                        all_customers = st.session_state.customer_data_manager.list_all_customers()
                        st.caption(f"📊 누적 고객 데이터: {len(all_customers)}건")
                    except Exception:
                        pass
            
            with st.expander(L.get("verification_info_input", "고객 검증 정보 입력"), expanded=True):
                # 고객이 처음에 첨부한 파일 표시
                if customer_has_attachment:
                    if st.session_state.customer_attachment_file:
                        attachment_file = st.session_state.customer_attachment_file
                        st.success(L.get("customer_initial_attachment", "📎 고객이 처음에 첨부한 파일: **{filename}** ({size} bytes, {type})").format(filename=attachment_file.name, size=attachment_file.size, type=attachment_file.type))
                        # 고객 첨부 파일을 검증 파일로도 사용 가능하도록 설정
                        if 'verification_file_info' not in st.session_state or not st.session_state.verification_file_info:
                            st.session_state.verification_file_info = {
                                "filename": attachment_file.name,
                                "size": attachment_file.size,
                                "type": attachment_file.type,
                                "source": "customer_initial_attachment"
                            }
                    elif st.session_state.sim_attachment_context_for_llm:
                        st.info(L.get("customer_attachment_info", "📎 고객이 첨부한 파일 정보: {info}").format(info=st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()))
                
                st.markdown("---")
                st.write(f"**{L.get('additional_verification_file_upload', '추가 검증 파일 업로드 (선택사항)')}**")
                # 파일 업로더 (스크린샷/사진 스캔용) - 추가 파일 업로드 가능
                verification_file = st.file_uploader(
                    L.get("verification_file_upload_label", "검증 파일 업로드 (스크린샷/사진)"),
                    type=["png", "jpg", "jpeg", "pdf"],
                    key="verification_file_uploader",
                    help=L.get("verification_file_upload_help", "고객이 제공한 영수증, 예약 확인서, 결제 내역 등의 스크린샷/사진을 추가로 업로드하세요. (고객이 처음에 첨부한 파일이 있으면 자동으로 포함됩니다.)")
                )
                
                # 검증에 사용할 파일 결정 (고객 첨부 파일 우선, 없으면 새로 업로드한 파일)
                file_to_verify = None
                file_verified = False
                ocr_extracted_info = {}  # OCR로 추출된 정보 저장
                
                if customer_has_attachment and st.session_state.customer_attachment_file:
                    file_to_verify = st.session_state.customer_attachment_file
                    file_verified = True
                    st.info(L.get("verification_file_using_customer_attachment", "✅ 검증에 사용할 파일: **{filename}** (고객이 처음에 첨부한 파일)").format(filename=file_to_verify.name))
                elif verification_file:
                    file_to_verify = verification_file
                    file_verified = True
                    st.info(L.get("file_upload_complete", "✅ 파일 업로드 완료: {filename} ({size} bytes)").format(filename=verification_file.name, size=verification_file.size))
                    # 파일 정보를 세션 상태에 저장
                    st.session_state.verification_file_info = {
                        "filename": verification_file.name,
                        "size": verification_file.size,
                        "type": verification_file.type,
                        "source": "verification_uploader"
                    }
                elif customer_has_attachment:
                    # 첨부 파일 정보만 있고 파일 객체는 없는 경우 (이전 세션에서 업로드)
                    file_verified = True  # 파일이 있었다는 정보만으로도 검증 가능
                    st.info(L.get("customer_attachment_info_confirmed", "✅ 고객이 첨부한 파일 정보가 확인되었습니다."))
                
                # OCR 기능: 파일이 업로드되면 자동으로 정보 추출
                if file_to_verify and file_to_verify.name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    if 'ocr_extracted_info' not in st.session_state or st.session_state.get('ocr_file_name') != file_to_verify.name:
                        with st.spinner(L.get("extracting_info_from_screenshot", "🔍 스크린샷에서 정보 추출 중 (OCR)...")):
                            try:
                                # 파일 읽기
                                file_to_verify.seek(0)
                                file_bytes = file_to_verify.getvalue()
                                file_type = file_to_verify.type
                                
                                # Gemini Vision API를 사용한 OCR
                                gemini_key = get_api_key("gemini")
                                if gemini_key:
                                    import google.generativeai as genai
                                    genai.configure(api_key=gemini_key)
                                    model = genai.GenerativeModel('gemini-2.5-flash')
                                    
                                    # 검증 정보 추출을 위한 특화 프롬프트
                                    ocr_verification_prompt = """이 이미지는 고객 검증을 위한 스크린샷입니다. 다음 정보를 추출해주세요:

    1. 영수증 번호 또는 예약 번호 (Receipt/Reservation Number)
    2. 고객 성함 (Customer Name)
    3. 고객 이메일 (Customer Email)
    4. 고객 전화번호 (Customer Phone)
    5. 결제 수단 (Payment Method: 신용카드, 체크카드, 카카오페이, 네이버페이, 온라인뱅킹 등)
    6. 카드 뒷자리 4자리 (Card Last 4 Digits) - 있는 경우
    7. 계좌번호 (Account Number) - 있는 경우

    각 정보를 JSON 형식으로 반환해주세요:
    {
  "receipt_number": "추출된 영수증/예약 번호 또는 빈 문자열",
  "customer_name": "추출된 고객 성함 또는 빈 문자열",
  "customer_email": "추출된 이메일 주소 또는 빈 문자열",
  "customer_phone": "추출된 전화번호 또는 빈 문자열",
  "payment_method": "추출된 결제 수단 또는 빈 문자열",
  "card_last4": "추출된 카드 뒷자리 4자리 또는 빈 문자열",
  "account_number": "추출된 계좌번호 또는 빈 문자열"
    }

    정보가 없으면 빈 문자열("")로 반환하세요. JSON 형식만 반환하고 다른 설명은 추가하지 마세요."""
                                    
                                    if file_to_verify.name.lower().endswith('.pdf'):
                                        # PDF는 텍스트 추출 후 OCR
                                        import tempfile
                                        import os
                                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                                        tmp.write(file_bytes)
                                        tmp.flush()
                                        tmp.close()
                                        try:
                                            loader = PyPDFLoader(tmp.name)
                                            file_docs = loader.load()
                                            pdf_text = "\n".join([doc.page_content for doc in file_docs])
                                            # PDF 텍스트가 있으면 그대로 사용, 없으면 이미지로 처리
                                            if pdf_text.strip():
                                                response = model.generate_content(f"{ocr_verification_prompt}\n\n추출된 텍스트:\n{pdf_text}")
                                            else:
                                                # PDF를 이미지로 변환하여 처리 (간단한 경우 텍스트만 사용)
                                                response = model.generate_content([
                                                    {"mime_type": "application/pdf", "data": file_bytes},
                                                    ocr_verification_prompt
                                                ])
                                        finally:
                                            try:
                                                os.remove(tmp.name)
                                            except:
                                                pass
                                    else:
                                        # 이미지 파일
                                        response = model.generate_content([
                                            {"mime_type": file_type, "data": file_bytes},
                                            ocr_verification_prompt
                                        ])
                                    
                                    ocr_result = response.text if response.text else ""
                                    
                                    # JSON 파싱 시도
                                    try:
                                        # JSON 부분만 추출 (코드 블록 제거)
                                        import json
                                        ocr_result_clean = ocr_result.strip()
                                        if ocr_result_clean.startswith("```"):
                                            # 코드 블록 제거
                                            lines = ocr_result_clean.split("\n")
                                            json_lines = [l for l in lines if not l.strip().startswith("```")]
                                            ocr_result_clean = "\n".join(json_lines)
                                        
                                        ocr_extracted_info = json.loads(ocr_result_clean)
                                        st.session_state.ocr_extracted_info = ocr_extracted_info
                                        st.session_state.ocr_file_name = file_to_verify.name
                                        
                                        # 추출된 정보 표시
                                        extracted_fields = []
                                        if ocr_extracted_info.get("receipt_number"):
                                            extracted_fields.append(f"영수증/예약 번호: {ocr_extracted_info['receipt_number']}")
                                        if ocr_extracted_info.get("customer_name"):
                                            extracted_fields.append(f"고객 성함: {ocr_extracted_info['customer_name']}")
                                        if ocr_extracted_info.get("customer_email"):
                                            extracted_fields.append(f"이메일: {ocr_extracted_info['customer_email']}")
                                        if ocr_extracted_info.get("customer_phone"):
                                            extracted_fields.append(f"전화번호: {ocr_extracted_info['customer_phone']}")
                                        if ocr_extracted_info.get("payment_method"):
                                            extracted_fields.append(f"결제 수단: {ocr_extracted_info['payment_method']}")
                                        if ocr_extracted_info.get("card_last4"):
                                            extracted_fields.append(f"카드 뒷자리: {ocr_extracted_info['card_last4']}")
                                        
                                        if extracted_fields:
                                            st.success(L.get("ocr_extracted_info", "✅ OCR로 다음 정보를 추출했습니다:") + "\n" + "\n".join(f"- {field}" for field in extracted_fields))
                                        else:
                                            st.info(L.get("ocr_no_verification_info", "ℹ️ OCR로 정보를 추출했지만 검증에 필요한 정보를 찾지 못했습니다."))
                                    except json.JSONDecodeError:
                                        # JSON 파싱 실패 시 텍스트에서 직접 추출 시도
                                        st.warning(L.get("ocr_json_parse_failed", "⚠️ OCR 결과를 JSON으로 파싱하지 못했습니다. 수동으로 입력해주세요."))
                                        st.text_area(L.get("ocr_raw_result_label", "OCR 원본 결과:"), ocr_result, height=100, key="ocr_raw_result")
                                        ocr_extracted_info = {}
                                else:
                                    st.warning(L.get("ocr_requires_gemini", "⚠️ OCR 기능을 사용하려면 Gemini API 키가 필요합니다. 수동으로 정보를 입력해주세요."))
                            except Exception as ocr_error:
                                st.warning(L.get("ocr_error_occurred", "⚠️ OCR 처리 중 오류가 발생했습니다: {error}").format(error=str(ocr_error)))
                                ocr_extracted_info = {}
                    else:
                        # 이전에 추출한 정보 재사용
                        ocr_extracted_info = st.session_state.get('ocr_extracted_info', {})
                        if ocr_extracted_info:
                            extracted_fields = []
                            if ocr_extracted_info.get("receipt_number"):
                                extracted_fields.append(f"{L.get('receipt_number_label', '영수증/예약 번호')}: {ocr_extracted_info['receipt_number']}")
                            if ocr_extracted_info.get("customer_name"):
                                extracted_fields.append(f"{L.get('customer_name_label', '고객 성함')}: {ocr_extracted_info['customer_name']}")
                            if ocr_extracted_info.get("customer_email"):
                                extracted_fields.append(f"{L.get('email_label', '이메일')}: {ocr_extracted_info['customer_email']}")
                            if ocr_extracted_info.get("customer_phone"):
                                extracted_fields.append(f"{L.get('phone_label', '전화번호')}: {ocr_extracted_info['customer_phone']}")
                            if extracted_fields:
                                st.info(L.get("previous_extracted_info", "ℹ️ 이전에 추출한 정보:") + " " + ", ".join(extracted_fields))
                
                # OCR로 추출된 정보가 있으면 세션 상태에서 가져오기
                if 'ocr_extracted_info' in st.session_state and st.session_state.ocr_extracted_info:
                    ocr_extracted_info = st.session_state.ocr_extracted_info
                
                verification_cols = st.columns(2)
                
                with verification_cols[0]:
                    # OCR로 추출한 정보가 있으면 기본값으로 사용
                    receipt_default = ocr_extracted_info.get("receipt_number", "") if ocr_extracted_info else ""
                    verification_receipt = st.text_input(
                        L['verification_receipt_label'],
                        value=receipt_default,
                        key="verification_receipt_input",
                        help=L.get("verification_receipt_help", "고객이 제공한 영수증 번호 또는 예약 번호를 입력하세요. (OCR로 자동 추출됨)")
                    )
                    
                    # 결제 수단 선택
                    payment_method_options = [
                        L.get("payment_method_card", "신용/체크카드"),
                        L.get("payment_method_kakaopay", "카카오페이"),
                        L.get("payment_method_naverpay", "네이버페이"),
                        L.get("payment_method_online_banking", "온라인뱅킹"),
                        L.get("payment_method_grabpay", "GrabPay"),
                        L.get("payment_method_tng", "Touch N Go"),
                        L.get("payment_method_other", "기타")
                    ]
                    
                    # OCR로 추출한 결제 수단이 있으면 매칭 시도
                    ocr_payment_method = ocr_extracted_info.get("payment_method", "") if ocr_extracted_info else ""
                    payment_method_index = 0
                    if ocr_payment_method:
                        # OCR 추출값과 옵션 매칭
                        ocr_payment_lower = ocr_payment_method.lower()
                        for idx, option in enumerate(payment_method_options):
                            if any(keyword in ocr_payment_lower for keyword in ["카드", "card", "신용", "credit", "체크", "check"]):
                                if "신용" in option or "체크" in option or "card" in option.lower():
                                    payment_method_index = idx
                                    break
                            elif any(keyword in ocr_payment_lower for keyword in ["카카오", "kakao"]):
                                if "카카오" in option:
                                    payment_method_index = idx
                                    break
                            elif any(keyword in ocr_payment_lower for keyword in ["네이버", "naver"]):
                                if "네이버" in option:
                                    payment_method_index = idx
                                    break
                            elif any(keyword in ocr_payment_lower for keyword in ["계좌", "account", "뱅킹", "banking"]):
                                if "뱅킹" in option or "banking" in option.lower():
                                    payment_method_index = idx
                                    break
                    
                    verification_payment_method = st.selectbox(
                        L['verification_payment_method_label'],
                        options=payment_method_options,
                        index=payment_method_index,
                        key="verification_payment_method_input",
                        help="고객이 사용한 결제 수단을 선택하세요. (OCR로 자동 추출됨)"
                    )
                    
                    # 결제 정보 입력 (카드 뒷자리 또는 계좌번호)
                    if verification_payment_method == L.get("payment_method_card", "신용/체크카드"):
                        card_default = ocr_extracted_info.get("card_last4", "") if ocr_extracted_info else ""
                        verification_card = st.text_input(
                            L['verification_card_label'],
                            value=card_default,
                            key="verification_card_input",
                            max_chars=4,
                            help=L.get("verification_card_help", "고객이 제공한 카드 뒷자리 4자리를 입력하세요. (OCR로 자동 추출됨)")
                        )
                        verification_account = ""
                    elif verification_payment_method == L.get("payment_method_online_banking", "온라인뱅킹"):
                        account_default = ocr_extracted_info.get("account_number", "") if ocr_extracted_info else ""
                        verification_account = st.text_input(
                            L['verification_account_label'],
                            value=account_default,
                            key="verification_account_input",
                            help="고객이 제공한 계좌번호를 입력하세요. (OCR로 자동 추출됨)"
                        )
                        verification_card = ""
                    else:
                        # 카카오페이, 네이버페이 등은 결제 수단 정보만으로 확인 가능
                        verification_card = ""
                        verification_account = ""
                    
                    name_default = ocr_extracted_info.get("customer_name", "") if ocr_extracted_info else ""
                    verification_name = st.text_input(
                        L['verification_name_label'],
                        value=name_default,
                        key="verification_name_input",
                        help=L.get("verification_name_help", "고객이 제공한 성함을 입력하세요. (OCR로 자동 추출됨)")
                    )
                
                with verification_cols[1]:
                    email_default = ocr_extracted_info.get("customer_email", "") if ocr_extracted_info else ""
                    verification_email = st.text_input(
                        L['verification_email_label'],
                        value=email_default,
                        key="verification_email_input",
                        help=L.get("verification_email_help", "고객이 제공한 이메일 주소를 입력하세요. (OCR로 자동 추출됨)")
                    )
                    phone_default = ocr_extracted_info.get("customer_phone", "") if ocr_extracted_info else ""
                    verification_phone = st.text_input(
                        L['verification_phone_label'],
                        value=phone_default,
                        key="verification_phone_input",
                        help=L.get("verification_phone_help", "고객이 제공한 연락처를 입력하세요. (OCR로 자동 추출됨)")
                    )
                
                # 시스템에 저장된 검증 정보 (시뮬레이션용 - 실제로는 DB에서 가져옴)
                stored_verification_info = st.session_state.verification_info.copy()
                
                # 검증 버튼 (길이 최소화)
                st.markdown("---")
                verify_cols = st.columns([1, 1])
                with verify_cols[0]:
                    if st.button(L['button_verify'], key="btn_verify_customer", type="primary"):
                        # 파일 검증 정보 확인 (고객 첨부 파일 또는 새로 업로드한 파일)
                        final_file_verified = False
                        file_info_for_verification = None
                        
                        if file_to_verify:
                            final_file_verified = True
                            file_info_for_verification = {
                                "filename": file_to_verify.name,
                                "size": file_to_verify.size if hasattr(file_to_verify, 'size') else 0,
                                "type": file_to_verify.type if hasattr(file_to_verify, 'type') else "unknown"
                            }
                            st.session_state.verification_file_verified = True
                        elif file_verified:  # 파일 정보만 있는 경우
                            final_file_verified = True
                            file_info_for_verification = st.session_state.verification_file_info if 'verification_file_info' in st.session_state else None
                        
                        # 결제 정보 구성 (payment_info 필드 추가)
                        payment_info = ""
                        if verification_payment_method == L.get("payment_method_card", "신용/체크카드"):
                            payment_info = f"{verification_payment_method} {verification_card}" if verification_card else verification_payment_method
                        elif verification_payment_method == L.get("payment_method_online_banking", "온라인뱅킹"):
                            payment_info = f"{verification_payment_method} {verification_account}" if verification_account else verification_payment_method
                        else:
                            payment_info = verification_payment_method
                        
                        # OCR로 추출한 정보가 있으면 우선 사용 (수동 입력값이 있으면 수동 입력값 우선)
                        final_receipt = verification_receipt if verification_receipt else (ocr_extracted_info.get("receipt_number", "") if ocr_extracted_info else "")
                        final_name = verification_name if verification_name else (ocr_extracted_info.get("customer_name", "") if ocr_extracted_info else "")
                        final_email = verification_email if verification_email else (ocr_extracted_info.get("customer_email", "") if ocr_extracted_info else "")
                        final_phone = verification_phone if verification_phone else (ocr_extracted_info.get("customer_phone", "") if ocr_extracted_info else "")
                        final_card = verification_card if verification_card else (ocr_extracted_info.get("card_last4", "") if ocr_extracted_info else "")
                        final_account = verification_account if verification_account else (ocr_extracted_info.get("account_number", "") if ocr_extracted_info else "")
                        
                        provided_info = {
                            "receipt_number": final_receipt,
                            "card_last4": final_card if verification_payment_method == L.get("payment_method_card", "신용/체크카드") else "",
                            "account_number": final_account if verification_payment_method == L.get("payment_method_online_banking", "온라인뱅킹") else "",
                            "payment_method": verification_payment_method,
                            "payment_info": payment_info,  # 결제 정보 통합 필드 추가
                            "customer_name": final_name,
                            "customer_email": final_email,
                            "customer_phone": final_phone,
                            "file_uploaded": final_file_verified,
                            "file_info": file_info_for_verification,  # 파일 상세 정보 추가
                            "ocr_extracted": ocr_extracted_info if ocr_extracted_info else {}  # OCR 추출 정보도 포함
                        }
                        
                        # 시스템에 저장된 검증 정보에도 파일 정보 추가 (시뮬레이션용)
                        stored_verification_info_with_file = stored_verification_info.copy()
                        if customer_has_attachment and st.session_state.customer_attachment_file:
                            stored_verification_info_with_file["file_uploaded"] = True
                            stored_verification_info_with_file["file_info"] = {
                                "filename": st.session_state.customer_attachment_file.name,
                                "size": st.session_state.customer_attachment_file.size if hasattr(st.session_state.customer_attachment_file, 'size') else 0,
                                "type": st.session_state.customer_attachment_file.type if hasattr(st.session_state.customer_attachment_file, 'type') else "unknown"
                            }
                        
                        # 검증 실행 (시스템 내부에서만 실행)
                        is_verified, verification_results = verify_customer_info(
                            provided_info, stored_verification_info_with_file
                        )
                        
                        if is_verified:
                            st.session_state.is_customer_verified = True
                            st.session_state.verification_stage = "VERIFIED"
                            st.session_state.verification_info["verification_attempts"] += 1
                            st.success(L['verification_success'])
                        else:
                            st.session_state.verification_stage = "VERIFICATION_FAILED"
                            st.session_state.verification_info["verification_attempts"] += 1
                            failed_fields = [k for k, v in verification_results.items() if not v]
                            
                            # 검증 실패 필드에 대한 상세 정보 제공 (보안: 시스템 저장값은 노출하지 않음)
                            failed_details = []
                            for field in failed_fields:
                                provided_value = provided_info.get(field, "")
                                
                                # 보안: 민감한 정보 마스킹 및 시스템 저장값은 노출하지 않음
                                if field == "file_uploaded":
                                    failed_details.append(f"{field}: 제공됨={provided_info.get('file_uploaded', False)}")
                                elif field == "file_info":
                                    provided_file = provided_info.get('file_info', {})
                                    failed_details.append(f"{field}: 제공된 파일={provided_file.get('filename', '없음')}")
                                elif field == "customer_email":
                                    # 이메일 마스킹
                                    masked_email = mask_email(provided_value) if provided_value else "없음"
                                    failed_details.append(f"{field}: 제공값='{masked_email}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "customer_phone":
                                    # 전화번호 마스킹 (뒷자리만 표시)
                                    if provided_value and len(provided_value) > 4:
                                        masked_phone = "***-" + provided_value[-4:]
                                    else:
                                        masked_phone = provided_value if provided_value else "없음"
                                    failed_details.append(f"{field}: 제공값='{masked_phone}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "card_last4":
                                    # 카드 번호는 이미 뒷자리 4자리만 있으므로 마스킹
                                    if provided_value:
                                        masked_card = "****" if len(provided_value) == 4 else provided_value
                                    else:
                                        masked_card = "없음"
                                    failed_details.append(f"{field}: 제공값='{masked_card}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "account_number":
                                    # 계좌번호 마스킹
                                    if provided_value and len(provided_value) > 4:
                                        masked_account = "***-" + provided_value[-4:]
                                    else:
                                        masked_account = provided_value if provided_value else "없음"
                                    failed_details.append(f"{field}: 제공값='{masked_account}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "customer_name":
                                    # 이름은 부분 마스킹
                                    if provided_value and len(provided_value) > 1:
                                        masked_name = (provided_value[0] if len(provided_value) > 0 else "*") + "*" * (len(provided_value) - 1) if len(provided_value) > 1 else "*"
                                    else:
                                        masked_name = provided_value if provided_value else "없음"
                                    failed_details.append(f"{field}: 제공값='{masked_name}' (시스템 저장값은 보안상 표시하지 않음)")
                                else:
                                    # 기타 필드는 값의 일부만 표시 (보안)
                                    if provided_value:
                                        if len(provided_value) > 8:
                                            masked_value = provided_value[:4] + "***" + provided_value[-2:]
                                        else:
                                            masked_value = "*" * len(provided_value)
                                    else:
                                        masked_value = "없음"
                                    failed_details.append(f"{field}: 제공값='{masked_value}' (시스템 저장값은 보안상 표시하지 않음)")
                            
                            error_message = L['verification_failed'].format(failed_fields=', '.join(failed_fields))
                            error_message += "\n\n⚠️ **보안 정책**: 시스템에 저장된 실제 검증 정보는 보안상 표시하지 않습니다."
                            if failed_details:
                                error_message += f"\n\n**제공된 정보 (일부 마스킹):**\n" + "\n".join(f"- {detail}" for detail in failed_details)
                            
                            st.error(error_message)
                
                with verify_cols[1]:
                    if st.button(L['button_retry_verification'], key="btn_retry_verification"):
                        st.session_state.verification_stage = "WAIT_VERIFICATION"
                        st.session_state.verification_info["verification_attempts"] = 0
                        # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                        # st.rerun()
                
                # 검증 시도 횟수 표시
                if st.session_state.verification_info.get("verification_attempts", 0) > 0:
                    st.info(L['verification_attempts'].format(count=st.session_state.verification_info['verification_attempts']))
            
            # ⭐ 수정: 검증 전 제한 사항도 버튼 클릭 시에만 표시 (고객 검증 버튼에 포함)
            # 검증되지 않은 상태에서는 힌트 및 초안 생성 제한
            st.markdown("---")
            st.markdown(f"### {L.get('verification_restrictions', '검증 전 제한 사항')}")
            st.info(L.get('verification_restrictions_text', '검증이 완료되기 전까지 일부 기능이 제한됩니다.'))
        
        elif is_login_inquiry and st.session_state.is_customer_verified:
            st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))

        # ⭐ 검증 UI가 표시될 때는 에이전트 응답 UI를 숨김
        # ⭐ AI 응답 초안 생성 기능 제거 (회사 정보 & FAQ 탭에 이미 있음)
        # 이 기능은 '회사 정보 & FAQ' > '고객 문의 재확인' 탭에서 사용할 수 있습니다.

        # ⭐ 전화 발신 버튼 제거 (메시지 말풍선에 버튼으로 이동)
        # 전화 발신 기능은 에이전트 응답 메시지 말풍선의 '업체에 전화' / '고객에게 전화' 버튼을 통해 사용할 수 있습니다.

        # Supervisor 정책 업로더 제거됨

        # --- 에이전트 첨부 파일 업로더는 숨김 처리 (버튼으로 대체) ---
        # 파일 업로더는 버튼 클릭 시에만 표시되도록 처리
        agent_attachment_files = None
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
                file_names = ", ".join([f["name"] for f in
                                        st.session_state.agent_attachment_file])
                st.info(L.get("agent_attachment_files_ready", "✅ {count}개 에이전트 첨부 파일 준비 완료: {files}").format(count=len(agent_attachment_files), files=file_names))
                st.session_state.show_agent_file_uploader = False  # 파일 선택 후 숨김
            else:
                st.session_state.agent_attachment_file = []
        else:
            st.session_state.agent_attachment_file = []

        # 마이크 녹음 처리 (전화 부분과 동일한 패턴: 종료 시 자동 전사)
        # 전사 로직: bytes_to_process에 데이터가 있을 때만 실행 (전화 부분과 동일)
        if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process is not None:
            # ⭐ 수정: OpenAI 또는 Gemini API 키가 있는지 확인
            has_openai = st.session_state.openai_client is not None
            has_gemini = bool(get_api_key("gemini"))
            
            if not has_openai and not has_gemini:
                st.error(L.get("whisper_client_error", "Whisper 클라이언트 오류") + " (OpenAI 또는 Gemini API Key 필요)")
                st.session_state.bytes_to_process = None
            else:
                # ⭐ 전사 결과를 저장할 변수 초기화
                agent_response_transcript = None

                # 전사 후 바이트 데이터 백업 (전사 전에 백업)
                audio_bytes_backup = st.session_state.bytes_to_process
                
                # 전사 후 바이트 데이터 즉시 삭제 (조건문 재평가 방지)
                st.session_state.bytes_to_process = None
                
                with st.spinner(L.get("whisper_processing", "전사 중...")):
                    try:
                        # Whisper 전사 (자동 언어 감지 사용)
                        agent_response_transcript = transcribe_bytes_with_whisper(
                            audio_bytes_backup,
                            "audio/wav",
                            lang_code=None,
                            auto_detect=True
                        )
                    except Exception as e:
                        agent_response_transcript = L.get("transcription_error_with_error", "❌ 전사 오류: {error}").format(error=str(e))

                # 2) 전사 실패 처리 (채팅/이메일과 동일한 패턴)
                if not agent_response_transcript or agent_response_transcript.startswith("❌"):
                    error_msg = agent_response_transcript if agent_response_transcript else L.get("transcription_no_result", "전사 결과가 없습니다.")
                    st.error(error_msg)
                    
                    # ⭐ [수정 4] 채팅/메일 탭에서 에러 발생 시 입력 필드를 비움
                    if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                        st.session_state.agent_response_area_text = ""
                        st.session_state.last_transcript = "" # 전사 실패 시 last_transcript 초기화
                    else:
                        # 전화 탭의 경우
                        st.session_state.current_agent_audio_text = L.get("transcription_error", "전사 오류")
                        if "agent_response_input_box_widget_call" in st.session_state:
                            st.session_state.agent_response_input_box_widget_call = "" # 전화 탭 입력 필드도 초기화
                        st.session_state.last_transcript = "" # 전사 실패 시 last_transcript 초기화

                elif not agent_response_transcript.strip(): # ⭐ 수정: 전사 결과가 비어 있거나 (공백만 있는 경우) 다음 단계로 진행하지 못하는 문제 해결
                    st.warning(L.get("transcription_empty_warning", "전사 결과가 비어 있습니다."))
                    if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                        st.session_state.agent_response_area_text = "" # 채팅/메일 탭도 초기화
                    else:
                        st.session_state.current_agent_audio_text = ""
                        if "agent_response_input_box_widget_call" in st.session_state:
                            st.session_state.agent_response_input_box_widget_call = ""
                    st.session_state.last_transcript = ""
                    # ⭐ 재실행 불필요: 전사 결과가 비어있어도 사용자가 다시 녹음할 수 있음
                    # st.rerun()

                elif agent_response_transcript.strip():
                    # 3) 전사 성공 - CC/입력창에 반영
                    agent_response_transcript = agent_response_transcript.strip()

                    # ⭐ [핵심 수정 5] 전사 결과를 last_transcript에 저장하고, AGENT_TURN 상태의 입력 필드에도 반영
                    st.session_state.last_transcript = agent_response_transcript
                    
                    # A. 채팅/메일 탭 처리
                    if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                        # AGENT_TURN 섹션의 st.text_area value로 사용되는 세션 상태 변수에 반영
                        st.session_state.agent_response_area_text = agent_response_transcript
                    
                    # B. 전화 탭 처리
                    else:
                        st.session_state.current_agent_audio_text = agent_response_transcript
                        # ⭐ [수정 3: 핵심 수정] 전화 탭 입력 칸에도 전사 결과 전달
                        if "agent_response_input_box_widget_call" in st.session_state:
                            st.session_state.agent_response_input_box_widget_call = agent_response_transcript
                    
                    # ⭐ 수정: 성공 메시지는 조건부로만 표시 (불필요한 rerun 방지)
                    # snippet = agent_response_transcript[:50].replace("\n", " ")
                    # if len(agent_response_transcript) > 50:
                    #     snippet += "..."
                    # st.success(L.get("whisper_success", "전사 완료") + f" **{L.get('recognized_content', '인식 내용')}:** *{snippet}*")
                    # st.info(L.get("transcription_auto_filled", "💡 전사된 텍스트가 CC 자막 및 입력창에 자동으로 입력되었습니다."))
                    # ⭐ 수정: 전사 결과는 위젯에 자동으로 반영되므로 별도 rerun 불필요

        # ⭐ 검증 UI나 응대 초안 UI가 표시되지 않을 때만 솔루션 체크박스 표시
        show_draft_ui = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui = st.session_state.get("show_customer_data_ui", False)
        if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
            # ⭐ admin.py 스타일로 간소화: 깔끔한 레이아웃
            # 솔루션 제공 체크박스 (기능 유지)
            st.session_state.is_solution_provided = st.checkbox(
                L["solution_check_label"],
                value=st.session_state.is_solution_provided,
                key="solution_checkbox_widget",
            )
        
        # ⭐ 메시지 입력 칸은 항상 표시 (어떤 기능 버튼을 클릭해도 항상 표시)
        # 위젯 생성 전에 초기화 플래그 확인 및 처리
        # ⭐ [핵심 수정 1] 전사 결과가 있으면 초기화하지 않도록 보장
        if st.session_state.get("reset_agent_response_area", False):
            # 전사 결과가 없거나 (last_transcript가 비어 있거나, 전사 중이 아닐 때)만 초기화
            if not st.session_state.get("last_transcript") or not st.session_state.last_transcript:
                st.session_state.agent_response_area_text = ""
            st.session_state.reset_agent_response_area = False
        
        # ⭐ 마이크 전사 결과 또는 자동 생성된 응대 초안이 있으면 입력창에 표시
        # 위젯 생성 전에만 값을 설정할 수 있으므로 여기서 처리
        # ⭐ [수정 1] 전사 결과가 입력 칸에 확실히 반영되도록 보장 (최우선 처리)
        if st.session_state.get("last_transcript") and st.session_state.last_transcript:
            # 전사 결과를 text_area의 value로 사용되는 세션 상태 변수에 반영
            st.session_state.agent_response_area_text = st.session_state.last_transcript
        # ⭐ [추가] 자동 생성된 응대 초안이 있으면 입력창에 표시 (전사 결과보다 우선순위 낮음)
        elif st.session_state.get("auto_generated_draft") and st.session_state.auto_generated_draft:
            if not st.session_state.get("agent_response_area_text") or not st.session_state.agent_response_area_text:
                st.session_state.agent_response_area_text = st.session_state.auto_generated_draft
                # 표시 후 초기화 (한 번만 표시)
                st.session_state.auto_generated_draft = None
        # ⭐ [추가 수정] agent_response_area_text가 비어있고 last_transcript가 있으면 반영
        elif not st.session_state.get("agent_response_area_text") and st.session_state.get("last_transcript") and st.session_state.last_transcript:
            st.session_state.agent_response_area_text = st.session_state.last_transcript

        # --- UI 개선: 에이전트 응답 헤더를 입력 칸 바로 위에 배치 ---
        # ⭐ 에이전트 응답 헤더 표시 (메시지 입력 칸 바로 위)
        if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
            st.markdown(f"### {L['agent_response_header']}")
        
        # --- UI 개선: 메시지 입력 칸과 파일 첨부 버튼을 한 줄에 배치 ---
        # ⭐ 메시지 입력 칸은 항상 표시 (어떤 기능 버튼을 클릭해도 항상 표시)
        
        # ⭐ 수정: 전사 결과는 입력 필드에만 표시하고, 자동 전송하지 않음
        # 사용자가 직접 입력하거나 전송 버튼을 눌러야 메시지가 전송됨
        # (자동 전송 로직 제거 - 순서 꼬임 방지)
        
        # 입력 칸과 파일 첨부 버튼을 한 줄에 배치
        input_cols = st.columns([10, 1])
        
        with input_cols[0]:
            # st.chat_input으로 입력 받기 (app.py 스타일)
            agent_response_input = st.chat_input(L.get("agent_response_placeholder", "고객에게 응답하세요..."))
        
        with input_cols[1]:
            # (+) 파일 첨부 버튼 (입력 칸 옆에 작은 버튼으로 배치)
            if st.button("➕", key="btn_add_attachment_unified", help=L.get("button_add_attachment", "파일 첨부"), use_container_width=True, type="secondary"):
                st.session_state.show_agent_file_uploader = True
        
        # 전사 결과 표시 (있는 경우) - 입력 칸 아래에 작은 텍스트로 표시
        if st.session_state.get("agent_response_area_text") and st.session_state.agent_response_area_text:
            transcript_preview = st.session_state.agent_response_area_text[:30]
            st.caption(L.get("transcription_label", "💬 전사: {text}...").format(text=transcript_preview))

        # 전송 로직 실행 (st.chat_input은 Enter 키 또는 전송 버튼으로 자동 전송됨)
        agent_response = None
        if agent_response_input:
            agent_response = agent_response_input.strip()
        
        # --- End of Unified Input UI ---
            
        if agent_response:
            if not agent_response.strip():
                st.warning(L["empty_response_warning"])
                # st.stop()
            else:
                # AHT 타이머 시작
                if st.session_state.start_time is None and len(st.session_state.simulator_messages) >= 1:
                    st.session_state.start_time = datetime.now()

                # --- 에이전트 첨부 파일 처리 (다중 파일 처리) ---
                final_response_content = agent_response
                if st.session_state.agent_attachment_file:
                    file_infos = st.session_state.agent_attachment_file
                    file_names = ", ".join([f["name"] for f in file_infos])
                    attachment_msg = L["agent_attachment_status"].format(
                        filename=file_names, filetype=f"총 {len(file_infos)}개 파일"
                    )
                    final_response_content = f"{agent_response}\n\n---\n{attachment_msg}"

                # 로그 업데이트
                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": final_response_content}
                )
                
                # ⭐ 고객 데이터 자동 저장 (에이전트 응답 시마다 업데이트) - 완벽한 구현
                try:
                    import logging
                    logger = logging.getLogger(__name__)
                    
                    # 고객 ID 결정 (이메일 > 전화번호 > 인스턴스 ID 순)
                    customer_id = st.session_state.get("customer_email", "") or st.session_state.get("customer_phone", "")
                    if not customer_id:
                        customer_id = f"customer_{st.session_state.sim_instance_id}"
                    
                    logger.info(f"[고객 데이터 저장 시작] customer_id={customer_id}")
                    
                    # 기존 고객 데이터 로드 (병합을 위해)
                    existing_data = st.session_state.customer_data_manager.load_customer_data(customer_id)
                    logger.debug(f"[고객 데이터 로드] 기존 데이터 존재 여부: {existing_data is not None}")
                    
                    # 현재 상담 요약 생성 (있는 경우)
                    consultation_summary = ""
                    try:
                        if st.session_state.simulator_messages:
                            consultation_summary = generate_chat_summary(
                                st.session_state.simulator_messages,
                                st.session_state.get("customer_query_text_area", ""),
                                st.session_state.get("customer_type_sim_select", ""),
                                st.session_state.get("language", "ko")
                            )
                            if consultation_summary:
                                consultation_summary = str(consultation_summary)
                    except Exception as e:
                        logger.warning(f"[상담 요약 생성 실패] {e}")
                    
                    # 고객 데이터 구성 (완전한 스키마)
                    current_time = datetime.now().isoformat()
                    customer_data = {
                        "customer_id": customer_id,
                        "data": {
                            "name": st.session_state.get("customer_name", ""),
                            "email": st.session_state.get("customer_email", ""),
                            "phone": st.session_state.get("customer_phone", ""),
                            "company": st.session_state.get("customer_company", ""),
                            "account_created_at": existing_data.get("data", {}).get("account_created_at", current_time) if existing_data else current_time,
                            "last_access_at": current_time,
                            "last_consultation_at": current_time,
                            "consultation_history": existing_data.get("data", {}).get("consultation_history", []) if existing_data else [],
                            "notes": existing_data.get("data", {}).get("notes", "") if existing_data else ""
                        },
                        "conversations": [
                            {
                                "role": msg.get("role", ""),
                                "content": msg.get("content", ""),
                                "timestamp": msg.get("timestamp", current_time)
                            }
                            for msg in st.session_state.simulator_messages  # 전체 메시지 저장
                        ],
                        "current_consultation": {
                            "consultation_id": st.session_state.sim_instance_id,
                            "started_at": st.session_state.get("consultation_started_at", current_time),
                            "last_updated_at": current_time,
                            "summary": consultation_summary,
                            "customer_type": st.session_state.get("customer_type_sim_select", ""),
                            "language": st.session_state.get("language", "ko"),
                            "messages_count": len(st.session_state.simulator_messages),
                            "is_ended": st.session_state.get("is_chat_ended", False)
                        }
                    }
                    
                    # 상담 이력에 현재 상담 추가 (중복 방지)
                    consultation_entry = {
                        "consultation_id": st.session_state.sim_instance_id,
                        "date": current_time,
                        "summary": consultation_summary[:200] if consultation_summary else "",
                        "customer_type": st.session_state.get("customer_type_sim_select", ""),
                        "language": st.session_state.get("language", "ko")
                    }
                    if existing_data and "data" in existing_data and "consultation_history" in existing_data["data"]:
                        # 기존 상담 이력에서 동일한 consultation_id가 있으면 업데이트, 없으면 추가
                        history = existing_data["data"]["consultation_history"]
                        existing_idx = next((i for i, h in enumerate(history) if h.get("consultation_id") == st.session_state.sim_instance_id), None)
                        if existing_idx is not None:
                            history[existing_idx] = consultation_entry
                        else:
                            history.append(consultation_entry)
                        customer_data["data"]["consultation_history"] = history
                    else:
                        customer_data["data"]["consultation_history"] = [consultation_entry]
                    
                    logger.debug(f"[고객 데이터 구성 완료] conversations={len(customer_data['conversations'])}, consultation_history={len(customer_data['data']['consultation_history'])}")
                    
                    # 고객 데이터 저장
                    save_success = st.session_state.customer_data_manager.save_customer_data(
                        customer_id,
                        customer_data,
                        merge=True
                    )
                    
                    if save_success:
                        logger.info(f"[고객 데이터 저장 성공] customer_id={customer_id}, conversations={len(customer_data['conversations'])}")
                        # 저장 성공 확인을 위해 다시 로드해서 검증
                        verify_data = st.session_state.customer_data_manager.load_customer_data(customer_id)
                        if verify_data:
                            logger.debug(f"[고객 데이터 검증 성공] 저장된 conversations={len(verify_data.get('conversations', []))}")
                        else:
                            logger.error(f"[고객 데이터 검증 실패] 저장 후 로드 실패: customer_id={customer_id}")
                    else:
                        logger.error(f"[고객 데이터 저장 실패] customer_id={customer_id}")
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"[고객 데이터 저장 중 오류] customer_id={customer_id}, error={e}", exc_info=True)

                # ⭐ 추가: 에이전트 응답에 메일 끝인사가 포함되어 있는지 확인
                email_closing_patterns = [
                    "추가 문의사항이 있으면 언제든지 연락", "추가 문의 사항이 있으면 언제든지 연락",
                    "추가 문의사항이 있으시면", "추가 문의 사항이 있으시면",
                    "언제든지 연락", "언제든지 연락 주세요",
                    "additional inquiries", "any additional questions", "any further questions",
                    "feel free to contact", "please feel free to contact",
                    "please don't hesitate to contact", "don't hesitate to contact",
                    "please let me know", "let me know", "let me know if",
                    "please let me know so", "let me know so",
                    "if you have any questions", "if you have any further questions",
                    "if you need any assistance", "if you need further assistance",
                    "if you encounter any issues", "if you still have", "if you remain unclear",
                    "I can assist further", "I can help further", "I can assist",
                    "so I can assist", "so I can help", "so I can assist further",
                    "追加のご質問", "追加のお問い合わせ", "ご質問がございましたら", "お問い合わせがございましたら"
                ]
                is_email_closing_in_response = any(pattern.lower() in final_response_content.lower() for pattern in email_closing_patterns)
                if is_email_closing_in_response:
                    st.session_state.has_email_closing = True  # 플래그 설정

                # 입력창/오디오/첨부 파일 초기화
                # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로,
                # 플래그를 사용하여 위젯이 다시 생성될 때 초기값이 적용되도록 합니다.
                st.session_state.sim_audio_bytes = None
                st.session_state.agent_attachment_file = []  # 첨부 파일 초기화
                st.session_state.language_transfer_requested = False
                st.session_state.realtime_hint_text = ""  # 힌트 초기화
                st.session_state.sim_call_outbound_summary = ""  # 전화 발신 요약 초기화
                st.session_state.last_transcript = ""  # 전사 결과 초기화

                # ⭐ 수정: agent_response_area_text는 위젯이 다시 생성될 때 초기화되도록
                # 플래그만 설정합니다. 위젯 생성 전에 이 플래그를 확인하여 값을 초기화합니다.
                # 위젯이 생성된 후에는 직접 수정할 수 없으므로 플래그만 사용합니다.
                st.session_state.reset_agent_response_area = True
                
                # ⭐ 수정: app.py 스타일 - 메시지 추가 후 바로 고객 반응 생성 (같은 렌더링 사이클에서 처리)
                # 플래그 대신 직접 처리하여 대화 흐름이 자연스럽게 진행되도록 함
                if st.session_state.is_llm_ready:
                    # 고객 반응 생성
                    with st.spinner(L["generating_customer_response"]):
                        customer_response = generate_customer_reaction(st.session_state.language, is_call=False)
                    
                    # 고객 반응을 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "customer", "content": customer_response}
                    )
                    
                    # ⭐ 응대 초안 자동 생성 (고객 메시지 수신 시)
                    try:
                        with st.spinner(L.get("generating_draft_auto", "응대 초안 자동 생성 중...")):
                            # 최근 고객 메시지 가져오기
                            recent_customer_messages = [
                                msg.get("content", "") 
                                for msg in st.session_state.simulator_messages 
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]
                            ]
                            latest_customer_query = recent_customer_messages[-1] if recent_customer_messages else customer_response
                            
                            # 응대 초안 생성
                            draft_text = _generate_initial_advice(
                                latest_customer_query,
                                st.session_state.get("customer_type_sim_select", ""),
                                st.session_state.customer_email,
                                st.session_state.customer_phone,
                                st.session_state.language,
                                st.session_state.customer_attachment_file
                            )
                            
                            # 응대 초안을 세션 상태에 저장 (입력창에 자동 표시용)
                            st.session_state.auto_generated_draft = draft_text
                            st.session_state.agent_response_area_text = draft_text
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(f"응대 초안 자동 생성 실패: {e}", exc_info=True)
                    
                    # ⭐ 추가: 메일 끝인사가 포함된 경우 고객 응답 확인 및 설문 조사 버튼 활성화
                    if st.session_state.get("has_email_closing", False):
                        # 고객의 긍정 반응 확인
                        positive_keywords = [
                            "No, that will be all", "no more", "없습니다", "감사합니다", "Thank you", "ありがとう",
                            "추가 문의 사항 없습니다", "추가 문의사항 없습니다", "no additional", "追加の質問はありません",
                            "알겠습니다", "알겠어요", "ok", "okay", "네", "yes", "좋습니다", "good", "fine", "괜찮습니다"
                        ]
                        is_positive = any(keyword.lower() in customer_response.lower() for keyword in positive_keywords)
                        
                        # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
                        import re
                        escaped = re.escape(L.get('customer_no_more_inquiries', ''))
                        no_more_pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
                        if is_positive or no_more_regex.search(customer_response):
                            # 설문 조사 버튼 활성화를 위해 WAIT_CUSTOMER_CLOSING_RESPONSE 단계로 이동
                            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                        else:
                            # 메일 끝인사가 있지만 고객이 추가 질문을 한 경우
                            st.session_state.sim_stage = "AGENT_TURN"
                    else:
                        # 고객 응답에 따라 다음 단계 결정
                        import re
                        escaped_no_more = re.escape(L.get("customer_no_more_inquiries", ""))
                        no_more_pattern = escaped_no_more.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
                        escaped_positive = re.escape(L.get("customer_positive_response", ""))
                        positive_pattern = escaped_positive.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                        positive_regex = re.compile(positive_pattern, re.IGNORECASE)
                        is_positive_closing = no_more_regex.search(customer_response) is not None or positive_regex.search(customer_response) is not None
                        
                        # 다음 단계 결정
                        if L.get("customer_positive_response", "") in customer_response:
                            if st.session_state.get("is_solution_provided", False):
                                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                            else:
                                st.session_state.sim_stage = "AGENT_TURN"
                        elif is_positive_closing:
                            if no_more_regex.search(customer_response):
                                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                            else:
                                if st.session_state.get("is_solution_provided", False):
                                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                                else:
                                    st.session_state.sim_stage = "AGENT_TURN"
                        elif customer_response.startswith(L.get("customer_escalation_start", "")):
                            st.session_state.sim_stage = "ESCALATION_REQUIRED"
                        else:
                            # 고객이 추가 질문하거나 정보 제공한 경우 -> 에이전트 턴으로 이동
                            st.session_state.sim_stage = "AGENT_TURN"
                else:
                    # LLM이 없는 경우 CUSTOMER_TURN 단계로 이동
                    st.session_state.sim_stage = "CUSTOMER_TURN"
                
                # ⭐ 재실행 불필요: 상태 변경 시 자동으로 rerun됨
                # st.rerun()
            

        # --- 언어 이관 버튼 (말풍선 스타일로 변경) ---
        st.markdown("---")
        st.markdown(f"**{L['transfer_header']}**")
        
        languages = list(LANG.keys())
        languages.remove(current_lang)
        
        # 말풍선 스타일로 버튼 배치 (작은 버튼들로 변경)
        transfer_cols = st.columns(len(languages))


        def transfer_session(target_lang: str, current_messages: List[Dict[str, str]]):
            # 언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다.

            # 현재 언어 확인 및 L 변수 정의
            current_lang_at_start = st.session_state.language  # Source language
            L = LANG.get(current_lang_at_start, LANG["ko"])  # L 변수 정의 추가

            # API 키 체크는 run_llm 내부에서 처리되지만, 명시적으로 Gemini 키를 요구함
            if not get_api_key("gemini"):
                st.error(L["simulation_no_key_warning"].replace('API Key', 'Gemini API Key'))
                # st.stop()
            else:
                # AHT 타이머 중지
                st.session_state.start_time = None

                # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
                with st.spinner(L["transfer_loading"]):
                    # 실제 대기 시간 5~10초 (3~10분 시뮬레이션)
                    time.sleep(np.random.uniform(5, 10))

                    # ⭐ [수정] 원본 언어로 핵심 요약 생성 후 번역
                    try:
                        # 원본 언어로 핵심 요약 생성
                        original_summary = summarize_history_with_ai(current_lang_at_start)
                        
                        if not original_summary or original_summary.startswith("❌"):
                            # 요약 생성 실패 시 대화 기록을 번역할 텍스트로 가공
                            history_text = ""
                            for msg in current_messages:
                                role = "Customer" if msg["role"].startswith("customer") or msg[
                                    "role"] == "initial_query" else "Agent"
                                if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                                   "customer_closing_response"]:
                                    history_text += f"{role}: {msg['content']}\n"
                            original_summary = history_text
                        
                        # 핵심 요약을 번역 대상 언어로 번역
                        translated_summary, is_success = translate_text_with_llm(
                            original_summary,
                            target_lang,
                            current_lang_at_start
                        )
                        
                        if not translated_summary:
                            # 번역 실패 시 번역 대상 언어로 요약 재생성
                            translated_summary = summarize_history_with_ai(target_lang)
                            is_success = True if translated_summary and not translated_summary.startswith("❌") else False
                        
                        # ⭐ [핵심 수정] 모든 메시지를 이관된 언어로 번역
                        # ⭐ 최적화: 배치 번역으로 변경하여 API 호출 횟수 감소 및 타이밍 문제 해결
                        translated_messages = []
                        messages_to_translate = []
                        message_indices = []
                        
                        # 번역할 메시지 수집
                        for idx, msg in enumerate(current_messages):
                            translated_msg = msg.copy()
                            if msg["role"] in ["initial_query", "customer", "customer_rebuttal", "agent_response", 
                                              "customer_closing_response", "supervisor"]:
                                if msg.get("content"):
                                    messages_to_translate.append((idx, msg))
                            translated_messages.append(translated_msg)
                        
                        # 배치 번역: 모든 메시지를 하나의 텍스트로 합쳐서 번역
                        if messages_to_translate:
                            try:
                                # 번역할 메시지들을 하나의 텍스트로 합치기
                                combined_text = "\n\n".join([
                                    f"[{msg['role']}]: {msg['content']}" 
                                    for _, msg in messages_to_translate
                                ])
                                
                                # 전체 텍스트를 한 번에 번역 (토큰 제한 고려하여 내부에서 청크 처리)
                                translated_combined, trans_success = translate_text_with_llm(
                                    combined_text,
                                    target_lang,
                                    current_lang_at_start
                                )
                                
                                if trans_success and translated_combined:
                                    # 번역된 텍스트를 다시 메시지로 분리
                                    translated_lines = translated_combined.split("\n\n")
                                    for i, (idx, original_msg) in enumerate(messages_to_translate):
                                        if i < len(translated_lines):
                                            # 번역된 라인에서 역할 제거
                                            translated_line = translated_lines[i]
                                            if "]: " in translated_line:
                                                translated_content = translated_line.split("]: ", 1)[1]
                                            else:
                                                translated_content = translated_line
                                            translated_messages[idx]["content"] = translated_content
                            except Exception as e:
                                # 배치 번역 실패 시 개별 번역으로 폴백
                                for idx, msg in messages_to_translate:
                                    try:
                                        translated_content, trans_success = translate_text_with_llm(
                                            msg["content"],
                                            target_lang,
                                            current_lang_at_start
                                        )
                                        if trans_success:
                                            translated_messages[idx]["content"] = translated_content
                                    except Exception:
                                        # 개별 번역도 실패하면 원본 유지
                                        pass
                        
                        # 번역된 메시지로 업데이트
                        st.session_state.simulator_messages = translated_messages
                        
                        # 이관 요약 저장
                        st.session_state.transfer_summary_text = translated_summary
                        st.session_state.translation_success = is_success
                        st.session_state.language_at_transfer_start = current_lang_at_start
                        
                        # 언어 변경
                        st.session_state.language = target_lang
                        L = LANG.get(target_lang, LANG["ko"])
                        
                        # 언어 이름 가져오기
                        lang_name_target = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(target_lang, "Korean")
                        
                        # 시스템 메시지 추가
                        system_msg = L["transfer_system_msg"].format(target_lang=lang_name_target)
                        st.session_state.simulator_messages.append(
                            {"role": "system_transfer", "content": system_msg}
                        )
                        
                        # 이관 요약을 supervisor 메시지로 추가
                        summary_msg = f"### {L['transfer_summary_header']}\n\n{translated_summary}"
                        st.session_state.simulator_messages.append(
                            {"role": "supervisor", "content": summary_msg}
                        )
                        
                        # 이력 저장
                        customer_type_display = st.session_state.get("customer_type_sim_select", "")
                        save_simulation_history_local(
                            st.session_state.customer_query_text_area,
                            customer_type_display,
                            st.session_state.simulator_messages,
                            is_chat_ended=False,
                            attachment_context=st.session_state.sim_attachment_context_for_llm,
                        )
                        
                        # AGENT_TURN으로 이동
                        st.session_state.sim_stage = "AGENT_TURN"
                        # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                        # st.rerun()
                    except Exception as e:
                        error_msg = L.get("transfer_error", "이관 처리 중 오류 발생: {error}").format(error=str(e))
                        st.error(error_msg)
                        summary_text = L.get("summary_generation_error", "요약 생성 오류: {error}").format(error=str(e))
        
        # 이관 버튼 렌더링 (말풍선 스타일 - 작은 버튼)
        for idx, lang_code in enumerate(languages):
            lang_name = {"ko": "한국어", "en": "영어", "ja": "일본어"}.get(lang_code, lang_code)
            # 말풍선 스타일 라벨 (짧게)
            if lang_code == "en":
                transfer_label = "US 영어 팀으로 이관"
            elif lang_code == "ja":
                transfer_label = "JP 일본어 팀으로 이관"
            else:
                transfer_label = f"{lang_name} 팀으로 이관"
            
            with transfer_cols[idx]:
                if st.button(
                    transfer_label,
                    key=f"btn_transfer_{lang_code}_{st.session_state.sim_instance_id}",
                    type="secondary"
                ):
                    transfer_session(lang_code, st.session_state.simulator_messages)
    
    # =========================
    # 5-B. 에스컬레이션 요청 단계 (ESCALATION_REQUIRED)
    # =========================
    elif st.session_state.sim_stage == "ESCALATION_REQUIRED":