# ========================================
# _pages/_chat_initial_query.py
# 채팅 시뮬레이터 - 초기 문의 입력 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.customer_analysis import (
    detect_text_language, analyze_customer_profile, find_similar_cases
)
from utils.history_handler import save_simulation_history_local
from utils.customer_verification import check_if_login_related_inquiry
from visualization import visualize_customer_profile_scores, visualize_similarity_cases
import uuid


def render_initial_query(L, current_lang):
    """초기 문의 입력 UI 렌더링 (간소화된 버전)"""
    # app.py 스타일의 3-column 레이아웃 적용
    col1, col2, col3 = st.columns([1, 2, 1.5])
    
    # col1은 비워둠 (고객 목록이 없으므로)
    with col1:
        pass
    
    # col3에 입력된 고객 정보 미리보기 표시
    with col3:
        st.subheader(L.get("customer_info", "고객 정보"))
        if st.session_state.get('customer_name') or st.session_state.get('customer_email') or st.session_state.get('customer_phone'):
            if st.session_state.get('customer_name'):
                st.markdown(f"**{L.get('name_label', '성함')}:** {st.session_state.customer_name}")
            if st.session_state.get('customer_email'):
                st.markdown(f"**{L.get('email_label', '이메일')}:** {st.session_state.customer_email}")
            if st.session_state.get('customer_phone'):
                st.markdown(f"**{L.get('contact_label', '연락처')}:** {st.session_state.customer_phone}")
            if st.session_state.get('customer_type_sim_select'):
                st.markdown(f"**{L.get('customer_type_label', '고객 유형')}:** {st.session_state.customer_type_sim_select}")
        else:
            st.info(L.get("customer_info_preview_placeholder", "고객 정보를 입력하면 여기에 표시됩니다."))
    
    with col2:
        st.markdown(f"### 💬 {L.get('customer_inquiry_input_header', '고객 문의 입력')}")
        
        # 문의 내용 입력 (크기 축소)
        customer_query = st.text_area(
            L["customer_query_label"],
            key="customer_query_text_area",
            height=100,  # 150 -> 100으로 축소
            placeholder=L["initial_query_sample"],
            label_visibility="visible"
        )

        # 고객 이름 입력 필드
        customer_name = st.text_input(
            L.get("customer_name_label", "고객님 성함"),
            key="customer_name_input",
            value=st.session_state.get("customer_name", ""),
            placeholder=L.get("customer_name_placeholder", "예: 홍길동")
        )
        st.session_state.customer_name = customer_name

        # 필수 입력 필드 (컴팩트하게)
        col_email, col_phone = st.columns(2)
        with col_email:
            customer_email = st.text_input(
                L["customer_email_label"],
                key="customer_email_input",
                value=st.session_state.customer_email,
            )
        with col_phone:
            customer_phone = st.text_input(
                L["customer_phone_label"],
                key="customer_phone_input",
                value=st.session_state.customer_phone,
            )
        st.session_state.customer_email = customer_email
        st.session_state.customer_phone = customer_phone

        # 고객 유형 선택 (컴팩트하게)
        customer_type_options = L["customer_type_options"]
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="customer_type_sim_select_widget",
        )

        # 첨부 파일 관련 상태 초기화
        st.session_state.customer_attachment_file = None
        st.session_state.sim_attachment_context_for_llm = ""

        # 고객 등록 폼 (채팅 시작 버튼 왼쪽에 배치)
        st.markdown("---")
        st.subheader(L.get("new_customer_registration", "새 고객 등록"))
        with st.form("customer_registration_form_chat"):
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                reg_customer_name = st.text_input(L.get("customer_name_required", "고객명 *"), key="reg_customer_name_chat")
                reg_phone = st.text_input(L.get("contact_required", "연락처 *"), key="reg_phone_chat")
                reg_email = st.text_input(L.get("email_required", "이메일 *"), key="reg_email_chat")
            with col_reg2:
                # ⭐ 언어별 고객 성향 옵션 사용
                personality_options = L.get("personality_options", ["일반", "신중형", "활발형", "가족형", "프리미엄형", "절약형", "자유형"])
                reg_personality = st.selectbox(
                    L.get("customer_personality", "고객 성향"), 
                    personality_options, 
                    key="reg_personality_chat"
                )
                reg_destination = st.text_input(L.get("preferred_destination", "선호 여행지"), key="reg_destination_chat")
            
            col_reg_btn1, col_reg_btn2 = st.columns([1, 1])
            with col_reg_btn1:
                if st.form_submit_button(L.get("customer_registration", "고객 등록"), type="primary", use_container_width=True):
                    if reg_customer_name and reg_phone and reg_email:
                        if hasattr(st.session_state, 'customer_data_manager') and st.session_state.customer_data_manager:
                            try:
                                customer_data = {
                                    'customer_name': reg_customer_name, 
                                    'phone': reg_phone, 
                                    'email': reg_email,
                                    'personality': reg_personality, 
                                    'preferred_destination': reg_destination
                                }
                                customer_id = st.session_state.customer_data_manager.create_customer(customer_data)
                                st.success(L.get("customer_registered_success", "고객이 등록되었습니다! 고객 ID: {customer_id}").format(customer_id=customer_id))
                                # 등록된 고객 정보를 세션 상태에 저장
                                st.session_state.customer_name = reg_customer_name
                                st.session_state.customer_phone = reg_phone
                                st.session_state.customer_email = reg_email
                            except Exception as e:
                                st.error(f"고객 등록 오류: {str(e)}")
                        else:
                            st.error("고객 데이터 관리자가 초기화되지 않았습니다.")
                    else:
                        st.error(L.get("customer_registration_required_fields", "고객명, 연락처, 이메일은 필수 항목입니다."))
            with col_reg_btn2:
                st.form_submit_button(L.get("button_cancel", "취소"), use_container_width=True)
        
        st.markdown("---")

        # 채팅 시작 버튼 (크기 축소)
        if st.button(
                L.get("button_start_chat", "채팅 시작"),
                key=f"btn_start_chat_{st.session_state.sim_instance_id}",
                use_container_width=True,
                type="primary"):
            if not customer_query.strip():
                st.warning(L["simulation_warning_query"])

            # 필수 입력 필드 검증
            if not st.session_state.customer_email.strip(
            ) or not st.session_state.customer_phone.strip():
                st.error(L["error_mandatory_contact"])

            # 초기 상태 리셋
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.is_chat_ended = False
            st.session_state.initial_advice_provided = False
            st.session_state.is_solution_provided = False
            st.session_state.language_transfer_requested = False
            st.session_state.transfer_summary_text = ""
            st.session_state.start_time = None
            st.session_state.sim_instance_id = str(
                uuid.uuid4())

            # UI 플래그 초기화
            st.session_state.show_verification_ui = False
            st.session_state.show_draft_ui = False
            st.session_state.show_customer_data_ui = False
            st.session_state.show_agent_response_ui = False

            # ⭐ 고객 데이터 자동 검색 및 불러오기 (이전 응대 이력 확인)
            if hasattr(st.session_state, 'customer_data_manager') and st.session_state.customer_data_manager:
                try:
                    customer_name = st.session_state.get("customer_name", "").strip()
                    customer_phone = st.session_state.customer_phone.strip()
                    customer_email = st.session_state.customer_email.strip()
                    
                    # 고객 정보로 이전 응대 이력 검색
                    found_customer = st.session_state.customer_data_manager.find_customer_by_info(
                        name=customer_name if customer_name else None,
                        phone=customer_phone if customer_phone else None,
                        email=customer_email if customer_email else None
                    )
                    
                    if found_customer:
                        st.session_state.customer_data = found_customer
                        customer_id = found_customer.get("basic_info", {}).get("customer_id", "")
                        consultation_count = found_customer.get("crm_profile", {}).get("total_consultations", 0)
                        
                        # 이전 응대 이력이 있다는 정보를 메시지에 추가
                        history_info = f"📋 **고객 기억**: 이전 응대 이력이 확인되었습니다. (고객 ID: {customer_id}, 이전 상담 건수: {consultation_count}회)"
                        st.session_state.simulator_messages.append({
                            "role": "system",
                            "content": history_info
                        })
                        print(f"✅ 고객 데이터 자동 불러오기 성공: {customer_id} (상담 건수: {consultation_count}회)")
                    else:
                        st.session_state.customer_data = None
                        print("ℹ️ 이전 응대 이력이 없는 신규 고객입니다.")
                except Exception as e:
                    print(f"⚠️ 고객 데이터 자동 검색 중 오류: {e}")
                    st.session_state.customer_data = None

            # 고객 검증 상태 초기화
            is_login_inquiry = check_if_login_related_inquiry(customer_query)
            if is_login_inquiry:
                st.session_state.is_customer_verified = False
                st.session_state.verification_stage = "WAIT_VERIFICATION"

                st.session_state.verification_info = {
                    "receipt_number": "",
                    "card_last4": "",
                    "customer_name": "",
                    "customer_email": st.session_state.customer_email,
                    "customer_phone": st.session_state.customer_phone,
                    "file_uploaded": False,
                    "file_info": None,
                    "verification_attempts": 0
                }
            else:
                st.session_state.is_customer_verified = True
                st.session_state.verification_stage = "NOT_REQUIRED"

            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None

            # 고객 첫 메시지 추가
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_query}
            )
            # ⭐ 초기 메시지 추가 시 즉시 화면 업데이트
            # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨

            # 언어 자동 감지
            try:
                detected_lang = detect_text_language(customer_query)
                if detected_lang not in ["ko", "en", "ja"]:
                    detected_lang = current_lang
                else:
                    if detected_lang != current_lang:
                        st.session_state.language = detected_lang
                        st.info(
                            f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
            except Exception as e:
                print(f"Language detection failed: {e}")
                detected_lang = current_lang

            # 고객 프로필 분석 (컴팩트하게)
            customer_profile = analyze_customer_profile(
                customer_query, detected_lang)
            similar_cases = find_similar_cases(
                customer_query, customer_profile, detected_lang, limit=5)

            # 프로필 분석을 expander로 감싸서 기본적으로 접힘
            with st.expander(f"📊 {L.get('customer_profile_analysis', '고객 프로필 분석')}", expanded=False):
                profile_chart = visualize_customer_profile_scores(
                    customer_profile, detected_lang)
                if profile_chart:
                    st.plotly_chart(profile_chart, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})  # 높이 제한 제거 (deprecated 파라미터)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        gender_display = customer_profile.get("gender", "unknown")
                        if gender_display == "male":
                            gender_display = "남자"
                        elif gender_display == "female":
                            gender_display = "여자"
                        else:
                            gender_display = "알 수 없음"
                        st.metric("성별", gender_display)
                    with col2:
                        st.metric(
                            L.get("sentiment_score_label", "감정 점수"),
                            f"{customer_profile.get('sentiment_score', 50)}/100"
                        )
                    with col3:
                        urgency_map = {"low": 25, "medium": 50, "high": 75}
                        urgency_score = urgency_map.get(
                            customer_profile.get(
                                "urgency_level", "medium").lower(), 50)
                        st.metric(
                            L.get("urgency_score_label", "긴급도"),
                            f"{urgency_score}/100"
                        )
                    with col4:
                        st.metric(
                            L.get(
                                "customer_type_label", "고객 유형"), customer_profile.get(
                                "predicted_customer_type", "normal"))

            # 유사 케이스 시각화 (컴팩트하게)
            if similar_cases:
                with st.expander(f"🔍 {L.get('similar_case_recommendation', '유사 케이스 추천')} ({len(similar_cases)}{L.get('count_unit', '개')})", expanded=False):
                    similarity_chart = visualize_similarity_cases(
                        similar_cases, detected_lang)
                    if similarity_chart:
                        st.plotly_chart(similarity_chart, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})  # 높이 제한 제거 (deprecated 파라미터)

                    with st.expander(f"💡 {len(similar_cases)}{L.get('similar_cases_detail_info', '개 유사 케이스 상세 정보')}", expanded=False):
                        for idx, similar_case in enumerate(similar_cases, 1):
                            case = similar_case["case"]
                            summary = similar_case["summary"]
                            similarity = similar_case["similarity_score"]
                            st.markdown(f"### {L.get('case_similarity_format', '케이스 {num} (유사도: {similarity}%)').format(num=idx, similarity=f'{similarity:.1f}')}")
                            st.markdown(
                                f"**{L.get('inquiry_content_label', '문의 내용:')}** {summary.get('main_inquiry', 'N/A')}")
                            st.markdown(
                                f"**{L.get('sentiment_score_label_short', '감정 점수:')}** {summary.get('customer_sentiment_score', 50)}/100")
                            st.markdown(
                                f"**{L.get('satisfaction_score_label_short', '만족도 점수:')}** {summary.get('customer_satisfaction_score', 50)}/100")
                            if summary.get("key_responses"):
                                st.markdown(f"**{L.get('key_response_label', '핵심 응답:')}**")
                                for response in summary.get(
                                        "key_responses", [])[:3]:
                                    st.markdown(f"- {response[:100]}...")
                            st.markdown("---")

            st.session_state.initial_advice_provided = False

            save_simulation_history_local(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.simulator_messages,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
                is_chat_ended=False,
            )
            # ⭐ 고객 체험 모드일 때는 AGENT_TURN으로 이동하여 AI가 자동 응답
            # ⭐ 상담원 테스트 모드일 때도 AGENT_TURN으로 이동하여 자동 응답 전송
            perspective = st.session_state.get("sim_perspective", "AGENT")
            st.session_state.sim_stage = "AGENT_TURN"
            
            # 응대 초안 자동 생성을 위한 플래그 초기화
            st.session_state.auto_draft_generated = False
            st.session_state.auto_generated_draft_text = ""
            st.session_state.auto_draft_auto_sent = False
            st.session_state.pending_customer_reaction = False
            # ⭐ 자동 응답 비활성화 플래그 초기화 (새로운 채팅 시작 시)
            st.session_state.auto_response_disabled = False
            st.session_state.requires_agent_response = False
            
            # ⭐ 고객 모드일 때는 AI 응답 생성 플래그 설정
            if perspective == "CUSTOMER":
                st.session_state.ai_agent_response_generated = False
            else:
                # ⭐ 상담원 모드: 초기 문의 입력 시 자동 응답 즉시 생성 및 전송 플래그 설정
                st.session_state.need_auto_response_on_agent_turn = True
            
            # ⭐ 화면 즉시 업데이트하여 자동 응답 생성 및 전송 트리거
            # st.rerun()  # 주석 처리: 버튼 클릭 후 Streamlit이 자동 rerun함

