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

def render_initial_setup():
    """초기 문의 입력 UI 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 3. 초기 문의 입력 (WAIT_FIRST_QUERY) - app.py 스타일: 바로 시작
    # ========================================
    if st.session_state.sim_stage == "WAIT_FIRST_QUERY":
        # admin.py 스타일: 깔끔한 레이아웃
        customer_query = st.text_area(
            L["customer_query_label"],
            key="customer_query_text_area",
            height=150,
            placeholder=L["initial_query_sample"],
        )

        st.divider()
        
        # 필수 입력 필드 (admin.py 스타일: 간단한 컬럼 구조)
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
        # 세션 상태 업데이트
        st.session_state.customer_email = customer_email
        st.session_state.customer_phone = customer_phone

        # 고객 유형 선택 (admin.py 스타일: 간단한 레이아웃)
        customer_type_options = L["customer_type_options"]
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="customer_type_sim_select_widget",
        )

        # ⭐ 수정: 고객 파일 업로드 기능 제거 (채팅/이메일 탭에서)
        # 첨부 파일 관련 상태 초기화
        st.session_state.customer_attachment_file = None
        st.session_state.sim_attachment_context_for_llm = ""

        st.divider()
        
        # ⭐ 수정: app.py 스타일로 바로 시작 (중복 기능 제거)
        # 채팅 시작 버튼 (간단한 버튼, "응대 조언 요청" 중복 기능 제거)
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button(L.get("button_start_chat", "채팅 시작"), key=f"btn_start_chat_{st.session_state.sim_instance_id}", use_container_width=True, type="primary"):
                if not customer_query.strip():
                    st.warning(L["simulation_warning_query"])
                    # st.stop()

                # --- 필수 입력 필드 검증 (요청 3 반영: 검증 로직 추가) ---
                if not st.session_state.customer_email.strip() or not st.session_state.customer_phone.strip():
                    st.error(L["error_mandatory_contact"])
                    # st.stop()
                # ------------------------------------------

                # 초기 상태 리셋
                st.session_state.simulator_messages = []
                st.session_state.simulator_memory.clear()
                st.session_state.is_chat_ended = False
                st.session_state.initial_advice_provided = False
                st.session_state.is_solution_provided = False  # 솔루션 플래그 리셋
                st.session_state.language_transfer_requested = False  # 언어 요청 플래그 리셋
                st.session_state.transfer_summary_text = ""  # 이관 요약 리셋
                st.session_state.start_time = None  # AHT 타이머 초기화 (첫 고객 반응 후 시작)
                st.session_state.sim_instance_id = str(uuid.uuid4())  # 새 시뮬레이션 ID 할당
                
                # ⭐ 추가: UI 플래그 초기화 (채팅 시작 시 모든 기능 UI 숨김, 에이전트 응답 입력만 표시)
                st.session_state.show_verification_ui = False
                st.session_state.show_draft_ui = False
                st.session_state.show_customer_data_ui = False
                st.session_state.show_agent_response_ui = False
                
                # 고객 검증 상태 초기화 (로그인/계정 관련 문의인 경우)
                is_login_inquiry = check_if_login_related_inquiry(customer_query)
                if is_login_inquiry:
                    # 검증 정보 초기화 및 고객이 제공한 정보를 시스템 검증 정보로 저장 (시뮬레이션용)
                    # 실제로는 DB에서 가져와야 하지만, 시뮬레이션에서는 고객이 제공한 정보를 저장
                    st.session_state.is_customer_verified = False
                    st.session_state.verification_stage = "WAIT_VERIFICATION"
                    
                    # ⭐ 수정: 고객 파일 업로드 기능 제거로 인해 첨부 파일 정보 없음
                    file_info_for_storage = None
                    
                    st.session_state.verification_info = {
                        "receipt_number": "",  # 실제로는 DB에서 가져와야 함
                        "card_last4": "",  # 실제로는 DB에서 가져와야 함
                        "customer_name": "",  # 실제로는 DB에서 가져와야 함
                        "customer_email": st.session_state.customer_email,  # 고객이 제공한 정보
                        "customer_phone": st.session_state.customer_phone,  # 고객이 제공한 정보
                        "file_uploaded": False,  # 채팅/이메일 탭에서는 파일 업로드 기능 제거
                        "file_info": None,  # 첨부 파일 상세 정보 없음
                        "verification_attempts": 0
                    }
                else:
                    # 로그인 관련 문의가 아닌 경우 검증 불필요
                    st.session_state.is_customer_verified = True
                    st.session_state.verification_stage = "NOT_REQUIRED"
                # 전화 발신 관련 상태 초기화
                st.session_state.sim_call_outbound_summary = ""
                st.session_state.sim_call_outbound_target = None

                # 1) 고객 첫 메시지 추가
                st.session_state.simulator_messages.append(
                    {"role": "customer", "content": customer_query}
                )

                # 2) Supervisor 가이드 + 초안 생성
                # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
                try:
                    detected_lang = detect_text_language(customer_query)
                    # 감지된 언어가 유효한지 확인
                    if detected_lang not in ["ko", "en", "ja"]:
                        detected_lang = current_lang
                    else:
                        # 언어가 감지되었고 현재 언어와 다르면 자동으로 언어 설정 업데이트
                        if detected_lang != current_lang:
                            st.session_state.language = detected_lang
                            st.info(f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
                except Exception as e:
                    print(f"Language detection failed: {e}")
                    detected_lang = current_lang  # 기본값으로 폴백
                
                # 고객 프로필 분석 (시각화를 위해 먼저 수행, 감지된 언어 사용)
                customer_profile = analyze_customer_profile(customer_query, detected_lang)
                similar_cases = find_similar_cases(customer_query, customer_profile, detected_lang, limit=5)

                # 시각화 차트 표시
                st.markdown("---")
                st.subheader("📊 고객 프로필 분석")

                # 고객 프로필 점수 차트 (감지된 언어 사용)
                profile_chart = visualize_customer_profile_scores(customer_profile, detected_lang)
                if profile_chart:
                    st.plotly_chart(profile_chart, use_container_width=True)
                else:
                    # Plotly가 없을 경우 텍스트로 표시
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        gender_display = customer_profile.get("gender", "unknown")
                        if gender_display == "male":
                            gender_display = "남자"
                        elif gender_display == "female":
                            gender_display = "여자"
                        else:
                            gender_display = "알 수 없음"
                        st.metric(
                            "성별",
                            gender_display
                        )
                    with col2:
                        st.metric(
                            L.get("sentiment_score_label", "감정 점수"),
                            f"{customer_profile.get('sentiment_score', 50)}/100"
                        )
                    with col3:
                        urgency_map = {"low": 25, "medium": 50, "high": 75}
                        urgency_score = urgency_map.get(customer_profile.get("urgency_level", "medium").lower(), 50)
                        st.metric(
                            L.get("urgency_score_label", "긴급도"),
                            f"{urgency_score}/100"
                        )
                    with col4:
                        st.metric(
                            L.get("customer_type_label", "고객 유형"),
                            customer_profile.get("predicted_customer_type", "normal")
                        )

                # 유사 케이스 시각화
                if similar_cases:
                    st.markdown("---")
                    st.subheader("🔍 유사 케이스 추천")
                    similarity_chart = visualize_similarity_cases(similar_cases, detected_lang)
                    if similarity_chart:
                        st.plotly_chart(similarity_chart, use_container_width=True)

                    # 유사 케이스 요약 표시
                    with st.expander(f"💡 {len(similar_cases)}개 유사 케이스 상세 정보"):
                        for idx, similar_case in enumerate(similar_cases, 1):
                            case = similar_case["case"]
                            summary = similar_case["summary"]
                            similarity = similar_case["similarity_score"]
                            st.markdown(f"### 케이스 {idx} (유사도: {similarity:.1f}%)")
                            st.markdown(f"**문의 내용:** {summary.get('main_inquiry', 'N/A')}")
                            st.markdown(f"**감정 점수:** {summary.get('customer_sentiment_score', 50)}/100")
                            st.markdown(f"**만족도 점수:** {summary.get('customer_satisfaction_score', 50)}/100")
                            if summary.get("key_responses"):
                                st.markdown("**핵심 응답:**")
                                for response in summary.get("key_responses", [])[:3]:
                                    st.markdown(f"- {response[:100]}...")
                            st.markdown("---")

                # ⭐ 수정: 자동으로 응대 가이드라인/초안 생성하지 않음 (버튼 클릭 시에만 생성)
                # 초기 조언은 버튼을 통해 수동으로 생성하도록 변경
                # st.session_state.initial_advice_provided는 버튼 클릭 시 설정됨
                st.session_state.initial_advice_provided = False
                
                # ⭐ 수정: AGENT_TURN으로 자동 변경하지 않음 (응대 가이드라인 버튼 클릭 시에만 변경)
                # 채팅 시작 후 고객 메시지가 표시되고, 버튼을 통해 기능 사용 가능
                save_simulation_history_local(
                    customer_query,
                    st.session_state.customer_type_sim_select,
                    st.session_state.simulator_messages,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                    is_chat_ended=False,
                )
                # ⭐ 수정: 에이전트 인사말 자동 생성 제거 - 에이전트가 직접 입력하도록 변경
                # 채팅 시작 시 고객의 초기 문의만 추가하고, 에이전트가 직접 인사말을 입력하도록 함
                # sim_stage는 AGENT_TURN으로 변경 (에이전트가 인사말을 입력할 수 있도록)
                st.session_state.sim_stage = "AGENT_TURN"