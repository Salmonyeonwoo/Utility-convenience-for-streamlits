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

    # 4. 대화 로그 표시 (공통)
    # =========================
    
    # 피드백 저장 콜백 함수
    def save_feedback(index):
        # 에이전트 응답에 대한 고객 피드백을 저장
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{index}"
        if feedback_key in st.session_state:
            feedback_value = st.session_state[feedback_key]
            # 메시지에 피드백 정보 저장
            if index < len(st.session_state.simulator_messages):
                st.session_state.simulator_messages[index]["feedback"] = feedback_value
    
    # ⭐ 카카오톡 스타일 채팅 UI CSS 추가
    st.markdown("""
    <style>
    /* 카카오톡 스타일 채팅 UI */
    .stChatMessage {
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 12px;
        max-width: 70%;
    }
    .stChatMessage[data-testid="user"] {
        background-color: #FEE500;
        margin-left: auto;
        margin-right: 0;
    }
    .stChatMessage[data-testid="assistant"] {
        background-color: #F5F5F5;
        margin-left: 0;
        margin-right: auto;
    }
    /* 작은 아이콘 버튼 스타일 */
    .compact-icon-button {
        padding: 4px 8px;
        font-size: 14px;
        min-width: auto;
        height: 28px;
    }
    /* 메시지 말풍선 내부 버튼 그룹 */
    .message-action-buttons {
        display: flex;
        gap: 4px;
        margin-top: 8px;
        flex-wrap: wrap;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 채팅/이메일 탭에서만 메시지 표시
    # ⭐ 카카오톡 스타일 채팅 UI
    if st.session_state.simulator_messages:
        for idx, msg in enumerate(st.session_state.simulator_messages):
            # ⭐ 수정: 안전한 딕셔너리 접근
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not role or not content:
                continue
            
            # 역할에 따른 표시 이름 및 아바타 설정
            if role == "customer" or role == "customer_rebuttal" or role == "initial_query":
                display_role = "user"
                avatar = "🙋"
            elif role == "agent_response":
                display_role = "assistant"
                avatar = "🧑‍💻"
            elif role == "supervisor":
                display_role = "assistant"
                avatar = "🤖"
            else:
                display_role = "assistant"
                avatar = "💬"
            
            with st.chat_message(display_role, avatar=avatar):
                st.write(content)
                
                # ⭐ 가이드라인 메시지는 메시지로만 표시 (에이전트 응답 UI는 AGENT_TURN 섹션에서 항상 표시)
                # 가이드라인 메시지 아래의 UI는 제거됨
                
                # ⭐ 메시지 말풍선 안에 버튼들 추가 (영상 스타일)
                # 버튼 레이아웃: 역할에 따라 다른 버튼 표시
                
                # 1. 음성으로 듣기 버튼 (모든 메시지에)
                tts_role = "customer" if role.startswith("customer") or role == "customer_rebuttal" else (
                    "agent" if role == "agent_response" else "supervisor")
                render_tts_button(content, st.session_state.language, role=tts_role, prefix=f"{role}_", index=idx)
                
                # 2. 에이전트 응답에 피드백 버튼만 표시 (응대 힌트, 전화 버튼은 입력 칸으로 이동)
                if role == "agent_response":
                    # 피드백 버튼 (기존 유지)
                    feedback_key = f"feedback_{st.session_state.sim_instance_id}_{idx}"
                    existing_feedback = msg.get("feedback", None)
                    if existing_feedback is not None:
                        st.session_state[feedback_key] = existing_feedback
                    
                    st.feedback(
                        "thumbs",
                        key=feedback_key,
                        disabled=existing_feedback is not None,
                        on_change=save_feedback,
                        args=[idx],
                    )
                
                # 3. 고객 메시지에 작은 아이콘 버튼들 (카카오톡 스타일)
                if role == "customer" or role == "customer_rebuttal":
                    # 작은 아이콘 버튼들 (한 줄에 여러 개)
                    icon_cols = st.columns([1, 1, 1, 1, 1, 1])
                    
                    # 응대 힌트 아이콘 버튼
                    with icon_cols[0]:
                        if st.button("💡", key=f"hint_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_hint", "응대 힌트"), use_container_width=True):
                            if st.session_state.is_llm_ready:
                                # 다른 플래그들 초기화 (하나만 보이도록)
                                st.session_state.show_verification_ui = False
                                st.session_state.show_draft_ui = False
                                st.session_state.show_customer_data_ui = False
                                st.session_state.show_agent_response_ui = False
                                
                                # ⭐ 수정: 이전 힌트 메시지 제거 (같은 타입의 supervisor 메시지 제거)
                                hint_label = L.get('hint_label', '응대 힌트')
                                st.session_state.simulator_messages = [
                                    msg for msg in st.session_state.simulator_messages 
                                    if not (msg.get("role") == "supervisor" and hint_label in msg.get("content", ""))
                                ]
                                
                                # ⭐ 수정: 세션 언어 설정을 직접 전달
                                session_lang = st.session_state.get("language", "ko")
                                if session_lang not in ["ko", "en", "ja"]:
                                    session_lang = "ko"
                                
                                with st.spinner(L.get("response_generating", "생성 중...")):
                                    hint = generate_realtime_hint(session_lang, is_call=False)
                                    st.session_state.realtime_hint_text = hint
                                    # 힌트를 supervisor 메시지로 추가하여 표시
                                    st.session_state.simulator_messages.append({
                                        "role": "supervisor",
                                        "content": f"💡 **{L.get('hint_label', '응대 힌트')}**: {hint}"
                                    })
                            else:
                                st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
                    
                    # 업체 전화 아이콘 버튼
                    with icon_cols[1]:
                        if st.button("📞", key=f"call_provider_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_call_company", "업체에 전화"), use_container_width=True):
                            st.session_state.show_verification_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = False
                            st.session_state.show_agent_response_ui = False
                            st.session_state.sim_call_outbound_target = L.get("call_target_provider", "현지 업체/파트너")
                            st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                    
                    # 고객 전화 아이콘 버튼
                    with icon_cols[2]:
                        if st.button("📱", key=f"call_customer_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_call_customer", "고객에게 전화"), use_container_width=True):
                            st.session_state.show_verification_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = False
                            st.session_state.show_agent_response_ui = False
                            st.session_state.sim_call_outbound_target = L.get("call_target_customer", "고객")
                            st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                    
                    # AI 응대 가이드라인 아이콘 버튼
                    with icon_cols[3]:
                        if st.button("📋", key=f"guideline_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_ai_guideline", "AI 응대 가이드라인"), use_container_width=True):
                            if st.session_state.is_llm_ready:
                                # 다른 플래그들 초기화 (하나만 보이도록)
                                st.session_state.show_verification_ui = False
                                st.session_state.show_draft_ui = False
                                st.session_state.show_customer_data_ui = False
                                st.session_state.show_agent_response_ui = False  # 가이드라인은 메시지만 표시
                                
                                # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                                guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
                                draft_label = L.get('draft_label', '응대 초안')
                                customer_data_label = L.get('customer_data_label', '고객 데이터')
                                customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
                                st.session_state.simulator_messages = [
                                    msg for msg in st.session_state.simulator_messages 
                                    if not (msg.get("role") == "supervisor" and (
                                        guideline_label in msg.get("content", "") or
                                        draft_label in msg.get("content", "") or
                                        customer_data_label in msg.get("content", "") or
                                        customer_data_loaded in msg.get("content", "")
                                    ))
                                ]
                                
                                with st.spinner(L.get("generating_guideline", "AI 응대 가이드라인 생성 중...")):
                                    # 초기 문의 가져오기
                                    initial_query = st.session_state.get('customer_query_text_area', content)
                                    customer_type_display = st.session_state.get("customer_type_sim_select", "")
                                    
                                    # ⭐ 수정: 세션 언어 설정을 직접 전달
                                    session_lang = st.session_state.get("language", "ko")
                                    if session_lang not in ["ko", "en", "ja"]:
                                        session_lang = "ko"
                                    
                                    # 응대 가이드라인 생성
                                    guideline_text = _generate_initial_advice(
                                        initial_query,
                                        customer_type_display,
                                        st.session_state.customer_email,
                                        st.session_state.customer_phone,
                                        session_lang,
                                        st.session_state.customer_attachment_file
                                    )
                                    
                                    # 가이드라인을 supervisor 메시지로 추가하여 표시
                                    st.session_state.simulator_messages.append({
                                        "role": "supervisor",
                                        "content": f"📋 **{L.get('guideline_label', 'AI 응대 가이드라인')}**:\n\n{guideline_text}"
                                    })
                                    
                                    # AGENT_TURN 단계로 변경하여 에이전트 응답 UI 표시 (항상 표시됨)
                                    st.session_state.sim_stage = "AGENT_TURN"
                            else:
                                st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
                    
                    # 고객 데이터 아이콘 버튼
                    with icon_cols[4]:
                        if st.button("👤", key=f"customer_data_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_customer_data", "고객 데이터"), use_container_width=True):
                            # 다른 플래그들 초기화 (하나만 보이도록)
                            st.session_state.show_agent_response_ui = False
                            st.session_state.show_verification_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = True
                            
                            # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                            guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
                            draft_label = L.get('draft_label', '응대 초안')
                            customer_data_label = L.get('customer_data_label', '고객 데이터')
                            customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
                            st.session_state.simulator_messages = [
                                msg for msg in st.session_state.simulator_messages 
                                if not (msg.get("role") == "supervisor" and (
                                    guideline_label in msg.get("content", "") or
                                    draft_label in msg.get("content", "") or
                                    customer_data_label in msg.get("content", "") or
                                    customer_data_loaded in msg.get("content", "")
                                ))
                            ]
                            
                            # 고객 ID는 이메일 또는 전화번호 기반으로 생성
                            customer_id = st.session_state.get("customer_email", "") or st.session_state.get("customer_phone", "")
                            if not customer_id:
                                customer_id = f"customer_{st.session_state.sim_instance_id}"
                            
                            # 고객 데이터 불러오기
                            customer_data = st.session_state.customer_data_manager.load_customer_data(customer_id)
                            
                            # ⭐ 추가: 누적 데이터 수 자동 확인
                            try:
                                all_customers = st.session_state.customer_data_manager.list_all_customers()
                                total_customers = len(all_customers)
                            except Exception:
                                total_customers = 0
                            
                            if customer_data:
                                st.session_state.customer_data = customer_data
                                customer_info = customer_data.get("data", {})
                                
                                # 고객 데이터를 supervisor 메시지로 추가하여 표시
                                info_message = f"📋 **{L.get('customer_data_loaded', '고객 데이터 불러옴')}**\n\n"
                                info_message += f"**{L.get('basic_info_label', '기본 정보')}:**\n"
                                info_message += f"- {L.get('name_label', '이름')}: {customer_info.get('name', 'N/A')}\n"
                                info_message += f"- {L.get('email_label', '이메일')}: {customer_info.get('email', 'N/A')}\n"
                                info_message += f"- {L.get('phone_label', '전화번호')}: {customer_info.get('phone', 'N/A')}\n"
                                info_message += f"- {L.get('company_label', '회사')}: {customer_info.get('company', 'N/A')}\n"
                                
                                # 누적 데이터 수 표시
                                info_message += f"\n**{L.get('accumulated_data_label', '누적 데이터')}:**\n"
                                info_message += f"- {L.get('total_customers_label', '총 고객 수')}: {total_customers}{L.get('cases_label', '건')}\n"
                                
                                if customer_info.get('purchase_history'):
                                    info_message += f"\n**{L.get('purchase_history_label', '구매 이력')}:** ({len(customer_info.get('purchase_history', []))}{L.get('cases_label', '건')})\n"
                                    for purchase in customer_info.get('purchase_history', [])[:5]:
                                        info_message += f"- {purchase.get('date', 'N/A')}: {purchase.get('item', 'N/A')} ({purchase.get('amount', 0):,}{L.get('currency_unit', '원')})\n"
                                if customer_info.get('notes'):
                                    info_message += f"\n**{L.get('notes_label', '메모')}:** {customer_info.get('notes', 'N/A')}"
                                
                                st.session_state.simulator_messages.append({
                                    "role": "supervisor",
                                    "content": info_message
                                })
                            else:
                                # 고객 데이터가 없으면 안내 메시지 (누적 데이터 수 포함)
                                info_message = f"📋 **{L.get('customer_data_label', '고객 데이터')}**: {L.get('no_customer_data', '저장된 고객 데이터가 없습니다.')}\n\n"
                                info_message += f"**{L.get('accumulated_data_label', '누적 데이터')}**: {L.get('total_label', '총')} {total_customers}{L.get('cases_label', '건')}"
                                st.session_state.simulator_messages.append({
                                    "role": "supervisor",
                                    "content": info_message
                                })
                    
                    # 응대 초안 아이콘 버튼
                    with icon_cols[5]:
                        if st.button("✍️", key=f"draft_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_draft", "응대 초안"), use_container_width=True):
                            if st.session_state.is_llm_ready:
                                # 다른 플래그들 초기화 (하나만 보이도록)
                                st.session_state.show_agent_response_ui = False
                                st.session_state.show_verification_ui = False
                                st.session_state.show_customer_data_ui = False
                                st.session_state.show_draft_ui = True
                                
                                # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                                guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
                                draft_label = L.get('draft_label', '응대 초안')
                                customer_data_label = L.get('customer_data_label', '고객 데이터')
                                customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
                                st.session_state.simulator_messages = [
                                    msg for msg in st.session_state.simulator_messages 
                                    if not (msg.get("role") == "supervisor" and (
                                        guideline_label in msg.get("content", "") or
                                        draft_label in msg.get("content", "") or
                                        customer_data_label in msg.get("content", "") or
                                        customer_data_loaded in msg.get("content", "")
                                    ))
                                ]
                                
                                with st.spinner(L.get("generating_draft", "응대 초안 생성 중...")):
                                    # 초기 문의 가져오기
                                    initial_query = st.session_state.get('customer_query_text_area', content)
                                    customer_type_display = st.session_state.get("customer_type_sim_select", "")
                                    
                                    # ⭐ 수정: 세션 언어 설정을 직접 전달
                                    session_lang = st.session_state.get("language", "ko")
                                    if session_lang not in ["ko", "en", "ja"]:
                                        session_lang = "ko"
                                    
                                    # 응대 초안 생성 (가이드라인과 동일한 함수 사용)
                                    draft_text = _generate_initial_advice(
                                        initial_query,
                                        customer_type_display,
                                        st.session_state.customer_email,
                                        st.session_state.customer_phone,
                                        session_lang,
                                        st.session_state.customer_attachment_file
                                    )
                                    
                                    # 초안을 supervisor 메시지로 추가하여 표시
                                    st.session_state.simulator_messages.append({
                                        "role": "supervisor",
                                        "content": f"✍️ **{L.get('draft_label', '응대 초안')}**:\n\n{draft_text}"
                                    })
                            else:
                                st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
                    
                    # 고객 검증 아이콘 버튼 (별도 행에 배치)
                    verification_col = st.columns([1, 4])
                    with verification_col[0]:
                        if st.button("🔐", key=f"verification_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_verification", "고객 검증"), use_container_width=False):
                            # 다른 플래그들 초기화 (하나만 보이도록)
                            st.session_state.show_agent_response_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = False
                            # 검증 UI 표시를 위한 플래그 설정
                            st.session_state.show_verification_ui = True
                            st.session_state.verification_message_idx = idx
                            
                            # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                            guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
                            draft_label = L.get('draft_label', '응대 초안')
                            customer_data_label = L.get('customer_data_label', '고객 데이터')
                            customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
                            st.session_state.simulator_messages = [
                                msg for msg in st.session_state.simulator_messages 
                                if not (msg.get("role") == "supervisor" and (
                                    guideline_label in msg.get("content", "") or
                                    draft_label in msg.get("content", "") or
                                    customer_data_label in msg.get("content", "") or
                                    customer_data_loaded in msg.get("content", "")
                                ))
                            ]
                            
                            st.session_state.sim_stage = "AGENT_TURN"  # 검증 UI를 표시하기 위해 AGENT_TURN으로 변경
                    
                    # 마지막 에이전트 응답에서 솔루션이 제공되었는지 확인
                    last_agent_response_idx = None
                    for i in range(idx - 1, -1, -1):
                        if i < len(st.session_state.simulator_messages) and st.session_state.simulator_messages[i].get("role") == "agent_response":
                            last_agent_response_idx = i
                            break
                    
                    # 솔루션 제공 여부 확인
                    solution_provided = False
                    if last_agent_response_idx is not None:
                        agent_msg_content = st.session_state.simulator_messages[last_agent_response_idx].get("content", "")
                        solution_keywords = ["해결", "도움", "안내", "제공", "solution", "help", "assist", "guide", "안내해드리", "도와드리"]
                        solution_provided = any(keyword in agent_msg_content.lower() for keyword in solution_keywords)
                    
                    # "알겠습니다" 또는 "감사합니다"가 포함된 경우 추가 문의 여부 확인 버튼 표시 (admin.py 스타일)
                    if solution_provided or st.session_state.is_solution_provided:
                        if "알겠습니다" in content or "감사합니다" in content or "ok" in content.lower() or "thank" in content.lower():
                            if st.button(L.get("button_additional_inquiry", "✅ 추가 문의 있나요?"), key=f"additional_inquiry_{idx}_{st.session_state.sim_instance_id}", use_container_width=True, type="secondary"):
                                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                    
                    # 4. 고객이 "없습니다. 감사합니다" 답변 시 설문 조사 버튼 (admin.py 스타일)
                    no_more_keywords = [
                        "없습니다", "감사합니다", "No, that will be all", "no more",
                        "추가 문의 사항 없습니다", "추가 문의사항 없습니다", "no additional", "結構です"
                    ]
                    # 키워드가 모두 포함되어 있거나 "없습니다"와 "감사합니다"가 함께 있는 경우
                    has_no_more = (
                        any(keyword in content for keyword in no_more_keywords) or
                        ("없습니다" in content and "감사합니다" in content) or
                        ("no" in content.lower() and "more" in content.lower() and "thank" in content.lower())
                    )
                    
                    if has_no_more:
                        if st.button(L.get("button_survey_end", "📋 설문 조사 전송 및 종료"), key=f"survey_end_{idx}_{st.session_state.sim_instance_id}", use_container_width=True, type="primary"):
                            # AHT 타이머 정지
                            st.session_state.start_time = None
                            
                            # 설문 조사 링크 전송 메시지 추가
                            end_msg = L.get("prompt_survey", "설문 조사 링크를 전송했습니다.")
                            st.session_state.simulator_messages.append(
                                {"role": "system_end", "content": end_msg}
                            )
                            
                            # 채팅 종료 처리
                            customer_type_display = st.session_state.get("customer_type_sim_select", "")
                            st.session_state.is_chat_ended = True
                            st.session_state.sim_stage = "CLOSING"
                            
                            # 이력 저장
                            save_simulation_history_local(
                                st.session_state.customer_query_text_area, customer_type_display,
                                st.session_state.simulator_messages, is_chat_ended=True,
                                attachment_context=st.session_state.sim_attachment_context_for_llm,
                            )
                            
                            # ⭐ 재실행 불필요: 이력 저장만으로 충분, 자동 업데이트됨
                            # st.rerun()

                # 고객 첨부 파일 표시 (기능 유지)
                if idx == 0 and role == "customer" and st.session_state.customer_attachment_b64:
                    mime = st.session_state.customer_attachment_mime or "image/png"
                    data_url = f"data:{mime};base64,{st.session_state.customer_attachment_b64}"

                    if mime.startswith("image/"):
                        caption_text = L.get("attachment_evidence_caption", "첨부된 증거물").format(filename=st.session_state.customer_attachment_file.name)
                        st.image(data_url, caption=caption_text, use_column_width=True)
                    elif mime == "application/pdf":
                        warning_text = L.get("attachment_pdf_warning", "첨부된 PDF 파일 ({filename})은 현재 인라인 미리보기가 지원되지 않습니다.").format(filename=st.session_state.customer_attachment_file.name)
                        st.warning(warning_text)

    # 이관 요약 표시 (이관 후에만) - ⭐ 수정: AI 응대 가이드라인 위에서는 표시하지 않음
    # AGENT_TURN 단계가 아니거나, 가이드라인/초안/고객데이터 UI가 표시되지 않을 때만 표시
    show_guideline_ui = st.session_state.get("show_draft_ui", False) or st.session_state.get("show_customer_data_ui", False)
    should_show_transfer_summary = (
        (st.session_state.transfer_summary_text or (st.session_state.language != st.session_state.language_at_transfer_start and st.session_state.language_at_transfer_start)) and
        st.session_state.sim_stage != "AGENT_TURN" and not show_guideline_ui
    )
    if should_show_transfer_summary:
                st.markdown("---")
                st.markdown(f"**{L['transfer_summary_header']}**")
                st.info(L["transfer_summary_intro"])

                # ⭐ [수정] 번역 성공 여부 확인 및 요약 표시
                is_translation_failed = not st.session_state.get("translation_success", True) or not st.session_state.transfer_summary_text

                # 번역 성공 시 요약 표시
                if st.session_state.transfer_summary_text and st.session_state.get("translation_success", True):
                    st.markdown(st.session_state.transfer_summary_text)
                elif st.session_state.transfer_summary_text:
                    # 번역 실패 시에도 원본 텍스트 표시
                    st.info(st.session_state.transfer_summary_text)
    
    # ⭐ 번역 재시도 버튼 (언어 이관 시 항상 표시 - should_show_transfer_summary 조건 밖으로 이동)
    # 채팅/이메일 탭에서 언어가 이관되었을 때 항상 번역 재시도 버튼 표시
    if (st.session_state.language_at_transfer_start and 
        st.session_state.language != st.session_state.language_at_transfer_start and
        st.session_state.get("feature_selection") == L["sim_tab_chat_email"]):
        st.markdown("---")
        st.markdown("**번역 재시도**")
        if st.button(L.get("button_retry_translation", "번역 다시 시도"),
                     key=f"btn_retry_translation_chat_{st.session_state.language_at_transfer_start}_{st.session_state.language}_{st.session_state.sim_instance_id}"):
            # 재시도 로직 실행
            try:
                source_lang = st.session_state.language_at_transfer_start
                target_lang = st.session_state.language
                
                if not source_lang or not target_lang:
                    st.error(L.get("invalid_language_info", "언어 정보가 올바르지 않습니다."))
                else:
                    # ⭐ 수정: 원본 언어로 요약을 먼저 생성한 후 번역 (전화 탭과 동일한 로직)
                    with st.spinner(L.get("transfer_loading", "번역 중...")):
                        # 원본 언어로 요약 생성
                        original_summary = summarize_history_with_ai(source_lang)
                        
                        if original_summary and not original_summary.startswith("❌"):
                            # 원본 핵심 요약을 번역 대상 언어로 번역
                            translated_summary, is_success = translate_text_with_llm(
                                original_summary,
                                target_lang,
                                source_lang
                            )
                            
                            if not translated_summary or not is_success:
                                # 번역 실패 시 현재 언어로 요약 재생성
                                translated_summary = summarize_history_with_ai(target_lang)
                                is_success = True if translated_summary and not translated_summary.startswith("❌") else False
                        else:
                            # 원본 요약 생성 실패 시 현재 언어로 요약 생성
                            translated_summary = summarize_history_with_ai(target_lang)
                            is_success = True if translated_summary and not translated_summary.startswith("❌") else False
                        
                        if not translated_summary:
                            st.warning(L.get("translation_empty", "번역 결과가 비어있습니다. 원본 텍스트를 사용합니다."))
                            translated_summary = original_summary if original_summary else ""
                            is_success = False
                        
                        # ⭐ [수정] 번역 재시도 시에도 배치 번역 사용 (요약 번역과 별도로 메시지 번역)
                        translated_messages = []
                        messages_to_translate = []
                        
                        # 번역할 메시지 수집
                        for idx, msg in enumerate(st.session_state.simulator_messages):
                            if not isinstance(msg, dict):
                                continue
                            translated_msg = msg.copy()
                            msg_role = msg.get("role", "")
                            if msg_role in ["initial_query", "customer", "customer_rebuttal", "agent_response", 
                                              "customer_closing_response", "supervisor"]:
                                if msg.get("content"):
                                    messages_to_translate.append((idx, msg))
                            translated_messages.append(translated_msg)
                        
                        # 배치 번역: 모든 메시지를 하나의 텍스트로 합쳐서 번역
                        if messages_to_translate:
                            try:
                                # 번역할 메시지들을 하나의 텍스트로 합치기
                                combined_text = "\n\n".join([
                                    f"[{msg.get('role', 'unknown')}]: {msg.get('content', '')}" 
                                    for _, msg in messages_to_translate
                                    if msg.get('content')
                                ])
                                
                                # 전체 텍스트를 한 번에 번역 (토큰 제한 고려하여 내부에서 청크 처리)
                                translated_combined, trans_success_batch = translate_text_with_llm(
                                    combined_text,
                                    target_lang,
                                    source_lang
                                )
                                
                                if trans_success_batch and translated_combined:
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
                                            source_lang
                                        )
                                        if trans_success:
                                            translated_messages[idx]["content"] = translated_content
                                    except Exception:
                                        # 개별 번역도 실패하면 원본 유지
                                        pass
                        
                        # 번역된 메시지로 업데이트
                        st.session_state.simulator_messages = translated_messages
                        
                        # 번역 결과 저장 (요약)
                        st.session_state.transfer_summary_text = translated_summary
                        st.session_state.translation_success = is_success
                    
                    # ⭐ 재실행 불필요: 버튼 클릭 시 자동으로 rerun됨
                    # st.rerun()
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(L.get("translation_retry_error", "번역 재시도 중 오류 발생: {error}").format(error=str(e)))
                st.code(error_details)
                st.session_state.transfer_summary_text = L.get("translation_error", "번역 오류: {error}").format(error=str(e))
                st.session_state.translation_success = False

    # =========================