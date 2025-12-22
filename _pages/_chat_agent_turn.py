# ========================================
# _pages/_chat_agent_turn.py
# 채팅 시뮬레이터 - 에이전트 입력 단계 처리 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import (
    generate_chat_summary, load_simulation_histories_local,
    recommend_guideline_for_customer, check_if_login_related_inquiry,
    check_if_customer_provided_verification_info, verify_customer_info,
    mask_email, transcribe_bytes_with_whisper, generate_customer_reaction,
    save_simulation_history_local, _generate_initial_advice,
    summarize_history_with_ai
)
from llm_client import get_api_key
from datetime import datetime
import re
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
import json


def render_agent_turn(L, current_lang):
    """에이전트 입력 단계 UI 렌더링"""
    show_verification_from_button = st.session_state.get(
        "show_verification_ui", False)
    show_draft_ui = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui = st.session_state.get(
        "show_customer_data_ui", False)

    if show_verification_from_button:
        pass
    elif show_draft_ui:
        pass
    elif show_customer_data_ui:
        pass
    else:
        st.markdown(f"### {L['agent_response_header']}")

    # 고객 성향 기반 가이드라인 추천
    if st.session_state.simulator_messages and len(
            st.session_state.simulator_messages) >= 2:
        try:
            temp_summary = generate_chat_summary(
                st.session_state.simulator_messages,
                st.session_state.customer_query_text_area,
                st.session_state.get("customer_type_sim_select", ""),
                st.session_state.language
            )

            if temp_summary and temp_summary.get("customer_sentiment_score"):
                all_histories = load_simulation_histories_local(
                    st.session_state.language)

                recommended_guideline = recommend_guideline_for_customer(
                    temp_summary,
                    all_histories,
                    st.session_state.language
                )

                if recommended_guideline:
                    with st.expander("💡 고객 성향 기반 응대 가이드라인 추천", expanded=False):
                        st.markdown(recommended_guideline)
                        st.caption(
                            "💡 이 가이드는 유사한 과거 고객 사례를 분석하여 자동 생성되었습니다.")
        except Exception:
            pass

    # 언어 이관 요청 강조 표시
    if st.session_state.language_transfer_requested:
        st.error(
            L.get(
                "language_transfer_requested_msg",
                "🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。"))

    # 고객 첨부 파일 정보 재표시
    if st.session_state.sim_attachment_context_for_llm:
        st.info(
            f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")

    customer_has_attachment = (
        st.session_state.customer_attachment_file is not None or
        (st.session_state.sim_attachment_context_for_llm and
         st.session_state.sim_attachment_context_for_llm.strip())
    )

    # 고객 검증 프로세스
    initial_query = st.session_state.get('customer_query_text_area', '')
    all_customer_texts = []
    if initial_query:
        all_customer_texts.append(initial_query)

    if st.session_state.simulator_messages:
        all_roles = [msg.get("role")
                     for msg in st.session_state.simulator_messages]
        customer_messages = [
            msg for msg in st.session_state.simulator_messages if msg.get("role") in [
                "customer", "customer_rebuttal", "initial_query"]]

        for msg in customer_messages:
            content = msg.get("content", "")
            if content and content not in all_customer_texts:
                all_customer_texts.append(content)

        combined_customer_text = " ".join(all_customer_texts)
        is_login_inquiry = check_if_login_related_inquiry(
            combined_customer_text)

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
                st.session_state.debug_customer_messages_count = len(
                    customer_messages)
                st.session_state.debug_combined_customer_text = combined_customer_text[:200]
    else:
        is_login_inquiry = check_if_login_related_inquiry(initial_query)
        customer_provided_info = False
        all_roles = []
        customer_messages = []

    # 고객 검증 UI 표시
    show_draft_ui_check = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui_check = st.session_state.get(
        "show_customer_data_ui", False)
    if show_verification_from_button and not show_draft_ui_check and not show_customer_data_ui_check:
        st.markdown("---")
        st.markdown(f"### {L.get('verification_header', '고객 검증')}")
        st.warning(L.get('verification_warning', '고객 검증이 필요합니다.'))

    # 검증 UI 렌더링
    if is_login_inquiry and show_verification_from_button:
        _render_verification_debug_info(L, is_login_inquiry, customer_provided_info, 
                                        customer_has_attachment, all_customer_texts, all_roles, customer_messages)

    show_draft_ui_check2 = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui_check2 = st.session_state.get("show_customer_data_ui", False)
    if is_login_inquiry and customer_provided_info and not st.session_state.is_customer_verified and show_verification_from_button and not show_draft_ui_check2 and not show_customer_data_ui_check2:
        _render_verification_ui(L, customer_has_attachment)

    elif is_login_inquiry and st.session_state.is_customer_verified:
        st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))

    # 에이전트 첨부 파일 업로더
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
            file_names = ", ".join(
                [f["name"] for f in st.session_state.agent_attachment_file])
            st.info(
                L.get(
                    "agent_attachment_files_ready",
                    "✅ {count}개 에이전트 첨부 파일 준비 완료: {files}").format(
                    count=len(agent_attachment_files),
                    files=file_names))
            st.session_state.show_agent_file_uploader = False
        else:
            st.session_state.agent_attachment_file = []
    else:
        st.session_state.agent_attachment_file = []

    # 마이크 녹음 처리
    if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process is not None:
        has_openai = st.session_state.openai_client is not None
        has_gemini = bool(get_api_key("gemini"))

        if not has_openai and not has_gemini:
            st.error(
                L.get(
                    "whisper_client_error",
                    "Whisper 클라이언트 오류") +
                " (OpenAI 또는 Gemini API Key 필요)")
            st.session_state.bytes_to_process = None
        else:
            agent_response_transcript = None
            audio_bytes_backup = st.session_state.bytes_to_process
            st.session_state.bytes_to_process = None

            with st.spinner(L.get("whisper_processing", "전사 중...")):
                try:
                    agent_response_transcript = transcribe_bytes_with_whisper(
                        audio_bytes_backup, "audio/wav", lang_code=None, auto_detect=True)
                except Exception as e:
                    agent_response_transcript = L.get(
                        "transcription_error_with_error",
                        "❌ 전사 오류: {error}").format(
                        error=str(e))

            if not agent_response_transcript or agent_response_transcript.startswith("❌"):
                error_msg = agent_response_transcript if agent_response_transcript else L.get(
                    "transcription_no_result", "전사 결과가 없습니다.")
                st.error(error_msg)

                if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                    st.session_state.agent_response_area_text = ""
                    st.session_state.last_transcript = ""
                else:
                    st.session_state.current_agent_audio_text = L.get(
                        "transcription_error", "전사 오류")
                    if "agent_response_input_box_widget_call" in st.session_state:
                        st.session_state.agent_response_input_box_widget_call = ""
                    st.session_state.last_transcript = ""

            elif not agent_response_transcript.strip():
                st.warning(
                    L.get(
                        "transcription_empty_warning",
                        "전사 결과가 비어 있습니다."))
                if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                    st.session_state.agent_response_area_text = ""
                else:
                    st.session_state.current_agent_audio_text = ""
                    if "agent_response_input_box_widget_call" in st.session_state:
                        st.session_state.agent_response_input_box_widget_call = ""
                st.session_state.last_transcript = ""

            elif agent_response_transcript.strip():
                agent_response_transcript = agent_response_transcript.strip()
                st.session_state.last_transcript = agent_response_transcript

                if st.session_state.get("feature_selection") == L["sim_tab_chat_email"]:
                    st.session_state.agent_response_area_text = agent_response_transcript
                else:
                    st.session_state.current_agent_audio_text = agent_response_transcript
                    if "agent_response_input_box_widget_call" in st.session_state:
                        st.session_state.agent_response_input_box_widget_call = agent_response_transcript

                snippet = agent_response_transcript[:50].replace("\n", " ")
                if len(agent_response_transcript) > 50:
                    snippet += "..."
                st.success(
                    L.get("whisper_success", "전사 완료") +
                    f" **{L.get('recognized_content', '인식 내용')}:** *{snippet}*")
                st.info(
                    L.get(
                        "transcription_auto_filled",
                        "💡 전사된 텍스트가 CC 자막 및 입력창에 자동으로 입력되었습니다."))

    # 솔루션 체크박스
    show_draft_ui = st.session_state.get("show_draft_ui", False)
    show_customer_data_ui = st.session_state.get("show_customer_data_ui", False)
    if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
        st.session_state.is_solution_provided = st.checkbox(
            L["solution_check_label"],
            value=st.session_state.is_solution_provided,
            key="solution_checkbox_widget",
        )

    # 메시지 입력 칸 초기화 처리
    if st.session_state.get("reset_agent_response_area", False):
        if not st.session_state.get("last_transcript") or not st.session_state.last_transcript:
            st.session_state.agent_response_area_text = ""
        st.session_state.reset_agent_response_area = False

    # 응대 초안 자동 생성 (고객 메시지 수신 시 - 5초 이내 빠른 응답)
    if st.session_state.is_llm_ready and st.session_state.sim_stage == "AGENT_TURN":
        # 마지막 고객 메시지 확인
        last_customer_msg = None
        last_customer_msg_idx = -1
        for idx, msg in enumerate(reversed(st.session_state.simulator_messages)):
            if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]:
                last_customer_msg = msg.get("content", "")
                last_customer_msg_idx = len(st.session_state.simulator_messages) - 1 - idx
                break
        
        # 고객 메시지가 있고, 아직 응대 초안이 생성되지 않았으면 자동 생성
        if last_customer_msg and not st.session_state.get("auto_draft_generated", False):
            # 이전에 생성한 응대 초안이 이 메시지에 대한 것인지 확인
            last_draft_for_idx = st.session_state.get("last_draft_for_message_idx", -1)
            if last_draft_for_idx != last_customer_msg_idx:
                try:
                    # 응대 초안 생성 (빠른 응답을 위해 최신 대화 맥락 사용)
                    # 전체 대화 맥락을 고려하여 응대 초안 생성
                    conversation_context = ""
                    for msg in st.session_state.simulator_messages[-5:]:  # 최근 5개 메시지만 사용
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role in ["customer", "customer_rebuttal", "initial_query"]:
                            conversation_context += f"고객: {content}\n"
                        elif role == "agent_response":
                            conversation_context += f"상담원: {content}\n"
                    
                    # 최신 고객 메시지를 주요 문의로 사용
                    initial_query = last_customer_msg if last_customer_msg else st.session_state.get('customer_query_text_area', '')
                    customer_type_display = st.session_state.get("customer_type_sim_select", "")
                    session_lang = st.session_state.get("language", "ko")
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = "ko"
                    
                    # 응대 초안 생성 (대화 맥락 포함)
                    draft_text = _generate_initial_advice(
                        initial_query,
                        customer_type_display,
                        st.session_state.customer_email,
                        st.session_state.customer_phone,
                        session_lang,
                        st.session_state.customer_attachment_file
                    )
                    
                    # 입력창에 자동으로 표시
                    if draft_text:
                        st.session_state.agent_response_area_text = draft_text
                        st.session_state.auto_draft_generated = True
                        st.session_state.auto_generated_draft_text = draft_text
                        st.session_state.last_draft_for_message_idx = last_customer_msg_idx
                except Exception as e:
                    # 오류 발생 시에도 계속 진행
                    st.session_state.auto_draft_generated = False

    # 전사 결과 반영 (응대 초안보다 우선순위 높음)
    if st.session_state.get("last_transcript") and st.session_state.last_transcript:
        st.session_state.agent_response_area_text = st.session_state.last_transcript
        st.session_state.auto_draft_generated = False  # 전사 결과가 있으면 초안 무시
    elif not st.session_state.get("agent_response_area_text") and st.session_state.get("last_transcript") and st.session_state.last_transcript:
        st.session_state.agent_response_area_text = st.session_state.last_transcript
        st.session_state.auto_draft_generated = False

    # 전사 결과 자동 전송 처리
    if st.session_state.get("last_transcript") and st.session_state.last_transcript:
        agent_response_auto = st.session_state.last_transcript.strip()
        if agent_response_auto:
            st.session_state.simulator_messages.append({
                "role": "agent_response",
                "content": agent_response_auto
            })
            st.session_state.last_transcript = ""
            st.session_state.agent_response_area_text = ""
            st.session_state.auto_draft_generated = False
            if st.session_state.is_llm_ready:
                # 고객 반응 즉시 생성 (5초 이내 빠른 응답)
                customer_response = generate_customer_reaction(
                    st.session_state.language, is_call=False)
                # 메시지 추가 및 즉시 화면 반영
                new_message = {"role": "customer", "content": customer_response}
                st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
                st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)

    # 응대 초안이 생성되었으면 표시 및 입력창에 자동 채우기
    if st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        draft_text = st.session_state.auto_generated_draft_text
        # 응대 초안을 입력창에 자동으로 채우기 위한 JavaScript
        draft_text_escaped = draft_text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
        st.markdown(f"""
        <script>
        (function() {{
            function fillChatInput() {{
                var chatInput = document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (chatInput && !chatInput.value.trim()) {{
                    chatInput.value = `{draft_text_escaped}`;
                    // 입력 이벤트 트리거
                    var event = new Event('input', {{ bubbles: true }});
                    chatInput.dispatchEvent(event);
                    // 변경 이벤트도 트리거
                    var changeEvent = new Event('change', {{ bubbles: true }});
                    chatInput.dispatchEvent(changeEvent);
                }}
            }}
            // 즉시 실행
            fillChatInput();
            // DOM이 준비될 때까지 대기
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', fillChatInput);
            }}
            // 추가로 짧은 지연 후에도 시도
            setTimeout(fillChatInput, 200);
            setTimeout(fillChatInput, 500);
        }})();
        </script>
        <div style="background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>💡 응대 초안이 자동 생성되었습니다.</strong> 아래 입력창에 자동으로 채워집니다. 수정 후 전송하세요.
        </div>
        """, unsafe_allow_html=True)
    
    # 채팅 입력 UI
    placeholder_text = L.get("agent_response_placeholder", "고객에게 응답하세요...")
    agent_response_input = st.chat_input(placeholder_text)

    col_extra_features = st.columns([1, 1])

    with col_extra_features[0]:
        if st.button(
                L.get("button_add_attachment", "➕ 파일 첨부"),
                key="btn_add_attachment_unified",
                use_container_width=True,
                type="secondary"):
            st.session_state.show_agent_file_uploader = True

    with col_extra_features[1]:
        if st.session_state.get("agent_response_area_text") and st.session_state.agent_response_area_text:
            transcript_preview = st.session_state.agent_response_area_text[:30]
            st.caption(
                L.get("transcription_label", "💬 전사: {text}...").format(
                    text=transcript_preview))

    # 전송 로직
    agent_response = None
    if agent_response_input:
        agent_response = agent_response_input.strip()

    if agent_response:
        if not agent_response.strip():
            st.warning(L["empty_response_warning"])
        else:
            # AHT 타이머 시작
            if st.session_state.start_time is None and len(
                    st.session_state.simulator_messages) >= 1:
                st.session_state.start_time = datetime.now()

            # 에이전트 첨부 파일 처리
            final_response_content = agent_response
            if st.session_state.agent_attachment_file:
                file_infos = st.session_state.agent_attachment_file
                file_names = ", ".join([f["name"] for f in file_infos])
                attachment_msg = L["agent_attachment_status"].format(
                    filename=file_names, filetype=f"총 {len(file_infos)}개 파일"
                )
                final_response_content = f"{agent_response}\n\n---\n{attachment_msg}"

            # 메시지 추가 및 즉시 화면 반영
            new_message = {"role": "agent_response", "content": final_response_content}
            st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)

            # 메일 끝인사 확인
            email_closing_patterns = [
                "추가 문의사항이 있으면 언제든지 연락",
                "추가 문의 사항이 있으면 언제든지 연락",
                "additional inquiries", "any additional questions",
                "feel free to contact", "please feel free to contact",
                "追加のご質問", "追加のお問い合わせ"]
            is_email_closing_in_response = any(pattern.lower(
            ) in final_response_content.lower() for pattern in email_closing_patterns)
            if is_email_closing_in_response:
                st.session_state.has_email_closing = True

            # 입력창 초기화
            st.session_state.sim_audio_bytes = None
            st.session_state.agent_attachment_file = []
            st.session_state.language_transfer_requested = False
            st.session_state.realtime_hint_text = ""
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.last_transcript = ""
            st.session_state.reset_agent_response_area = True
            st.session_state.auto_draft_generated = False  # 다음 고객 메시지에서 다시 생성
            st.session_state.auto_generated_draft_text = ""

            # 고객 반응 자동 생성
            if st.session_state.is_llm_ready:
                with st.spinner(L["generating_customer_response"]):
                    customer_response = generate_customer_reaction(
                        st.session_state.language, is_call=False)

                st.session_state.simulator_messages.append(
                    {"role": "customer", "content": customer_response}
                )

                # 다음 단계 결정
                if st.session_state.get("has_email_closing", False):
                    positive_keywords = [
                        "No, that will be all", "no more", "없습니다", "감사합니다",
                        "Thank you", "ありがとう", "추가 문의 사항 없습니다",
                        "no additional", "追加の質問はありません", "알겠습니다", "ok", "네", "yes"]
                    is_positive = any(
                        keyword.lower() in customer_response.lower() for keyword in positive_keywords)

                    escaped = re.escape(L.get('customer_no_more_inquiries', ''))
                    no_more_pattern = escaped.replace(
                        r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                    no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
                    if is_positive or no_more_regex.search(customer_response):
                        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"
                else:
                    escaped_no_more = re.escape(L.get("customer_no_more_inquiries", ""))
                    no_more_pattern = escaped_no_more.replace(
                        r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                    no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
                    escaped_positive = re.escape(L.get("customer_positive_response", ""))
                    positive_pattern = escaped_positive.replace(
                        r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                    positive_regex = re.compile(positive_pattern, re.IGNORECASE)
                    is_positive_closing = no_more_regex.search(
                        customer_response) is not None or positive_regex.search(customer_response) is not None

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
                        st.session_state.sim_stage = "AGENT_TURN"
            else:
                st.session_state.need_customer_response = True
                st.session_state.sim_stage = "CUSTOMER_TURN"

    # 언어 이관 버튼
    st.markdown("---")
    st.markdown(f"**{L['transfer_header']}**")
    transfer_cols = st.columns(len(LANG) - 1)

    languages = list(LANG.keys())
    languages.remove(current_lang)

    def transfer_session(target_lang: str, current_messages):
        current_lang_at_start = st.session_state.language
        L = LANG.get(current_lang_at_start, LANG["ko"])

        if not get_api_key("gemini"):
            st.error(
                L["simulation_no_key_warning"].replace(
                    'API Key', 'Gemini API Key'))
        else:
            st.session_state.start_time = None

            with st.spinner(L["transfer_loading"]):
                import time
                import numpy as np
                time.sleep(np.random.uniform(5, 10))

                try:
                    original_summary = summarize_history_with_ai(
                        current_lang_at_start)

                    if not original_summary or original_summary.startswith("❌"):
                        history_text = ""
                        for msg in current_messages:
                            role = "Customer" if msg["role"].startswith(
                                "customer") or msg["role"] == "initial_query" else "Agent"
                            if msg["role"] in [
                                "initial_query",
                                "customer_rebuttal",
                                "agent_response",
                                    "customer_closing_response"]:
                                history_text += f"{role}: {msg['content']}\n"
                        original_summary = history_text

                    from simulation_handler import translate_text_with_llm
                    translated_summary, is_success = translate_text_with_llm(
                        original_summary,
                        target_lang,
                        current_lang_at_start
                    )

                    if not translated_summary:
                        translated_summary = summarize_history_with_ai(
                            target_lang)
                        is_success = True if translated_summary and not translated_summary.startswith(
                            "❌") else False

                    translated_messages = []
                    for msg in current_messages:
                        translated_msg = msg.copy()
                        if msg["role"] in [
                            "initial_query",
                            "customer",
                            "customer_rebuttal",
                            "agent_response",
                            "customer_closing_response",
                                "supervisor"]:
                            if msg.get("content"):
                                try:
                                    translated_content, trans_success = translate_text_with_llm(
                                        msg["content"],
                                        target_lang,
                                        current_lang_at_start
                                    )
                                    if trans_success:
                                        translated_msg["content"] = translated_content
                                except Exception:
                                    pass
                        translated_messages.append(translated_msg)

                    st.session_state.simulator_messages = translated_messages
                    st.session_state.transfer_summary_text = translated_summary
                    st.session_state.translation_success = is_success
                    st.session_state.language_at_transfer_start = current_lang_at_start

                    st.session_state.language = target_lang
                    L = LANG.get(target_lang, LANG["ko"])

                    lang_name_target = {
                        "ko": "Korean",
                        "en": "English",
                        "ja": "Japanese"}.get(
                        target_lang,
                        "Korean")

                    system_msg = L["transfer_system_msg"].format(
                        target_lang=lang_name_target)
                    st.session_state.simulator_messages.append(
                        {"role": "system_transfer", "content": system_msg}
                    )
                    
                    # 이관 후 언어 설정이 사이드바에 반영되도록 즉시 새로고침
                    st.rerun()

                    summary_msg = f"### {L['transfer_summary_header']}\n\n{translated_summary}"
                    st.session_state.simulator_messages.append(
                        {"role": "supervisor", "content": summary_msg}
                    )

                    customer_type_display = st.session_state.get(
                        "customer_type_sim_select", "")
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=False,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )

                    st.session_state.sim_stage = "AGENT_TURN"
                except Exception as e:
                    error_msg = L.get(
                        "transfer_error",
                        "이관 처리 중 오류 발생: {error}").format(
                        error=str(e))
                    st.error(error_msg)

    for idx, lang_code in enumerate(languages):
        lang_name = {
            "ko": "Korean",
            "en": "English",
            "ja": "Japanese"}.get(
            lang_code,
            lang_code)
        transfer_label = L.get(
            f"transfer_to_{lang_code}",
            f"Transfer to {lang_name} Team")

        with transfer_cols[idx]:
            if st.button(
                    transfer_label,
                    key=f"btn_transfer_{lang_code}_{st.session_state.sim_instance_id}",
                    use_container_width=True):
                transfer_session(
                    lang_code, st.session_state.simulator_messages)


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
        st.warning(
            "⚠️ 고객이 검증 정보를 제공하면 검증 UI가 표시됩니다. 위의 디버깅 정보를 확인하세요.")


def _render_verification_ui(L, customer_has_attachment):
    """고객 검증 UI 렌더링"""
    # 검증 UI는 매우 길기 때문에 별도 파일로 분리하는 것이 좋지만,
    # 여기서는 핵심 부분만 포함합니다.
    # 전체 검증 UI는 원본 파일의 1884-2500줄 부분을 참고하세요.
    st.markdown("---")
    st.markdown(f"### {L.get('verification_header', '고객 검증')}")
    st.warning(L.get('verification_warning', '고객 검증이 필요합니다.'))

    # 검증 UI의 나머지 부분은 원본 파일을 참고하여 구현하세요.
    # (OCR, 파일 업로드, 검증 정보 입력 등)

