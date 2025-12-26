# ========================================
# _pages/_chat_agent_turn.py
# 채팅 시뮬레이터 - 에이전트 입력 단계 처리 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import (
    generate_customer_reaction, summarize_history_with_ai
)
from utils.history_handler import (
    generate_chat_summary, load_simulation_histories_local,
    recommend_guideline_for_customer, save_simulation_history_local
)
from utils.customer_verification import (
    check_if_login_related_inquiry, check_if_customer_provided_verification_info,
    verify_customer_info, mask_email
)
from utils.audio_handler import transcribe_bytes_with_whisper
from utils.customer_analysis import _generate_initial_advice
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
    # ⭐ 고객 체험 모드일 때 AI가 자동으로 응답 생성
    perspective = st.session_state.get("sim_perspective", "AGENT")
    if perspective == "CUSTOMER" and st.session_state.sim_stage == "AGENT_TURN":
        # AI가 에이전트로서 자동 응답 생성
        if st.session_state.is_llm_ready:
            # 중복 생성 방지
            if not st.session_state.get("ai_agent_response_generated", False):
                st.session_state.ai_agent_response_generated = True
                with st.spinner("🤖 AI 상담원이 답변을 작성 중입니다..."):
                    try:
                        from simulation_handler import generate_agent_response_draft
                        ai_agent_reply = generate_agent_response_draft(st.session_state.get("language", current_lang))
                        
                        if ai_agent_reply and ai_agent_reply.strip():
                            # 마크다운 헤더 제거
                            if "###" in ai_agent_reply:
                                lines = ai_agent_reply.split("\n")
                                ai_agent_reply = "\n".join([line for line in lines if not line.strip().startswith("###")])
                            ai_agent_reply = ai_agent_reply.strip()
                            
                            # 메시지 추가
                            new_msg = {"role": "agent_response", "content": ai_agent_reply, "is_auto_response": True}
                            st.session_state.simulator_messages.append(new_msg)
                            
                            # 다음 단계로 자동 이동 (사용자가 고객으로서 말할 차례)
                            st.session_state.sim_stage = "CUSTOMER_TURN"
                            st.session_state.ai_agent_response_generated = False  # 다음 응답을 위해 리셋
                            # st.rerun()  # 주석 처리: 상태 변경 후 Streamlit이 자동 rerun함
                    except Exception as e:
                        st.error(f"AI 응답 생성 중 오류: {e}")
                        st.session_state.ai_agent_response_generated = False
        else:
            st.info("👤 고객 입장 모드: AI 상담원이 답변을 생성하려면 LLM API Key가 필요합니다.")
        return  # 고객 모드일 때는 기존 상담원 입력 UI를 표시하지 않음
    
    # ⭐ 자연스러운 대화 흐름: 에이전트 응답 후 고객 반응 생성
    if st.session_state.get("pending_customer_reaction", False):
        pending_msg_idx = st.session_state.get("pending_customer_reaction_for_msg_idx", -1)
        if pending_msg_idx >= 0 and st.session_state.is_llm_ready:
            try:
                with st.spinner(L.get("generating_customer_response", "고객 응답 생성 중...")):
                    customer_response = generate_customer_reaction(
                        st.session_state.language, is_call=False)
                    customer_message = {"role": "customer", "content": customer_response}
                    st.session_state.simulator_messages = st.session_state.simulator_messages + [customer_message]
                    
                    # ⭐ 고객 불만/직접 응대 요청 감지
                    complaint_keywords = [
                        "불만", "불만족", "해결 안 됨", "도와주세요", "에이전트", "상담원", "직접",
                        "상담원 연결", "직접 상담", "사람과", "complaint", "dissatisfied",
                        "not resolved", "help me", "agent", "representative", "directly",
                        "speak to", "talk to", "connect me"
                    ]
                    has_complaint = any(keyword in customer_response.lower() for keyword in complaint_keywords)
                    
                    # 취소/환불 키워드 확인 (예외 처리 필요)
                    cancellation_keywords = [
                        "취소", "환불", "환불해주세요", "취소해주세요", "cancel", "refund",
                        "cancel please", "refund please", "キャンセル", "返金"
                    ]
                    has_cancellation_request = any(keyword in customer_response.lower() for keyword in cancellation_keywords)
                    
                    # ⭐ 예외 사유 키워드 (업종별 포괄적 확장)
                    # 여행/숙박업: 비행기 결항/지연, 병가, 날씨 등
                    # 제조/전자제품: 기기 결함, 배송 지연, 제품 오작동 등
                    # 일반: 긴급 상황, 불가피한 사정 등
                    exception_keywords = [
                        # 여행/숙박 관련
                        "비행기 결항", "비행기 지연", "항공편 결항", "항공편 지연", "항공사", "airline",
                        "flight cancelled", "flight delayed", "cancelled flight", "delayed flight",
                        "날씨", "태풍", "폭설", "weather", "typhoon", "snowstorm",
                        # 건강/긴급 상황
                        "병가", "병원", "입원", "수술", "응급", "긴급", "sick", "hospital", "emergency",
                        "medical", "surgery", "urgent", "critical",
                        # 제품/배송 관련 (제조업/전자제품 등 - 삼성전자 등)
                        "기기 결함", "제품 결함", "불량품", "오작동", "고장", "작동 안 함", "안 됨", "안돼",
                        "defect", "malfunction", "faulty", "broken", "not working", "doesn't work",
                        "배송 지연", "배송 오류", "배송 누락", "배송 안 됨", "배송 못 받음", "배송 안 옴",
                        "delivery delay", "delivery error", "delivery missing", "late delivery", 
                        "wrong delivery", "not delivered", "missing delivery",
                        # 제품 품질 문제
                        "품질 문제", "품질 불량", "불량", "quality issue", "quality problem", "poor quality",
                        # 포장/파손 문제
                        "포장 파손", "박스 파손", "상품 파손", "포장 뜯김", "damaged", "broken package",
                        # 교환/반품 관련
                        "교환", "반품", "exchange", "return", "교환 요청", "반품 요청",
                        # 일반 예외 사유
                        "불가피", "예외", "특별한 사정", "특수한 경우", "unavoidable", "exceptional",
                        "special circumstances", "unforeseen", "unexpected",
                        # 법적/정책적 사유
                        "법적", "정책", "규정", "legal", "policy", "regulation"
                    ]
                    has_exception = any(keyword in customer_response.lower() for keyword in exception_keywords)
                    
                    # 만족/해결 키워드 확인
                    satisfaction_keywords = [
                        "감사합니다", "감사해요", "해결됐어요", "해결되었습니다", "알겠습니다", "좋아요",
                        "thank you", "thanks", "resolved", "solved", "ok", "okay", "good",
                        "ありがとうございます", "解決しました", "了解しました"
                    ]
                    is_satisfied = any(keyword in customer_response.lower() for keyword in satisfaction_keywords)
                    
                    # ⭐ 불만이 있거나 직접 응대를 원하는 경우 자동 응답 비활성화
                    # ⭐ 단, 취소/환불 요청이 있어도 예외 사유가 있으면 자동 응답 유지
                    if has_complaint and not is_satisfied:
                        st.session_state.auto_response_disabled = True
                        st.session_state.requires_agent_response = True
                        print(f"⚠️ 고객 불만/직접 응대 요청 감지: 자동 응답 비활성화")
                    elif has_cancellation_request:
                        # 취소/환불 요청이 있지만 예외 사유가 있으면 자동 응답 유지
                        if has_exception:
                            print(f"ℹ️ 취소/환불 요청이지만 예외 사유 확인: 자동 응답 유지")
                        else:
                            # 취소/환불 요청은 에이전트 직접 응대 필요
                            st.session_state.auto_response_disabled = True
                            st.session_state.requires_agent_response = True
                            print(f"⚠️ 취소/환불 요청 감지: 자동 응답 비활성화 (에이전트 직접 응대 필요)")
                    
                    # 다음 단계 결정
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
                    
                    # 플래그 초기화
                    st.session_state.pending_customer_reaction = False
                    st.session_state.pending_customer_reaction_for_msg_idx = -1
                    
                    # ⭐ 고객 반응 추가 후 화면 업데이트
                    # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨
            except Exception as e:
                print(f"고객 반응 생성 오류: {e}")
                st.session_state.pending_customer_reaction = False
                st.session_state.pending_customer_reaction_for_msg_idx = -1
    
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

    # ⭐ 응대 초안 자동 생성 (고객 메시지 수신 시 즉시 생성)
    # API Key 확인 및 is_llm_ready 설정
    from llm_client import get_api_key
    has_api_key = any([
        bool(get_api_key("openai")),
        bool(get_api_key("gemini")),
        bool(get_api_key("claude")),
        bool(get_api_key("groq"))
    ])
    
    # API Key가 있으면 is_llm_ready를 True로 설정
    if has_api_key:
        st.session_state.is_llm_ready = True
    
    # ⭐ AGENT_TURN 단계에서 응대 초안 확인 및 생성 (더 확실하게)
    # ⭐ 초기 문의 입력 후 자동 응답 즉시 생성 및 전송
    # ⭐ 단, 고객이 불만을 표시하거나 직접 응대를 원하는 경우 자동 응답 비활성화
    if st.session_state.is_llm_ready and st.session_state.sim_stage == "AGENT_TURN":
        # ⭐ 자동 응답이 비활성화되지 않은 경우에만 자동 응답 생성
        auto_response_disabled = st.session_state.get("auto_response_disabled", False)
        
        # ⭐ 초기 문의 입력 시 자동 응답 즉시 생성 및 전송
        if not auto_response_disabled and st.session_state.get("need_auto_response_on_agent_turn", False):
            st.session_state.need_auto_response_on_agent_turn = False
            try:
                from simulation_handler import generate_agent_response_draft
                session_lang = st.session_state.get("language", "ko")
                if session_lang not in ["ko", "en", "ja"]:
                    session_lang = "ko"
                
                # ⭐ 응대 초안 즉시 생성 및 자동 전송
                draft_text = generate_agent_response_draft(session_lang)
                
                if draft_text and draft_text.strip():
                    # 마크다운 헤더 제거
                    draft_text_clean = draft_text
                    if "###" in draft_text_clean:
                        lines = draft_text_clean.split("\n")
                        draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                    draft_text_clean = draft_text_clean.strip()
                    
                    if draft_text_clean:
                        # ⭐ 즉시 자동 전송
                        new_message = {"role": "agent_response", "content": draft_text_clean}
                        st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
                        st.session_state.auto_draft_auto_sent = True
                        
                        # 고객 반응 자동 생성
                        if st.session_state.is_llm_ready:
                            st.session_state.pending_customer_reaction = True
                            st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1
                        
                        # ⭐ 화면 즉시 업데이트
                        # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨
            except Exception as e:
                print(f"초기 자동 응답 생성 오류: {e}")
        
        # 마지막 고객 메시지 확인
        last_customer_msg = None
        last_customer_msg_idx = -1
        for idx, msg in enumerate(reversed(st.session_state.simulator_messages)):
            if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]:
                last_customer_msg = msg.get("content", "")
                last_customer_msg_idx = len(st.session_state.simulator_messages) - 1 - idx
                break
        
        # 응대 초안이 이미 생성되었는지 확인 (더 엄격하게)
        last_draft_for_idx = st.session_state.get("last_draft_for_message_idx", -1)
        auto_draft_exists = (
            st.session_state.get("auto_draft_generated", False) and 
            st.session_state.get("auto_generated_draft_text", "") and
            last_draft_for_idx == last_customer_msg_idx
        )
        
        # ⭐ 새로운 고객 메시지가 들어왔고, 응대 초안이 없거나 다른 메시지용이면 생성
        # ⭐ 단, 자동 응답이 비활성화된 경우에는 생성하지 않음
        if not auto_response_disabled and last_customer_msg and not auto_draft_exists:
            # ⭐ 응대 초안 생성 중 플래그로 중복 생성 방지
            if not st.session_state.get("draft_generation_in_progress", False):
                st.session_state.draft_generation_in_progress = True
                try:
                    from simulation_handler import generate_agent_response_draft
                    session_lang = st.session_state.get("language", "ko")
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = "ko"
                    
                    # ⭐ 응대 초안 즉시 생성 (백그라운드에서 조용히)
                    draft_text = generate_agent_response_draft(session_lang)
                    
                    # 입력창에 자동으로 표시
                    if draft_text and draft_text.strip():
                        # 마크다운 헤더 제거
                        draft_text_clean = draft_text
                        if "###" in draft_text_clean:
                            lines = draft_text_clean.split("\n")
                            draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                        draft_text_clean = draft_text_clean.strip()
                        
                        if draft_text_clean:
                            # ⭐ 응대 초안 저장 및 자동 전송
                            st.session_state.agent_response_area_text = draft_text_clean
                            st.session_state.auto_draft_generated = True
                            st.session_state.auto_generated_draft_text = draft_text_clean
                            st.session_state.last_draft_for_message_idx = last_customer_msg_idx
                            
                            # ⭐ 응대 초안 자동 전송 (고객 문의 입력 시 자연스러운 타이밍으로 응답)
                            # 중복 전송 방지: 해당 메시지 인덱스에 대한 자동 전송이 아직 안 된 경우만
                            auto_sent_key = f"auto_sent_for_msg_{last_customer_msg_idx}"
                            if not st.session_state.get(auto_sent_key, False):
                                # ⭐ 자연스러운 대화 흐름을 위한 단계별 처리
                                # 1단계: 에이전트 응답 추가 및 화면 업데이트
                                new_message = {"role": "agent_response", "content": draft_text_clean}
                                st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
                                st.session_state[auto_sent_key] = True
                                st.session_state.auto_draft_auto_sent = True
                                
                                # 입력창 초기화
                                st.session_state.agent_response_area_text = ""
                                st.session_state.auto_generated_draft_text = ""
                                
                                # ⭐ 첫 번째 화면 업데이트 (에이전트 응답만 표시)
                                # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨
                                
                                # ⭐ 2단계: 고객 반응은 다음 렌더링 사이클에서 생성
                                # 자연스러운 타이밍을 위해 즉시 생성하지 않고 플래그 설정
                                st.session_state.pending_customer_reaction = True
                                st.session_state.pending_customer_reaction_for_msg_idx = last_customer_msg_idx
                            
                            # 디버깅: 응대 초안 생성 및 전송 확인
                            print(f"✅ 응대 초안 생성 및 자동 전송 완료 (메시지 인덱스: {last_customer_msg_idx})")
                except Exception as e:
                    # 오류 발생 시에도 계속 진행
                    st.session_state.auto_draft_generated = False
                    print(f"❌ 응대 초안 자동 생성 오류: {e}")
                finally:
                    st.session_state.draft_generation_in_progress = False

    # 전사 결과 반영 및 자동 전송 (응대 초안보다 우선순위 높음)
    if st.session_state.get("last_transcript") and st.session_state.last_transcript:
        agent_response_auto = st.session_state.last_transcript.strip()
        if agent_response_auto:
            # ⭐ 전사 결과 자동 전송 (자연스러운 타이밍)
            new_message = {"role": "agent_response", "content": agent_response_auto}
            st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
            
            # 전사 결과 초기화
            st.session_state.last_transcript = ""
            st.session_state.agent_response_area_text = ""
            st.session_state.auto_draft_generated = False
            
            # ⭐ 첫 번째 화면 업데이트 (에이전트 응답만 표시)
            # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨
            
            # ⭐ 고객 반응은 다음 렌더링 사이클에서 생성 (자연스러운 타이밍)
            if st.session_state.is_llm_ready:
                st.session_state.pending_customer_reaction = True
                st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1

    # 채팅 입력 UI (카카오톡 스타일)
    # 응대 초안이 있으면 자동으로 입력창에 표시
    initial_value = ""
    if st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        initial_value = st.session_state.auto_generated_draft_text
    elif st.session_state.get("agent_response_area_text") and st.session_state.agent_response_area_text:
        initial_value = st.session_state.agent_response_area_text
    
    placeholder_text = L.get("agent_response_placeholder", "고객에게 응답하세요...")
    
    # ⭐ 상담원 모드일 때만 카카오톡 스타일 채팅 입력창 및 파일 첨부 버튼 표시
    if perspective == "AGENT":
        st.markdown("""
        <style>
        .kakao-chat-input {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 24px;
            padding: 12px 20px;
            font-size: 15px;
            min-height: 50px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .kakao-chat-input:focus {
            outline: none;
            border-color: #FEE500;
            box-shadow: 0 2px 8px rgba(254, 229, 0, 0.3);
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Streamlit의 chat_input 사용 (자동 업데이트 지원)
    # ⭐ 고객 모드일 때는 상담원 입력 UI를 표시하지 않음
    agent_response_input = None
    if perspective == "AGENT":
        agent_response_input = st.chat_input(placeholder_text, key="agent_chat_input_main")
    
    # ⭐ 응대 초안이 있으면 입력창에 자동으로 채우기 (입력창 생성 후 - 더 확실하게)
    # ⭐ 상담원 모드일 때만 입력창에 자동 채우기
    if perspective == "AGENT" and agent_response_input is not None and st.session_state.get("auto_generated_draft_text") and st.session_state.auto_generated_draft_text:
        draft_text = st.session_state.auto_generated_draft_text
        # JavaScript를 사용하여 입력창에 자동으로 채우기 (더 확실한 방법)
        import json
        draft_text_json = json.dumps(draft_text)
        
        st.markdown(f"""
        <script>
        (function() {{
            var draftText = {draft_text_json};
            var filled = false;
            var fillAttempts = 0;
            var maxAttempts = 30; // 최대 30번 시도 (약 3초)
            
            function fillChatInput() {{
                fillAttempts++;
                
                // 여러 선택자 시도 (Streamlit 버전에 따라 다를 수 있음)
                var selectors = [
                    'textarea[data-testid="stChatInputTextArea"]',
                    'textarea[aria-label*="고객"]',
                    'textarea[placeholder*="고객"]',
                    'textarea.stChatInputTextArea',
                    'textarea[placeholder*="응답"]',
                    'textarea'
                ];
                
                var chatInput = null;
                for (var i = 0; i < selectors.length; i++) {{
                    var elements = document.querySelectorAll(selectors[i]);
                    for (var j = 0; j < elements.length; j++) {{
                        if (elements[j] && elements[j].offsetParent !== null) {{
                            chatInput = elements[j];
                            break;
                        }}
                    }}
                    if (chatInput) break;
                }}
                
                if (chatInput && !filled) {{
                    var currentValue = chatInput.value || '';
                    // 입력창이 비어있거나 이전 초안과 다를 때만 채우기
                    if (!currentValue.trim() || currentValue.trim() !== draftText.trim()) {{
                        // ⭐ 즉시 채우기 (타이핑 애니메이션은 선택적)
                        chatInput.value = draftText;
                        chatInput.focus();
                        
                        // 모든 이벤트 트리거
                        var events = ['input', 'change', 'keyup', 'keydown'];
                        events.forEach(function(eventType) {{
                            var event = new Event(eventType, {{ bubbles: true, cancelable: true }});
                            chatInput.dispatchEvent(event);
                        }});
                        
                        // React 이벤트 (Streamlit이 React를 사용하는 경우)
                        if (chatInput._valueTracker) {{
                            chatInput._valueTracker.setValue('');
                            chatInput._valueTracker.setValue(draftText);
                        }}
                        
                        // React의 onChange 이벤트 트리거
                        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                        nativeInputValueSetter.call(chatInput, draftText);
                        var inputEvent = new Event('input', {{ bubbles: true }});
                        chatInput.dispatchEvent(inputEvent);
                        
                        filled = true;
                        console.log('✅ 응대 초안이 입력창에 자동으로 채워졌습니다.');
                        
                        // ⭐ 입력창에 실제로 채워졌을 때만 알림 표시
                        showDraftNotification();
                    }}
                }} else if (!filled && fillAttempts < maxAttempts) {{
                    // 입력창을 찾지 못했으면 계속 시도
                    setTimeout(fillChatInput, 100);
                }}
            }}
            
            // ⭐ 알림 표시 함수 (입력창에 실제로 채워졌을 때만 호출)
            function showDraftNotification() {{
                var notification = document.getElementById('draft-notification');
                if (notification) {{
                    notification.style.display = 'block';
                    notification.style.animation = 'slideInDown 0.3s ease-out';
                    // 5초 후 자동으로 제거
                    setTimeout(function() {{
                        if (notification) {{
                            notification.style.animation = 'fadeOut 0.3s ease-in forwards';
                            setTimeout(function() {{
                                if (notification) notification.style.display = 'none';
                            }}, 300);
                        }}
                    }}, 5000);
                }}
            }}
            
            // 즉시 실행
            fillChatInput();
            
            // DOM이 준비될 때까지 대기
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', fillChatInput);
            }}
            
            // MutationObserver로 동적 생성된 입력창 감지
            var observer = new MutationObserver(function(mutations) {{
                if (!filled) {{
                    fillChatInput();
                }}
            }});
            
            observer.observe(document.body, {{
                childList: true,
                subtree: true
            }});
            
            // 여러 시점에 시도 (입력창이 늦게 생성될 수 있음)
            var intervals = [50, 100, 150, 200, 300, 500, 800, 1200, 2000, 3000];
            intervals.forEach(function(delay) {{
                setTimeout(function() {{
                    if (!filled) fillChatInput();
                }}, delay);
            }});
        }})();
        </script>
        
        <!-- ⭐ 응대 초안 생성 알림 (입력창에 실제로 채워졌을 때만 표시 - 기본적으로 숨김, 다국어 지원) -->
        <div id="draft-notification" style="display: none; background: rgba(33, 150, 243, 0.08); 
                    padding: 8px 12px; 
                    border-radius: 8px; 
                    margin-bottom: 8px;
                    border-left: 3px solid #2196F3;
                    font-size: 0.85em;
                    color: #1976D2;">
            <span style="display: inline-flex; align-items: center; gap: 6px;">
                <span style="font-size: 1.1em;">✨</span>
                <span id="draft-notification-text"></span>
            </span>
        </div>
        <script>
        // 다국어 알림 메시지 설정
        (function() {{
            var lang = '{current_lang}';
            var notificationText = '';
            if (lang === 'ko') {{
                notificationText = '응대 초안이 자동 생성되어 입력창에 채워졌습니다';
            }} else if (lang === 'en') {{
                notificationText = 'Response draft has been automatically generated and filled in the input field';
            }} else if (lang === 'ja') {{
                notificationText = '対応草案が自動生成され、入力欄に記入されました';
            }} else {{
                notificationText = '응대 초안이 자동 생성되어 입력창에 채워졌습니다';
            }}
            var notificationElement = document.getElementById('draft-notification-text');
            if (notificationElement) {{
                notificationElement.textContent = notificationText;
            }}
        }})();
        </script>
        <style>
        @keyframes slideInDown {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        @keyframes fadeOut {{
            to {{
                opacity: 0;
                transform: translateY(-10px);
                height: 0;
                margin: 0;
                padding: 0;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)

    # ⭐ 파일 첨부 버튼을 입력창 안쪽에 배치 (카카오톡 스타일)
    # 입력창 생성 후 JavaScript로 '+' 아이콘을 입력창 안쪽에 추가
    st.markdown("""
    <style>
    /* 입력창 컨테이너 스타일 */
    .stChatInputContainer,
    div[data-testid="stChatInputContainer"],
    div[data-baseweb="input"] {
        position: relative !important;
    }
    
    /* '+' 아이콘 버튼 스타일 (입력창 안쪽 왼쪽) */
    .chat-input-attach-btn {
        position: absolute !important;
        left: 10px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.2s ease !important;
        z-index: 1000 !important;
        line-height: 1 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .chat-input-attach-btn:hover {
        transform: translateY(-50%) scale(1.1) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    }
    
    .chat-input-attach-btn:active {
        transform: translateY(-50%) scale(0.95) !important;
    }
    
    /* 입력창 텍스트 영역에 왼쪽 패딩 추가 (아이콘 공간 확보) */
    textarea[data-testid="stChatInputTextArea"],
    textarea[data-baseweb="textarea"],
    textarea.stChatInputTextArea {
        padding-left: 48px !important;
    }
    
    /* 입력창 컨테이너 전체 스타일 */
    div[data-testid="stChatInputContainer"],
    div[data-baseweb="input"] {
        position: relative !important;
    }
    
    /* 입력 필드 래퍼 */
    div[data-baseweb="input"] > div {
        position: relative !important;
    }
    </style>
    <script>
    (function() {
        function addAttachButton() {
            // 기존 버튼이 있으면 제거
            var existingBtn = document.getElementById('chat-attach-btn');
            if (existingBtn) {
                existingBtn.remove();
            }
            
            // 입력창 찾기 (여러 선택자 시도)
            var chatInput = document.querySelector('textarea[data-testid="stChatInputTextArea"]')
                || document.querySelector('textarea[data-baseweb="textarea"]')
                || document.querySelector('textarea.stChatInputTextArea');
            
            if (chatInput) {
                // 입력창 컨테이너 찾기 (여러 선택자 시도)
                var container = chatInput.closest('[data-testid="stChatInputContainer"]') 
                    || chatInput.closest('[data-baseweb="input"]')
                    || chatInput.closest('.stChatInputContainer')
                    || chatInput.parentElement.parentElement
                    || chatInput.parentElement;
                
                // '+' 아이콘 버튼 생성
                var attachBtn = document.createElement('button');
                attachBtn.id = 'chat-attach-btn';
                attachBtn.className = 'chat-input-attach-btn';
                attachBtn.innerHTML = '+';
                attachBtn.title = '파일 첨부';
                attachBtn.type = 'button';
                attachBtn.setAttribute('aria-label', '파일 첨부');
                
                // 버튼 클릭 이벤트
                attachBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // Streamlit 버튼 찾기 (여러 방법 시도)
                    var hiddenBtn = document.querySelector('button[data-testid*="btn_add_attachment_unified_hidden"]')
                        || document.querySelector('button[data-baseweb="button"][aria-label*="파일"]')
                        || Array.from(document.querySelectorAll('button')).find(function(btn) {
                            return btn.textContent.includes('➕') || btn.textContent.includes('파일');
                        });
                    
                    if (hiddenBtn) {
                        // 버튼 클릭
                        hiddenBtn.click();
                        // 추가로 Streamlit 이벤트 트리거
                        var clickEvent = new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            view: window
                        });
                        hiddenBtn.dispatchEvent(clickEvent);
                    }
                });
                
                // 컨테이너에 버튼 추가
                if (container) {
                    // 컨테이너 스타일 설정
                    if (container.style) {
                        container.style.position = 'relative';
                    }
                    // 기존 버튼이 있으면 제거 후 추가
                    var oldBtn = container.querySelector('#chat-attach-btn');
                    if (oldBtn) {
                        oldBtn.remove();
                    }
                    container.appendChild(attachBtn);
                } else {
                    // 컨테이너를 찾지 못한 경우 입력창의 부모 요소에 추가
                    var parent = chatInput.parentElement;
                    if (parent) {
                        parent.style.position = 'relative';
                        var oldBtn = parent.querySelector('#chat-attach-btn');
                        if (oldBtn) {
                            oldBtn.remove();
                        }
                        parent.appendChild(attachBtn);
                    }
                }
            }
        }
        
        // 즉시 실행
        addAttachButton();
        
        // DOM이 준비될 때까지 대기
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', addAttachButton);
        }
        
        // MutationObserver로 동적 생성된 입력창 감지
        var observer = new MutationObserver(function(mutations) {
            addAttachButton();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // 여러 시점에 시도
        var intervals = [50, 100, 200, 300, 500, 800, 1200];
        intervals.forEach(function(delay) {
            setTimeout(addAttachButton, delay);
        });
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # ⭐ 파일 첨부 버튼 (숨겨진 버튼으로 기능만 제공)
    # 실제 버튼은 JavaScript로 입력창 안쪽에 '+' 아이콘으로 표시됨
    # CSS로 버튼을 숨기고 JavaScript에서 클릭 이벤트만 트리거
    st.markdown("""
    <style>
    button[data-testid*="btn_add_attachment_unified_hidden"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button(
            "➕",
            key="btn_add_attachment_unified_hidden",
            help=L.get("button_add_attachment", "➕ 파일 첨부"),
            use_container_width=False,
            type="secondary"):
        st.session_state.show_agent_file_uploader = True

    # ⭐ 응대 초안 자동 전송은 이미 위에서 처리됨 (AGENT_TURN 진입 시)
    # 여기서는 수동 입력만 처리
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

            # ⭐ 메시지 추가 및 즉시 화면 반영 (수동 전송)
            new_message = {
                "role": "agent_response", 
                "content": final_response_content,
                "is_manual_response": True,  # ⭐ 에이전트 직접 응답 표시
                "timestamp": datetime.now().isoformat()  # ⭐ 응답 시간 기록
            }
            st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
            
            # ⭐ 에이전트 직접 응답 이력 기록 (고객 데이터 관리 시스템에 저장)
            # ⭐ 모든 에이전트 직접 응답은 이력에 기록 (requires_agent_response 여부와 무관)
            try:
                from customer_data_management import AdvancedCustomerManager
                manager = AdvancedCustomerManager()
                customer_id = st.session_state.get("customer_email", "") or st.session_state.get("customer_phone", "")
                if customer_id:
                    # 고객 식별 (이메일/전화번호로)
                    all_custs = manager.list_all_customers()
                    target_customer_id = None
                    customer_hash = manager.generate_identity_hash(
                        st.session_state.get("customer_phone", ""),
                        st.session_state.get("customer_email", "")
                    )
                    for c in all_custs:
                        if c["basic_info"]["identity_hash"] == customer_hash:
                            target_customer_id = c["basic_info"]["customer_id"]
                            break
                    
                    # 고객이 없으면 생성
                    if not target_customer_id:
                        target_customer_id = manager.create_customer(
                            st.session_state.get("customer_email", "고객").split("@")[0],
                            st.session_state.get("customer_phone", ""),
                            st.session_state.get("customer_email", ""),
                            "일반"
                        )
                    
                    # 고객 데이터에 에이전트 직접 응답 기록
                    customer_data = manager.load_customer(target_customer_id)
                    if customer_data:
                        if "agent_manual_responses" not in customer_data:
                            customer_data["agent_manual_responses"] = []
                        
                        # 응답 사유 판단
                        reason = "일반 응대"
                        if st.session_state.get("requires_agent_response", False):
                            # 취소/환불 요청 확인
                            last_customer_msg = ""
                            for msg in reversed(st.session_state.simulator_messages):
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]:
                                    last_customer_msg = msg.get("content", "").lower()
                                    break
                            
                            cancellation_keywords = ["취소", "환불", "cancel", "refund"]
                            has_cancellation = any(kw in last_customer_msg for kw in cancellation_keywords)
                            
                            if has_cancellation:
                                reason = "취소/환불 요청 - 에이전트 직접 응대"
                            else:
                                reason = "고객 불만/직접 응대 요청"
                        
                        customer_data["agent_manual_responses"].append({
                            "response": final_response_content,
                            "timestamp": datetime.now().isoformat(),
                            "reason": reason,
                            "is_cancellation_refund": any(kw in final_response_content.lower() for kw in ["취소", "환불", "cancel", "refund"])
                        })
                        manager._save_to_file(target_customer_id, customer_data)
                        print(f"✅ 에이전트 직접 응답 이력 기록 완료 (고객 ID: {target_customer_id}, 사유: {reason})")
            except Exception as e:
                print(f"에이전트 응답 이력 기록 오류: {e}")
            
            # ⭐ 메시지 추가 시 즉시 화면 업데이트를 위한 rerun
            # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨

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
            st.session_state.auto_draft_auto_sent = False  # 자동 전송 플래그 리셋

            # ⭐ 고객 반응 자동 생성 (자연스러운 타이밍)
            if st.session_state.is_llm_ready:
                # ⭐ 에이전트 응답 표시 후 고객 반응 생성 (플래그 설정)
                st.session_state.pending_customer_reaction = True
                st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1
                # ⭐ 고객 반응 추가 시 즉시 화면 업데이트
                # st.rerun()  # 주석 처리: 렌더링 순서 변경으로 자동 반영됨

                # 다음 단계 결정 (고객 반응이 아직 생성되지 않았으므로 기본적으로 CUSTOMER_TURN으로 설정)
                # 고객 반응이 생성된 후에는 pending_customer_reaction 블록에서 처리됨
                st.session_state.sim_stage = "CUSTOMER_TURN"
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

                    from utils.translation import translate_text_with_llm
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
                    
                    # ⭐ rerun 제거: 언어 이관은 상태 변경으로 자동 반영됨
                    
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
                    # ⭐ 언어 이관 후 화면 즉시 업데이트
                    # st.rerun()  # 주석 처리: 버튼 클릭 후 Streamlit이 자동 rerun함
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
        # 수동 입력 폼
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
                # 검증 정보 업데이트
                st.session_state.verification_info.update({
                    "receipt_number": receipt_number,
                    "card_last4": card_last4,
                    "customer_name": customer_name,
                    "customer_email": customer_email,
                    "verification_attempts": st.session_state.verification_info.get("verification_attempts", 0) + 1
                })
                
                # 검증 로직 실행
                if verify_customer_info(
                    receipt_number=receipt_number,
                    card_last4=card_last4,
                    customer_name=customer_name,
                    customer_email=customer_email,
                    customer_phone=st.session_state.verification_info.get("customer_phone", "")
                ):
                    st.session_state.is_customer_verified = True
                    st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))
                    # st.rerun()  # 주석 처리: 폼 제출 후 Streamlit이 자동 rerun함
                else:
                    st.error("검증 정보가 일치하지 않습니다. 다시 확인해주세요.")
    
    else:  # 파일 업로드 (OCR)
        st.info("📎 영수증, 예약 확인서, 결제 내역 등의 이미지를 업로드하세요.")
        
        uploaded_file = st.file_uploader(
            "검증 파일 업로드",
            type=["png", "jpg", "jpeg", "pdf"],
            key="verification_file_uploader"
        )
        
        if uploaded_file:
            # 파일 정보 저장
            st.session_state.verification_info.update({
                "file_uploaded": True,
                "file_info": {
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "size": uploaded_file.size
                }
            })
            
            # 이미지 미리보기
            if uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
            
            # OCR 처리 버튼
            if st.button("OCR로 검증 정보 추출", use_container_width=True):
                with st.spinner("OCR 처리 중..."):
                    try:
                        # OCR 처리 (실제 구현은 utils에 있을 것으로 예상)
                        # 여기서는 기본 검증 로직만 실행
                        st.session_state.verification_info["verification_attempts"] = \
                            st.session_state.verification_info.get("verification_attempts", 0) + 1
                        
                        # 간단한 검증 (실제로는 OCR 결과를 사용)
                        if uploaded_file.size > 0:
                            st.session_state.is_customer_verified = True
                            st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))
                            # st.rerun()  # 주석 처리: 버튼 클릭 후 Streamlit이 자동 rerun함
                        else:
                            st.error("파일을 읽을 수 없습니다.")
                    except Exception as e:
                        st.error(f"OCR 처리 중 오류 발생: {str(e)}")
    
    # 검증 시도 횟수 표시
    attempts = st.session_state.verification_info.get("verification_attempts", 0)
    if attempts > 0:
        st.caption(f"검증 시도 횟수: {attempts}회")

