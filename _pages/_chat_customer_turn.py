# ========================================
# _pages/_chat_customer_turn.py
# 채팅 시뮬레이터 - 고객 반응 생성 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import generate_customer_reaction
from utils.history_handler import save_simulation_history_local

# 응대 초안 자동 생성을 위한 플래그 초기화
# 고객 메시지가 생성되면 다음 AGENT_TURN에서 자동으로 응대 초안 생성
import re


def render_customer_turn(L, current_lang):
    """고객 반응 생성 단계 UI 렌더링"""
    customer_type_display = st.session_state.get(
        "customer_type_sim_select", L["customer_type_options"][0])
    st.info(L["customer_turn_info"])

    # 고객 반응 생성
    last_customer_message = None
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "customer" and msg.get("content"):
            last_customer_message = msg.get("content", "")
            break

    if last_customer_message is None:
        # 고객 반응 즉시 생성 (5초 이내 빠른 응답)
        customer_response = generate_customer_reaction(
            st.session_state.language, is_call=False)

        # 메시지 추가 및 즉시 화면 반영을 위한 상태 업데이트
        new_message = {"role": "customer", "content": customer_response}
        st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
        
        # 상태 변경을 명시적으로 트리거하여 즉시 화면 업데이트
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
        
        # ⭐ 응대초안 즉시 자동 생성 (고객 메시지 수신 시 - 백그라운드에서 조용히)
        # ⭐ API Key 확인
        from llm_client import get_api_key
        has_api_key = any([
            bool(get_api_key("openai")),
            bool(get_api_key("gemini")),
            bool(get_api_key("claude")),
            bool(get_api_key("groq"))
        ])
        
        if has_api_key:
            st.session_state.is_llm_ready = True
            
            try:
                from simulation_handler import generate_agent_response_draft
                session_lang = st.session_state.get("language", "ko")
                if session_lang not in ["ko", "en", "ja"]:
                    session_lang = "ko"
                
                # ⭐ 응대 초안 즉시 생성 (백그라운드에서 조용히 - spinner 없이)
                draft_text = generate_agent_response_draft(session_lang)
                
                if draft_text and draft_text.strip():
                    # 마크다운 헤더 제거
                    draft_text_clean = draft_text
                    if "###" in draft_text_clean:
                        lines = draft_text_clean.split("\n")
                        draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                    draft_text_clean = draft_text_clean.strip()
                    
                    if draft_text_clean:
                        # ⭐ 응대 초안 즉시 저장 (AGENT_TURN에서 바로 사용)
                        st.session_state.agent_response_area_text = draft_text_clean
                        st.session_state.auto_draft_generated = True
                        st.session_state.auto_generated_draft_text = draft_text_clean
                        st.session_state.last_draft_for_message_idx = len(st.session_state.simulator_messages) - 1
                        
                        # ⭐ 응대 초안은 입력창에만 표시 (자동 전송하지 않음)
                        # 사용자가 수정 후 직접 전송하도록 함
                        
                        # 디버깅: 응대 초안 생성 확인
                        print(f"✅ 고객 메시지 수신 시 응대 초안 생성 완료 (메시지 인덱스: {len(st.session_state.simulator_messages) - 1})")
            except Exception as e:
                # 오류 발생 시에도 계속 진행 (조용히)
                print(f"❌ 응대 초안 자동 생성 오류: {e}")
                st.session_state.auto_draft_generated = False
        else:
            # 응대초안 자동 생성을 위한 플래그 리셋
            st.session_state.auto_draft_generated = False
            st.session_state.auto_generated_draft_text = ""
            st.session_state.last_draft_for_message_idx = -1

        # 다음 단계 결정
        escaped_no_more = re.escape(L["customer_no_more_inquiries"])
        no_more_pattern = escaped_no_more.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
        escaped_positive = re.escape(L["customer_positive_response"])
        positive_pattern = escaped_positive.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        positive_regex = re.compile(positive_pattern, re.IGNORECASE)
        is_positive_closing = no_more_regex.search(
            customer_response) is not None or positive_regex.search(customer_response) is not None

        if L["customer_positive_response"] in customer_response:
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                st.session_state.sim_stage = "AGENT_TURN"
        elif is_positive_closing:
            escaped = re.escape(L['customer_no_more_inquiries'])
            no_more_pattern = escaped.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
            if no_more_regex.search(customer_response):
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                if st.session_state.is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    st.session_state.sim_stage = "AGENT_TURN"
        elif customer_response.startswith(L["customer_escalation_start"]):
            st.session_state.sim_stage = "ESCALATION_REQUIRED"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
    else:
        customer_response = last_customer_message

    # 종료 조건 검토
    escaped_no_more = re.escape(L["customer_no_more_inquiries"])
    no_more_pattern = escaped_no_more.replace(
        r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
    escaped_positive = re.escape(L["customer_positive_response"])
    positive_pattern = escaped_positive.replace(
        r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    positive_regex = re.compile(positive_pattern, re.IGNORECASE)
    is_positive_closing = no_more_regex.search(
        customer_response) is not None or positive_regex.search(customer_response) is not None

    # 메일 응대 종료 문구 확인
    is_email_closing = st.session_state.get("has_email_closing", False)

    if not is_email_closing:
        last_agent_response = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "agent_response" and msg.get("content"):
                last_agent_response = msg.get("content", "")
                break

        email_closing_patterns = [
            "추가 문의사항이 있으면 언제든지 연락",
            "추가 문의 사항이 있으면 언제든지 연락",
            "additional inquiries", "any additional questions",
            "feel free to contact", "please feel free to contact",
            "追加のご質問", "追加のお問い合わせ"]
        if last_agent_response:
            is_email_closing = any(pattern.lower() in last_agent_response.lower(
            ) for pattern in email_closing_patterns)
            if is_email_closing:
                st.session_state.has_email_closing = True

    # 메일 끝인사 처리
    if is_email_closing:
        no_more_keywords = [
            L['customer_no_more_inquiries'],
            "No, that will be all", "no more", "없습니다", "감사합니다",
            "Thank you", "ありがとう", "추가 문의 사항 없습니다",
            "no additional", "追加の質問はありません", "알겠습니다", "ok", "네", "yes"]
        has_no_more_inquiry = False
        for keyword in no_more_keywords:
            escaped = re.escape(keyword)
            pattern = escaped.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            regex = re.compile(pattern, re.IGNORECASE)
            if regex.search(customer_response):
                has_no_more_inquiry = True
                break
        if "없습니다" in customer_response and "감사합니다" in customer_response:
            has_no_more_inquiry = True

        positive_keywords = [
            "알겠습니다", "알겠어요", "네", "yes", "ok", "okay",
            "감사합니다", "thank you", "ありがとう", "좋습니다", "good", "fine", "괜찮습니다"]
        is_positive_response = any(
            keyword.lower() in customer_response.lower() for keyword in positive_keywords)

        escaped_check = re.escape(L['customer_no_more_inquiries'])
        no_more_pattern_check = escaped_check.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)
        if is_positive_closing or has_no_more_inquiry or no_more_regex_check.search(
                customer_response) or is_positive_response:
            agent_closing_added = False
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") == "agent_response":
                    agent_msg_content = msg.get("content", "")
                    if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                        agent_closing_added = True
                    break

            if not agent_closing_added:
                agent_name = st.session_state.get("agent_name", "000")
                if current_lang == "ko":
                    agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                elif current_lang == "en":
                    agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                else:
                    agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": agent_closing_msg}
                )

            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
    elif L["customer_positive_response"] in customer_response or ("알겠습니다" in customer_response and "감사합니다" in customer_response):
        if st.session_state.is_solution_provided:
            st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
    elif is_positive_closing:
        escaped = re.escape(L['customer_no_more_inquiries'])
        no_more_pattern = escaped.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
        if no_more_regex.search(customer_response):
            st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                st.session_state.sim_stage = "AGENT_TURN"
    elif customer_response.startswith(L["customer_escalation_start"]):
        st.session_state.sim_stage = "ESCALATION_REQUIRED"
    else:
        st.session_state.sim_stage = "AGENT_TURN"
        # 응대 초안 자동 생성을 위한 플래그 리셋
        st.session_state.auto_draft_generated = False

    st.session_state.is_solution_provided = False

    # 이력 저장
    if st.session_state.sim_stage != "CLOSING":
        save_simulation_history_local(
            st.session_state.customer_query_text_area,
            customer_type_display,
            st.session_state.simulator_messages,
            is_chat_ended=False,
            attachment_context=st.session_state.sim_attachment_context_for_llm,
        )

    st.session_state.realtime_hint_text = ""

