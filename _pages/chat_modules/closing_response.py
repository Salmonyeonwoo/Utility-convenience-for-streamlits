# ========================================
# chat_modules/closing_response.py
# 고객 최종 응답 생성 및 처리 단계 모듈
# ========================================

import streamlit as st
from simulation_handler import generate_customer_closing_response
from utils.history_handler import save_simulation_history_local
import re


def render_wait_customer_closing_response(L, current_lang):
    """고객 최종 응답 생성 및 처리 단계 렌더링 (백업 파일의 원본 기능 유지)"""
    customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])

    # 메일 응대 종료 문구 확인
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
    is_email_closing = False
    if last_agent_response:
        is_email_closing = any(pattern.lower() in last_agent_response.lower() for pattern in email_closing_patterns)

    # 고객 응답 확인
    last_customer_message = None
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "customer_rebuttal"):
            last_customer_message = msg.get("content", "")
            break
        elif msg.get("role") == "customer" and is_email_closing:
            last_customer_message = msg.get("content", "")
            break

    # 고객 응답 생성
    if last_customer_message is None:
        if not st.session_state.is_llm_ready:
            st.warning(L["llm_key_missing_customer_response"])
            if st.button(L["customer_generate_response_button"], key="btn_generate_final_response"):
                st.session_state.sim_stage = "AGENT_TURN"
            st.stop()

        st.info(L["agent_confirmed_additional_inquiry"])
        with st.spinner(L["generating_customer_response"]):
            final_customer_reaction = generate_customer_closing_response(st.session_state.language)

        st.session_state.simulator_messages.append({"role": "customer_rebuttal", "content": final_customer_reaction})
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
        last_customer_message = final_customer_reaction

    # 고객 응답 처리
    if last_customer_message is None:
        st.warning(L["customer_response_generation_failed"])
    else:
        final_customer_reaction = last_customer_message
        _process_customer_closing_response(L, current_lang, final_customer_reaction, is_email_closing, customer_type_display)


def _process_customer_closing_response(L, current_lang, final_customer_reaction, is_email_closing, customer_type_display):
    """고객 종료 응답 처리 (백업 파일의 원본 기능 유지)"""
    no_more_keywords = [
        L['customer_no_more_inquiries'],
        "No, that will be all", "no more", "없습니다", "감사합니다",
        "結構です", "ありがとう", "추가 문의 사항 없습니다",
        "no additional", "追加の質問はありません"]
    has_no_more_inquiry = False
    for keyword in no_more_keywords:
        escaped = re.escape(keyword)
        pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        regex = re.compile(pattern, re.IGNORECASE)
        if regex.search(final_customer_reaction):
            has_no_more_inquiry = True
            break

    positive_keywords = [
        "알겠습니다", "알겠어요", "네", "yes", "ok", "okay",
        "감사합니다", "thank you", "ありがとう"]
    is_positive_response = any(keyword.lower() in final_customer_reaction.lower() for keyword in positive_keywords)

    escaped_check = re.escape(L['customer_no_more_inquiries'])
    no_more_pattern_check = escaped_check.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)

    if is_email_closing and (has_no_more_inquiry or no_more_regex_check.search(final_customer_reaction) or is_positive_response):
        _add_agent_closing_if_needed(L, current_lang)
        _render_survey_button(L, customer_type_display, "btn_final_end_chat_email_closing")
    elif not is_email_closing:
        if no_more_regex_check.search(final_customer_reaction) or has_no_more_inquiry:
            _add_agent_closing_if_needed(L, current_lang)
            _render_survey_button(L, customer_type_display, "btn_final_end_chat_in_wait")
        elif L['customer_has_additional_inquiries'] in final_customer_reaction:
            st.session_state.sim_stage = "AGENT_TURN"
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            st.session_state.realtime_hint_text = ""
        else:
            _add_agent_closing_if_needed(L, current_lang)
            _render_survey_button(L, customer_type_display, "btn_final_end_chat_fallback")


def _add_agent_closing_if_needed(L, current_lang):
    """에이전트 감사 인사 추가 (필요한 경우)"""
    agent_closing_added = False
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "agent_response"):
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

        st.session_state.simulator_messages.append({"role": "agent_response", "content": agent_closing_msg})
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)


def _render_survey_button(L, customer_type_display, button_key):
    """설문 조사 버튼 렌더링"""
    st.markdown("---")
    st.success(L["no_more_inquiries_confirmed"])
    st.markdown(f"### {L['consultation_end_header']}")
    st.info(L["click_survey_button_to_end"])
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        end_chat_button = st.button(L["sim_end_chat_button"], key=button_key, use_container_width=True, type="primary")

        if end_chat_button:
            st.session_state.start_time = None
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append({"role": "system_end", "content": end_msg})
            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            st.session_state.realtime_hint_text = ""
