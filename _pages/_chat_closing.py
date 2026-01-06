# ========================================
# _pages/_chat_closing.py
# 채팅 시뮬레이터 - 종료 관련 단계들 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import generate_customer_closing_response
from utils.history_handler import save_simulation_history_local
import re
import time


def render_closing_stages(L, current_lang):
    """종료 관련 단계들 렌더링"""
    # 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    if st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
        render_wait_closing_confirmation(L, current_lang)

    # 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":
        render_wait_customer_closing_response(L, current_lang)

    # 최종 종료 행동 (FINAL_CLOSING_ACTION)
    elif st.session_state.sim_stage == "FINAL_CLOSING_ACTION":
        render_final_closing_action(L, current_lang)


def render_escalation(L, current_lang):
    """에스컬레이션 요청 단계 렌더링"""
    st.warning(
        L.get(
            "escalation_required_msg",
            "🚨 고객이 에스컬레이션을 요청했습니다. 상급자나 전문 팀으로 이관이 필요합니다."))

    col_escalate, col_continue = st.columns(2)

    with col_escalate:
        if st.button(
                L.get("button_escalate", "에스컬레이션 처리"),
                key=f"btn_escalate_{st.session_state.sim_instance_id}"):
            escalation_msg = L.get(
                "escalation_system_msg",
                "📌 시스템 메시지: 고객 요청에 따라 상급자/전문 팀으로 이관되었습니다.")
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": escalation_msg}
            )

            customer_type_display = st.session_state.get(
                "customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

            st.session_state.sim_stage = "CLOSING"

    with col_continue:
        if st.button(
                L.get("button_continue", "계속 응대"),
                key=f"btn_continue_{st.session_state.sim_instance_id}"):
            st.session_state.sim_stage = "AGENT_TURN"


def render_wait_closing_confirmation(L, current_lang):
    """종료 확인 메시지 대기 단계 렌더링"""
    # ⭐ 고객 모드 확인
    perspective = st.session_state.get("sim_perspective", "AGENT")
    is_customer_mode = (perspective == "CUSTOMER")
    
    if is_customer_mode:
        # 고객 모드: 고객이 추가 문의 여부를 선택할 수 있는 버튼 표시
        st.success(
            L.get(
                "customer_positive_solution_reaction",
                "고객이 솔루션에 만족했습니다."))
        st.info(
            L.get(
                "info_use_buttons",
                "💡 아래 버튼을 사용하여 추가 문의 여부를 확인하거나 상담을 종료하세요."))
    else:
        # 에이전트 모드: 에이전트가 종료 확인 메시지를 보낼 수 있는 버튼 표시
        st.success(
            L.get(
                "customer_positive_solution_reaction",
                "고객이 솔루션에 만족했습니다."))
        st.info(
            L.get(
                "info_use_buttons",
                "💡 아래 버튼을 사용하여 추가 문의 여부를 확인하거나 상담을 종료하세요."))

    col_chat_end, col_email_end = st.columns(2)

    with col_chat_end:
        # 언어별 기본값 설정
        default_texts = {
            "ko": "✅ 추가 문의 있나요?",
            "en": "✅ Any Additional Questions?",
            "ja": "✅ 追加のご質問はありますか？"
        }
        button_text = L.get("send_closing_confirm_button", default_texts.get(current_lang, default_texts["ko"]))
        if st.button(
                button_text,
                key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}",
                use_container_width=True):
            agent_name = st.session_state.get("agent_name", "000")
            if current_lang == "ko":
                closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. {L.get('customer_closing_confirm', '추가 문의사항이 있으시면 언제든지 연락 주세요.')} 즐거운 하루 되세요."
            elif current_lang == "en":
                closing_msg = f"Thank you for contacting us. This was {agent_name}. {L.get('customer_closing_confirm', 'Please feel free to contact us if you have any additional questions.')} Have a great day!"
            else:
                closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。{L.get('customer_closing_confirm', '追加のご質問がございましたら、お気軽にお問い合わせください。')} 良い一日をお過ごしください。"

            st.session_state.simulator_messages.append(
                {"role": "agent_response", "content": closing_msg}
            )

            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
            # ⭐ 메시지 추가 후 즉시 화면 업데이트 (주석 처리: Streamlit이 자동으로 rerun함)
            # st.rerun()

    with col_email_end:
        # 언어별 기본값 설정
        default_end_texts = {
            "ko": "응대 종료 (설문 요청)",
            "en": "End Support (Survey Request)",
            "ja": "応対終了（アンケート）"
        }
        button_text_end = L.get("button_email_end_chat", default_end_texts.get(current_lang, default_end_texts["ko"]))
        if st.button(
                button_text_end,
                key=f"btn_email_end_chat_{st.session_state.sim_instance_id}",
                use_container_width=True,
                type="primary"):
            st.session_state.start_time = None

            end_msg = L.get("prompt_survey", "설문 조사 링크를 전송했습니다.")
            system_msg_prefix = L.get("email_end_system_msg", "시스템: 이메일 상담 종료")
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": f"({system_msg_prefix}) {end_msg}"}
            )

            time.sleep(0.1)
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"

            customer_type_display = st.session_state.get(
                "customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )


def render_wait_customer_closing_response(L, current_lang):
    """고객 최종 응답 생성 및 처리 단계 렌더링"""
    customer_type_display = st.session_state.get(
        "customer_type_sim_select", L["customer_type_options"][0])

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
        is_email_closing = any(pattern.lower() in last_agent_response.lower(
        ) for pattern in email_closing_patterns)

    # 고객 응답 확인
    last_customer_message = None
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "customer_rebuttal":
            last_customer_message = msg.get("content", "")
            break
        elif msg.get("role") == "customer" and is_email_closing:
            last_customer_message = msg.get("content", "")
            break

    # 고객 응답 생성
    if last_customer_message is None:
        if not st.session_state.is_llm_ready:
            st.warning(L["llm_key_missing_customer_response"])
            if st.button(
                    L["customer_generate_response_button"],
                    key="btn_generate_final_response"):
                st.session_state.sim_stage = "AGENT_TURN"
            st.stop()

        st.info(L["agent_confirmed_additional_inquiry"])
        with st.spinner(L["generating_customer_response"]):
            final_customer_reaction = generate_customer_closing_response(
                st.session_state.language)

        st.session_state.simulator_messages.append(
            {"role": "customer_rebuttal", "content": final_customer_reaction}
        )
        last_customer_message = final_customer_reaction

    # 고객 응답 처리
    if last_customer_message is None:
        st.warning(L["customer_response_generation_failed"])
    else:
        final_customer_reaction = last_customer_message
        _process_customer_closing_response(L, current_lang, final_customer_reaction, 
                                          is_email_closing, customer_type_display)


def _process_customer_closing_response(L, current_lang, final_customer_reaction, 
                                      is_email_closing, customer_type_display):
    """고객 종료 응답 처리"""
    no_more_keywords = [
        L['customer_no_more_inquiries'],
        "No, that will be all", "no more", "없습니다", "감사합니다",
        "結構です", "ありがとうございます", "추가 문의 사항 없습니다",
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
        "감사합니다", "thank you", "ありがとうございます"]
    is_positive_response = any(keyword.lower(
    ) in final_customer_reaction.lower() for keyword in positive_keywords)

    escaped_check = re.escape(L['customer_no_more_inquiries'])
    no_more_pattern_check = escaped_check.replace(
        r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
    no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)

    if is_email_closing and (has_no_more_inquiry or no_more_regex_check.search(
            final_customer_reaction) or is_positive_response):
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


def _render_survey_button(L, customer_type_display, button_key):
    """설문 조사 버튼 렌더링"""
    st.markdown("---")
    st.success(L["no_more_inquiries_confirmed"])
    st.markdown(f"### {L['consultation_end_header']}")
    st.info(L["click_survey_button_to_end"])
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        end_chat_button = st.button(
            L["sim_end_chat_button"],
            key=button_key,
            use_container_width=True,
            type="primary"
        )

        if end_chat_button:
            st.session_state.start_time = None

            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

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


def render_final_closing_action(L, current_lang):
    """최종 종료 행동 단계 렌더링"""
    st.markdown("---")
    st.success(L["no_more_inquiries_confirmed"])
    st.markdown(f"### {L['consultation_end_header']}")
    st.info(L["click_survey_button_to_end"])
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        end_chat_button = st.button(
            L["sim_end_chat_button"],
            key="btn_final_end_chat",
            use_container_width=True,
            type="primary"
        )

        if end_chat_button:
            st.session_state.start_time = None

            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"

            customer_type_display = st.session_state.get(
                "customer_type_sim_select", L["customer_type_options"][0])
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

            st.session_state.realtime_hint_text = ""


