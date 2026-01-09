# ========================================
# _pages/chat_modules/agent_turn_language_transfer.py
# 에이전트 턴 - 언어 이관 처리
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import summarize_history_with_ai
from utils.history_handler import save_simulation_history_local
from utils.translation import translate_text_with_llm
from llm_client import get_api_key
import time
import numpy as np

def render_language_transfer(L, current_lang):
    """언어 이관 버튼 렌더링"""
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

