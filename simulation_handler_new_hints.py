# ========================================
# simulation_handler_new_hints.py
# 실시간 힌트 생성 함수
# ========================================

import streamlit as st
from lang_pack import LANG
from llm_client import run_llm
from simulation_handler_new_chat_history import get_chat_history_for_prompt

def generate_realtime_hint(current_lang_key: str, is_call: bool = False):
    """현재 대화 맥락을 기반으로 에이전트에게 실시간 응대 힌트(키워드/정책/액션)를 제공"""
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    
    if is_call:
        website_url = st.session_state.get("call_website_url", "").strip()
        website_context = f"\nWebsite URL: {website_url}" if website_url else ""
        history_text = (
            f"Initial Query: {st.session_state.call_initial_query}\n"
            f"Previous Customer Utterance: {st.session_state.current_customer_audio_text}\n"
            f"Previous Agent Utterance: {st.session_state.current_agent_audio_text}{website_context}"
        )
    else:
        history_text = get_chat_history_for_prompt(include_attachment=True)

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    hint_prompt = f"""
You are an AI Supervisor providing an **urgent, internal hint** to a human agent whose AHT is being monitored.
Analyze the conversation history, especially the customer's last message, which might be about complex issues like JR Pass, Universal Studio Japan (USJ), or a complex refund policy.

Provide ONE concise, actionable hint for the agent. The purpose is to save AHT time.

Output MUST be a single paragraph/sentence in {lang_name} containing actionable advice.
DO NOT use markdown headers or titles.
Do NOT direct the agent to check the general website.
Provide an actionable fact or the next specific step (e.g., check policy section, confirm coverage).

Examples of good hints (based on the content):
- Check the official JR Pass site for current exchange rates.
- The 'Universal Express Pass' is non-refundable; clearly cite policy section 3.2.
- Ask for the order confirmation number before proceeding with any action.
- The solution lies in the section of the Klook site titled '~'.

Conversation History:
{history_text}

HINT:
"""
    if not st.session_state.is_llm_ready:
        return "(Mock Hint: LLM Key is missing. Ask the customer for the booking number.)"

    with st.spinner(f"💡 {L['button_request_hint']}..."):
        try:
            return run_llm(hint_prompt).strip()
        except Exception as e:
            return f"❌ Hint Generation Error. (Try again or check API Key: {e})"

