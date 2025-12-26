# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
시뮬레이션 요약 생성 모듈
"""

from typing import List, Dict
import streamlit as st
from llm_client import run_llm

def summarize_history_with_ai(current_lang_key: str) -> str:
    """전화 통화 로그를 정리하여 LLM에 전달하고 요약 텍스트를 받는 함수"""
    conversation_text = ""
    initial_query = st.session_state.get("call_initial_query", "N/A")
    website_url = st.session_state.get("call_website_url", "").strip()
    if initial_query and initial_query != "N/A":
        conversation_text += f"Initial Query: {initial_query}\n"
    if website_url:
        conversation_text += f"Website URL: {website_url}\n"

    for msg in st.session_state.simulator_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "phone_exchange":
            conversation_text += f"{content}\n"

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    summary_prompt = f"""
You are an AI Analyst specialized in summarizing customer phone calls. 
Analyze the full conversation log below, identify the main issue, the steps taken by the agent, and the customer's sentiment.

Provide a concise, easy-to-read summary of the key exchange STRICTLY in {lang_name}.

--- Conversation Log ---
{conversation_text}
---

Summary:
"""
    if not st.session_state.is_llm_ready:
        return "LLM Key가 없어 요약 생성이 불가합니다."

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ AI 요약 생성 오류: {e}"


def summarize_history_for_call(call_logs: List[Dict[str, str]], initial_query: str, current_lang_key: str) -> str:
    """전화 통화 로그와 초기 문의를 바탕으로 요약본을 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    full_log_text = f"--- Initial Customer Query ---\nCustomer: {initial_query}\n"
    for log in call_logs:
        if log["role"] == "phone_exchange":
            full_log_text += f"{log['content']}\n"
        elif log["role"] == "agent" and "content" in log:
            full_log_text += f"Agent (Greeting): {log['content']}\n"

    summary_prompt = f"""
You are an AI Supervisor. Analyze the following telephone support conversation log.
Provide a concise, neutral summary of the key issue, the steps taken by the agent, and the final outcome.
The summary MUST be STRICTLY in {lang_name}.

--- Conversation Log ---
{full_log_text}
---

Summary:
"""
    if not st.session_state.is_llm_ready:
        return f"❌ LLM Key is missing. Cannot generate summary. Log length: {len(full_log_text.splitlines())}"

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ Summary Generation Error: {e}"

