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
시뮬레이션 AI 응대 가이드라인 생성 모듈
고객 문의 내용 및 대화 맥락을 분석하여 상담원을 위한 3단계 실무 가이드라인을 생성합니다.
"""

import streamlit as st
from llm_client import run_llm
from lang_pack import LANG
from simulation_handler_base import get_chat_history_for_prompt

def generate_ai_guideline(current_lang_key: str = "ko", customer_query: str = "") -> str:
    """
    고객 문의 내용 및 대화 맥락을 기반으로 상담원을 위한 AI 응대 가이드라인(단계별 지침)을 생성합니다.
    """
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
            
    L = LANG.get(current_lang_key, LANG["ko"])
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    # 1. 대상 고객 문의 내용 결정 (전달받은 내용 우선, 없으면 대화 기록에서 최신 고객 메시지 추출)
    query_to_use = customer_query.strip() if customer_query else ""
    if not query_to_use:
        if "simulator_messages" in st.session_state and st.session_state.simulator_messages:
            cust_msgs = [
                m.get("content", "") for m in st.session_state.simulator_messages
                if m.get("role") in ["customer", "customer_rebuttal", "initial_query"] and m.get("content")
            ]
            if cust_msgs:
                query_to_use = cust_msgs[-1]
                
    if not query_to_use:
        query_to_use = st.session_state.get("customer_query_text_area", "")
        
    # 2. 대화 기록 추출
    history_text = get_chat_history_for_prompt(include_attachment=True)
    customer_type = st.session_state.get("customer_type_sim_select", "일반 고객")
    
    # 3. RAG 지식 베이스 검색 및 출처 인용
    citation_md = ""
    grounding_section = ""
    try:
        from utils.rag_knowledge_engine import retrieve_grounding_knowledge, format_citation_markdown
        citation = retrieve_grounding_knowledge(query_to_use, current_lang_key)
        if citation:
            st.session_state.last_rag_citation = citation
            grounding_section = f"\n[Verified Enterprise Knowledge Grounding]\n{citation['grounding_context']}\n"
            citation_md = format_citation_markdown(citation, current_lang_key)
    except Exception as e:
        print(f"RAG retrieval in guideline error: {e}")

    # 4. AI 가이드라인 프롬프트 구성
    guideline_prompt = f"""
You are an AI Customer Support Supervisor providing a step-by-step **Response Guideline** for a human agent.
Target Customer Type: **{customer_type}**
Language Requirement: Output ALL text STRICTLY in {lang_name}.

[Conversation History]
{history_text}

[Target Customer Inquiry to Address]:
{query_to_use}
{grounding_section}
Please provide a concise, highly practical 3-step response guideline for the agent:
1. Inquiry Analysis (고객 문의 핵심 파악)
2. Policy & System Verification (규정 및 정책/시스템 확인)
3. Recommended Agent Action (상담원 권장 조치 및 마무리 확인)

CRITICAL INSTRUCTIONS:
- Output ONLY the response guideline in {lang_name}.
- Do NOT include draft replies, customer greetings, or meta commentary.
- Address the customer's specific inquiry directly and accurately.
"""
    if not st.session_state.get("is_llm_ready", True):
        from llm_client import generate_fast_cs_simulation
        return generate_fast_cs_simulation(guideline_prompt)

    try:
        res = run_llm(guideline_prompt)
    except Exception as e:
        from llm_client import generate_fast_cs_simulation
        res = generate_fast_cs_simulation(guideline_prompt)

    if citation_md:
        res = f"{res.strip()}\n\n{citation_md.strip()}"
    return res
