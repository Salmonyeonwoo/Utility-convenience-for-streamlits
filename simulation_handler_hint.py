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
시뮬레이션 힌트 생성 모듈
"""

import streamlit as st
from llm_client import run_llm
from lang_pack import LANG
from simulation_handler_base import get_chat_history_for_prompt

def generate_realtime_hint(current_lang_key: str, is_call: bool = False):
    """현재 대화 맥락을 기반으로 에이전트에게 실시간 응대 힌트(키워드/정책/액션)를 제공
    확장 기능: 고객 감정 분석, 문의 내용 및 상품별 홈페이지 정보 포함"""
    # 언어 키 검증 및 기본값 처리
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    
    # 채팅/전화 구분하여 이력 사용
    if is_call:
        website_url = st.session_state.get("call_website_url", "").strip()
        website_context = f"\nWebsite URL: {website_url}" if website_url else ""
        history_text = (
            f"Initial Query: {st.session_state.call_initial_query}\n"
            f"Previous Customer Utterance: {st.session_state.current_customer_audio_text}\n"
            f"Previous Agent Utterance: {st.session_state.current_agent_audio_text}{website_context}"
        )
        customer_emotion = st.session_state.get("customer_avatar", {}).get("state", "NEUTRAL") if st.session_state.get("customer_avatar") else "NEUTRAL"
        initial_query = st.session_state.get("call_initial_query", "")
    else:
        history_text = get_chat_history_for_prompt(include_attachment=True)
        initial_query = st.session_state.get("customer_query_text_area", "")
        customer_profile = None
        customer_emotion = "NEUTRAL"
        try:
            from utils.customer_analysis import analyze_customer_profile
            if initial_query:
                customer_profile = analyze_customer_profile(initial_query, current_lang_key)
                sentiment_score = customer_profile.get("sentiment_score", 50)
                if sentiment_score >= 75:
                    customer_emotion = "HAPPY"
                elif sentiment_score >= 50:
                    customer_emotion = "NEUTRAL"
                elif sentiment_score >= 25:
                    customer_emotion = "SAD"
                else:
                    customer_emotion = "ANGRY"
        except Exception as e:
            print(f"Customer profile analysis error: {e}")
            customer_profile = None

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    # 감정별 응대 가이드 생성
    emotion_guidance = {
        "HAPPY": "The customer is in a positive mood. Be friendly and efficient. You can be more casual and use positive language.",
        "NEUTRAL": "The customer is in a neutral state. Maintain professional and polite communication.",
        "SAD": "The customer seems disappointed or frustrated. Show empathy, be patient, and focus on resolving their issue with extra care.",
        "ANGRY": "The customer is angry or very dissatisfied. Stay calm, acknowledge their frustration, apologize sincerely, and prioritize finding a solution quickly."
    }
    emotion_guide = emotion_guidance.get(customer_emotion, emotion_guidance["NEUTRAL"])
    
    # 문의 내용에서 상품/서비스 추출 및 홈페이지 정보 추가
    product_info = ""
    website_info = ""
    if initial_query:
        product_keywords = ["JR Pass", "USJ", "Universal Studio", "도쿄", "오사카", "교토", "호텔", "항공권", "티켓", "투어", "패스"]
        detected_products = [kw for kw in product_keywords if kw.lower() in initial_query.lower()]
        if detected_products:
            product_info = f"\nDetected Products/Services: {', '.join(detected_products)}"
        
        website_url = st.session_state.get("call_website_url", "") or st.session_state.get("website_url", "")
        if website_url:
            website_info = f"\nRelevant Website: {website_url} - Check this website for specific product information, policies, and FAQs."

    hint_prompt = f"""
You are an AI Supervisor providing an **urgent, internal hint** to a human agent whose AHT is being monitored.
Analyze the conversation history, especially the customer's last message, which might be about complex issues like JR Pass, Universal Studio Japan (USJ), or a complex refund policy.

**IMPORTANT CONTEXT:**
- Customer Emotional State: {customer_emotion}
- Emotion-Based Response Guidance: {emotion_guide}
{product_info}
{website_info}

Provide ONE concise, actionable hint for the agent. The purpose is to save AHT time.

Output MUST be a single paragraph/sentence in {lang_name} containing actionable advice.
DO NOT use markdown headers or titles.
DO NOT just say "check the website" - provide specific actionable steps or facts.
Consider the customer's emotional state when providing the hint.

Provide an actionable fact or the next specific step (e.g., check policy section, confirm coverage, specific website page URL, product-specific information).

Examples of good hints (based on the content):
- Check the official JR Pass site for current exchange rates.
- The 'Universal Express Pass' is non-refundable; clearly cite policy section 3.2.
- Ask for the order confirmation number before proceeding with any action.
- For this product, check the cancellation policy on the product page: [specific URL if available]
- The customer seems frustrated - acknowledge their concern first, then provide the solution.

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





