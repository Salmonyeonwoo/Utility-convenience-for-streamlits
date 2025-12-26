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
시뮬레이션 처리 모듈
고객 응대 시뮬레이션, 채팅/전화 대화 생성, 힌트 생성 등의 핵심 기능을 제공합니다.
"""

import random
from typing import List, Dict, Any
import streamlit as st

from llm_client import run_llm
from lang_pack import LANG
from utils.customer_verification import mask_email, check_if_login_related_inquiry


def get_chat_history_for_prompt(include_attachment=False):
    """메모리에서 대화 기록을 추출하여 프롬프트에 사용할 문자열 형태로 반환 (채팅용)"""
    history_str = ""
    for msg in st.session_state.simulator_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "customer" or role == "customer_rebuttal":
            history_str += f"Customer: {content}\n"
        elif role == "agent_response":
            history_str += f"Agent: {content}\n"
        # supervisor 메시지는 LLM에 전달하지 않아 역할 혼동 방지
    return history_str


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
        # 전화 시뮬레이터에서는 현재 CC 영역에 표시된 텍스트와 초기 문의를 함께 사용
        website_url = st.session_state.get("call_website_url", "").strip()
        website_context = f"\nWebsite URL: {website_url}" if website_url else ""
        history_text = (
            f"Initial Query: {st.session_state.call_initial_query}\n"
            f"Previous Customer Utterance: {st.session_state.current_customer_audio_text}\n"
            f"Previous Agent Utterance: {st.session_state.current_agent_audio_text}{website_context}"
        )
        # 전화에서 고객 감정 정보 가져오기
        customer_emotion = st.session_state.get("customer_avatar", {}).get("state", "NEUTRAL") if st.session_state.get("customer_avatar") else "NEUTRAL"
        initial_query = st.session_state.get("call_initial_query", "")
    else:
        history_text = get_chat_history_for_prompt(include_attachment=True)
        # 채팅에서 고객 프로필 분석
        initial_query = st.session_state.get("customer_query_text_area", "")
        customer_profile = None
        customer_emotion = "NEUTRAL"
        try:
            from utils.customer_analysis import analyze_customer_profile
            if initial_query:
                customer_profile = analyze_customer_profile(initial_query, current_lang_key)
                sentiment_score = customer_profile.get("sentiment_score", 50)
                # 감정 점수를 감정 상태로 변환
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
        # 상품명/서비스명 추출 시도
        product_keywords = ["JR Pass", "USJ", "Universal Studio", "도쿄", "오사카", "교토", "호텔", "항공권", "티켓", "투어", "패스"]
        detected_products = [kw for kw in product_keywords if kw.lower() in initial_query.lower()]
        if detected_products:
            product_info = f"\nDetected Products/Services: {', '.join(detected_products)}"
        
        # 홈페이지 URL이 있으면 추가
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


def generate_agent_response_draft(current_lang_key: str) -> str:
    """고객 응답을 기반으로 AI가 에이전트 응답 초안을 생성"""
    # 언어 키 검증 및 기본값 처리
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    history_text = get_chat_history_for_prompt(include_attachment=True)
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 고객의 최신 문의 내용 추출 및 분석
    latest_customer_message = ""
    initial_customer_query = st.session_state.get('customer_query_text_area', '')
    customer_query_analysis = ""
    
    # 모든 고객 메시지 수집
    all_customer_messages = []
    if st.session_state.simulator_messages:
        all_customer_messages = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]]
    
    # 초기 문의도 포함
    if initial_customer_query and initial_customer_query not in all_customer_messages:
        all_customer_messages.insert(0, initial_customer_query)
    
    if all_customer_messages:
        latest_customer_message = all_customer_messages[-1]
        
        # 짧은 답변 감지
        short_response_keywords = ["네", "예", "아니요", "Yes", "No", "はい", "いいえ", "좋아요", "알겠습니다", "OK", "ok"]
        is_short_response = len(latest_customer_message.strip()) <= 10 or any(
            keyword in latest_customer_message.strip() for keyword in short_response_keywords
        )
        
        short_response_instruction = ""
        if is_short_response:
            short_response_instruction = """
**⚠️ CRITICAL: CUSTOMER GAVE A SHORT RESPONSE**

The customer's last message was very short (e.g., "네", "예", "아니요", "Yes", "No", "좋아요", "알겠습니다").

**YOU MUST:**
1. **Ask for more specific information** to understand their exact need or concern
2. **Request clarification** on what they need help with
3. **Ask follow-up questions** to get the details needed to provide proper assistance
4. **DO NOT** just acknowledge their short response - actively seek more information

**Example good responses:**
- "네, 알겠습니다. 어떤 부분이 궁금하신지 좀 더 자세히 말씀해주실 수 있을까요?"
- "감사합니다. 정확히 어떤 도움이 필요하신지 구체적으로 알려주시면 더 정확한 안내를 드릴 수 있습니다."
- "네, 이해했습니다. 혹시 [구체적인 정보]에 대해 더 자세히 알려주실 수 있나요?"

**IMPORTANT**: If the customer's response is too short to provide proper assistance, you MUST ask for more details.
"""
        
        # 핵심 문의 내용 요약
        inquiry_summary = f"""
**CUSTOMER INQUIRY DETAILS:**

Initial Query: "{initial_customer_query if initial_customer_query else 'Not provided'}"

Latest Customer Message: "{latest_customer_message}"

All Customer Messages Context:
{chr(10).join([f"- {msg[:150]}..." if len(msg) > 150 else f"- {msg}" for msg in all_customer_messages[-3:]])}

**YOUR RESPONSE MUST DIRECTLY ADDRESS:**

1. **SPECIFIC ISSUE IDENTIFICATION**: 
   - What EXACT problem or question did the customer mention?
   - Extract and reference specific details: order numbers, dates, product names, locations, error messages, etc.
   - If multiple issues were mentioned, address EACH one specifically

2. **CONCRETE SOLUTION PROVIDED**:
   - Provide STEP-BY-STEP instructions tailored to their EXACT situation
   - Include specific actions they need to take
   - Reference the exact products/services they mentioned
   - If they mentioned a location, reference it in your solution

3. **PERSONALIZATION**:
   - Use the customer's specific words/phrases when appropriate
   - Reference their exact situation
   - Acknowledge their specific concern or frustration point

4. **COMPLETENESS**:
   - Answer ALL questions they asked
   - Address ALL problems they mentioned
   - If they asked "why", explain the specific reason for their situation
   - If they asked "how", provide detailed steps for their exact case

**CRITICAL REQUIREMENTS:**
- DO NOT use generic templates like "Thank you for contacting us" without addressing their specific issue
- DO NOT give vague answers like "Please check your settings" - be SPECIFIC about which settings and what to do
- DO NOT ignore specific details they mentioned (order numbers, dates, locations, etc.)
- Your response must read as if it was written SPECIFICALLY for this customer's exact inquiry

**NOW GENERATE YOUR RESPONSE** following these requirements:
{short_response_instruction if short_response_instruction else ""}
"""
        
        customer_query_analysis = inquiry_summary

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"\n[고객 첨부 파일 정보: {attachment_context}]\n"
    else:
        attachment_context = ""

    # 고객 검증 상태 확인
    is_login_inquiry = check_if_login_related_inquiry(initial_customer_query)
    is_customer_verified = st.session_state.get("is_customer_verified", False)
    verification_warning = ""
    
    if is_login_inquiry and not is_customer_verified:
        customer_email = st.session_state.get("customer_email", "")
        masked_email = mask_email(customer_email, show_chars=2) if customer_email else ""
        
        verification_warning = f"""
**⚠️ CRITICAL SECURITY REQUIREMENT - CUSTOMER VERIFICATION NOT COMPLETED:**

This is a LOGIN/ACCOUNT related inquiry, but the customer has NOT been verified yet.

**STRICT RULES YOU MUST FOLLOW:**
1. **DO NOT provide ANY customer information hints** (email, phone, name, receipt number, card number) in your response
2. **EXCEPTION**: You MAY provide a masked email hint ONLY if absolutely necessary: "{masked_email}"
3. **DO NOT reveal**: Full email addresses, phone numbers, customer names, receipt numbers, card numbers, or any other personal information
4. **You MUST request verification information** from the customer before proceeding with account-related assistance

**ONLY AFTER VERIFICATION IS COMPLETED** can you provide full information hints and proceed with account assistance.

**CURRENT STATUS**: Customer verification: NOT COMPLETED ❌
"""
    elif is_login_inquiry and is_customer_verified:
        verification_warning = """
**✅ CUSTOMER VERIFICATION COMPLETED:**

The customer has been successfully verified. You may now provide information hints and proceed with account-related assistance.
"""

    # 고객 유형 및 반복 불만 패턴 분석
    customer_type = st.session_state.get('customer_type_sim_select', '일반적인 문의')
    is_difficult_customer = customer_type in ["까다로운 고객", "매우 불만족스러운 고객", "Difficult Customer",
                                              "Highly Dissatisfied Customer", "難しい顧客", "非常に不満な顧客"]

    customer_message_count = sum(
        1 for msg in st.session_state.simulator_messages if msg.get("role") in ["customer", "customer_rebuttal"])
    agent_message_count = sum(1 for msg in st.session_state.simulator_messages if msg.get("role") == "agent_response")

    # 이전 에이전트 응답들 추출
    previous_agent_responses = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") == "agent_response"]
    previous_responses_context = ""
    if previous_agent_responses:
        previous_responses_context = f"\n[이전 에이전트 응답들 (참고용, 동일하게 반복하지 말 것):\n"
        for i, prev_resp in enumerate(previous_agent_responses[-3:], 1):
            prev_resp_preview = prev_resp[:200] + "..." if len(prev_resp) > 200 else prev_resp
            previous_responses_context += f"{i}. {prev_resp_preview}\n"
        previous_responses_context += "]\n"

    # 고객이 계속 따지거나 화내는 패턴 감지
    is_repeating_complaints = False
    if customer_message_count > agent_message_count and customer_message_count >= 2:
        recent_customer_messages = [msg["content"].lower() for msg in st.session_state.simulator_messages if
                                    msg.get("role") in ["customer", "customer_rebuttal"]][-2:]
        complaint_keywords = ["왜", "이유", "설명", "말이 안", "이해가 안", "화나", "짜증", "불만", "why", "reason", "explain",
                              "angry", "frustrated", "complaint", "なぜ", "理由", "説明", "怒り", "不満"]
        if any(any(keyword in msg for keyword in complaint_keywords) for msg in recent_customer_messages):
            is_repeating_complaints = True

    needs_coping_strategy = is_difficult_customer or (is_repeating_complaints and customer_message_count >= 2)

    # 대처법 가이드라인 생성
    coping_guidance = ""
    if needs_coping_strategy:
        coping_guidance = f"""

[CRITICAL: Handling Difficult Customer Situation]
The customer type is "{customer_type}" and the customer has sent {customer_message_count} messages while the agent has sent {agent_message_count} messages.

**INCLUDE THE FOLLOWING COPING STRATEGY FORMAT IN YOUR RESPONSE:**

1. **Immediate Acknowledgment** (1-2 sentences)
2. **Specific Solution Recap** (2-3 sentences)
3. **Escalation or Follow-up Offer** (1-2 sentences)
4. **Closing with Assurance** (1 sentence)

**IMPORTANT NOTES:**
- DO NOT repeat the exact same solution that was already provided
- DO NOT sound dismissive or automated
- DO sound genuinely concerned and willing to go the extra mile
- Use warm, respectful tone while being firm about what can/cannot be done

Now generate the agent's response draft following this structure:
"""

    # 다양성 확보를 위한 추가 지시사항
    diversity_instruction = ""
    if previous_agent_responses:
        diversity_instruction = """
**CRITICAL DIVERSITY REQUIREMENT - STRICTLY ENFORCED:**
- You MUST generate a COMPLETELY DIFFERENT response from ALL previous agent responses
- Use COMPLETELY DIFFERENT wording, phrasing, sentence structures, and vocabulary
- If similar solutions are needed, present them in a COMPLETELY DIFFERENT way
- Vary your opening sentences, transition phrases, and closing statements - NO REPETITION
- DO NOT copy, paraphrase, or reuse ANY phrases from previous responses
"""

    variation_approaches = [
        "Start with a different greeting or acknowledgment style",
        "Use a different problem-solving approach or framework",
        "Present information in a different order",
        "Use different examples or analogies",
        "Vary the level of formality or warmth",
    ]
    selected_approaches = random.sample(variation_approaches, min(3, len(variation_approaches)))
    variation_note = "\n".join([f"- {approach}" for approach in selected_approaches])

    draft_prompt = f"""
You are an AI assistant helping a customer support agent write a professional, tailored response.

**PRIMARY OBJECTIVE:**
Generate a response draft that is SPECIFICALLY tailored to the customer's EXACT inquiry, providing concrete, actionable solutions.

**CRITICAL REQUIREMENTS:**
1. Address the customer's SPECIFIC inquiry/question with DETAILED, ACTIONABLE solutions
2. The response MUST be in {lang_name}
3. Be professional, empathetic, and solution-oriented
4. Reference specific details from their inquiry (order numbers, dates, products, locations, etc.) if mentioned
5. Keep the tone appropriate for the customer type: {customer_type}
6. Do NOT include any markdown formatting, just plain text
7. Generate a COMPLETELY UNIQUE and VARIED response - avoid repeating ANY similar phrases from previous responses

**VARIATION TECHNIQUES TO APPLY:**
{variation_note}

{customer_query_analysis}

**FULL CONVERSATION HISTORY:**
{history_text}
{attachment_context}

{verification_warning}

**PREVIOUS RESPONSES TO AVOID REPEATING:**
{previous_responses_context if previous_responses_context else "No previous responses to compare against."}

**DIVERSITY REQUIREMENTS:**
{diversity_instruction if diversity_instruction else "This is the first response, so no previous responses to avoid."}

{coping_guidance if needs_coping_strategy else ''}

**NOW GENERATE THE RESPONSE:**
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        draft = run_llm(draft_prompt).strip()
        # 마크다운 제거
        if draft.startswith("```"):
            lines = draft.split("\n")
            draft = "\n".join(lines[1:-1]) if len(lines) > 2 else draft
        return draft
    except Exception as e:
        return f"❌ 응답 초안 생성 오류: {e}"


def generate_outbound_call_summary(customer_query: str, current_lang_key: str, target: str) -> str:
    """전화 발신 시뮬레이션 요약 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    history_text = get_chat_history_for_prompt(include_attachment=True)
    if not history_text:
        history_text = f"Initial Customer Query: {customer_query}"

    policy_context = st.session_state.supervisor_policy_context or ""

    summary_prompt = f"""
You are an AI simulating a quick, high-stakes phone call placed by the customer support agent to a '{target}' (either a local partner/vendor or the customer).

The purpose of the call is to resolve a complex, policy-restricted issue (like an exceptional refund for a non-refundable item, or urgent confirmation of an airport transfer change).

Analyze the conversation history, the initial query, and any provided supervisor policy.
Generate a concise summary of the OUTCOME of this simulated phone call.
The summary MUST be professional and strictly in {lang_name}.

[CRITICAL RULE]: For non-refundable items, the local partner should only grant an exception IF the customer has provided strong, unavoidable proof. If no such proof is evident, the outcome should usually be a denial or a request for more proof, but keep the tone professional.

Conversation History:
{history_text}

Supervisor Policy Context (If any):
{policy_context}

Target of Call: {target}

Generate the phone call summary (Outcome ONLY):
"""
    if not st.session_state.is_llm_ready:
        return f"❌ LLM Key missing. (Simulated Outcome: The {target} requested the agent to send proof via email.)"

    try:
        summary = run_llm(summary_prompt).strip()
        if summary.startswith("```"):
            lines = summary.split("\n")
            summary = "\n".join(lines[1:-1]) if len(lines) > 2 else summary
        return summary
    except Exception as e:
        return f"❌ Phone call simulation error: {e}"


def generate_customer_reaction(current_lang_key: str, is_call: bool = False) -> str:
    """고객의 다음 반응을 생성하는 LLM 호출 (채팅 전용)"""
    history_text = get_chat_history_for_prompt()
    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG.get(current_lang_key, LANG["ko"])

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"[INITIAL ATTACHMENT CONTEXT (for customer reference only, do not repeat to agent)]\n{attachment_context}\n\n"
    else:
        attachment_context = ""

    next_prompt = f"""
{attachment_context}
You are now ROLEPLAYING as the CUSTOMER.

Read the following conversation and respond naturally in {lang_name}.

Conversation so far:
{history_text}

RULES:
1. You are only the customer. Do not write as the agent.
2. **[CRITICAL: Mandatory Information Submission]** If the agent requested any critical information, you MUST provide it.
3. **[CRITICAL: Short Response Handling]** If your previous response was very short and the agent asks for more information, you MUST provide the requested information actively and in detail.
4. **[Solution Acknowledgment]** If the agent provided a clear and accurate solution, you MUST respond with appreciation and satisfaction.
5. If the agent's LAST message was the closing confirmation: "{L_local['customer_closing_confirm']}"
    - If you have NO additional questions: You MUST reply with "{L_local['customer_no_more_inquiries']}".
   - If you DO have additional questions: You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS IMMEDIATELY.
6. Do NOT repeat your initial message or previous responses unless necessary.
7. Output ONLY the customer's next message.
"""
    try:
        reaction = run_llm(next_prompt)

        if not reaction or len(reaction.strip()) < 5:
            print("LLM returned insufficient response. Using positive closing fallback.")
            return L_local['customer_positive_response']

        return reaction.strip()
    except Exception as e:
        print(f"LLM Customer Reaction generation failed: {e}. Falling back to positive closing.")
        return L_local['customer_positive_response']


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


def generate_customer_reaction_for_call(current_lang_key: str, last_agent_response: str) -> str:
    """전화 시뮬레이터 전용 고객 반응 생성 (마지막 에이전트 응답 중심)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]
    
    # 고객 성별 및 감정 상태 가져오기
    if "customer_avatar" not in st.session_state:
        st.session_state.customer_avatar = {"gender": "male", "state": "NEUTRAL"}
    
    customer_gender = st.session_state.customer_avatar.get("gender", "male")
    customer_emotion = st.session_state.customer_avatar.get("state", "NEUTRAL")
    
    emotion_tone_map = {
        "HAPPY": "friendly, positive, and satisfied",
        "ASKING": "slightly frustrated, questioning, and seeking clarification",
        "ANGRY": "angry, frustrated, and demanding",
        "SAD": "sad, depressed, and disappointed",
        "NEUTRAL": "neutral, calm, and polite"
    }
    emotion_tone = emotion_tone_map.get(customer_emotion, "neutral, calm, and polite")
    
    closing_msg = L_local['customer_closing_confirm']
    # ⭐ 개선: 종료 확인 질문 판단 로직 강화 (더 정확하게 판단)
    closing_keywords = [
        "다른 문의 사항", "추가 문의 사항", "다른 문의가", "추가 문의가",
        "다른 도움이 필요", "다른 문의 없으", "추가 문의 없으",
        "anything else", "other questions", "any other inquiries", "any other questions",
        "other inquiries", "additional inquiries", "anything else we can",
        "他のお問合せ", "追加の問い合わせ", "他にご質問"
    ]
    
    # 종료 확인 질문인지 판단 (더 엄격하게)
    is_closing_question = (
        closing_msg in last_agent_response or 
        any(phrase in last_agent_response for phrase in closing_keywords) or
        # "또 다른 문의 사항 없으십니까?" 같은 패턴도 포함
        ("또 다른" in last_agent_response and ("문의" in last_agent_response or "질문" in last_agent_response) and ("없" in last_agent_response or "없으" in last_agent_response)) or
        ("다른" in last_agent_response and "문의" in last_agent_response and ("없" in last_agent_response or "없으" in last_agent_response or "있" in last_agent_response))
    )
    
    initial_inquiry = st.session_state.get("inquiry_text", "")
    agent_asking_for_details = any(phrase in last_agent_response.lower() for phrase in [
        "문의 내용", "상세히", "자세히", "구체적으로", "설명", "어떤 문의", "무엇을",
        "inquiry details", "more details", "explain", "what inquiry", "what is"
    ])
    
    # 최근 대화 이력만 추출
    recent_exchanges = []
    for msg in reversed(st.session_state.simulator_messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "phone_exchange":
            recent_exchanges.insert(0, content)
            if len(recent_exchanges) >= 3:
                break
        elif role == "agent" or role == "agent_response":
            recent_exchanges.insert(0, f"Agent: {content}")
            if len(recent_exchanges) >= 3:
                break
    
    recent_history = "\n".join(recent_exchanges) if recent_exchanges else "(No previous exchanges)"
    
    website_url = st.session_state.get("call_website_url", "").strip()
    website_context = f"\nWebsite URL: {website_url}" if website_url else ""
    
    last_agent_text = last_agent_response.strip() if last_agent_response else "None"
    
    initial_inquiry_context = ""
    if initial_inquiry and agent_asking_for_details:
        initial_inquiry_context = f"""
═══════════════════════════════════════════════════════════════════
📋 YOUR INITIAL INQUIRY (for reference when agent asks for details):
"{initial_inquiry}"
═══════════════════════════════════════════════════════════════════
"""
    
    history_text = f"""[Recent Conversation Context - For Reference Only]
{recent_history}{website_context}
{initial_inquiry_context}
═══════════════════════════════════════════════════════════════════
🎯 YOUR TASK: Respond ONLY to the Agent's message below
═══════════════════════════════════════════════════════════════════

Agent just said: "{last_agent_text}"

═══════════════════════════════════════════════════════════════════
IMPORTANT: 
- Respond DIRECTLY to what the agent JUST SAID above
- If the agent asks about your inquiry details, explain your INITIAL INQUIRY in detail
- DO NOT mention "추가 문의 사항" unless the agent explicitly asks "다른 문의가 있나요?"
- Keep your response short and conversational
- Your emotional state: {customer_emotion} - respond with {emotion_tone} tone
═══════════════════════════════════════════════════════════════════"""

    if is_closing_question:
        # ⭐ 종료 확인 질문일 때만 "없습니다. 감사합니다" 또는 "추가 문의 사항도 있습니다" 답변
        call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

{history_text}

The agent just asked: "{last_agent_text}"

═══════════════════════════════════════════════════════════════════
CRITICAL RULES FOR CLOSING CONFIRMATION (종료 확인 질문):
═══════════════════════════════════════════════════════════════════

The agent is asking if you have any OTHER or ADDITIONAL inquiries/questions.

YOU MUST CHOOSE ONE OF THE FOLLOWING TWO OPTIONS:

OPTION 1 - If you have NO additional questions and everything is resolved:
   → You MUST reply EXACTLY: "{L_local['customer_no_more_inquiries']}"
   → This means the conversation is complete and you are satisfied.

OPTION 2 - If you DO have additional questions or the issue is NOT fully resolved:
   → You MUST reply with: "{L_local['customer_has_additional_inquiries']}" 
   → AND immediately state your additional question clearly.
   → Example: "{L_local['customer_has_additional_inquiries']} [your additional question here]"

CRITICAL REQUIREMENTS:
1. Your response MUST be ONLY one of the two options above.
2. DO NOT add any other text if choosing OPTION 1.
3. If choosing OPTION 2, you MUST include your additional question.
4. Output ONLY the customer's response in {lang_name}.
5. DO NOT say anything else - just choose one option.

Your response (choose ONLY one option above):
"""
    else:
        # ⭐ 일반 질문일 때는 적절한 답변만 생성 (종료 문구 사용 금지)
        call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

{history_text}

═══════════════════════════════════════════════════════════════════
CRITICAL RULES FOR REGULAR RESPONSES (일반 질문 답변):
═══════════════════════════════════════════════════════════════════

The agent just said: "{last_agent_text}"

YOU MUST:
1. **CRITICAL**: Respond DIRECTLY and ACCURATELY to what the agent JUST SAID above
2. **If agent asked a question** → Answer it SPECIFICALLY and DIRECTLY with the requested information
3. **If agent requested information** → Provide the EXACT information requested (e.g., phone model, order number, date, etc.)
4. **If agent gave a solution or instruction** → Acknowledge it clearly and indicate if you understand or need clarification
5. Keep your response short (1-2 sentences max) and focused ONLY on what the agent just said
6. **CRITICAL - ANSWER THE AGENT'S QUESTION DIRECTLY** - Do not avoid the question

ABSOLUTELY FORBIDDEN (절대 금지):
1. **DO NOT say "없습니다. 감사합니다"** - This is ONLY for closing confirmation questions
2. **DO NOT say "추가 문의 사항도 있습니다"** - This is ONLY for closing confirmation questions
3. **DO NOT mention "다른 문의" or "추가 문의"** - This is ONLY for closing confirmation questions
4. **DO NOT end the conversation** - The agent is asking for information, not closing
5. **DO NOT switch topics** - Answer what the agent asked, nothing else

EXAMPLES:
- If agent asks "스마트폰 기종명은 어떻게 되십니까?" → Answer with your phone model (e.g., "아이폰 14입니다" or "갤럭시 S23입니다")
- If agent asks "주문 번호를 알려주세요" → Answer with an order number (e.g., "주문 번호는 12345입니다")
- If agent explains something → Acknowledge (e.g., "네, 이해했습니다" or "좀 더 자세히 설명해주실 수 있나요?")

Your response (respond ONLY to the agent's question above, with {emotion_tone} tone):
"""
    try:
        reaction = run_llm(call_prompt)
        reaction_text = reaction.strip()
        
        if is_closing_question:
            # ⭐ 종료 확인 질문일 때만 두 가지 옵션 중 하나 반환
            no_more_text = L_local['customer_no_more_inquiries']
            has_additional_text = L_local['customer_has_additional_inquiries']
            
            # "없습니다. 감사합니다" 또는 유사한 답변인지 확인
            if no_more_text in reaction_text or any(keyword in reaction_text for keyword in [
                "없습니다", "감사합니다", "No, that will be all", "no more", "結構です"
            ]):
                return no_more_text
            # "추가 문의 사항도 있습니다" 또는 유사한 답변인지 확인
            elif has_additional_text in reaction_text or any(phrase in reaction_text for phrase in [
                "추가 문의", "다른 문의", "additional", "other inquiries", "追加の問い合わせ"
            ]):
                # 추가 문의가 있다고 답한 경우, 추가 문의 내용이 포함되어 있으면 그대로 반환
                if len(reaction_text) > len(has_additional_text):
                    return reaction_text
                else:
                    # 추가 문의 내용이 없으면 기본 메시지만 반환
                    return has_additional_text
            else:
                # 명확하지 않으면 기본적으로 "없습니다. 감사합니다" 반환
                return no_more_text
        
        # ⭐ 일반 질문일 때는 종료 문구 필터링 강화
        else:
            # 종료 문구가 포함되어 있으면 재생성 요청
            no_more_keywords = [
                "없습니다. 감사합니다", "없습니다 감사합니다",
                "No, that will be all", "no more", "thank you",
                "추가 문의 사항 없습니다", "no additional", "結構です"
            ]
            
            additional_inquiry_phrases = [
                "추가 문의 사항도 있습니다", "다른 문의 사항도 있습니다",
                "additional inquiries", "other inquiries", "I also have"
            ]
            
            has_no_more = any(keyword in reaction_text for keyword in no_more_keywords)
            has_additional_inquiry_mention = any(phrase in reaction_text for phrase in additional_inquiry_phrases)
            
            # 종료 문구가 포함되어 있으면 제거하고 적절한 답변으로 재생성
            if has_no_more:
                # "없습니다. 감사합니다" 같은 종료 문구가 포함되어 있으면 제거
                # 에이전트의 질문에 적절히 답변하도록 재생성
                if "기종" in last_agent_text or "model" in last_agent_text.lower() or "phone" in last_agent_text.lower():
                    reaction_text = "아이폰 14입니다." if current_lang_key == "ko" else "iPhone 14."
                elif "번호" in last_agent_text or "number" in last_agent_text.lower():
                    reaction_text = "주문 번호는 12345입니다." if current_lang_key == "ko" else "The order number is 12345."
                elif "날짜" in last_agent_text or "date" in last_agent_text.lower():
                    reaction_text = "12월 12일입니다." if current_lang_key == "ko" else "December 12th."
                else:
                    # 일반적인 확인 답변
                    reaction_text = "네, 알겠습니다." if current_lang_key == "ko" else "Yes, I understand."
            
            # 추가 문의 언급이 포함되어 있으면 제거 (일반 질문이므로)
            if has_additional_inquiry_mention:
                # 추가 문의 언급 부분 제거
                for phrase in additional_inquiry_phrases:
                    reaction_text = reaction_text.replace(phrase, "").strip()
                # 빈 답변이 되면 적절한 답변으로 대체
                if not reaction_text or len(reaction_text) < 3:
                    if "기종" in last_agent_text or "model" in last_agent_text.lower():
                        reaction_text = "아이폰 14입니다." if current_lang_key == "ko" else "iPhone 14."
                    else:
                        reaction_text = "네, 알겠습니다." if current_lang_key == "ko" else "Yes, I understand."
        
        return reaction_text
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def generate_customer_reaction_for_first_greeting(current_lang_key: str, agent_greeting: str, initial_query: str) -> str:
    """전화 시뮬레이터 전용: 첫 인사말에 대한 고객의 맞춤형 반응 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]
    
    if "customer_avatar" not in st.session_state:
        st.session_state.customer_avatar = {"gender": "male", "state": "NEUTRAL"}
    
    customer_gender = st.session_state.customer_avatar.get("gender", "male")
    customer_emotion = st.session_state.customer_avatar.get("state", "NEUTRAL")
    
    emotion_tone_map = {
        "HAPPY": "friendly, positive, and satisfied",
        "ASKING": "slightly frustrated, questioning, and seeking clarification",
        "ANGRY": "angry, frustrated, and demanding",
        "SAD": "sad, depressed, and disappointed",
        "NEUTRAL": "neutral, calm, and polite"
    }
    emotion_tone = emotion_tone_map.get(customer_emotion, "neutral, calm, and polite")
    
    website_url = st.session_state.get("call_website_url", "").strip()
    website_context = f"\nWebsite URL: {website_url}" if website_url else ""
    
    agent_greeting_text = agent_greeting.strip() if agent_greeting else "None"
    initial_query_text = initial_query.strip() if initial_query else "None"
    
    call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

You called because: "{initial_query_text}"

The agent just greeted you and said: "{agent_greeting_text}"
{website_context}

YOUR TASK: Respond to the agent's greeting in a way that:
1. Acknowledge the agent's greeting naturally
2. Briefly mention your inquiry/concern: "{initial_query_text}"
3. Show that you're ready to discuss your issue
4. Keep it conversational and natural (1-2 sentences max)
5. IMPORTANT: Match your tone to your emotional state ({customer_emotion}) - be {emotion_tone}

**CRITICAL RULES:**
- You MUST mention your inquiry/concern: "{initial_query_text}"
- DO NOT say "없습니다. 감사합니다" or similar closing phrases
- DO NOT end the conversation - you are just starting it
- Keep it brief (1-2 sentences) but make sure to mention your inquiry

Your response (respond naturally to the greeting and briefly mention your inquiry, with {emotion_tone} tone):
"""
    try:
        reaction = run_llm(call_prompt)
        reaction_text = reaction.strip()
        
        # 종료 메시지 필터링
        no_more_keywords = [
            "없습니다", "감사합니다", "No, that will be all", "no more",
            "추가 문의 사항 없습니다", "no additional", "結構です"
        ]
        
        has_no_more = any(keyword in reaction_text for keyword in no_more_keywords)
        has_inquiry_mention = initial_query_text.lower() in reaction_text.lower() or any(
            word in reaction_text.lower() for word in initial_query_text.split()[:3]
        )
        
        if has_no_more and not has_inquiry_mention:
                reaction_text = f"안녕하세요. {initial_query_text}에 대해 문의드리고 싶어서 연락드렸습니다."
        
        return reaction_text
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


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


def generate_customer_closing_response(current_lang_key: str) -> str:
    """에이전트의 마지막 확인 질문에 대한 고객의 최종 답변 생성 (채팅용)"""
    history_text = get_chat_history_for_prompt()
    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG.get(current_lang_key, LANG["ko"])

    closing_msg = L_local['customer_closing_confirm']

    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"[INITIAL ATTACHMENT CONTEXT (for customer reference only, do not repeat to agent)]\n{attachment_context}\n\n"
    else:
        attachment_context = ""

    final_prompt = f"""
{attachment_context}
You are now ROLEPLAYING as the CUSTOMER.

The agent's final message was the closing confirmation: "{closing_msg}".
You MUST respond to this confirmation based on the overall conversation.

Conversation history:
{history_text}

RULES:
1. If the conversation seems resolved and you have NO additional questions:
    - You MUST reply with "{L_local['customer_no_more_inquiries']}".
2. If the conversation is NOT fully resolved and you DO have additional questions:
    - You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS.
3. Your reply MUST be ONLY one of the two options above, in {lang_name}.
4. Output ONLY the customer's next message (must be one of the two rule options).
"""
    try:
        reaction = run_llm(final_prompt)
        reaction_text = reaction.strip()
        if L_local['customer_no_more_inquiries'] in reaction_text:
            return L_local['customer_no_more_inquiries']
        elif L_local['customer_has_additional_inquiries'] in reaction_text:
            return reaction_text
        else:
            return L_local['customer_has_additional_inquiries']
    except Exception as e:
        st.error(f"고객 최종 반응 생성 오류: {e}")
        return L_local['customer_has_additional_inquiries']


def generate_agent_first_greeting(lang_key: str, initial_query: str) -> str:
    """전화 통화 시작 시 에이전트의 첫 인사말을 생성"""
    # 언어 키 검증
    if lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
    L_local = LANG.get(lang_key, LANG["ko"])
    # 문의 내용의 첫 15자만 사용
    topic = initial_query.strip()[:15].replace('\n', ' ')
    if len(initial_query.strip()) > 15:
        topic += "..."

    if lang_key == 'ko':
        return f"안녕하세요, {topic} 관련 문의 주셨죠? 상담원 000입니다. 무엇을 도와드릴까요?"
    elif lang_key == 'en':
        return f"Hello, thank you for calling. I see you're calling about {topic}. My name is 000. How may I help you today?"
    elif lang_key == 'ja':
        return f"お電話ありがとうございます。{topic}の件ですね。担当の000と申します。どのようなご用件でしょうか?"
    return "Hello, how may I help you?"
