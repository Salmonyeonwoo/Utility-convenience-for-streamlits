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
시뮬레이션 에이전트 응답 생성 모듈
"""

import random
import streamlit as st
from llm_client import run_llm
from lang_pack import LANG
from utils.customer_verification import mask_email, check_if_login_related_inquiry
from simulation_handler_base import get_chat_history_for_prompt

def generate_agent_response_draft(current_lang_key: str) -> str:
    """고객 응답을 기반으로 AI가 에이전트 응답 초안을 생성"""
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    history_text = get_chat_history_for_prompt(include_attachment=True)
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    latest_customer_message = ""
    initial_customer_query = st.session_state.get('customer_query_text_area', '')
    customer_query_analysis = ""
    
    all_customer_messages = []
    if st.session_state.simulator_messages:
        all_customer_messages = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]]
    
    if initial_customer_query and initial_customer_query not in all_customer_messages:
        all_customer_messages.insert(0, initial_customer_query)
    
    if all_customer_messages:
        latest_customer_message = all_customer_messages[-1]
        
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

    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"\n[고객 첨부 파일 정보: {attachment_context}]\n"
    else:
        attachment_context = ""

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

    customer_type = st.session_state.get('customer_type_sim_select', '일반적인 문의')
    is_difficult_customer = customer_type in ["까다로운 고객", "매우 불만족스러운 고객", "Difficult Customer",
                                              "Highly Dissatisfied Customer", "難しい顧客", "非常に不満な顧客"]

    customer_message_count = sum(
        1 for msg in st.session_state.simulator_messages if msg.get("role") in ["customer", "customer_rebuttal"])
    agent_message_count = sum(1 for msg in st.session_state.simulator_messages if msg.get("role") == "agent_response")

    previous_agent_responses = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") == "agent_response"]
    previous_responses_context = ""
    if previous_agent_responses:
        previous_responses_context = f"\n[이전 에이전트 응답들 (참고용, 동일하게 반복하지 말 것):\n"
        for i, prev_resp in enumerate(previous_agent_responses[-3:], 1):
            prev_resp_preview = prev_resp[:200] + "..." if len(prev_resp) > 200 else prev_resp
            previous_responses_context += f"{i}. {prev_resp_preview}\n"
        previous_responses_context += "]\n"

    is_repeating_complaints = False
    if customer_message_count > agent_message_count and customer_message_count >= 2:
        recent_customer_messages = [msg["content"].lower() for msg in st.session_state.simulator_messages if
                                    msg.get("role") in ["customer", "customer_rebuttal"]][-2:]
        complaint_keywords = ["왜", "이유", "설명", "말이 안", "이해가 안", "화나", "짜증", "불만", "why", "reason", "explain",
                              "angry", "frustrated", "complaint", "なぜ", "理由", "説明", "怒り", "不満"]
        if any(any(keyword in msg for keyword in complaint_keywords) for msg in recent_customer_messages):
            is_repeating_complaints = True

    needs_coping_strategy = is_difficult_customer or (is_repeating_complaints and customer_message_count >= 2)

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


def generate_agent_first_greeting(lang_key: str, initial_query: str) -> str:
    """전화 통화 시작 시 에이전트의 첫 인사말을 생성"""
    if lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
    L_local = LANG.get(lang_key, LANG["ko"])
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



