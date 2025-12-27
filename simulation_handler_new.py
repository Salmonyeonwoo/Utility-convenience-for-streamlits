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
시뮬레이션 처리 모듈 (핵심 기능만 포함)
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
    return history_str


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


def generate_agent_response_draft(current_lang_key: str) -> str:
    """고객 응답을 기반으로 AI가 에이전트 응답 초안을 생성"""
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    history_text = get_chat_history_for_prompt(include_attachment=True)
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    initial_customer_query = st.session_state.get('customer_query_text_area', '')
    all_customer_messages = []
    if st.session_state.simulator_messages:
        all_customer_messages = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]]
    
    if initial_customer_query and initial_customer_query not in all_customer_messages:
        all_customer_messages.insert(0, initial_customer_query)
    
    customer_query_analysis = ""
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
"""
        
        customer_query_analysis = f"""
**CUSTOMER INQUIRY DETAILS:**

Initial Query: "{initial_customer_query if initial_customer_query else 'Not provided'}"
Latest Customer Message: "{latest_customer_message}"

All Customer Messages Context:
{chr(10).join([f"- {msg[:150]}..." if len(msg) > 150 else f"- {msg}" for msg in all_customer_messages[-3:]])}

**YOUR RESPONSE MUST DIRECTLY ADDRESS:**
1. **SPECIFIC ISSUE IDENTIFICATION**: What EXACT problem or question did the customer mention?
2. **CONCRETE SOLUTION PROVIDED**: Provide STEP-BY-STEP instructions tailored to their EXACT situation
3. **PERSONALIZATION**: Use the customer's specific words/phrases when appropriate
4. **COMPLETENESS**: Answer ALL questions they asked completely

{short_response_instruction if short_response_instruction else ""}
"""

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
1. **DO NOT provide ANY customer information hints** (email, phone, name, receipt number, card number)
2. **EXCEPTION**: You MAY provide a masked email hint ONLY if absolutely necessary: "{masked_email}"
3. **You MUST request verification information** from the customer before proceeding
"""
    elif is_login_inquiry and is_customer_verified:
        verification_warning = "**✅ CUSTOMER VERIFICATION COMPLETED:** The customer has been successfully verified."

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
The customer type is "{customer_type}" and the customer has sent {customer_message_count} messages.
The customer may be showing signs of continued frustration or dissatisfaction.

**INCLUDE THE FOLLOWING COPING STRATEGY FORMAT IN YOUR RESPONSE:**
1. **Immediate Acknowledgment** (1-2 sentences): Acknowledge their frustration/specific concern explicitly
2. **Specific Solution Recap** (2-3 sentences): Clearly restate the solution/step provided previously
3. **Escalation or Follow-up Offer** (1-2 sentences): Offer to escalate to supervisor/higher level support
4. **Closing with Assurance** (1 sentence): Reassure that their concern is being taken seriously
"""

    diversity_instruction = ""
    if previous_agent_responses:
        diversity_instruction = """
**CRITICAL DIVERSITY REQUIREMENT - STRICTLY ENFORCED:**
- You MUST generate a COMPLETELY DIFFERENT response from ALL previous agent responses
- Use COMPLETELY DIFFERENT wording, phrasing, sentence structures, and vocabulary
- DO NOT copy, paraphrase, or reuse ANY phrases from previous responses
"""

    variation_approaches = [
        "Start with a different greeting or acknowledgment style",
        "Use a different problem-solving approach or framework",
        "Present information in a different order",
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
4. Reference specific details from their inquiry (order numbers, dates, products, locations, etc.)
5. Keep the tone appropriate for the customer type: {customer_type}
6. Do NOT include any markdown formatting, just plain text
7. Generate a COMPLETELY UNIQUE and VARIED response - avoid repeating ANY similar phrases

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


def generate_customer_reaction(current_lang_key: str, is_call: bool = False) -> str:
    """고객의 다음 반응을 생성하는 LLM 호출 (채팅 전용)"""
    history_text = get_chat_history_for_prompt()
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG.get(current_lang_key, LANG["ko"])

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
2. **[CRITICAL: Mandatory Information Submission]** If the agent requested any critical information, you MUST provide it:
   - Order/Booking Number, eSIM related details, Child-related product details, Exception/Refund Reason
3. **[CRITICAL: Solution Acknowledgment]** If the agent provided a clear and accurate solution:
   - You MUST respond with appreciation and satisfaction, like "{L_local['customer_positive_response']}"
4. If the agent's LAST message was the closing confirmation: "{L_local['customer_closing_confirm']}"
   - If you have NO additional questions: You MUST reply with "{L_local['customer_no_more_inquiries']}".
   - If you DO have additional questions: You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS IMMEDIATELY.
5. Do NOT repeat your initial message or previous responses unless necessary.
6. Output ONLY the customer's next message.
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


def generate_customer_reaction_for_call(current_lang_key: str, last_agent_response: str) -> str:
    """전화 시뮬레이터 전용 고객 반응 생성 (마지막 에이전트 응답 중심)"""
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
    
    closing_msg = L_local['customer_closing_confirm']
    is_closing_question = (
        closing_msg in last_agent_response or 
        any(phrase in last_agent_response for phrase in [
            "다른 문의가 있으신가요", "추가 문의가 있으신가요", "anything else", "other questions"
        ])
    )
    
    initial_inquiry = st.session_state.get("inquiry_text", "")
    agent_asking_for_details = any(phrase in last_agent_response.lower() for phrase in [
        "문의 내용", "상세히", "자세히", "구체적으로", "설명", "inquiry details", "more details", "explain"
    ])
    
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
- Keep your response short and conversational
- Your emotional state: {customer_emotion} - respond with {emotion_tone} tone
═══════════════════════════════════════════════════════════════════"""

    if is_closing_question:
        call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

{history_text}

The agent just asked: "{last_agent_text}"

CRITICAL RULES FOR CLOSING CONFIRMATION:
1. If you have NO additional questions and the conversation is resolved:
   - You MUST reply with: "{L_local['customer_no_more_inquiries']}"
2. If you DO have additional questions or the issue is NOT fully resolved:
   - You MUST reply with: "{L_local['customer_has_additional_inquiries']}" AND immediately state your additional question
3. Your response MUST be ONLY one of the two options above, in {lang_name}.

Your response:
"""
    else:
        call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

{history_text}

RULES:
1. **CRITICAL**: Respond DIRECTLY and ACCURATELY to what the agent JUST SAID: "{last_agent_text}"
2. **If agent asked a question** → Answer it SPECIFICALLY and DIRECTLY
3. **If agent requested information** → Provide the EXACT information requested
4. **If agent gave a solution** → Acknowledge it clearly and indicate if you understand or need clarification
5. Keep your response short (1-2 sentences max) and focused ONLY on what the agent just said
6. Match your tone to your emotional state ({customer_emotion}) - be {emotion_tone}
7. **CRITICAL - NO CLOSING PHRASES**: DO NOT say "없습니다. 감사합니다" unless the agent explicitly asks if you have any other questions

Your response:
"""
    try:
        reaction = run_llm(call_prompt)
        reaction_text = reaction.strip()
        
        if is_closing_question:
            if L_local['customer_no_more_inquiries'] in reaction_text:
                return L_local['customer_no_more_inquiries']
            elif L_local['customer_has_additional_inquiries'] in reaction_text:
                return reaction_text
            else:
                return L_local['customer_no_more_inquiries']
        
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

Your response:
"""
    try:
        reaction = run_llm(call_prompt)
        reaction_text = reaction.strip()
        
        no_more_keywords = ["없습니다", "감사합니다", "No, that will be all", "no more"]
        has_no_more = any(keyword in reaction_text for keyword in no_more_keywords)
        has_inquiry_mention = initial_query_text.lower() in reaction_text.lower() or any(
            word in reaction_text.lower() for word in initial_query_text.split()[:3]
        )
        
        if has_no_more and not has_inquiry_mention:
            retry_prompt = f"""
You are a CUSTOMER in a phone call. The agent just greeted you.

You called because: "{initial_query_text}"
The agent said: "{agent_greeting_text}"

**CRITICAL - YOU MUST:**
1. Acknowledge the greeting (e.g., "안녕하세요" or "네")
2. IMMEDIATELY mention your inquiry: "{initial_query_text}"
3. DO NOT say "없습니다. 감사합니다" or any closing phrases
4. You are STARTING the conversation, not ending it

Your response (in {lang_name}, MUST mention your inquiry):
"""
            reaction = run_llm(retry_prompt)
            reaction_text = reaction.strip()
            
            if any(keyword in reaction_text for keyword in no_more_keywords) and not has_inquiry_mention:
                reaction_text = f"안녕하세요. {initial_query_text}에 대해 문의드리고 싶어서 연락드렸습니다."
        
        return reaction_text
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def generate_customer_closing_response(current_lang_key: str) -> str:
    """에이전트의 마지막 확인 질문에 대한 고객의 최종 답변 생성 (채팅용)"""
    history_text = get_chat_history_for_prompt()
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
4. Output ONLY the customer's next message.
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
        return L_local['customer_has_additional_inquiries']


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
        return f"❌ LLM Key is missing. Cannot generate summary."

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ Summary Generation Error: {e}"


def generate_outbound_call_summary(customer_query: str, current_lang_key: str, target: str) -> str:
    """Simulates an outbound call to a local partner or customer and generates a summary of the outcome."""
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








