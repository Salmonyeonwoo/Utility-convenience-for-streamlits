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
시뮬레이션 고객 반응 생성 모듈
"""

import streamlit as st
from llm_client import run_llm
from lang_pack import LANG
from simulation_handler_base import get_chat_history_for_prompt

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
    closing_keywords = [
        "다른 문의 사항", "추가 문의 사항", "다른 문의가", "추가 문의가",
        "다른 도움이 필요", "다른 문의 없으", "추가 문의 없으",
        "anything else", "other questions", "any other inquiries", "any other questions",
        "other inquiries", "additional inquiries", "anything else we can",
        "他のお問合せ", "追加の問い合わせ", "他にご質問"
    ]
    
    is_closing_question = (
        closing_msg in last_agent_response or 
        any(phrase in last_agent_response for phrase in closing_keywords) or
        ("또 다른" in last_agent_response and ("문의" in last_agent_response or "질문" in last_agent_response) and ("없" in last_agent_response or "없으" in last_agent_response)) or
        ("다른" in last_agent_response and "문의" in last_agent_response and ("없" in last_agent_response or "없으" in last_agent_response or "있" in last_agent_response))
    )
    
    initial_inquiry = st.session_state.get("inquiry_text", "")
    agent_asking_for_details = any(phrase in last_agent_response.lower() for phrase in [
        "문의 내용", "상세히", "자세히", "구체적으로", "설명", "어떤 문의", "무엇을",
        "inquiry details", "more details", "explain", "what inquiry", "what is"
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
- DO NOT mention "추가 문의 사항" unless the agent explicitly asks "다른 문의가 있나요?"
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
            no_more_text = L_local['customer_no_more_inquiries']
            has_additional_text = L_local['customer_has_additional_inquiries']
            
            if no_more_text in reaction_text or any(keyword in reaction_text for keyword in [
                "없습니다", "감사합니다", "No, that will be all", "no more", "結構です"
            ]):
                return no_more_text
            elif has_additional_text in reaction_text or any(phrase in reaction_text for phrase in [
                "추가 문의", "다른 문의", "additional", "other inquiries", "追加の問い合わせ"
            ]):
                if len(reaction_text) > len(has_additional_text):
                    return reaction_text
                else:
                    return has_additional_text
            else:
                return no_more_text
        
        else:
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
            
            if has_no_more:
                if "기종" in last_agent_text or "model" in last_agent_text.lower() or "phone" in last_agent_text.lower():
                    reaction_text = "아이폰 14입니다." if current_lang_key == "ko" else "iPhone 14."
                elif "번호" in last_agent_text or "number" in last_agent_text.lower():
                    reaction_text = "주문 번호는 12345입니다." if current_lang_key == "ko" else "The order number is 12345."
                elif "날짜" in last_agent_text or "date" in last_agent_text.lower():
                    reaction_text = "12월 12일입니다." if current_lang_key == "ko" else "December 12th."
                else:
                    reaction_text = "네, 알겠습니다." if current_lang_key == "ko" else "Yes, I understand."
            
            if has_additional_inquiry_mention:
                for phrase in additional_inquiry_phrases:
                    reaction_text = reaction_text.replace(phrase, "").strip()
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
        import streamlit as st
        st.error(f"고객 최종 반응 생성 오류: {e}")
        return L_local['customer_has_additional_inquiries']

