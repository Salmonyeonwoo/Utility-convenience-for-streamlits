"""대화 처리 및 응답 생성 모듈 (LLM 기반)"""

import streamlit as st
from llm_client import run_llm
from lang_pack import LANG

def get_call_history_for_prompt():
    """call_messages에서 대화 기록을 추출하여 프롬프트에 사용할 문자열 형태로 반환"""
    history_str = ""
    if "call_messages" in st.session_state:
        for msg in st.session_state.call_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "customer":
                history_str += f"고객: {content}\n"
            elif role == "agent":
                history_str += f"AI 상담원: {content}\n"
    return history_str

def generate_agent_response(user_text, customer_insight=None, needs_more_info=False, info_requested=None):
    """AI 에이전트 응답 생성 (LLM 기반, 대화 맥락 고려)"""
    if info_requested is None:
        info_requested = []
    
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang]
    
    # LLM 준비 상태 확인
    if not st.session_state.get("is_llm_ready", False):
        # LLM이 없을 때 fallback 응답
        return f"네, '{user_text}' 말씀하신 내용 확인했습니다. 감사합니다. 처리해드리겠습니다."
    
    # 대화 기록 가져오기
    history_text = get_call_history_for_prompt()
    
    # 이전 에이전트 응답들 확인 (반복 방지)
    previous_agent_responses = []
    if "call_messages" in st.session_state:
        previous_agent_responses = [
            msg.get("content", "") for msg in st.session_state.call_messages 
            if msg.get("role") == "agent"
        ][-3:]  # 최근 3개만
    
    previous_responses_context = ""
    if previous_agent_responses:
        previous_responses_context = f"\n[이전 에이전트 응답들 (참고용, 동일하게 반복하지 말 것):\n"
        for i, prev_resp in enumerate(previous_agent_responses, 1):
            prev_resp_preview = prev_resp[:200] + "..." if len(prev_resp) > 200 else prev_resp
            previous_responses_context += f"{i}. {prev_resp_preview}\n"
        previous_responses_context += "]\n"
    
    # 고객 인사이트 정보
    customer_insight_context = ""
    if customer_insight:
        intent = customer_insight.get('intent', '일반 문의')
        emotion = customer_insight.get('emotion', 'NEUTRAL')
        customer_insight_context = f"""
[고객 인사이트 정보]
- 의도: {intent}
- 감정: {emotion}
"""
    
    # 첫 인사 여부 확인
    is_first_greeting = False
    agent_message_count = 0
    if "call_messages" in st.session_state:
        agent_message_count = sum(1 for msg in st.session_state.call_messages if msg.get("role") == "agent")
        is_first_greeting = (agent_message_count == 0)
    
    # 프롬프트 생성
    greeting_instruction = ""
    if is_first_greeting:
        greeting_instruction = """
**FIRST GREETING RULE:**
- This is the FIRST message from the agent to the customer
- Start with a greeting like "안녕하세요" or "안녕하세요, 고객님"
- After the first greeting, DO NOT use "안녕하세요 고객님" again in subsequent responses
"""
    else:
        greeting_instruction = """
**IMPORTANT: NOT THE FIRST MESSAGE**
- DO NOT start with "안녕하세요 고객님" or formal greetings
- This is a continuation of the conversation, so respond naturally without repeating greetings
- Get straight to addressing the customer's inquiry
"""
    
    # 프롬프트 최적화 (더 짧고 빠른 응답 생성)
    prompt = f"""You are a customer service agent. Respond naturally in {lang_name}.

{greeting_instruction}

**REQUIREMENTS:**
- Respond in {lang_name}
- Address the customer's SPECIFIC inquiry directly
- Reference specific details they mentioned
- Be concise but helpful (2-4 sentences typically)
- Do NOT repeat previous responses
- Do NOT add "추가 문의 사항 있으신가요?" unless you provided a complete solution

**RECENT CONVERSATION:**
{history_text[-500:] if len(history_text) > 500 else history_text}

**CUSTOMER:** {user_text}

**PREVIOUS RESPONSES (avoid repeating):**
{previous_responses_context[-200:] if previous_responses_context and len(previous_responses_context) > 200 else previous_responses_context if previous_responses_context else "None"}

**RESPONSE (concise, direct, helpful):**
"""
    
    try:
        response = run_llm(prompt).strip()
        # 마크다운 코드 블록 제거
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
        return response
    except Exception as e:
        # LLM 오류 시 fallback
        return f"네, '{user_text}' 말씀하신 내용 확인했습니다. 감사합니다. 처리해드리겠습니다."

