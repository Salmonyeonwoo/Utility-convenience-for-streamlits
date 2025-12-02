"""
프롬프트 생성 함수 모듈
고객 반응 생성, 에이전트 응답 초안, 이력 요약 등의 프롬프트를 생성합니다.
"""
import streamlit as st
from typing import List, Dict, Any
from utils.llm_clients import run_llm, get_llm_client
from utils.i18n import LANG


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


def generate_customer_reaction(current_lang_key: str, is_call: bool = False) -> str:
    """고객의 다음 반응을 생성하는 LLM 호출 (채팅 전용)"""
    history_text = get_chat_history_for_prompt()
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    # 고객 유형 가져오기
    customer_type = st.session_state.get('customer_type_sim_select', '일반적인 문의')
    
    prompt = f"""
You are simulating a customer in a customer support chat.
Generate the customer's next response based on the conversation history.

Requirements:
1. The response MUST be in {lang_name}
2. The customer type is: {customer_type}
3. Be natural and realistic
4. If the agent provided a solution, the customer should acknowledge it (positively or with follow-up questions)
5. If the agent didn't provide a solution, the customer may express frustration or ask for clarification
6. Keep responses concise (1-3 sentences)

Conversation History:
{history_text}

Generate the customer's next response:
"""
    
    if not st.session_state.is_llm_ready:
        return ""
    
    try:
        reaction = run_llm(prompt).strip()
        # 마크다운 제거
        if reaction.startswith("```"):
            lines = reaction.split("\n")
            reaction = "\n".join(lines[1:-1]) if len(lines) > 2 else reaction
        return reaction
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def generate_agent_response_draft(current_lang_key: str) -> str:
    """고객 응답을 기반으로 AI가 에이전트 응답 초안을 생성"""
    L = LANG[current_lang_key]
    history_text = get_chat_history_for_prompt(include_attachment=True)
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"\n[고객 첨부 파일 정보: {attachment_context}]\n"
    else:
        attachment_context = ""

    # 고객 유형
    customer_type = st.session_state.get('customer_type_sim_select', '일반적인 문의')

    draft_prompt = f"""
You are an AI assistant helping a customer support agent write a professional response.

Based on the conversation history below, generate a draft response that the agent can review and modify before sending.

Requirements:
1. The response MUST be in {lang_name}
2. Be professional, empathetic, and solution-oriented
3. Address the customer's latest inquiry or concern
4. If the customer asked a question, provide a clear answer
5. If the customer expressed dissatisfaction, show empathy and offer solutions
6. Keep the tone appropriate for the customer type: {customer_type}
7. Do NOT include any markdown formatting, just plain text

Conversation History:
{history_text}
{attachment_context}

Generate the agent's response draft:
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


def summarize_history_with_ai(current_lang_key: str) -> str:
    """대화 이력을 AI로 요약"""
    history_text = get_chat_history_for_prompt()
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    prompt = f"""
Summarize the following customer support conversation in {lang_name}.

Include:
1. Customer's main issue
2. Solutions provided
3. Current status

Conversation:
{history_text}

Summary (in {lang_name}):
"""
    
    if not st.session_state.is_llm_ready:
        return ""
    
    try:
        summary = run_llm(prompt).strip()
        return summary
    except Exception as e:
        return f"❌ 요약 생성 오류: {e}"


def generate_customer_reaction_for_call(current_lang_key: str, last_agent_response: str) -> str:
    """전화 통화용 고객 반응 생성"""
    history_text = get_chat_history_for_prompt()
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    prompt = f"""
You are simulating a customer in a phone call with customer support.
Generate the customer's verbal response to the agent's last statement.

Last Agent Response: {last_agent_response}

Conversation History:
{history_text}

Requirements:
1. Response MUST be in {lang_name}
2. Be natural and conversational (as if speaking on the phone)
3. Keep it brief (1-2 sentences)
4. Show appropriate reaction to the agent's response

Generate the customer's verbal response:
"""
    
    if not st.session_state.is_llm_ready:
        return ""
    
    try:
        reaction = run_llm(prompt).strip()
        return reaction
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def summarize_history_for_call(call_logs: List[Dict[str, str]], initial_query: str, current_lang_key: str) -> str:
    """전화 통화 이력 요약"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    call_text = "\n".join([f"{log.get('role', 'unknown')}: {log.get('content', '')}" for log in call_logs])
    
    prompt = f"""
Summarize the following phone call conversation in {lang_name}.

Initial Query: {initial_query}

Call Logs:
{call_text}

Summary (in {lang_name}):
"""
    
    if not st.session_state.is_llm_ready:
        return ""
    
    try:
        summary = run_llm(prompt).strip()
        return summary
    except Exception as e:
        return f"❌ 통화 요약 생성 오류: {e}"


def generate_customer_closing_response(current_lang_key: str) -> str:
    """고객의 종료 응답 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L = LANG[current_lang_key]
    
    # 간단한 종료 응답 (LLM 호출 없이)
    responses = {
        "ko": ["알겠습니다. 감사합니다.", "네, 이해했습니다. 좋은 하루 되세요.", "감사합니다."],
        "en": ["I understand. Thank you.", "Got it. Have a nice day.", "Thank you."],
        "ja": ["承知いたしました。ありがとうございます。", "分かりました。良い一日をお過ごしください。", "ありがとうございます。"]
    }
    
    import random
    return random.choice(responses.get(current_lang_key, responses["en"]))


def generate_agent_first_greeting(lang_key: str, initial_query: str) -> str:
    """에이전트의 첫 인사말 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[lang_key]
    L = LANG[lang_key]
    
    prompt = f"""
Generate a professional greeting for a customer support agent starting a conversation.

Customer's initial query: {initial_query}

Requirements:
1. Response MUST be in {lang_name}
2. Be warm, professional, and empathetic
3. Acknowledge the customer's query
4. Show willingness to help
5. Keep it concise (2-3 sentences)

Generate the greeting:
"""
    
    if not st.session_state.is_llm_ready:
        return L.get("simulator_header", "안녕하세요. 고객 지원팀입니다.")
    
    try:
        greeting = run_llm(prompt).strip()
        return greeting
    except Exception as e:
        return L.get("simulator_header", "안녕하세요. 고객 지원팀입니다.")


def generate_outbound_call_summary(customer_query: str, current_lang_key: str, target: str) -> str:
    """전화 발신 시뮬레이션 요약 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    history_text = get_chat_history_for_prompt(include_attachment=True)
    
    if not history_text:
        history_text = f"Initial Customer Query: {customer_query}"
    
    policy_context = st.session_state.supervisor_policy_context or ""
    
    summary_prompt = f"""
You are an AI simulating a quick, high-stakes phone call placed by the customer support agent to a '{target}' (either a local partner/vendor or the customer).

Generate a concise summary of the OUTCOME of this simulated phone call.
The summary MUST be professional and strictly in {lang_name}.

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


def analyze_customer_profile(customer_query: str, current_lang_key: str) -> Dict[str, Any]:
    """고객 프로필 분석 (감정 점수, 긴급도 등)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    
    prompt = f"""
Analyze the following customer query and provide a JSON response with:
- sentiment_score: 0-100 (0=very negative, 100=very positive)
- urgency_score: 0-100 (0=not urgent, 100=very urgent)
- customer_type: "general", "difficult", or "highly_dissatisfied"

Customer Query: {customer_query}

Respond ONLY with valid JSON in this format:
{{"sentiment_score": 50, "urgency_score": 70, "customer_type": "general"}}
"""
    
    if not st.session_state.is_llm_ready:
        return {"sentiment_score": 50, "urgency_score": 50, "customer_type": "general"}
    
    try:
        response = run_llm(prompt).strip()
        # JSON 파싱
        import json
        if response.startswith("```"):
            response = response.strip("```json").strip("```").strip()
        profile = json.loads(response)
        return profile
    except Exception as e:
        return {"sentiment_score": 50, "urgency_score": 50, "customer_type": "general"}







