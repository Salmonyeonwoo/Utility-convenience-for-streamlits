import streamlit as st
import json
import re
from datetime import datetime
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


def get_api_key(api_name="openai"):
    """API 키 가져오기 - llm_client에서 가져오기"""
    try:
        from llm_client import get_api_key as llm_get_api_key
        return llm_get_api_key(api_name)
    except ImportError:
        # Fallback: 환경변수에서 직접 가져오기
        import os
        return os.getenv(f"{api_name.upper()}_API_KEY", "")


def get_rag_chatbot_response(user_query: str, context: List[dict] = None) -> str:
    """RAG 챗봇 응답 생성"""
    api_key = get_api_key("openai") or get_api_key("gemini")
    if not api_key:
        return "OpenAI 또는 Gemini API 키가 설정되지 않았습니다."
    
    try:
        openai_key = get_api_key("openai")
        if not openai_key:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_key
        )
        
        # 컨텍스트 구성
        context_text = ""
        if context:
            context_text = "\n".join([f"- {item}" for item in context[-5:]])
        
        system_prompt = f"""당신은 여행사 정보를 제공하는 AI 어시스턴트입니다.
        
사용 가능한 컨텍스트 정보:
{context_text}

사용자의 질문에 대해 컨텍스트 정보를 활용하여 정확하고 도움이 되는 답변을 제공하세요.
컨텍스트에 없는 정보는 추측하지 말고, 정확히 모른다고 답변하세요."""
        
        messages = [SystemMessage(content=system_prompt)]
        messages.append(HumanMessage(content=user_query))
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"챗봇 응답 생성 중 오류: {str(e)}"


def perform_rag_analysis(customer_message: str, customer_info: dict) -> dict:
    """RAG 분석 수행 (고객 메시지 분석)"""
    api_key = get_api_key("openai") or get_api_key("gemini")
    if not api_key:
        return {
            "sentiment": "neutral",
            "intent": "일반 문의",
            "keywords": [],
            "suggested_response": "",
            "confidence": 0.0
        }
    
    try:
        openai_key = get_api_key("openai")
        if not openai_key:
            return {
                "sentiment": "neutral",
                "intent": "일반 문의",
                "keywords": [],
                "suggested_response": "API 키가 설정되지 않았습니다.",
                "confidence": 0.0
            }
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=openai_key
        )
        
        analysis_prompt = f"""다음 고객 메시지를 분석하여 다음 정보를 JSON 형식으로 제공해주세요:
1. sentiment: 감정 분석 (positive, neutral, negative)
2. intent: 의도 (패키지 문의, 예약, 취소/변경, 일반 문의 등)
3. keywords: 주요 키워드 리스트
4. suggested_response: 추천 응답
5. confidence: 신뢰도 (0.0-1.0)

고객 정보:
- 이름: {customer_info.get('customer_name', 'N/A')}
- 성향: {customer_info.get('personality', 'N/A')}

고객 메시지: {customer_message}

JSON 형식으로만 응답해주세요."""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # JSON 파싱 시도
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # 파싱 실패 시 기본값 반환
            return {
                "sentiment": "neutral",
                "intent": "일반 문의",
                "keywords": customer_message.split()[:5],
                "suggested_response": response.content,
                "confidence": 0.7
            }
    except Exception as e:
        return {
            "sentiment": "neutral",
            "intent": "일반 문의",
            "keywords": [],
            "suggested_response": f"분석 중 오류: {str(e)}",
            "confidence": 0.0
        }


def get_ai_response(customer_message: str, customer_info: dict, chat_history: List[dict]) -> str:
    """OpenAI를 사용하여 상담원 응답 생성"""
    api_key = get_api_key("openai") or get_api_key("gemini")
    if not api_key:
        return "OpenAI 또는 Gemini API 키가 설정되지 않았습니다. 환경변수나 세션 상태에서 설정해주세요."
    
    try:
        openai_key = get_api_key("openai")
        if not openai_key:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_key
        )
        
        # 시스템 프롬프트 구성
        system_prompt = f"""당신은 여행사 상담원입니다. 고객에게 친절하고 전문적으로 응대해야 합니다.

고객 정보:
- 이름: {customer_info.get('customer_name', 'N/A')}
- 성향: {customer_info.get('personality', 'N/A')}
- 성향 요약: {customer_info.get('personality_summary', 'N/A')}
- 선호 여행지: {customer_info.get('preferred_destination', 'N/A')}
- 예산: {customer_info.get('travel_budget', 'N/A')}

고객의 성향에 맞춰 맞춤형 여행 상담을 제공하세요. 친절하고 전문적인 톤을 유지하며, 구체적인 정보를 제공하세요."""
        
        # 채팅 히스토리 구성
        messages = [SystemMessage(content=system_prompt)]
        
        # 최근 10개 메시지만 포함 (컨텍스트 길이 제한)
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        
        for msg in recent_history:
            if msg['sender'] == 'customer':
                messages.append(HumanMessage(content=msg['message']))
            elif msg['sender'] == 'operator':
                messages.append(AIMessage(content=msg['message']))
        
        # 현재 고객 메시지 추가
        messages.append(HumanMessage(content=customer_message))
        
        # AI 응답 생성
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}"


def translate_text(text: str, target_lang: str) -> str:
    """텍스트 번역"""
    api_key = get_api_key("openai") or get_api_key("gemini")
    if not api_key:
        return text
    
    try:
        openai_key = get_api_key("openai")
        if not openai_key:
            return text
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=openai_key
        )
        
        lang_map = {
            "ko": "한국어",
            "en": "영어",
            "ja": "일본어"
        }
        
        prompt = f"다음 텍스트를 {lang_map.get(target_lang, target_lang)}로 번역해주세요. 번역만 출력하세요:\n\n{text}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"번역 오류: {str(e)}"


def summarize_conversation(chat_history: List[dict], target_lang: str = "ko") -> str:
    """대화 요약"""
    api_key = get_api_key("openai") or get_api_key("gemini")
    if not api_key:
        return "요약을 생성할 수 없습니다. API 키를 설정해주세요."
    
    try:
        openai_key = get_api_key("openai")
        if not openai_key:
            return "요약을 생성할 수 없습니다. API 키를 설정해주세요."
        
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=openai_key
        )
        
        # 대화 내용 추출
        conversation_text = "\n".join([
            f"{msg['sender_name']}: {msg['message']}" 
            for msg in chat_history[-20:]  # 최근 20개 메시지
        ])
        
        lang_map = {
            "ko": "한국어",
            "en": "영어",
            "ja": "일본어"
        }
        
        prompt = f"""다음 고객 상담 대화를 {lang_map.get(target_lang, target_lang)}로 요약해주세요.
주요 내용, 고객 문의 사항, 해결 방안을 포함하여 간결하게 작성해주세요.

대화 내용:
{conversation_text}

요약:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"요약 생성 오류: {str(e)}"


def transfer_to_language_team(customer_id: str, target_lang: str, summary: str):
    """언어별 팀으로 이관"""
    # 이관 기록 저장
    transfer_record = {
        "customer_id": customer_id,
        "from_lang": st.session_state.get('language', 'ko'),
        "to_lang": target_lang,
        "summary": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 이관 기록 파일에 저장
    try:
        import os
        os.makedirs('data', exist_ok=True)
        with open('data/transfers.json', 'r', encoding='utf-8') as f:
            transfers = json.load(f)
    except FileNotFoundError:
        transfers = []
    
    transfers.append(transfer_record)
    
    with open('data/transfers.json', 'w', encoding='utf-8') as f:
        json.dump(transfers, f, ensure_ascii=False, indent=2)
    
    return transfer_record

