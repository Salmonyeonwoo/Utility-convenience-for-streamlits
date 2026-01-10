# ========================================
# video_handler_analysis.py
# 비디오 처리 - 텍스트 분석 모듈
# ========================================

import json
import re
import streamlit as st
from typing import List, Dict, Any
from llm_client import run_llm
from lang_pack import LANG


def analyze_text_for_video_selection(text: str, current_lang_key: str, 
                                     agent_last_response: str = None,
                                     conversation_context: List[Dict] = None) -> Dict[str, Any]:
    """
    LLM을 사용하여 텍스트를 분석하고 적절한 감정 상태와 제스처를 판단합니다.
    OpenAI/Gemini API를 활용한 영상 RAG의 핵심 기능입니다.
    
    Args:
        text: 분석할 텍스트 (고객의 질문/응답)
        current_lang_key: 현재 언어 키
        agent_last_response: 에이전트의 마지막 답변 (선택적)
        conversation_context: 대화 컨텍스트 (선택적)
    
    Returns:
        {
            "emotion": "NEUTRAL" | "HAPPY" | "ANGRY" | "ASKING" | "SAD",
            "gesture": "NONE" | "HAND_WAVE" | "NOD" | "SHAKE_HEAD" | "POINT",
            "urgency": "LOW" | "MEDIUM" | "HIGH",
            "satisfaction_delta": -1.0 to 1.0,
            "confidence": 0.0-1.0
        }
    """
    if not text or not text.strip():
        return {
            "emotion": "NEUTRAL", 
            "gesture": "NONE", 
            "urgency": "LOW",
            "satisfaction_delta": 0.0,
            "confidence": 0.5
        }
    
    L = LANG.get(current_lang_key, LANG["ko"])
    
    # 에이전트 답변 기반 예측 컨텍스트 구성
    context_info = ""
    if agent_last_response:
        context_info = f"""
에이전트의 마지막 답변: "{agent_last_response}"

에이전트의 답변을 고려했을 때, 고객이 지금 말하는 내용은 어떤 감정을 수반할 것인지 예측하세요.
예를 들어:
- 에이전트가 솔루션을 제시했다면 → 고객은 HAPPY 또는 ASKING (추가 질문)
- 에이전트가 거절했다면 → 고객은 ANGRY 또는 SAD
- 에이전트가 질문을 했다면 → 고객은 ASKING (답변) 또는 NEUTRAL
"""
    
    # 만족도 변화 분석 컨텍스트
    satisfaction_context = ""
    if conversation_context and len(conversation_context) > 1:
        recent_emotions = []
        for msg in conversation_context[-3:]:
            if msg.get("role") == "customer_rebuttal" or msg.get("role") == "customer":
                recent_emotions.append(msg.get("content", ""))
        
        if len(recent_emotions) >= 2:
            satisfaction_context = f"""
최근 대화 흐름:
- 이전 고객 메시지: "{recent_emotions[-2] if len(recent_emotions) >= 2 else ''}"
- 현재 고객 메시지: "{recent_emotions[-1]}"

만족도 변화를 분석하세요:
- 이전보다 더 긍정적이면 satisfaction_delta > 0
- 이전보다 더 부정적이면 satisfaction_delta < 0
- 비슷하면 satisfaction_delta ≈ 0
"""
    
    prompt = _build_analysis_prompt(text, context_info, satisfaction_context)
    
    try:
        if st.session_state.is_llm_ready:
            response_text = run_llm(prompt)
            result = _parse_llm_response(response_text)
            if result:
                return result
    except Exception as e:
        print(f"LLM 분석 오류: {e}")
    
    # 키워드 기반 분석 (폴백)
    return _analyze_by_keywords(text)


def _build_analysis_prompt(text: str, context_info: str, satisfaction_context: str) -> str:
    """분석 프롬프트 생성"""
    return f"""다음 고객의 텍스트를 분석하여 적절한 감정 상태, 제스처, 긴급도, 만족도 변화를 판단하세요.

고객 텍스트: "{text}"
{context_info}
{satisfaction_context}

다음 JSON 형식으로만 응답하세요 (다른 설명 없이):
{{
    "emotion": "NEUTRAL" | "HAPPY" | "ANGRY" | "ASKING" | "SAD",
    "gesture": "NONE" | "HAND_WAVE" | "NOD" | "SHAKE_HEAD" | "POINT",
    "urgency": "LOW" | "MEDIUM" | "HIGH",
    "satisfaction_delta": -1.0 to 1.0,
    "confidence": 0.0-1.0
}}

감정 판단 기준:
- HAPPY: 긍정적 표현, 감사, 만족, 해결됨
- ANGRY: 불만, 화남, 거부, 강한 부정
- ASKING: 질문, 궁금함, 확인 요청
- SAD: 슬픔, 실망, 좌절
- NEUTRAL: 중립적 표현 (기본값)

제스처 판단 기준:
- HAND_WAVE: 인사, 환영
- NOD: 동의, 긍정, 이해
- SHAKE_HEAD: 부정, 거부
- POINT: 설명, 지시, 특정 항목 언급
- NONE: 특별한 제스처 없음 (기본값)

긴급도 판단 기준:
- HIGH: 즉시 해결 필요
- MEDIUM: 빠른 해결 선호
- LOW: 일반적인 문의 (기본값)

만족도 변화 (satisfaction_delta):
- 1.0: 매우 만족
- 0.0: 중립
- -1.0: 매우 불만족

JSON만 응답하세요:"""


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """LLM 응답 파싱"""
    try:
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            valid_emotions = ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"]
            valid_gestures = ["NONE", "HAND_WAVE", "NOD", "SHAKE_HEAD", "POINT"]
            valid_urgencies = ["LOW", "MEDIUM", "HIGH"]
            
            emotion = result.get("emotion", "NEUTRAL")
            gesture = result.get("gesture", "NONE")
            urgency = result.get("urgency", "LOW")
            satisfaction_delta = float(result.get("satisfaction_delta", 0.0))
            confidence = float(result.get("confidence", 0.7))
            
            if emotion not in valid_emotions:
                emotion = "NEUTRAL"
            if gesture not in valid_gestures:
                gesture = "NONE"
            if urgency not in valid_urgencies:
                urgency = "LOW"
            
            context_keywords = _extract_context_keywords(response_text, emotion, satisfaction_delta)
            
            return {
                "emotion": emotion,
                "gesture": gesture,
                "urgency": urgency,
                "satisfaction_delta": max(-1.0, min(1.0, satisfaction_delta)),
                "context_keywords": context_keywords,
                "confidence": max(0.0, min(1.0, confidence))
            }
    except json.JSONDecodeError:
        pass
    return None


def _extract_context_keywords(text: str, emotion: str, satisfaction_delta: float) -> List[str]:
    """상황별 키워드 추출"""
    context_keywords = []
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["주문번호", "order number", "주문 번호"]):
        context_keywords.append("order_number")
    if any(word in text_lower for word in ["해결", "완료", "감사", "solution", "resolved"]):
        if satisfaction_delta > 0.3:
            context_keywords.append("solution_accepted")
    if any(word in text_lower for word in ["거절", "불가", "안 됩니다", "denied", "cannot"]):
        if emotion == "ANGRY":
            context_keywords.append("policy_denial")
    
    return context_keywords


def _analyze_by_keywords(text: str) -> Dict[str, Any]:
    """키워드 기반 분석 (폴백)"""
    text_lower = text.lower()
    emotion = "NEUTRAL"
    gesture = "NONE"
    urgency = "LOW"
    satisfaction_delta = 0.0
    
    # 감정 키워드 분석
    if any(word in text_lower for word in ["감사", "좋아", "완벽", "만족", "고마워", "해결"]):
        emotion = "HAPPY"
        satisfaction_delta = 0.5
    elif any(word in text_lower for word in ["화", "불만", "거절", "불가능", "안 됩니다", "말도 안 돼"]):
        emotion = "ANGRY"
        satisfaction_delta = -0.5
    elif any(word in text_lower for word in ["어떻게", "왜", "알려", "질문", "궁금", "주문번호"]):
        emotion = "ASKING"
    elif any(word in text_lower for word in ["슬프", "실망", "아쉽", "그렇다면"]):
        emotion = "SAD"
        satisfaction_delta = -0.3
    
    # 긴급도 키워드 분석
    if any(word in text_lower for word in ["지금 당장", "바로", "긴급", "중요해요", "즉시"]):
        urgency = "HIGH"
    elif any(word in text_lower for word in ["빨리", "가능한 한", "최대한"]):
        urgency = "MEDIUM"
    
    # 제스처 키워드 분석
    if any(word in text_lower for word in ["안녕", "반갑", "인사"]):
        gesture = "HAND_WAVE"
    elif any(word in text_lower for word in ["네", "맞아", "그래", "동의", "알겠습니다"]):
        gesture = "NOD"
        if emotion == "HAPPY":
            satisfaction_delta = 0.3
    elif any(word in text_lower for word in ["아니", "안 됩니다", "거절"]):
        gesture = "SHAKE_HEAD"
        satisfaction_delta = -0.2
    elif any(word in text_lower for word in ["여기", "이것", "저것", "이거", "주문번호"]):
        gesture = "POINT"
    
    context_keywords = _extract_context_keywords(text, emotion, satisfaction_delta)
    
    return {
        "emotion": emotion,
        "gesture": gesture,
        "urgency": urgency,
        "satisfaction_delta": satisfaction_delta,
        "context_keywords": context_keywords,
        "confidence": 0.6
    }
