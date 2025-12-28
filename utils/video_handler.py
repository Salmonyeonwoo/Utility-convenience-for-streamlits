"""
비디오 처리 모듈
비디오 선택, 렌더링, 매핑 데이터베이스 관리 기능을 제공합니다.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st

from config import DATA_DIR, VIDEO_MAPPING_DB_FILE
from utils.audio_handler import synthesize_tts


def analyze_text_for_video_selection(text: str, current_lang_key: str, 
                                     agent_last_response: str = None,
                                     conversation_context: List[Dict] = None) -> Dict[str, Any]:
    """
    LLM을 사용하여 텍스트를 분석하고 적절한 감정 상태와 제스처를 판단합니다.
    OpenAI/Gemini API를 활용한 영상 RAG의 핵심 기능입니다.
    
    ⭐ Gemini 제안 적용: 긴급도, 만족도 변화, 에이전트 답변 기반 예측 추가
    
    Args:
        text: 분석할 텍스트 (고객의 질문/응답)
        current_lang_key: 현재 언어 키
        agent_last_response: 에이전트의 마지막 답변 (선택적, 예측 정확도 향상)
        conversation_context: 대화 컨텍스트 (선택적, 만족도 변화 분석용)
    
    Returns:
        {
            "emotion": "NEUTRAL" | "HAPPY" | "ANGRY" | "ASKING" | "SAD",
            "gesture": "NONE" | "HAND_WAVE" | "NOD" | "SHAKE_HEAD" | "POINT",
            "urgency": "LOW" | "MEDIUM" | "HIGH",  # ⭐ 추가: 긴급도
            "satisfaction_delta": -1.0 to 1.0,  # ⭐ 추가: 만족도 변화 (-1: 감소, 0: 유지, 1: 증가)
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
    
    from lang_pack import LANG
    L = LANG.get(current_lang_key, LANG["ko"])
    
    # ⭐ Gemini 제안: 에이전트 답변 기반 예측 컨텍스트 구성
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
    
    # ⭐ Gemini 제안: 만족도 변화 분석 컨텍스트
    satisfaction_context = ""
    if conversation_context and len(conversation_context) > 1:
        # 최근 대화의 감정 변화 추적
        recent_emotions = []
        for msg in conversation_context[-3:]:  # 최근 3개 메시지
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
    
    # ⭐ Gemini 제안: 개선된 LLM 프롬프트 구성
    from llm_client import run_llm
    prompt = f"""다음 고객의 텍스트를 분석하여 적절한 감정 상태, 제스처, 긴급도, 만족도 변화를 판단하세요.

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

감정 판단 기준 (세분화):
- HAPPY: 긍정적 표현, 감사, 만족, 해결됨 ("감사합니다", "좋아요", "완벽해요", "이제 이해했어요")
- ANGRY: 불만, 화남, 거부, 강한 부정 ("화가 나요", "불가능해요", "거절합니다", "말도 안 돼요")
- ASKING: 질문, 궁금함, 확인 요청, 정보 요구 ("어떻게", "왜", "알려주세요", "주문번호가 뭐예요?")
- SAD: 슬픔, 실망, 좌절 ("슬프네요", "실망했어요", "아쉽습니다", "그렇다면 어쩔 수 없네요")
- NEUTRAL: 중립적 표현, 단순 정보 전달 (기본값)

제스처 판단 기준:
- HAND_WAVE: 인사, 환영 ("안녕하세요", "반갑습니다")
- NOD: 동의, 긍정, 이해 ("네", "맞아요", "그렇습니다", "알겠습니다")
- SHAKE_HEAD: 부정, 거부, 불만족 ("아니요", "안 됩니다", "그건 아니에요")
- POINT: 설명, 지시, 특정 항목 언급 ("여기", "이것", "저것", "주문번호는")
- NONE: 특별한 제스처 없음 (기본값)

긴급도 판단 기준:
- HIGH: 즉시 해결 필요, 긴급한 문제 ("지금 당장", "바로", "긴급", "중요해요")
- MEDIUM: 빠른 해결 선호, 중요하지만 긴급하지 않음
- LOW: 일반적인 문의, 긴급하지 않음 (기본값)

만족도 변화 (satisfaction_delta):
- 1.0: 매우 만족, 문제 해결됨, 감사 표현
- 0.5: 만족, 긍정적 반응
- 0.0: 중립, 변화 없음
- -0.5: 불만족, 부정적 반응
- -1.0: 매우 불만족, 화남, 거부

JSON만 응답하세요:"""

    try:
        # LLM 호출
        if st.session_state.is_llm_ready:
            response_text = run_llm(prompt)
            
            # JSON 파싱 시도 (강화된 오류 처리)
            try:
                # JSON 부분만 추출 (코드 블록 제거)
                import re
                import json
                
                # 마크다운 코드 블록 제거
                if "```" in response_text:
                    json_match_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if json_match_block:
                        json_text = json_match_block.group(1)
                    else:
                        json_text = response_text
                else:
                    json_text = response_text
                
                # JSON 객체 찾기 (중첩 중괄호 지원)
                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError as json_err:
                        # 파싱 실패 시 기본값으로 계속 진행
                        print(f"비디오 핸들러 JSON 파싱 오류: {json_err}")
                        raise  # 외부 except로 전달
                    
                    # 유효성 검사
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
                    
                    # ⭐ Gemini 제안: 상황별 키워드 추출
                    context_keywords = []
                    text_lower_for_context = text.lower()
                    
                    # 주요 상황별 키워드 매핑
                    if any(word in text_lower_for_context for word in ["주문번호", "order number", "주문 번호"]):
                        context_keywords.append("order_number")
                    if any(word in text_lower_for_context for word in ["해결", "완료", "감사", "solution", "resolved"]):
                        if satisfaction_delta > 0.3:
                            context_keywords.append("solution_accepted")
                    if any(word in text_lower_for_context for word in ["거절", "불가", "안 됩니다", "denied", "cannot"]):
                        if emotion == "ANGRY":
                            context_keywords.append("policy_denial")
                    
                    return {
                        "emotion": emotion,
                        "gesture": gesture,
                        "urgency": urgency,
                        "satisfaction_delta": max(-1.0, min(1.0, satisfaction_delta)),
                        "context_keywords": context_keywords,  # ⭐ 추가
                        "confidence": max(0.0, min(1.0, confidence))
                    }
            except json.JSONDecodeError:
                pass
        
        # LLM 호출 실패 시 키워드 기반 간단한 분석
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
        
        # ⭐ Gemini 제안: 상황별 키워드 추출 (키워드 기반 분석)
        context_keywords = []
        if any(word in text_lower for word in ["주문번호", "order number", "주문 번호"]):
            context_keywords.append("order_number")
        if any(word in text_lower for word in ["해결", "완료", "감사", "solution"]):
            if satisfaction_delta > 0.3:
                context_keywords.append("solution_accepted")
        if any(word in text_lower for word in ["거절", "불가", "안 됩니다"]):
            if emotion == "ANGRY":
                context_keywords.append("policy_denial")
        
        return {
            "emotion": emotion,
            "gesture": gesture,
            "urgency": urgency,
            "satisfaction_delta": satisfaction_delta,
            "context_keywords": context_keywords,  # ⭐ 추가
            "confidence": 0.6  # 키워드 기반 분석은 낮은 신뢰도
        }
    
    except Exception as e:
        print(f"텍스트 분석 오류: {e}")
        return {
            "emotion": "NEUTRAL", 
            "gesture": "NONE", 
            "urgency": "LOW",
            "satisfaction_delta": 0.0,
            "context_keywords": [],  # ⭐ 추가
            "confidence": 0.5
        }


def get_video_path_by_avatar(gender: str, emotion: str, is_speaking: bool = False, 
                             gesture: str = "NONE", context_keywords: List[str] = None) -> str:
    """
    고객 아바타 정보(성별, 감정 상태, 제스처, 상황)에 따라 적절한 비디오 경로를 반환합니다.
    OpenAI/Gemini 기반 영상 RAG: LLM이 분석한 감정/제스처에 따라 비디오 클립을 선택합니다.
    
    ⭐ Gemini 제안: 상황별 비디오 클립 패턴 확장 (예: male_asking_order_number.mp4)
    
    Args:
        gender: "male" 또는 "female"
        emotion: "NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD"
        is_speaking: 말하는 중인지 여부
        gesture: "NONE", "HAND_WAVE", "NOD", "SHAKE_HEAD", "POINT"
        context_keywords: 상황별 키워드 리스트 (예: ["order_number", "solution_accepted", "policy_denial"])
    
    Returns:
        비디오 파일 경로 (없으면 None)
    """
    # 비디오 디렉토리 경로 (사용자가 설정한 비디오 파일들이 저장된 위치)
    video_base_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(video_base_dir, exist_ok=True)
    
    # ⭐ Gemini 제안: 우선순위 -1 - 데이터베이스 기반 추천 비디오 (가장 우선)
    if context_keywords:
        db_recommended = get_recommended_video_from_database(emotion, gesture, context_keywords)
        if db_recommended:
            return db_recommended
    else:
        db_recommended = get_recommended_video_from_database(emotion, gesture, [])
        if db_recommended:
            return db_recommended
    
    # ⭐ Gemini 제안: 우선순위 0 - 상황별 비디오 클립 (가장 구체적)
    if context_keywords:
        for keyword in context_keywords:
            # 상황별 파일명 패턴 시도 (예: male_asking_order_number.mp4)
            context_filename = f"{gender}_{emotion.lower()}_{keyword}"
            if is_speaking:
                context_filename += "_speaking"
            context_filename += ".mp4"
            context_path = os.path.join(video_base_dir, context_filename)
            if os.path.exists(context_path):
                return context_path
            
            # 세션 상태에서도 확인
            context_video_key = f"video_{gender}_{emotion.lower()}_{keyword}"
            if context_video_key in st.session_state and st.session_state[context_video_key]:
                video_path = st.session_state[context_video_key]
                if os.path.exists(video_path):
                    return video_path
    
    # 우선순위 1: 제스처가 있는 경우 제스처별 비디오 시도
    if gesture != "NONE" and gesture:
        gesture_video_key = f"video_{gender}_{emotion.lower()}_{gesture.lower()}"
        if gesture_video_key in st.session_state and st.session_state[gesture_video_key]:
            video_path = st.session_state[gesture_video_key]
            if os.path.exists(video_path):
                return video_path
        
        # 제스처별 파일명 패턴 시도
        gesture_filename = f"{gender}_{emotion.lower()}_{gesture.lower()}"
        if is_speaking:
            gesture_filename += "_speaking"
        gesture_filename += ".mp4"
        gesture_path = os.path.join(video_base_dir, gesture_filename)
        if os.path.exists(gesture_path):
            return gesture_path
    
    # 우선순위 2: 감정 상태별 비디오 (제스처 없이)
    video_key = f"video_{gender}_{emotion.lower()}"
    if is_speaking:
        video_key += "_speaking"
    
    # 세션 상태에 저장된 비디오 경로가 있으면 사용
    if video_key in st.session_state and st.session_state[video_key]:
        video_path = st.session_state[video_key]
        if os.path.exists(video_path):
            return video_path
    
    # 기본 비디오 파일명 패턴 시도
    video_filename = f"{gender}_{emotion.lower()}"
    if is_speaking:
        video_filename += "_speaking"
    video_filename += ".mp4"
    
    video_path = os.path.join(video_base_dir, video_filename)
    if os.path.exists(video_path):
        return video_path
    
    # 우선순위 3: 기본 비디오 파일 시도 (중립 상태)
    default_video = os.path.join(video_base_dir, f"{gender}_neutral.mp4")
    if os.path.exists(default_video):
        return default_video
    
    # 우선순위 4: 세션 상태에서 업로드된 비디오 확인
    if "current_customer_video" in st.session_state and st.session_state.current_customer_video:
        return st.session_state.current_customer_video
    
    return None


# ⭐ Gemini 제안: 비디오 매핑 데이터베이스 관리 함수
def load_video_mapping_database() -> Dict[str, Any]:
    """비디오 매핑 데이터베이스를 로드합니다."""
    if os.path.exists(VIDEO_MAPPING_DB_FILE):
        try:
            with open(VIDEO_MAPPING_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"비디오 매핑 데이터베이스 로드 오류: {e}")
            return {"mappings": [], "feedback_history": []}
    return {"mappings": [], "feedback_history": []}


def save_video_mapping_database(db_data: Dict[str, Any]):
    """비디오 매핑 데이터베이스를 저장합니다."""
    try:
        with open(VIDEO_MAPPING_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"비디오 매핑 데이터베이스 저장 오류: {e}")


def add_video_mapping_feedback(
    customer_text: str,
    selected_video_path: str,
    emotion: str,
    gesture: str,
    context_keywords: List[str],
    user_rating: int,  # 1-5 점수
    user_comment: str = ""
) -> None:
    """
    ⭐ Gemini 제안: 사용자 피드백을 비디오 매핑 데이터베이스에 추가합니다.
    
    Args:
        customer_text: 고객의 텍스트
        selected_video_path: 선택된 비디오 경로
        emotion: 분석된 감정
        gesture: 분석된 제스처
        context_keywords: 상황별 키워드
        user_rating: 사용자 평가 (1-5)
        user_comment: 사용자 코멘트 (선택적)
    """
    db_data = load_video_mapping_database()
    
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_text": customer_text[:200],  # 최대 200자
        "selected_video": os.path.basename(selected_video_path) if selected_video_path else None,
        "video_path": selected_video_path,
        "emotion": emotion,
        "gesture": gesture,
        "context_keywords": context_keywords,
        "user_rating": user_rating,
        "user_comment": user_comment[:500] if user_comment else "",  # 최대 500자
        "is_natural_match": user_rating >= 4  # 4점 이상이면 자연스러운 매칭으로 간주
    }
    
    db_data["feedback_history"].append(feedback_entry)
    
    # 매핑 규칙 업데이트 (평가가 높은 경우)
    if user_rating >= 4:
        mapping_key = f"{emotion}_{gesture}_{'_'.join(context_keywords) if context_keywords else 'none'}"
        
        # 기존 매핑 찾기
        existing_mapping = None
        for mapping in db_data["mappings"]:
            if mapping.get("key") == mapping_key:
                existing_mapping = mapping
                break
        
        if existing_mapping:
            # 기존 매핑 업데이트 (평균 점수 계산)
            total_rating = existing_mapping.get("total_rating", 0) + user_rating
            count = existing_mapping.get("count", 0) + 1
            existing_mapping["total_rating"] = total_rating
            existing_mapping["count"] = count
            existing_mapping["avg_rating"] = total_rating / count
            existing_mapping["last_updated"] = datetime.now().isoformat()
        else:
            # 새 매핑 추가
            db_data["mappings"].append({
                "key": mapping_key,
                "emotion": emotion,
                "gesture": gesture,
                "context_keywords": context_keywords,
                "recommended_video": os.path.basename(selected_video_path) if selected_video_path else None,
                "video_path": selected_video_path,
                "total_rating": user_rating,
                "count": 1,
                "avg_rating": float(user_rating),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            })
    
    save_video_mapping_database(db_data)


def get_recommended_video_from_database(
    emotion: str,
    gesture: str,
    context_keywords: List[str]
) -> str:
    """
    ⭐ Gemini 제안: 데이터베이스에서 추천 비디오 경로를 가져옵니다.
    
    Args:
        emotion: 감정 상태
        gesture: 제스처
        context_keywords: 상황별 키워드
    
    Returns:
        추천 비디오 경로 (없으면 None)
    """
    db_data = load_video_mapping_database()
    
    mapping_key = f"{emotion}_{gesture}_{'_'.join(context_keywords) if context_keywords else 'none'}"
    
    # 정확한 매칭 찾기
    for mapping in db_data["mappings"]:
        if mapping.get("key") == mapping_key and mapping.get("avg_rating", 0) >= 4.0:
            video_path = mapping.get("video_path")
            if video_path and os.path.exists(video_path):
                return video_path
    
    # 부분 매칭 시도 (감정과 제스처만)
    partial_key = f"{emotion}_{gesture}_none"
    for mapping in db_data["mappings"]:
        if mapping.get("key") == partial_key and mapping.get("avg_rating", 0) >= 4.0:
            video_path = mapping.get("video_path")
            if video_path and os.path.exists(video_path):
                return video_path
    
    return None


def render_synchronized_video(text: str, audio_bytes: bytes, gender: str, emotion: str, 
                               role: str = "customer", autoplay: bool = True,
                               gesture: str = "NONE", context_keywords: List[str] = None):
    """
    TTS 오디오와 동기화된 비디오를 렌더링합니다.
    
    ⭐ Gemini 제안: 피드백 평가 기능 추가
    
    Args:
        text: 말하는 텍스트 내용
        audio_bytes: TTS로 생성된 오디오 바이트
        gender: 고객 성별 ("male" 또는 "female")
        emotion: 감정 상태 ("NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD")
        role: 역할 ("customer" 또는 "agent")
        autoplay: 자동 재생 여부
        gesture: 제스처 (선택적)
        context_keywords: 상황별 키워드 (선택적)
    """
    if role == "customer":
        is_speaking = True
        if context_keywords is None:
            context_keywords = []
        
        # ⭐ Gemini 제안: 데이터베이스 기반 추천 비디오 우선 사용
        video_path = get_video_path_by_avatar(gender, emotion, is_speaking, gesture, context_keywords)
        
        if video_path and os.path.exists(video_path):
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                
                # 비디오와 오디오를 함께 재생
                # Streamlit의 st.video는 오디오 트랙이 있는 비디오를 지원합니다
                # 여기서는 비디오만 표시하고, 오디오는 별도로 재생합니다
                st.video(video_bytes, format="video/mp4", autoplay=autoplay, loop=False, muted=False)
                
                # 오디오도 함께 재생 (동기화)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
                
                # ⭐ Gemini 제안: 사용자 피드백 평가 UI 추가 (채팅/이메일 탭용)
                if not autoplay:  # 자동 재생이 아닌 경우에만 피드백 UI 표시
                    st.markdown("---")
                    st.markdown("**💬 비디오 매칭 평가**")
                    st.caption("이 비디오가 고객의 텍스트와 감정에 자연스럽게 매칭되었습니까?")
                    
                    feedback_key = f"video_feedback_chat_{st.session_state.get('sim_instance_id', 'default')}_{hash(text) % 10000}"
                    
                    col_rating, col_comment = st.columns([2, 3])
                    with col_rating:
                        rating = st.slider(
                            "평가 점수 (1-5점)",
                            min_value=1,
                            max_value=5,
                            value=3,
                            key=f"{feedback_key}_rating",
                            help="1점: 매우 부자연스러움, 5점: 매우 자연스러움"
                        )
                    
                    with col_comment:
                        comment = st.text_input(
                            "의견 (선택사항)",
                            key=f"{feedback_key}_comment",
                            placeholder="예: 비디오가 텍스트와 잘 맞았습니다"
                        )
                    
                    if st.button("피드백 제출", key=f"{feedback_key}_submit"):
                        # 피드백을 데이터베이스에 저장
                        add_video_mapping_feedback(
                            customer_text=text[:200],
                            selected_video_path=video_path,
                            emotion=emotion,
                            gesture=gesture,
                            context_keywords=context_keywords,
                            user_rating=rating,
                            user_comment=comment
                        )
                        st.success(f"✅ 피드백이 저장되었습니다! (점수: {rating}/5)")
                        st.info("💡 이 피드백은 향후 비디오 선택 정확도를 개선하는 데 사용됩니다.")
                
                return True
            except Exception as e:
                st.warning(f"비디오 재생 오류: {e}")
                # 비디오 재생 실패 시 오디오만 재생
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
                return False
        else:
            # 비디오가 없으면 오디오만 재생
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
            return False
    else:
        # 에이전트는 비디오 없이 오디오만 재생
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
        return False


def generate_virtual_human_video(text: str, audio_bytes: bytes, gender: str, emotion: str, 
                                 provider: str = "hyperclova") -> bytes:
    """
    가상 휴먼 기술을 사용하여 텍스트와 오디오에 맞는 비디오를 생성합니다.
    
    ⚠️ 주의: OpenAI/Gemini API만으로는 입모양 동기화 비디오 생성이 불가능합니다.
    가상 휴먼 비디오 생성은 별도의 가상 휴먼 API (예: Hyperclova)가 필요합니다.
    
    현재는 미리 준비된 비디오 파일을 사용하는 방식을 권장합니다.
    
    Args:
        text: 말하는 텍스트 내용
        audio_bytes: TTS로 생성된 오디오 바이트
        gender: 고객 성별 ("male" 또는 "female")
        emotion: 감정 상태 ("NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD")
        provider: 가상 휴먼 제공자 ("hyperclova", "other")
    
    Returns:
        생성된 비디오 바이트 (없으면 None)
    """
    from llm_client import get_api_key
    # 가상 휴먼 API 키 확인
    if provider == "hyperclova":
        api_key = get_api_key("hyperclova")
        if not api_key:
            return None
        
        # TODO: Hyperclova API 연동 구현 (별도 API 필요)
        # OpenAI/Gemini API만으로는 불가능하므로, 실제 가상 휴먼 API가 필요합니다.
        # 예시 구조:
        # response = requests.post(
        #     "https://api.hyperclova.com/virtual-human/generate",
        #     headers={"Authorization": f"Bearer {api_key}"},
        #     json={
        #         "text": text,
        #         "audio": base64.b64encode(audio_bytes).decode(),
        #         "gender": gender,
        #         "emotion": emotion
        #     }
        # )
        # return response.content
    
    # 다른 제공자도 여기에 추가 가능
    # elif provider == "other":
    #     ...
    
    return None


def get_virtual_human_config() -> Dict[str, Any]:
    """
    가상 휴먼 설정을 반환합니다.
    
    Returns:
        가상 휴먼 설정 딕셔너리
    """
    from llm_client import get_api_key
    return {
        "enabled": st.session_state.get("virtual_human_enabled", False),
        "provider": st.session_state.get("virtual_human_provider", "hyperclova"),
        "api_key": get_api_key("hyperclova") if st.session_state.get("virtual_human_provider", "hyperclova") == "hyperclova" else None
    }









