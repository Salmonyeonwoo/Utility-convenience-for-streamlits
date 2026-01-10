# ========================================
# video_handler_virtual.py
# 비디오 처리 - 가상 휴먼 모듈
# ========================================

import streamlit as st
from typing import Dict, Any, Optional
from llm_client import get_api_key


def generate_virtual_human_video(text: str, audio_bytes: bytes, gender: str, emotion: str, 
                                 provider: str = "hyperclova") -> Optional[bytes]:
    """
    가상 휴먼 기술을 사용하여 텍스트와 오디오에 맞는 비디오를 생성합니다.
    
    ⚠️ 주의: OpenAI/Gemini API만으로는 입모양 동기화 비디오 생성이 불가능합니다.
    가상 휴먼 비디오 생성은 별도의 가상 휴먼 API (예: Hyperclova)가 필요합니다.
    
    Args:
        text: 말하는 텍스트 내용
        audio_bytes: TTS로 생성된 오디오 바이트
        gender: 고객 성별 ("male" 또는 "female")
        emotion: 감정 상태
        provider: 가상 휴먼 제공자 ("hyperclova", "other")
    
    Returns:
        생성된 비디오 바이트 (없으면 None)
    """
    if provider == "hyperclova":
        api_key = get_api_key("hyperclova")
        if not api_key:
            return None
        
        # TODO: Hyperclova API 연동 구현 (별도 API 필요)
        # OpenAI/Gemini API만으로는 불가능하므로, 실제 가상 휴먼 API가 필요합니다.
    
    return None


def get_virtual_human_config() -> Dict[str, Any]:
    """가상 휴먼 설정을 반환합니다."""
    return {
        "enabled": st.session_state.get("virtual_human_enabled", False),
        "provider": st.session_state.get("virtual_human_provider", "hyperclova"),
        "api_key": get_api_key("hyperclova") if st.session_state.get("virtual_human_provider", "hyperclova") == "hyperclova" else None
    }
