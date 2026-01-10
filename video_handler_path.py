# ========================================
# video_handler_path.py
# 비디오 처리 - 경로 관리 모듈
# ========================================

import os
import streamlit as st
from typing import List
from config import DATA_DIR
from video_handler_database import get_recommended_video_from_database


def get_video_path_by_avatar(gender: str, emotion: str, is_speaking: bool = False, 
                             gesture: str = "NONE", context_keywords: List[str] = None) -> str:
    """
    고객 아바타 정보에 따라 적절한 비디오 경로를 반환합니다.
    
    Args:
        gender: "male" 또는 "female"
        emotion: "NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD"
        is_speaking: 말하는 중인지 여부
        gesture: "NONE", "HAND_WAVE", "NOD", "SHAKE_HEAD", "POINT"
        context_keywords: 상황별 키워드 리스트
    
    Returns:
        비디오 파일 경로 (없으면 None)
    """
    video_base_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(video_base_dir, exist_ok=True)
    
    # 우선순위 0: 데이터베이스 기반 추천 비디오
    db_recommended = get_recommended_video_from_database(emotion, gesture, context_keywords or [])
    if db_recommended:
        return db_recommended
    
    # 우선순위 1: 상황별 비디오 클립
    if context_keywords:
        for keyword in context_keywords:
            context_filename = f"{gender}_{emotion.lower()}_{keyword}"
            if is_speaking:
                context_filename += "_speaking"
            context_filename += ".mp4"
            context_path = os.path.join(video_base_dir, context_filename)
            if os.path.exists(context_path):
                return context_path
            
            context_video_key = f"video_{gender}_{emotion.lower()}_{keyword}"
            if context_video_key in st.session_state and st.session_state[context_video_key]:
                video_path = st.session_state[context_video_key]
                if os.path.exists(video_path):
                    return video_path
    
    # 우선순위 2: 제스처별 비디오
    if gesture != "NONE" and gesture:
        gesture_video_key = f"video_{gender}_{emotion.lower()}_{gesture.lower()}"
        if gesture_video_key in st.session_state and st.session_state[gesture_video_key]:
            video_path = st.session_state[gesture_video_key]
            if os.path.exists(video_path):
                return video_path
        
        gesture_filename = f"{gender}_{emotion.lower()}_{gesture.lower()}"
        if is_speaking:
            gesture_filename += "_speaking"
        gesture_filename += ".mp4"
        gesture_path = os.path.join(video_base_dir, gesture_filename)
        if os.path.exists(gesture_path):
            return gesture_path
    
    # 우선순위 3: 감정 상태별 비디오
    video_key = f"video_{gender}_{emotion.lower()}"
    if is_speaking:
        video_key += "_speaking"
    
    if video_key in st.session_state and st.session_state[video_key]:
        video_path = st.session_state[video_key]
        if os.path.exists(video_path):
            return video_path
    
    video_filename = f"{gender}_{emotion.lower()}"
    if is_speaking:
        video_filename += "_speaking"
    video_filename += ".mp4"
    
    video_path = os.path.join(video_base_dir, video_filename)
    if os.path.exists(video_path):
        return video_path
    
    # 우선순위 4: 기본 비디오 파일
    default_video = os.path.join(video_base_dir, f"{gender}_neutral.mp4")
    if os.path.exists(default_video):
        return default_video
    
    # 우선순위 5: 세션 상태에서 업로드된 비디오
    if "current_customer_video" in st.session_state and st.session_state.current_customer_video:
        return st.session_state.current_customer_video
    
    return None
