# ========================================
# video_handler_rendering.py
# 비디오 처리 - 렌더링 모듈
# ========================================

import os
import streamlit as st
from typing import List, Optional
from video_handler_path import get_video_path_by_avatar
from video_handler_database import add_video_mapping_feedback


def render_synchronized_video(text: str, audio_bytes: bytes, gender: str, emotion: str, 
                               role: str = "customer", autoplay: bool = True,
                               gesture: str = "NONE", context_keywords: List[str] = None):
    """
    TTS 오디오와 동기화된 비디오를 렌더링합니다.
    
    Args:
        text: 말하는 텍스트 내용
        audio_bytes: TTS로 생성된 오디오 바이트
        gender: 고객 성별 ("male" 또는 "female")
        emotion: 감정 상태
        role: 역할 ("customer" 또는 "agent")
        autoplay: 자동 재생 여부
        gesture: 제스처
        context_keywords: 상황별 키워드
    """
    if role == "customer":
        is_speaking = True
        if context_keywords is None:
            context_keywords = []
        
        video_path = get_video_path_by_avatar(gender, emotion, is_speaking, gesture, context_keywords)
        
        if video_path and os.path.exists(video_path):
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                
                st.video(video_bytes, format="video/mp4", autoplay=autoplay, loop=False, muted=False)
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
                
                # 피드백 평가 UI
                if not autoplay:
                    _render_feedback_ui(text, video_path, emotion, gesture, context_keywords)
                
                return True
            except Exception as e:
                st.warning(f"비디오 재생 오류: {e}")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
                return False
        else:
            # 비디오가 없으면 Lottie 애니메이션 fallback
            _render_fallback_animation(text)
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
            return False
    else:
        # 에이전트는 비디오 없이 오디오만 재생
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
        return False


def _render_feedback_ui(text: str, video_path: str, emotion: str, gesture: str, context_keywords: List[str]):
    """피드백 평가 UI 렌더링"""
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


def _render_fallback_animation(text: str):
    """Lottie 애니메이션 fallback 렌더링"""
    try:
        from streamlit_lottie import st_lottie
        import requests
        import json
        
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json"
        
        try:
            lottie_json = requests.get(lottie_url, timeout=2).json()
        except:
            lottie_file_path = os.path.join(os.path.dirname(__file__), "assets", "speaking_character.json")
            if os.path.exists(lottie_file_path):
                with open(lottie_file_path, "r", encoding="utf-8") as f:
                    lottie_json = json.load(f)
            else:
                raise Exception("Lottie 파일을 찾을 수 없습니다")
        
        st_lottie(lottie_json, height=300, key=f"lottie_fallback_{hash(text) % 10000}")
        st.caption("💬 캐릭터가 말하고 있습니다...")
    except ImportError:
        st.info("🎤 캐릭터가 말하고 있습니다...")
        speaking_indicator = "●" * (len(text) % 10 + 1)
        st.markdown(f"<div style='text-align: center; font-size: 24px;'>{speaking_indicator}</div>", unsafe_allow_html=True)
    except Exception:
        st.info("🎤 캐릭터가 말하고 있습니다...")
