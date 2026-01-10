# ========================================
# _pages/_call_video.py
# 전화 통화 비디오 표시 모듈
# ========================================

import streamlit as st
import time
from PIL import Image
import io
import numpy as np
from lang_pack import LANG


def render_video_section(L):
    """비디오 섹션 렌더링"""
    if not st.session_state.get("video_enabled", False):
        return
    
    video_col1, video_col2 = st.columns(2)
    
    with video_col1:
        _render_my_video(L)
    
    with video_col2:
        _render_opponent_video(L)
    
    st.markdown("---")


def _render_my_video(L):
    """내 비디오 렌더링"""
    st.markdown(f"**{L.get('my_screen', '📹 내 화면')}**")
    camera_image = st.camera_input(
        L.get("webcam_label", "웹캠"),
        key="my_camera_call",
        help=L.get("webcam_help", "내 웹캠 영상"),
    )
    
    if camera_image:
        st.image(camera_image, use_container_width=True)
        if 'opponent_video_frames' not in st.session_state:
            st.session_state.opponent_video_frames = []
        if 'last_camera_frame' not in st.session_state:
            st.session_state.last_camera_frame = None
        
        st.session_state.last_camera_frame = camera_image
        
        if len(st.session_state.opponent_video_frames) >= 3:
            st.session_state.opponent_video_frames.pop(0)
        
        st.session_state.opponent_video_frames.append({
            'image': camera_image,
            'timestamp': time.time()
        })


def _render_opponent_video(L):
    """상대방 비디오 렌더링"""
    st.markdown(f"**{L.get('opponent_screen', '📹 상대방 화면')}**")
    
    if st.session_state.get("opponent_video_frames"):
        display_frame_idx = max(0, len(st.session_state.opponent_video_frames) - 2)
        if display_frame_idx < len(st.session_state.opponent_video_frames):
            opponent_frame = st.session_state.opponent_video_frames[display_frame_idx]['image']
            try:
                img = Image.open(io.BytesIO(opponent_frame.getvalue()))
                mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_array = np.array(mirrored_img)
                img_array = (img_array * 0.9).astype(np.uint8)
                processed_img = Image.fromarray(img_array)
                st.image(processed_img, use_container_width=True, 
                        caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
            except Exception:
                st.image(opponent_frame, use_container_width=True,
                        caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
        else:
            st.info(L.get("opponent_video_preparing", "상대방 비디오를 준비하는 중..."))
    elif st.session_state.get("last_camera_frame"):
        try:
            img = Image.open(io.BytesIO(st.session_state.last_camera_frame.getvalue()))
            mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_array = np.array(mirrored_img)
            img_array = (img_array * 0.9).astype(np.uint8)
            processed_img = Image.fromarray(img_array)
            st.image(processed_img, use_container_width=True,
                    caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
        except:
            st.image(st.session_state.last_camera_frame, use_container_width=True,
                    caption=L.get("opponent_screen_simulation", "상대방 화면 (시뮬레이션)"))
    else:
        st.info(L.get("opponent_video_waiting", "상대방 비디오 스트림을 기다리는 중..."))
