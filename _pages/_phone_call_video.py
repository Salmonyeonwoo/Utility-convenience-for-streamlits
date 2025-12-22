# ========================================
# pages/phone_call_video.py
# 비디오 통화 처리 모듈
# ========================================

import io
import time
import streamlit as st
import numpy as np
from PIL import Image
from lang_pack import LANG


def render_video_call_ui():
    """비디오 통화 UI 렌더링"""
    current_lang = st.session_state.language
    L = LANG[current_lang]

    video_col1, video_col2 = st.columns(2)
    
    with video_col1:
        st.subheader(L.get("my_screen", "📹 내 화면"))
        camera_image = st.camera_input(
            L.get("webcam_label", "웹캠"),
            key="my_camera",
            help=L.get("webcam_help", "내 웹캠 영상")
        )
        if camera_image:
            st.image(camera_image, use_container_width=True)
            st.session_state.last_camera_frame = camera_image
            if len(st.session_state.opponent_video_frames) >= 3:
                st.session_state.opponent_video_frames.pop(0)
            st.session_state.opponent_video_frames.append({
                'image': camera_image,
                'timestamp': time.time()
            })
    
    with video_col2:
        st.subheader("📹 상대방 화면")
        if st.session_state.opponent_video_frames:
            display_frame_idx = max(0, len(st.session_state.opponent_video_frames) - 2)
            if display_frame_idx < len(st.session_state.opponent_video_frames):
                opponent_frame = st.session_state.opponent_video_frames[display_frame_idx]['image']
                
                try:
                    img = Image.open(io.BytesIO(opponent_frame.getvalue()))
                    mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    img_array = np.array(mirrored_img)
                    img_array = (img_array * 0.9).astype(np.uint8)
                    processed_img = Image.fromarray(img_array)
                    st.image(processed_img, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
                except Exception as e:
                    st.image(opponent_frame, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
            else:
                st.info("상대방 비디오를 준비하는 중...")
        elif st.session_state.last_camera_frame:
            try:
                img = Image.open(io.BytesIO(st.session_state.last_camera_frame.getvalue()))
                mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_array = np.array(mirrored_img)
                img_array = (img_array * 0.9).astype(np.uint8)
                processed_img = Image.fromarray(img_array)
                st.image(processed_img, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
            except:
                st.image(st.session_state.last_camera_frame, use_container_width=True, caption="상대방 화면 (시뮬레이션)")
        else:
            st.info("상대방 비디오 스트림을 기다리는 중...")
            st.caption("💡 팁: 내 화면이 상대방 화면으로 시뮬레이션됩니다")
        
        if st.session_state.opponent_video_frames:
            st.caption("📹 가상 상대방 비디오 (API 키 불필요)")
        
        st.markdown("---")
        with st.expander(L["video_upload_expander"], expanded=False):
            st.session_state.is_video_sync_enabled = st.checkbox(
                L["video_sync_enable"],
                value=st.session_state.is_video_sync_enabled,
                key="video_sync_checkbox_call"
            )
            
            st.markdown("---")
            st.markdown(f"**{L['video_rag_title']}**")
            st.success(L["video_rag_desc"])
            
            st.session_state.virtual_human_enabled = False
            
            st.markdown(f"**{L['video_gender_emotion_setting']}**")
            col_gender_video, col_emotion_video = st.columns(2)
            
            with col_gender_video:
                video_gender = st.radio(L["video_gender_label"], [L["video_gender_male"], L["video_gender_female"]], key="video_gender_select_call", horizontal=True)
                gender_key = "male" if video_gender == L["video_gender_male"] else "female"
            
            with col_emotion_video:
                video_emotion = st.selectbox(
                    L["video_emotion_label"],
                    ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"],
                    key="video_emotion_select_call"
                )
                emotion_key = video_emotion.lower()
            
            video_key = f"video_{gender_key}_{emotion_key}"
            uploaded_video = st.file_uploader(
                L["video_upload_label"].format(gender=video_gender, emotion=video_emotion),
                type=["mp4", "webm", "ogg"],
                key=f"customer_video_uploader_call_{gender_key}_{emotion_key}"
            )
            
            if uploaded_video is not None:
                try:
                    video_bytes = uploaded_video.read()
                    if video_bytes and len(video_bytes) > 0:
                        upload_key = f"last_uploaded_video_{gender_key}_{emotion_key}"
                        video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"
                        st.session_state[video_bytes_key] = video_bytes
                        st.session_state[video_key] = video_bytes_key
                        st.success(f"✅ 비디오 업로드 완료: {uploaded_video.name}")
                except Exception as e:
                    st.error(f"비디오 업로드 오류: {e}")

