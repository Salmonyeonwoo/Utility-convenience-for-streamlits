# 이 모듈은 _chat_simulator.py에서 분리된 부분입니다
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
from simulation_handler import *
from visualization import *
from audio_handler import *
from llm_client import get_api_key
from typing import List, Dict, Any
import uuid
import time
import os

        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        # ⭐ 수정: 명확한 안내 메시지와 함께 버튼 표시
        st.markdown("---")
        st.success(L["no_more_inquiries_confirmed"])
        st.markdown(f"### {L['consultation_end_header']}")
        st.info(L["click_survey_button_to_end"])
        st.markdown("---")
        
        # 버튼을 중앙에 크게 표시
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            end_chat_button = st.button(
                L["sim_end_chat_button"], 
                key="btn_final_end_chat", 
                use_container_width=True, 
                type="primary"
            )
        
        if end_chat_button:
            # AHT 타이머 정지
            st.session_state.start_time = None

            # 설문 조사 링크 전송 메시지 추가
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

            # 채팅 종료 처리
            st.session_state.is_chat_ended = True
