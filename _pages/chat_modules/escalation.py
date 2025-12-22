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
        
        st.warning(L.get("escalation_required_msg", "🚨 고객이 에스컬레이션을 요청했습니다. 상급자나 전문 팀으로 이관이 필요합니다."))
        
        # 에스컬레이션 처리 옵션
        col_escalate, col_continue = st.columns(2)
        
        with col_escalate:
            if st.button(L.get("button_escalate", "에스컬레이션 처리"), key=f"btn_escalate_{st.session_state.sim_instance_id}"):
                # 에스컬레이션 시스템 메시지 추가
                escalation_msg = L.get("escalation_system_msg", "📌 시스템 메시지: 고객 요청에 따라 상급자/전문 팀으로 이관되었습니다.")
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": escalation_msg}
                )
                
                # 이력 저장
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=True,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                
                # 종료 단계로 이동
                st.session_state.sim_stage = "CLOSING"
        
        with col_continue:
            if st.button(L.get("button_continue", "계속 응대"), key=f"btn_continue_{st.session_state.sim_instance_id}"):
                # 계속 응대하는 경우 AGENT_TURN으로 이동
                st.session_state.sim_stage = "AGENT_TURN"
    
    # =========================
    # 6. 고객 반응 생성 단계 (CUSTOMER_TURN)
    # =========================
    elif st.session_state.sim_stage == "CUSTOMER_TURN":