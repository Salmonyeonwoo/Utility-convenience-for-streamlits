# 전화 시뮬레이터
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
# ⭐ 수정: 사용하지 않는 import를 try-except로 감싸서 import 실패해도 계속 진행
try:
    from simulation_handler import *
except ImportError:
    pass
try:
    from visualization import *
except ImportError:
    pass
try:
    from audio_handler import *
except ImportError:
    pass
try:
    from llm_client import get_api_key
except ImportError:
    pass
from typing import List, Dict, Any
import uuid
import time
import os
from PIL import Image
import io

def render_call_simulator():
    """전화 시뮬레이터 렌더링 (전화 수신, 문의 입력 포함)"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # AHT 타이머는 streamlit_app.py의 우측 상단에서 표시됨 (제거됨)

    # ========================================
    # 화면 구분 (애니메이션 / CC)
    # ========================================
    # ⭐ 수정: 왼쪽 비디오 업로드 섹션 제거, col_cc만 사용
    col_cc = st.columns([1])[0]

    with col_cc:
        # ⭐ 수정: "전화 수신 중" 메시지를 더 깔끔한 위치로 이동
        if st.session_state.call_sim_stage == "IN_CALL":
            if st.session_state.call_sim_mode == "INBOUND":
                st.markdown(
                    f"## {L['call_status_ringing'].format(number=st.session_state.incoming_phone_number)}"
                )
            else:
                st.markdown(
                    f"## {L['button_call_outbound']} ({st.session_state.incoming_phone_number})"
                )
        st.markdown("---")

    # ⭐ 왼쪽 비디오 섹션 제거 (비디오 업로드 내용은 상대방 화면 밑으로 이동)

    # ⭐ col_cc는 위에서 이미 처리됨

    # ========================================
    # WAITING / RINGING 상태 - 전화 수신, 문의 입력 포함
    # ========================================
    try:
        # IDLE 상태도 WAITING_CALL로 처리 (초기 상태)
        if st.session_state.call_sim_stage in ["WAITING_CALL", "RINGING", "IDLE", None]:
            # IDLE이나 None이면 WAITING_CALL로 초기화
            if st.session_state.call_sim_stage in ["IDLE", None]:
                st.session_state.call_sim_stage = "WAITING_CALL"
            
            # _call_waiting 모듈 사용 (중복 제거)
            try:
                from _pages._call_waiting import render_call_waiting
                render_call_waiting()
            except Exception as e:
                st.error(f"❌ _call_waiting 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())

        # ------------------
        # IN_CALL 상태 (통화 중)
        # ------------------
        elif st.session_state.call_sim_stage == "IN_CALL":
            try:
                from _pages._call_in_call import render_call_in_call
                render_call_in_call()
            except Exception as e:
                st.error(f"❌ _call_in_call 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.info("📞 통화 중입니다...")
                if st.button("📴 통화 종료", use_container_width=True, type="primary"):
                    st.session_state.call_sim_stage = "CALL_ENDED"
                    st.session_state.call_active = False
                    st.session_state.start_time = None
        
        elif st.session_state.call_sim_stage == "CALL_ENDED":
            try:
                from _pages._call_ended import render_call_ended
                render_call_ended()
            except ImportError:
                # call_ended 모듈이 없으면 기본 종료 화면 표시
                st.success(L.get("call_ended_message", "통화가 종료되었습니다."))
                if st.button(L.get("new_call_button", "새 통화 시작"), key="btn_new_call"):
                    st.session_state.call_sim_stage = "WAITING_CALL"
            except Exception as e:
                st.error(f"❌ _call_ended 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.success("통화가 종료되었습니다.")
                if st.button("새 통화 시작", key="btn_new_call_fallback"):
                    st.session_state.call_sim_stage = "WAITING_CALL"
        else:
            # 알 수 없는 상태일 때 WAITING_CALL로 초기화하고 전화 수신 화면 표시
            st.session_state.call_sim_stage = "WAITING_CALL"
            try:
                from _pages._call_waiting import render_call_waiting
                render_call_waiting()
            except Exception as e:
                st.error(f"❌ _call_waiting 로드 오류: {e}")
                import traceback
                st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"❌ 전화 시뮬레이터 렌더링 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        # 기본 폴백: _call_waiting 모듈 사용
        try:
            from _pages._call_waiting import render_call_waiting
            render_call_waiting()
        except:
            st.info("전화 시뮬레이터를 초기화할 수 없습니다.")

