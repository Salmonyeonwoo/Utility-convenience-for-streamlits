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
        st.success(L.get("customer_positive_solution_reaction", "고객이 솔루션에 만족했습니다."))

        # ⭐ 버튼들을 메시지 말풍선 스타일로 표시 (간소화)
        st.info(L.get("info_use_buttons", "💡 아래 버튼을 사용하여 추가 문의 여부를 확인하거나 상담을 종료하세요."))
        
        col_chat_end, col_email_end = st.columns(2)  # 버튼을 나란히 배치

        # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼
        with col_chat_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L.get("send_closing_confirm_button", "✅ 추가 문의 있나요?"),
                         key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}", use_container_width=True):
                # ⭐ 수정: 에이전트가 감사 인사를 포함한 종료 메시지 전송
                # 언어별 감사 인사 메시지 생성
                agent_name = st.session_state.get("agent_name", "000")
                if current_lang == "ko":
                    closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. {L.get('customer_closing_confirm', '추가 문의사항이 있으시면 언제든지 연락 주세요.')} 즐거운 하루 되세요."
                elif current_lang == "en":
                    closing_msg = f"Thank you for contacting us. This was {agent_name}. {L.get('customer_closing_confirm', 'Please feel free to contact us if you have any additional questions.')} Have a great day!"
                else:  # ja
                    closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。{L.get('customer_closing_confirm', '追加のご質問がございましたら、お気軽にお問い合わせください。')} 良い一日をお過ごしください。"

                # 에이전트 응답으로 로그 기록
                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": closing_msg}
                )

                # ⭐ time.sleep 제거: 불필요한 지연
                st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                # st.rerun()

        # [2] 이메일 - 상담 종료 버튼 (즉시 종료)
        with col_email_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L.get("button_email_end_chat", "📋 설문 조사 전송 및 종료"), 
                        key=f"btn_email_end_chat_{st.session_state.sim_instance_id}", use_container_width=True, type="primary"):
                # AHT 타이머 정지
                st.session_state.start_time = None

                # [수정 1] 다국어 레이블 사용
                end_msg = L.get("prompt_survey", "설문 조사 링크를 전송했습니다.")
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": "(시스템: 이메일 상담 종료) " + end_msg}
                )

                time.sleep(0.1)
                st.session_state.is_chat_ended = True
                st.session_state.sim_stage = "CLOSING"  # 바로 CLOSING으로 전환
                
                # 이력 저장
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.customer_query_text_area, customer_type_display,
                    st.session_state.simulator_messages, is_chat_ended=True,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                # ⭐ 재실행 불필요: 채팅 종료 상태는 이미 반영됨, 다음 렌더링에서 자동 표시됨
                # st.rerun()

    # =========================
    # 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":