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

                try:
                    t = datetime.fromisoformat(h["timestamp"])
                    t_str = t.strftime("%m-%d %H:%M")
                except Exception:
                    t_str = h.get("timestamp", "")

                # 요약 데이터가 있으면 요약 정보 표시, 없으면 초기 문의 표시
                summary = h.get("summary")
                if summary and isinstance(summary, dict):
                    main_inquiry = summary.get("main_inquiry", h["initial_query"][:30])
                    sentiment = summary.get("customer_sentiment_score", 50)
                    satisfaction = summary.get("customer_satisfaction_score", 50)
                    q = main_inquiry[:30].replace("\n", " ")
                    # 첨부 파일 여부 표시 추가
                    attachment_icon = "📎" if h.get("attachment_context") else ""
                    # 요약 데이터 표시 (감정/만족도 점수 포함)
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} | 감정:{sentiment} 만족:{satisfaction} - {q}..."
                else:
                    q = h["initial_query"][:30].replace("\n", " ")
                    attachment_icon = "📎" if h.get("attachment_context") else ""
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} - {q}..."


            options_map = {_label(h): h for h in filtered_for_display}

            # Show a message indicating what is displayed if filters were applied
            if is_searching_or_filtering:
                st.caption(f"🔎 총 {len(filtered_for_display)}개 이력 검색됨 (전화 이력 제외)")
            else:
                st.caption(f"⭐ 최근 {len(filtered_for_display)}개 이력 표시 중 (전화 이력 제외)")

            sel_key = st.selectbox(L["history_selectbox_label"], options=list(options_map.keys()))

            if st.button(L["history_load_button"], key="load_hist_btn"):
                h = options_map[sel_key]
                st.session_state.customer_query_text_area = h["initial_query"]

                # 메시지가 비어있고 요약 데이터가 있는 경우, 요약을 기반으로 최소한의 메시지 재구성
                if not h.get("messages") and h.get("summary"):
                    summary = h["summary"]
                    # 요약 데이터를 기반으로 기본 메시지 구조 생성
                    reconstructed_messages = [
                        {"role": "customer", "content": h["initial_query"]}
                    ]
                    # 요약에서 핵심 응답 추가
                    if summary.get("key_responses"):
                        for response in summary.get("key_responses", [])[:3]:  # 최대 3개만
                            reconstructed_messages.append({"role": "agent_response", "content": response})
                    # 요약 정보를 supervisor 메시지로 추가
                    summary_text = f"**요약된 상담 이력**\n\n"
                    summary_text += f"주요 문의: {summary.get('main_inquiry', 'N/A')}\n"
                    summary_text += f"고객 감정 점수: {summary.get('customer_sentiment_score', 50)}/100\n"
                    summary_text += f"고객 만족도: {summary.get('customer_satisfaction_score', 50)}/100\n"
                    summary_text += f"\n전체 요약:\n{summary.get('summary', 'N/A')}"
                    reconstructed_messages.append({"role": "supervisor", "content": summary_text})
                    st.session_state.simulator_messages = reconstructed_messages

                    # 요약 데이터 시각화
                    st.markdown("---")
                    st.subheader("📊 로드된 케이스 분석")

                    # 요약 데이터를 프로필 형식으로 변환
                    loaded_profile = {
                        "sentiment_score": summary.get("customer_sentiment_score", 50),
                        "urgency_level": "medium",  # 기본값
                        "predicted_customer_type": h.get("customer_type", "normal")
                    }

                    # 프로필 점수 차트
                    profile_chart = visualize_customer_profile_scores(loaded_profile, current_lang)
                    if profile_chart:
                        st.plotly_chart(profile_chart, use_container_width=True)
                    else:
                        # Plotly가 없을 경우 텍스트로 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(L.get("sentiment_score_label", "감정 점수"),
                                      f"{summary.get('customer_sentiment_score', 50)}/100")
                        with col2:
                            st.metric(L.get("urgency_score_label", "긴급도"), f"50/100")
                        with col3:
                            st.metric(L.get("customer_type_label", "고객 유형"), h.get("customer_type", "normal"))

                    # 고객 특성 시각화
                    if summary.get("customer_characteristics") or summary.get("privacy_info"):
                        characteristics_chart = visualize_customer_characteristics(summary, current_lang)
                        if characteristics_chart:
                            st.plotly_chart(characteristics_chart, use_container_width=True)
                else:
                    # 기존 메시지가 있는 경우 그대로 사용
                    st.session_state.simulator_messages = h.get("messages", [])

                st.session_state.initial_advice_provided = True
                st.session_state.is_chat_ended = h.get("is_chat_ended", False)
                st.session_state.sim_attachment_context_for_llm = h.get("attachment_context", "")  # 컨텍스트 로드
                st.session_state.customer_attachment_file = []  # 로드된 이력에는 파일 객체 대신 컨텍스트 문자열만 사용
                st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화

                # 상태 복원
                if st.session_state.is_chat_ended:
                    st.session_state.sim_stage = "CLOSING"
                else:
                    messages = st.session_state.simulator_messages
                    last_role = messages[-1].get("role", "") if messages and len(messages) > 0 else None
                    if last_role == "agent_response":
                        st.session_state.sim_stage = "CUSTOMER_TURN"
                    elif last_role == "customer_rebuttal":
                        st.session_state.sim_stage = "AGENT_TURN"
                    elif last_role == "supervisor" and messages and len(messages) > 0 and messages[-1].get("content", "") == L[
                        "customer_closing_confirm"]:
                        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"

                st.session_state.simulator_memory.clear()  # 메모리 초기화
        else:
            st.info(L["no_history_found"])

    # =========================
    # AHT 타이머 (화면 최상단)
    # =========================
    if st.session_state.sim_stage not in ["WAIT_FIRST_QUERY", "CLOSING", "idle"]:
        elapsed_placeholder = st.empty()

        if st.session_state.start_time is not None:
            # 실시간 업데이트를 위해 페이지 로드 시마다 현재 시간 계산
            elapsed_time = datetime.now() - st.session_state.start_time
            total_seconds = elapsed_time.total_seconds()

            # Hold 시간 제외 (채팅/이메일은 Hold 없음, 전화 탭과 로직 통일 위해 유지)
            # total_seconds -= st.session_state.total_hold_duration.total_seconds()

            # 시간 형식 포맷팅
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 경고 기준
            if total_seconds > 900:  # 15분
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:  # 10분
                delta_str = L["timer_info_warn"]
                delta_color = "off"
            else:
                delta_str = L["timer_info_ok"]
                delta_color = "normal"

            elapsed_placeholder.metric(
                L["timer_metric"],
                time_str,
                delta=delta_str,
                delta_color=delta_color
            )

            # ⭐ 수정: 3초마다 재실행하여 AHT 실시간성 확보
            if seconds % 3 == 0 and total_seconds < 1000:
                time.sleep(1)

        st.markdown("---")

    # =========================
    # 2. LLM 준비 체크 & 채팅 종료 상태
    # =========================
    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.sim_stage == "CLOSING":
        st.success(L["survey_sent_confirm"])
        st.info(L["new_simulation_ready"])
        
        # ⭐ 추가: 현재 세션 이력 다운로드 기능
        st.markdown("---")
        st.markdown("**📥 현재 세션 이력 다운로드**")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        # 현재 세션의 이력을 생성
        current_session_history = None
        if st.session_state.simulator_messages:
            try:
                customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
                current_session_summary = generate_chat_summary(
                    st.session_state.simulator_messages,
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.language
                )
                current_session_history = [{
                    "id": f"session_{st.session_state.sim_instance_id}",
                    "timestamp": datetime.now().isoformat(),
                    "initial_query": st.session_state.customer_query_text_area,
                    "customer_type": customer_type_display,
                    "language_key": st.session_state.language,
                    "messages": st.session_state.simulator_messages,
                    "summary": current_session_summary,
                    "is_chat_ended": True,
                    "attachment_context": st.session_state.sim_attachment_context_for_llm
                }]
            except Exception as e:
                st.warning(L.get("history_generation_error", "이력 생성 중 오류 발생: {error}").format(error=e))
        
        # 다운로드 버튼들을 직접 표시
        if current_session_history:
            # 현재 언어 가져오기
            current_lang = st.session_state.get("language", "ko")
            if current_lang not in ["ko", "en", "ja"]:
                current_lang = "ko"
            
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history, lang=current_lang)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_word", "📥 이력 다운로드 (Word)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_word_file"
                        )
                except Exception as e:
                    st.error(L.get("word_download_error", "Word 다운로드 오류: {error}").format(error=e))
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history, lang=current_lang)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pptx", "📥 이력 다운로드 (PPTX)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_pptx_file"
                        )
                except Exception as e:
                    st.error(L.get("pptx_download_error", "PPTX 다운로드 오류: {error}").format(error=e))
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history, lang=current_lang)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pdf", "📥 이력 다운로드 (PDF)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pdf),
                            mime="application/pdf",
                            key="download_pdf_file"
                        )
                except Exception as e:
                    st.error(L.get("pdf_download_error", "PDF 다운로드 오류: {error}").format(error=e))
        else:
            st.warning(L.get("no_history_to_download", "다운로드할 이력이 없습니다."))
        
        st.markdown("---")
        
        if st.button(L["new_simulation_button"], key="new_simulation_btn"):
            # 초기화 로직
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.initial_advice_provided = False
            st.session_state.is_chat_ended = False
            st.session_state.agent_response_area_text = ""
            st.session_state.customer_query_text_area = ""
            st.session_state.last_transcript = ""
            st.session_state.sim_audio_bytes = None
            st.session_state.sim_stage = "WAIT_FIRST_QUERY"
            st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
            st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
            st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
            st.session_state.start_time = None
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None
        # st.stop()

    # =========================