# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 통화 종료 모듈
통화 종료 후 이력 다운로드 기능 제공
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime
import os
import json
import csv
import io
from utils.history_handler import (
    save_simulation_history_local,
    generate_chat_summary,
    export_history_to_word,
    export_history_to_pptx,
    export_history_to_pdf
)


def render_call_ended():
    """통화 종료 화면 렌더링 및 이력 다운로드 기능"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 통화 시간 표시
    call_duration = st.session_state.get("call_duration", 0)
    minutes = int(call_duration // 60)
    seconds = int(call_duration % 60)
    if minutes > 0:
        duration_msg = f"통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)"
    else:
        duration_msg = f"통화가 종료되었습니다. (통화 시간: {seconds}초)"
    st.success(duration_msg)
    
    # 통화 이력 저장 (자동)
    if st.session_state.get("call_messages"):
        try:
            inquiry_text = st.session_state.get("inquiry_text", "")
            customer_type = st.session_state.get("customer_type_sim_select", "일반 고객")
            if not customer_type:
                customer_type = "일반 고객"
            
            # 통화 이력을 채팅 형식으로 변환
            call_messages = st.session_state.get("call_messages", [])
            converted_messages = []
            for msg in call_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "agent":
                    converted_messages.append({"role": "agent_response", "content": content})
                elif role == "customer":
                    converted_messages.append({"role": "customer", "content": content})
                elif role in ["system_transfer", "supervisor"]:
                    converted_messages.append({"role": "supervisor", "content": content})
            
            # 이력 저장
            save_simulation_history_local(
                initial_query=inquiry_text or "전화 통화",
                customer_type=customer_type,
                messages=converted_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.get("call_content", ""),
                is_call=True
            )
        except Exception as e:
            st.warning(f"통화 이력 저장 중 오류 발생: {e}")
    
    # 이력 다운로드 섹션 (채팅 탭과 동일한 기능)
    st.markdown("---")
    st.markdown("**📥 현재 통화 이력 다운로드**")
    download_col1, download_col2, download_col3, download_col4, download_col5 = st.columns(5)
    
    current_session_history = None
    if st.session_state.get("call_messages"):
        try:
            inquiry_text = st.session_state.get("inquiry_text", "")
            customer_type = st.session_state.get("customer_type_sim_select", "일반 고객")
            if not customer_type:
                customer_type = "일반 고객"
            
            # 통화 메시지를 채팅 형식으로 변환
            call_messages = st.session_state.get("call_messages", [])
            converted_messages = []
            for msg in call_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "agent":
                    converted_messages.append({"role": "agent_response", "content": content})
                elif role == "customer":
                    converted_messages.append({"role": "customer", "content": content})
                elif role in ["system_transfer", "supervisor"]:
                    converted_messages.append({"role": "supervisor", "content": content})
            
            # 요약 생성
            current_session_summary = generate_chat_summary(
                converted_messages,
                inquiry_text or "전화 통화",
                customer_type,
                st.session_state.language
            )
            
            current_session_history = [{
                "id": f"call_{st.session_state.get('current_call_id', 'unknown')}",
                "timestamp": datetime.now().isoformat(),
                "initial_query": inquiry_text or "전화 통화",
                "customer_type": customer_type,
                "language_key": st.session_state.language,
                "messages": converted_messages,
                "summary": current_session_summary,
                "is_chat_ended": True,
                "attachment_context": st.session_state.get("call_content", ""),
                "is_call": True
            }]
        except Exception as e:
            st.warning(
                L.get(
                    "history_generation_error",
                    "이력 생성 중 오류 발생: {error}").format(
                    error=e))
    
    if current_session_history:
        # Word 다운로드
        with download_col1:
            try:
                filepath_word = export_history_to_word(
                    current_session_history, lang=current_lang)
                with open(filepath_word, "rb") as f:
                    st.download_button(
                        label=L.get(
                            "download_history_word",
                            "📥 이력 다운로드 (Word)"),
                        data=f.read(),
                        file_name=os.path.basename(filepath_word),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_call_word_file")
            except Exception as e:
                st.error(
                    L.get(
                        "word_download_error",
                        "Word 다운로드 오류: {error}").format(
                        error=e))
        
        # PPTX 다운로드
        with download_col2:
            try:
                filepath_pptx = export_history_to_pptx(
                    current_session_history, lang=current_lang)
                with open(filepath_pptx, "rb") as f:
                    st.download_button(
                        label=L.get(
                            "download_history_pptx",
                            "📥 이력 다운로드 (PPTX)"),
                        data=f.read(),
                        file_name=os.path.basename(filepath_pptx),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key="download_call_pptx_file")
            except Exception as e:
                st.error(
                    L.get(
                        "pptx_download_error",
                        "PPTX 다운로드 오류: {error}").format(
                        error=e))
        
        # PDF 다운로드
        with download_col3:
            try:
                filepath_pdf = export_history_to_pdf(
                    current_session_history, lang=current_lang)
                with open(filepath_pdf, "rb") as f:
                    st.download_button(
                        label=L.get(
                            "download_history_pdf",
                            "📥 이력 다운로드 (PDF)"),
                        data=f.read(),
                        file_name=os.path.basename(filepath_pdf),
                        mime="application/pdf",
                        key="download_call_pdf_file")
            except Exception as e:
                st.error(
                    L.get(
                        "pdf_download_error",
                        "PDF 다운로드 오류: {error}").format(
                        error=e))
        
        # JSON 다운로드
        with download_col4:
            try:
                json_data = json.dumps(
                    current_session_history, ensure_ascii=False, indent=2)
                call_id = st.session_state.get("current_call_id", "unknown")
                st.download_button(
                    label=L.get(
                        "download_history_json",
                        "📥 이력 다운로드 (JSON)"),
                    data=json_data.encode('utf-8'),
                    file_name=f"call_history_{call_id}.json",
                    mime="application/json",
                    key="download_call_json_file")
            except Exception as e:
                st.error(
                    L.get(
                        "json_download_error",
                        "JSON 다운로드 오류: {error}").format(
                        error=e))
        
        # CSV 다운로드
        with download_col5:
            try:
                output = io.StringIO()
                writer = csv.writer(output)
                
                writer.writerow(["Role", "Content", "Timestamp"])
                
                for msg in current_session_history[0].get("messages", []):
                    writer.writerow([
                        msg.get("role", ""),
                        msg.get("content", ""),
                        current_session_history[0].get("timestamp", "")
                    ])
                
                csv_data = output.getvalue()
                call_id = st.session_state.get("current_call_id", "unknown")
                st.download_button(
                    label=L.get("download_history_csv", "📥 이력 다운로드 (CSV)"),
                    data=csv_data.encode('utf-8-sig'),
                    file_name=f"call_history_{call_id}.csv",
                    mime="text/csv",
                    key="download_call_csv_file"
                )
            except Exception as e:
                st.error(
                    L.get(
                        "csv_download_error",
                        "CSV 다운로드 오류: {error}").format(
                        error=e))
    else:
        st.warning(L.get("no_history_to_download", "다운로드할 이력이 없습니다."))
    
    st.markdown("---")
    
    # 새 통화 시작 버튼
    if st.button(L.get("new_call_button", "새 통화 시작"), key="btn_new_call"):
        # 모든 통화 관련 상태 완전 초기화
        st.session_state.call_sim_stage = "WAITING_CALL"
        st.session_state.call_messages = []
        st.session_state.inquiry_text = ""
        st.session_state.call_content = ""
        st.session_state.incoming_phone_number = None
        st.session_state.incoming_call = None
        st.session_state.call_active = False
        st.session_state.start_time = None
        st.session_state.call_duration = None
        st.session_state.transfer_summary_text = ""
        st.session_state.language_at_transfer_start = None
        st.session_state.current_call_id = None
        st.success("✅ 새 통화를 시작할 수 있습니다.")

