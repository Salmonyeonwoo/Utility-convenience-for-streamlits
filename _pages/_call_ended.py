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
        duration_msg = L.get("call_ended_with_duration", "통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)").format(minutes=minutes, seconds=seconds)
    else:
        duration_msg = L.get("call_ended_with_seconds", "통화가 종료되었습니다. (통화 시간: {seconds}초)").format(seconds=seconds)
    st.success(duration_msg)
    
    # ⭐ 고객 정보 입력 폼 (통화 종료 후, 이력 다운로드 전)
    if "call_customer_info_saved" not in st.session_state:
        st.session_state.call_customer_info_saved = False
    
    if not st.session_state.call_customer_info_saved:
        st.markdown("---")
        st.subheader(L.get("customer_info_input_header", "📝 고객 정보 입력"))
        st.info(L.get("customer_info_input_note", "통화 이력을 저장하기 위해 고객 정보를 입력해주세요."))
        
        with st.form("call_customer_info_form"):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                call_customer_name = st.text_input(
                    L.get("customer_name_label", "고객명") + " *",
                    key="call_ended_customer_name",
                    value=st.session_state.get("call_customer_name", "")
                )
                call_customer_phone = st.text_input(
                    L.get("phone_label", "연락처") + " *",
                    key="call_ended_customer_phone",
                    value=st.session_state.get("call_customer_phone", "")
                )
            with col_info2:
                call_customer_email = st.text_input(
                    L.get("email_label", "이메일") + " *",
                    key="call_ended_customer_email",
                    value=st.session_state.get("call_customer_email", "")
                )
                # ⭐ 언어별 고객 성향 옵션 사용
                personality_options = L.get("personality_options", ["일반", "신중형", "활발형", "가족형", "프리미엄형", "절약형", "자유형"])
                call_customer_personality = st.selectbox(
                    L.get("personality_label", "고객 성향"),
                    personality_options,
                    key="call_ended_customer_personality",
                    index=0
                )
            
            # ⭐ 고객과의 응대 내용 요약 메모 칸 추가
            call_summary_memo = st.text_area(
                L.get("call_summary_memo_label", "고객과의 응대 내용 요약"),
                key="call_ended_summary_memo",
                value=st.session_state.get("call_summary_memo", ""),
                height=100,
                placeholder=L.get("call_summary_memo_placeholder", "고객과의 응대 내용을 요약하여 입력하세요...")
            )
            
            if st.form_submit_button(L.get("button_save_customer_info", "고객 정보 저장 및 이력 저장"), type="primary", use_container_width=True):
                if call_customer_name and call_customer_phone and call_customer_email:
                    # 고객 정보를 session state에 저장
                    st.session_state.call_customer_name = call_customer_name
                    st.session_state.call_customer_phone = call_customer_phone
                    st.session_state.call_customer_email = call_customer_email
                    st.session_state.call_customer_personality = call_customer_personality
                    st.session_state.call_summary_memo = call_summary_memo
                    st.session_state.call_customer_info_saved = True
                    
                    # 고객 정보를 CustomerDataManager에 저장 (선택사항)
                    try:
                        from customer_data_manager import CustomerDataManager
                        manager = CustomerDataManager()
                        customer_data = {
                            'customer_name': call_customer_name,
                            'phone': call_customer_phone,
                            'email': call_customer_email,
                            'personality': call_customer_personality,
                            'inquiry_summary': call_summary_memo  # ⭐ 추가: 문의 내용 요약 저장
                        }
                        customer_id = manager.create_customer(customer_data)
                        st.session_state.call_customer_id = customer_id
                        st.success(L.get("customer_info_saved_success", "고객 정보가 저장되었습니다. (고객 ID: {id})").format(id=customer_id))
                    except Exception as e:
                        st.warning(L.get("customer_info_save_warning", "고객 정보 저장 중 오류 발생 (이력은 저장됩니다): {error}").format(error=str(e)))
                else:
                    st.error(L.get("error_mandatory_fields", "고객명, 연락처, 이메일은 필수 항목입니다."))
    else:
        # 고객 정보가 이미 저장된 경우 표시
        st.markdown("---")
        st.success(L.get("customer_info_already_saved", "✅ 고객 정보: {name} ({phone}, {email})").format(
            name=st.session_state.get("call_customer_name", ""),
            phone=st.session_state.get("call_customer_phone", ""),
            email=st.session_state.get("call_customer_email", "")
        ))
    
    # 통화 이력 저장 (고객 정보 포함, 자동)
    if st.session_state.get("call_messages") and st.session_state.call_customer_info_saved:
        # 이력 저장이 아직 안 된 경우에만 저장
        if "call_history_saved" not in st.session_state:
            try:
                inquiry_text = st.session_state.get("inquiry_text", "")
                customer_type = st.session_state.get("customer_type_sim_select", L.get("default_customer_type", "일반 고객"))
                if not customer_type:
                    customer_type = L.get("default_customer_type", "일반 고객")
                
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
                
                # 고객 정보 가져오기
                customer_name = st.session_state.get("call_customer_name", "")
                customer_phone = st.session_state.get("call_customer_phone", "")
                customer_email = st.session_state.get("call_customer_email", "")
                customer_id = st.session_state.get("call_customer_id", "")
                summary_memo = st.session_state.get("call_summary_memo", "")
                
                # 이력 저장 (고객 정보 및 요약 메모 포함)
                save_simulation_history_local(
                    initial_query=inquiry_text or L.get("phone_call_default", "전화 통화"),
                    customer_type=customer_type,
                    messages=converted_messages,
                    is_chat_ended=True,
                    attachment_context=st.session_state.get("call_content", "") or summary_memo,
                    is_call=True,
                    customer_name=customer_name,
                    customer_phone=customer_phone,
                    customer_email=customer_email,
                    customer_id=customer_id
                )
                
                st.session_state.call_history_saved = True
                st.success(L.get("call_history_saved_success", "✅ 통화 이력이 저장되었습니다."))
            except Exception as e:
                st.warning(L.get("call_history_save_error", "통화 이력 저장 중 오류 발생: {error}").format(error=e))
    
    # 이력 다운로드 섹션 (채팅 탭과 동일한 기능)
    st.markdown("---")
    st.markdown(f"**{L.get('download_current_call_history', '📥 현재 통화 이력 다운로드')}**")
    download_col1, download_col2, download_col3, download_col4, download_col5 = st.columns(5)
    
    current_session_history = None
    if st.session_state.get("call_messages"):
        try:
            inquiry_text = st.session_state.get("inquiry_text", "")
            customer_type = st.session_state.get("customer_type_sim_select", L.get("default_customer_type", "일반 고객"))
            if not customer_type:
                customer_type = L.get("default_customer_type", "일반 고객")
            
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
                inquiry_text or L.get("phone_call_default", "전화 통화"),
                customer_type,
                st.session_state.language
            )
            
            current_session_history = [{
                "id": f"call_{st.session_state.get('current_call_id', 'unknown')}",
                "timestamp": datetime.now().isoformat(),
                "initial_query": inquiry_text or L.get("phone_call_default", "전화 통화"),
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
        st.session_state.is_on_hold = False
        st.session_state.hold_start_time = None
        st.session_state.hold_total_seconds = 0
        st.session_state.provider_call_active = False
        st.session_state.call_direction = "inbound"
        st.success(L.get("new_call_ready", "✅ 새 통화를 시작할 수 있습니다."))



