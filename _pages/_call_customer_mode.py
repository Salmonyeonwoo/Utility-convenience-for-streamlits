# -*- coding: utf-8 -*-
"""
전화 시뮬레이터 - 고객 모드 (사용자=고객)
고객 입장에서 AI 상담원과 통화하는 모드
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime

def render_call_customer_mode():
    """고객 모드 전화 시뮬레이터 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # sim_perspective 초기화
    if "sim_perspective" not in st.session_state:
        st.session_state.sim_perspective = "CUSTOMER"
    
    if st.session_state.sim_perspective != "CUSTOMER":
        st.session_state.sim_perspective = "CUSTOMER"
    
    # 전화 시뮬레이터 세션 상태 초기화
    if "call_sim_stage" not in st.session_state:
        st.session_state.call_sim_stage = "WAITING_CALL"
    if "call_messages" not in st.session_state:
        st.session_state.call_messages = []
    if "incoming_phone_number" not in st.session_state:
        st.session_state.incoming_phone_number = ""
    if "current_call_id" not in st.session_state:
        st.session_state.current_call_id = None
    
    # 상태별 라우팅
    if st.session_state.call_sim_stage == "WAITING_CALL":
        try:
            from _pages._call_customer_waiting import render_customer_waiting
            render_customer_waiting()
        except ImportError:
            from _call_customer_waiting import render_customer_waiting
            render_customer_waiting()
    
    elif st.session_state.call_sim_stage == "RINGING":
        try:
            from _pages._call_customer_ringing import render_customer_ringing
            render_customer_ringing()
        except ImportError:
            from _call_customer_ringing import render_customer_ringing
            render_customer_ringing()
    
    elif st.session_state.call_sim_stage == "IN_CALL":
        try:
            from _pages._call_customer_in_call import render_customer_in_call
            render_customer_in_call()
        except ImportError:
            from _call_customer_in_call import render_customer_in_call
            render_customer_in_call()
    
    elif st.session_state.call_sim_stage == "CALL_ENDED":
        call_duration = st.session_state.get("call_duration", 0)
        minutes = int(call_duration // 60)
        seconds = int(call_duration % 60)
        if minutes > 0:
            duration_msg = L.get("call_ended_with_duration", "통화가 종료되었습니다. (통화 시간: {minutes}분 {seconds}초)").format(minutes=minutes, seconds=seconds)
        else:
            duration_msg = L.get("call_ended_with_seconds", "통화가 종료되었습니다. (통화 시간: {seconds}초)").format(seconds=seconds)
        st.success(duration_msg)
        
        # 통화 요약 생성 및 표시
        st.markdown("---")
        st.markdown(f"### 📋 {L.get('call_summary_header', '통화 요약')}")
        
        if st.session_state.call_messages:
            # 요약 생성
            if "call_summary_generated" not in st.session_state:
                st.session_state.call_summary_generated = False
            
            if not st.session_state.call_summary_generated:
                with st.spinner("통화 요약 생성 중..."):
                    try:
                        from utils.history_handler import generate_call_summary
                        call_summary = generate_call_summary(
                            messages=st.session_state.call_messages,
                            initial_query=st.session_state.get("inquiry_text", ""),
                            customer_type=L.get("default_customer_type", "일반 고객"),
                            current_lang_key=current_lang
                        )
                        if call_summary:
                            st.session_state.call_summary = call_summary
                            st.session_state.call_summary_generated = True
                    except Exception as e:
                        st.error(f"요약 생성 오류: {e}")
                        st.session_state.call_summary = None
                        st.session_state.call_summary_generated = True
            
            # 요약 표시 및 재생성 버튼
            col_summary1, col_summary2 = st.columns([3, 1])
            with col_summary2:
                if st.button(f"🔄 {L.get('regenerate_summary', '요약 재생성')}", key="regenerate_summary"):
                    st.session_state.call_summary_generated = False
                    st.session_state.call_summary = None
            
            # 재생성 요청 시 즉시 생성
            if not st.session_state.call_summary_generated:
                with st.spinner(L.get("generating_call_summary", "통화 요약 생성 중...")):
                    try:
                        from utils.history_handler import generate_call_summary
                        call_summary = generate_call_summary(
                            messages=st.session_state.call_messages,
                            initial_query=st.session_state.get("inquiry_text", ""),
                            customer_type=L.get("default_customer_type", "일반 고객"),
                            current_lang_key=current_lang
                        )
                        if call_summary:
                            st.session_state.call_summary = call_summary
                            st.session_state.call_summary_generated = True
                    except Exception as e:
                        st.error(f"{L.get('summary_generation_error', '요약 생성 오류')}: {e}")
                        st.session_state.call_summary = None
                        st.session_state.call_summary_generated = True
            
            if st.session_state.get("call_summary"):
                summary = st.session_state.call_summary
                if isinstance(summary, dict):
                    st.markdown(f"#### {L.get('customer_inquiry_label', '주요 문의')}")
                    st.info(summary.get("customer_inquiry", L.get("no_summary_info", "요약 정보 없음")))
                    
                    st.markdown(f"#### {L.get('key_solutions', '핵심 솔루션')}")
                    key_solutions = summary.get("key_solutions", [])
                    if key_solutions:
                        for i, solution in enumerate(key_solutions, 1):
                            st.write(f"{i}. {solution}")
                    else:
                        st.info(L.get("no_solution_info", "솔루션 정보 없음"))
                    
                    st.markdown(f"#### {L.get('overall_summary_label', '전체 요약')}")
                    st.write(summary.get("summary", L.get("no_summary_info", "요약 정보 없음")))
                else:
                    st.write(summary)
            else:
                st.info(L.get("no_summary_available", "요약 정보가 없습니다."))
        
        st.markdown("---")
        
        # 이력 다운로드 기능 추가 (Word, PDF, PPTX, JSON, CSV)
        st.markdown(f"### 📥 {L.get('download_current_call_history', '통화 이력 다운로드')}")
        
        if st.session_state.call_messages:
            # 통화 메시지를 채팅 형식으로 변환
            converted_messages = []
            for msg in st.session_state.call_messages:
                msg_copy = msg.copy()
                role = msg.get("role", "")
                content = msg.get("content", "")
                # audio 필드 제거 (bytes는 직렬화 불가)
                if "audio" in msg_copy and isinstance(msg_copy["audio"], bytes):
                    msg_copy["audio"] = "[Audio data - binary]"
                
                if role == "agent":
                    converted_messages.append({"role": "agent_response", "content": content})
                elif role == "customer":
                    converted_messages.append({"role": "customer", "content": content})
                elif role in ["system", "system_transfer", "supervisor"]:
                    converted_messages.append({"role": "supervisor", "content": content})
            
            # 요약 생성 (generate_chat_summary 사용)
            current_session_history = None
            try:
                from utils.history_handler import generate_chat_summary
                inquiry_text = st.session_state.get("inquiry_text", "")
                customer_type = st.session_state.get("customer_type_sim_select", L.get("default_customer_type", "일반 고객"))
                
                current_session_summary = generate_chat_summary(
                    converted_messages,
                    inquiry_text or "전화 통화",
                    customer_type,
                    current_lang
                )
                
                current_session_history = [{
                    "id": f"call_{st.session_state.get('current_call_id', 'unknown')}",
                    "timestamp": datetime.now().isoformat(),
                    "initial_query": inquiry_text or "전화 통화",
                    "customer_type": customer_type,
                    "language_key": current_lang,
                    "messages": converted_messages,
                    "summary": current_session_summary,
                    "is_chat_ended": True,
                    "is_call": True,
                    "call_duration": call_duration,
                    "customer_info": st.session_state.get("call_customer_info", {}),
                    "call_summary": st.session_state.get("call_summary", {})
                }]
            except Exception as e:
                st.warning(f"{L.get('history_generation_error', '이력 생성 중 오류')}: {e}")
                current_session_history = None
            
            # 다운로드 버튼들 (5개 컬럼: Word, PPTX, PDF, JSON, CSV)
            download_col1, download_col2, download_col3, download_col4, download_col5 = st.columns(5)
            
            if current_session_history:
                # Word 다운로드
                with download_col1:
                    try:
                        from utils.history_handler import export_history_to_word
                        import os
                        filepath_word = export_history_to_word(current_session_history, lang=current_lang)
                        with open(filepath_word, "rb") as f:
                            st.download_button(
                                label=f"📄 {L.get('download_history_word', 'Word 다운로드')}",
                                data=f.read(),
                                file_name=os.path.basename(filepath_word),
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key="download_call_word"
                            )
                    except Exception as e:
                        st.error(f"{L.get('word_download_error', 'Word 다운로드 오류')}: {e}")
                
                # PPTX 다운로드
                with download_col2:
                    try:
                        from utils.history_handler import export_history_to_pptx
                        import os
                        filepath_pptx = export_history_to_pptx(current_session_history, lang=current_lang)
                        with open(filepath_pptx, "rb") as f:
                            st.download_button(
                                label=f"📊 {L.get('download_history_pptx', 'PPTX 다운로드')}",
                                data=f.read(),
                                file_name=os.path.basename(filepath_pptx),
                                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                key="download_call_pptx"
                            )
                    except Exception as e:
                        st.error(f"{L.get('pptx_download_error', 'PPTX 다운로드 오류')}: {e}")
                
                # PDF 다운로드
                with download_col3:
                    try:
                        from utils.history_handler import export_history_to_pdf
                        import os
                        filepath_pdf = export_history_to_pdf(current_session_history, lang=current_lang)
                        with open(filepath_pdf, "rb") as f:
                            st.download_button(
                                label=f"📑 {L.get('download_history_pdf', 'PDF 다운로드')}",
                                data=f.read(),
                                file_name=os.path.basename(filepath_pdf),
                                mime="application/pdf",
                                key="download_call_pdf"
                            )
                    except Exception as e:
                        st.error(f"{L.get('pdf_download_error', 'PDF 다운로드 오류')}: {e}")
                
                # JSON 다운로드
                with download_col4:
                    try:
                        import json
                        history_data = {
                            "initial_query": st.session_state.get("inquiry_text", ""),
                            "customer_type": L.get("default_customer_type", "일반 고객"),
                            "messages": converted_messages,
                            "language_key": current_lang,
                            "is_call": True,
                            "call_duration": call_duration,
                            "timestamp": datetime.now().isoformat(),
                            "customer_info": st.session_state.get("call_customer_info", {}),
                            "summary": st.session_state.get("call_summary", {}),
                            "chat_summary": current_session_summary
                        }
                        history_json = json.dumps(history_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            label=f"📋 {L.get('download_history_json', 'JSON 다운로드')}",
                            data=history_json.encode('utf-8'),
                            file_name=f"call_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_call_json"
                        )
                    except Exception as e:
                        st.error(f"{L.get('json_download_error', 'JSON 다운로드 오류')}: {e}")
                
                # CSV 다운로드
                with download_col5:
                    try:
                        import csv
                        import io
                        output = io.StringIO()
                        writer = csv.writer(output)
                        
                        # 헤더
                        writer.writerow(["Role", "Content", "Timestamp"])
                        
                        # 메시지 데이터
                        for msg in converted_messages:
                            writer.writerow([
                                msg.get("role", ""),
                                msg.get("content", ""),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ])
                        
                        csv_data = output.getvalue()
                        st.download_button(
                            label=f"📊 {L.get('download_history_csv', 'CSV 다운로드')}",
                            data=csv_data.encode('utf-8-sig'),
                            file_name=f"call_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_call_csv"
                        )
                    except Exception as e:
                        st.error(f"{L.get('csv_download_error', 'CSV 다운로드 오류')}: {e}")
            else:
                st.warning(L.get("no_history_to_download", "다운로드할 이력이 없습니다."))
            
            # 이력 저장 버튼 (별도 행)
            st.markdown("---")
            if st.button(f"💾 {L.get('save_history_button', '이력 저장')}", key="save_call_history", use_container_width=True):
                try:
                    from utils.history_handler import save_simulation_history_local
                    save_simulation_history_local(
                        initial_query=st.session_state.get("inquiry_text", ""),
                        customer_type=L.get("default_customer_type", "일반 고객"),
                        messages=converted_messages,
                        is_chat_ended=True,
                        attachment_context="",
                        is_call=True,
                        customer_name=st.session_state.get("call_customer_info", {}).get("name", ""),
                        customer_phone=st.session_state.get("call_customer_info", {}).get("phone", ""),
                        customer_email=st.session_state.get("call_customer_info", {}).get("email", ""),
                        customer_id=st.session_state.get("call_customer_id", "")
                    )
                    st.success(L.get("call_history_saved_success", "✅ 통화 이력이 저장되었습니다."))
                except Exception as e:
                    st.error(f"{L.get('save_history_error', '이력 저장 오류')}: {e}")
        
        st.markdown("---")
        
        if st.button(L.get("new_call_button", "새 통화 시작"), key="btn_new_call_customer"):
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.call_messages = []
            st.session_state.inquiry_text = ""
            st.session_state.incoming_phone_number = None
            st.session_state.call_active = False
            st.session_state.start_time = None
            st.session_state.call_duration = None
