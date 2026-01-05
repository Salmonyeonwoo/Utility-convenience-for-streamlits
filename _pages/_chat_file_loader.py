# ========================================
# _pages/_chat_file_loader.py
# 채팅 시뮬레이터 - 파일 로더 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import os


def render_file_loader_panel(L, current_lang):
    """파일에서 이력 불러오기 패널 렌더링"""
    try:
        from utils.file_loader import scan_data_directory, load_file_by_type, parse_history_from_file_data
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dirs = [
            os.path.join(base_dir, "data"),
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\data",
        ]
        
        scanned_files = []
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = scan_data_directory(data_dir)
                scanned_files.extend(files)
        
        seen_files = set()
        unique_files = []
        for file_meta in scanned_files:
            file_name = file_meta.get("file_name", "")
            if file_name not in seen_files:
                seen_files.add(file_name)
                unique_files.append(file_meta)
        
        scanned_files = unique_files
        
        try:
            from utils.file_loader import scan_github_repository
            from llm_client import get_api_key
            
            github_token = get_api_key("github")
            github_files = scan_github_repository(github_token=github_token)
            if github_files:
                scanned_files.extend(github_files)
        except Exception:
            pass
        
        # ⭐ 매일 20~30개씩만 가져오도록 필터링 (오늘 날짜 기준)
        today = datetime.now().date()
        
        # 오늘 날짜에 수정된 파일만 필터링
        today_files = []
        for file_meta in scanned_files:
            try:
                modified_time_str = file_meta.get("modified_time", "")
                if not modified_time_str:
                    continue
                
                modified_time = None
                
                # 다양한 날짜 형식 파싱 시도
                try:
                    # ISO 형식에서 날짜 부분만 추출
                    # 예: "2025-01-01T12:00:00" -> "2025-01-01"
                    # 예: "2025-01-01T12:00:00Z" -> "2025-01-01"
                    date_str = modified_time_str.split('T')[0] if 'T' in modified_time_str else modified_time_str[:10]
                    
                    # 날짜 파싱
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    
                    # 오늘 날짜인지 확인
                    if file_date == today:
                        today_files.append(file_meta)
                except (ValueError, IndexError):
                    # 날짜 파싱 실패 시 건너뛰기
                    continue
            except Exception:
                # 예외 발생 시 건너뛰기
                continue
        
        # 날짜 기준 최신순 정렬
        today_files.sort(key=lambda x: x.get("modified_time", ""), reverse=True)
        
        # 매일 정확히 20개 제한
        daily_limit = 20
        filtered_files = today_files[:daily_limit] if len(today_files) > 0 else []
        
        if filtered_files:
            with st.expander(f"📁 {L.get('load_history_from_file', '파일에서 이력 불러오기')} ({L.get('today', '오늘')}: {len(filtered_files)}{L.get('files_count', '개')})", expanded=False):
                local_files = [f for f in filtered_files if f.get("source") == "local"]
                github_files = [f for f in filtered_files if f.get("source") in ["github", "github_api"]]
                
                if local_files:
                    st.markdown(f"**📂 {L.get('local_files', '로컬 파일')}** ({L.get('today_modified', '오늘 수정')}: {len(local_files)}{L.get('files_count', '개')})")
                if github_files:
                    st.markdown(f"**🌐 {L.get('github_files', 'GitHub 파일')}** ({L.get('today_modified', '오늘 수정')}: {len(github_files)}{L.get('files_count', '개')})")
                
                file_groups = {}
                for file_meta in filtered_files:
                    file_type = file_meta.get("file_type", "unknown")
                    if file_type not in file_groups:
                        file_groups[file_type] = []
                    file_groups[file_type].append(file_meta)
                
                for file_type, files in file_groups.items():
                    file_type_label = {
                        "json": "📄 JSON",
                        "docx": "📝 Word",
                        "pdf": "📕 PDF",
                        "pptx": "📊 PPTX",
                        "csv": "📋 CSV"
                    }.get(file_type, f"📎 {file_type.upper()}")
                    
                    st.markdown(f"**{file_type_label} {L.get('file_label', '파일')}** ({len(files)}{L.get('files_count', '개')})")
                    for file_meta in files:
                        _render_file_item(L, file_meta)
                
                # 오늘 가져온 파일 수 정보 표시
                if len(filtered_files) > 0:
                    st.info(f"ℹ️ {L.get('today_files_info', '오늘 날짜')}({today.strftime('%Y-%m-%d')}){L.get('modified_files_display', '에 수정된 파일 중')} {len(filtered_files)}{L.get('files_displayed', '개를 표시합니다')}. ({L.get('daily_limit', '제한')}: 20{L.get('files_per_day', '개/일')})")
        elif scanned_files:
            # 오늘 날짜 파일이 없는 경우 안내
            st.info(f"ℹ️ {L.get('today_files_info', '오늘 날짜')}({today.strftime('%Y-%m-%d')}){L.get('no_modified_files', '에 수정된 파일이 없습니다')}. {L.get('total_files_scanned', '총')} {len(scanned_files)}{L.get('files_found', '개의 파일이 검색되었습니다')}.")
    except ImportError:
        pass
    except Exception:
        pass


def _render_file_item(L, file_meta):
    """개별 파일 항목 렌더링"""
    file_name = file_meta.get("file_name", "")
    file_path = file_meta.get("file_path", "")
    file_type = file_meta.get("file_type", "unknown")
    file_size = file_meta.get("file_size", 0)
    modified_time = file_meta.get("modified_time", "")
    
    if file_size < 1024:
        size_str = f"{file_size}B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f}KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.1f}MB"
    
    try:
        mod_time = datetime.fromisoformat(modified_time)
        time_str = mod_time.strftime("%m-%d %H:%M")
    except:
        time_str = modified_time[:10] if modified_time else ""
    
    col_file1, col_file2 = st.columns([3, 1])
    with col_file1:
        st.caption(f"{file_name} ({size_str})")
    with col_file2:
        if st.button(L.get("load", "로드"), key=f"load_file_{file_name}_{st.session_state.sim_instance_id}", 
                   use_container_width=True):
            _handle_file_load(L, file_meta, file_path, file_type)
    
    st.caption(f"{L.get('modified', '수정')}: {time_str}")
    st.markdown("---")


def _handle_file_load(L, file_meta, file_path, file_type):
    """파일 로드 처리"""
    file_name = file_meta.get("file_name", "")
    with st.spinner(f"{L.get('file_loading', '파일 로드 중')}: {file_name}..."):
        try:
            from utils.file_loader import load_file_by_type, parse_history_from_file_data
            
            github_token = None
            if file_meta.get("source") == "github_api":
                from llm_client import get_api_key
                github_token = get_api_key("github") or file_meta.get("github_token")
            
            file_data = load_file_by_type(file_path, file_type, github_token=github_token)
            
            if file_data:
                history = parse_history_from_file_data(file_data, file_name)
                
                if history:
                    if "parse_error" in history:
                        st.warning(f"⚠️ {L.get('file_parse_warning', '파일 파싱 중 일부 오류가 발생했습니다')}: {history.get('parse_error', '')}")
                    if "raw_data" in history:
                        st.info(f"ℹ️ {L.get('raw_data_stored', '원본 데이터가 보관되었습니다. 필요시 확인하세요')}.")
                    
                    if "initial_query" in history:
                        st.session_state.customer_query_text_area = history["initial_query"]
                    
                    if "messages" in history and history["messages"]:
                        st.session_state.simulator_messages = history["messages"]
                    elif "initial_query" in history:
                        st.session_state.simulator_messages = [
                            {"role": "customer", "content": history["initial_query"]}
                        ]
                    
                    if "customer_type" in history:
                        st.session_state.customer_type_sim_select = history["customer_type"]
                    
                    if "summary" in history:
                        st.session_state.initial_advice_provided = True
                    
                    st.session_state.is_chat_ended = history.get("is_chat_ended", False)
                    
                    if st.session_state.is_chat_ended:
                        st.session_state.sim_stage = "CLOSING"
                    else:
                        messages = st.session_state.simulator_messages
                        if messages:
                            last_role = messages[-1].get("role") if messages else None
                            if last_role == "agent_response":
                                st.session_state.sim_stage = "CUSTOMER_TURN"
                            else:
                                st.session_state.sim_stage = "AGENT_TURN"
                        else:
                            st.session_state.sim_stage = "AGENT_TURN"
                    
                    st.success(f"✅ {L.get('file_loaded_success', '파일 로드 완료')}: {file_name}")
                else:
                    st.warning(f"⚠️ {L.get('file_parse_error', '파일을 이력 형식으로 변환할 수 없습니다')}: {file_name}")
            else:
                st.error(f"❌ {L.get('file_load_failed', '파일 로드 실패')}: {file_name}")
        except Exception as e:
            st.error(f"❌ {L.get('file_load_error', '파일 로드 오류')}: {str(e)}")



