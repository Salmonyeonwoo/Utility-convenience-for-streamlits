# ========================================
# _pages/_chat_simulator_panels.py
# 채팅 시뮬레이터의 패널 렌더링 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.history_handler import get_daily_data_statistics
from datetime import datetime
import os

def _render_customer_list_panel(L, current_lang):
    """고객 목록 패널 렌더링 (col1) - 스크린샷 스타일 + 파일 자동 로드"""
    st.subheader("고객 목록")
    
    # 스크린샷 스타일: 고객 목록 버튼 스타일 개선
    st.markdown("""
    <style>
    /* 고객 목록 버튼 스타일 (스크린샷 스타일) */
    div[data-testid="stButton"] > button[kind="primary"] {
        border: 2px solid #FF69B4;
        background-color: #FFFFFF;
        color: #333;
        font-weight: 500;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #FFF0F5;
        border-color: #FF1493;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 1px solid #E0E0E0;
        background-color: #FFFFFF;
        color: #333;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #F5F5F5;
        border-color: #BDBDBD;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 파일 자동 인식 및 로드 기능
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
        
        if scanned_files:
            with st.expander("📁 파일에서 이력 불러오기", expanded=False):
                local_files = [f for f in scanned_files if f.get("source") == "local"]
                github_files = [f for f in scanned_files if f.get("source") in ["github", "github_api"]]
                
                if local_files:
                    st.markdown("**📂 로컬 파일**")
                elif github_files:
                    st.markdown("**🌐 GitHub 파일**")
                
                file_groups = {}
                for file_meta in scanned_files[:30]:
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
                    
                    st.markdown(f"**{file_type_label} 파일**")
                    for file_meta in files:
                        file_name = file_meta.get("file_name", "")
                        file_path = file_meta.get("file_path", "")
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
                            if st.button("로드", key=f"load_file_{file_name}_{st.session_state.sim_instance_id}", 
                                       use_container_width=True):
                                with st.spinner(f"파일 로드 중: {file_name}..."):
                                    try:
                                        github_token = None
                                        if file_meta.get("source") == "github_api":
                                            from llm_client import get_api_key
                                            github_token = get_api_key("github") or file_meta.get("github_token")
                                        
                                        file_data = load_file_by_type(file_path, file_type, github_token=github_token)
                                        
                                        if file_data:
                                            history = parse_history_from_file_data(file_data, file_name)
                                            
                                            if history:
                                                if "parse_error" in history:
                                                    st.warning(f"⚠️ 파일 파싱 중 일부 오류가 발생했습니다: {history.get('parse_error', '')}")
                                                if "raw_data" in history:
                                                    st.info(f"ℹ️ 원본 데이터가 보관되었습니다. 필요시 확인하세요.")
                                                
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
                                                
                                                st.success(f"✅ 파일 로드 완료: {file_name}")
                                            else:
                                                st.warning(f"⚠️ 파일을 이력 형식으로 변환할 수 없습니다: {file_name}")
                                        else:
                                            st.error(f"❌ 파일 로드 실패: {file_name}")
                                    except Exception as e:
                                        st.error(f"❌ 파일 로드 오류: {str(e)}")
                        
                        st.caption(f"수정: {time_str}")
                        st.markdown("---")
    except ImportError:
        pass
    except Exception:
        pass
    
    # 데이터 디렉토리에서 고객 목록 추출
    try:
        from utils.customer_list_extractor import extract_customers_from_data_directories
        from utils.history_handler import load_simulation_histories_local
        from utils.customer_list_extractor import extract_customers_from_histories
        
        # 데이터 디렉토리 경로 설정
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dirs = [
            os.path.join(base_dir, "data"),
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\data",
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\customer data histories via streamlits",
        ]
        
        # 실제 존재하는 디렉토리만 필터링
        existing_dirs = [d for d in data_dirs if os.path.exists(d)]
        
        # 데이터 디렉토리에서 고객 정보 추출
        customers_from_files = extract_customers_from_data_directories(existing_dirs)
        
        # 시뮬레이션 이력에서도 고객 정보 추출
        histories = load_simulation_histories_local(current_lang)
        customers_from_histories = extract_customers_from_histories(histories)
        
        # 두 소스의 고객 정보 병합
        all_customers_dict = {}
        for customer in customers_from_files:
            name = customer.get('customer_name', '')
            if name:
                if name not in all_customers_dict:
                    all_customers_dict[name] = customer
                else:
                    # 상담 횟수 합산
                    all_customers_dict[name]['consultation_count'] += customer.get('consultation_count', 0)
        
        for customer in customers_from_histories:
            name = customer.get('customer_name', '')
            if name:
                if name not in all_customers_dict:
                    all_customers_dict[name] = customer
                else:
                    # 상담 횟수 합산
                    all_customers_dict[name]['consultation_count'] += customer.get('consultation_count', 0)
        
        # 고객 데이터 관리자에서도 가져오기
        try:
            if hasattr(st.session_state, 'customer_data_manager') and st.session_state.customer_data_manager:
                manager_customers = st.session_state.customer_data_manager.load_all_customers()
                for customer in manager_customers:
                    basic_info = customer.get("basic_info", {})
                    customer_name = basic_info.get("customer_name", "")
                    customer_id = basic_info.get("customer_id", "")
                    
                    if customer_name:
                        if customer_name not in all_customers_dict:
                            # 상담 이력에서 횟수 계산
                            consultation_history = customer.get("data", {}).get("consultation_history", [])
                            consultation_count = len(consultation_history) if consultation_history else 1
                            
                            all_customers_dict[customer_name] = {
                                'customer_name': customer_name,
                                'customer_id': customer_id,
                                'consultation_count': consultation_count,
                                'last_consultation_date': '',
                                'customer_data': customer
                            }
                        else:
                            # 상담 횟수 추가
                            consultation_history = customer.get("data", {}).get("consultation_history", [])
                            if consultation_history:
                                all_customers_dict[customer_name]['consultation_count'] += len(consultation_history)
        except Exception:
            pass
        
        # 고객 목록을 리스트로 변환하고 정렬
        all_customers_list = list(all_customers_dict.values())
        all_customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
        
        # 현재 선택된 고객 확인
        current_customer_name = None
        if st.session_state.get("customer_data"):
            basic_info = st.session_state.customer_data.get("basic_info", {})
            current_customer_name = basic_info.get("customer_name", "")
        if not current_customer_name:
            current_customer_name = st.session_state.get('customer_name', '')
        
        # 고객 목록 표시 (두 번째 스크린샷 스타일)
        if all_customers_list:
            # 고객 목록 스타일 추가
            st.markdown("""
            <style>
            .customer-badge {
                background-color: #FFB6C1;
                color: #333;
                border-radius: 12px;
                padding: 2px 8px;
                font-size: 0.85em;
                font-weight: 500;
                display: inline-block;
            }
            </style>
            """, unsafe_allow_html=True)
            
            for customer in all_customers_list[:20]:  # 최대 20명 표시
                customer_name = customer.get('customer_name', '고객')
                consultation_count = customer.get('consultation_count', 0)
                is_selected = current_customer_name == customer_name
                
                # 고객 이름과 배지를 한 줄에 표시
                col_name, col_badge = st.columns([4, 1])
                
                with col_name:
                    if st.button(f"👤 {customer_name}", 
                               key=f"customer_list_{customer_name}_{st.session_state.sim_instance_id}",
                               use_container_width=True, 
                               type="primary" if is_selected else "secondary"):
                        # 고객 데이터 설정
                        customer_data = customer.get('customer_data', {})
                        if customer_data:
                            st.session_state.customer_data = customer_data
                        else:
                            # customer_data가 없으면 기본 구조 생성
                            st.session_state.customer_data = {
                                "basic_info": {
                                    "customer_name": customer_name,
                                    "customer_id": customer.get('customer_id', '')
                                },
                                "data": {}
                            }
                        st.session_state.customer_name = customer_name
                
                with col_badge:
                    if consultation_count > 0:
                        # 배지를 버튼 옆에 표시
                        st.markdown(f'<div style="text-align: center; margin-top: 8px;"><span class="customer-badge">{consultation_count}개</span></div>', unsafe_allow_html=True)
        else:
            st.info("등록된 고객이 없습니다.")
    except ImportError as e:
        st.info(f"고객 목록 추출 모듈을 불러올 수 없습니다: {e}")
    except Exception as e:
        st.info(f"고객 목록을 불러올 수 없습니다: {e}")


def _render_customer_info_panel(L, current_lang):
    """고객 정보 패널 렌더링 (col3) - app.py 스타일로 간소화"""
    st.subheader("고객 정보")
    
    customer_data = st.session_state.get("customer_data", None)
    
    if customer_data:
        customer_info = customer_data.get("data", {})
        basic_info = customer_data.get("basic_info", {})
        
        customer_name = (
            basic_info.get('customer_name', '') or 
            customer_info.get('name', '') or 
            st.session_state.get('customer_name', '고객')
        )
        
        st.markdown(f"### 👤 {customer_name}")
        
        customer_id = basic_info.get("customer_id", "N/A")
        email = customer_info.get('email', st.session_state.get('customer_email', 'N/A'))
        phone = customer_info.get('phone', st.session_state.get('customer_phone', 'N/A'))
        
        st.markdown(f"**고객 ID:** {customer_id}")
        if customer_name and customer_name != '고객':
            st.markdown(f"**성함:** {customer_name}")
        st.markdown(f"**연락처:** {phone}")
        st.markdown(f"**이메일:** {email}")
        
        crm_profile = customer_info.get("crm_profile", {})
        if crm_profile:
            personality = crm_profile.get('personality', 'N/A')
            st.markdown(f"**성향:** {personality}")
            
            survey_score = crm_profile.get('survey_score', 4.5)
            st.metric("설문 점수", f"{survey_score:.1f} / 5.0")
    else:
        initial_query_msg = None
        for msg in st.session_state.get("simulator_messages", []):
            if msg.get("role") == "initial_query" or msg.get("role") == "customer":
                initial_query_msg = msg
                break
        
        if st.session_state.get('customer_name') or st.session_state.get('customer_email') or st.session_state.get('customer_phone'):
            customer_display_name = st.session_state.get('customer_name', '고객')
            st.markdown(f"### 👤 {customer_display_name}")
            if st.session_state.get('customer_name'):
                st.markdown(f"**성함:** {st.session_state.customer_name}")
            if st.session_state.get('customer_email'):
                st.markdown(f"**이메일:** {st.session_state.customer_email}")
            if st.session_state.get('customer_phone'):
                st.markdown(f"**연락처:** {st.session_state.customer_phone}")
        elif initial_query_msg:
            st.info("고객 정보를 불러오려면 고객 데이터 버튼을 클릭하세요.")
        else:
            st.info("고객을 선택하면 상세 정보가 표시됩니다.")
    
    # 일일 통계를 col3 하단에 배치 (축소된 버전)
    if st.session_state.sim_stage not in ["WAIT_ROLE_SELECTION", "WAIT_FIRST_QUERY", "idle"]:
        st.markdown("---")
        st.markdown("**📊 일일 통계**")
        daily_stats = get_daily_data_statistics(st.session_state.language)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric(L.get("daily_stats_cases_collected", "수집 케이스"), daily_stats["total_cases"], help="오늘 수집된 케이스 수")
        with col_stat2:
            st.metric(L.get("daily_stats_unique_customers", "고유 고객"), daily_stats["unique_customers"], 
                     delta=L.get("daily_stats_target_met", "목표: 5인 이상") if daily_stats["target_met"] else L.get("daily_stats_target_not_met", "목표 미달"))
        
        col_stat3, col_stat4 = st.columns(2)
        with col_stat3:
            st.metric(L.get("daily_stats_summary_completed", "요약 완료"), daily_stats["cases_with_summary"], help="요약 완료된 케이스 수")
        with col_stat4:
            status_icon = "✅" if daily_stats["target_met"] else "⚠️"
            st.metric(L.get("daily_stats_goal_achievement", "목표 달성"), status_icon,
                     delta=L.get("daily_stats_achieved", "달성") if daily_stats["target_met"] else L.get("daily_stats_not_achieved", "미달성"))

