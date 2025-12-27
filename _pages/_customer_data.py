# 고객 데이터 관리 페이지 모듈
import streamlit as st
from customer_data_manager import CustomerDataManager
from lang_pack import LANG
import os


def render_customer_data_page():
    """고객 데이터 조회 페이지 렌더링 (등록 기능 제거, 조회만 유지)"""
    try:
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        st.title(f"📋 {L.get('customer_data_inquiry_tab', '고객 데이터 조회')}")
        manager = CustomerDataManager()
        
        st.subheader(L.get("customer_inquiry_tab", "고객 조회"))
        
        # 검색 필드 (고객 ID 검색 추가)
        col_search1, col_search2, col_search3, col_search4, col_search_btn = st.columns([1.5, 1.5, 1.5, 1.5, 1])
        with col_search1:
            search_customer_id = st.text_input(L.get("customer_id_label", "고객 ID"), key="search_customer_id", placeholder=L.get("customer_id_search_placeholder", "CUST..."))
        with col_search2:
            search_name = st.text_input(L.get("customer_search_name", "고객명"), key="search_customer_name", placeholder=L.get("customer_search_name", "고객명"))
        with col_search3:
            search_phone = st.text_input(L.get("customer_search_phone", "전화번호"), key="search_customer_phone", placeholder=L.get("customer_search_phone", "전화번호"))
        with col_search4:
            search_email = st.text_input(L.get("customer_search_email", "이메일 주소"), key="search_customer_email", placeholder=L.get("customer_search_email", "이메일 주소"))
        with col_search_btn:
            st.write("")  # 공간 확보
            st.write("")  # 공간 확보
            search_clicked = st.button(L.get("customer_search", "검색"), type="primary", use_container_width=True)
        
        # ⭐ 수정: 채팅/전화 시뮬레이터와 동일하게 여러 소스에서 고객 정보 가져오기
        all_customers_list = []
        
        # 1. CustomerDataManager에서 고객 가져오기
        manager_customers = manager.load_all_customers()
        for customer in manager_customers:
            all_customers_list.append({
                'customer_id': customer.get('customer_id', ''),
                'customer_name': customer.get('customer_name', ''),
                'phone': customer.get('phone', ''),
                'email': customer.get('email', ''),
                'source': 'customer_database',
                'customer_data': customer
            })
        
        # 2. 시뮬레이션 이력에서 고객 정보 추출
        try:
            from utils.history_handler import load_simulation_histories_local
            from utils.customer_list_extractor import extract_customers_from_histories
            
            current_lang = st.session_state.get("language", "ko")
            histories = load_simulation_histories_local(current_lang)
            customers_from_histories = extract_customers_from_histories(histories)
            
            for customer in customers_from_histories:
                customer_name = customer.get('customer_name', '')
                if customer_name:
                    # 중복 확인 (고객명 기준)
                    existing = next((c for c in all_customers_list if c.get('customer_name') == customer_name), None)
                    if not existing:
                        all_customers_list.append({
                            'customer_id': customer.get('customer_id', ''),
                            'customer_name': customer_name,
                            'phone': customer.get('phone', ''),
                            'email': customer.get('email', ''),
                            'source': 'simulation_history',
                            'customer_data': customer.get('customer_data', {})
                        })
        except Exception as e:
            pass  # 오류 발생해도 계속 진행
        
        # base_dir 정의 (이력 불러오기에서 사용)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 3. 데이터 디렉토리에서 고객 정보 추출
        try:
            from utils.customer_list_extractor import extract_customers_from_data_directories
            data_dirs = [
                os.path.join(base_dir, "customer data histories via streamlits"),
                os.path.join(base_dir, "data"),
            ]
            existing_dirs = [d for d in data_dirs if os.path.exists(d)]
            if existing_dirs:
                customers_from_files = extract_customers_from_data_directories(existing_dirs)
                for customer in customers_from_files:
                    customer_name = customer.get('customer_name', '')
                    if customer_name:
                        # 중복 확인 (고객명 기준)
                        existing = next((c for c in all_customers_list if c.get('customer_name') == customer_name), None)
                        if not existing:
                            all_customers_list.append({
                                'customer_id': customer.get('customer_id', ''),
                                'customer_name': customer_name,
                                'phone': customer.get('phone', ''),
                                'email': customer.get('email', ''),
                                'source': 'data_directory',
                                'customer_data': customer.get('customer_data', {})
                            })
        except Exception as e:
            pass  # 오류 발생해도 계속 진행
        
        # 검색 필터 적용 (고객 ID 검색 포함)
        if search_clicked or search_customer_id or search_name or search_phone or search_email:
            filtered_customers = []
            for customer in all_customers_list:
                # 고객 ID 검색 (우선순위)
                id_match = not search_customer_id or (search_customer_id.upper() in customer.get('customer_id', '').upper())
                name_match = not search_name or (search_name.lower() in customer.get('customer_name', '').lower())
                phone_match = not search_phone or (search_phone in customer.get('phone', ''))
                email_match = not search_email or (search_email.lower() in customer.get('email', '').lower())
                
                # 고객 ID가 입력된 경우 ID 우선 매칭, 아니면 다른 필드 매칭 (OR 조건)
                if search_customer_id:
                    if id_match:
                        filtered_customers.append(customer)
                else:
                    # 이름, 전화번호, 이메일 중 하나라도 일치하면 표시
                    if name_match or phone_match or email_match:
                        filtered_customers.append(customer)
            all_customers_list = filtered_customers
        
        if all_customers_list:
            # 고객 목록 정렬 (이름순)
            all_customers_list.sort(key=lambda x: x.get('customer_name', ''))
            
            customer_options = {}
            for idx, c in enumerate(all_customers_list):
                display_name = f"{c.get('customer_name', 'N/A')}"
                if c.get('customer_id'):
                    display_name += f" ({c.get('customer_id')})"
                customer_options[display_name] = idx
            
            selected_customer_display = st.selectbox(L.get("select_customer_to_view_details", "고객을 선택하면 상세 정보가 표시됩니다"), list(customer_options.keys()), key="select_customer_view")
            if selected_customer_display:
                selected_idx = customer_options[selected_customer_display]
                customer = all_customers_list[selected_idx]
                
                st.markdown(f"### 👤 {customer.get('customer_name', 'N/A')} {L.get('customer_info', '고객 정보')}")
                st.markdown(f"**{L.get('customer_id_label', '고객 ID')}:** {customer.get('customer_id', 'N/A')} | **{L.get('contact_label', '연락처')}:** {customer.get('phone', 'N/A')} | **{L.get('email_label', '이메일')}:** {customer.get('email', 'N/A')}")
                if customer.get('source'):
                    st.caption(f"{L.get('data_source', '데이터 소스')}: {customer.get('source')}")
                
                # ⭐ 이전 응대 이력 불러오기 (로컬 폴더에서 자동 인식)
                try:
                    import json
                    from utils.history_handler import load_simulation_histories_local
                    from utils.customer_list_extractor import extract_customers_from_histories, extract_customers_from_data_directories
                    
                    customer_histories = []
                    
                    # 고객 정보
                    customer_id = customer.get('customer_id', '')
                    customer_name = customer.get('customer_name', '')
                    customer_phone = customer.get('phone', '')
                    customer_email = customer.get('email', '')
                    
                    # 1. 시뮬레이션 이력에서 검색 (고객 ID 우선, 그 다음 고객 정보로 매칭)
                    try:
                        histories = load_simulation_histories_local(current_lang)
                        for history in histories:
                            # 다양한 형식에서 고객 정보 추출
                            history_customer_id = history.get('customer_id', '')
                            history_customer_name = history.get('customer_name', '') or history.get('summary', {}).get('customer_name', '')
                            history_customer_phone = history.get('customer_phone', '') or history.get('phone', '')
                            history_customer_email = history.get('customer_email', '') or history.get('email', '')
                            
                            # 고객 ID 매칭 (우선순위)
                            id_match = customer_id and history_customer_id and (customer_id.upper() == history_customer_id.upper())
                            
                            # 고객 정보 매칭 (이름, 전화번호, 이메일 중 하나라도 일치하면)
                            name_match = customer_name and history_customer_name and (customer_name.lower() in history_customer_name.lower() or history_customer_name.lower() in customer_name.lower())
                            phone_match = customer_phone and history_customer_phone and (customer_phone in history_customer_phone or history_customer_phone in customer_phone)
                            email_match = customer_email and history_customer_email and (customer_email.lower() in history_customer_email.lower() or history_customer_email.lower() in customer_email.lower())
                            
                            # 고객 ID가 있으면 ID 우선 매칭, 없으면 다른 정보로 매칭
                            if id_match or (not customer_id and (name_match or phone_match or email_match)):
                                customer_histories.append(history)
                    except Exception as e:
                        pass  # 오류 발생해도 계속 진행
                    
                    # 2. 로컬 폴더에서 직접 JSON 파일 검색
                    try:
                        history_dir = os.path.join(base_dir, "customer data histories via streamlits")
                        if os.path.exists(history_dir):
                            for root, dirs, files in os.walk(history_dir):
                                for file in files:
                                    if not file.endswith('.json'):
                                        continue
                                    
                                    file_path = os.path.join(root, file)
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            history_data = json.load(f)
                                        
                                        # 고객 정보 추출
                                        history_customer_id = ''
                                        history_customer_name = ''
                                        history_customer_phone = ''
                                        history_customer_email = ''
                                        
                                        # 다양한 형식 지원
                                        if isinstance(history_data, dict):
                                            # 형식 1: basic_info가 있는 경우
                                            if 'basic_info' in history_data:
                                                basic_info = history_data.get('basic_info', {})
                                                history_customer_id = basic_info.get('customer_id', '')
                                                history_customer_name = basic_info.get('customer_name', '')
                                                history_customer_phone = basic_info.get('phone', '')
                                                history_customer_email = basic_info.get('email', '')
                                            # 형식 2: 직접 고객 정보가 있는 경우
                                            else:
                                                history_customer_id = history_data.get('customer_id', '')
                                                history_customer_name = history_data.get('customer_name', '') or history_data.get('summary', {}).get('customer_name', '')
                                                history_customer_phone = history_data.get('customer_phone', '') or history_data.get('phone', '')
                                                history_customer_email = history_data.get('customer_email', '') or history_data.get('email', '')
                                        
                                        # 고객 정보 매칭 (고객 ID 우선)
                                        history_customer_id = history_data.get('customer_id', '')
                                        id_match = customer_id and history_customer_id and (customer_id.upper() == history_customer_id.upper())
                                        name_match = customer_name and history_customer_name and (customer_name.lower() in history_customer_name.lower() or history_customer_name.lower() in customer_name.lower())
                                        phone_match = customer_phone and history_customer_phone and (customer_phone in history_customer_phone or history_customer_phone in customer_phone)
                                        email_match = customer_email and history_customer_email and (customer_email.lower() in history_customer_email.lower() or history_customer_email.lower() in customer_email.lower())
                                        
                                        if id_match or (not customer_id and (name_match or phone_match or email_match)):
                                            # 이력 형식으로 변환
                                            history_item = {
                                                'timestamp': history_data.get('timestamp', history_data.get('date', '')),
                                                'customer_inquiry': history_data.get('initial_query', history_data.get('customer_inquiry', '')),
                                                'summary': history_data.get('summary', {}),
                                                'messages': history_data.get('messages', []),
                                                'source_file': file_path
                                            }
                                            customer_histories.append(history_item)
                                    except Exception as e:
                                        continue  # 파일 읽기 오류는 무시하고 계속 진행
                    except Exception as e:
                        pass  # 오류 발생해도 계속 진행
                    
                    if customer_histories:
                        st.markdown("---")
                        st.subheader(f"📋 {L.get('previous_consultation_history', '이전 응대 이력')} ({len(customer_histories)}건)")
                        
                        # 최근 이력부터 표시
                        customer_histories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                        
                        for idx, history in enumerate(customer_histories[:10]):  # 최근 10건만 표시
                            # 이력 요약 정보 추출
                            summary = history.get('summary', {})
                            if isinstance(summary, dict):
                                customer_inquiry = summary.get('customer_inquiry', history.get('initial_query', 'N/A'))
                                key_solutions = summary.get('key_solutions', [])
                                summary_text = summary.get('summary', '')
                                
                                # 다국어 요약 지원
                                current_lang_key = history.get('language_key', current_lang)
                                if current_lang_key == "en" and summary.get('summary_en'):
                                    summary_text = summary.get('summary_en', summary_text)
                                elif current_lang_key == "ja" and summary.get('summary_ja'):
                                    summary_text = summary.get('summary_ja', summary_text)
                                elif summary.get('summary_ko'):
                                    summary_text = summary.get('summary_ko', summary_text)
                            else:
                                customer_inquiry = history.get('initial_query', history.get('customer_inquiry', 'N/A'))
                                key_solutions = []
                                summary_text = str(summary) if summary else ''
                            
                            timestamp = history.get('timestamp', 'N/A')
                            inquiry_display = customer_inquiry[:50] + "..." if len(customer_inquiry) > 50 and customer_inquiry != 'N/A' else customer_inquiry
                            
                            with st.expander(f"{idx+1}. {timestamp} - {inquiry_display}"):
                                st.markdown(f"**{L.get('timestamp', '일시')}:** {timestamp}")
                                st.markdown(f"**{L.get('customer_inquiry', '고객 문의')}:** {customer_inquiry}")
                                
                                if key_solutions:
                                    st.markdown(f"**{L.get('key_solutions', '주요 솔루션')}:**")
                                    for sol_idx, solution in enumerate(key_solutions[:3], 1):
                                        st.markdown(f"  {sol_idx}. {solution}")
                                
                                if summary_text:
                                    st.markdown(f"**{L.get('summary', '요약')}:** {summary_text}")
                                
                                if history.get('messages'):
                                    st.markdown(f"**{L.get('messages_count', '메시지')}:** {len(history.get('messages', []))}개")
                                
                                if history.get('source_file'):
                                    st.caption(f"**{L.get('source_file', '출처 파일')}:** {os.path.basename(history.get('source_file', ''))}")
                    else:
                        st.info(L.get("no_previous_history", "이전 응대 이력이 없습니다."))
                except Exception as e:
                    st.warning(f"{L.get('history_load_error', '이력 불러오기 오류')}: {str(e)}")
        else:
            st.info(L.get("no_customers_registered", "등록된 고객이 없습니다."))
    except Exception as e:
        st.error(f"{L.get('customer_data_module_error', '고객 데이터 관리 모듈 로드 오류')}: {e}")

