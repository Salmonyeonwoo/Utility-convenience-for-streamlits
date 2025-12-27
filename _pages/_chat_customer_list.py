# ========================================
# _pages/_chat_customer_list.py
# 채팅 시뮬레이터 - 고객 목록 추출 및 표시 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
import os


def render_customer_list_display(L, current_lang):
    """고객 목록 표시"""
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
        
        # 고객 정보 병합
        all_customers_dict = _merge_customer_sources(customers_from_files, customers_from_histories)
        
        # 고객 목록을 리스트로 변환하고 정렬
        all_customers_list = list(all_customers_dict.values())
        all_customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
        
        # 고객 목록 표시
        _display_customer_list(L, all_customers_list)
    except ImportError as e:
        st.info(f"{L.get('cannot_load_customer_extractor', '고객 목록 추출 모듈을 불러올 수 없습니다')}: {e}")
    except Exception as e:
        st.info(f"{L.get('cannot_load_customer_list', '고객 목록을 불러올 수 없습니다')}: {e}")


def _merge_customer_sources(customers_from_files, customers_from_histories):
    """여러 소스의 고객 정보 병합"""
    all_customers_dict = {}
    
    for customer in customers_from_files:
        name = customer.get('customer_name', '')
        if name:
            if name not in all_customers_dict:
                all_customers_dict[name] = customer
            else:
                all_customers_dict[name]['consultation_count'] += customer.get('consultation_count', 0)
    
    for customer in customers_from_histories:
        name = customer.get('customer_name', '')
        if name:
            if name not in all_customers_dict:
                all_customers_dict[name] = customer
            else:
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
                        consultation_history = customer.get("data", {}).get("consultation_history", [])
                        if consultation_history:
                            all_customers_dict[customer_name]['consultation_count'] += len(consultation_history)
    except Exception:
        pass
    
    return all_customers_dict


def _display_customer_list(L, all_customers_list):
    """고객 목록 화면에 표시"""
    # 현재 선택된 고객 확인
    current_customer_name = None
    if st.session_state.get("customer_data"):
        basic_info = st.session_state.customer_data.get("basic_info", {})
        current_customer_name = basic_info.get("customer_name", "")
    if not current_customer_name:
        current_customer_name = st.session_state.get('customer_name', '')
    
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
            customer_name = customer.get('customer_name', L.get('customer_label', '고객'))
            consultation_count = customer.get('consultation_count', 0)
            is_selected = current_customer_name == customer_name
            
            # 고객 이름과 배지를 한 줄에 표시
            col_name, col_badge = st.columns([4, 1])
            
            with col_name:
                if st.button(f"👤 {customer_name}", 
                           key=f"customer_list_{customer_name}_{st.session_state.sim_instance_id}",
                           use_container_width=True, 
                           type="primary" if is_selected else "secondary"):
                    _select_customer(customer_name, customer)
            
            with col_badge:
                if consultation_count > 0:
                    st.markdown(f'<div style="text-align: center; margin-top: 8px;"><span class="customer-badge">{consultation_count}{L.get("items", "개")}</span></div>', unsafe_allow_html=True)
    else:
        st.info(L.get("no_customers_registered", "등록된 고객이 없습니다."))


def _select_customer(customer_name, customer):
    """고객 선택 처리"""
    customer_data = customer.get('customer_data', {})
    if customer_data:
        st.session_state.customer_data = customer_data
    else:
        st.session_state.customer_data = {
            "basic_info": {
                "customer_name": customer_name,
                "customer_id": customer.get('customer_id', '')
            },
            "data": {}
        }
    st.session_state.customer_name = customer_name


