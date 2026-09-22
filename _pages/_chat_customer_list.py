# ========================================
# _pages/_chat_customer_list.py
# 채팅 시뮬레이터 - 고객 목록 추출 및 표시 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
import os
from utils.customer_list_extractor import is_valid_customer_name

def render_customer_list_display(L, current_lang):
    """고객 목록 표시"""
    try:
        from utils.customer_list_extractor import extract_customers_from_data_directories
        from utils.history_handler import load_simulation_histories_local
        from utils.customer_list_extractor import extract_customers_from_histories
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dirs = [
            os.path.join(base_dir, "data"),
            os.path.join(base_dir, "data", "customers"),
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\data",
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\data\customers",
            r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\customer data histories via streamlits",
        ]
        
        existing_dirs = [d for d in data_dirs if os.path.exists(d)]
        customers_from_files = extract_customers_from_data_directories(existing_dirs)
        
        histories = load_simulation_histories_local(current_lang)
        customers_from_histories = extract_customers_from_histories(histories)
        
        all_customers_dict = _merge_customer_sources(customers_from_files, customers_from_histories)
        
        all_customers_list = list(all_customers_dict.values())
        all_customers_list.sort(key=lambda x: x.get('last_consultation_date', ''), reverse=True)
        
        _display_customer_list(L, all_customers_list)
    except ImportError as e:
        st.info(f"{L.get('cannot_load_customer_extractor', '고객 목록 추출 모듈을 불러올 수 없습니다')}: {e}")
    except Exception as e:
        st.info(f"{L.get('cannot_load_customer_list', '고객 목록을 불러올 수 없습니다')}: {e}")


def _merge_customer_sources(customers_from_files, customers_from_histories):
    """여러 소스의 고객 정보 병합 (CustomerDataManager 연동 포함)"""
    all_customers_dict = {}
    
    # 1. 파일에서 추출한 고객들
    for customer in customers_from_files:
        name = customer.get('customer_name', '')
        if name and is_valid_customer_name(name):
            if name not in all_customers_dict:
                all_customers_dict[name] = customer.copy()
            else:
                all_customers_dict[name]['consultation_count'] += customer.get('consultation_count', 0)
    
    # 2. 시뮬레이션 이력에서 추출한 고객들
    for customer in customers_from_histories:
        name = customer.get('customer_name', '')
        if name and is_valid_customer_name(name):
            if name not in all_customers_dict:
                all_customers_dict[name] = customer.copy()
            else:
                all_customers_dict[name]['consultation_count'] += customer.get('consultation_count', 0)
    
    # 3. CustomerDataManager (데이터베이스 파일)에서 가져오기
    try:
        if hasattr(st.session_state, 'customer_data_manager') and st.session_state.customer_data_manager:
            manager_customers = st.session_state.customer_data_manager.load_all_customers()
            for customer in manager_customers:
                basic_info = customer.get("basic_info", {})
                customer_name = customer.get("customer_name") or basic_info.get("customer_name", "")
                customer_id = customer.get("customer_id") or basic_info.get("customer_id", "")
                phone = customer.get("phone") or basic_info.get("phone", "")
                email = customer.get("email") or basic_info.get("email", "")
                personality = customer.get("personality") or basic_info.get("personality", "")
                
                if customer_name and is_valid_customer_name(customer_name):
                    consultation_history = customer.get("consultation_history") or customer.get("data", {}).get("consultation_history", [])
                    c_count = len(consultation_history) if consultation_history else 1
                    
                    if customer_name not in all_customers_dict:
                        all_customers_dict[customer_name] = {
                            'customer_name': customer_name,
                            'customer_id': customer_id,
                            'phone': phone,
                            'email': email,
                            'personality': personality,
                            'consultation_count': c_count,
                            'last_consultation_date': customer.get('last_consultation', ''),
                            'customer_data': customer
                        }
                    else:
                        all_customers_dict[customer_name]['consultation_count'] += c_count
                        if phone: all_customers_dict[customer_name]['phone'] = phone
                        if email: all_customers_dict[customer_name]['email'] = email
                        if personality: all_customers_dict[customer_name]['personality'] = personality
                        all_customers_dict[customer_name]['customer_data'] = customer
    except Exception:
        pass
    
    return all_customers_dict


def _display_customer_list(L, all_customers_list):
    """고객 목록 화면에 표시"""
    current_customer_name = None
    if st.session_state.get("customer_data"):
        c_data = st.session_state.customer_data
        current_customer_name = c_data.get('customer_name') or c_data.get('basic_info', {}).get('customer_name', '')
    if not current_customer_name:
        current_customer_name = st.session_state.get('customer_name', '')
    
    if all_customers_list:
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
        
        for customer in all_customers_list[:25]:
            customer_name = customer.get('customer_name', L.get('customer_label', '고객'))
            consultation_count = customer.get('consultation_count', 1)
            is_selected = (current_customer_name == customer_name)
            
            col_name, col_badge = st.columns([4, 1])
            
            with col_name:
                btn_key = f"cust_list_btn_{customer_name}_{st.session_state.get('sim_instance_id', 'default')}"
                if st.button(f"👤 {customer_name}", 
                           key=btn_key,
                           use_container_width=True, 
                           type="primary" if is_selected else "secondary"):
                    _select_customer(customer_name, customer)
            
            with col_badge:
                if consultation_count > 0:
                    st.markdown(f'<div style="text-align: center; margin-top: 8px;"><span class="customer-badge">{consultation_count}{L.get("items", "개")}</span></div>', unsafe_allow_html=True)
    else:
        st.info(L.get("no_customers_registered", "등록된 고객이 없습니다."))


def _select_customer(customer_name, customer):
    """고객 선택 처리 (실시간 호출 및 세션 연동)"""
    customer_data = customer.get('customer_data') or {}
    st.session_state.customer_data = customer_data
    st.session_state.customer_name = customer_name
    
    c_id = customer.get('customer_id') or customer_data.get('customer_id', '')
    phone = customer.get('phone') or customer_data.get('phone') or customer_data.get('basic_info', {}).get('phone', '')
    email = customer.get('email') or customer_data.get('email') or customer_data.get('basic_info', {}).get('email', '')
    
    st.session_state.customer_id = c_id
    st.session_state.selected_customer_id = c_id
    if phone:
        st.session_state.customer_phone = phone
    if email:
        st.session_state.customer_email = email
    
    st.rerun()
