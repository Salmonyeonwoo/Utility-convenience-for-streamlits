# ========================================
# _pages/_chat_simulator_panels.py
# 채팅 시뮬레이터의 패널 렌더링 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.history_handler import get_daily_data_statistics


def _render_customer_list_panel(L, current_lang):
    """고객 목록 패널 렌더링 (col1)"""
    st.subheader(L.get("customer_list", "고객 목록"))
    
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button[kind="primary"] {
        border: 2px solid #FF69B4;
        background-color: #FFFFFF;
        color: #333;
        font-weight: 600;
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
    
    try:
        from _pages._chat_file_loader import render_file_loader_panel
        render_file_loader_panel(L, current_lang)
    except ImportError:
        pass
    
    try:
        from _pages._chat_customer_list import render_customer_list_display
        render_customer_list_display(L, current_lang)
    except ImportError:
        pass


def _render_customer_info_panel(L, current_lang):
    """고객 정보 패널 렌더링 (col3) - flat/nested 데이터 모두 완벽 지원"""
    st.subheader(L.get("customer_info", "고객 정보"))
    
    customer_data = st.session_state.get("customer_data", None)
    
    if customer_data:
        customer_info = customer_data.get("data", {})
        basic_info = customer_data.get("basic_info", {})
        crm_profile = customer_info.get("crm_profile", {})
        
        # 고객 이름
        customer_name = (
            customer_data.get('customer_name') or 
            basic_info.get('customer_name') or 
            customer_info.get('name') or 
            st.session_state.get('customer_name', '')
        )
        default_label = L.get('customer_label', '고객')
        if not customer_name:
            customer_name = default_label
        
        st.markdown(f"### 👤 {customer_name}")
        
        # ID, 연락처, 이메일
        customer_id = customer_data.get('customer_id') or basic_info.get("customer_id") or st.session_state.get('customer_id', 'N/A')
        email = customer_data.get('email') or customer_info.get('email') or basic_info.get('email') or st.session_state.get('customer_email', 'N/A')
        phone = customer_data.get('phone') or customer_info.get('phone') or basic_info.get('phone') or st.session_state.get('customer_phone', 'N/A')
        
        st.markdown(f"**{L.get('customer_id_label', '고객 ID')}:** `{customer_id}`")
        st.markdown(f"**{L.get('contact_label', '연락처')}:** {phone}")
        st.markdown(f"**{L.get('email_label', '이메일')}:** {email}")
        
        # 계정 생성일 / 마지막 접속일 / 마지막 상담
        account_created = customer_data.get('account_created') or basic_info.get('account_created')
        if account_created:
            st.markdown(f"**계정 생성일:** {account_created}")
        
        last_login = customer_data.get('last_login') or basic_info.get('last_login')
        if last_login:
            st.markdown(f"**마지막 접속일:** {last_login}")
            
        last_consultation = customer_data.get('last_consultation') or basic_info.get('last_consultation')
        if last_consultation:
            st.markdown(f"**마지막 상담일자:** {last_consultation}")
        
        # 성향
        personality = customer_data.get('personality') or crm_profile.get('personality') or basic_info.get('personality', '일반')
        st.markdown(f"**{L.get('personality_label', '성향')}:** {personality}")
        
        # 성향 요약
        personality_summary = customer_data.get('personality_summary') or crm_profile.get('personality_summary', '')
        if personality_summary:
            st.markdown("**고객 성향 요약:**")
            st.info(personality_summary)
        
        # 점수 메트릭
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            survey_score = float(customer_data.get('survey_score') or crm_profile.get('survey_score', 4.2))
            st.metric(L.get("survey_score_label", "설문 점수"), f"{survey_score:.1f} / 5.0")
        with col_m2:
            service_rating = float(customer_data.get('service_rating') or crm_profile.get('service_rating', 4.5))
            st.metric("응대 평가 점수", f"{service_rating:.1f} / 5.0")
            
    else:
        # customer_data가 아직 없을 때 세션 상태의 기본 입력 표시
        c_name = st.session_state.get('customer_name', '')
        c_email = st.session_state.get('customer_email', '')
        c_phone = st.session_state.get('customer_phone', '')
        
        if c_name or c_email or c_phone:
            customer_display_name = c_name if c_name else L.get('customer_label', '고객')
            st.markdown(f"### 👤 {customer_display_name}")
            if c_name:
                st.markdown(f"**{L.get('name_label', '성함')}:** {c_name}")
            if c_phone:
                st.markdown(f"**{L.get('contact_label', '연락처')}:** {c_phone}")
            if c_email:
                st.markdown(f"**{L.get('email_label', '이메일')}:** {c_email}")
            if st.session_state.get('customer_type_sim_select'):
                st.markdown(f"**{L.get('customer_type_label', '고객 유형')}:** {st.session_state.customer_type_sim_select}")
        else:
            st.info(L.get("customer_info_preview_placeholder", "왼쪽에서 고객을 선택하거나 정보를 입력하면 여기에 상세 정보가 표시됩니다."))
