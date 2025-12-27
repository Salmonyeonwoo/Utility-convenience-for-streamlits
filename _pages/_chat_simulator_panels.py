# ========================================
# _pages/_chat_simulator_panels.py
# 채팅 시뮬레이터의 패널 렌더링 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from utils.history_handler import get_daily_data_statistics


def _render_customer_list_panel(L, current_lang):
    """고객 목록 패널 렌더링 (col1) - 스크린샷 스타일 + 파일 자동 로드"""
    st.subheader(L.get("customer_list", "고객 목록"))
    
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
    
    # 파일 로더 패널
    try:
        from _pages._chat_file_loader import render_file_loader_panel
        render_file_loader_panel(L, current_lang)
    except ImportError:
        pass
    
    # 고객 목록 표시
    try:
        from _pages._chat_customer_list import render_customer_list_display
        render_customer_list_display(L, current_lang)
    except ImportError:
        pass


def _render_customer_info_panel(L, current_lang):
    """고객 정보 패널 렌더링 (col3) - app.py 스타일로 간소화"""
    st.subheader(L.get("customer_info", "고객 정보"))
    
    customer_data = st.session_state.get("customer_data", None)
    
    if customer_data:
        customer_info = customer_data.get("data", {})
        basic_info = customer_data.get("basic_info", {})
        
        # 고객 이름 추출 (여러 소스에서 시도)
        customer_name = (
            basic_info.get('customer_name', '') or 
            customer_info.get('name', '') or 
            st.session_state.get('customer_name', '')
        )
        
        # 고객 이름이 없거나 기본 라벨과 같은 경우에만 기본값 사용
        default_label = L.get('customer_label', '고객')
        if not customer_name or customer_name == default_label:
            customer_name = default_label
        
        st.markdown(f"### 👤 {customer_name}")
        
        customer_id = basic_info.get("customer_id", "N/A")
        email = customer_info.get('email', st.session_state.get('customer_email', 'N/A'))
        phone = customer_info.get('phone', st.session_state.get('customer_phone', 'N/A'))
        
        st.markdown(f"**{L.get('customer_id_label', '고객 ID')}:** {customer_id}")
        # 고객 이름이 기본 라벨이 아닌 실제 이름인 경우에만 표시
        if customer_name and customer_name != default_label:
            st.markdown(f"**{L.get('name_label', '성함')}:** {customer_name}")
        st.markdown(f"**{L.get('contact_label', '연락처')}:** {phone}")
        st.markdown(f"**{L.get('email_label', '이메일')}:** {email}")
        
        crm_profile = customer_info.get("crm_profile", {})
        if crm_profile:
            personality = crm_profile.get('personality', 'N/A')
            st.markdown(f"**{L.get('personality_label', '성향')}:** {personality}")
            
            survey_score = crm_profile.get('survey_score', 4.5)
            st.metric(L.get("survey_score_label", "설문 점수"), f"{survey_score:.1f} / 5.0")
    else:
        initial_query_msg = None
        for msg in st.session_state.get("simulator_messages", []):
            if msg.get("role") == "initial_query" or msg.get("role") == "customer":
                initial_query_msg = msg
                break
        
        if st.session_state.get('customer_name') or st.session_state.get('customer_email') or st.session_state.get('customer_phone'):
            # 실제 고객 이름이 있는지 확인
            customer_display_name = st.session_state.get('customer_name', '')
            default_label = L.get('customer_label', '고객')
            if not customer_display_name:
                customer_display_name = default_label
            st.markdown(f"### 👤 {customer_display_name}")
            if st.session_state.get('customer_name'):
                st.markdown(f"**{L.get('name_label', '성함')}:** {st.session_state.customer_name}")
            if st.session_state.get('customer_email'):
                st.markdown(f"**{L.get('email_label', '이메일')}:** {st.session_state.customer_email}")
            if st.session_state.get('customer_phone'):
                st.markdown(f"**{L.get('contact_label', '연락처')}:** {st.session_state.customer_phone}")
        elif initial_query_msg:
            st.info(L.get("click_customer_data_button", "고객 정보를 불러오려면 고객 데이터 버튼을 클릭하세요."))
        else:
            st.info(L.get("select_customer_to_view_details", "고객을 선택하면 상세 정보가 표시됩니다."))
    
    # 일일 통계를 col3 하단에 배치 (축소된 버전)
    if st.session_state.sim_stage not in ["WAIT_FIRST_QUERY", "idle"]:
        st.markdown("---")
        st.markdown(f"**📊 {L.get('daily_statistics', '일일 통계')}**")
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

