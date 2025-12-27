"""
app.py의 고객 데이터 관리 페이지 렌더링 로직
"""

import streamlit as st

def render_customer_data_page():
    """고객 데이터 관리 페이지 렌더링"""
    try:
        from customer_data_manager import CustomerDataManager
        st.title("📋 고객 데이터 관리")
        manager = CustomerDataManager()
        
        tab1, tab2 = st.tabs(["📝 고객 등록", "🔍 고객 조회"])
        
        with tab1:
            st.subheader("새 고객 등록")
            with st.form("customer_registration_form"):
                col1, col2 = st.columns(2)
                with col1:
                    customer_name = st.text_input("고객명 *", key="reg_customer_name")
                    phone = st.text_input("연락처 *", key="reg_phone")
                    email = st.text_input("이메일 *", key="reg_email")
                with col2:
                    personality = st.selectbox("고객 성향", 
                                             ["일반", "신중형", "활발형", "가족형", "프리미엄형", "절약형", "자유형"], 
                                             key="reg_personality")
                    preferred_destination = st.text_input("선호 여행지", key="reg_destination")
                
                if st.form_submit_button("고객 등록", type="primary", use_container_width=True):
                    if customer_name and phone and email:
                        customer_data = {
                            'customer_name': customer_name, 
                            'phone': phone, 
                            'email': email,
                            'personality': personality, 
                            'preferred_destination': preferred_destination
                        }
                        customer_id = manager.create_customer(customer_data)
                        st.success(f"고객이 등록되었습니다! 고객 ID: {customer_id}")
                    else:
                        st.error("고객명, 연락처, 이메일은 필수 항목입니다.")
        
        with tab2:
            st.subheader("고객 정보 조회")
            customers = manager.load_all_customers()
            if customers:
                customer_options = {f"{c['customer_name']} ({c['customer_id']})": c['customer_id'] for c in customers}
                selected_customer_name = st.selectbox("고객 선택:", 
                                                     list(customer_options.keys()), 
                                                     key="select_customer_view")
                if selected_customer_name:
                    customer = manager.get_customer_by_id(customer_options[selected_customer_name])
                    if customer:
                        st.markdown(f"### 👤 {customer['customer_name']} 고객 정보")
                        st.markdown(f"**고객 ID:** {customer['customer_id']} | **연락처:** {customer.get('phone', 'N/A')} | **이메일:** {customer.get('email', 'N/A')}")
            else:
                st.info("등록된 고객이 없습니다.")
    except Exception as e:
        st.error(f"고객 데이터 관리 모듈 로드 오류: {e}")




