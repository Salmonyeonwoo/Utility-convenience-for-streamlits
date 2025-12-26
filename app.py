import streamlit as st
from config import init_page_config, init_session_state, get_css_styles
from data_manager import load_customers, load_chats, save_chats, save_customers
from datetime import datetime
from app_sidebar import render_operator_sidebar
from app_home import render_home_page
from app_chat import render_chat_page
from app_customer_data import render_customer_data_page

# 페이지 설정 및 초기화
init_page_config()
init_session_state()
st.markdown(get_css_styles(), unsafe_allow_html=True)

# 사용자 타입 선택 화면
def show_mode_selection():
    """모드 선택 화면 표시"""
    st.title("AI 고객응대 시뮬레이터")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👨‍💼 상담원 모드")
        st.markdown("고객과의 채팅을 관리하고 AI 응대 가이드를 받을 수 있습니다.")
        if st.button("상담원으로 접속", type="primary", use_container_width=True):
            st.session_state.user_type = "operator"
            st.session_state.current_page = "home"
    
    with col2:
        st.markdown("### 👤 고객 모드")
        st.markdown("고객으로서 상담원과 채팅할 수 있습니다.")
        if st.button("고객으로 접속", type="secondary", use_container_width=True):
            st.session_state.user_type = "customer"
            st.session_state.current_page = "chat"
    
    st.stop()

# 전화 페이지
def render_call_page():
    """전화 페이지 렌더링 (간소화 버전)"""
    st.title("📞 전화 기능")
    st.info("전화 기능은 채팅 페이지에서 사용할 수 있습니다.")

# 상담원 모드
if st.session_state.user_type == "operator":
    render_operator_sidebar()
    
    if st.session_state.current_page is None:
        st.session_state.current_page = 'home'
    
    if st.session_state.current_page == 'home':
        render_home_page()
    elif st.session_state.current_page == 'chat':
        render_chat_page()
    elif st.session_state.current_page == 'call':
        render_call_page()
    elif st.session_state.current_page == 'customer_data':
        render_customer_data_page()

# 모드 선택 화면 (user_type이 None일 때만)
elif st.session_state.user_type is None:
    show_mode_selection()

# 고객 모드
elif st.session_state.user_type == "customer":
    customers = load_customers()
    chats = load_chats()
    
    if 'customer_selected' not in st.session_state:
        st.title("고객 모드")
        customer_options = {f"{c['customer_name']} ({c['customer_id']})": c['customer_id'] for c in customers}
        selected_name = st.selectbox("고객을 선택하세요:", list(customer_options.keys()))
        if st.button("선택", type="primary"):
            st.session_state.customer_selected = customer_options[selected_name]
        if st.button("🔄 모드 변경", use_container_width=True):
            st.session_state.user_type = None
        st.stop()
    
    selected_customer = next((c for c in customers if c['customer_id'] == st.session_state.customer_selected), None)
    if selected_customer:
        st.title(f"💬 {selected_customer['customer_name']}님의 채팅")
        customer_id = selected_customer['customer_id']
        if customer_id not in chats:
            chats[customer_id] = []
        
        chat_container = st.container(height=400)
        with chat_container:
            for msg in chats[customer_id]:
                sender_class = "message-operator" if msg['sender'] == 'operator' else "message-customer"
                st.markdown(f"""<div class="{sender_class}"><strong>{msg['sender_name']}</strong><br>{msg['message']}<br><small style="color: #666;">{msg['timestamp']}</small></div>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            customer_input = st.text_input("메시지 입력", key="customer_input", placeholder="메시지를 입력하세요...", label_visibility="collapsed")
        with col2:
            if st.button("전송", type="primary", use_container_width=True) and customer_input:
                new_message = {
                    "message_id": f"MSG{len(chats[customer_id]) + 1:03d}", 
                    "sender": "customer",
                    "sender_name": selected_customer['customer_name'], 
                    "message": customer_input,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                chats[customer_id].append(new_message)
                save_chats(chats)
                selected_customer['last_login'] = datetime.now().strftime("%Y-%m-%d")
                save_customers(customers)
        
        if st.button("🔄 모드 변경", use_container_width=True):
            st.session_state.user_type = None
            st.session_state.customer_selected = None
