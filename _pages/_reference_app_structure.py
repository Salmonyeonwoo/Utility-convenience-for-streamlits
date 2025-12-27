"""
참고용 app.py의 전체 구조 (홈, 사이드바, 채팅, 전화, 고객 데이터)
"""
import streamlit as st
import json
from datetime import datetime
from data_manager import (
    load_customers, load_chats, load_dashboard_stats, load_calls, load_auto_responses,
    load_rag_analysis, load_company_info, search_company,
    save_chats, save_customers, save_dashboard_stats, save_calls, save_rag_analysis
)
from ai_services import (
    get_rag_chatbot_response, perform_rag_analysis, get_ai_response,
    translate_text, summarize_conversation, transfer_to_language_team
)
from config import get_api_key


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


def render_operator_sidebar():
    """상담원 사이드바 렌더링"""
    with st.sidebar:
        st.title("💬 AI 고객응대 시뮬레이터")
        
        # 언어 선택
        st.markdown("### 🌐 언어 선택")
        selected_language = st.radio(
            "언어:",
            ["한국어", "English", "日本語"],
            index=["한국어", "English", "日本語"].index(
                {"ko": "한국어", "en": "English", "ja": "日本語"}.get(st.session_state.get('language', 'ko'), "한국어")
            ),
            key="language_select"
        )
        
        lang_map = {"한국어": "ko", "English": "en", "日本語": "ja"}
        if lang_map[selected_language] != st.session_state.get('language', 'ko'):
            st.session_state.language = lang_map[selected_language]
        
        st.divider()
        
        # 네비게이션
        if st.button("🏠 홈", key="nav_home", use_container_width=True):
            st.session_state.current_page = 'home'
        
        if st.button("💬 채팅", key="nav_chat", use_container_width=True):
            st.session_state.current_page = 'chat'
        
        if st.button("📞 전화", key="nav_call", use_container_width=True):
            st.session_state.current_page = 'call'
        
        if st.button("📋 고객 데이터", key="nav_customer_data", use_container_width=True):
            st.session_state.current_page = 'customer_data'
        
        st.divider()
        st.markdown("### 상담원 프로필")
        st.markdown("**이름:** 상담원")
        st.markdown("**상태:** 🟢 온라인")
        st.divider()
        
        if st.button("🔄 모드 변경", use_container_width=True):
            st.session_state.user_type = None
            st.session_state.current_page = None
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        st.session_state.auto_refresh = st.checkbox("🔄 자동 새로고침", value=st.session_state.auto_refresh)


def render_home_page():
    """홈 대시보드 페이지 렌더링"""
    st.title("📊 대시보드")
    
    stats = load_dashboard_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="오늘 CS 인입 케이스", value=stats['today_cases'], delta=f"목표: {stats['daily_goal']}")
    with col2:
        st.metric(label="담당 고객 수", value=stats['assigned_customers'])
    with col3:
        st.metric(label="상담 목표 달성 개수", value=stats['goal_achievements'], delta=f"{stats['completion_rate']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("평균 응답 시간", stats.get('average_response_time', '2분 30초'))
    with col2:
        st.metric("고객 만족도", f"{stats.get('customer_satisfaction', 4.5):.1f} / 5.0")
    
    st.divider()
    st.markdown("## 🛠️ 주요 기능")
    
    func_col1, func_col2, func_col3, func_col4 = st.columns(4)
    with func_col1:
        if st.button("🏢 회사 정보 및 FAQ", use_container_width=True, key="home_company_info"):
            st.session_state.show_home_company_info = True
    with func_col2:
        if st.button("📊 LSTM 점수 분석", use_container_width=True, key="home_lstm"):
            st.session_state.show_home_lstm = True
    with func_col3:
        if st.button("✨ 맞춤형 콘텐츠 생성", use_container_width=True, key="home_content"):
            st.session_state.show_home_content = True
    with func_col4:
        if st.button("🔍 RAG 챗봇", use_container_width=True, key="home_rag"):
            st.session_state.show_home_rag = True
    
    st.divider()
    
    # 회사 정보 및 FAQ
    if st.session_state.get('show_home_company_info', False):
        with st.expander("🏢 회사 정보 및 FAQ", expanded=True):
            search_query = st.text_input("검색어 입력:", key="home_company_search", placeholder="회사명, 업종, 서비스 등으로 검색...")
            if search_query:
                results = search_company(search_query)
                if results:
                    st.markdown(f"**검색 결과: {len(results)}개**")
                    for company in results[:5]:
                        with st.expander(f"🏢 {company['company_name']}", expanded=False):
                            st.markdown(f"**업종:** {company.get('industry', 'N/A')}")
                            st.markdown(f"**설명:** {company.get('description', 'N/A')}")
                            company_query = st.text_input("질문:", key=f"home_company_query_{company['company_id']}", placeholder="이 회사에 대해 질문하세요...")
                            if st.button("질문하기", key=f"home_ask_company_{company['company_id']}"):
                                context = [f"회사명: {company['company_name']}", f"업종: {company.get('industry', '')}", f"설명: {company.get('description', '')}"]
                                response = get_rag_chatbot_response(company_query, context)
                                st.info(f"🤖 {response}")
            if st.button("닫기", key="close_home_company_info"):
                st.session_state.show_home_company_info = False
    
    # LSTM 점수 분석
    if st.session_state.get('show_home_lstm', False):
        with st.expander("📊 LSTM 점수 분석", expanded=True):
            if st.session_state.get('selected_customer_id'):
                customer = next((c for c in load_customers() if c['customer_id'] == st.session_state.selected_customer_id), None)
                if customer:
                    st.markdown(f"**고객:** {customer['customer_name']}")
                    st.markdown(f"**LSTM 감정 점수:** 0.75 (긍정적)")
                    st.markdown(f"**의도 예측:** 패키지 문의 (신뢰도: 0.82)")
            else:
                st.info("고객을 선택하면 LSTM 분석 결과를 확인할 수 있습니다.")
            if st.button("닫기", key="close_home_lstm"):
                st.session_state.show_home_lstm = False
    
    # 맞춤형 콘텐츠 생성
    if st.session_state.get('show_home_content', False):
        with st.expander("✨ 맞춤형 콘텐츠 생성", expanded=True):
            content_type = st.selectbox("콘텐츠 유형:", ["이메일", "안내문", "제안서", "응답 템플릿"], key="content_type")
            content_topic = st.text_input("주제:", key="content_topic", placeholder="콘텐츠 주제를 입력하세요...")
            if st.button("생성", key="generate_content"):
                api_key = get_api_key("openai")
                if content_topic and api_key:
                    try:
                        from langchain_openai import ChatOpenAI
                        try:
                            from langchain.schema import HumanMessage
                        except ImportError:
                            from langchain_core.messages import HumanMessage
                        with st.spinner("콘텐츠 생성 중..."):
                            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
                            prompt = f"""여행사 상담원을 위한 {content_type}를 작성해주세요.\n\n주제: {content_topic}\n\n친절하고 전문적인 톤으로 작성해주세요."""
                            response = llm.invoke([HumanMessage(content=prompt)])
                            st.text_area("생성된 콘텐츠:", value=response.content, height=200, key="generated_content")
                    except Exception as e:
                        st.error(f"콘텐츠 생성 중 오류: {str(e)}")
                else:
                    st.warning("주제를 입력하고 OpenAI API 키를 설정해주세요.")
            if st.button("닫기", key="close_home_content"):
                st.session_state.show_home_content = False
    
    # RAG 챗봇
    if st.session_state.get('show_home_rag', False):
        with st.expander("🔍 RAG 챗봇", expanded=True):
            rag_query = st.text_input("질문:", key="home_rag_query", placeholder="질문을 입력하세요...")
            if st.button("질문하기", key="ask_home_rag"):
                if rag_query:
                    response = get_rag_chatbot_response(rag_query)
                    st.info(f"🤖 {response}")
            if st.button("닫기", key="close_home_rag"):
                st.session_state.show_home_rag = False


def render_chat_page():
    """채팅 페이지 렌더링 (참고용 app.py와 동일)"""
    customers = load_customers()
    chats = load_chats()
    
    col1, col2, col3 = st.columns([1, 2, 1.5])
    
    # 고객 리스트
    with col1:
        st.subheader("고객 목록")
        unread_counts = {}
        for customer in customers:
            customer_id = customer['customer_id']
            if customer_id in chats:
                customer_messages = [msg for msg in chats[customer_id] if msg['sender'] == 'customer']
                unread_counts[customer_id] = len(customer_messages)
        
        for customer in customers:
            customer_id = customer['customer_id']
            is_selected = st.session_state.get('selected_customer_id') == customer_id
            
            if st.button(f"👤 {customer['customer_name']}", key=f"customer_{customer_id}", 
                        use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.selected_customer_id = customer_id
            
            if customer_id in unread_counts and unread_counts[customer_id] > 0:
                st.caption(f"🔴 {unread_counts[customer_id]}개")
    
    # 채팅 화면
    with col2:
        if st.session_state.get('selected_customer_id'):
            selected_customer = next((c for c in customers if c['customer_id'] == st.session_state.selected_customer_id), None)
            
            if selected_customer:
                st.subheader(f"💬 {selected_customer['customer_name']}님과의 대화")
                
                customer_id = selected_customer['customer_id']
                if customer_id not in chats:
                    chats[customer_id] = []
                
                current_chats = chats[customer_id]
                last_msg_id = st.session_state.get('last_message_id', {}).get(customer_id, "")
                
                # AI 응답 생성
                if current_chats:
                    last_msg = current_chats[-1]
                    current_last_id = last_msg.get('message_id', '')
                    api_key_auto = get_api_key("openai") or get_api_key("gemini")
                    if (last_msg['sender'] == 'customer' and current_last_id != last_msg_id and api_key_auto):
                        if f'ai_processing_{customer_id}' not in st.session_state:
                            st.session_state[f'ai_processing_{customer_id}'] = True
                            try:
                                ai_response = get_ai_response(last_msg['message'], selected_customer, current_chats)
                                st.session_state.ai_suggestion = {
                                    'customer_id': customer_id, 'message': ai_response,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            except Exception as e:
                                st.session_state.ai_suggestion = {
                                    'customer_id': customer_id, 'message': f"오류: {str(e)}",
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            finally:
                                st.session_state[f'ai_processing_{customer_id}'] = False
                        if 'last_message_id' not in st.session_state:
                            st.session_state.last_message_id = {}
                        st.session_state.last_message_id[customer_id] = current_last_id
                
                # 메시지 표시
                chat_container = st.container(height=400)
                with chat_container:
                    for msg in current_chats:
                        sender_class = "message-operator" if msg['sender'] == 'operator' else "message-customer"
                        st.markdown(f"""
                        <div class="{sender_class}">
                            <strong>{msg['sender_name']}</strong><br>
                            {msg['message']}<br>
                            <small style="color: #666;">{msg['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI 제안 표시
                    if (current_chats and current_chats[-1]['sender'] == 'customer' and
                        st.session_state.get('ai_suggestion', {}).get('customer_id') == customer_id):
                        ai_suggestion = st.session_state.ai_suggestion
                        st.markdown(f"""
                        <div class="message-ai-suggestion">
                            <strong>🤖 AI 제안 응답</strong><br>
                            {ai_suggestion['message']}<br>
                            <small style="color: #666;">{ai_suggestion['timestamp']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("✅ 사용", key=f"use_ai_{customer_id}", use_container_width=True):
                            new_message = {
                                "message_id": f"MSG{len(current_chats) + 1:03d}",
                                "sender": "operator", "sender_name": "상담원",
                                "message": ai_suggestion['message'],
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            chats[customer_id].append(new_message)
                            save_chats(chats)
                            st.session_state.ai_suggestion = {}
                    
                    if st.session_state.get(f'ai_processing_{customer_id}', False):
                        st.info("🤖 AI가 응답을 생성하는 중...")
                
                st.divider()
                
                # 입력 영역
                chat_input = st.text_input("메시지 입력", key=f"chat_input_{customer_id}", placeholder="메시지를 입력하세요...", label_visibility="collapsed")
                if st.button("전송", type="primary", use_container_width=True, key=f"send_{customer_id}"):
                    if chat_input:
                        new_message = {
                            "message_id": f"MSG{len(chats[customer_id]) + 1:03d}",
                            "sender": "operator", "sender_name": "상담원",
                            "message": chat_input, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        chats[customer_id].append(new_message)
                        save_chats(chats)
                        stats = load_dashboard_stats()
                        stats['today_cases'] += 1
                        save_dashboard_stats(stats)
            else:
                st.info("고객을 선택해주세요.")
        else:
            st.info("왼쪽에서 고객을 선택하여 채팅을 시작하세요.")
    
    # 고객 정보
    with col3:
        if st.session_state.get('selected_customer_id'):
            selected_customer = next((c for c in customers if c['customer_id'] == st.session_state.selected_customer_id), None)
            if selected_customer:
                st.subheader("고객 정보")
                st.markdown(f"### 👤 {selected_customer['customer_name']}")
                st.markdown(f"**고객 ID:** {selected_customer['customer_id']}")
                st.markdown(f"**연락처:** {selected_customer.get('phone', 'N/A')}")
                st.markdown(f"**이메일:** {selected_customer.get('email', 'N/A')}")
                st.markdown(f"**성향:** {selected_customer.get('personality', 'N/A')}")
                st.metric("설문 점수", f"{selected_customer.get('survey_score', 0.0):.1f} / 5.0")
            else:
                st.info("고객 정보를 불러올 수 없습니다.")
        else:
            st.info("고객을 선택하면 상세 정보가 표시됩니다.")


def render_call_page():
    """전화 페이지 렌더링 (간소화 버전)"""
    st.title("📞 전화 기능")
    st.info("전화 기능은 채팅 페이지에서 사용할 수 있습니다.")


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
                    personality = st.selectbox("고객 성향", ["일반", "신중형", "활발형", "가족형", "프리미엄형", "절약형", "자유형"], key="reg_personality")
                    preferred_destination = st.text_input("선호 여행지", key="reg_destination")
                
                if st.form_submit_button("고객 등록", type="primary", use_container_width=True):
                    if customer_name and phone and email:
                        customer_data = {'customer_name': customer_name, 'phone': phone, 'email': email,
                                       'personality': personality, 'preferred_destination': preferred_destination}
                        customer_id = manager.create_customer(customer_data)
                        st.success(f"고객이 등록되었습니다! 고객 ID: {customer_id}")
                    else:
                        st.error("고객명, 연락처, 이메일은 필수 항목입니다.")
        
        with tab2:
            st.subheader("고객 정보 조회")
            customers = manager.load_all_customers()
            if customers:
                customer_options = {f"{c['customer_name']} ({c['customer_id']})": c['customer_id'] for c in customers}
                selected_customer_name = st.selectbox("고객 선택:", list(customer_options.keys()), key="select_customer_view")
                if selected_customer_name:
                    customer = manager.get_customer_by_id(customer_options[selected_customer_name])
                    if customer:
                        st.markdown(f"### 👤 {customer['customer_name']} 고객 정보")
                        st.markdown(f"**고객 ID:** {customer['customer_id']} | **연락처:** {customer.get('phone', 'N/A')} | **이메일:** {customer.get('email', 'N/A')}")
            else:
                st.info("등록된 고객이 없습니다.")
    except Exception as e:
        st.error(f"고객 데이터 관리 모듈 로드 오류: {e}")


