"""
app.py의 홈 페이지 렌더링 로직
"""

import streamlit as st
from data_manager import load_dashboard_stats, load_customers, search_company
from ai_services import get_rag_chatbot_response, get_ai_response
from config import get_api_key

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
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
    with func_col2:
        if st.button("📊 LSTM 점수 분석", use_container_width=True, key="home_lstm"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = True
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
    with func_col3:
        if st.button("✨ 맞춤형 콘텐츠 생성", use_container_width=True, key="home_content"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = True
            st.session_state.show_home_rag = False
    with func_col4:
        if st.button("🔍 RAG 챗봇", use_container_width=True, key="home_rag"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = True
    
    st.divider()
    
    # 회사 정보 및 FAQ
    if st.session_state.get('show_home_company_info', False):
        with st.expander("🏢 회사 정보 및 FAQ", expanded=True):
            col_search_input, col_search_btn = st.columns([4, 1])
            with col_search_input:
                search_query = st.text_input("검색어 입력:", key="home_company_search", 
                                           placeholder="회사명, 업종, 서비스 등으로 검색...", 
                                           label_visibility="visible", 
                                           value=st.session_state.get('home_company_search_query', ''))
            with col_search_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                search_clicked = st.button("🔍 검색", key="home_company_search_btn", use_container_width=True)
            
            if search_clicked:
                st.session_state.home_company_search_query = search_query
                if search_query:
                    try:
                        results = search_company(search_query)
                        st.session_state.home_company_search_results = results
                        if not results:
                            st.info("검색 결과가 없습니다.")
                    except Exception as e:
                        st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("검색어를 입력해주세요.")
                    st.session_state.home_company_search_results = None
            
            if st.session_state.get('home_company_search_results') is not None:
                results = st.session_state.home_company_search_results
                if results:
                    st.markdown(f"**검색 결과: {len(results)}개**")
                    for company in results[:5]:
                        with st.expander(f"🏢 {company.get('company_name', 'N/A')}", expanded=False):
                            st.markdown(f"**업종:** {company.get('industry', 'N/A')}")
                            st.markdown(f"**설명:** {company.get('description', 'N/A')}")
                            company_query = st.text_input("질문:", 
                                                         key=f"home_company_query_{company.get('company_id', 'unknown')}", 
                                                         placeholder="이 회사에 대해 질문하세요...")
                            if st.button("질문하기", key=f"home_ask_company_{company.get('company_id', 'unknown')}"):
                                context = [f"회사명: {company.get('company_name', '')}", 
                                          f"업종: {company.get('industry', '')}", 
                                          f"설명: {company.get('description', '')}"]
                                response = get_rag_chatbot_response(company_query, context)
                                st.info(f"🤖 {response}")
            if st.button("닫기", key="close_home_company_info"):
                st.session_state.show_home_company_info = False
    
    # LSTM 점수 분석
    if st.session_state.get('show_home_lstm', False):
        with st.expander("📊 LSTM 점수 분석", expanded=True):
            if st.session_state.selected_customer_id:
                customer = next((c for c in load_customers() if c['customer_id'] == st.session_state.selected_customer_id), None)
                if customer:
                    st.markdown(f"**고객:** {customer['customer_name']}")
                    st.markdown(f"**LSTM 감정 점수:** 0.75 (긍정적)")
                    st.markdown(f"**의도 예측:** 패키지 문의 (신뢰도: 0.82)")
                else:
                    st.info("고객 정보를 불러올 수 없습니다.")
            else:
                st.info("고객을 선택하면 LSTM 분석 결과를 확인할 수 있습니다.")
            if st.button("닫기", key="close_home_lstm"):
                st.session_state.show_home_lstm = False
    
    # 맞춤형 콘텐츠 생성
    if st.session_state.get('show_home_content', False):
        with st.expander("✨ 맞춤형 콘텐츠 생성", expanded=True):
            try:
                from _pages._content import render_content
                content_type = st.selectbox("콘텐츠 유형:", ["이메일", "안내문", "제안서", "응답 템플릿"], key="home_content_type")
                content_topic = st.text_input("주제:", key="home_content_topic", placeholder="콘텐츠 주제를 입력하세요...")
                if st.button("생성", key="home_generate_content"):
                    api_key = get_api_key("openai") or get_api_key("gemini")
                    if content_topic and api_key:
                        try:
                            from langchain_openai import ChatOpenAI
                            from langchain.schema import HumanMessage
                            with st.spinner("콘텐츠 생성 중..."):
                                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
                                prompt = f"""여행사 상담원을 위한 {content_type}를 작성해주세요.\n\n주제: {content_topic}\n\n친절하고 전문적인 톤으로 작성해주세요."""
                                response = llm.invoke([HumanMessage(content=prompt)])
                                st.text_area("생성된 콘텐츠:", value=response.content, height=200, key="home_generated_content")
                        except Exception as e:
                            st.error(f"콘텐츠 생성 중 오류: {str(e)}")
                    else:
                        st.warning("주제를 입력하고 API 키를 설정해주세요.")
                if st.button("닫기", key="close_home_content"):
                    st.session_state.show_home_content = False
            except ImportError:
                st.info("콘텐츠 생성 기능을 사용할 수 없습니다.")
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

