"""
app.py의 홈 페이지 렌더링 로직
"""

import streamlit as st
from data_manager import load_dashboard_stats, load_customers, search_company
from ai_services import get_rag_chatbot_response, get_ai_response
from config import get_api_key
from lang_pack import LANG

def render_home_page():
    """홈 대시보드 페이지 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    st.title(L.get("dashboard_title", "📊 대시보드"))
    
    stats = load_dashboard_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=L.get("today_cs_cases", "오늘 CS 인입 케이스"), value=stats['today_cases'], delta=f"{L.get('target_label', '목표')}: {stats['daily_goal']}")
    with col2:
        st.metric(label=L.get("assigned_customers", "담당 고객 수"), value=stats['assigned_customers'])
    with col3:
        st.metric(label=L.get("consultation_goal_achievements", "상담 목표 달성 개수"), value=stats['goal_achievements'], delta=f"{stats['completion_rate']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(L.get("average_response_time", "평균 응답 시간"), stats.get('average_response_time', '2분 30초'))
    with col2:
        st.metric(L.get("customer_satisfaction", "고객 만족도"), f"{stats.get('customer_satisfaction', 4.5):.1f} / 5.0")
    
    st.divider()
    st.markdown(f"## 🛠️ {L.get('key_features', '주요 기능')}")
    
    func_col1, func_col2, func_col3, func_col4 = st.columns(4)
    with func_col1:
        if st.button(f"🏢 {L.get('company_info_faq', '회사 정보 및 FAQ')}", use_container_width=True, key="home_company_info"):
            st.session_state.show_home_company_info = True
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
    with func_col2:
        if st.button(f"📊 {L.get('lstm_score_analysis', 'LSTM 점수 분석')}", use_container_width=True, key="home_lstm"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = True
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
    with func_col3:
        if st.button(f"✨ {L.get('custom_content_generation', '맞춤형 콘텐츠 생성')}", use_container_width=True, key="home_content"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = True
            st.session_state.show_home_rag = False
    with func_col4:
        if st.button(f"🔍 {L.get('rag_chatbot', 'RAG 챗봇')}", use_container_width=True, key="home_rag"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = True
    
    st.divider()
    
    # 회사 정보 및 FAQ
    if st.session_state.get('show_home_company_info', False):
        with st.expander(f"🏢 {L.get('company_info_faq', '회사 정보 및 FAQ')}", expanded=True):
            col_search_input, col_search_btn = st.columns([4, 1])
            with col_search_input:
                search_query = st.text_input(L.get("search_query_input", "검색어 입력:"), key="home_company_search", 
                                           placeholder=L.get("search_placeholder", "회사명, 업종, 서비스 등으로 검색..."), 
                                           label_visibility="visible", 
                                           value=st.session_state.get('home_company_search_query', ''))
            with col_search_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                search_clicked = st.button(L.get("search_button", "🔍 검색"), key="home_company_search_btn", use_container_width=True)
            
            if search_clicked:
                st.session_state.home_company_search_query = search_query
                if search_query:
                    try:
                        results = search_company(search_query)
                        st.session_state.home_company_search_results = results
                        if not results:
                            st.info(L.get("no_search_results", "검색 결과가 없습니다."))
                    except Exception as e:
                        st.error(L.get("search_error", "검색 중 오류가 발생했습니다: {error}").format(error=str(e)))
                else:
                    st.warning(L.get("enter_search_query", "검색어를 입력해주세요."))
                    st.session_state.home_company_search_results = None
            
            if st.session_state.get('home_company_search_results') is not None:
                results = st.session_state.home_company_search_results
                if results:
                    st.markdown(f"**{L.get('search_results', '검색 결과: {count}개').format(count=len(results))}**")
                    for company in results[:5]:
                        with st.expander(f"🏢 {company.get('company_name', 'N/A')}", expanded=False):
                            st.markdown(f"**{L.get('industry_label', '업종:')}** {company.get('industry', 'N/A')}")
                            st.markdown(f"**{L.get('description_label', '설명:')}** {company.get('description', 'N/A')}")
                            company_query = st.text_input(L.get("question_label", "질문:"), 
                                                         key=f"home_company_query_{company.get('company_id', 'unknown')}", 
                                                         placeholder=L.get("question_placeholder", "이 회사에 대해 질문하세요..."))
                            if st.button(L.get("ask_button", "질문하기"), key=f"home_ask_company_{company.get('company_id', 'unknown')}"):
                                context = [f"회사명: {company.get('company_name', '')}", 
                                          f"업종: {company.get('industry', '')}", 
                                          f"설명: {company.get('description', '')}"]
                                response = get_rag_chatbot_response(company_query, context)
                                st.info(f"🤖 {response}")
            if st.button(L.get("close_button", "닫기"), key="close_home_company_info"):
                st.session_state.show_home_company_info = False
    
    # LSTM 점수 분석
    if st.session_state.get('show_home_lstm', False):
        with st.expander(f"📊 {L.get('lstm_score_analysis', 'LSTM 점수 분석')}", expanded=True):
            if st.session_state.selected_customer_id:
                customer = next((c for c in load_customers() if c['customer_id'] == st.session_state.selected_customer_id), None)
                if customer:
                    st.markdown(f"**{L.get('customer_label', '고객:')}** {customer['customer_name']}")
                    st.markdown(f"**{L.get('lstm_sentiment_score', 'LSTM 감정 점수'):}** 0.75 ({L.get('positive', '긍정적')})")
                    st.markdown(f"**{L.get('intent_prediction', '의도 예측'):}** {L.get('package_inquiry', '패키지 문의')} ({L.get('confidence', '신뢰도')}: 0.82)")
                else:
                    st.info(L.get("cannot_load_customer_info", "고객 정보를 불러올 수 없습니다."))
            else:
                st.info(L.get("select_customer_for_lstm", "고객을 선택하면 LSTM 분석 결과를 확인할 수 있습니다."))
            if st.button(L.get("close_button", "닫기"), key="close_home_lstm"):
                st.session_state.show_home_lstm = False
    
    # 맞춤형 콘텐츠 생성
    if st.session_state.get('show_home_content', False):
        with st.expander(f"✨ {L.get('custom_content_generation', '맞춤형 콘텐츠 생성')}", expanded=True):
            try:
                from _pages._content import render_content
                content_type_options = [
                    L.get("content_type_email", "이메일"),
                    L.get("content_type_notice", "안내문"),
                    L.get("content_type_proposal", "제안서"),
                    L.get("content_type_template", "응답 템플릿")
                ]
                content_type = st.selectbox(L.get("content_type_label", "콘텐츠 유형:"), content_type_options, key="home_content_type")
                content_topic = st.text_input(L.get("topic_label", "주제:"), key="home_content_topic", placeholder=L.get("topic_placeholder", "콘텐츠 주제를 입력하세요..."))
                if st.button(L.get("generate_button", "생성"), key="home_generate_content"):
                    api_key = get_api_key("openai") or get_api_key("gemini")
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
                                st.text_area("생성된 콘텐츠:", value=response.content, height=200, key="home_generated_content")
                        except Exception as e:
                            st.error(f"콘텐츠 생성 중 오류: {str(e)}")
                    else:
                        st.warning(L.get("enter_topic_and_api_key", "주제를 입력하고 API 키를 설정해주세요."))
                if st.button(L.get("close_button", "닫기"), key="close_home_content"):
                    st.session_state.show_home_content = False
            except ImportError:
                st.info(L.get("content_generation_error", "콘텐츠 생성 기능을 사용할 수 없습니다."))
                if st.button(L.get("close_button", "닫기"), key="close_home_content"):
                    st.session_state.show_home_content = False
    
    # RAG 챗봇
    if st.session_state.get('show_home_rag', False):
        with st.expander(f"🔍 {L.get('rag_chatbot', 'RAG 챗봇')}", expanded=True):
            rag_query = st.text_input(L.get("question_label", "질문:"), key="home_rag_query", placeholder=L.get("question_placeholder", "질문을 입력하세요..."))
            if st.button(L.get("ask_button", "질문하기"), key="ask_home_rag"):
                if rag_query:
                    response = get_rag_chatbot_response(rag_query)
                    st.info(f"🤖 {response}")
            if st.button(L.get("close_button", "닫기"), key="close_home_rag"):
                st.session_state.show_home_rag = False

