"""
참고용 app.py의 홈 페이지 렌더링
"""
import streamlit as st
import json
import uuid
from data_manager import load_dashboard_stats, load_customers
from ai_services import get_rag_chatbot_response
from config import get_api_key
from data_manager import search_company
from lang_pack import LANG


def render_home_page():
    """홈 대시보드 페이지 렌더링 (참고용 app.py와 동일)"""
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
            # 다른 섹션 모두 닫기
            st.session_state.show_home_company_info = True
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
    with func_col2:
        if st.button(f"📊 {L.get('lstm_score_analysis', 'LSTM 점수 분석')}", use_container_width=True, key="home_lstm"):
            # 다른 섹션 모두 닫기
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = True
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
    with func_col3:
        if st.button(f"✨ {L.get('custom_content_generation', '맞춤형 콘텐츠 생성')}", use_container_width=True, key="home_content"):
            # 다른 섹션 모두 닫기
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = True
            st.session_state.show_home_rag = False
    with func_col4:
        if st.button(f"🔍 {L.get('rag_chatbot', 'RAG 챗봇')}", use_container_width=True, key="home_rag"):
            # 다른 섹션 모두 닫기
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
                search_query = st.text_input(L.get("search_query_input", "검색어 입력:"), key="home_company_search", placeholder=L.get("search_placeholder", "회사명, 업종, 서비스 등으로 검색..."), label_visibility="visible", value=st.session_state.get('home_company_search_query', ''))
            with col_search_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # 버튼을 입력 필드와 정렬
                search_clicked = st.button(L.get("search_button", "🔍 검색"), key="home_company_search_btn", use_container_width=True)
            
            # 검색 버튼 클릭 시 검색 수행 (GitHub 기능 활용: LLM으로 회사 정보 생성)
            if search_clicked:
                st.session_state.home_company_search_query = search_query
                if search_query:
                    try:
                        # 먼저 로컬에서 검색 시도
                        results = search_company(search_query)
                        if results:
                            st.session_state.home_company_search_results = results
                        else:
                            # 로컬 검색 결과가 없으면 LLM으로 회사 정보 생성 시도 (GitHub 기능)
                            try:
                                from faq_manager import generate_company_info_with_llm, load_faq_database, save_faq_database
                                current_lang = st.session_state.get("language", "ko")
                                if current_lang not in ["ko", "en", "ja"]:
                                    current_lang = "ko"
                                
                                api_key = get_api_key("openai") or get_api_key("gemini")
                                if api_key:
                                    with st.spinner(f"{search_query} 회사 정보 생성 중..."):
                                        generated_data = generate_company_info_with_llm(search_query, current_lang)
                                        if generated_data:
                                            # 생성된 데이터를 결과 형식에 맞게 변환
                                            company_result = {
                                                'company_name': search_query,
                                                'company_id': search_query.lower().replace(' ', '_'),
                                                'industry': generated_data.get('company_info', '').split('\n')[0] if generated_data.get('company_info') else 'N/A',
                                                'description': generated_data.get('company_info', 'N/A'),
                                                'generated_data': generated_data
                                            }
                                            st.session_state.home_company_search_results = [company_result]
                                            st.success(L.get("company_info_generated", "✅ {company} 회사 정보를 생성했습니다!").format(company=search_query))
                                            
                                            # FAQ 데이터베이스에 저장
                                            faq_data = load_faq_database()
                                            if search_query not in faq_data.get("companies", {}):
                                                faq_data.setdefault("companies", {})[search_query] = {
                                                    f"info_{current_lang}": generated_data.get("company_info", ""),
                                                    "info_ko": generated_data.get("company_info", ""),
                                                    "info_en": "",
                                                    "info_ja": "",
                                                    "popular_products": generated_data.get("popular_products", []),
                                                    "trending_topics": generated_data.get("trending_topics", []),
                                                    "faqs": generated_data.get("faqs", []),
                                                    "interview_questions": generated_data.get("interview_questions", []),
                                                    "ceo_info": generated_data.get("ceo_info", {})
                                                }
                                                save_faq_database(faq_data)
                                        else:
                                            st.info(L.get("no_search_results", "검색 결과가 없습니다."))
                                else:
                                    # API 키가 없으면 로컬 검색 결과만 표시
                                    st.session_state.home_company_search_results = []
                                    st.info(L.get("no_api_key_for_company_info", "검색 결과가 없습니다. API 키를 설정하면 LLM으로 회사 정보를 생성할 수 있습니다."))
                            except ImportError:
                                # faq_manager가 없으면 로컬 검색만 사용
                                st.session_state.home_company_search_results = []
                                st.info("검색 결과가 없습니다.")
                            except Exception as e:
                                st.error(L.get("company_info_generation_error", "회사 정보 생성 중 오류: {error}").format(error=str(e)))
                                st.session_state.home_company_search_results = []
                    except Exception as e:
                        st.error(L.get("search_error", "검색 중 오류가 발생했습니다: {error}").format(error=str(e)))
                        st.session_state.home_company_search_results = []
                else:
                    st.warning(L.get("enter_search_query", "검색어를 입력해주세요."))
                    st.session_state.home_company_search_results = None
            
            # 검색 결과 표시
            if st.session_state.get('home_company_search_results') is not None:
                results = st.session_state.home_company_search_results
                if results:
                    st.markdown(f"**{L.get('search_results', '검색 결과: {count}개').format(count=len(results))}**")
                    for company in results[:5]:  # 최대 5개만 표시
                        with st.expander(f"🏢 {company.get('company_name', 'N/A')}", expanded=False):
                            st.markdown(f"**{L.get('industry_label', '업종:')}** {company.get('industry', 'N/A')}")
                            st.markdown(f"**{L.get('description_label', '설명:')}** {company.get('description', 'N/A')}")
                            
                            # 생성된 데이터가 있으면 추가 정보 표시
                            if company.get('generated_data'):
                                gen_data = company['generated_data']
                                if gen_data.get('popular_products'):
                                    st.markdown(f"**{L.get('popular_products', '인기 제품:')}**")
                                    for product in gen_data['popular_products'][:3]:
                                        st.markdown(f"- {product}")
                                if gen_data.get('faqs'):
                                    st.markdown(f"**{L.get('faq_label', 'FAQ:')}**")
                                    for faq in gen_data['faqs'][:3]:
                                        st.markdown(f"**Q:** {faq.get('question', '')}")
                                        st.markdown(f"**A:** {faq.get('answer', '')}")
                            
                            company_query = st.text_input(L.get("question_label", "질문:"), key=f"home_company_query_{company.get('company_id', 'unknown')}", placeholder=L.get("question_placeholder", "이 회사에 대해 질문하세요..."))
                            if st.button(L.get("ask_button", "질문하기"), key=f"home_ask_company_{company.get('company_id', 'unknown')}"):
                                context = [f"회사명: {company.get('company_name', '')}", f"업종: {company.get('industry', '')}", f"설명: {company.get('description', '')}"]
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
                st.info(L.get("select_customer_for_lstm", "고객을 선택하면 LSTM 분석 결과를 확인할 수 있습니다."))
            if st.button(L.get("close_button", "닫기"), key="close_home_lstm"):
                st.session_state.show_home_lstm = False
    
    # 맞춤형 콘텐츠 생성 (GitHub _pages/_content.py 기능 활용 - 핵심 요약, 객관식 10문항, 실습 예제)
    if st.session_state.get('show_home_content', False):
        with st.expander(f"✨ {L.get('custom_content_generation', '맞춤형 콘텐츠 생성')}", expanded=True):
            # GitHub 기능 활용: 핵심 요약 노트, 객관식 퀴즈 10문항, 실습 예제 아이디어
            content_type_options = [
                L.get("content_type_summary", "핵심 요약 노트"),
                L.get("content_type_quiz", "객관식 퀴즈 10문항"),
                L.get("content_type_example", "실습 예제 아이디어")
            ]
            content_type = st.selectbox(L.get("content_type_label", "콘텐츠 유형:"), content_type_options, key="home_content_type")
            content_topic = st.text_input(L.get("topic_label", "주제:"), key="home_content_topic", placeholder=L.get("topic_placeholder", "학습할 주제를 입력하세요..."))
            
            if st.button(L.get("generate_button", "생성"), key="home_generate_content"):
                api_key = get_api_key("openai") or get_api_key("gemini")
                if content_topic and api_key:
                    try:
                        with st.spinner(f"{content_type} 생성 중..."):
                            if content_type == L.get("content_type_quiz", "객관식 퀴즈 10문항"):
                                # 퀴즈 생성 로직
                                def extract_json_from_text(text):
                                    """텍스트에서 JSON 객체를 추출하는 함수 (마크다운 코드 블록 제거)"""
                                    if not text:
                                        return None
                                    
                                    text = text.strip()
                                    
                                    # 1. Markdown 코드 블록 제거
                                    if "```json" in text:
                                        start = text.find("```json") + 7
                                        end = text.find("```", start)
                                        if end != -1:
                                            text = text[start:end].strip()
                                    elif "```" in text:
                                        start = text.find("```") + 3
                                        end = text.find("```", start)
                                        if end != -1:
                                            text = text[start:end].strip()
                                    
                                    # 2. 첫 번째 '{' 부터 마지막 '}' 까지 추출
                                    first_brace = text.find('{')
                                    if first_brace == -1:
                                        return None
                                    
                                    # 중괄호 매칭으로 JSON 객체 끝 찾기
                                    brace_count = 0
                                    last_brace = -1
                                    for i in range(first_brace, len(text)):
                                        if text[i] == '{':
                                            brace_count += 1
                                        elif text[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                last_brace = i
                                                break
                                    
                                    if last_brace != -1:
                                        json_str = text[first_brace:last_brace + 1]
                                        return json_str.strip()
                                    
                                    return None
                                
                                from langchain_openai import ChatOpenAI
                                import json
                                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
                                quiz_prompt = f"""주제 '{content_topic}'에 대한 객관식 문제 10개를 생성해주세요. 
각 문제는 4개의 선택지를 가지며, 정답과 상세한 해설을 포함해야 합니다.
JSON 형식으로 출력하되, 다음 구조를 따라주세요:
{{
  "quiz_questions": [
    {{
      "question": "문제 내용",
      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "answer": 1,
      "explanation": "정답 해설"
    }}
  ]
}}
중요: 마크다운 코드 블록(```json 등) 없이 순수 JSON만 출력해주세요."""
                                response = llm.invoke([{"role": "user", "content": quiz_prompt}])
                                
                                # 응답에서 JSON 추출
                                response_text = response.content if hasattr(response, 'content') else str(response)
                                extracted_json = extract_json_from_text(response_text) or response_text
                                
                                # JSON 파싱 시도 (GitHub _content.py 스타일)
                                try:
                                    quiz_data_obj = json.loads(extracted_json)
                                    quiz_data = quiz_data_obj.get("quiz_questions", [])
                                    
                                    if quiz_data:
                                        # 세션 상태에 퀴즈 데이터 저장 (GitHub _content.py 스타일)
                                        st.session_state.home_quiz_data = quiz_data
                                        st.session_state.home_current_question_index = 0
                                        st.session_state.home_quiz_score = 0
                                        st.session_state.home_quiz_answers = [1] * len(quiz_data)
                                        st.session_state.home_show_explanation = False
                                        st.session_state.home_is_quiz_active = True
                                        st.session_state.home_quiz_type_key = str(uuid.uuid4())
                                        
                                        st.success(f"✅ {content_topic} {L.get('topic_label', '주제')}의 {L.get('quiz_progress', '퀴즈')} 10개가 생성되었습니다!")
                                    else:
                                        st.error(L.get("content_generation_error", "콘텐츠 생성 중 오류"))
                                except json.JSONDecodeError as e:
                                    st.error(L.get("content_generation_error", "콘텐츠 생성 중 오류") + f": {str(e)}")
                                    st.code(extracted_json, language="text")
                                    st.info("💡 " + L.get("content_generation_error", "원본 응답을 확인하세요. LLM이 올바른 JSON 형식으로 응답하지 않았을 수 있습니다."))
                            elif content_type == L.get("content_type_summary", "핵심 요약 노트"):
                                # 요약 노트 생성
                                from langchain_openai import ChatOpenAI
                                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
                                summary_prompt = f"""주제 '{content_topic}'에 대한 핵심 요약 노트를 작성해주세요.
다음 내용을 포함해주세요:
1. 핵심 개념 정리
2. 주요 포인트 (3-5개)
3. 실무 적용 팁
4. 주의사항

마크다운 형식으로 작성해주세요."""
                                response = llm.invoke([{"role": "user", "content": summary_prompt}])
                                st.markdown(response.content)
                            else:  # 실습 예제 아이디어
                                # 실습 예제 생성
                                from langchain_openai import ChatOpenAI
                                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)
                                example_prompt = f"""주제 '{content_topic}'에 대한 실습 예제 아이디어 5개를 제안해주세요.
각 예제는 다음을 포함해야 합니다:
1. 예제 제목
2. 학습 목표
3. 실습 내용 개요
4. 예상 소요 시간
5. 난이도 (초급/중급/고급)

마크다운 형식으로 작성해주세요."""
                                response = llm.invoke([{"role": "user", "content": example_prompt}])
                                st.markdown(response.content)
                    except Exception as e:
                        st.error(L.get("content_generation_error", "콘텐츠 생성 중 오류") + f": {str(e)}")
                else:
                    st.warning(L.get("enter_topic_and_api_key", "주제를 입력하고 API 키를 설정해주세요."))
            if st.button(L.get("close_button", "닫기"), key="close_home_content"):
                st.session_state.show_home_content = False
            
            # 퀴즈 렌더링 (GitHub _content.py 스타일)
            if st.session_state.get("home_is_quiz_active", False) and st.session_state.get("home_quiz_data"):
                quiz_data = st.session_state.home_quiz_data
                idx = st.session_state.home_current_question_index
                
                st.markdown("---")
                st.markdown(f"### {L.get('quiz_progress', '📝 퀴즈 진행')}")
                
                if idx >= len(quiz_data):
                    # 퀴즈 완료
                    st.success(L.get("quiz_complete", "퀴즈 완료!"))
                    total_questions = len(quiz_data)
                    score = st.session_state.home_quiz_score
                    incorrect_count = total_questions - score
                    st.subheader(f"{L.get('score_label', '점수:')} {score} / {total_questions} ({(score / total_questions) * 100:.1f}%)")
                    
                    # 결과 표시
                    st.markdown(f"### {L.get('question_results', '문제 결과')}")
                    for i, question_item in enumerate(quiz_data):
                        user_answer = st.session_state.home_quiz_answers[i] if i < len(st.session_state.home_quiz_answers) else None
                        is_correct = user_answer == 'Correctly Scored'
                        correct_answer_idx = question_item.get('answer', 1)
                        correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                        
                        if is_correct:
                            status_icon = "✅"
                            status_color = "green"
                        else:
                            status_icon = "❌"
                            status_color = "red"
                        
                        st.markdown(f"**{status_icon} {L.get('question_number', '문항 {num}').format(num=i+1)}:** {question_item['question']}")
                        if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                            user_answer_text = question_item['options'][user_answer - 1]
                        else:
                            user_answer_text = L.get("no_answer", "미응답")
                        st.markdown(f"- {L.get('my_answer', '내 답안:')} {user_answer_text}")
                        st.markdown(f"- {L.get('correct_answer', '정답:')} {correct_answer_text}")
                        st.markdown("---")
                    
                    if st.button(L.get("retake_quiz", "다시 풀기"), key="home_retake_quiz"):
                        st.session_state.home_current_question_index = 0
                        st.session_state.home_quiz_score = 0
                        st.session_state.home_quiz_answers = [1] * len(quiz_data)
                        st.session_state.home_show_explanation = False
                else:
                    # 퀴즈 진행
                    question_data = quiz_data[idx]
                    st.subheader(f"{L.get('question_number', '문항 {num}').format(num=idx + 1)}/{len(quiz_data)}")
                    st.markdown(f"**{question_data['question']}**")
                    
                    options = question_data['options']
                    current_answer = st.session_state.home_quiz_answers[idx]
                    
                    if current_answer is None or not isinstance(current_answer, int) or current_answer <= 0:
                        radio_index = 0
                    else:
                        radio_index = min(current_answer - 1, len(options) - 1)
                    
                    selected_option = st.radio(
                        L.get("answer_choice", "답안 선택"),
                        options,
                        index=radio_index,
                        key=f"home_quiz_radio_{st.session_state.home_quiz_type_key}_{idx}"
                    )
                    
                    selected_option_index = options.index(selected_option) + 1 if selected_option in options else None
                    
                    check_col, next_col = st.columns([1, 1])
                    
                    if check_col.button(L.get("check_answer", "답안 확인"), key=f"home_check_answer_btn_{idx}"):
                        if selected_option_index is None:
                            st.warning(L.get("please_select_option", "선택지를 선택해 주세요."))
                        else:
                            if st.session_state.home_quiz_answers[idx] != 'Correctly Scored':
                                correct_answer = question_data.get('answer')
                                if selected_option_index == correct_answer:
                                    st.session_state.home_quiz_score += 1
                                    st.session_state.home_quiz_answers[idx] = 'Correctly Scored'
                                    st.success(L.get("correct", "정답입니다!"))
                                else:
                                    st.session_state.home_quiz_answers[idx] = selected_option_index
                                    st.error(L.get("incorrect", "틀렸습니다."))
                            
                            st.session_state.home_show_explanation = True
                    
                    if st.session_state.home_show_explanation:
                        correct_index = question_data.get('answer', 1)
                        correct_answer_text = question_data['options'][correct_index - 1] if 0 < correct_index <= len(question_data['options']) else "N/A"
                        
                        st.markdown("---")
                        st.markdown(f"**{L.get('correct_answer', '정답:')}** {correct_answer_text}")
                        with st.expander(f"**{L.get('explanation', '해설')}**", expanded=True):
                            st.info(question_data.get('explanation', L.get('no_explanation', '해설이 제공되지 않았습니다.')))
                        
                        if next_col.button(L.get("next_question", "다음 문항"), key=f"home_next_question_btn_{idx}"):
                            st.session_state.home_current_question_index += 1
                            st.session_state.home_show_explanation = False
                    else:
                        # 이미 답안 확인했으면 다음 버튼 바로 표시
                        if st.session_state.home_quiz_answers[idx] == 'Correctly Scored' or (isinstance(st.session_state.home_quiz_answers[idx], int) and st.session_state.home_quiz_answers[idx] > 0):
                            if next_col.button(L.get("next_question", "다음 문항"), key=f"home_next_question_btn_after_check_{idx}"):
                                st.session_state.home_current_question_index += 1
                                st.session_state.home_show_explanation = False
    
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

