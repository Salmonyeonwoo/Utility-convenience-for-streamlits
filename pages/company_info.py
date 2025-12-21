# ========================================
# pages/company_info.py
# Company Info & FAQ Tab 모듈
# ========================================

import os
import base64
import tempfile
import html as html_escape
import streamlit as st
import requests
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

from lang_pack import LANG
from llm_client import get_api_key, run_llm
from faq_manager import (
    load_faq_database, save_faq_database, get_company_info_faq,
    visualize_company_data, get_product_image_url,
    generate_company_info_with_llm
)


def render_company_info_tab():
    """Company Info & FAQ Tab 렌더링 함수"""
    # 현재 언어 확인 및 L 변수 정의
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # FAQ 데이터베이스 로드
    faq_data = load_faq_database()
    companies = list(faq_data.get("companies", {}).keys())
    
    # 회사명 검색 입력 (상단에 배치) - 입력란은 글로벌 기업 영문명 고려하여 원래 크기 유지
    col_search_header, col_search_input, col_search_btn = st.columns([0.5, 1.2, 0.2])
    with col_search_header:
        st.write(f"**{L['search_company']}**")
    with col_search_input:
        company_search_input = st.text_input(
            "",
            placeholder=L["company_search_placeholder"],
            key="company_search_input",
            value=st.session_state.get("searched_company", ""),
            label_visibility="collapsed"
        )
    with col_search_btn:
        search_button = st.button(f"🔍 {L['company_search_button']}", key="company_search_btn", type="primary", use_container_width=True)
    
    # 검색된 회사 정보 저장
    searched_company = st.session_state.get("searched_company", "")
    searched_company_data = st.session_state.get("searched_company_data", None)
    
    # 검색 버튼 클릭 시 LLM으로 회사 정보 생성
    if search_button and company_search_input:
        with st.spinner(f"{company_search_input} {L['generating_company_info']}"):
            generated_data = generate_company_info_with_llm(company_search_input, current_lang)
            st.session_state.searched_company = company_search_input
            st.session_state.searched_company_data = generated_data
            searched_company = company_search_input
            searched_company_data = generated_data
            
            # 생성된 데이터를 데이터베이스에 저장
            if company_search_input not in faq_data.get("companies", {}):
                faq_data.setdefault("companies", {})[company_search_input] = {
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
    
    # 검색된 회사가 있으면 해당 데이터 사용, 없으면 기존 회사 선택
    if searched_company and searched_company_data:
        display_company = searched_company
        display_data = searched_company_data
        # 데이터베이스에도 저장되어 있으면 업데이트
        if display_company in faq_data.get("companies", {}):
            faq_data["companies"][display_company].update({
                f"info_{current_lang}": display_data.get("company_info", ""),
                "popular_products": display_data.get("popular_products", []),
                "trending_topics": display_data.get("trending_topics", []),
                "faqs": display_data.get("faqs", []),
                "interview_questions": display_data.get("interview_questions", []),
                "ceo_info": display_data.get("ceo_info", {})
            })
            save_faq_database(faq_data)
    elif companies:
        display_company = st.selectbox(
            L["select_company"],
            options=companies,
            key="company_select_display"
        )
        company_db_data = faq_data["companies"][display_company]
        display_data = {
            "company_info": company_db_data.get(f"info_{current_lang}", company_db_data.get("info_ko", "")),
            "popular_products": company_db_data.get("popular_products", []),
            "trending_topics": company_db_data.get("trending_topics", []),
            "faqs": company_db_data.get("faqs", []),
            "interview_questions": company_db_data.get("interview_questions", []),
            "ceo_info": company_db_data.get("ceo_info", {})
        }
    else:
        display_company = None
        display_data = None
    
    # 탭 생성 (FAQ 검색 탭 제거, FAQ 탭에 통합) - 공백 축소
    tab1, tab2, tab3 = st.tabs([
        L["company_info"], 
        L["company_faq"], 
        L["button_add_company"]
    ])
    
    # 탭 1: 회사 소개 및 시각화
    with tab1:
        if display_company and display_data:
            # 제목을 더 간결하게 표시
            st.markdown(f"#### {display_company} - {L['company_info']}")
            
            # 회사 소개 표시
            if display_data.get("company_info"):
                st.markdown(display_data["company_info"])
            
            # 시각화 차트 표시
            if display_data.get("popular_products") or display_data.get("trending_topics"):
                charts = visualize_company_data(
                    {
                        "popular_products": display_data.get("popular_products", []),
                        "trending_topics": display_data.get("trending_topics", [])
                    },
                    current_lang
                )
                
                if charts:
                    # 막대 그래프 표시 - 공백 축소
                    st.markdown(f"#### 📊 {L['visualization_chart']}")
                    col1_bar, col2_bar = st.columns(2)
                    
                    if "products_bar" in charts:
                        with col1_bar:
                            st.plotly_chart(charts["products_bar"], use_container_width=True)
                    
                    if "topics_bar" in charts:
                        with col2_bar:
                            st.plotly_chart(charts["topics_bar"], use_container_width=True)
                    
                    # 선형 그래프 표시
                    col1_line, col2_line = st.columns(2)
                    
                    if "products_line" in charts:
                        with col1_line:
                            st.plotly_chart(charts["products_line"], use_container_width=True)
                    
                    if "topics_line" in charts:
                        with col2_line:
                            st.plotly_chart(charts["topics_line"], use_container_width=True)
            
            # 인기 상품 목록 (이미지 포함) - 공백 축소
            if display_data.get("popular_products"):
                st.markdown(f"#### {L['popular_products']}")
                # 상품을 그리드 형태로 표시
                product_cols = st.columns(min(3, len(display_data["popular_products"])))
                for idx, product in enumerate(display_data["popular_products"]):
                    product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                    product_score = product.get("score", 0)
                    product_image_url = product.get("image_url", "")
                    
                    with product_cols[idx % len(product_cols)]:
                        # 이미지 표시 - 상품명 기반으로 동적 이미지 검색
                        if not product_image_url:
                            # 모든 언어 버전의 상품명을 확인하여 이미지 URL 생성
                            # 우선순위: 현재 언어 > 한국어 > 영어 > 일본어
                            image_found = False
                            for lang_key in [current_lang, "ko", "en", "ja"]:
                                check_text = product.get(f"text_{lang_key}", "")
                                if check_text:
                                    check_url = get_product_image_url(check_text)
                                    if check_url:
                                        product_image_url = check_url
                                        image_found = True
                                        break
                            
                            # 모든 언어에서 이미지를 찾지 못한 경우 기본 이미지 사용
                            if not image_found:
                                product_image_url = get_product_image_url(product_text)
                        
                        # 이미지 표시 시도 (로컬 파일 및 URL 모두 지원)
                        image_displayed = False
                        if product_image_url:
                            try:
                                # 로컬 파일 경로인 경우
                                if os.path.exists(product_image_url):
                                    st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                    image_displayed = True
                                # URL인 경우
                                elif product_image_url.startswith("http://") or product_image_url.startswith("https://"):
                                    try:
                                        # HEAD 요청으로 이미지 존재 여부 확인 (타임아웃 2초)
                                        response = requests.head(product_image_url, timeout=2, allow_redirects=True)
                                        if response.status_code == 200:
                                            st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                            image_displayed = True
                                        else:
                                            image_displayed = False
                                    except Exception:
                                        # HEAD 요청 실패 시에도 이미지 표시 시도 (일부 서버는 HEAD를 지원하지 않음)
                                        try:
                                            st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                            image_displayed = True
                                        except Exception:
                                            image_displayed = False
                                else:
                                    # 기타 경로 시도
                                    try:
                                        st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                                        image_displayed = True
                                    except Exception:
                                        image_displayed = False
                            except Exception as img_error:
                                # 이미지 로딩 실패
                                image_displayed = False
                        
                        # 이미지 표시 실패 시 이모지 카드 표시
                        if not image_displayed:
                            product_emoji = "🎫" if "티켓" in product_text or "ticket" in product_text.lower() else \
                                          "🎢" if "테마파크" in product_text or "theme" in product_text.lower() or "디즈니" in product_text or "유니버셜" in product_text or "스튜디오" in product_text else \
                                          "✈️" if "항공" in product_text or "flight" in product_text.lower() else \
                                          "🏨" if "호텔" in product_text or "hotel" in product_text.lower() else \
                                          "🍔" if "음식" in product_text or "food" in product_text.lower() else \
                                          "🌏" if "여행" in product_text or "travel" in product_text.lower() or "사파리" in product_text else \
                                          "📦"
                            product_html = """<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 10px; color: white; min-height: 200px; display: flex; flex-direction: column; justify-content: center;'>
                                    <h1 style='font-size: 64px; margin: 0;'>""" + product_emoji + """</h1>
                                    <p style='font-size: 16px; margin-top: 15px; font-weight: bold;'>""" + product_text[:25] + """</p>
                                </div>"""
                            st.markdown(product_html, unsafe_allow_html=True)
                        
                        st.write(f"**{product_text}**")
                        st.caption(f"{L.get('popularity', '인기도')}: {product_score}")
                        st.markdown("---")
            
            # 화제의 소식 목록 (상세 내용 포함) - 공백 축소
            if display_data.get("trending_topics"):
                st.markdown(f"#### {L['trending_topics']}")
                for idx, topic in enumerate(display_data["trending_topics"], 1):
                    topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                    topic_score = topic.get("score", 0)
                    topic_detail = topic.get(f"detail_{current_lang}", topic.get("detail_ko", ""))
                    
                    with st.expander(f"{idx}. **{topic_text}** ({L.get('trend_score', '화제도')}: {topic_score})"):
                        if topic_detail:
                            st.write(topic_detail)
                        else:
                            # 상세 내용이 없으면 LLM으로 생성
                            if display_company:
                                try:
                                    # 언어별 프롬프트
                                    detail_prompts = {
                                        "ko": f"{display_company}의 '{topic_text}'에 대한 상세 내용을 200자 이상 작성해주세요.",
                                        "en": f"Please write detailed content of at least 200 characters about '{topic_text}' from {display_company}.",
                                        "ja": f"{display_company}の「{topic_text}」に関する詳細内容を200文字以上で作成してください。"
                                    }
                                    detail_prompt = detail_prompts.get(current_lang, detail_prompts["ko"])
                                    generated_detail = run_llm(detail_prompt)
                                    if generated_detail and not generated_detail.startswith("❌"):
                                        st.write(generated_detail)
                                        # 생성된 상세 내용을 데이터베이스에 저장
                                        if display_company in faq_data.get("companies", {}):
                                            topic_idx = idx - 1
                                            if topic_idx < len(faq_data["companies"][display_company].get("trending_topics", [])):
                                                faq_data["companies"][display_company]["trending_topics"][topic_idx][f"detail_{current_lang}"] = generated_detail
                                                save_faq_database(faq_data)
                                    else:
                                        st.write(L.get("generating_detail", "상세 내용을 생성하는 중입니다..."))
                                except Exception as e:
                                    st.write(L.get("checking_additional_info", "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.").format(topic=topic_text))
                            else:
                                st.write(L.get("checking_additional_info", "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.").format(topic=topic_text))
            
            # CEO/대표이사 정보 표시
            if display_data.get("ceo_info"):
                ceo_info = display_data["ceo_info"]
                ceo_name = ceo_info.get(f"name_{current_lang}", ceo_info.get("name_ko", ""))
                ceo_position = ceo_info.get(f"position_{current_lang}", ceo_info.get("position_ko", ""))
                ceo_bio = ceo_info.get(f"bio_{current_lang}", ceo_info.get("bio_ko", ""))
                ceo_tenure = ceo_info.get(f"tenure_{current_lang}", ceo_info.get("tenure_ko", ""))
                ceo_education = ceo_info.get(f"education_{current_lang}", ceo_info.get("education_ko", ""))
                ceo_career = ceo_info.get(f"career_{current_lang}", ceo_info.get("career_ko", ""))
                
                if ceo_name or ceo_position:
                    st.markdown(f"#### 👔 {L.get('ceo_info', 'CEO/대표이사 정보')}")
                    st.markdown("---")
                    
                    # CEO 정보 카드 형태로 표시
                    col_ceo_left, col_ceo_right = st.columns([1, 2])
                    
                    with col_ceo_left:
                        # CEO 이름과 직책
                        if ceo_name:
                            st.markdown(f"### {ceo_name}")
                        if ceo_position:
                            st.markdown(f"**{L.get('position', '직책')}:** {ceo_position}")
                        if ceo_tenure:
                            st.markdown(f"**{L.get('tenure', '재임 기간')}:** {ceo_tenure}")
                    
                    with col_ceo_right:
                        # 상세 소개
                        if ceo_bio:
                            st.markdown(f"**{L.get('ceo_bio', '소개')}**")
                            st.markdown(ceo_bio)
                    
                    # 학력 및 경력 정보
                    if ceo_education or ceo_career:
                        st.markdown("---")
                        col_edu, col_career = st.columns(2)
                        
                        with col_edu:
                            if ceo_education:
                                st.markdown(f"**{L.get('education', '학력')}**")
                                st.markdown(ceo_education)
                        
                        with col_career:
                            if ceo_career:
                                st.markdown(f"**{L.get('career', '주요 경력')}**")
                                st.markdown(ceo_career)
                    
                    st.markdown("---")
            
            # 면접 질문 목록 표시
            if display_data.get("interview_questions"):
                st.markdown(f"#### 💼 {L.get('interview_questions', '면접 예상 질문')}")
                st.markdown(f"*{L.get('interview_questions_desc', '면접에서 나올 만한 핵심 질문들과 상세한 답변입니다. 면접 준비와 회사 이해에 도움이 됩니다.')}*")
                st.markdown("---")
                
                # 카테고리별로 그룹화
                interview_by_category = {}
                for idx, iq in enumerate(display_data["interview_questions"]):
                    question = iq.get(f"question_{current_lang}", iq.get("question_ko", ""))
                    answer = iq.get(f"answer_{current_lang}", iq.get("answer_ko", ""))
                    category = iq.get(f"category_{current_lang}", iq.get("category_ko", L.get("interview_category_other", "기타")))
                    
                    if category not in interview_by_category:
                        interview_by_category[category] = []
                    interview_by_category[category].append({
                        "question": question,
                        "answer": answer,
                        "index": idx + 1
                    })
                
                # 카테고리별로 표시
                for category, questions in interview_by_category.items():
                    with st.expander(f"📋 **{category}** ({len(questions)}{L.get('items', '개')})"):
                        for item in questions:
                            st.markdown(f"**{item['index']}. {item['question']}**")
                            st.markdown(item['answer'])
                            st.markdown("---")
        else:
            st.info(L["company_search_or_select"])
    
    # 탭 2: 자주 묻는 질문 (FAQ) - 검색 기능 포함
    with tab2:
        if display_company and display_data:
            # 제목을 더 간결하게 표시
            st.markdown(f"#### {display_company} - {L['company_faq']}")
            
            # FAQ 검색 기능 (탭 내부에 통합) - 검색 범위 확대, 공백 축소
            col_search_faq, col_btn_faq = st.columns([3.5, 1])
            with col_search_faq:
                faq_search_query = st.text_input(
                    L["faq_search_placeholder"],
                    key="faq_search_in_tab",
                    placeholder=L.get("faq_search_placeholder_extended", L["faq_search_placeholder"])
                )
            with col_btn_faq:
                faq_search_btn = st.button(L["button_search_faq"], key="faq_search_btn_in_tab")
            
            faqs = display_data.get("faqs", [])
            popular_products = display_data.get("popular_products", [])
            trending_topics = display_data.get("trending_topics", [])
            company_info = display_data.get("company_info", "")
            
            # 검색 관련 변수 초기화
            matched_products = []
            matched_topics = []
            matched_info = False
            
            # 검색어가 있으면 확장된 검색 (FAQ, 상품, 화제 소식, 회사 소개 모두 검색)
            if faq_search_query and faq_search_btn:
                query_lower = faq_search_query.lower()
                filtered_faqs = []
                
                # 1. FAQ 검색 (기본 FAQ + 상품명 관련 FAQ)
                for faq in faqs:
                    question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                    answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                    if query_lower in question.lower() or query_lower in answer.lower():
                        filtered_faqs.append(faq)
                
                # 2. 상품명으로 FAQ 검색 (상품명이 검색어와 일치하거나 포함되는 경우)
                # 검색어가 상품명에 포함되면 해당 상품과 관련된 FAQ를 찾아서 표시
                for product in popular_products:
                    product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                    product_text_lower = product_text.lower()
                    
                    # 검색어가 상품명에 포함되는 경우
                    if query_lower in product_text_lower:
                        # 해당 상품명이 FAQ 질문/답변에 포함된 경우 찾기
                        product_related_faqs = []
                        for faq in faqs:
                            question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                            answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                            # 상품명이 FAQ에 언급되어 있으면 추가
                            if product_text_lower in question.lower() or product_text_lower in answer.lower():
                                if faq not in filtered_faqs:
                                    filtered_faqs.append(faq)
                                    product_related_faqs.append(faq)
                        
                        # 상품명이 매칭되었지만 관련 FAQ가 없는 경우, 상품 정보만 표시
                        if not product_related_faqs:
                            matched_products.append(product)
                
                # 2. 인기 상품 검색
                for product in popular_products:
                    product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                    if query_lower in product_text.lower():
                        matched_products.append(product)
                
                # 3. 화제의 소식 검색
                for topic in trending_topics:
                    topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                    if query_lower in topic_text.lower():
                        matched_topics.append(topic)
                
                # 4. 회사 소개 검색
                if query_lower in company_info.lower():
                    matched_info = True
                
                # 검색 결과가 있으면 표시
                if filtered_faqs or matched_products or matched_topics or matched_info:
                    # 매칭된 상품 표시 (FAQ가 없는 경우에만)
                    if matched_products and not filtered_faqs:
                        st.subheader(f"🔍 {L.get('related_products', '관련 상품')} ({len(matched_products)}{L.get('items', '개')})")
                        st.info(L.get("no_faq_for_product", "해당 상품과 관련된 FAQ를 찾을 수 없습니다. 상품 정보만 표시됩니다."))
                        for idx, product in enumerate(matched_products, 1):
                            product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                            product_score = product.get("score", 0)
                            st.write(f"• **{product_text}** ({L.get('popularity', '인기도')}: {product_score})")
                        st.markdown("---")
                    
                    # 매칭된 화제 소식 표시
                    if matched_topics:
                        st.subheader(f"🔍 {L.get('related_trending_news', '관련 화제 소식')} ({len(matched_topics)}{L.get('items', '개')})")
                        for idx, topic in enumerate(matched_topics, 1):
                            topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                            topic_score = topic.get("score", 0)
                            st.write(f"• **{topic_text}** ({L.get('trend_score', '화제도')}: {topic_score})")
                        st.markdown("---")
                    
                    # 매칭된 회사 소개 표시
                    if matched_info:
                        st.subheader(f"🔍 {L.get('related_company_info', '관련 회사 소개 내용')}")
                        # 검색어가 포함된 부분 강조하여 표시
                        info_lower = company_info.lower()
                        query_pos = info_lower.find(query_lower)
                        if query_pos != -1:
                            start = max(0, query_pos - 100)
                            end = min(len(company_info), query_pos + len(query_lower) + 100)
                            snippet = company_info[start:end]
                            if start > 0:
                                snippet = "..." + snippet
                            if end < len(company_info):
                                snippet = snippet + "..."
                            # 검색어 강조
                            highlighted = snippet.replace(
                                query_lower, 
                                f"**{query_lower}**"
                            )
                            st.write(highlighted)
                        st.markdown("---")
                    
                    # FAQ 결과
                    faqs = filtered_faqs
                else:
                    faqs = []
            
            # FAQ 목록 표시
            if faqs:
                if faq_search_query and faq_search_btn:
                    st.subheader(f"🔍 {L.get('related_faq', '관련 FAQ')} ({len(faqs)}{L.get('items', '개')})")
                else:
                    st.subheader(f"{L['company_faq']} ({len(faqs)}{L.get('items', '개')})")
                for idx, faq in enumerate(faqs, 1):
                    question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                    answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                    with st.expander(f"{L['faq_question_prefix'].format(num=idx)} {question}"):
                        st.write(f"**{L['faq_answer']}:** {answer}")
            else:
                if faq_search_query and faq_search_btn:
                    # 검색 결과가 없을 때만 메시지 표시 (위에서 이미 관련 상품/소식 등이 표시되었을 수 있음)
                    if not (matched_products or matched_topics or matched_info):
                        st.info(L["no_faq_results"])
                else:
                    st.info(L.get("no_faq_for_company", f"{display_company}의 FAQ가 없습니다.").format(company=display_company))
        else:
            st.info(L.get("no_company_selected", "회사명을 검색하거나 선택해주세요."))
    
    # 탭 3: 고객 문의 재확인 (에이전트용)
    with tab3:
        # 제목과 설명을 한 줄로 간결하게 표시
        st.markdown(f"#### {L['customer_inquiry_review']}")
        st.caption(L.get("customer_inquiry_review_desc", "에이전트가 상사들에게 고객 문의 내용을 재확인하고, AI 답안 및 힌트를 생성할 수 있는 기능입니다."))
        
        # 세션 상태 초기화
        if "generated_ai_answer" not in st.session_state:
            st.session_state.generated_ai_answer = None
        if "generated_hint" not in st.session_state:
            st.session_state.generated_hint = None
        
        # 회사 선택 (선택사항)
        selected_company_for_inquiry = None
        if companies:
            all_option = L.get("all_companies", "전체")
            selected_company_for_inquiry = st.selectbox(
                f"{L['select_company']} ({L.get('optional', '선택사항')})",
                options=[all_option] + companies,
                key="inquiry_company_select"
            )
            if selected_company_for_inquiry == all_option:
                selected_company_for_inquiry = None
        
        # 고객 문의 내용 입력
        customer_inquiry = st.text_area(
            L["inquiry_question_label"],
            placeholder=L["inquiry_question_placeholder"],
            key="customer_inquiry_input",
            height=150
        )
        
        # 고객 첨부 파일 업로드
        uploaded_file = st.file_uploader(
            L.get("inquiry_attachment_label", "📎 고객 첨부 파일 업로드 (사진/스크린샷)"),
            type=["png", "jpg", "jpeg", "pdf"],
            key="customer_inquiry_attachment",
            help=L.get("inquiry_attachment_help", "특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 반드시 사진이나 스크린샷을 첨부해주세요.")
        )
        
        # 업로드된 파일 정보 저장
        attachment_info = ""
        uploaded_file_info = None
        file_content_extracted = ""
        file_content_translated = ""
        
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_type = uploaded_file.type
            file_size = len(uploaded_file.getvalue())
            st.success(L.get("inquiry_attachment_uploaded", "✅ 첨부 파일이 업로드되었습니다: {filename}").format(filename=file_name))
            
            # 파일 정보 저장
            uploaded_file_info = {
                "name": file_name,
                "type": file_type,
                "size": file_size
            }
            
            # 파일 내용 추출 (PDF, TXT, 이미지 파일인 경우)
            if file_name.lower().endswith(('.pdf', '.txt', '.png', '.jpg', '.jpeg')):
                try:
                    with st.spinner(L.get("extracting_file_content", "파일 내용 추출 중...")):
                        if file_name.lower().endswith('.pdf'):
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                            tmp.write(uploaded_file.getvalue())
                            tmp.flush()
                            tmp.close()
                            try:
                                loader = PyPDFLoader(tmp.name)
                                file_docs = loader.load()
                                file_content_extracted = "\n".join([doc.page_content for doc in file_docs])
                            finally:
                                try:
                                    os.remove(tmp.name)
                                except:
                                    pass
                        elif file_name.lower().endswith('.txt'):
                            uploaded_file.seek(0)  # 파일 포인터를 처음으로 이동
                            file_content_extracted = uploaded_file.read().decode("utf-8", errors="ignore")
                        elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # 이미지 파일의 경우 OCR을 사용하여 텍스트 추출
                            uploaded_file.seek(0)
                            image_bytes = uploaded_file.getvalue()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            
                            # Gemini Vision API를 사용하여 이미지에서 텍스트 추출
                            ocr_prompt = """이 이미지에 있는 모든 텍스트를 정확히 추출해주세요. 
이미지에 한국어, 일본어, 영어 등 어떤 언어의 텍스트가 있든 모두 추출하고, 
텍스트의 구조와 순서를 유지해주세요. 
이미지에 텍스트가 없으면 "텍스트 없음"이라고 답변하세요.

추출된 텍스트:"""
                            
                            try:
                                # Gemini Vision API 호출
                                gemini_key = get_api_key("gemini")
                                if gemini_key:
                                    genai.configure(api_key=gemini_key)
                                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                                    
                                    # 이미지와 프롬프트를 함께 전송
                                    response = model.generate_content([
                                        {
                                            "mime_type": file_type,
                                            "data": image_bytes
                                        },
                                        ocr_prompt
                                    ])
                                    file_content_extracted = response.text if response.text else ""
                                else:
                                    # Gemini 키가 없으면 LLM에 base64 이미지를 전송하여 OCR 요청
                                    ocr_llm_prompt = """{ocr_prompt}

이미지는 base64로 인코딩되어 전송되었습니다. 이미지에서 텍스트를 추출해주세요."""
                                    # LLM이 이미지를 직접 처리할 수 없으므로, 사용자에게 안내
                                    file_content_extracted = ""
                                    st.info(L.get("ocr_requires_manual", "이미지 OCR을 위해서는 Gemini API 키가 필요합니다. 이미지의 텍스트를 수동으로 입력해주세요."))
                            except Exception as ocr_error:
                                error_msg = L.get("ocr_error", "이미지 텍스트 추출 중 오류: {error}")
                                st.warning(error_msg.format(error=str(ocr_error)))
                                file_content_extracted = ""
                        
                        # 파일 내용이 추출된 경우 언어 감지 및 번역 (일본어/영어 버전에서 한국어 파일 번역)
                        if file_content_extracted and current_lang in ["ja", "en"]:
                            # 한국어 내용인지 확인하고 번역
                            with st.spinner(L.get("detecting_language", "언어 감지 중...")):
                                # 언어 감지 프롬프트 (현재 언어에 맞춤)
                                detect_prompts = {
                                    "ja": f"""次のテキストの言語を検出してください。韓国語、日本語、英語のいずれかで答えてください。

テキスト:
{file_content_extracted[:500]}

言語:""",
                                    "en": f"""Detect the language of the following text. Answer with only one of: Korean, Japanese, or English.

Text:
{file_content_extracted[:500]}

Language:""",
                                    "ko": f"""다음 텍스트의 언어를 감지해주세요. 한국어, 일본어, 영어 중 하나로만 답변하세요.

텍스트:
{file_content_extracted[:500]}

언어:"""
                                }
                                detect_prompt = detect_prompts.get(current_lang, detect_prompts["ko"])
                                detected_lang = run_llm(detect_prompt).strip().lower()
                                
                                # 한국어로 감지된 경우 현재 언어로 번역
                                if "한국어" in detected_lang or "korean" in detected_lang or "ko" in detected_lang:
                                    with st.spinner(L.get("translating_content", "파일 내용 번역 중...")):
                                        # 번역 프롬프트 (현재 언어에 맞춤)
                                        translate_prompts = {
                                            "ja": f"""次の韓国語テキストを日本語に翻訳してください。原文の意味とトーンを正確に維持しながら、自然な日本語で翻訳してください。

韓国語テキスト:
{file_content_extracted}

日本語翻訳:""",
                                            "en": f"""Please translate the following Korean text into English. Maintain the exact meaning and tone of the original text while translating into natural English.

Korean text:
{file_content_extracted}

English translation:"""
                                        }
                                        translate_prompt = translate_prompts.get(current_lang)
                                        if translate_prompt:
                                            file_content_translated = run_llm(translate_prompt)
                                            if file_content_translated and not file_content_translated.startswith("❌"):
                                                st.info(L.get("file_translated", "✅ 파일 내용이 번역되었습니다."))
                                            else:
                                                file_content_translated = ""
                except Exception as e:
                    error_msg = L.get("file_extraction_error", "파일 내용 추출 중 오류가 발생했습니다: {error}")
                    st.warning(error_msg.format(error=str(e)))
            
            # 언어별 파일 정보 텍스트 생성
            file_content_to_include = file_content_translated if file_content_translated else file_content_extracted
            content_section = ""
            if file_content_to_include:
                content_section = f"\n\n[파일 내용]\n{file_content_to_include[:2000]}"  # 최대 2000자만 포함
                if len(file_content_to_include) > 2000:
                    content_section += "\n...(내용이 길어 일부만 표시됨)"
            
            attachment_info_by_lang = {
                "ko": f"\n\n[고객 첨부 파일 정보]\n- 파일명: {file_name}\n- 파일 타입: {file_type}\n- 파일 크기: {file_size} bytes\n- 참고: 고객이 {file_name} 파일을 첨부했습니다. 이 파일은 비행기 지연, 여권 이슈, 질병 등 불가피한 사유로 인한 취소 불가 여행상품 관련 증빙 자료일 수 있습니다. 파일 내용을 참고하여 응대하세요.{content_section}",
                "en": f"\n\n[Customer Attachment Information]\n- File name: {file_name}\n- File type: {file_type}\n- File size: {file_size} bytes\n- Note: The customer has attached the file {file_name}. This file may be evidence related to non-refundable travel products due to unavoidable reasons such as flight delays, passport issues, illness, etc. Please refer to the file content when responding.{content_section}",
                "ja": f"\n\n[顧客添付ファイル情報]\n- ファイル名: {file_name}\n- ファイルタイプ: {file_type}\n- ファイルサイズ: {file_size} bytes\n- 参考: 顧客が{file_name}ファイルを添付しました。このファイルは、飛行機の遅延、パスポートの問題、病気などやむを得ない理由によるキャンセル不可の旅行商品に関連する証拠資料である可能性があります。ファイルの内容を参照して対応してください。{content_section}"
            }
            attachment_info = attachment_info_by_lang.get(current_lang, attachment_info_by_lang["ko"])
            
            # 이미지 파일인 경우 미리보기 표시
            if file_type and file_type.startswith("image/"):
                st.image(uploaded_file, caption=file_name, use_container_width=True)
        
        col_ai_answer, col_hint = st.columns(2)
        
        # AI 답안 생성
        with col_ai_answer:
            if st.button(L["button_generate_ai_answer"], key="generate_ai_answer_btn", type="primary"):
                if customer_inquiry:
                    with st.spinner(L["generating_ai_answer"]):
                        # 회사 정보가 있으면 포함하여 답안 생성
                        company_context = ""
                        if selected_company_for_inquiry and selected_company_for_inquiry in faq_data.get("companies", {}):
                            company_data = get_company_info_faq(selected_company_for_inquiry, current_lang)
                            company_info_label = L.get("company_info", "회사 정보")
                            company_context = f"\n\n{company_info_label}: {company_data.get('info', '')}"
                            # 관련 FAQ도 포함
                            related_faqs = company_data.get("faqs", [])[:5]  # 상위 5개만
                            if related_faqs:
                                faq_label = L.get("company_faq", "자주 나오는 질문")
                                faq_context = f"\n\n{faq_label}:\n"
                                for faq in related_faqs:
                                    q = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                                    a = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                                    faq_context += f"Q: {q}\nA: {a}\n"
                                company_context += faq_context
                        
                        # 언어별 프롬프트
                        lang_prompts_inquiry = {
                            "ko": f"""다음 고객 문의에 대한 전문적이고 친절한 답안을 작성해주세요.

고객 문의: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

답안은 다음을 포함해야 합니다:
1. 고객의 문의에 대한 명확한 답변
2. 필요한 경우 추가 정보나 안내
3. 친절하고 전문적인 톤
4. 첨부 파일이 있는 경우, 해당 파일 내용을 참고하여 응대하세요. 특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 첨부된 증빙 자료를 확인하고 적절히 대응하세요.

답안:""",
                            "en": f"""Please write a professional and friendly answer to the following customer inquiry.

Customer Inquiry: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

The answer should include:
1. Clear answer to the customer's inquiry
2. Additional information or guidance if needed
3. Friendly and professional tone
4. If there is an attachment, please reference the file content in your response. For non-refundable travel products with unavoidable reasons (flight delays, passport issues, etc.), review the attached evidence and respond appropriately.

Answer:""",
                            "ja": f"""次の顧客問い合わせに対する専門的で親切な回答を作成してください。

顧客問い合わせ: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

回答には以下を含める必要があります:
1. 顧客の問い合わせに対する明確な回答
2. 必要に応じて追加情報や案内
3. 親切で専門的なトーン
4. 添付ファイルがある場合は、そのファイルの内容を参照して対応してください。特にキャンセル不可の旅行商品で、飛行機の遅延、パスポートの問題などやむを得ない理由がある場合は、添付された証拠資料を確認し、適切に対応してください。

回答:"""
                        }
                        prompt = lang_prompts_inquiry.get(current_lang, lang_prompts_inquiry["ko"])
                        
                        ai_answer = run_llm(prompt)
                        st.session_state.generated_ai_answer = ai_answer
                        st.success(f"✅ {L.get('ai_answer_generated', 'AI 답안이 생성되었습니다.')}")
                else:
                    st.warning(L.get("warning_enter_inquiry", "고객 문의 내용을 입력해주세요."))
        
        # 응대 힌트 생성
        with col_hint:
            if st.button(L["button_generate_hint"], key="generate_hint_btn", type="primary"):
                if customer_inquiry:
                    with st.spinner(L["generating_hint"]):
                        # 회사 정보가 있으면 포함하여 힌트 생성
                        company_context = ""
                        if selected_company_for_inquiry and selected_company_for_inquiry in faq_data.get("companies", {}):
                            company_data = get_company_info_faq(selected_company_for_inquiry, current_lang)
                            company_info_label = L.get("company_info", "회사 정보")
                            company_context = f"\n\n{company_info_label}: {company_data.get('info', '')}"
                        
                        # 언어별 프롬프트
                        lang_prompts_hint = {
                            "ko": f"""다음 고객 문의에 대한 응대 힌트를 작성해주세요.

고객 문의: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

응대 힌트는 다음을 포함해야 합니다:
1. 고객 문의의 핵심 포인트
2. 응대 시 주의사항
3. 권장 응대 방식
4. 추가 확인이 필요한 사항 (있는 경우)
5. 첨부 파일이 있는 경우, 해당 파일을 확인하고 증빙 자료로 활용하세요. 특히 취소 불가 여행상품의 경우, 첨부된 사진이나 스크린샷을 통해 불가피한 사유를 확인하고 적절한 조치를 취하세요.

응대 힌트:""",
                            "en": f"""Please write response hints for the following customer inquiry.

Customer Inquiry: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

Response hints should include:
1. Key points of the customer inquiry
2. Precautions when responding
3. Recommended response method
4. Items that need additional confirmation (if any)
5. If there is an attachment, review the file and use it as evidence. For non-refundable travel products, verify unavoidable reasons through attached photos or screenshots and take appropriate action.

Response Hints:""",
                            "ja": f"""次の顧客問い合わせに対する対応ヒントを作成してください。

顧客問い合わせ: {customer_inquiry}
{company_context}
{attachment_info if attachment_info else ""}

対応ヒントには以下を含める必要があります:
1. 顧客問い合わせの核心ポイント
2. 対応時の注意事項
3. 推奨対応方法
4. 追加確認が必要な事項（ある場合）
5. 添付ファイルがある場合は、そのファイルを確認し、証拠資料として活用してください。特にキャンセル不可の旅行商品の場合、添付された写真やスクリーンショットを通じてやむを得ない理由を確認し、適切な措置を取ってください。

対応ヒント:"""
                        }
                        prompt = lang_prompts_hint.get(current_lang, lang_prompts_hint["ko"])
                        
                        hint = run_llm(prompt)
                        st.session_state.generated_hint = hint
                        st.success(f"✅ {L.get('hint_generated', '응대 힌트가 생성되었습니다.')}")
                else:
                    st.warning(L.get("warning_enter_inquiry", "고객 문의 내용을 입력해주세요."))
        
        # 생성된 결과 표시
        if st.session_state.get("generated_ai_answer"):
            st.markdown("---")
            st.subheader(L["ai_answer_header"])
            
            answer_text = st.session_state.generated_ai_answer
            
            # 답안을 선택 가능한 텍스트로 표시 (폰트 크기 확대)
            answer_escaped = html_escape.escape(answer_text)
            st.markdown(f"""
            <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{answer_escaped}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # 다운로드 버튼 추가 (더 안정적인 복사 방법)
            col_copy, col_download = st.columns(2)
            with col_copy:
                st.info(L.get("copy_instruction", "💡 위 텍스트를 선택하고 Ctrl+C (Mac: Cmd+C)로 복사하세요."))
            with col_download:
                st.download_button(
                    label=f"📥 {L.get('button_download_answer', '답안 다운로드')}",
                    data=answer_text.encode('utf-8'),
                    file_name=f"ai_answer_{st.session_state.get('copy_answer_id', 0)}.txt",
                    mime="text/plain",
                    key="download_answer_btn"
                )
        
        if st.session_state.get("generated_hint"):
            st.markdown("---")
            st.subheader(L["hint_header"])
            
            hint_text = st.session_state.generated_hint
            
            # 힌트를 선택 가능한 텍스트로 표시 (폰트 크기 확대)
            hint_escaped = html_escape.escape(hint_text)
            st.markdown(f"""
            <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{hint_escaped}</pre>
            </div>
            """, unsafe_allow_html=True)
