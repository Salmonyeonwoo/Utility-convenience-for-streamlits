# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
탭2: FAQ 검색 및 표시 모듈
"""

import streamlit as st


def render_tab2_faq(display_company: str, display_data: dict, current_lang: str, L: dict):
    """탭2: 자주 묻는 질문 (FAQ) 렌더링"""
    if display_company and display_data:
        st.markdown(f"#### {display_company} - {L['company_faq']}")

        # FAQ 검색 기능
        col_search_faq, col_btn_faq = st.columns([3.5, 1])
        with col_search_faq:
            faq_search_query = st.text_input(
                L["faq_search_placeholder"],
                key="faq_search_in_tab",
                placeholder=L.get("faq_search_placeholder_extended", L["faq_search_placeholder"]))
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

        # 검색어가 있으면 확장된 검색
        if faq_search_query and faq_search_btn:
            query_lower = faq_search_query.lower()
            filtered_faqs = []

            # 1. FAQ 검색
            for faq in faqs:
                question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                if query_lower in question.lower() or query_lower in answer.lower():
                    filtered_faqs.append(faq)

            # 2. 상품명으로 FAQ 검색
            for product in popular_products:
                product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                product_text_lower = product_text.lower()

                if query_lower in product_text_lower:
                    product_related_faqs = []
                    for faq in faqs:
                        question = faq.get(f"question_{current_lang}", faq.get("question_ko", ""))
                        answer = faq.get(f"answer_{current_lang}", faq.get("answer_ko", ""))
                        if product_text_lower in question.lower() or product_text_lower in answer.lower():
                            if faq not in filtered_faqs:
                                filtered_faqs.append(faq)
                                product_related_faqs.append(faq)

                    if not product_related_faqs:
                        matched_products.append(product)

            # 3. 인기 상품 검색
            for product in popular_products:
                product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
                if query_lower in product_text.lower():
                    matched_products.append(product)

            # 4. 화제의 소식 검색
            for topic in trending_topics:
                topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
                if query_lower in topic_text.lower():
                    matched_topics.append(topic)

            # 5. 회사 소개 검색
            if query_lower in company_info.lower():
                matched_info = True

            # 검색 결과 표시
            if filtered_faqs or matched_products or matched_topics or matched_info:
                _render_search_results(
                    filtered_faqs, matched_products, matched_topics, 
                    matched_info, company_info, query_lower, 
                    faq_search_query, faq_search_btn, current_lang, L)
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
                if not (matched_products or matched_topics or matched_info):
                    st.info(L["no_faq_results"])
            else:
                st.info(L.get("no_faq_for_company", f"{display_company}의 FAQ가 없습니다.").format(company=display_company))
    else:
        st.info(L.get("no_company_selected", "회사명을 검색하거나 선택해주세요."))


def _render_search_results(
    filtered_faqs: list, matched_products: list, matched_topics: list,
    matched_info: bool, company_info: str, query_lower: str,
    faq_search_query: str, faq_search_btn: bool, current_lang: str, L: dict):
    """검색 결과 렌더링"""
    # 매칭된 상품 표시
    if matched_products and not filtered_faqs:
        st.subheader(f"🔍 {L.get('related_products', '관련 상품')} ({len(matched_products)}{L.get('items', '개')})")
        st.info(L.get("no_faq_for_product", "해당 상품과 관련된 FAQ를 찾을 수 없습니다. 상품 정보만 표시됩니다."))
        for product in matched_products:
            product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
            product_score = product.get("score", 0)
            st.write(f"• **{product_text}** ({L.get('popularity', '인기도')}: {product_score})")
        st.markdown("---")

    # 매칭된 화제 소식 표시
    if matched_topics:
        st.subheader(f"🔍 {L.get('related_trending_news', '관련 화제 소식')} ({len(matched_topics)}{L.get('items', '개')})")
        for topic in matched_topics:
            topic_text = topic.get(f"text_{current_lang}", topic.get("text_ko", ""))
            topic_score = topic.get("score", 0)
            st.write(f"• **{topic_text}** ({L.get('trend_score', '화제도')}: {topic_score})")
        st.markdown("---")

    # 매칭된 회사 소개 표시
    if matched_info:
        st.subheader(f"🔍 {L.get('related_company_info', '관련 회사 소개 내용')}")
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
            highlighted = snippet.replace(query_lower, f"**{query_lower}**")
            st.write(highlighted)
        st.markdown("---")

