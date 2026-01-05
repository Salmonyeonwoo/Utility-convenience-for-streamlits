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
탭1: 회사 소개 및 시각화 렌더링 모듈
"""

import streamlit as st
from faq.visualization import visualize_company_data
from .tab1_products import render_products_section
from .tab1_topics import render_topics_section
from .tab1_ceo import render_ceo_section
from .tab1_interview import render_interview_section


def render_tab1_company_info(display_company: str, display_data: dict, current_lang: str, L: dict, faq_data: dict):
    """탭1: 회사 소개 및 시각화 렌더링"""
    if display_company and display_data:
        # 제목 표시
        st.markdown(f"#### {display_company} - {L['company_info']}")

        # 회사 소개 표시
        if display_data.get("company_info"):
            st.markdown(display_data["company_info"])

        # 시각화 차트 표시
        if display_data.get("popular_products") or display_data.get("trending_topics"):
            try:
                charts = visualize_company_data(
                    {
                        "popular_products": display_data.get("popular_products", []), 
                        "trending_topics": display_data.get("trending_topics", [])
                    }, 
                    current_lang
                )

                if charts and len(charts) > 0:
                    st.markdown(f"#### 📊 {L.get('visualization_chart', '📊 시각화 차트')}")
                    
                    # 제품 차트 표시
                    if "products_bar" in charts or "topics_bar" in charts:
                        col1_bar, col2_bar = st.columns(2)
                        
                        if "products_bar" in charts:
                            with col1_bar:
                                try:
                                    st.plotly_chart(
                                        charts["products_bar"], use_container_width=True, key=f"products_bar_chart")
                                except Exception as e:
                                    st.warning(f"제품 막대 그래프 표시 오류: {str(e)}")

                        if "topics_bar" in charts:
                            with col2_bar:
                                try:
                                    st.plotly_chart(
                                        charts["topics_bar"], use_container_width=True, key=f"topics_bar_chart")
                                except Exception as e:
                                    st.warning(f"화제 소식 막대 그래프 표시 오류: {str(e)}")

                    # 선형 그래프 표시
                    if "products_line" in charts or "topics_line" in charts:
                        col1_line, col2_line = st.columns(2)

                        if "products_line" in charts:
                            with col1_line:
                                try:
                                    st.plotly_chart(
                                        charts["products_line"], use_container_width=True, key=f"products_line_chart")
                                except Exception as e:
                                    st.warning(f"제품 선형 그래프 표시 오류: {str(e)}")

                        if "topics_line" in charts:
                            with col2_line:
                                try:
                                    st.plotly_chart(
                                        charts["topics_line"], use_container_width=True, key=f"topics_line_chart")
                                except Exception as e:
                                    st.warning(f"화제 소식 선형 그래프 표시 오류: {str(e)}")
            except Exception as e:
                st.warning(f"차트 생성 중 오류가 발생했습니다: {str(e)}")

        # 인기 상품 목록
        if display_data.get("popular_products"):
            render_products_section(display_data["popular_products"], current_lang, L)

        # 화제의 소식 목록
        if display_data.get("trending_topics"):
            render_topics_section(
                display_data["trending_topics"], 
                display_company, 
                current_lang, 
                L, 
                faq_data
            )

        # CEO/대표이사 정보
        if display_data.get("ceo_info"):
            render_ceo_section(display_data["ceo_info"], current_lang, L)

        # 면접 질문 목록
        if display_data.get("interview_questions"):
            render_interview_section(display_data["interview_questions"], current_lang, L)
    else:
        st.info(L["company_search_or_select"])

