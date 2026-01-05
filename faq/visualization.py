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
회사 데이터 시각화 모듈
"""

from typing import Dict, Any

# Plotly 사용 가능 여부 확인
try:
    import plotly.graph_objects as go
    import plotly.express as px
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False


def visualize_company_data(company_data: Dict[str, Any], lang: str = "ko") -> Dict[str, Any]:
    """회사 데이터 시각화 (Plotly 사용)"""
    charts = {}
    
    if not IS_PLOTLY_AVAILABLE:
        return charts
    
    try:
        import plotly.graph_objects as go
        
        # 언어별 레이블
        lang_labels = {
            "ko": {
                "popular_products": "인기 상품",
                "product_name": "상품명",
                "popularity": "인기도",
                "trending_topics": "화제의 소식",
                "topic": "소식",
                "trend_score": "화제도"
            },
            "en": {
                "popular_products": "Popular Products",
                "product_name": "Product Name",
                "popularity": "Popularity",
                "trending_topics": "Trending News",
                "topic": "News",
                "trend_score": "Trend Score"
            },
            "ja": {
                "popular_products": "人気商品",
                "product_name": "商品名",
                "popularity": "人気度",
                "trending_topics": "話題のニュース",
                "topic": "ニュース",
                "trend_score": "話題度"
            }
        }
        labels = lang_labels.get(lang, lang_labels["ko"])
        
        # 인기 상품 시각화
        popular_products = company_data.get("popular_products", [])
        if popular_products:
            product_names = []
            product_scores = []
            for product in popular_products:
                name = product.get(f"text_{lang}", product.get("text_ko", ""))
                score = product.get("score", 0)
                if name:
                    product_names.append(name[:20])  # 이름이 너무 길면 자름
                    product_scores.append(score if score > 0 else 50)  # 기본값 50
            
            if product_names:
                # 막대 그래프
                fig_products_bar = go.Figure(data=[
                    go.Bar(
                        x=product_names,
                        y=product_scores,
                        marker_color='lightblue',
                        text=product_scores,
                        textposition='auto',
                    )
                ])
                fig_products_bar.update_layout(
                    title=f"{labels['popular_products']} (막대 그래프)",
                    xaxis_title=labels["product_name"],
                    yaxis_title=labels["popularity"],
                    height=300,
                    showlegend=False
                )
                charts["products_bar"] = fig_products_bar
                
                # 선형 그래프
                fig_products_line = go.Figure(data=[
                    go.Scatter(
                        x=product_names,
                        y=product_scores,
                        mode='lines+markers',
                        marker=dict(size=10, color='lightblue'),
                        line=dict(width=3, color='lightblue'),
                        text=product_scores,
                        textposition='top center',
                    )
                ])
                fig_products_line.update_layout(
                    title=f"{labels['popular_products']} (선형 그래프)",
                    xaxis_title=labels["product_name"],
                    yaxis_title=labels["popularity"],
                    height=300,
                    showlegend=False
                )
                charts["products_line"] = fig_products_line
        
        # 화제의 소식 시각화
        trending_topics = company_data.get("trending_topics", [])
        if trending_topics:
            topic_names = []
            topic_scores = []
            for topic in trending_topics:
                name = topic.get(f"text_{lang}", topic.get("text_ko", ""))
                score = topic.get("score", 0)
                if name:
                    topic_names.append(name[:20])
                    topic_scores.append(score if score > 0 else 50)
            
            if topic_names:
                # 막대 그래프
                fig_topics_bar = go.Figure(data=[
                    go.Bar(
                        x=topic_names,
                        y=topic_scores,
                        marker_color='lightcoral',
                        text=topic_scores,
                        textposition='auto',
                    )
                ])
                fig_topics_bar.update_layout(
                    title=f"{labels['trending_topics']} (막대 그래프)",
                    xaxis_title=labels["topic"],
                    yaxis_title=labels["trend_score"],
                    height=300,
                    showlegend=False
                )
                charts["topics_bar"] = fig_topics_bar
                
                # 선형 그래프
                fig_topics_line = go.Figure(data=[
                    go.Scatter(
                        x=topic_names,
                        y=topic_scores,
                        mode='lines+markers',
                        marker=dict(size=10, color='lightcoral'),
                        line=dict(width=3, color='lightcoral'),
                        text=topic_scores,
                        textposition='top center',
                    )
                ])
                fig_topics_line.update_layout(
                    title=f"{labels['trending_topics']} (선형 그래프)",
                    xaxis_title=labels["topic"],
                    yaxis_title=labels["trend_score"],
                    height=300,
                    showlegend=False
                )
                charts["topics_line"] = fig_topics_line
        
    except Exception as e:
        pass  # 시각화 실패해도 계속 진행
    
    return charts

