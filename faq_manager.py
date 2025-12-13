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
FAQ 관리 모듈
회사 정보, FAQ 검색, 제품 이미지 생성 등의 기능을 제공합니다.
"""

import os
import json
import hashlib
import requests
import re
from typing import List, Dict, Any
from openai import OpenAI

from config import (
    FAQ_DB_FILE,
    PRODUCT_IMAGE_CACHE_FILE,
    PRODUCT_IMAGE_DIR
)
from utils import _load_json, _save_json

# Plotly 사용 가능 여부 확인
try:
    import plotly.graph_objects as go
    import plotly.express as px
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False

# run_llm과 get_api_key는 llm_client에서 import
# 순환 참조 방지를 위해 함수 내부에서 import

def load_faq_database() -> Dict[str, Any]:
    """FAQ 데이터베이스 로드"""
    return _load_json(FAQ_DB_FILE, {"companies": {}})



def save_faq_database(faq_data: Dict[str, Any]):
    """FAQ 데이터베이스 저장"""
    _save_json(FAQ_DB_FILE, faq_data)



def get_company_info_faq(company: str, lang: str = "ko") -> Dict[str, Any]:
    """회사 소개 및 FAQ 가져오기"""
    faq_data = load_faq_database()
    if company in faq_data.get("companies", {}):
        company_data = faq_data["companies"][company]
        return {
            "info": company_data.get(f"info_{lang}", company_data.get("info_ko", "")),
            "popular_products": company_data.get("popular_products", []),
            "trending_topics": company_data.get("trending_topics", []),
            "faqs": company_data.get("faqs", []),
            "interview_questions": company_data.get("interview_questions", []),
            "ceo_info": company_data.get("ceo_info", {})
        }
    return {"info": "", "popular_products": [], "trending_topics": [], "faqs": [], "interview_questions": [], "ceo_info": {}}



def visualize_company_data(company_data: Dict[str, Any], lang: str = "ko") -> Dict[str, Any]:
    """회사 데이터 시각화 (Plotly 사용)"""
    charts = {}
    
    if not IS_PLOTLY_AVAILABLE:
        return charts
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
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
                
                # 선형 그래프 (LSTM 스타일)
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



def load_product_image_cache() -> Dict[str, str]:
    """제품 이미지 캐시 로드"""
    return _load_json(PRODUCT_IMAGE_CACHE_FILE, {})



def save_product_image_cache(cache_data: Dict[str, str]):
    """제품 이미지 캐시 저장"""
    _save_json(PRODUCT_IMAGE_CACHE_FILE, cache_data)



def generate_product_image_prompt(product_name: str) -> str:
    """제품명을 기반으로 이미지 생성 프롬프트 생성"""
    product_lower = product_name.lower()
    
    # 언어별 제품명 추출 (한국어, 영어, 일본어)
    lang_versions = []
    if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in product_name):  # 한글 포함
        lang_versions.append(("ko", product_name))
    if any(ord(c) >= 0x3040 and ord(c) <= 0x309F or ord(c) >= 0x30A0 and ord(c) <= 0x30FF for c in product_name):  # 일본어 포함
        lang_versions.append(("ja", product_name))
    if any(c.isalpha() and ord(c) < 128 for c in product_name):  # 영어 포함
        lang_versions.append(("en", product_name))
    
    # 기본 프롬프트 구성
    base_prompt = f"Professional product photo of {product_name}, "
    
    # 카테고리별 상세 프롬프트 추가
    if "디즈니" in product_name or "disney" in product_lower or "ディズニー" in product_name:
        return f"Beautiful, vibrant photo of Disneyland theme park entrance ticket for {product_name}, magical atmosphere, colorful, professional product photography, high quality, commercial style"
    elif "유니버셜" in product_name or "universal" in product_lower or "ユニバーサル" in product_name:
        return f"Professional photo of Universal Studios theme park ticket for {product_name}, exciting theme park atmosphere, high quality product photography, commercial style"
    elif "스카이트리" in product_name or "skytree" in product_lower or "도쿄 타워" in product_name or "tokyo tower" in product_lower or "スカイツリー" in product_name or "東京タワー" in product_name:
        return f"Beautiful photo of Tokyo Skytree or Tokyo Tower admission ticket for {product_name}, modern Tokyo cityscape background, professional product photography, high quality"
    elif "갤럭시" in product_name or "galaxy" in product_lower:
        return f"Professional product photo of Samsung Galaxy smartphone {product_name}, sleek modern design, premium quality, white background, commercial product photography, high resolution"
    elif "qled" in product_lower or "tv" in product_lower or "티비" in product_name or "텔레비전" in product_name:
        return f"Professional product photo of Samsung QLED TV {product_name}, modern sleek design, premium quality, minimalist background, commercial product photography, high resolution"
    elif "티켓" in product_name or "ticket" in product_lower or "チケット" in product_name:
        return f"Professional photo of admission ticket for {product_name}, clean design, high quality product photography, commercial style"
    elif "호텔" in product_name or "hotel" in product_lower or "ホテル" in product_name:
        return f"Beautiful photo of hotel booking voucher or hotel room for {product_name}, luxurious atmosphere, professional photography, high quality"
    elif "항공" in product_name or "flight" in product_lower or "航空" in product_name:
        return f"Professional photo of airline ticket or boarding pass for {product_name}, clean design, high quality product photography"
    elif "여행" in product_name or "travel" in product_lower or "투어" in product_name or "tour" in product_lower or "旅行" in product_name or "ツアー" in product_name:
        return f"Beautiful travel-related photo for {product_name}, scenic destination, professional photography, high quality, travel brochure style"
    elif "음식" in product_name or "food" in product_lower or "레스토랑" in product_name or "restaurant" in product_lower or "食事" in product_name or "レストラン" in product_name:
        return f"Appetizing food photo for {product_name}, restaurant dish, professional food photography, high quality, commercial style"
    else:
        return f"Professional product photo of {product_name}, clean background, high quality product photography, commercial style, well-lit"



def generate_product_image_with_ai(product_name: str) -> str:
    """AI를 사용하여 제품 이미지 생성 (DALL-E 사용)"""
    try:
        # 캐시 확인
        cache = load_product_image_cache()
        cache_key = product_name.lower().strip()
        
        if cache_key in cache:
            cached_path = cache[cache_key]
            if os.path.exists(cached_path):
                return cached_path
        
        # OpenAI API 키 확인
        from llm_client import get_api_key
        openai_key = get_api_key("openai")
        if not openai_key:
            # OpenAI 키가 없으면 기본 이미지 URL 반환
            return ""
        
        # 이미지 생성 프롬프트 생성
        image_prompt = generate_product_image_prompt(product_name)
        
        # DALL-E API 호출
        client = OpenAI(api_key=openai_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # 생성된 이미지 URL 가져오기
        image_url = response.data[0].url
        
        # 이미지를 로컬에 저장
        import hashlib
        image_hash = hashlib.md5(product_name.encode('utf-8')).hexdigest()
        image_filename = f"{image_hash}.png"
        image_path = os.path.join(PRODUCT_IMAGE_DIR, image_filename)
        
        # 이미지 다운로드 및 저장
        img_response = requests.get(image_url, timeout=10)
        if img_response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(img_response.content)
            
            # 캐시에 저장
            cache[cache_key] = image_path
            save_product_image_cache(cache)
            
            return image_path
        else:
            return ""
            
    except Exception as e:
        print(f"⚠️ AI 이미지 생성 실패 ({product_name}): {e}")
        return ""



def get_product_image_url(product_name: str) -> str:
    """상품명을 기반으로 이미지 URL 생성 - AI 이미지 생성 우선 사용"""
    try:
        # ⭐ 1순위: AI 이미지 생성 시도 (DALL-E)
        ai_image_path = generate_product_image_with_ai(product_name)
        if ai_image_path and os.path.exists(ai_image_path):
            return ai_image_path
        
        # ⭐ 2순위: 기존 키워드 기반 이미지 매칭 (폴백)
        product_lower = product_name.lower()
        
        # 디즈니랜드 관련 상품 - 미키마우스 이미지 (한국어, 영어, 일본어 모두 체크)
        if ("디즈니" in product_name or "disney" in product_lower or "disneyland" in product_lower or 
            "tokyo disneyland" in product_lower or "hong kong disneyland" in product_lower or
            "ディズニー" in product_name or "ディズニーランド" in product_name):
            return "https://images.unsplash.com/photo-1606813907291-d86efa9b94db?w=400&h=300&fit=crop&q=80"
        
        # 유니버셜 스튜디오 관련 상품 - 유니버셜 로고/지구본 이미지 (한국어, 영어, 일본어 모두 체크)
        if ("유니버셜" in product_name or "universal" in product_lower or "universal studio" in product_lower or
            "universal studios" in product_lower or "ユニバーサル" in product_name or "ユニバーサルスタジオ" in product_name):
            return "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?w=400&h=300&fit=crop&q=80"
        
        # 도쿄 스카이트리 관련 상품 - 스카이트리 건물 이미지 (한국어, 영어, 일본어 모두 체크)
        if ("스카이트리" in product_name or "skytree" in product_lower or "도쿄 타워" in product_name or 
            "tokyo tower" in product_lower or "tokyo skytree" in product_lower or
            "スカイツリー" in product_name or "東京タワー" in product_name or "東京スカイツリー" in product_name):
            return "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400&h=300&fit=crop&q=80"
        
        # 홍콩 관련 상품 (디즈니랜드 외)
        if ("홍콩" in product_name or "hong kong" in product_lower or "香港" in product_name):
            if "disney" not in product_lower and "디즈니" not in product_name:
                # 홍콩 공항 익스프레스 등
                return "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?w=400&h=300&fit=crop&q=80"
        
        # 방콕 관련 상품 (한국어, 영어, 일본어 모두 체크)
        if ("방콕" in product_name or "bangkok" in product_lower or "バンコク" in product_name):
            return "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?w=400&h=300&fit=crop&q=80"
        
        # 삼성 갤럭시 S 시리즈 관련 상품
        if ("갤럭시 s" in product_lower or "galaxy s" in product_lower or "galaxy s24" in product_lower or
            "galaxy s23" in product_lower or "galaxy s22" in product_lower or "galaxy s21" in product_lower or
            "galaxy s20" in product_lower or "samsung galaxy s" in product_lower):
            return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
        
        # 삼성 갤럭시 노트 시리즈 관련 상품
        if ("갤럭시 노트" in product_lower or "galaxy note" in product_lower or "galaxy note24" in product_lower or
            "galaxy note23" in product_lower or "galaxy note22" in product_lower or "galaxy note21" in product_lower or
            "galaxy note20" in product_lower or "samsung galaxy note" in product_lower):
            return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
        
        # 삼성 QLED TV 관련 상품
        if ("qled" in product_lower or "삼성 qled" in product_lower or "samsung qled" in product_lower or
            "삼성 tv" in product_lower or "samsung tv" in product_lower):
            return "https://images.unsplash.com/photo-1593359677879-a4b92c0a3b8b?w=400&h=300&fit=crop&q=80"
        
        # 삼성 제품 일반 (위에서 매칭되지 않은 경우)
        if ("삼성" in product_name or "samsung" in product_lower):
            # 스마트폰 관련
            if ("스마트폰" in product_name or "smartphone" in product_lower or "phone" in product_lower or
                "갤럭시" in product_name or "galaxy" in product_lower):
                return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
            # TV 관련
            elif ("tv" in product_lower or "티비" in product_name or "텔레비전" in product_name):
                return "https://images.unsplash.com/photo-1593359677879-a4b92c0a3b8b?w=400&h=300&fit=crop&q=80"
            # 기본 삼성 제품
            else:
                return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
        
        # 티켓 관련 상품
        if ("티켓" in product_name or "ticket" in product_lower or "チケット" in product_name):
            return "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80"
        
        # 호텔 관련 상품
        if ("호텔" in product_name or "hotel" in product_lower or "ホテル" in product_name):
            return "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop&q=80"
        
        # 항공 관련 상품
        if ("항공" in product_name or "flight" in product_lower or "航空" in product_name or "airline" in product_lower):
            return "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=300&fit=crop&q=80"
        
        # 여행/투어 관련 상품
        if ("여행" in product_name or "travel" in product_lower or "투어" in product_name or "tour" in product_lower or
            "旅行" in product_name or "ツアー" in product_name):
            return "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&h=300&fit=crop&q=80"
        
        # 음식/레스토랑 관련 상품
        if ("음식" in product_name or "food" in product_lower or "레스토랑" in product_name or "restaurant" in product_lower or
            "食事" in product_name or "レストラン" in product_name):
            return "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400&h=300&fit=crop&q=80"
        
        # 기본값: 상품명 기반으로 일관된 이미지 생성
        # 제품 카테고리를 추론하여 적절한 이미지 선택
        import hashlib
        
        # 제품명을 해시하여 일관된 이미지 ID 생성
        hash_obj = hashlib.md5(product_name.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        image_seed = hash_int % 1000
        
        # 카테고리별 Unsplash 이미지 (더 안정적인 이미지 ID 사용)
        category_images = [
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80",  # 티켓/여행
            "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&h=300&fit=crop&q=80",  # 여행지
            "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop&q=80",  # 호텔
            "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=300&fit=crop&q=80",  # 항공
            "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400&h=300&fit=crop&q=80",  # 음식
        ]
        
        # 해시 기반으로 일관된 이미지 선택
        selected_image = category_images[image_seed % len(category_images)]
        return selected_image
    except Exception:
        return ""



def search_faq(faq_data: Dict[str, Any], company: str, query: str, lang: str = "ko") -> List[Dict[str, str]]:
    """FAQ 검색"""
    if not query or not query.strip():
        return []
    
    results = []
    query_lower = query.lower().strip()
    
    # 회사별 FAQ 검색
    if company and company in faq_data.get("companies", {}):
        company_faqs = faq_data["companies"][company].get("faqs", [])
        for faq in company_faqs:
            question = faq.get(f"question_{lang}", faq.get("question_ko", ""))
            answer = faq.get(f"answer_{lang}", faq.get("answer_ko", ""))
            
            if query_lower in question.lower() or query_lower in answer.lower():
                results.append({
                    "question": question,
                    "answer": answer,
                    "company": company
                })
    
    # 기본 FAQ 검색 (회사가 없거나 기본 설정인 경우)
    default_settings_texts = ["기본 설정", "Default Settings", "デフォルト設定"]
    if not company or company in default_settings_texts:
        default_faqs = faq_data.get("default", {}).get("faqs", [])
        for faq in default_faqs:
            question = faq.get(f"question_{lang}", faq.get("question_ko", ""))
            answer = faq.get(f"answer_{lang}", faq.get("answer_ko", ""))
            
            if query_lower in question.lower() or query_lower in answer.lower():
                results.append({
                    "question": question,
                    "answer": answer,
                    "company": "기본"
                })
    
    return results



def get_common_product_faqs(company_name: str, lang: str = "ko") -> List[Dict[str, str]]:
    """공동 대표 제품 FAQ 반환"""
    company_lower = company_name.lower()
    common_faqs = []
    
    # Klook 공동 제품 FAQ
    if "klook" in company_lower or "클룩" in company_name:
        if lang == "ko":
            common_faqs = [
                {
                    "question_ko": "eSIM은 어떤 국가에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 전 세계 대부분의 국가에서 사용 가능합니다. 주요 여행지인 유럽, 아시아, 아메리카, 오세아니아 등 전 세계 190개 이상의 국가와 지역에서 사용할 수 있습니다. 각 국가별 데이터 요금제와 사용 가능 여부는 상품 페이지에서 확인하실 수 있습니다.",
                    "question_en": "Which countries can I use eSIM in?",
                    "answer_en": "eSIM can be used in most countries around the world. It is available in over 190 countries and regions including major travel destinations in Europe, Asia, Americas, and Oceania. Data plans and availability for each country can be checked on the product page.",
                    "question_ja": "eSIMはどの国で使用できますか？",
                    "answer_ja": "eSIMは世界中のほとんどの国で使用できます。主要な旅行先であるヨーロッパ、アジア、アメリカ、オセアニアなど、世界190以上の国と地域で使用可能です。各国のデータプランと利用可否は商品ページでご確認いただけます。"
                },
                {
                    "question_ko": "eSIM 활성화는 어떻게 하나요?",
                    "answer_ko": "eSIM 활성화는 매우 간단합니다. 1) 구매 후 이메일로 받은 QR 코드를 확인하세요. 2) 여행지에 도착한 후 스마트폰 설정에서 eSIM을 추가하세요. 3) QR 코드를 스캔하거나 수동으로 입력하세요. 4) 데이터 요금제를 활성화하세요. 대부분의 경우 자동으로 활성화되며, 수동 활성화가 필요한 경우 상품 페이지의 안내를 따르시면 됩니다.",
                    "question_en": "How do I activate eSIM?",
                    "answer_en": "Activating eSIM is very simple. 1) Check the QR code received via email after purchase. 2) After arriving at your destination, add eSIM in your smartphone settings. 3) Scan the QR code or enter manually. 4) Activate your data plan. In most cases, it activates automatically, and if manual activation is required, please follow the instructions on the product page.",
                    "question_ja": "eSIMの有効化はどうすればいいですか？",
                    "answer_ja": "eSIMの有効化は非常に簡単です。1) 購入後メールで受け取ったQRコードを確認してください。2) 旅行先に到着したら、スマートフォンの設定でeSIMを追加してください。3) QRコードをスキャンするか、手動で入力してください。4) データプランを有効化してください。ほとんどの場合、自動的に有効化されますが、手動有効化が必要な場合は、商品ページの案内に従ってください。"
                },
                {
                    "question_ko": "eSIM을 여러 국가에서 사용할 수 있나요?",
                    "answer_ko": "네, 일부 eSIM 요금제는 여러 국가에서 사용할 수 있는 글로벌 플랜을 제공합니다. 지역별 플랜(예: 유럽 여러 국가, 아시아 여러 국가)도 있습니다. 구매 전 상품 설명에서 지원 국가 목록을 확인하시기 바랍니다. 단일 국가 전용 플랜도 있으므로 여행 계획에 맞는 플랜을 선택하시면 됩니다.",
                    "question_en": "Can I use eSIM in multiple countries?",
                    "answer_en": "Yes, some eSIM plans offer global plans that can be used in multiple countries. There are also regional plans (e.g., multiple European countries, multiple Asian countries). Please check the list of supported countries in the product description before purchase. There are also single-country exclusive plans, so please choose a plan that suits your travel plans.",
                    "question_ja": "eSIMを複数の国で使用できますか？",
                    "answer_ja": "はい、一部のeSIMプランは複数の国で使用できるグローバルプランを提供しています。地域別プラン（例：ヨーロッパ複数国、アジア複数国）もあります。購入前に商品説明でサポート国リストをご確認ください。単一国専用プランもあるため、旅行計画に合ったプランを選択してください。"
                },
                {
                    "question_ko": "eSIM은 어떤 기기에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 eSIM 기능을 지원하는 스마트폰, 태블릿, 스마트워치 등에서 사용할 수 있습니다. 주요 호환 기종은 다음과 같습니다:\n\n**iPhone:**\n- iPhone XS, XS Max, XR 이후 모델 (iPhone 14 시리즈 이상 권장)\n- iPhone SE (2020년 이후 모델)\n\n**Android:**\n- Google Pixel 3 이후 모델\n- Samsung Galaxy S20 시리즈 이후 (S21, S22, S23, S24, S25 등)\n- Samsung Galaxy Note 20 시리즈 이후\n- Samsung Galaxy Z Fold, Z Flip 시리즈\n- Samsung Galaxy Tab 시리즈 (일부 모델)\n- OnePlus 6 이후 모델\n- Xiaomi, Huawei, Oppo 등 주요 브랜드의 최신 모델\n\n**기타:**\n- iPad Pro (2018년 이후), iPad Air (2020년 이후), iPad mini (2019년 이후)\n- Apple Watch Series 3 이후 (셀룰러 모델)\n\n기기 호환성은 제조사와 모델에 따라 다를 수 있으므로, 구매 전 상품 페이지에서 사용하시는 기기 모델의 호환 여부를 확인하시기 바랍니다. 또한 일부 기기는 특정 국가나 통신사에서만 eSIM을 지원할 수 있습니다.",
                    "question_en": "Which devices support eSIM?",
                    "answer_en": "eSIM can be used on smartphones, tablets, smartwatches, and other devices that support eSIM functionality. Main compatible devices include:\n\n**iPhone:**\n- iPhone XS, XS Max, XR and later models (iPhone 14 series and above recommended)\n- iPhone SE (2020 and later models)\n\n**Android:**\n- Google Pixel 3 and later models\n- Samsung Galaxy S20 series and later (S21, S22, S23, S24, S25, etc.)\n- Samsung Galaxy Note 20 series and later\n- Samsung Galaxy Z Fold, Z Flip series\n- Samsung Galaxy Tab series (some models)\n- OnePlus 6 and later models\n- Latest models from Xiaomi, Huawei, Oppo, and other major brands\n\n**Others:**\n- iPad Pro (2018 and later), iPad Air (2020 and later), iPad mini (2019 and later)\n- Apple Watch Series 3 and later (cellular models)\n\nDevice compatibility may vary by manufacturer and model, so please check the product page before purchase to confirm compatibility with your device model. Some devices may only support eSIM in specific countries or with specific carriers.",
                    "question_ja": "eSIMはどのデバイスで使用できますか？",
                    "answer_ja": "eSIMは、eSIM機能をサポートするスマートフォン、タブレット、スマートウォッチなどで使用できます。主な互換デバイスは以下の通りです：\n\n**iPhone:**\n- iPhone XS、XS Max、XR以降のモデル（iPhone 14シリーズ以降推奨）\n- iPhone SE（2020年以降のモデル）\n\n**Android:**\n- Google Pixel 3以降のモデル\n- Samsung Galaxy S20シリーズ以降（S21、S22、S23、S24、S25など）\n- Samsung Galaxy Note 20シリーズ以降\n- Samsung Galaxy Z Fold、Z Flipシリーズ\n- Samsung Galaxy Tabシリーズ（一部モデル）\n- OnePlus 6以降のモデル\n- Xiaomi、Huawei、Oppoなどの主要ブランドの最新モデル\n\n**その他:**\n- iPad Pro（2018年以降）、iPad Air（2020年以降）、iPad mini（2019年以降）\n- Apple Watch Series 3以降（セルラーモデル）\n\nデバイスの互換性はメーカーやモデルによって異なる場合があるため、購入前に商品ページで使用するデバイスモデルの互換性をご確認ください。また、一部のデバイスは特定の国や通信事業者でのみeSIMをサポートする場合があります。"
                }
            ]
        elif lang == "en":
            common_faqs = [
                {
                    "question_en": "Which countries can I use eSIM in?",
                    "answer_en": "eSIM can be used in most countries around the world. It is available in over 190 countries and regions including major travel destinations in Europe, Asia, Americas, and Oceania. Data plans and availability for each country can be checked on the product page.",
                    "question_ko": "eSIM은 어떤 국가에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 전 세계 대부분의 국가에서 사용 가능합니다.",
                    "question_ja": "eSIMはどの国で使用できますか？",
                    "answer_ja": "eSIMは世界中のほとんどの国で使用できます。"
                },
                {
                    "question_en": "How do I activate eSIM?",
                    "answer_en": "Activating eSIM is very simple. 1) Check the QR code received via email after purchase. 2) After arriving at your destination, add eSIM in your smartphone settings. 3) Scan the QR code or enter manually. 4) Activate your data plan.",
                    "question_ko": "eSIM 활성화는 어떻게 하나요?",
                    "answer_ko": "eSIM 활성화는 매우 간단합니다.",
                    "question_ja": "eSIMの有効化はどうすればいいですか？",
                    "answer_ja": "eSIMの有効化は非常に簡単です。"
                },
                {
                    "question_en": "Which devices support eSIM?",
                    "answer_en": "eSIM can be used on smartphones, tablets, smartwatches, and other devices that support eSIM functionality. Main compatible devices include:\n\n**iPhone:**\n- iPhone XS, XS Max, XR and later models (iPhone 14 series and above recommended)\n- iPhone SE (2020 and later models)\n\n**Android:**\n- Google Pixel 3 and later models\n- Samsung Galaxy S20 series and later (S21, S22, S23, S24, S25, etc.)\n- Samsung Galaxy Note 20 series and later\n- Samsung Galaxy Z Fold, Z Flip series\n- Samsung Galaxy Tab series (some models)\n- OnePlus 6 and later models\n- Latest models from Xiaomi, Huawei, Oppo, and other major brands\n\n**Others:**\n- iPad Pro (2018 and later), iPad Air (2020 and later), iPad mini (2019 and later)\n- Apple Watch Series 3 and later (cellular models)\n\nDevice compatibility may vary by manufacturer and model, so please check the product page before purchase to confirm compatibility with your device model. Some devices may only support eSIM in specific countries or with specific carriers.",
                    "question_ko": "eSIM은 어떤 기기에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 eSIM 기능을 지원하는 스마트폰, 태블릿, 스마트워치 등에서 사용할 수 있습니다.",
                    "question_ja": "eSIMはどのデバイスで使用できますか？",
                    "answer_ja": "eSIMは、eSIM機能をサポートするスマートフォン、タブレット、スマートウォッチなどで使用できます。"
                }
            ]
        else:  # ja
            common_faqs = [
                {
                    "question_ja": "eSIMはどの国で使用できますか？",
                    "answer_ja": "eSIMは世界中のほとんどの国で使用できます。主要な旅行先であるヨーロッパ、アジア、アメリカ、オセアニアなど、世界190以上の国と地域で使用可能です。",
                    "question_ko": "eSIM은 어떤 국가에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 전 세계 대부분의 국가에서 사용 가능합니다.",
                    "question_en": "Which countries can I use eSIM in?",
                    "answer_en": "eSIM can be used in most countries around the world."
                },
                {
                    "question_ja": "eSIMはどのデバイスで使用できますか？",
                    "answer_ja": "eSIMは、eSIM機能をサポートするスマートフォン、タブレット、スマートウォッチなどで使用できます。主な互換デバイスは以下の通りです：\n\n**iPhone:**\n- iPhone XS、XS Max、XR以降のモデル（iPhone 14シリーズ以降推奨）\n- iPhone SE（2020年以降のモデル）\n\n**Android:**\n- Google Pixel 3以降のモデル\n- Samsung Galaxy S20シリーズ以降（S21、S22、S23、S24、S25など）\n- Samsung Galaxy Note 20シリーズ以降\n- Samsung Galaxy Z Fold、Z Flipシリーズ\n- Samsung Galaxy Tabシリーズ（一部モデル）\n- OnePlus 6以降のモデル\n- Xiaomi、Huawei、Oppoなどの主要ブランドの最新モデル\n\n**その他:**\n- iPad Pro（2018年以降）、iPad Air（2020年以降）、iPad mini（2019年以降）\n- Apple Watch Series 3以降（セルラーモデル）\n\nデバイスの互換性はメーカーやモデルによって異なる場合があるため、購入前に商品ページで使用するデバイスモデルの互換性をご確認ください。また、一部のデバイスは特定の国や通信事業者でのみeSIMをサポートする場合があります。",
                    "question_ko": "eSIM은 어떤 기기에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 eSIM 기능을 지원하는 스마트폰, 태블릿, 스마트워치 등에서 사용할 수 있습니다.",
                    "question_en": "Which devices support eSIM?",
                    "answer_en": "eSIM can be used on smartphones, tablets, smartwatches, and other devices that support eSIM functionality."
                }
            ]
    
    # 삼성 공동 제품 FAQ
    elif "samsung" in company_lower or "삼성" in company_name:
        if lang == "ko":
            common_faqs = [
                {
                    "question_ko": "Galaxy S25 Ultra의 주요 특징은 무엇인가요?",
                    "answer_ko": "Galaxy S25 Ultra는 삼성의 최신 플래그십 스마트폰으로, 고성능 프로세서, 고해상도 카메라 시스템, 긴 배터리 수명, 빠른 충전 기능을 제공합니다. 특히 AI 기능이 강화되어 사진 촬영, 생산성 향상, 일상 작업 자동화에 도움이 됩니다.",
                    "question_en": "What are the main features of Galaxy S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra is Samsung's latest flagship smartphone, offering high-performance processor, high-resolution camera system, long battery life, and fast charging. AI features are particularly enhanced to help with photography, productivity, and daily task automation.",
                    "question_ja": "Galaxy S25 Ultraの主な特徴は何ですか？",
                    "answer_ja": "Galaxy S25 Ultraはサムスンの最新フラグシップスマートフォンで、高性能プロセッサー、高解像度カメラシステム、長いバッテリー寿命、高速充電機能を提供します。特にAI機能が強化され、写真撮影、生産性向上、日常作業の自動化に役立ちます。"
                },
                {
                    "question_ko": "신규 출시 예정 제품은 언제 출시되나요?",
                    "answer_ko": "삼성은 정기적으로 신제품을 출시합니다. 정확한 출시 일정은 공식 발표를 통해 확인하실 수 있으며, 일반적으로 갤럭시 시리즈는 연 1-2회 주요 업데이트가 있습니다. 신제품 출시 소식은 삼성 공식 웹사이트나 공식 채널을 통해 확인하시기 바랍니다.",
                    "question_en": "When will the new products be released?",
                    "answer_en": "Samsung regularly releases new products. Exact release schedules can be confirmed through official announcements, and generally, the Galaxy series has 1-2 major updates per year. Please check Samsung's official website or official channels for new product release news.",
                    "question_ja": "新製品はいつ発売されますか？",
                    "answer_ja": "サムスンは定期的に新製品を発売しています。正確な発売スケジュールは公式発表で確認でき、一般的にギャラクシーシリーズは年間1-2回の主要アップデートがあります。新製品発売のニュースはサムスン公式ウェブサイトまたは公式チャンネルでご確認ください。"
                }
            ]
        elif lang == "en":
            common_faqs = [
                {
                    "question_en": "What are the main features of Galaxy S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra is Samsung's latest flagship smartphone, offering high-performance processor, high-resolution camera system, long battery life, and fast charging.",
                    "question_ko": "Galaxy S25 Ultra의 주요 특징은 무엇인가요?",
                    "answer_ko": "Galaxy S25 Ultra는 삼성의 최신 플래그십 스마트폰입니다.",
                    "question_ja": "Galaxy S25 Ultraの主な特徴は何ですか？",
                    "answer_ja": "Galaxy S25 Ultraはサムスンの最新フラグシップスマートフォンです。"
                }
            ]
        else:  # ja
            common_faqs = [
                {
                    "question_ja": "Galaxy S25 Ultraの主な特徴は何ですか？",
                    "answer_ja": "Galaxy S25 Ultraはサムスンの最新フラグシップスマートフォンで、高性能プロセッサー、高解像度カメラシステムを提供します。",
                    "question_ko": "Galaxy S25 Ultra의 주요 특징은 무엇인가요?",
                    "answer_ko": "Galaxy S25 Ultra는 삼성의 최신 플래그십 스마트폰입니다.",
                    "question_en": "What are the main features of Galaxy S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra is Samsung's latest flagship smartphone."
                }
            ]
    
    return common_faqs



def generate_company_info_with_llm(company_name: str, lang: str = "ko") -> Dict[str, Any]:
    """LLM을 사용하여 회사 정보 생성"""
    lang_prompts = {
        "ko": f"""다음 회사에 대한 상세 정보를 제공해주세요: {company_name}

다음 형식으로 JSON으로 응답해주세요:
{{
    "company_info": "회사 소개 (500자 이상)",
    "popular_products": [
        {{"text_ko": "상품명1", "score": 85, "image_url": ""}},
        {{"text_ko": "상품명2", "score": 80, "image_url": ""}},
        {{"text_ko": "상품명3", "score": 75, "image_url": ""}}
    ],
    "trending_topics": [
        {{"text_ko": "화제 소식1", "score": 90, "detail_ko": "화제 소식1에 대한 상세 내용입니다. 구체적인 설명과 배경 정보를 포함합니다."}},
        {{"text_ko": "화제 소식2", "score": 85, "detail_ko": "화제 소식2에 대한 상세 내용입니다. 구체적인 설명과 배경 정보를 포함합니다."}},
        {{"text_ko": "화제 소식3", "score": 80, "detail_ko": "화제 소식3에 대한 상세 내용입니다. 구체적인 설명과 배경 정보를 포함합니다."}}
    ],
    "faqs": [
        {{"question_ko": "질문1", "answer_ko": "답변1"}},
        {{"question_ko": "질문2", "answer_ko": "답변2"}},
        {{"question_ko": "질문3", "answer_ko": "답변3"}},
        {{"question_ko": "질문4", "answer_ko": "답변4"}},
        {{"question_ko": "질문5", "answer_ko": "답변5"}},
        {{"question_ko": "질문6", "answer_ko": "답변6"}},
        {{"question_ko": "질문7", "answer_ko": "답변7"}},
        {{"question_ko": "질문8", "answer_ko": "답변8"}},
        {{"question_ko": "질문9", "answer_ko": "답변9"}},
        {{"question_ko": "질문10", "answer_ko": "답변10"}}
    ],
    "interview_questions": [
        {{"question_ko": "면접 질문1", "answer_ko": "면접 질문1에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "회사 이해"}},
        {{"question_ko": "면접 질문2", "answer_ko": "면접 질문2에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "업무 관련"}},
        {{"question_ko": "면접 질문3", "answer_ko": "면접 질문3에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "직무 이해"}},
        {{"question_ko": "면접 질문4", "answer_ko": "면접 질문4에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "회사 이해"}},
        {{"question_ko": "면접 질문5", "answer_ko": "면접 질문5에 대한 상세한 답변입니다. 회사의 비전, 문화, 업무 환경 등을 고려하여 구체적이고 실용적인 답변을 제공합니다. (200자 이상)", "category_ko": "업무 관련"}}
    ],
    "ceo_info": {{
        "name_ko": "대표이사/CEO 이름",
        "position_ko": "직책 (예: 대표이사, CEO, 공동대표이사 등)",
        "bio_ko": "대표이사/CEO에 대한 상세한 소개입니다. 학력, 경력, 주요 성과, 리더십 스타일 등을 포함하여 300자 이상 작성해주세요.",
        "tenure_ko": "재임 기간 (예: 2020년 ~ 현재)",
        "education_ko": "학력 정보",
        "career_ko": "주요 경력 및 성과"
    }}
}}

FAQ는 10개를 생성해주세요. 실제로 자주 묻는 질문과 답변을 포함해주세요.
면접 질문은 5개 이상 생성해주세요. 실제 면접에서 나올 만한 핵심 질문들을 포함하고, 각 질문에 대한 상세한 답변(200자 이상)과 카테고리(회사 이해, 업무 관련, 직무 이해 등)를 제공해주세요.
CEO/대표이사 정보는 현재 재임 중인 대표이사 또는 CEO의 실제 정보를 포함해주세요. 이름, 직책, 상세 소개(300자 이상), 재임 기간, 학력, 주요 경력을 포함해주세요.""",
        "en": f"""Please provide detailed information about the following company: {company_name}

Respond in JSON format as follows:
{{
    "company_info": "Company introduction (500+ characters)",
    "popular_products": [
        {{"text_en": "Product1", "score": 85, "image_url": ""}},
        {{"text_en": "Product2", "score": 80, "image_url": ""}},
        {{"text_en": "Product3", "score": 75, "image_url": ""}}
    ],
    "trending_topics": [
        {{"text_en": "Trending news1", "score": 90, "detail_en": "Detailed content about trending news1, including specific explanations and background information."}},
        {{"text_en": "Trending news2", "score": 85, "detail_en": "Detailed content about trending news2, including specific explanations and background information."}},
        {{"text_en": "Trending news3", "score": 80, "detail_en": "Detailed content about trending news3, including specific explanations and background information."}}
    ],
    "faqs": [
        {{"question_en": "Question1", "answer_en": "Answer1"}},
        {{"question_en": "Question2", "answer_en": "Answer2"}},
        {{"question_en": "Question3", "answer_en": "Answer3"}},
        {{"question_en": "Question4", "answer_en": "Answer4"}},
        {{"question_en": "Question5", "answer_en": "Answer5"}},
        {{"question_en": "Question6", "answer_en": "Answer6"}},
        {{"question_en": "Question7", "answer_en": "Answer7"}},
        {{"question_en": "Question8", "answer_en": "Answer8"}},
        {{"question_en": "Question9", "answer_en": "Answer9"}},
        {{"question_en": "Question10", "answer_en": "Answer10"}}
    ],
    "interview_questions": [
        {{"question_en": "Interview Question 1", "answer_en": "Detailed answer for interview question 1. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Company Understanding"}},
        {{"question_en": "Interview Question 2", "answer_en": "Detailed answer for interview question 2. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Work Related"}},
        {{"question_en": "Interview Question 3", "answer_en": "Detailed answer for interview question 3. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Job Understanding"}},
        {{"question_en": "Interview Question 4", "answer_en": "Detailed answer for interview question 4. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Company Understanding"}},
        {{"question_en": "Interview Question 5", "answer_en": "Detailed answer for interview question 5. Provide specific and practical answers considering the company's vision, culture, work environment, etc. (200+ characters)", "category_en": "Work Related"}}
    ],
    "ceo_info": {{
        "name_en": "CEO/President Name",
        "position_en": "Position (e.g., CEO, President, Co-CEO, etc.)",
        "bio_en": "Detailed introduction of the CEO/President. Include education, career, major achievements, leadership style, etc. (300+ characters)",
        "tenure_en": "Tenure (e.g., 2020 - Present)",
        "education_en": "Education Information",
        "career_en": "Major Career and Achievements"
    }}
}}

Generate 10 FAQs with real frequently asked questions and answers.
Generate at least 5 interview questions that are likely to be asked in actual interviews. Include core questions with detailed answers (200+ characters) and categories (Company Understanding, Work Related, Job Understanding, etc.) for each question.
Include CEO/President information for the current CEO or President. Include name, position, detailed introduction (300+ characters), tenure, education, and major career achievements.""",
        "ja": f"""次の会社に関する詳細情報を提供してください: {company_name}

次の形式でJSONで応答してください:
{{
    "company_info": "会社紹介 (500文字以上)",
    "popular_products": [
        {{"text_ja": "商品名1", "score": 85, "image_url": ""}},
        {{"text_ja": "商品名2", "score": 80, "image_url": ""}},
        {{"text_ja": "商品名3", "score": 75, "image_url": ""}}
    ],
    "trending_topics": [
        {{"text_ja": "話題のニュース1", "score": 90, "detail_ja": "話題のニュース1に関する詳細内容です。具体的な説明と背景情報を含みます。"}},
        {{"text_ja": "話題のニュース2", "score": 85, "detail_ja": "話題のニュース2に関する詳細内容です。具体的な説明と背景情報を含みます。"}},
        {{"text_ja": "話題のニュース3", "score": 80, "detail_ja": "話題のニュース3に関する詳細内容です。具体的な説明と背景情報を含みます。"}}
    ],
    "faqs": [
        {{"question_ja": "質問1", "answer_ja": "回答1"}},
        {{"question_ja": "質問2", "answer_ja": "回答2"}},
        {{"question_ja": "質問3", "answer_ja": "回答3"}},
        {{"question_ja": "質問4", "answer_ja": "回答4"}},
        {{"question_ja": "質問5", "answer_ja": "回答5"}},
        {{"question_ja": "質問6", "answer_ja": "回答6"}},
        {{"question_ja": "質問7", "answer_ja": "回答7"}},
        {{"question_ja": "質問8", "answer_ja": "回答8"}},
        {{"question_ja": "質問9", "answer_ja": "回答9"}},
        {{"question_ja": "質問10", "answer_ja": "回答10"}}
    ],
    "interview_questions": [
        {{"question_ja": "面接質問1", "answer_ja": "面接質問1に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "会社理解"}},
        {{"question_ja": "面接質問2", "answer_ja": "面接質問2に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "業務関連"}},
        {{"question_ja": "面接質問3", "answer_ja": "面接質問3に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "職務理解"}},
        {{"question_ja": "面接質問4", "answer_ja": "面接質問4に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "会社理解"}},
        {{"question_ja": "面接質問5", "answer_ja": "面接質問5に対する詳細な回答です。会社のビジョン、文化、業務環境などを考慮して、具体的で実用的な回答を提供します。（200文字以上）", "category_ja": "業務関連"}}
    ],
    "ceo_info": {{
        "name_ja": "代表取締役/CEO名",
        "position_ja": "役職（例：代表取締役、CEO、共同代表取締役など）",
        "bio_ja": "代表取締役/CEOに関する詳細な紹介です。学歴、経歴、主要な成果、リーダーシップスタイルなどを含めて300文字以上で作成してください。",
        "tenure_ja": "在任期間（例：2020年～現在）",
        "education_ja": "学歴情報",
        "career_ja": "主要な経歴および成果"
    }}
}}

FAQは10個生成してください。実際によくある質問と回答を含めてください。
面接質問は5個以上生成してください。実際の面接で出る可能性のある核心的な質問を含め、各質問に対する詳細な回答（200文字以上）とカテゴリー（会社理解、業務関連、職務理解など）を提供してください。
CEO/代表取締役情報は現在在任中の代表取締役またはCEOの実際の情報を含めてください。名前、役職、詳細な紹介（300文字以上）、在任期間、学歴、主要な経歴を含めてください。"""
    }
    
    prompt = lang_prompts.get(lang, lang_prompts["ko"])
    
    try:
        from llm_client import run_llm
        response = run_llm(prompt)
        
        # JSON 파싱 시도
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group()
            try:
                company_data = json.loads(json_str)
                
                # 공동 대표 제품 FAQ 추가
                common_faqs = get_common_product_faqs(company_name, lang)
                if common_faqs:
                    existing_faqs = company_data.get("faqs", [])
                    # 공동 FAQ를 기존 FAQ 앞에 추가
                    company_data["faqs"] = common_faqs + existing_faqs
                
                # FAQ가 10개 미만이면 기본 FAQ 추가
                if len(company_data.get("faqs", [])) < 10:
                    # 언어별 기본 FAQ
                    default_faqs_by_lang = {
                        "ko": [
                        {"question_ko": "회사 설립일은 언제인가요?", "answer_ko": "회사 설립일에 대한 정보를 확인 중입니다."},
                        {"question_ko": "주요 사업 분야는 무엇인가요?", "answer_ko": "주요 사업 분야에 대한 정보를 확인 중입니다."},
                        {"question_ko": "본사 위치는 어디인가요?", "answer_ko": "본사 위치에 대한 정보를 확인 중입니다."},
                        {"question_ko": "직원 수는 얼마나 되나요?", "answer_ko": "직원 수에 대한 정보를 확인 중입니다."},
                        {"question_ko": "주요 제품/서비스는 무엇인가요?", "answer_ko": "주요 제품/서비스에 대한 정보를 확인 중입니다."},
                        ],
                        "en": [
                            {"question_en": "When was the company founded?", "answer_en": "We are checking information about the company's founding date."},
                            {"question_en": "What are the main business areas?", "answer_en": "We are checking information about the main business areas."},
                            {"question_en": "Where is the headquarters located?", "answer_en": "We are checking information about the headquarters location."},
                            {"question_en": "How many employees does the company have?", "answer_en": "We are checking information about the number of employees."},
                            {"question_en": "What are the main products/services?", "answer_en": "We are checking information about the main products/services."},
                        ],
                        "ja": [
                            {"question_ja": "会社の設立日はいつですか？", "answer_ja": "会社の設立日に関する情報を確認中です。"},
                            {"question_ja": "主要な事業分野は何ですか？", "answer_ja": "主要な事業分野に関する情報を確認中です。"},
                            {"question_ja": "本社の所在地はどこですか？", "answer_ja": "本社の所在地に関する情報を確認中です。"},
                            {"question_ja": "従業員数は何人ですか？", "answer_ja": "従業員数に関する情報を確認中です。"},
                            {"question_ja": "主要な製品・サービスは何ですか？", "answer_ja": "主要な製品・サービスに関する情報を確認中です。"},
                        ]
                    }
                    default_faqs = default_faqs_by_lang.get(lang, default_faqs_by_lang["ko"])
                    existing_faqs = company_data.get("faqs", [])
                    # 부족한 만큼 기본 FAQ 추가
                    while len(existing_faqs) < 10:
                        idx = len(existing_faqs) % len(default_faqs)
                        existing_faqs.append(default_faqs[idx])
                    company_data["faqs"] = existing_faqs[:10]
                return company_data
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본 구조 반환
                return {
                    "company_info": response[:1000] if len(response) > 1000 else response,
                    "popular_products": [],
                    "trending_topics": [],
                    "faqs": [],
                    "interview_questions": [],
                    "ceo_info": {}
                }
        else:
            # JSON이 아닌 경우 기본 구조 반환
            return {
                "company_info": response[:1000] if len(response) > 1000 else response,
                "popular_products": [],
                "trending_topics": [],
                "faqs": [],
                "interview_questions": [],
                "ceo_info": {}
            }
    except Exception as e:
        # 언어별 에러 메시지
        error_messages = {
            "ko": f"회사 정보 생성 중 오류가 발생했습니다: {str(e)}",
            "en": f"An error occurred while generating company information: {str(e)}",
            "ja": f"会社情報の生成中にエラーが発生しました: {str(e)}"
        }
        return {
            "company_info": error_messages.get(lang, error_messages["ko"]),
            "popular_products": [],
            "trending_topics": [],
            "faqs": [],
            "interview_questions": [],
            "ceo_info": {}
        }





# ========================================
# 1. 다국어 설정 (전화 발신 관련 텍스트 추가)
# ========================================
DEFAULT_LANG = "ko"


