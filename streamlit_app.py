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

# ========================================
# streamlit_app.py (전체 수정된 코드)
#
# 주요 개선 사항:
# 1. 채팅/이메일 탭에 '전화 발신 (현지 업체/고객)' 버튼 및 기능 추가 (예외 처리 대응)
# 2. 전화 탭에 '전화 발신' 버튼 추가 및 발신 통화 시뮬레이션 모드 지원
# 3. 관련 언어 팩 추가 및 세션 상태 업데이트
# 4. 퀴즈 기능의 정답 확인, 해설, 점수 표시 로직 완성
# 5. [BUG FIX] 언어 이관 시 '번역 다시 시도' 버튼의 DuplicateWidgetID 오류 해결
# 6. [BUG FIX] 콘텐츠 생성 탭의 LLM 응답 및 라디오 버튼 초기화 오류 해결
# ⭐ [전화 아바타 버그 수정]
# 7. 전화 응답 후 인사말 미출력 오류 수정 (just_entered_call 플래그 위치 수정)
# 8. 아바타 Lottie 파일 로딩 경로 수정 (업로드된 파일명 참조)
# ========================================

# ⭐ OpenMP 라이브러리 충돌 해결
# 여러 OpenMP 런타임이 동시에 로드되는 것을 방지하기 위한 환경 변수 설정
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import io
import json
import time
import uuid
import base64
import tempfile
import hashlib
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union, Tuple
import google.generativeai as genai
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import requests  # ⭐ 추가: requests 라이브러리 필요

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False

from openai import OpenAI
from anthropic import Anthropic

# mic_recorder (0.0.8) - returns dict with key "bytes"
from streamlit_mic_recorder import mic_recorder

# LangChain / RAG 관련
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError(
        "❌ 'langchain-text-splitters' 패키지가 설치되지 않았습니다.\n"
        "다음 명령어로 설치해주세요: pip install langchain-text-splitters\n"
        "또는 requirements.txt의 모든 패키지를 설치: pip install -r requirements.txt"
    )
from langchain_core.prompts import PromptTemplate
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    raise ImportError(
        "❌ 'langchain' 패키지가 설치되지 않았거나 'langchain.memory' 모듈을 찾을 수 없습니다.\n"
        "다음 명령어로 설치해주세요: pip install langchain\n"
        "또는 requirements.txt의 모든 패키지를 설치: pip install -r requirements.txt"
    )
try:
    from langchain.chains import ConversationChain
except ImportError:
    raise ImportError(
        "❌ 'langchain' 패키지가 설치되지 않았거나 'langchain.chains' 모듈을 찾을 수 없습니다.\n"
        "다음 명령어로 설치해주세요: pip install langchain\n"
        "또는 requirements.txt의 모든 패키지를 설치: pip install -r requirements.txt"
    )

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Word, PPTX, PDF 생성을 위한 라이브러리
try:
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    IS_DOCX_AVAILABLE = True
except ImportError:
    IS_DOCX_AVAILABLE = False

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    IS_PPTX_AVAILABLE = True
except ImportError:
    IS_PPTX_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.colors import black
    IS_REPORTLAB_AVAILABLE = True
except ImportError:
    IS_REPORTLAB_AVAILABLE = False



try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    IS_GEMINI_EMBEDDING_AVAILABLE = True
except ImportError:
    IS_GEMINI_EMBEDDING_AVAILABLE = False

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    IS_NVIDIA_EMBEDDING_AVAILABLE = True
except ImportError:
    IS_NVIDIA_EMBEDDING_AVAILABLE = False

# ========================================
# Streamlit 페이지 설정 (반드시 최상단에 위치)
# ========================================
st.set_page_config(
    page_title="AI Study Coach & Customer Service Simulator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 0. 기본 경로/로컬 DB 설정
# ========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "local_db")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RAG_INDEX_DIR = os.path.join(DATA_DIR, "rag_index")

VOICE_META_FILE = os.path.join(DATA_DIR, "voice_records.json")
SIM_META_FILE = os.path.join(DATA_DIR, "simulation_histories.json")
VIDEO_MAPPING_DB_FILE = os.path.join(DATA_DIR, "video_mapping_database.json")  # ⭐ Gemini 제안: 비디오 매핑 데이터베이스
FAQ_DB_FILE = os.path.join(DATA_DIR, "faq_database.json")  # FAQ 데이터베이스 파일
PRODUCT_IMAGE_CACHE_FILE = os.path.join(DATA_DIR, "product_image_cache.json")  # 제품 이미지 캐시 파일
PRODUCT_IMAGE_DIR = os.path.join(DATA_DIR, "product_images")  # 생성된 제품 이미지 저장 디렉토리

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRODUCT_IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RAG_INDEX_DIR, exist_ok=True)

# 비디오 디렉토리도 초기화 시 생성
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)




# ----------------------------------------
# JSON Helper
# ----------------------------------------
def _load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# FAQ 데이터베이스 관리 함수
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
            "faqs": company_data.get("faqs", [])
        }
    return {"info": "", "popular_products": [], "trending_topics": [], "faqs": []}


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
    ]
}}

FAQ는 10개를 생성해주세요. 실제로 자주 묻는 질문과 답변을 포함해주세요.""",
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
    ]
}}

Generate 10 FAQs with real frequently asked questions and answers.""",
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
    ]
}}

FAQは10個生成してください。実際によくある質問と回答を含めてください。"""
    }
    
    prompt = lang_prompts.get(lang, lang_prompts["ko"])
    
    try:
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
                    "faqs": []
                }
        else:
            # JSON이 아닌 경우 기본 구조 반환
            return {
                "company_info": response[:1000] if len(response) > 1000 else response,
                "popular_products": [],
                "trending_topics": [],
                "faqs": []
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
            "faqs": []
        }





# ========================================
# 1. 다국어 설정 (전화 발신 관련 텍스트 추가)
# ========================================
DEFAULT_LANG = "ko"

LANG: Dict[str, Dict[str, str]] = {
    "ko": {
        "title": "개인 맞춤형 AI 학습 코치 (음성 및 DB 통합)",
        "sidebar_title": "📚 AI Study Coach 설정",
        "file_uploader": "학습 자료 업로드 (PDF, TXT, HTML)",
        "button_start_analysis": "자료 분석 시작 (RAG Indexing)",
        "rag_tab": "RAG 지식 챗봇",
        "content_tab": "맞춤형 학습 콘텐츠 생성",
        "lstm_tab": "LSTM 성취도 예측 대시보드",
        "sim_tab_chat_email": "AI 고객 응대 시뮬레이터 (채팅/이메일)",
        "sim_tab_phone": "AI 고객 응대 시뮬레이터 (전화)",
        "simulator_tab": "AI 고객 응대 시뮬레이터",
        "company_info_tab": "회사 정보 및 FAQ",
        "company_info_tab_desc": "회사별 상세 정보, 인기 상품, 화제의 소식, FAQ를 검색하고 관리할 수 있는 기능입니다.",
        "sim_tab_chat_email_desc": "고객 응대 업무에서 채팅 및 이메일로 실제로 문의 응대가 될 수 있는 실전 대비 가상 시나리오입니다. AI가 응대 가이드라인과 초안을 생성하며, 고객 반응을 시뮬레이션하여 실전 대비 훈련이 가능합니다.",
        "sim_tab_phone_desc": "고객 응대 업무에서 전화로 실제로 문의 응대가 될 수 있는 실전 대비 가상 시나리오입니다. 음성 녹음 및 실시간 CC 자막 기능을 제공하며, 전화 통화 시뮬레이션을 통해 실전 응대 능력을 향상시킬 수 있습니다.",
        "rag_tab_desc": "업로드된 문서를 기반으로 질문에 답변하는 지식 챗봇입니다. PDF, TXT, HTML 파일을 업로드하여 RAG(Retrieval-Augmented Generation) 인덱스를 구축하고, 문서 내용을 기반으로 정확한 답변을 제공합니다.",
        "content_tab_desc": "AI를 활용하여 개인 맞춤형 학습 콘텐츠를 생성하는 기능입니다. 학습 주제와 난이도에 맞춰 핵심 요약 노트, 객관식 퀴즈, 실습 예제 등을 생성할 수 있습니다.",
        "lstm_tab_desc": "LSTM 모델을 활용하여 학습자의 성취도를 예측하고 대시보드로 시각화하는 기능입니다. 과거 퀴즈 점수 데이터를 분석하여 미래 성취도를 예측하고, 학습 성과를 시각적으로 확인할 수 있습니다.",
        "company_info_tab_desc": "회사별 상세 정보, 인기 상품, 화제의 소식, FAQ를 검색하고 관리할 수 있는 기능입니다. 회사 소개, 인기 상품, 화제의 소식을 시각화하여 한눈에 확인할 수 있습니다.",
        "voice_rec_header_desc": "음성 녹음 및 전사 결과를 관리하고 저장하는 기능입니다. 마이크로 녹음하거나 파일을 업로드하여 Whisper API를 통해 음성을 텍스트로 변환하고, 전사 결과를 저장 및 관리할 수 있습니다.",
        "more_features_label": "더보기 기능",
        "rag_header": "RAG 지식 챗봇 (문서 기반 Q&A)",
        "rag_desc": "업로드된 문서 기반으로 질문에 답변합니다。",
        "rag_input_placeholder": "학습 자료에 대해 질문해 보세요",
        "llm_error_key": "⚠️ 경고: GEMINI API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요。",
        "llm_error_init": "LLM 초기화 오류: API 키를 확인해 주세요。",
        "content_header": "맞춤형 학습 콘텐츠 생성",
        "content_desc": "학습 주제와 난이도에 맞춰 콘텐츠 생성",
        "topic_label": "학습 주제",
        "level_label": "난이도",
        "content_type_label": "콘텐츠 형식",
        "level_options": ["초급", "중급", "고급"],
        "content_options": ["핵심 요약 노트", "객관식 퀴즈 10문항", "실습 예제 아이디어"],
        "button_generate": "콘텐츠 생성",
        "warning_topic": "학습 주제를 입력해 주세요。",
        "lstm_header": "LSTM 기반 학습 성취도 예측 대시보드",
        "lstm_desc": "가상의 과거 퀴즈 점수 데이터를 바탕으로 LSTM 모델을 훈련하고 미래 성취도를 예측하여 보여줍니다。",
        "lang_select": "언어 선택",
        "company_info_faq_settings": "회사별 상세 정보 및 FAQ",
        "search_company": "회사명 검색",
        "company_info": "회사 소개",
        "company_faq": "자주 나오는 질문",
        "faq_question": "질문",
        "faq_answer": "답변",
        "popular_products": "인기 상품",
        "trending_topics": "화제의 소식",
        "company_details": "회사 상세 정보",
        "no_company_found": "에 해당하는 회사를 찾을 수 없습니다.",
        "no_company_selected": "회사명을 검색하거나 선택해주세요.",
        "product_popularity": "상품 인기도",
        "topic_trends": "화제 트렌드",
        "select_company": "회사 선택",
        "faq_search": "FAQ 검색",
        "faq_search_placeholder": "FAQ 검색어를 입력하세요",
        "faq_search_placeholder_extended": "FAQ 검색어를 입력하세요 (상품명, 서비스명 등도 검색 가능)",
        "button_search_faq": "검색",
        "company_search_placeholder": "예: 삼성, 네이버, 구글, 애플 등",
        "company_search_button": "검색",
        "generating_company_info": "회사 정보를 생성하는 중...",
        "button_copy_answer": "답안 복사",
        "button_copy_hint": "힌트 복사",
        "button_download_answer": "답안 다운로드",
        "button_download_hint": "힌트 다운로드",
        "copy_instruction": "💡 위 텍스트를 선택하고 Ctrl+C (Mac: Cmd+C)로 복사하세요.",
        "copy_help_text": "텍스트를 선택하고 Ctrl+C (또는 Cmd+C)로 복사하세요.",
        "button_reset": "새로 시작",
        "answer_displayed": "답안이 표시되었습니다. 위의 텍스트를 복사하세요.",
        "hint_displayed": "힌트가 표시되었습니다. 위의 텍스트를 복사하세요.",
        "ai_answer_generated": "AI 답안이 생성되었습니다.",
        "hint_generated": "응대 힌트가 생성되었습니다.",
        "warning_enter_inquiry": "고객 문의 내용을 입력해주세요.",
        "customer_inquiry_review_desc": "에이전트가 상사들에게 고객 문의 내용을 재확인하고, AI 답안 및 힌트를 생성할 수 있는 기능입니다.",
        "all_companies": "전체",
        "optional": "선택사항",
        "no_faq_for_company": "{company}의 FAQ가 없습니다.",
        "related_products": "관련 상품",
        "related_trending_news": "관련 화제 소식",
        "related_company_info": "관련 회사 소개 내용",
        "related_faq": "관련 FAQ",
        "items": "개",
        "popularity": "인기도",
        "no_faq_for_product": "해당 상품과 관련된 FAQ를 찾을 수 없습니다. 상품 정보만 표시됩니다.",
        "generating_detail": "상세 내용을 생성하는 중입니다...",
        "checking_additional_info": "상세 내용: {topic}에 대한 추가 정보를 확인 중입니다.",
        "button_generate_faq": "FAQ 생성",
        "button_add_company": "고객 문의 재확인",
        "customer_inquiry_review": "고객 문의 재확인",
        "inquiry_question_label": "고객 문의 내용",
        "inquiry_question_placeholder": "고객이 문의한 내용을 입력하세요",
        "inquiry_attachment_label": "📎 고객 첨부 파일 업로드 (사진/스크린샷)",
        "inquiry_attachment_help": "특히 취소 불가 여행상품의 비행기 지연, 여권 이슈 등 불가피한 사유의 경우, 반드시 사진이나 스크린샷을 첨부해주세요.",
        "inquiry_attachment_uploaded": "✅ 첨부 파일이 업로드되었습니다: {filename}",
        "extracting_file_content": "파일 내용 추출 중...",
        "detecting_language": "언어 감지 중...",
        "translating_content": "파일 내용 번역 중...",
        "file_translated": "✅ 파일 내용이 번역되었습니다.",
        "file_extraction_error": "파일 내용 추출 중 오류가 발생했습니다: {error}",
        "ocr_requires_manual": "이미지 OCR을 위해서는 Gemini API 키가 필요합니다. 이미지의 텍스트를 수동으로 입력해주세요.",
        "ocr_error": "이미지 텍스트 추출 중 오류: {error}",
        "button_generate_ai_answer": "AI 답안 생성",
        "button_generate_hint": "응대 힌트 생성",
        "ai_answer_header": "AI 추천 답안",
        "hint_header": "응대 힌트",
        "generating_ai_answer": "AI 답안을 생성하는 중...",
        "generating_hint": "응대 힌트를 생성하는 중...",
        "button_edit_company": "회사 정보 수정",
        "button_show_company_info": "회사 소개 보기",
        "no_faq_results": "검색 결과가 없습니다.",
        "faq_search_results": "FAQ 검색 결과",
        "add_company_name": "회사명",
        "add_company_info": "회사 소개",
        "generate_faq_question": "질문",
        "generate_faq_answer": "답변",
        "button_save_faq": "FAQ 저장",
        "button_cancel": "취소",
        "faq_saved_success": "FAQ가 저장되었습니다.",
        "company_added_success": "회사가 추가되었습니다.",
        "company_updated_success": "회사 정보가 업데이트되었습니다.",
        "embed_success": "총 {count}개 청크로 학습 DB 구축 완료!",
        "embed_fail": "임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제。",
        "warning_no_files": "먼저 학습 자료를 업로드하세요。",
        "warning_rag_not_ready": "RAG가 준비되지 않았습니다. 학습 자료를 업로드하고 분석하세요。",
        "quiz_fail_structure": "퀴즈 데이터 구조가 올바르지 않습니다。",
        "select_answer": "정답을 선택하세요",
        "check_answer": "정답 확인",
        "next_question": "다음 문항",
        "correct_answer": "정답입니다! 🎉",
        "incorrect_answer": "오답입니다。😞",
        "correct_is": "정답",
        "explanation": "해설",
        "quiz_complete": "퀴즈 완료!",
        "score": "점수",
        "retake_quiz": "퀴즈 다시 풀기",
        "question_label": "문항",
        "correct_questions": "맞은 문제",
        "incorrect_questions": "틀린 문제",
        "question_result": "문제 결과",
        "your_answer": "내 답안",
        "correct_answer_label": "정답",
        "quiz_error_llm": "퀴즈 생성 실패: LLM이 올바른 JSON 형식을 반환하지 않았습니다。",
        "quiz_original_response": "LLM 원본 응답",
        "firestore_loading": "데이터베이스에서 RAG 인덱스 로드 중...",
        "firestore_no_index": "데이터베이스에서 기존 RAG 인덱스를 찾을 수 없습니다. 파일을 업로드하여 새로 만드세요。",
        "db_save_complete": "(DB 저장 완료)",
        "data_analysis_progress": "자료 분석 및 학습 DB 구축 중...",
        "response_generating": "답변 생성 중...",
        "lstm_result_header": "학습 성취도 예측 결과",
        "lstm_score_metric": "현재 예측 성취도",
        "lstm_score_info": "다음 퀴즈 예상 점수는 약 **{predicted_score:.1f}점**입니다. 학습 성과를 유지하거나 개선하세요!",
        "lstm_rerun_button": "새로운 가상 데이터로 예측",

        # --- 토스트 메시지 추가 ---
        "toast_like": "🔥 컨텐츠가 맘에 드셨군요! (좋아요 카운트 +1)",
        "toast_dislike": "😔 더 나은 콘텐츠를 위해 피드백을 반영하겠습니다。",
        "toast_share": "🌐 콘텐츠 링크가 생성되었습니다。",
        "toast_copy": "✅ 콘텐츠가 클립보드에 복사되었습니다!",
        "toast_more": "ℹ️ 추가 옵션 (PDF, 인쇄본 저장 등)",
        "mock_pdf_save": "📥 PDF 저장",
        "mock_word_open": "📑 Word로 열기",
        "mock_print": "🖨 인쇄",

        # --- 시뮬레이터 ---
        "simulator_header": "AI 고객 응대 시뮬레이터",
        "simulator_desc": "까다로운 고객 문의에 AI의 응대 초안 및 가이드라인을 제공합니다。",
        "customer_query_label": "고객 문의 내용 (링크 포함 가능)",
        "customer_type_label": "고객 성향",
        "customer_type_options": ["일반적인 문의", "까다로운 고객", "매우 불만족스러운 고객"],
        "button_simulate": "응대 조언 요청",
        "customer_generate_response_button": "고객 반응 생성",
        "send_closing_confirm_button": "추가 문의 여부 확인 메시지 보내기",
        "simulation_warning_query": "고객 문의 내용을 입력해 주세요。",
        "simulation_no_key_warning": "⚠️ API Key가 없기 때문에 응답 생성은 실행되지 않습니다。",
        "simulation_advice_header": "AI의 응대 가이드라인",
        "simulation_draft_header": "추천 응대 초안",
        "button_listen_audio": "음성으로 듣기",
        "tts_status_ready": "음성으로 듣기 준비됨",
        "tts_status_generating": "오디오 생성 중...",
        "tts_status_success": "✅ 오디오 재생 완료!",
        "tts_status_error": "❌ TTS 오류 발생",
        "history_expander_title": "📝 이전 상담 이력 로드 (최근 10건)",
        "initial_query_sample": "프랑스 파리에 도착했는데, 클룩에서 구매한 eSIM이 활성화가 안 됩니다...",
        "button_mic_input": "🎙 음성 입력",
        "button_mic_stop": "⏹️ 녹음 종료",
        "prompt_customer_end": "고객님의 추가 문의 사항이 없어, 이 상담을 종료합니다。",
        "prompt_survey": "지금까지 상담원 000였습니다. 즐거운 하루 되시기 바랍니다. [설문 조사 링크]",
        "customer_closing_confirm": "다른 문의 사항은 없으십니까?",
        "customer_positive_response": "알겠습니다. 감사합니다。",
        "button_email_end_chat": "응대 종료 (설문 요청)",
        "error_mandatory_contact": "이메일과 전화번호 입력은 필수입니다。",
        "customer_attachment_label": "📎 고객 첨부 파일 업로드",
        "attachment_info_llm": "[고객 첨부 파일: {filename}이(가) 확인되었습니다. 이 파일을 참고하여 응대하세요.]",
        "button_retry_translation": "번역 다시 시도",
        "button_request_hint": "💡 응대 힌트 요청 (AHT 모니터링 중)",
        "button_generate_draft": "🤖 AI 응답 초안 생성",
        "draft_generating": "AI가 응답 초안을 생성 중입니다...",
        "draft_success": "✅ AI 응답 초안이 생성되었습니다. 아래에서 확인하고 수정하세요.",
        "hint_placeholder": "문의 응대에 대한 힌트:",
        "survey_sent_confirm": "📨 설문조사 링크가 전송되었으며, 이 상담은 종료되었습니다。",
        "new_simulation_ready": "새 시뮬레이션을 시작할 수 있습니다。",
        "agent_response_header": "✍️ 에이전트 응답",
        "agent_response_placeholder": "고객에게 응답하세요...",
        "send_response_button": "응답 전송",
        "customer_turn_info": "에이전트 응답 전송 완료. 고객 반응을 자동으로 생성 중입니다。",
        "generating_customer_response": "고객 반응 생성 중...",
        "call_started_message": "통화가 시작되었습니다. 아래 마이크 버튼을 눌러 인사말을 녹음하세요.",
        "call_on_hold_message": "통화가 Hold 중입니다. 통화 재개 후 녹음이 가능합니다.",
        "recording_complete_transcribing": "🎙️ 녹음 완료. 전사 처리 중...",
        "transcription_empty_warning": "⚠️ 전사 결과가 비어있습니다. 다시 녹음해주세요. (마이크 입력이 없거나 음소거된 경우)",
        "transcription_error": "[ERROR: 전사 실패]",
        "transcription_no_result": "❌ 전사 결과가 없습니다.",
        "customer_escalation_start": "상급자와 이야기하고 싶습니다",
        "request_rebuttal_button": "고객의 다음 반응 요청",
        "new_simulation_button": "새 시뮬레이션 시작",
        "history_selectbox_label": "로드할 이력을 선택하세요:",
        "history_load_button": "선택된 이력 로드",
        "delete_history_button": "❌ 모든 이력 삭제",
        "delete_confirm_message": "정말로 모든 상담 이력을 삭제하시겠습니까?",
        "delete_confirm_yes": "예, 삭제합니다",
        "download_history_word": "📥 이력 다운로드 (Word)",
        "download_history_pptx": "📥 이력 다운로드 (PPTX)",
        "download_history_pdf": "📥 이력 다운로드 (PDF)",
        "download_current_session": "📥 현재 세션 다운로드",
        "delete_confirm_no": "아니오, 유지합니다",
        "delete_success": "✅ 삭제 완료!",
        "deleting_history_progress": "이력 삭제 중...",
        "search_history_label": "이력 검색",
        "date_range_label": "날짜 범위 필터",
        "history_search_button": "🔍 검색",
        "no_history_found": "검색 조건에 맞는 이력이 없습니다。",
        "customer_email_label": "고객 이메일 (필수)",
        "customer_phone_label": "고객 연락처 / 전화번호 (필수)",
        "transfer_header": "언어 이관 요청 (다른 팀)",
        "transfer_to_en": "🇺🇸 영어 팀으로 이관",
        "transfer_to_ja": "🇯🇵 일본어 팀으로 이관",
        "transfer_to_ko": "🇰🇷 한국어 팀으로 이관",
        "transfer_system_msg": "📌 시스템 메시지: 고객 요청에 따라 상담 언어가 {target_lang} 팀으로 이관되었습니다. 새로운 상담원(AI)이 응대합니다。",
        "transfer_loading": "이관 처리 중: 이전 대화 이력 번역 및 검토 (고객님께 3~10분 양해 요청)",
        "transfer_summary_header": "🔍 이관된 상담원을 위한 요약 (번역됨)",
        "transfer_summary_intro": "고객님과의 이전 대화 이력입니다. 이 내용을 바탕으로 응대를 이어나가세요。",
        "llm_translation_error": "❌ 번역 실패: LLM 응답 오류",
        "timer_metric": "상담 경과 시간 (AHT)",
        "timer_info_ok": "AHT (15분 기준)",
        "timer_info_warn": "AHT (10분 초과)",
        "timer_info_risk": "🚨 15분 초과: 높은 리스크",
        "solution_check_label": "✅ 이 응답에 솔루션/해결책이 포함되어 있습니다。",
        "sentiment_score_label": "고객 감정 점수",
        "urgency_score_label": "긴급도 점수",
        "customer_gender_label": "고객 성별",
        "customer_emotion_label": "고객 감정 상태",
        "gender_male": "남성",
        "gender_female": "여성",
        "emotion_happy": "기분 좋은 고객",
        "emotion_dissatisfied": "불만인 고객",
        "emotion_angry": "화난 고객",
        "emotion_sad": "슬픈/우울한 고객",
        "emotion_neutral": "중립",
        "similarity_chart_title": "유사 케이스 유사도",
        "scores_comparison_title": "감정 및 만족도 점수 비교",
        "similarity_score_label": "유사도",
        "satisfaction_score_label": "만족도",
        "sentiment_trend_label": "감정 점수 추이",
        "satisfaction_trend_label": "만족도 점수 추이",
        "case_trends_title": "과거 케이스 점수 추이",
        "date_label": "날짜",
        "score_label": "점수 (0-100)",
        "customer_characteristics_title": "고객 특성 분포",
        "language_label": "언어",
        "email_provided_label": "이메일 제공",
        "phone_provided_label": "전화번호 제공",
        "region_label": "지역",
        "btn_request_phone_summary": "이력 요약 요청",

        # --- 추가된 전화 발신 기능 관련 ---
        "button_call_outbound": "전화 발신",
        "button_call_outbound_to_customer": "고객에게 전화 발신",
        "button_call_outbound_to_provider": "현지 업체에게 전화 발신",
        "call_outbound_system_msg": "📌 시스템 메시지: 에이전트가 {target}에게 전화 발신을 시도했습니다。",
        "call_outbound_simulation_header": "📞 전화 발신 시뮬레이션 결과",
        "call_outbound_summary_header": "📞 현지 업체/고객과의 통화 요약",
        "call_outbound_loading": "전화 연결 및 통화 결과 정리 중... (LLM 호출)",
        "call_target_select_label": "발신 대상 선택",
        "call_target_customer": "고객에게 발신",
        "call_target_partner": "현지 업체 발신",

        # --- 음성 기록 ---
        "voice_rec_header": "음성 기록 & 관리",
        "record_help": "마이크 버튼을 눌러 녹음하거나 파일을 업로드하세요。",
        "uploaded_file": "오디오 파일 업로드",
        "rec_list_title": "저장된 음성 기록",
        "transcribe_btn": "전사(Whisper)",
        "save_btn": "음성 기록 저장",
        "transcribing": "음성 전사 중...",
        "transcript_result": "전사 결과:",
        "transcript_text": "전사 텍스트",
        "openai_missing": "OpenAI API Key가 없습니다。",
        "whisper_client_error": "❌ Whisper API Client 초기화 실패",
        "whisper_auth_error": "❌ Whisper API 인증 실패",
        "whisper_format_error": "❌ 지원하지 않는 오디오 형식입니다。",
        "whisper_success": "✅ 음성 전사 완료!",
        "playback": "녹음 재생",
        "retranscribe": "재전사",
        "delete": "삭제",
        "no_records": "저장된 음성 기록이 없습니다。",
        "saved_success": "저장 완료!",
        "delete_confirm_rec": "정말 삭제하시겠습니까?",
        "gcs_not_conf": "GCS 미설정",
        "gcs_playback_fail": "오디오 재생 실패",
        "gcs_no_audio": "오디오 없음",
        "error": "오류:",
        "firestore_no_db_connect": "DB 연결 실패",
        "save_history_success": "상담 이력이 저장되었습니다。",
        "save_history_fail": "상담 이력 저장 실패",
        "delete_fail": "삭제 실패",
        "rec_header": "음성 입력 및 전사",
        "whisper_processing": "음성 전사 처리 중..",
        "empty_response_warning": "응답을 입력하세요.",
        "customer_no_more_inquiries": "없습니다. 감사합니다.",
        "customer_has_additional_inquiries": "추가 문의 사항도 있습니다.",
        "agent_confirmed_inquiry": "에이전트가 추가 문의 여부를 확인했습니다. 고객의 최종 답변을 자동으로 생성합니다.",
        "customer_no_more_inquiries_confirmed": "고객이 더 이상 문의할 사항이 없다고 확인했습니다。",
        "consultation_end": "상담 종료",
        "consultation_end_instruction": "아래 **설문 조사 링크 전송 및 응대 종료** 버튼을 클릭하여 상담을 종료하세요.",
        "sim_end_chat_button": "설문 조사 링크 전송 및 응대 종료",
        "delete_mic_record": "❌ 녹음 삭제",

        # --- 첨부 파일 기능 추가 ---
        "attachment_label": "고객 첨부 파일 업로드 (스크린샷 등)",
        "attachment_placeholder": "파일을 첨부하여 상황을 설명하세요 (선택 사항)",
        "attachment_info_llm": "[고객 첨부 파일: {filename}이(가) 확인되었습니다. 이 파일을 참고하여 응대하세요.]",
        "agent_attachment_label": "에이전트 첨부 파일 (스크린샷 등)",
        "agent_attachment_placeholder": "응답에 첨부할 파일을 선택하세요 (선택 사항)",
        "agent_attachment_status": "📎 에이전트가 **{filename}** 파일을 응답에 첨부했습니다. (파일 타입: {filetype})",

        # --- RAG 오류 메시지 추가 ---
        "rag_embed_error_openai": "RAG 임베딩 실패: OpenAI API Key가 유효하지 않거나 설정되지 않았습니다。",
        "rag_embed_error_gemini": "RAG 임베딩 실패: Gemini API Key가 유효하지 않거나 설정되지 않았습니다。",
        "rag_embed_error_nvidia": "RAG 임베딩 실패: NVIDIA API Key가 유효하지 않거나 설정되지 않았습니다。",
        "rag_embed_error_none": "RAG 임베딩에 필요한 모든 키(OpenAI, Gemini, NVIDIA)가 유효하지 않습니다. 키를 설정해 주세요。",

        # --- 전화 기능 관련 추가 ---
        "phone_header": "AI 고객 응대 시뮬레이터 (전화)",
        "call_status_waiting": "수신 대기 중...",
        "call_status_ringing": "전화 수신 중: {number}",
        "button_answer": "📞 전화 응답",
        "button_hangup": "🔴 전화 끊기",
        "button_hold": "⏸️ Hold (소음 차단)",
        "button_resume": "▶️ 통화 재개",
        "hold_status": "통화 Hold 중 (누적 Hold 시간: {duration})",
        "cc_live_transcript": "🎤 실시간 CC 자막 / 전사",
        "mic_input_status": "🎙️ 에이전트 음성 입력",
        "customer_audio_playback": "🗣️ 고객 음성 재생",
        "agent_response_prompt": "고객에게 말할 응답을 녹음하세요。",
        "agent_response_stop_and_send": "⏹️ 녹음 종료 및 응답 전송",
        "call_end_message": "통화가 종료되었습니다. AHT 및 이력을 확인하세요。",
        "call_query_placeholder": "고객 문의 내용을 입력하세요。",
        "call_number_placeholder": "+82 10-xxxx-xxxx (가상 번호)",
        "website_url_label": "홈페이지 웹 주소 (선택사항)",
        "website_url_placeholder": "https://example.com (홈페이지 주소가 있으면 입력하세요)",
        "call_summary_header": "AI 통화 요약",
        "customer_audio_header": "고객 최초 문의 (음성)",
        "aht_not_recorded": "⚠️ 통화 시작 시간이 기록되지 않아 AHT를 계산할 수 없습니다。",
        "no_audio_record": "고객의 최초 음성 기록이 없습니다。",
        "customer_query_playing": "🔊 고객 문의 재생 중입니다.",
        "query_content_label": "📝 문의 내용:",
        "auto_play_failed": "자동 재생 실패: {error}. 수동으로 재생해주세요.",
        "generating_customized_response": "고객 맞춤형 반응 생성 중...",
        "customer_responded": "🗣️ 고객이 응답했습니다: {reaction}",
        "customer_voice_generation_error": "❌ 고객 음성 생성 오류: {error}",
        "button_retry_translation": "번역 다시 시도",
        "customer_waiting_hold": "[고객: 잠시 대기 중입니다...]",
        "agent_hold_message": "[에이전트: Hold 중입니다. 통화 재개 버튼을 눌러주세요.]",
        
        # --- 비디오 파일 업로드 관련 ---
        "video_upload_expander": "비디오 파일 업로드/로드",
        "video_sync_enable": "비디오 동기화 활성화 (TTS와 함께 재생)",
        "video_rag_title": "🎥 OpenAI/Gemini 기반 영상 RAG 기능",
        "video_rag_desc": "✅ **현재 구현 방식 (영상 RAG):**\n\n1. **LLM 텍스트 분석**: OpenAI/Gemini API가 고객의 텍스트를 분석하여 감정 상태와 제스처를 자동 판단합니다.\n\n2. **지능형 비디오 선택**: 분석 결과에 따라 적절한 비디오 클립을 자동으로 선택합니다.\n   - 감정 상태: HAPPY, ANGRY, ASKING, SAD, NEUTRAL\n   - 제스처: HAND_WAVE, NOD, SHAKE_HEAD, POINT, NONE\n\n3. **TTS 동기화 재생**: 선택된 비디오와 TTS로 생성된 음성을 동시에 재생합니다.\n\n**사용 방법:**\n- 성별(남자/여자)과 감정 상태별로 비디오 파일을 업로드하세요.\n- 제스처별 비디오도 업로드 가능합니다 (예: `male_happy_hand_wave.mp4`).\n- 고객이 말하는 내용에 따라 LLM이 자동으로 적절한 비디오를 선택합니다.",
        "video_gender_emotion_setting": "성별 및 감정 상태별 비디오 설정",
        "video_gender_label": "성별",
        "video_gender_male": "남자",
        "video_gender_female": "여자",
        "video_emotion_label": "감정 상태",
        "video_upload_label": "비디오 파일 업로드 ({gender} - {emotion})",
        "video_current_selection": "📹 현재 선택: {gender} - {emotion}",
        "video_upload_prompt": "💡 '{filename}' 비디오 파일을 업로드하세요.",
        "video_save_path": "📂 비디오 저장 경로:",
        "video_directory_empty": "⚠️ 비디오 디렉토리에 파일이 없습니다. 파일을 업로드하세요.",
        "video_directory_not_exist": "⚠️ 비디오 디렉토리가 존재하지 않습니다: {path}",
        "video_local_path_input": "또는 로컬 파일 경로 입력",
        "video_local_path_placeholder": "예: C:\\Users\\Admin\\Downloads\\video.mp4 또는 video.mp4",
        "video_current_avatar": "📺 현재 고객 아바타 영상",
        "video_avatar_upload_prompt": "💡 '{filename}' 비디오 파일을 업로드하면 영상이 표시됩니다.",
        "video_uploaded_files": "📁 업로드된 비디오 파일:",
        "video_bytes_saved": "✅ 비디오 바이트 저장 완료: {name} ({size} MB)",
        "video_empty_error": "❌ 비디오 파일이 비어있습니다. 다시 업로드해주세요.",
        "video_upload_error": "❌ 비디오 업로드 중 오류 발생: {error}",
        "video_playback_error": "❌ 비디오 재생에 실패했습니다.",
        "video_auto_play_info": "💡 이 비디오는 '{gender} - {emotion}' 상태에서 자동으로 재생됩니다.",
        "video_preview_error": "비디오 미리보기 오류",
        "video_similar_gender": "같은 성별의 다른 비디오",
        "video_rename_hint": "💡 위 비디오 중 하나를 사용하려면 파일명을 변경하거나 새로 업로드하세요.",
        "video_more_files": "... 외 {count}개",
        "avatar_status_info": "상태: {state} | 성별: {gender}",
        "customer_video_simulation": "고객 영상 시뮬레이션",
        "customer_avatar": "고객 아바타",
        "faq_question_prefix": "Q{num}.",
        "visualization_chart": "시각화 차트",
        "company_search_or_select": "회사명을 검색하거나 선택해주세요.",
    },

    # --- ⭐ 영어 버전 (한국어 100% 매칭) ---
    "en": {
        "title": "Personalized AI Study Coach (Voice & Local DB)",
        "sidebar_title": "📚 AI Study Coach Settings",
        "file_uploader": "Upload Study Materials (PDF, TXT, HTML)",
        "button_start_analysis": "Start Analysis (RAG Indexing)",
        "rag_tab": "RAG Knowledge Chatbot",
        "content_tab": "Custom Content Generation",
        "lstm_tab": "LSTM Achievement Prediction Dashboard",
        "sim_tab_chat_email": "AI Customer Support Simulator (Chat / Email)",
        "sim_tab_phone": "AI Customer Support Simulator (Phone)",
        "simulator_tab": "AI Customer Support Simulator",
        "company_info_tab": "Company Information & FAQ",
        "sim_tab_chat_email_desc": "A virtual scenario for practical training in handling customer inquiries via chat and email in customer service work. AI generates response guidelines and drafts, and simulates customer reactions for real-world training.",
        "sim_tab_phone_desc": "A virtual scenario for practical training in handling customer inquiries via phone in customer service work. Provides voice recording and real-time CC subtitle features, allowing you to improve your practical response skills through phone call simulations.",
        "rag_tab_desc": "A knowledge chatbot that answers questions based on uploaded documents. Upload PDF, TXT, or HTML files to build a RAG (Retrieval-Augmented Generation) index and provide accurate answers based on document content.",
        "content_tab_desc": "A feature that generates personalized learning content using AI. You can generate key summary notes, multiple-choice quizzes, and practical examples tailored to learning topics and difficulty levels.",
        "lstm_tab_desc": "A feature that predicts learner achievement using LSTM models and visualizes it in a dashboard. Analyzes past quiz score data to predict future achievement and visually check learning performance.",
        "company_info_tab_desc": "Search and manage company-specific detailed information, popular products, trending news, and FAQs. Visualize company introductions, popular products, and trending news at a glance.",
        "voice_rec_header_desc": "A feature for managing and storing voice recordings and transcription results. Record with a microphone or upload files to convert speech to text via Whisper API, and save and manage transcription results.",
        "more_features_label": "More Features",
        "rag_header": "RAG Knowledge Chatbot (Document Q&A)",
        "rag_desc": "Answer questions based on uploaded documents.",
        "rag_input_placeholder": "Ask a question about your study materials",
        "llm_error_key": "⚠️ Warning: GEMINI_API_KEY is not set.",
        "llm_error_init": "LLM initialization error. Please check your API key.",
        "content_header": "Custom Learning Content Generation",
        "content_desc": "Generate content based on the topic and difficulty.",
        "topic_label": "Learning Topic",
        "level_label": "Difficulty",
        "content_type_label": "Content Type",
        "level_options": ["Beginner", "Intermediate", "Advanced"],
        "content_options": ["Key Summary Note", "10 MCQ Questions", "Practical Example Idea"],
        "button_generate": "Generate Content",
        "warning_topic": "Please enter a learning topic.",
        "lstm_header": "LSTM Achievement Prediction Dashboard",
        "lstm_desc": "Train an LSTM model on hypothetical quiz scores and predict performance.",
        "lang_select": "Select Language",
        "company_info_faq_settings": "Company Details & FAQ",
        "search_company": "Search Company",
        "company_info": "Company Information",
        "company_faq": "Frequently Asked Questions",
        "faq_question": "Question",
        "faq_answer": "Answer",
        "popular_products": "Popular Products",
        "trending_topics": "Trending News",
        "company_details": "Company Details",
        "no_company_found": "No matching company found.",
        "no_company_selected": "Please search or select a company name.",
        "product_popularity": "Product Popularity",
        "topic_trends": "Topic Trends",
        "select_company": "Select Company",
        "faq_search": "FAQ Search",
        "faq_search_placeholder": "Enter FAQ search term",
        "faq_search_placeholder_extended": "Enter FAQ search term (product names, service names, etc. can also be searched)",
        "button_search_faq": "Search",
        "company_search_placeholder": "e.g., Samsung, Naver, Google, Apple",
        "company_search_button": "Search",
        "generating_company_info": "Generating company information...",
        "button_copy_answer": "Copy Answer",
        "button_copy_hint": "Copy Hint",
        "button_download_answer": "Download Answer",
        "button_download_hint": "Download Hint",
        "copy_instruction": "💡 Select the text above and press Ctrl+C (Mac: Cmd+C) to copy.",
        "copy_help_text": "Select the text and press Ctrl+C (or Cmd+C) to copy.",
        "button_reset": "Reset",
        "answer_displayed": "Answer displayed. Please copy the text above.",
        "hint_displayed": "Hint displayed. Please copy the text above.",
        "ai_answer_generated": "AI answer has been generated.",
        "hint_generated": "Response hint has been generated.",
        "warning_enter_inquiry": "Please enter the customer inquiry.",
        "customer_inquiry_review_desc": "A feature that allows agents to reconfirm customer inquiries with supervisors and generate AI answers and hints.",
        "all_companies": "All",
        "optional": "Optional",
        "no_faq_for_company": "No FAQs available for {company}.",
        "related_products": "Related Products",
        "related_trending_news": "Related Trending News",
        "related_company_info": "Related Company Information",
        "related_faq": "Related FAQ",
        "items": "",
        "popularity": "Popularity",
        "no_faq_for_product": "No FAQs related to this product were found. Only product information is displayed.",
        "generating_detail": "Generating detailed content...",
        "checking_additional_info": "Checking additional information for: {topic}",
        "button_generate_faq": "Generate FAQ",
        "button_add_company": "Customer Inquiry Review",
        "customer_inquiry_review": "Customer Inquiry Review",
        "inquiry_question_label": "Customer Inquiry",
        "inquiry_question_placeholder": "Enter the customer's inquiry",
        "inquiry_attachment_label": "📎 Customer Attachment Upload (Photo/Screenshot)",
        "inquiry_attachment_help": "For non-refundable travel products with unavoidable reasons (flight delays, passport issues, etc.), please attach photos or screenshots.",
        "inquiry_attachment_uploaded": "✅ Attachment uploaded: {filename}",
        "extracting_file_content": "Extracting file content...",
        "detecting_language": "Detecting language...",
        "translating_content": "Translating file content...",
        "file_translated": "✅ File content has been translated.",
        "file_extraction_error": "Error occurred while extracting file content: {error}",
        "ocr_requires_manual": "Gemini API key is required for image OCR. Please manually enter the text from the image.",
        "ocr_error": "Error extracting text from image: {error}",
        "button_generate_ai_answer": "Generate AI Answer",
        "button_generate_hint": "Generate Response Hint",
        "ai_answer_header": "AI Recommended Answer",
        "hint_header": "Response Hint",
        "generating_ai_answer": "Generating AI answer...",
        "generating_hint": "Generating response hint...",
        "button_edit_company": "Edit Company Info",
        "button_show_company_info": "Show Company Info",
        "no_faq_results": "No search results found.",
        "faq_search_results": "FAQ Search Results",
        "add_company_name": "Company Name",
        "add_company_info": "Company Information",
        "generate_faq_question": "Question",
        "generate_faq_answer": "Answer",
        "button_save_faq": "Save FAQ",
        "button_cancel": "Cancel",
        "faq_saved_success": "FAQ saved successfully.",
        "company_added_success": "Company added successfully.",
        "company_updated_success": "Company information updated successfully.",
        "embed_success": "Learning DB built with {count} chunks!",
        "embed_fail": "Embedding failed: quota exceeded or network issue.",
        "warning_no_files": "Please upload study materials first.",
        "warning_rag_not_ready": "RAG is not ready. Upload materials and analyze.",
        "quiz_fail_structure": "Quiz data structure is invalid.",
        "select_answer": "Select answer",
        "check_answer": "Check answer",
        "next_question": "Next question",
        "correct_answer": "Correct! 🎉",
        "incorrect_answer": "Incorrect 😞",
        "correct_is": "Correct answer",
        "explanation": "Explanation",
        "quiz_complete": "Quiz Complete!",
        "score": "Score",
        "retake_quiz": "Retake Quiz",
        "question_label": "Question",
        "correct_questions": "Correct",
        "incorrect_questions": "Incorrect",
        "question_result": "Question Results",
        "your_answer": "Your Answer",
        "correct_answer_label": "Correct Answer",
        "quiz_error_llm": "Quiz generation failed: invalid JSON.",
        "quiz_original_response": "Original LLM Response",
        "firestore_loading": "Loading RAG index...",
        "firestore_no_index": "No existing RAG index found.",
        "db_save_complete": "(DB Save Complete)",
        "data_analysis_progress": "Analyzing materials and building DB...",
        "response_generating": "Generating response...",
        "lstm_result_header": "Achievement Prediction",
        "lstm_score_metric": "Predicted Achievement",
        "lstm_score_info": "Estimated next quiz score: **{predicted_score:.1f}**.",
        "lstm_rerun_button": "Predict with New Data",

        # --- 토스트 메시지 추가 ---
        "toast_like": "🔥 Content liked! (+1 Count Reflected)",
        "toast_dislike": "😔 Feedback recorded for better content.",
        "toast_share": "🌐 Content link generated.",
        "toast_copy": "✅ Content copied to clipboard!",
        "toast_more": "ℹ️ Additional options (Print, PDF Save, etc.)",
        "mock_pdf_save": "📥 Save as PDF",
        "mock_word_open": "📑 Open via Word",
        "mock_print": "🖨 Print",

        # --- 토스트 메시지 끝 ---

        # Simulator
        "simulator_header": "AI Customer Response Simulator",
        "simulator_desc": "AI generates draft responses and guidelines for customer inquiries.",
        "customer_query_label": "Customer Message (links allowed)",
        "customer_type_label": "Customer Type",
        "customer_type_options": ["General Inquiry", "Difficult Customer", "Highly Dissatisfied Customer"],
        "button_simulate": "Generate Response",
        "customer_generate_response_button": "Generate Customer Response",
        "send_closing_confirm_button": "Send Closing Confirmation",
        "simulation_warning_query": "Please enter the customer’s message.",
        "simulation_no_key_warning": "⚠️ API Key missing. Simulation cannot proceed.",
        "simulation_advice_header": "AI Response Guidelines",
        "simulation_draft_header": "Recommended Response Draft",
        "button_listen_audio": "Play as Audio",
        "tts_status_ready": "Ready to generate audio",
        "tts_status_generating": "Generating audio...",
        "tts_status_success": "Audio ready!",
        "tts_status_error": "TTS error occurred",
        "history_expander_title": "📝 Load Previous Sessions (Last 10)",
        "initial_query_sample": "I arrived in Paris but my Klook eSIM won't activate…",
        "button_mic_input": "🎙 Voice Input",
        "button_mic_stop": "⏹️ Stop recording",
        "prompt_customer_end": "No further inquiries. Ending chat.",
        "prompt_survey": "This was Agent 000. Have a nice day. [Survey Link]",
        "customer_closing_confirm": "Is there anything else we can assist you with?",
        "customer_positive_response": "I understand. Thank you.",
        "button_email_end_chat": "End supports (Survey Request)",
        "error_mandatory_contact": "Email and Phone number input are mandatory.",
        "customer_attachment_label": "📎 Customer Attachment Upload",
        "attachment_info_llm": "[Customer Attachment: {filename} is confirmed. Reference this file in your response.]",
        "button_retry_translation": "Retry Translation",
        "button_request_hint": "💡 Request Response Hint (AHT Monitored)",
        "button_generate_draft": "🤖 Generate AI Response Draft",
        "draft_generating": "AI is generating a response draft...",
        "draft_success": "✅ AI response draft has been generated. Please review and modify below.",
        "hint_placeholder": "Hints for responses",
        "survey_sent_confirm": "📨 The survey link has been sent. This chat session is now closed。",
        "new_simulation_ready": "You can now start a new simulation.",
        "agent_response_header": "✍️ Agent Response",
        "agent_response_placeholder": "Write a response...",
        "send_response_button": "Send Response",
        "customer_turn_info": "Agent response sent. Generating customer reaction automatically。",
        "generating_customer_response": "Generating customer response...",
        "call_started_message": "Call started. Please click the microphone button below to record your greeting.",
        "call_on_hold_message": "Call is on hold. Recording is available after resuming the call.",
        "recording_complete_transcribing": "🎙️ Recording complete. Transcribing...",
        "transcription_empty_warning": "⚠️ Transcription result is empty. Please record again. (No microphone input or muted)",
        "transcription_error": "[ERROR: Transcription failed]",
        "transcription_no_result": "❌ No transcription result.",
        "customer_escalation_start": "I want to speak to a supervisor",
        "request_rebuttal_button": "Request Customer Reaction",
        "new_simulation_button": "Start New Simulation",
        "history_selectbox_label": "Choose a record to load:",
        "history_load_button": "Load Selected Record",
        "delete_history_button": "❌ Delete All History",
        "delete_confirm_message": "Are you sure you want to delete all records?",
        "delete_confirm_yes": "Yes, Delete",
        "delete_confirm_no": "Cancel",
        "download_history_word": "📥 Download History (Word)",
        "download_history_pptx": "📥 Download History (PPTX)",
        "download_history_pdf": "📥 Download History (PDF)",
        "download_current_session": "📥 Download Current Session",
        "delete_success": "Deleted successfully!",
        "deleting_history_progress": "Deleting history...",
        "search_history_label": "Search History",
        "date_range_label": "Date Filter",
        "history_search_button": "🔍 Search",
        "no_history_found": "No matching history found.",
        "customer_email_label": "Customer Email (Mandatory)",
        "customer_phone_label": "Customer Phone / WhatsApp (Mandatory)",
        "transfer_header": "Language Transfer Request (To Other Teams)",
        "transfer_to_en": "🇺🇸 English Team Transfer",
        "transfer_to_ja": "🇯🇵 Japanese Team Transfer",
        "transfer_to_ko": "🇰🇷 Korean Team Transfer",
        "transfer_system_msg": "📌 System Message: The session language has been transferred to the {target_lang} team per customer request. A new agent (AI) will now respond。",
        "transfer_loading": "Transferring: Translating and reviewing chat history (3-10 minute wait requested from customer)",
        "transfer_summary_header": "🔍 Summary for Transferred Agent (Translated)",
        "transfer_summary_intro": "This is the previous chat history. Please continue the support based on this summary。",
        "llm_translation_error": "❌ Translation failed: LLM response error",
        "timer_metric": "Elapsed Time",
        "timer_info_ok": "AHT (15 min standard)",
        "timer_info_warn": "AHT (Over 10 min)",
        "timer_info_risk": "🚨 Over 15 min: High Risk",
        "solution_check_label": "✅ This response includes a solution/fix。",
        "sentiment_score_label": "Customer Sentiment Score",
        "urgency_score_label": "Urgency Score",
        "customer_gender_label": "Customer Gender",
        "customer_emotion_label": "Customer Emotional State",
        "gender_male": "Male",
        "gender_female": "Female",
        "emotion_happy": "Happy Customer",
        "emotion_dissatisfied": "Dissatisfied Customer",
        "emotion_angry": "Angry Customer",
        "emotion_sad": "Sad/Depressed Customer",
        "emotion_neutral": "Neutral",
        "similarity_chart_title": "Case Similarity",
        "scores_comparison_title": "Sentiment & Satisfaction Scores",
        "similarity_score_label": "Similarity",
        "satisfaction_score_label": "Satisfaction",
        "sentiment_trend_label": "Sentiment Trend",
        "satisfaction_trend_label": "Satisfaction Trend",
        "case_trends_title": "Case Score Trends",
        "date_label": "Date",
        "score_label": "Score (0-100)",
        "customer_characteristics_title": "Customer Characteristics",
        "language_label": "Language",
        "email_provided_label": "Email Provided",
        "phone_provided_label": "Phone Provided",
        "region_label": "Region",
        "btn_request_phone_summary": "Request to summarize histories",

        # --- 추가된 전화 발신 기능 관련 ---
        "button_call_outbound": "Call Outbound",
        "button_call_outbound_to_customer": "Call Outbound to Customer",
        "button_call_outbound_to_provider": "Call Outbound to Provider",
        "call_outbound_system_msg": "📌 System Message: Agent attempted an outbound call to {target}。",
        "call_outbound_simulation_header": "📞 Outbound Call Simulation Result",
        "call_outbound_summary_header": "📞 Summary of Call with Local Partner/Customer",
        "call_outbound_loading": "Connecting call and summarizing outcome... (LLM Call)",
        "call_target_select_label": "Select Call Target",
        "call_target_customer": "Call Customer",
        "call_target_partner": "Call Local Partner",

        # --- 음성 기록 ---
        "voice_rec_header": "Voice Record & Management",
        "record_help": "Record using the microphone or upload a file。",
        "uploaded_file": "Upload Audio File",
        "rec_list_title": "Saved Voice Records",
        "transcribe_btn": "Transcribe (Whisper)",
        "save_btn": "Save Record",
        "transcribing": "Transcribing...",
        "transcript_result": "Transcription:",
        "transcript_text": "Transcribed Text",
        "openai_missing": "OpenAI API Key is missing。",
        "whisper_client_error": "❌ Error: Whisper API client not initialized。",
        "whisper_auth_error": "❌ Whisper API authentication failed。",
        "whisper_format_error": "❌ Error: Unsupported audio format。",
        "whisper_success": "✅ Voice Transcription Complete!",
        "playback": "Playback Recording",
        "retranscribe": "Re-transcribe",
        "delete": "Delete",
        "no_records": "No saved voice records。",
        "saved_success": "Saved successfully!",
        "delete_confirm_rec": "Are you sure you want to delete this voice record?",
        "gcs_not_conf": "GCS not configured or no audio available",
        "gcs_playback_fail": "Failed to play audio",
        "gcs_no_audio": "No audio file found",
        "error": "Error:",
        "firestore_no_db_connect": "DB connection failed",
        "save_history_success": "Saved successfully。",
        "save_history_fail": "Save failed。",
        "delete_fail": "Delete failed",
        "rec_header": "Voice Input & Transcription",
        "whisper_processing": "Processing...",
        "empty_response_warning": "Please enter a response。",
        "customer_no_more_inquiries": "No, that will be all, thank you。",
        "customer_has_additional_inquiries": "Yes, I have an additional question。",
        "agent_confirmed_inquiry": "The agent has confirmed if there are additional inquiries. Automatically generating the customer's final response.",
        "customer_no_more_inquiries_confirmed": "Confirmed that the customer has no further inquiries.",
        "consultation_end": "End of Consultation",
        "consultation_end_instruction": "Click the **Send Survey Link and End Response** button below to end the consultation.",
        "sim_end_chat_button": "Send Survey Link and End Consultations",
        "delete_mic_record": "❌ Delete recordings",

        # --- 첨부 파일 기능 추가 ---
        "attachment_label": "Customer Attachment Upload (Screenshot, etc。)",
        "attachment_placeholder": "Attach a file to explain the situation (optional)",
        "attachment_info_llm": "[Customer Attachment: {filename} is confirmed. Reference this file in your response。]",
        "agent_attachment_label": "Agent Attachment (Screenshot, etc。)",
        "agent_attachment_placeholder": "Select a file to attach to the response (optional)",
        "agent_attachment_status": "📎 Agent attached **{filename}** file to the response。 (File type: {filetype})",

        # --- RAG 오류 메시지 추가 ---
        "rag_embed_error_openai": "RAG embedding failed: OpenAI API Key is invalid or not set。",
        "rag_embed_error_gemini": "RAG embedding failed: Gemini API Key is invalid or not set。",
        "rag_embed_error_nvidia": "RAG embedding failed: NVIDIA API Key is invalid or not set。",
        "rag_embed_error_none": "RAG embedding failed: All required keys (OpenAI, Gemini, NVIDIA) are invalid or not set。 Please configure a key。",

        # --- 전화 기능 관련 추가 ---
        "phone_header": "AI Customer Support Simulator (Phone)",
        "call_status_waiting": "Waiting for incoming call...",
        "call_status_ringing": "Incoming Call from: {number}",
        "button_answer": "📞 Answer Call",
        "button_hangup": "🔴 Hang Up",
        "button_hold": "⏸️ Hold (Mute)",
        "button_resume": "▶️ Resume Call",
        "hold_status": "On Hold (Total Hold Time: {duration})",
        "cc_live_transcript": "🎤 Live CC Transcript",
        "mic_input_status": "🎙️ Agent Voice Input",
        "customer_audio_playback": "🗣️ Customer Audio Playback",
        "agent_response_prompt": "Record your response to the customer。",
        "agent_response_stop_and_send": "⏹️ Stop and share recording to customers",
        "call_end_message": "Call ended. Check AHT and history。",
        "call_query_placeholder": "Enter customer's initial query。",
        "call_number_placeholder": "+1 (555) 123-4567 (Mock Number)",
        "website_url_label": "Website URL (Optional)",
        "website_url_placeholder": "https://example.com (Enter website URL if available)",
        "call_summary_header": "AI Call Summary",
        "customer_audio_header": "Customer Initial Query (Voice)",
        "aht_not_recorded": "⚠️ Call start time not recorded。 Cannot calculate AHT。",
        "no_audio_record": "No initial customer voice record。",
        "customer_query_playing": "🔊 Playing customer inquiry...",
        "query_content_label": "📝 Inquiry content:",
        "auto_play_failed": "Auto-play failed: {error}. Please play manually.",
        "generating_customized_response": "Generating customized customer response...",
        "customer_responded": "🗣️ Customer responded: {reaction}",
        "customer_voice_generation_error": "❌ Customer voice generation error: {error}",
        "button_retry_translation": "Retry Translation",
        "customer_waiting_hold": "[Customer: Please wait...]",
        "agent_hold_message": "[Agent: Call is on hold. Please click the resume button.]",
        
        # --- Video File Upload Related ---
        "video_upload_expander": "Video File Upload/Load",
        "video_sync_enable": "Enable Video Synchronization (Play with TTS)",
        "video_rag_title": "🎥 OpenAI/Gemini Based Video RAG Feature",
        "video_rag_desc": "✅ **Current Implementation (Video RAG):**\n\n1. **LLM Text Analysis**: OpenAI/Gemini API analyzes customer's text to automatically determine emotional state and gestures.\n\n2. **Intelligent Video Selection**: Automatically selects appropriate video clips based on analysis results.\n   - Emotional State: HAPPY, ANGRY, ASKING, SAD, NEUTRAL\n   - Gestures: HAND_WAVE, NOD, SHAKE_HEAD, POINT, NONE\n\n3. **TTS Synchronized Playback**: Plays selected video and TTS-generated audio simultaneously.\n\n**Usage:**\n- Upload video files by gender (male/female) and emotional state.\n- Gesture-specific videos can also be uploaded (e.g., `male_happy_hand_wave.mp4`).\n- LLM automatically selects appropriate videos based on customer's speech content.",
        "video_gender_emotion_setting": "Video Settings by Gender and Emotional State",
        "video_gender_label": "Gender",
        "video_gender_male": "Male",
        "video_gender_female": "Female",
        "video_emotion_label": "Emotional State",
        "video_upload_label": "Video File Upload ({gender} - {emotion})",
        "video_current_selection": "📹 Current Selection: {gender} - {emotion}",
        "video_upload_prompt": "💡 Please upload the '{filename}' video file.",
        "video_save_path": "📂 Video Save Path:",
        "video_directory_empty": "⚠️ There is no file in the video directory. Please upload the file.",
        "video_directory_not_exist": "⚠️ Video directory does not exist: {path}",
        "video_local_path_input": "Or Enter Local File Path",
        "video_local_path_placeholder": "e.g., C:\\Users\\Admin\\Downloads\\video.mp4 or video.mp4",
        "video_current_avatar": "📺 Current Customer Avatar Video",
        "video_avatar_upload_prompt": "💡 Upload the '{filename}' video file to display the video.",
        "video_uploaded_files": "📁 Uploaded Video Files:",
        "video_bytes_saved": "✅ Video bytes saved: {name} ({size} MB)",
        "video_empty_error": "❌ Video file is empty. Please upload again.",
        "video_upload_error": "❌ Error occurred during video upload: {error}",
        "video_playback_error": "❌ Failed to play video.",
        "video_auto_play_info": "💡 This video will automatically play in '{gender} - {emotion}' state.",
        "video_preview_error": "Video preview error",
        "video_similar_gender": "Other videos of the same gender",
        "video_rename_hint": "💡 To use one of the videos above, rename the file or upload a new one.",
        "video_more_files": "... and {count} more",
        "avatar_status_info": "Status: {state} | Gender: {gender}",
        "customer_video_simulation": "Customer Video Simulation",
        "customer_avatar": "Customer Avatar",
        "faq_question_prefix": "Q{num}.",
        "visualization_chart": "Visualization Chart",
        "company_search_or_select": "Please search or select a company name.",

    },

    # --- ⭐ 일본어 버전 (한국어 100% 매칭) ---
    "ja": {
        "title": "パーソナライズAI学習コーチ (音声・ローカルDB)",
        "sidebar_title": "📚 AI学習コーチ設定",
        "file_uploader": "学習資料をアップロード (PDF, TXT, HTML)",
        "button_start_analysis": "資料分析開始 (RAGインデックス作成)",
        "rag_tab": "RAG知識チャットボット",
        "content_tab": "カスタム学習コンテンツ生成",
        "lstm_tab": "LSTM達成度予測ダッシュボード",
        "sim_tab_chat_email": "AI顧客対応シミュレーター(チャット・メール)",
        "sim_tab_phone": "AI顧客対応シミュレーター(電話)",
        "simulator_tab": "AI顧客対応シミュレーター",
        "company_info_tab": "会社情報およびFAQ",
        "sim_tab_chat_email_desc": "顧客対応業務において、チャットやメールで実際に問い合わせ対応ができる実戦向けの仮想シナリオです。AIが対応ガイドラインと草案を生成し、顧客の反応をシミュレートして実戦向けの訓練が可能です。",
        "sim_tab_phone_desc": "顧客対応業務において、電話で実際に問い合わせ対応ができる実戦向けの仮想シナリオです。音声録音およびリアルタイムCC字幕機能を提供し、電話通話シミュレーションを通じて実戦対応能力を向上させることができます。",
        "rag_tab_desc": "アップロードされた文書に基づいて質問に答える知識チャットボットです。PDF、TXT、HTMLファイルをアップロードしてRAG（Retrieval-Augmented Generation）インデックスを構築し、文書内容に基づいて正確な回答を提供します。",
        "content_tab_desc": "AIを活用して個人向けの学習コンテンツを生成する機能です。学習テーマと難易度に合わせて要点サマリー、選択式クイズ、実践例などを生成できます。",
        "lstm_tab_desc": "LSTMモデルを活用して学習者の達成度を予測し、ダッシュボードで可視化する機能です。過去のクイズスコアデータを分析して将来の達成度を予測し、学習成果を視覚的に確認できます。",
        "company_info_tab_desc": "会社別の詳細情報、人気商品、話題のニュース、FAQを検索・管理できる機能です。会社紹介、人気商品、話題のニュースを視覚化して一目で確認できます。",
        "voice_rec_header_desc": "音声録音および転写結果を管理・保存する機能です。マイクで録音するかファイルをアップロードしてWhisper APIを通じて音声をテキストに変換し、転写結果を保存・管理できます。",
        "more_features_label": "その他の機能",
        "rag_header": "RAG知識チャットボット (ドキュメントQ&A)",
        "rag_desc": "アップロードされた資料に基づいて質問に回答します。",
        "rag_input_placeholder": "資料について質問してください",
        "llm_error_key": "⚠️ 注意: GEMINI_API_KEY が設定されていません。",
        "llm_error_init": "LLM 初期化エラー：APIキーを確認してください。",
        "content_header": "カスタム学習コンテンツ生成",
        "content_desc": "学習テーマと難易度に応じてコンテンツを生成します。",
        "topic_label": "学習テーマ",
        "level_label": "難易度",
        "content_type_label": "コンテンツ種類",
        "level_options": ["初級", "中級", "上級"],
        "content_options": ["要点サマリー", "選択式クイズ10問", "実践例アイデア"],
        "button_generate": "生成する",
        "warning_topic": "学習テーマを入力してください。",
        "lstm_header": "LSTM達成度予測ダッシュボード",
        "lstm_desc": "仮想クイズスコアを使用して達成度を予測します。",
        "lang_select": "言語選択",
        "company_info_faq_settings": "会社別詳細情報とFAQ",
        "search_company": "会社名検索",
        "company_info": "会社情報",
        "company_faq": "よくある質問",
        "faq_question": "質問",
        "faq_answer": "回答",
        "popular_products": "人気商品",
        "trending_topics": "話題のニュース",
        "company_details": "会社詳細情報",
        "no_company_found": "に該当する会社が見つかりません。",
        "no_company_selected": "会社名を検索または選択してください。",
        "product_popularity": "商品人気度",
        "topic_trends": "話題トレンド",
        "select_company": "会社選択",
        "faq_search": "FAQ検索",
        "faq_search_placeholder": "FAQ検索語を入力してください",
        "faq_search_placeholder_extended": "FAQ検索語を入力してください（商品名、サービス名なども検索可能）",
        "button_search_faq": "検索",
        "company_search_placeholder": "例: サムスン、ネイバー、グーグル、アップルなど",
        "company_search_button": "検索",
        "generating_company_info": "会社情報を生成中...",
        "button_copy_answer": "回答コピー",
        "button_copy_hint": "ヒントコピー",
        "button_download_answer": "回答ダウンロード",
        "button_download_hint": "ヒントダウンロード",
        "copy_instruction": "💡 上のテキストを選択してCtrl+C（Mac: Cmd+C）でコピーしてください。",
        "copy_help_text": "テキストを選択してCtrl+C（またはCmd+C）でコピーしてください。",
        "button_reset": "リセット",
        "answer_displayed": "回答が表示されました。上のテキストをコピーしてください。",
        "hint_displayed": "ヒントが表示されました。上のテキストをコピーしてください。",
        "ai_answer_generated": "AI回答が生成されました。",
        "hint_generated": "対応ヒントが生成されました。",
        "warning_enter_inquiry": "顧客問い合わせ内容を入力してください。",
        "customer_inquiry_review_desc": "エージェントが上司に顧客問い合わせ内容を再確認し、AI回答とヒントを生成できる機能です。",
        "all_companies": "すべて",
        "optional": "任意",
        "no_faq_for_company": "{company}のFAQがありません。",
        "related_products": "関連商品",
        "related_trending_news": "関連話題のニュース",
        "related_company_info": "関連会社紹介内容",
        "related_faq": "関連FAQ",
        "items": "件",
        "popularity": "人気度",
        "no_faq_for_product": "該当商品に関連するFAQが見つかりません。商品情報のみ表示されます。",
        "generating_detail": "詳細内容を生成中...",
        "checking_additional_info": "詳細内容: {topic}に関する追加情報を確認中です。",
        "button_generate_faq": "FAQ生成",
        "button_add_company": "顧客問い合わせ再確認",
        "customer_inquiry_review": "顧客問い合わせ再確認",
        "inquiry_question_label": "顧客問い合わせ内容",
        "inquiry_question_placeholder": "顧客が問い合わせた内容を入力してください",
        "inquiry_attachment_label": "📎 顧客添付ファイルアップロード (写真/スクリーンショット)",
        "inquiry_attachment_help": "特にキャンセル不可の旅行商品で、飛行機の遅延、パスポートの問題などやむを得ない理由がある場合は、必ず写真やスクリーンショットを添付してください。",
        "inquiry_attachment_uploaded": "✅ 添付ファイルがアップロードされました: {filename}",
        "extracting_file_content": "ファイル内容を抽出中...",
        "detecting_language": "言語を検出中...",
        "translating_content": "ファイル内容を翻訳中...",
        "file_translated": "✅ ファイル内容が翻訳されました。",
        "file_extraction_error": "ファイル内容の抽出中にエラーが発生しました: {error}",
        "ocr_requires_manual": "画像OCRにはGemini APIキーが必要です。画像のテキストを手動で入力してください。",
        "ocr_error": "画像からのテキスト抽出中にエラーが発生しました: {error}",
        "button_generate_ai_answer": "AI回答生成",
        "button_generate_hint": "対応ヒント生成",
        "ai_answer_header": "AI推奨回答",
        "hint_header": "対応ヒント",
        "generating_ai_answer": "AI回答を生成中...",
        "generating_hint": "対応ヒントを生成中...",
        "button_edit_company": "会社情報編集",
        "button_show_company_info": "会社紹介を見る",
        "no_faq_results": "検索結果がありません。",
        "faq_search_results": "FAQ検索結果",
        "add_company_name": "会社名",
        "add_company_info": "会社情報",
        "generate_faq_question": "質問",
        "generate_faq_answer": "回答",
        "button_save_faq": "FAQ保存",
        "button_cancel": "キャンセル",
        "faq_saved_success": "FAQが保存されました。",
        "company_added_success": "会社が追加されました。",
        "company_updated_success": "会社情報が更新されました。",
        "embed_success": "{count}個のチャンクでDB構築完了!",
        "embed_fail": "埋め込み失敗：クォータ超過またはネットワーク問題。",
        "warning_no_files": "資料をアップロードしてください。",
        "warning_rag_not_ready": "RAGが準備できていません。",
        "quiz_fail_structure": "クイズデータの形式が正しくありません。",
        "select_answer": "回答を選択してください",
        "check_answer": "回答を確認",
        "next_question": "次の質問",
        "correct_answer": "正解！ 🎉",
        "incorrect_answer": "不正解 😞",
        "correct_is": "正解",
        "explanation": "解説",
        "quiz_complete": "クイズ完了!",
        "score": "スコア",
        "retake_quiz": "再挑戦",
        "question_label": "質問",
        "correct_questions": "正解",
        "incorrect_questions": "不正解",
        "question_result": "問題結果",
        "your_answer": "あなたの答え",
        "correct_answer_label": "正解",
        "quiz_error_llm": "퀴즈 생성 실패：JSON形式が正しくありません。",
        "quiz_original_response": "LLM 原本回答",
        "firestore_loading": "RAGインデックス読み込み中...",
        "firestore_no_index": "保存されたRAGインデックスが見つかりません。",
        "db_save_complete": "(DB保存完了)",
        "data_analysis_progress": "資料分析中...",
        "response_generating": "応答生成中...",
        "lstm_result_header": "達成度予測結果",
        "lstm_score_metric": "予測達成度",
        "lstm_score_info": "次のスコア予測: **{predicted_score:.1f}점**",
        "lstm_rerun_button": "新しいデータで再予測",

        # --- 토스트 메시지 추가 ---
        "toast_like": "🔥 コンテンツを気に入っていただけました！ (+1 カウント反映)",
        "toast_dislike": "😔 より良いコンテンツのためフィードバックを記録しました。",
        "toast_share": "🌐 コンテンツリンクが生成されました。",
        "toast_copy": "✅ コンテンツがクリップボードにコピーされました！",
        "toast_more": "ℹ️ その他のオプション（印刷、PDF保存など）",
        "mock_pdf_save": "📥 PDFで保存",
        "mock_word_open": "📑 Wordで開く",
        "mock_print": "🖨 印刷",
        # --- 토스트 메시지 끝 ---

        # --- Simulator ---
        "simulator_header": "AI顧客対応シミュレーター",
        "simulator_desc": "難しい顧客問い合わせに対するAIのガイドラインと草案を生成します。",
        "customer_query_label": "顧客からの問い合わせ内容 (リンク可)",
        "customer_type_label": "顧客タイプ",
        "customer_type_options": ["一般的な問い合わせ", "難しい顧客", "非常に不満な顧客"],
        "button_simulate": "応対ガイド生成",
        "customer_generate_response_button": "顧客の返信を生成",
        "send_closing_confirm_button": "追加のご質問有無を確認するメッセージを送信",
        "simulation_warning_query": "お問い合わせ内容を入力してください。",
        "simulation_no_key_warning": "⚠️ APIキー不足のため応対生成不可。",
        "simulation_advice_header": "AI対応ガイドライン",
        "simulation_draft_header": "推奨応対草案",
        "button_listen_audio": "音声で聞く",
        "tts_status_ready": "音声生成準備完了",
        "tts_status_generating": "音声生成中...",
        "tts_status_success": "音声準備完了！",
        "tts_status_error": "TTS エラーが発生しました",
        "history_expander_title": "📝 過去の対応履歴を読み込む (最新10件)",
        "initial_query_sample": "パリに到着しましたが、KlookのeSIMが使えません…",
        "button_mic_input": "🎙 音声入力",
        "button_mic_stop": "⏹️ 録音終了",
        "prompt_customer_end": "追加の質問がないためチャットを終了します。",
        "prompt_survey": "担当エージェント000でした。良い一日をお過ごしください。 [アンケートリンク]",
        "customer_closing_confirm": "他のお問合せはございませんでしょうか。",
        "customer_positive_response": "承知いたしました。ありがとうございます。",
        "button_email_end_chat": "応対終了（アンケート）",
        "error_mandatory_contact": "メールアドレスと電話番号の入力は必須です。",
        "customer_attachment_label": "📎 顧客添付ファイルアップロード",
        "attachment_info_llm": "[顧客添付ファイル: {filename}が確認されました。このファイルを参照して対応してください。]",
        "button_retry_translation": "翻訳を再試行",
        "button_request_hint": "💡 応対ヒントを要請 (AHT モニタリング中)",
        "button_generate_draft": "🤖 AI応答草案生成",
        "draft_generating": "AIが応答草案を生成中です...",
        "draft_success": "✅ AI応答草案が生成されました。以下で確認して修正してください。",
        "hint_placeholder": "お問合せの応対に対するヒント：",
        "new_simulation_ready": "新しいシミュレーションを開始できます。",
        "survey_sent_confirm": "📨 アンケートリンクを送信しました。このチャットは終了しました。",
        "agent_response_header": "✍️ エージェント応答",
        "agent_response_placeholder": "顧客へ返信内容を入力…",
        "generating_customer_response": "顧客の返信を生成中...",
        "call_started_message": "通話が開始されました。下のマイクボタンをクリックして挨拶を録音してください。",
        "call_on_hold_message": "通話が保留中です。通話を再開した後、録音が可能です。",
        "recording_complete_transcribing": "🎙️ 録音完了。転写処理中...",
        "transcription_empty_warning": "⚠️ 転写結果が空です。もう一度録音してください。（マイク入力がないか、ミュートされています）",
        "transcription_error": "[ERROR: 転写失敗]",
        "transcription_no_result": "❌ 転写結果がありません。",
        "send_response_button": "返信送信",
        "customer_turn_info": "エージェント応答送信完了。顧客の反応を自動生成中です。",
        "generating_customer_response": "顧客の反応を生成中...",
        "customer_escalation_start": "上級の担当者と話したい",
        "request_rebuttal_button": "顧客の反応を生成",
        "new_simulation_button": "新規シミュレーション",
        "history_selectbox_label": "履歴を選択:",
        "history_load_button": "履歴を読み込む",
        "delete_history_button": "❌ 全履歴削除",
        "delete_confirm_message": "すべての履歴を削除しますか？",
        "delete_confirm_yes": "はい、削除します。",
        "download_history_word": "📥 履歴ダウンロード (Word)",
        "download_history_pptx": "📥 履歴ダウンロード (PPTX)",
        "download_history_pdf": "📥 履歴ダウンロード (PDF)",
        "download_current_session": "📥 現在のセッションをダウンロード",
        "delete_confirm_no": "いいえ、維持します。",
        "delete_success": "削除完了！",
        "deleting_history_progress": "削除中...",
        "search_history_label": "履歴検索",
        "date_range_label": "日付フィルター",
        "history_search_button": "🔍 検索",
        "no_history_found": "該当する履歴はありません。",
        "customer_email_label": "顧客メールアドレス（必修）",
        "customer_phone_label": "顧客連絡先 / 電話番号（必修）",
        "transfer_header": "言語切り替え要請（他チームへ）",
        "transfer_to_en": "🇺🇸 英語チームへ転送",
        "transfer_to_ja": "🇯🇵 日本語チームへ転送",
        "transfer_to_ko": "🇰🇷 韓国語チームへ転送",
        "transfer_system_msg": "📌 システムメッセージ: 顧客の要請により、対応言語が {target_lang} チームへ切り替えられました。新しい担当者(AI)が対応します。",
        "transfer_loading": "転送中: 過去のチャット履歴を翻訳およびレビューしています (お客様には3〜10分のお時間をいただいています)",
        "transfer_summary_header": "🔍 転送された担当者向けの要約 (翻訳済み)",
        "transfer_summary_intro": "これが顧客との過去のチャット履歴です。この要約に基づいてサポートを続けてください。",
        "llm_translation_error": "❌ 翻訳失敗: LLM応答エラー",
        "timer_metric": "経過時間",
        "timer_info_ok": "AHT (15分基準)",
        "timer_info_warn": "AHT (10分経過)",
        "timer_info_risk": "🚨 15分経過: 高いリスク",
        "solution_check_label": "✅ この応答に解決策/対応策が含まれています。",
        "sentiment_score_label": "顧客の感情スコア",  # <--- 추가/수정
        "urgency_score_label": "緊急度スコア",
        "customer_gender_label": "顧客性別",
        "customer_emotion_label": "顧客感情状態",
        "gender_male": "男性",
        "gender_female": "女性",
        "emotion_happy": "気分良い顧客",
        "emotion_dissatisfied": "不満な顧客",
        "emotion_angry": "怒った顧客",
        "emotion_sad": "悲しい/憂鬱な顧客",
        "emotion_neutral": "中立",
        "similarity_chart_title": "類似性ケースの比率",
        "scores_comparison_title": "感情及び満足度のスコア",
        "similarity_score_label": "類似性",
        "satisfaction_score_label": "満足度",
        "sentiment_trend_label": "感情のスコアの推測",
        "satisfaction_trend_label": "満足度のスコアの推測",
        "case_trends_title": "過去に推定されたスコア",
        "date_label": "日付",
        "score_label": "スコア (0-100)",
        "customer_characteristics_title": "顧客の性格",
        "language_label": "言語",
        "email_provided_label": "提供されたメールアドレス",
        "phone_provided_label": "提供された電話番号",
        "region_label": "地域",
        "btn_request_phone_summary": "履歴を要約する",

        # --- 추가된 전화 발신 기능 관련 ---
        "button_call_outbound": "電話発信",
        "button_call_outbound_to_customer": "顧客へ電話発信",
        "button_call_outbound_to_provider": "現地業者へ電話発信",
        "call_outbound_system_msg": "📌 システムメッセージ: エージェントが{target}へ電話発信を試みました。",
        "call_outbound_simulation_header": "📞 電話発信シミュレーション結果",
        "call_outbound_summary_header": "📞 現地業者/顧客との通話要約",
        "call_outbound_loading": "電話接続と通話結果の整理中... (LLMコール)",
        "call_target_select_label": "発信先選択",
        "call_target_customer": "顧客へ電話発信",
        "call_target_partner": "現地業者へ電話発信",

        # --- Voice ---
        "voice_rec_header": "音声記録＆管理",
        "record_help": "録音するか音声ファイルをアップロードします。",
        "uploaded_file": "音声ファイルをアップロード",
        "rec_list_title": "保存された音声記録",
        "transcribe_btn": "転写 (Whisper)",
        "save_btn": "音声記録を保存",
        "transcribing": "音声を転写中...",
        "transcript_result": "転写結果:",
        "transcript_text": "転写テキスト",
        "openai_missing": "OpenAI APIキーがありません。",
        "whisper_client_error": "❌ エラー: Whisper APIクライアントが初期化されていません。",
        "whisper_auth_error": "❌ Whisper API認証に失敗しました。",
        "whisper_format_error": "❌ エラー: この音声形式はサポートされていません。",
        "whisper_success": "✅ 音声転写完了！",
        "playback": "録音再生",
        "retranscribe": "再転写",
        "delete": "削除",
        "no_records": "保存された音声記録はありません。",
        "saved_success": "保存しました！",
        "delete_confirm_rec": "この音声記録を削除しますか？",
        "gcs_not_conf": "GCSが設定されていないか、音声がありません。",
        "gcs_playback_fail": "音声の再生に失敗しました。",
        "gcs_no_audio": "音声ファイルがありません。",
        "error": "エラー:",
        "firestore_no_db_connect": "DB接続失敗",
        "save_history_success": "保存完了。",
        "save_history_fail": "保存失敗。",
        "delete_fail": "削除失敗",
        "rec_header": "音声入力＆転写",
        "whisper_processing": "処理中...",
        "empty_response_warning": "応答を入力してください。",
        "customer_no_more_inquiries": "いいえ、結構です。大丈夫です。有難う御座いました。",
        "customer_has_additional_inquiries": "はい、追加の問い合わせがあります。",
        "agent_confirmed_inquiry": "エージェントが追加の問い合わせの有無を確認しました。お客様の最終回答を自動生成します。",
        "customer_no_more_inquiries_confirmed": "お客様がこれ以上問い合わせがないことを確認しました。",
        "consultation_end": "相談終了",
        "consultation_end_instruction": "以下の**アンケートリンク送信および応対終了**ボタンをクリックして相談を終了してください。",
        "sim_end_chat_button": "アンケートリンクを送信して応対終了",
        "delete_mic_record": "録音を削除する",

        # --- 첨부 파일 기능 추가 ---
        "attachment_label": "顧客の添付ファイルアップロード (スクリーンショットなど)",
        "attachment_placeholder": "ファイルを添付して状況を説明してください（オプション）",
        "attachment_status_llm": "顧客が **{filename}** 파일을 첨부했습니다. 이 파일을 스크린샷이라고 가정하고 응대 초안과 가이드라인에 반영해주세요. (ファイルタイプ: {filetype})",
        "agent_attachment_label": "エージェント添付ファイル (スクリーンショットなど)",
        "agent_attachment_placeholder": "応答に添付するファイルを選択してください（オプション）",
        "agent_attachment_status": "📎 エージェントが **{filename}** ファイルを応答に添付しました。(ファイルタイプ: {filetype})",

        # --- RAG 오류 메시지 추가 ---
        "rag_embed_error_openai": "RAG embedding failed: OpenAI API Key is invalid or not set。",
        "rag_embed_error_gemini": "RAG embedding failed: Gemini API Key is invalid or not set。",
        "rag_embed_error_nvidia": "RAG embedding failed: NVIDIA API Key is invalid or not set。",
        "rag_embed_error_none": "RAG embedding failed: All required keys (OpenAI, Gemini, NVIDIA) are invalid or not set。 Please configure a key。",

        # --- 電話機能関連追加 ---
        "phone_header": "AI顧客対応シミュレーター(電話)",
        "call_status_waiting": "着信待ち...",
        "call_status_ringing": "着信中: {number}",
        "button_answer": "📞 電話に出る",
        "button_hangup": "🔴 電話を切る",
        "button_hold": "⏸️ 保留 (ノイズ遮断)",
        "button_resume": "▶️ 通話再開",
        "hold_status": "保留中 (累計保留時間: {duration})",
        "cc_live_transcript": "🎤 リアルタイムCC字幕 / 転写",
        "mic_input_status": "🎙️ エージェントの音声入力",
        "customer_audio_playback": "🗣️ 顧客の音声再生",
        "agent_response_prompt": "顧客への応答を録音してください。",
        "agent_response_stop_and_send": "⏹️録音を終了して、顧客へ転送する",
        "call_end_message": "通話が終了しました。AHTと履歴を確認してください。",
        "call_query_placeholder": "顧客からの最初の問い合わせ内容を入力してください。",
        "call_number_placeholder": "+81 90-xxxx-xxxx (仮想番号)",
        "website_url_label": "ホームページのウェブアドレス (任意)",
        "website_url_placeholder": "https://example.com (ホームページのアドレスがある場合は入力してください)",
        "call_summary_header": "AI 通話要約",
        "customer_audio_header": "顧客の最初の問い合わせ (音声)",
        "aht_not_recorded": "⚠️ 通話開始時間が記録されていないため、AHTを計算できません。",
        "no_audio_record": "顧客の最初の音声記録はありません。",
        "customer_query_playing": "🔊 顧客の問い合わせを再生中です。",
        "query_content_label": "📝 問い合わせ内容:",
        "auto_play_failed": "自動再生に失敗しました: {error}。手動で再生してください。",
        "generating_customized_response": "顧客向けカスタマイズされた反応を生成中...",
        "customer_responded": "🗣️ 顧客が応答しました: {reaction}",
        "customer_voice_generation_error": "❌ 顧客の音声生成エラー: {error}",
        "button_retry_translation": "翻訳を再試行",
        "customer_waiting_hold": "[顧客: お待ちください...]",
        "agent_hold_message": "[エージェント: 通話が保留中です。通話再開ボタンをクリックしてください。]",
        
        # --- ビデオファイルアップロード関連 ---
        "video_upload_expander": "ビデオファイルアップロード/ロード",
        "video_sync_enable": "ビデオ同期を有効化 (TTSと一緒に再生)",
        "video_rag_title": "🎥 OpenAI/GeminiベースのビデオRAG機能",
        "video_rag_desc": "✅ **現在の実装方式 (ビデオRAG):**\n\n1. **LLMテキスト分析**: OpenAI/Gemini APIが顧客のテキストを分析し、感情状態とジェスチャーを自動判定します。\n\n2. **インテリジェントビデオ選択**: 分析結果に基づいて適切なビデオクリップを自動選択します。\n   - 感情状態: HAPPY, ANGRY, ASKING, SAD, NEUTRAL\n   - ジェスチャー: HAND_WAVE, NOD, SHAKE_HEAD, POINT, NONE\n\n3. **TTS同期再生**: 選択されたビデオとTTSで生成された音声を同時に再生します。\n\n**使用方法:**\n- 性別(男性/女性)と感情状態別にビデオファイルをアップロードしてください。\n- ジェスチャー別のビデオもアップロード可能です (例: `male_happy_hand_wave.mp4`)。\n- 顧客が話す内容に応じてLLMが自動的に適切なビデオを選択します。",
        "video_gender_emotion_setting": "性別および感情状態別ビデオ設定",
        "video_gender_label": "性別",
        "video_gender_male": "男性",
        "video_gender_female": "女性",
        "video_emotion_label": "感情状態",
        "video_upload_label": "ビデオファイルアップロード ({gender} - {emotion})",
        "video_current_selection": "📹 現在の選択: {gender} - {emotion}",
        "video_upload_prompt": "💡 '{filename}' ビデオファイルをアップロードしてください。",
        "video_save_path": "📂 ビデオ保存パス:",
        "video_directory_empty": "⚠️ ビデオディレクトリにファイルがありません。ファイルをアップロードしてください。",
        "video_directory_not_exist": "⚠️ ビデオディレクトリが存在しません: {path}",
        "video_local_path_input": "またはローカルファイルパス入力",
        "video_local_path_placeholder": "例: C:\\Users\\Admin\\Downloads\\video.mp4 または video.mp4",
        "video_current_avatar": "📺 現在の顧客アバター映像",
        "video_avatar_upload_prompt": "💡 '{filename}' ビデオファイルをアップロードすると映像が表示されます。",
        "video_uploaded_files": "📁 アップロードされたビデオファイル:",
        "video_bytes_saved": "✅ ビデオバイト保存完了: {name} ({size} MB)",
        "video_empty_error": "❌ ビデオファイルが空です。再度アップロードしてください。",
        "video_upload_error": "❌ ビデオアップロード中にエラーが発生しました: {error}",
        "video_playback_error": "❌ ビデオ再生に失敗しました。",
        "video_auto_play_info": "💡 このビデオは '{gender} - {emotion}' 状態で自動的に再生されます。",
        "video_preview_error": "ビデオプレビューエラー",
        "video_similar_gender": "同じ性別の他のビデオ",
        "video_rename_hint": "💡 上記のビデオのいずれかを使用するには、ファイル名を変更するか、新しくアップロードしてください。",
        "video_more_files": "... 他 {count}件",
        "avatar_status_info": "状態: {state} | 性別: {gender}",
        "customer_video_simulation": "顧客映像シミュレーション",
        "customer_avatar": "顧客アバター",
        "faq_question_prefix": "Q{num}.",
        "visualization_chart": "可視化チャート",
        "company_search_or_select": "会社名を検索または選択してください。",
    }
}

# ========================================
# 1-1. Session State 초기화 (전화 발신 관련 상태 추가)
# ========================================
# ⭐ 사이드바 버튼은 사이드바 블록 안으로 이동해야 함
# 여기서는 세션 상태만 초기화

if "language" not in st.session_state:
    st.session_state.language = DEFAULT_LANG
if "is_llm_ready" not in st.session_state:
    st.session_state.is_llm_ready = False
if "llm_init_error_msg" not in st.session_state:
    st.session_state.llm_init_error_msg = ""
if "uploaded_files_state" not in st.session_state:
    st.session_state.uploaded_files_state = None
if "is_rag_ready" not in st.session_state:
    st.session_state.is_rag_ready = False
if "rag_vectorstore" not in st.session_state:
    st.session_state.rag_vectorstore = None
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "agent_input" not in st.session_state:
    st.session_state.agent_input = ""
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
if "simulator_messages" not in st.session_state:
    st.session_state.simulator_messages = []
if "simulator_memory" not in st.session_state:
    st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history")
if "simulator_chain" not in st.session_state:
    st.session_state.simulator_chain = None
if "initial_advice_provided" not in st.session_state:
    st.session_state.initial_advice_provided = False
if "is_chat_ended" not in st.session_state:
    st.session_state.is_chat_ended = False
if "show_delete_confirm" not in st.session_state:
    st.session_state.show_delete_confirm = False
if "customer_query_text_area" not in st.session_state:
    st.session_state.customer_query_text_area = ""
if "agent_response_area_text" not in st.session_state:
    st.session_state.agent_response_area_text = ""
if "reset_agent_response_area" not in st.session_state:
    st.session_state.reset_agent_response_area = False
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "sim_audio_bytes" not in st.session_state:
    st.session_state.sim_audio_bytes = None
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "idle"
    # idle → initial_customer → supervisor_advice → agent_turn → customer_turn → closing
if "openai_client" not in st.session_state:
    st.session_state.openai_client = None
if "openai_init_msg" not in st.session_state:
    st.session_state.openai_init_msg = ""
if "sim_stage" not in st.session_state:
    st.session_state.sim_stage = "WAIT_FIRST_QUERY"
    # WAIT_FIRST_QUERY (초기 문의 입력)
    # AGENT_TURN (에이전트 응답 입력)
    # CUSTOMER_TURN (고객 반응 생성 요청)
    # WAIT_CLOSING_CONFIRMATION_FROM_AGENT (고객이 감사, 에이전트가 종료 확인 메시지 보내기 대기)
    # WAIT_CUSTOMER_CLOSING_RESPONSE (종료 확인 메시지 보냄, 고객의 마지막 응답 대기)
    # FINAL_CLOSING_ACTION (최종 종료 버튼 대기)
    # CLOSING (채팅 종료)
    # ⭐ 추가: OUTBOUND_CALL_IN_PROGRESS (전화 발신 진행 중)
if "start_time" not in st.session_state:  # AHT 타이머 시작 시간
    st.session_state.start_time = None
if "is_solution_provided" not in st.session_state:  # 솔루션 제공 여부 플래그
    st.session_state.is_solution_provided = False
if "transfer_summary_text" not in st.session_state:  # 이관 시 번역된 요약
    st.session_state.transfer_summary_text = ""
if "translation_success" not in st.session_state:  # 번역 성공 여부 추적
    st.session_state.translation_success = True
if "language_transfer_requested" not in st.session_state:  # 고객의 언어 이관 요청 여부
    st.session_state.language_transfer_requested = False
if "customer_attachment_file" not in st.session_state:  # 고객 첨부 파일 정보
    st.session_state.customer_attachment_file = None
if "language_at_transfer" not in st.session_state:  # 현재 언어와 비교를 위한 변수
    st.session_state.language_at_transfer = st.session_state.language
if "language_at_transfer_start" not in st.session_state:  # 번역 재시도를 위한 원본 언어
    st.session_state.language_at_transfer_start = st.session_state.language
if "transfer_retry_count" not in st.session_state:
    st.session_state.transfer_retry_count = 0
if "customer_type_sim_select" not in st.session_state:  # FIX: Attribute Error 해결
    # LANG이 정의되기 전이므로 기본값을 직접 설정
    default_customer_type = "까다로운 고객"  # 한국어 기본값
    if st.session_state.language == "en":
        default_customer_type = "Difficult Customer"
    elif st.session_state.language == "ja":
        default_customer_type = "難しい顧客"
    st.session_state.customer_type_sim_select = default_customer_type
if "customer_email" not in st.session_state:  # FIX: customer_email 초기화
    st.session_state.customer_email = ""
if "customer_phone" not in st.session_state:  # FIX: customer_phone 초기화
    st.session_state.customer_phone = ""
if "agent_response_input_box_widget" not in st.session_state:  # FIX: customer_phone 초기화
    st.session_state.agent_response_input_box_widget = ""
if "sim_instance_id" not in st.session_state:  # FIX: DuplicateWidgetID 방지용 인스턴스 ID 초기화
    st.session_state.sim_instance_id = str(uuid.uuid4())
if "sim_attachment_context_for_llm" not in st.session_state:
    st.session_state.sim_attachment_context_for_llm = ""
if "realtime_hint_text" not in st.session_state:
    st.session_state.realtime_hint_text = ""
# ⭐ 추가: 전화 발신 관련 상태
if "sim_call_outbound_summary" not in st.session_state:
    st.session_state.sim_call_outbound_summary = ""
if "sim_call_outbound_target" not in st.session_state:
    st.session_state.sim_call_outbound_target = None
# ----------------------------------------------------------------------
# ⭐ 전화 기능 관련 상태 추가
if "call_sim_stage" not in st.session_state:
    st.session_state.call_sim_stage = "WAITING_CALL"  # WAITING_CALL, RINGING, IN_CALL, CALL_ENDED
if "call_sim_mode" not in st.session_state:
    st.session_state.call_sim_mode = "INBOUND"  # INBOUND or OUTBOUND
if "incoming_phone_number" not in st.session_state:
    st.session_state.incoming_phone_number = "+82 10-1234-5678"
if "is_on_hold" not in st.session_state:
    st.session_state.is_on_hold = False
if "hold_start_time" not in st.session_state:
    st.session_state.hold_start_time = None
if "total_hold_duration" not in st.session_state:
    st.session_state.total_hold_duration = timedelta(0)
if "current_customer_audio_text" not in st.session_state:
    st.session_state.current_customer_audio_text = ""
if "current_agent_audio_text" not in st.session_state:
    st.session_state.current_agent_audio_text = ""
if "agent_response_input_box_widget_call" not in st.session_state:  # 전화 탭 전용 입력창
    st.session_state.agent_response_input_box_widget_call = ""
if "call_initial_query" not in st.session_state:  # 전화 탭 전용 초기 문의
    st.session_state.call_initial_query = ""
if "call_website_url" not in st.session_state:  # 전화 탭 전용 홈페이지 주소
    st.session_state.call_website_url = ""
# ⭐ 추가: 통화 요약 및 초기 고객 음성 저장소
if "call_summary_text" not in st.session_state:
    st.session_state.call_summary_text = ""
if "customer_initial_audio_bytes" not in st.session_state:  # 고객의 첫 음성 (TTS 결과) 저장
    st.session_state.customer_initial_audio_bytes = None
if "supervisor_policy_context" not in st.session_state:
    # Supervisor가 업로드한 예외 정책 텍스트를 저장합니다.
    st.session_state.supervisor_policy_context = ""
if "agent_policy_attachment_content" not in st.session_state:
    # 에이전트가 업로드한 정책 파일 객체(또는 내용)를 저장합니다.
    st.session_state.agent_policy_attachment_content = ""
if "customer_attachment_b64" not in st.session_state:
    st.session_state.customer_attachment_b64 = ""
if "customer_history_summary" not in st.session_state:
    st.session_state.customer_history_summary = ""
if "customer_avatar" not in st.session_state:
    st.session_state.customer_avatar = {
        "gender": "male",  # 기본값
        "state": "NEUTRAL",  # 기본 아바타 상태
    }
# ⭐ 추가: 비디오 동기화 관련 세션 상태
if "current_customer_video" not in st.session_state:
    st.session_state.current_customer_video = None  # 현재 재생 중인 고객 비디오 경로
if "current_customer_video_bytes" not in st.session_state:
    st.session_state.current_customer_video_bytes = None  # 현재 재생 중인 고객 비디오 바이트
if "is_video_sync_enabled" not in st.session_state:
    st.session_state.is_video_sync_enabled = True  # 비디오 동기화 활성화 여부
if "video_male_neutral" not in st.session_state:
    st.session_state.video_male_neutral = None  # 남자 중립 비디오 경로
if "video_male_happy" not in st.session_state:
    st.session_state.video_male_happy = None
if "video_male_angry" not in st.session_state:
    st.session_state.video_male_angry = None
if "video_male_asking" not in st.session_state:
    st.session_state.video_male_asking = None
if "video_male_sad" not in st.session_state:
    st.session_state.video_male_sad = None
if "video_female_neutral" not in st.session_state:
    st.session_state.video_female_neutral = None  # 여자 중립 비디오 경로
if "video_female_happy" not in st.session_state:
    st.session_state.video_female_happy = None
if "video_female_angry" not in st.session_state:
    st.session_state.video_female_angry = None
if "video_female_asking" not in st.session_state:
    st.session_state.video_female_asking = None
if "video_female_sad" not in st.session_state:
    st.session_state.video_female_sad = None
# ⭐ 추가: 전사할 오디오 바이트 임시 저장소
if "bytes_to_process" not in st.session_state:
    st.session_state.bytes_to_process = None

# 언어 키 안전하게 가져오기
current_lang = st.session_state.get("language", "ko")
if current_lang not in ["ko", "en", "ja"]:
    current_lang = "ko"
L = LANG.get(current_lang, LANG["ko"])

# ⭐ 2-A. Gemini 키 초기화 (잘못된 키 잔존 방지)
if "user_gemini_key" in st.session_state and st.session_state["user_gemini_key"].startswith("AIza"):
    pass

# ========================================
# 0. 멀티 모델 API Key 안전 구조 (Secrets + Env Var만 사용)
# ========================================

# 1) 지원하는 API 목록 정의
SUPPORTED_APIS = {
    "openai": {
        "label": "OpenAI API Key",
        "secret_key": "OPENAI_API_KEY",
        "session_key": "user_openai_key",
        "placeholder": "sk-proj-**************************",
    },
    "gemini": {
        "label": "Google Gemini API Key",
        "secret_key": "GEMINI_API_KEY",
        "session_key": "user_gemini_key",
        "placeholder": "AIza***********************************",
    },
    "hyperclova": {
        "label": "Hyperclova API Key",
        "secret_key": "HYPERCLOVA_API_KEY",
        "session_key": "user_hyperclova_key",
        "placeholder": "hyperclova-**************************",
    },
    "nvidia": {
        "label": "NVIDIA NIM API Key",
        "secret_key": "NVIDIA_API_KEY",
        "session_key": "user_nvidia_key",
        "placeholder": "nvapi-**************************",
    },
    "claude": {
        "label": "Anthropic Claude API Key",
        "secret_key": "CLAUDE_API_KEY",
        "session_key": "user_claude_key",
        "placeholder": "sk-ant-api-**************************",
    },
    "groq": {
        "label": "Groq API Key",
        "secret_key": "GROQ_API_KEY",
        "session_key": "user_groq_key",
        "placeholder": "gsk_**************************",
    },
    "hyperclova": {
        "label": "Hyperclova API Key",
        "secret_key": "HYPERCLOVA_API_KEY",
        "session_key": "user_hyperclova_key",
        "placeholder": "hyperclova-**************************",
    },
}

# 2) 세션 초기화
for api, cfg in SUPPORTED_APIS.items():
    if cfg["session_key"] not in st.session_state:
        st.session_state[cfg["session_key"]] = ""

if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "openai_gpt4"


def get_api_key(api):
    cfg = SUPPORTED_APIS[api]

    # ⭐ 1. Streamlit Secrets (.streamlit/secrets.toml) - 최우선
    try:
        if hasattr(st, "secrets") and cfg["secret_key"] in st.secrets:
            return st.secrets[cfg["secret_key"]]
    except Exception:
        pass

    # 2. Environment Variable (os.environ)
    env_key = os.environ.get(cfg["secret_key"])
    if env_key:
        return env_key

    # 3. User Input (Session State - 제거됨)
    user_key = st.session_state.get(cfg["session_key"], "")
    if user_key:
        return user_key

    return ""


# ========================================
# 1. Sidebar UI: API Key 입력 제거
# ========================================
# API Key 입력 UI는 제거하고, 환경변수와 Streamlit Secrets만 사용하도록 함.


# ========================================
# 2. LLM 클라이언트 라우팅 & 실행
# ========================================
def get_llm_client():
    """선택된 모델에 맞는 클라이언트 + 모델코드 반환"""
    model_key = st.session_state.get("selected_llm", "openai_gpt4")

    # --- OpenAI ---
    if model_key.startswith("openai"):
        key = get_api_key("openai")
        if not key: return None, None
        try:
            client = OpenAI(api_key=key)
            model_name = "gpt-4o" if model_key == "openai_gpt4" else "gpt-3.5-turbo"
            return client, ("openai", model_name)
        except Exception:
            return None, None

    # --- Gemini ---
    if model_key.startswith("gemini"):
        key = get_api_key("gemini")
        if not key: return None, None
        try:
            genai.configure(api_key=key)
            model_name = "gemini-2.5-pro" if model_key == "gemini_pro" else "gemini-2.5-flash"
            return genai, ("gemini", model_name)
        except Exception:
            return None, None

    # --- Claude ---
    if model_key.startswith("claude"):
        key = get_api_key("claude")
        if not key: return None, None
        try:
            client = Anthropic(api_key=key)
            model_name = "claude-3-5-sonnet-latest"
            return client, ("claude", model_name)
        except Exception:
            return None, None

    # --- Groq ---
    if model_key.startswith("groq"):
        from groq import Groq
        key = get_api_key("groq")
        if not key: return None, None
        try:
            client = Groq(api_key=key)
            model_name = (
                "llama3-70b-8192"
                if "llama3" in model_key
                else "mixtral-8x7b-32768"
            )
            return client, ("groq", model_name)
        except Exception:
            return None, None

    return None, None


def run_llm(prompt: str) -> str:
    """선택된 LLM으로 프롬프트 실행 (Gemini 우선순위 변경 적용)"""
    client, info = get_llm_client()

    # Note: info는 사이드바에서 선택된 주력 모델의 정보를 담고 있습니다.
    provider, model_name = info if info else (None, None)

    # Fallback 순서를 정의합니다. (Gemini 우선)
    llm_attempts = []

    # 1. Gemini를 최우선 Fallback으로 시도 (Keys 확인)
    gemini_key = get_api_key("gemini")
    if gemini_key:
        llm_attempts.append(("gemini", gemini_key, "gemini-2.5-pro" if "pro" in model_name else "gemini-2.5-flash"))

    # 2. OpenAI를 2순위 Fallback으로 시도 (Keys 확인)
    openai_key = get_api_key("openai")
    if openai_key:
        llm_attempts.append(("openai", openai_key, "gpt-4o" if "4" in model_name else "gpt-3.5-turbo"))

    # 3. Claude를 3순위 Fallback으로 시도 (Keys 확인)
    claude_key = get_api_key("claude")
    if claude_key:
        llm_attempts.append(("claude", claude_key, "claude-3-5-sonnet-latest"))

    # 4. Groq를 4순위 Fallback으로 시도 (Keys 확인)
    groq_key = get_api_key("groq")
    if groq_key:
        groq_model = "llama3-70b-8192" if "llama3" in model_name else "mixtral-8x7b-32768"
        llm_attempts.append(("groq", groq_key, groq_model))

    # ⭐ 순서 조정: 주력 모델(사용자가 사이드바에서 선택한 모델)을 가장 먼저 시도합니다.
    # 만약 주력 모델이 Fallback 리스트에 포함되어 있다면, 그 모델을 첫 순서로 올립니다.
    if provider and provider in [attempt[0] for attempt in llm_attempts]:
        # 주력 모델을 리스트에서 찾아 제거
        primary_attempt = next((attempt for attempt in llm_attempts if attempt[0] == provider), None)
        if primary_attempt:
            llm_attempts.remove(primary_attempt)
            # 주력 모델이 Gemini나 OpenAI가 아니라면, Fallback 순서와 관계없이 가장 먼저 시도하도록 삽입
            llm_attempts.insert(0, primary_attempt)

    # LLM 순차 실행
    for provider, key, model in llm_attempts:
        if not key: continue

        try:
            if provider == "gemini":
                genai.configure(api_key=key)
                gen_model = genai.GenerativeModel(model)
                resp = gen_model.generate_content(prompt)
                return resp.text

            elif provider == "openai":
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

            elif provider == "claude":
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text

            elif provider == "groq":
                from groq import Groq
                g_client = Groq(api_key=key)
                resp = g_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

        except Exception as e:
            # 해당 API가 실패하면 다음 API로 넘어갑니다.
            print(f"LLM {provider} ({model}) failed: {e}")
            continue

    # 모든 시도가 실패했을 때
    return "❌ 모든 LLM API 키가 작동하지 않거나 할당량이 소진되었습니다."


# ========================================
# 2-A. Whisper / TTS 용 OpenAI Client 별도로 초기화
# ========================================

def init_openai_audio_client():
    key = get_api_key("openai")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except:
        return None


# ⭐ 최적화: LLM 클라이언트 초기화 캐싱 (매번 재생성하지 않도록)
# OpenAI 클라이언트 캐싱
# ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
if "openai_client" not in st.session_state or st.session_state.openai_client is None:
    try:
        st.session_state.openai_client = init_openai_audio_client()
    except Exception as e:
        st.session_state.openai_client = None
        print(f"OpenAI 클라이언트 초기화 중 오류 (무시됨): {e}")

# LLM 준비 상태 캐싱 (API 키 변경 시에만 재확인)
# ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
if "is_llm_ready" not in st.session_state or "llm_ready_checked" not in st.session_state:
    try:
        probe_client, _ = get_llm_client()
        st.session_state.is_llm_ready = probe_client is not None
    except Exception as e:
        # 초기화 실패 시에도 앱이 계속 실행되도록 False로 설정
        st.session_state.is_llm_ready = False
        print(f"LLM 초기화 중 오류 (무시됨): {e}")
    st.session_state.llm_ready_checked = True

# API 키 변경 감지를 위한 해시 체크
current_api_keys_hash = hashlib.md5(
    f"{get_api_key('openai')}{get_api_key('gemini')}{get_api_key('claude')}{get_api_key('groq')}".encode()
).hexdigest()

if "api_keys_hash" not in st.session_state:
    st.session_state.api_keys_hash = current_api_keys_hash
elif st.session_state.api_keys_hash != current_api_keys_hash:
    # API 키가 변경된 경우만 재확인
    # ⭐ 수정: 초기화 시 블로킹 방지를 위해 try-except 추가
    try:
        probe_client, _ = get_llm_client()
        st.session_state.is_llm_ready = probe_client is not None
    except Exception as e:
        st.session_state.is_llm_ready = False
        print(f"LLM 재초기화 중 오류 (무시됨): {e}")
    st.session_state.api_keys_hash = current_api_keys_hash
    # OpenAI 클라이언트도 재초기화
    try:
        st.session_state.openai_client = init_openai_audio_client()
    except Exception as e:
        st.session_state.openai_client = None
        print(f"OpenAI 클라이언트 재초기화 중 오류 (무시됨): {e}")

if st.session_state.openai_client:
    # 키를 찾았고 클라이언트 객체는 생성되었으나, 실제 인증은 API 호출 시 이루어짐 (401 오류는 여기서 발생)
    st.session_state.openai_init_msg = "✅ OpenAI TTS/Whisper 클라이언트 준비 완료 (Key 확인됨)"
else:
    # 키를 찾지 못한 경우
    st.session_state.openai_init_msg = L["openai_missing"]

if not st.session_state.is_llm_ready:
    st.session_state.llm_init_error_msg = L["simulation_no_key_warning"]
else:
    st.session_state.llm_init_error_msg = ""


# ----------------------------------------
# LLM 번역 함수 (Gemini 클라이언트 의존성 제거 및 강화)
# ----------------------------------------
def translate_text_with_llm(text_content: str, target_lang_code: str, source_lang_code: str) -> Tuple[str, bool]:
    """
    주어진 텍스트를 LLM을 사용하여 대상 언어로 번역합니다. (안정화된 텍스트 출력)
    **수정 사항:** LLM Fallback 순서를 OpenAI 우선으로 조정하고, 응답이 비어있을 경우 원본 텍스트를 반환
    
    Returns:
        tuple: (translated_text, is_success) - 번역된 텍스트와 성공 여부
    """
    target_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(target_lang_code, "English")
    source_lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang_code, "English")

    # 순수한 텍스트 번역 결과만 출력하도록 강제
    system_prompt = (
        f"You are a professional translation AI. Translate the entire following customer support chat history "
        f"from '{source_lang_name}' to '{target_lang_name}'. "
        f"You MUST translate the content to {target_lang_name} ONLY. "
        f"Do not include any mixed languages, the source text, or any introductory/concluding remarks. "
        f"Output ONLY the translated chat history text. "
    )
    prompt = f"Original Chat History:\n\n{text_content}"

    # LLM Fallback 순서: OpenAI -> Gemini -> Claude (OpenAI를 최우선으로 조정)
    llm_attempts = [
        ("openai", get_api_key("openai"), "gpt-4o"),  # 1순위: OpenAI (가장 안정적)
        ("gemini", get_api_key("gemini"), "gemini-2.5-flash"),  # 2순위
        ("claude", get_api_key("claude"), "claude-3-5-sonnet-latest"),  # 3순위
    ]

    last_error = ""

    for provider, key, model_name in llm_attempts:
        if not key: continue

        try:
            translated_text = ""

            if provider == "openai":
                o_client = OpenAI(api_key=key)
                resp = o_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.1
                )
                translated_text = resp.choices[0].message.content.strip()

            elif provider == "gemini":
                genai.configure(api_key=key)
                g_model = genai.GenerativeModel(model_name)
                resp = g_model.generate_content(
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
                )
                translated_text = resp.text.strip()

            elif provider == "claude":
                from anthropic import Anthropic
                c_client = Anthropic(api_key=key)
                resp = c_client.messages.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    system=system_prompt
                )
                translated_text = resp.content[0].text.strip()

            # 번역 결과가 유효한지 확인
            if translated_text and len(translated_text.strip()) > 0:
                return translated_text, True  # 번역 성공
            else:
                last_error = f"Translation failed: {provider} returned empty response."
                continue  # 다음 LLM 시도

        except Exception as e:
            last_error = f"Translation API call failed with {provider} ({model_name}): {e}"  # 모델명 추가
            print(last_error)
            continue  # 다음 LLM 시도

    # 모든 시도가 실패했을 때, 원본 텍스트를 반환하여 프로세스가 계속 진행되도록 함
    # (오류 메시지 대신 원본 텍스트를 반환하여 번역 실패해도 다음 단계로 진행 가능)
    print(f"Translation failed: {last_error or 'No active API key found.'}. Returning original text.")
    return text_content, False  # 원본 텍스트 반환, 번역 실패 표시


# ----------------------------------------
# Realtime Hint Generation (요청 2 반영)
# ----------------------------------------
def generate_realtime_hint(current_lang_key: str, is_call: bool = False):
    """현재 대화 맥락을 기반으로 에이전트에게 실시간 응대 힌트(키워드/정책/액션)를 제공"""
    # 언어 키 검증 및 기본값 처리
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    # 채팅/전화 구분하여 이력 사용
    if is_call:
        # 전화 시뮬레이터에서는 현재 CC 영역에 표시된 텍스트와 초기 문의를 함께 사용
        website_url = st.session_state.get("call_website_url", "").strip()
        website_context = f"\nWebsite URL: {website_url}" if website_url else ""
        history_text = (
            f"Initial Query: {st.session_state.call_initial_query}\n"
            f"Previous Customer Utterance: {st.session_state.current_customer_audio_text}\n"
            f"Previous Agent Utterance: {st.session_state.current_agent_audio_text}{website_context}"
        )
    else:
        history_text = get_chat_history_for_prompt(include_attachment=True)

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    hint_prompt = f"""
You are an AI Supervisor providing an **urgent, internal hint** to a human agent whose AHT is being monitored.
Analyze the conversation history, especially the customer's last message, which might be about complex issues like JR Pass, Universal Studio Japan (USJ), or a complex refund policy.

Provide ONE concise, actionable hint for the agent. The purpose is to save AHT time.

Output MUST be a single paragraph/sentence in {lang_name} containing actionable advice.
DO NOT use markdown headers or titles.
Do NOT direct the agent to check the general website.
Provide an actionable fact or the next specific step (e.g., check policy section, confirm coverage).

Examples of good hints (based on the content):
- Check the official JR Pass site for current exchange rates.
- The 'Universal Express Pass' is non-refundable; clearly cite policy section 3.2.
- Ask for the order confirmation number before proceeding with any action.
- The solution lies in the section of the Klook site titled '~'.

Conversation History:
{history_text}

HINT:
"""
    if not st.session_state.is_llm_ready:
        return "(Mock Hint: LLM Key is missing. Ask the customer for the booking number.)"

    with st.spinner(f"💡 {L['button_request_hint']}..."):
        try:
            return run_llm(hint_prompt).strip()
        except Exception as e:
            return f"❌ Hint Generation Error. (Try again or check API Key: {e})"


def generate_agent_response_draft(current_lang_key: str) -> str:
    """고객 응답을 기반으로 AI가 에이전트 응답 초안을 생성 (요청 1 반영)"""
    # 언어 키 검증 및 기본값 처리
    if not current_lang_key or current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])
    history_text = get_chat_history_for_prompt(include_attachment=True)
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 고객의 최신 문의 내용 추출 및 분석 (강화)
    latest_customer_message = ""
    initial_customer_query = st.session_state.get('customer_query_text_area', '')
    customer_query_analysis = ""
    
    # 모든 고객 메시지 수집
    all_customer_messages = []
    if st.session_state.simulator_messages:
        all_customer_messages = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]]
    
    # 초기 문의도 포함
    if initial_customer_query and initial_customer_query not in all_customer_messages:
        all_customer_messages.insert(0, initial_customer_query)
    
    if all_customer_messages:
        latest_customer_message = all_customer_messages[-1]
        
        # 문의 내용에서 핵심 정보 추출 (간단한 키워드 추출)
        inquiry_keywords = []
        inquiry_text = " ".join(all_customer_messages).lower()
        
        # 일반적인 문의 키워드 패턴
        important_patterns = [
            r'\b\d{4,}\b',  # 주문번호, 전화번호 등 숫자
            r'\b(주문|order|注文)\b',
            r'\b(환불|refund|返金)\b',
            r'\b(취소|cancel|キャンセル)\b',
            r'\b(배송|delivery|配送)\b',
            r'\b(변경|change|変更)\b',
            r'\b(문제|problem|issue|問題)\b',
            r'\b(도움|help|助け)\b',
        ]
        
        # 핵심 문의 내용 요약
        inquiry_summary = f"""
**CUSTOMER INQUIRY DETAILS:**

Initial Query: "{initial_customer_query if initial_customer_query else 'Not provided'}"

Latest Customer Message: "{latest_customer_message}"

All Customer Messages Context:
{chr(10).join([f"- {msg[:150]}..." if len(msg) > 150 else f"- {msg}" for msg in all_customer_messages[-3:]])}

**YOUR RESPONSE MUST DIRECTLY ADDRESS:**

1. **SPECIFIC ISSUE IDENTIFICATION**: 
   - What EXACT problem or question did the customer mention?
   - Extract and reference specific details: order numbers, dates, product names, locations, error messages, etc.
   - If multiple issues were mentioned, address EACH one specifically

2. **CONCRETE SOLUTION PROVIDED**:
   - Provide STEP-BY-STEP instructions tailored to their EXACT situation
   - Include specific actions they need to take (e.g., "Go to Settings > Account > Order History and click on order #12345")
   - Reference the exact products/services they mentioned
   - If they mentioned a location, reference it in your solution

3. **PERSONALIZATION**:
   - Use the customer's specific words/phrases when appropriate
   - Reference their exact situation (e.g., "Since you mentioned your eSIM isn't activating in Paris...")
   - Acknowledge their specific concern or frustration point

4. **COMPLETENESS**:
   - Answer ALL questions they asked
   - Address ALL problems they mentioned
   - If they asked "why", explain the specific reason for their situation
   - If they asked "how", provide detailed steps for their exact case

**CRITICAL REQUIREMENTS:**
- DO NOT use generic templates like "Thank you for contacting us" without addressing their specific issue
- DO NOT give vague answers like "Please check your settings" - be SPECIFIC about which settings and what to do
- DO NOT ignore specific details they mentioned (order numbers, dates, locations, etc.)
- Your response must read as if it was written SPECIFICALLY for this customer's exact inquiry
- If the customer mentioned "eSIM activation in Paris", your response MUST specifically address eSIM activation and Paris
- If the customer mentioned an order number, your response MUST reference that order number

**EXAMPLE OF GOOD RESPONSE:**
Bad: "Thank you for contacting us. We understand your concern and will help you resolve this issue."
Good: "I understand you're having trouble activating your eSIM in Paris. Let me help you resolve this step by step. First, please check if your phone's APN settings are configured correctly for the Paris network..."

**NOW GENERATE YOUR RESPONSE** following these requirements:
"""
        
        customer_query_analysis = inquiry_summary

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"\n[고객 첨부 파일 정보: {attachment_context}]\n"
    else:
        attachment_context = ""

    # 고객 유형 및 반복 불만 패턴 분석
    customer_type = st.session_state.get('customer_type_sim_select', '일반적인 문의')
    is_difficult_customer = customer_type in ["까다로운 고객", "매우 불만족스러운 고객", "Difficult Customer",
                                              "Highly Dissatisfied Customer", "難しい顧客", "非常に不満な顧客"]

    # 고객 메시지 수 및 감정 분석
    customer_message_count = sum(
        1 for msg in st.session_state.simulator_messages if msg.get("role") in ["customer", "customer_rebuttal"])
    agent_message_count = sum(1 for msg in st.session_state.simulator_messages if msg.get("role") == "agent_response")

    # 이전 에이전트 응답들 추출 (다양성 확보를 위해)
    previous_agent_responses = [msg["content"] for msg in st.session_state.simulator_messages 
                                if msg.get("role") == "agent_response"]
    previous_responses_context = ""
    if previous_agent_responses:
        previous_responses_context = f"\n[이전 에이전트 응답들 (참고용, 동일하게 반복하지 말 것):\n"
        for i, prev_resp in enumerate(previous_agent_responses[-3:], 1):  # 최근 3개만
            prev_resp_preview = prev_resp[:200] + "..." if len(prev_resp) > 200 else prev_resp
            previous_responses_context += f"{i}. {prev_resp_preview}\n"
        previous_responses_context += "]\n"

    # 고객이 계속 따지거나 화내는 패턴 감지 (고객 메시지가 에이전트 메시지보다 많거나, 반복적인 불만 표현)
    is_repeating_complaints = False
    if customer_message_count > agent_message_count and customer_message_count >= 2:
        # 마지막 2개 고객 메시지 분석
        recent_customer_messages = [msg["content"].lower() for msg in st.session_state.simulator_messages if
                                    msg.get("role") in ["customer", "customer_rebuttal"]][-2:]
        complaint_keywords = ["왜", "이유", "설명", "말이 안", "이해가 안", "화나", "짜증", "불만", "왜", "why", "reason", "explain",
                              "angry", "frustrated", "complaint", "なぜ", "理由", "説明", "怒り", "不満"]
        if any(any(keyword in msg for keyword in complaint_keywords) for msg in recent_customer_messages):
            is_repeating_complaints = True

    # 대처법 포메이션 추가 여부 결정
    needs_coping_strategy = is_difficult_customer or (is_repeating_complaints and customer_message_count >= 2)

    # 대처법 가이드라인 생성
    coping_guidance = ""
    if needs_coping_strategy:
        coping_guidance = f"""

[CRITICAL: Handling Difficult Customer Situation]
The customer type is "{customer_type}" and the customer has sent {customer_message_count} messages while the agent has sent {agent_message_count} messages.
The customer may be showing signs of continued frustration or dissatisfaction.

**INCLUDE THE FOLLOWING COPING STRATEGY FORMAT IN YOUR RESPONSE:**

1. **Immediate Acknowledgment** (1-2 sentences):
   - Acknowledge their frustration/specific concern explicitly
   - Show deep empathy and understanding
   - Example formats:
     * "{'죄송합니다. 불편을 드려 정말 죄송합니다. 고객님의 상황을 충분히 이해하고 있습니다.' if current_lang_key == 'ko' else ('I sincerely apologize for the inconvenience. I fully understand your situation and frustration.' if current_lang_key == 'en' else '大変申し訳ございません。お客様の状況とご不便を十分に理解しております。')}"
     * "{'고객님의 소중한 의견을 잘 듣고 있습니다. 정말 답답하셨을 것 같습니다.' if current_lang_key == 'ko' else ('I hear your concerns clearly. This must have been very frustrating for you.' if current_lang_key == 'en' else 'お客様のご意見をしっかりと受け止めています。本当にお困りだったと思います。')}"

2. **Specific Solution Recap** (2-3 sentences):
   - Clearly restate the solution/step provided previously (if any)
   - Offer a NEW concrete action or alternative solution
   - Be specific and actionable
   - Example formats:
     * "{'앞서 안내드린 [구체적 해결책] 외에도, [새로운 대안/추가 조치]를 진행해드릴 수 있습니다.' if current_lang_key == 'ko' else ('In addition to the [specific solution] I mentioned earlier, I can also [new alternative/additional action] for you.' if current_lang_key == 'en' else '先ほどご案内した[具体的解決策]に加えて、[新しい代替案/追加措置]も進めることができます。')}"
     * "{'혹시 [구체적 문제점] 때문에 불편하셨다면, [구체적 해결 방법]을 바로 진행해드리겠습니다.' if current_lang_key == 'ko' else ('If you are experiencing [specific issue], I can immediately proceed with [specific solution].' if current_lang_key == 'en' else 'もし[具体的問題]でご不便でしたら、[具体的解決方法]をすぐに進めさせていただきます。')}"

3. **Escalation or Follow-up Offer** (1-2 sentences):
   - Offer to escalate to supervisor/higher level support
   - Promise immediate follow-up within specific time
   - Example formats:
     * "{'만약 여전히 불만이 해소되지 않으신다면, 즉시 상급 관리자에게 이관하여 더 나은 해결책을 찾아드리겠습니다.' if current_lang_key == 'ko' else ('If your concern is still not resolved, I can immediately escalate this to a supervisor to find a better solution.' if current_lang_key == 'en' else 'もしご不満が解消されない場合は、すぐに上級管理者にエスカレートして、より良い解決策を見つけさせていただきます。')}"
     * "{'24시간 이내에 [구체적 조치/결과]를 확인하여 고객님께 다시 연락드리겠습니다.' if current_lang_key == 'ko' else ('I will follow up with you within 24 hours regarding [specific action/result].' if current_lang_key == 'en' else '24時間以内に[具体的措置/結果]を確認し、お客様に再度ご連絡いたします。')}"

4. **Closing with Assurance** (1 sentence):
   - Reassure that their concern is being taken seriously
   - Example formats:
     * "{'고객님의 모든 문의사항을 최우선으로 처리하겠습니다.' if current_lang_key == 'ko' else ('I will prioritize resolving all of your concerns.' if current_lang_key == 'en' else 'お客様のすべてのご質問を最優先で処理いたします。')}"

**IMPORTANT NOTES:**
- DO NOT repeat the exact same solution that was already provided
- DO NOT sound dismissive or automated
- DO sound genuinely concerned and willing to go the extra mile
- If policy restrictions exist, acknowledge them but still offer alternatives
- Use warm, respectful tone while being firm about what can/cannot be done

**RESPONSE STRUCTURE:**
[Immediate Acknowledgment]
[Specific Solution Recap + New Action]
[Escalation/Follow-up Offer]
[Closing with Assurance]

Now generate the agent's response draft following this structure:
"""

    # 다양성 확보를 위한 추가 지시사항 (더 강화)
    diversity_instruction = ""
    if previous_agent_responses:
        # 이전 응답들의 주요 키워드/구문 추출 (반복 방지)
        previous_keywords = []
        for prev_resp in previous_agent_responses[-3:]:
            # 간단한 키워드 추출 (2-3단어 구문)
            words = prev_resp.split()[:20]  # 처음 20단어만
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:
                    previous_keywords.append(f"{words[i]} {words[i+1]}")
        
        keywords_warning = ""
        if previous_keywords:
            unique_keywords = list(set(previous_keywords))[:10]  # 최대 10개만
            keywords_warning = f"\n- AVOID using these exact phrases from previous responses: {', '.join(unique_keywords[:5])}"
        
        diversity_instruction = f"""
**CRITICAL DIVERSITY REQUIREMENT - STRICTLY ENFORCED:**
- You MUST generate a COMPLETELY DIFFERENT response from ALL previous agent responses shown above
- Use COMPLETELY DIFFERENT wording, phrasing, sentence structures, and vocabulary
- If similar solutions are needed, present them in a COMPLETELY DIFFERENT way or from a COMPLETELY DIFFERENT angle
- Vary your opening sentences, transition phrases, and closing statements - NO REPETITION
- DO NOT copy, paraphrase, or reuse ANY phrases from previous responses - be CREATIVE and UNIQUE while maintaining professionalism
- Consider COMPLETELY different approaches: 
  * If previous responses were formal, try a warmer, more personal tone (or vice versa)
  * If previous responses focused on one aspect, emphasize a COMPLETELY different aspect this time
  * Use different examples, analogies, or explanations
  * Change the order of information presentation
  * Use different sentence lengths and structures
{keywords_warning}
- IMPORTANT: Read ALL previous responses carefully and ensure your response is DISTINCTLY different in style, tone, structure, and content
- If you find yourself writing something similar to a previous response, STOP and rewrite it completely differently
"""

    # 랜덤 요소 추가를 위한 변형 지시사항
    variation_approaches = [
        "Start with a different greeting or acknowledgment style",
        "Use a different problem-solving approach or framework",
        "Present information in a different order",
        "Use different examples or analogies",
        "Vary the level of formality or warmth",
        "Focus on different aspects of the solution",
        "Use different transition words and phrases",
        "Change the length and complexity of sentences"
    ]
    selected_approaches = random.sample(variation_approaches, min(3, len(variation_approaches)))
    variation_note = "\n".join([f"- {approach}" for approach in selected_approaches])

    draft_prompt = f"""
You are an AI assistant helping a customer support agent write a professional, tailored response.

**PRIMARY OBJECTIVE:**
Generate a response draft that is SPECIFICALLY tailored to the customer's EXACT inquiry, providing concrete, actionable solutions that directly address their specific situation. The response must read as if it was written personally for this customer's unique case.

**CRITICAL REQUIREMENTS (IN ORDER OF PRIORITY):**
1. **MOST CRITICAL**: Address the customer's SPECIFIC inquiry/question with DETAILED, ACTIONABLE solutions
   - Extract and reference specific details from their message (order numbers, dates, product names, locations, error messages, etc.)
   - Provide step-by-step instructions tailored to their EXACT situation
   - Answer ALL questions they asked completely
   - Address ALL problems they mentioned specifically

2. The response MUST be in {lang_name}

3. Be professional, empathetic, and solution-oriented

4. If the customer asked a question, provide a COMPLETE and SPECIFIC answer - do NOT give vague or generic responses
   - Bad: "Please check your settings"
   - Good: "Please go to Settings > Mobile Network > APN Settings and ensure the APN is set to 'internet'"

5. If the customer mentioned a problem, acknowledge it SPECIFICALLY and provide STEP-BY-STEP solutions
   - Reference their exact problem description
   - Provide solutions that directly address their specific issue

6. Reference specific details from their inquiry (order numbers, dates, products, locations, etc.) if mentioned
   - If they mentioned "order #12345", your response MUST include "order #12345"
   - If they mentioned "Paris", your response should reference Paris specifically
   - If they mentioned "eSIM", address eSIM specifically, not just "SIM card"

7. Keep the tone appropriate for the customer type: {customer_type}

8. Do NOT include any markdown formatting, just plain text

9. {f'**FOLLOW THE COPING STRATEGY FORMAT BELOW**' if needs_coping_strategy else 'Use natural, conversational flow'}

10. **CRITICAL**: Generate a COMPLETELY UNIQUE and VARIED response - avoid repeating ANY similar phrases, structures, or approaches from previous responses

11. **CRITICAL**: Your response must be HIGHLY RELEVANT to the customer's specific inquiry - generic template responses are NOT acceptable
    - DO NOT start with generic greetings without immediately addressing their specific issue
    - DO NOT use placeholder text like "[specific solution]" - provide ACTUAL specific solutions
    - Your response should make the customer feel like you read and understood THEIR specific message

**VARIATION TECHNIQUES TO APPLY:**
{variation_note}

{customer_query_analysis}

**FULL CONVERSATION HISTORY:**
{history_text}
{attachment_context}

**PREVIOUS RESPONSES TO AVOID REPEATING:**
{previous_responses_context if previous_responses_context else "No previous responses to compare against."}

**DIVERSITY REQUIREMENTS:**
{diversity_instruction if diversity_instruction else "This is the first response, so no previous responses to avoid."}

{coping_guidance if needs_coping_strategy else ''}

**YOUR TASK:**
Generate the agent's response draft NOW. The response must:

1. **FIRST**: Read the customer inquiry analysis above CAREFULLY and identify:
   - What is their EXACT problem or question?
   - What specific details did they mention (order numbers, dates, locations, products)?
   - What do they need help with specifically?

2. **SECOND**: Write a response that:
   - Addresses their EXACT problem/question (not a generic version)
   - References the specific details they mentioned
   - Provides concrete, actionable steps tailored to their situation
   - Answers ALL their questions completely
   - Makes them feel understood

3. **THIRD**: Ensure the response is:
   - COMPLETELY DIFFERENT and UNIQUE from any previous responses
   - Professional, empathetic, and solution-oriented
   - Written in {lang_name}
   - Free of markdown formatting

**BEFORE YOU WRITE**: Ask yourself:
- "Does this response address the customer's SPECIFIC inquiry?"
- "Would a generic template response work here?" (If yes, rewrite it to be more specific)
- "Does this response reference specific details from the customer's message?"
- "Would the customer feel like I read and understood THEIR specific message?"

**NOW GENERATE THE RESPONSE:**
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        draft = run_llm(draft_prompt).strip()
        # 마크다운 제거 (``` 등)
        if draft.startswith("```"):
            lines = draft.split("\n")
            draft = "\n".join(lines[1:-1]) if len(lines) > 2 else draft
        return draft
    except Exception as e:
        return f"❌ 응답 초안 생성 오류: {e}"


# ⭐ 새로운 함수: 전화 발신 시뮬레이션 요약 생성
def generate_outbound_call_summary(customer_query: str, current_lang_key: str, target: str) -> str:
    """
    Simulates an outbound call to a local partner or customer and generates a summary of the outcome.
    """
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # Get the current chat history for context
    history_text = get_chat_history_for_prompt(include_attachment=True)
    if not history_text:
        history_text = f"Initial Customer Query: {customer_query}"

    # Policy context (from supervisor) should be included to guide the outcome
    policy_context = st.session_state.supervisor_policy_context or ""

    summary_prompt = f"""
You are an AI simulating a quick, high-stakes phone call placed by the customer support agent to a '{target}' (either a local partner/vendor or the customer).

The purpose of the call is to resolve a complex, policy-restricted issue (like an exceptional refund for a non-refundable item, or urgent confirmation of an airport transfer change).

Analyze the conversation history, the initial query, and any provided supervisor policy.
Generate a concise summary of the OUTCOME of this simulated phone call.
The summary MUST be professional and strictly in {lang_name}.

[CRITICAL RULE]: For non-refundable items (e.g., Universal Studio Express Pass, non-refundable hotel/transfer), the local partner should only grant an exception IF the customer has provided strong, unavoidable proof (like a flight cancellation notice, doctor's note, or natural disaster notice). If no such proof is evident in the chat history, the outcome should usually be a denial or a request for more proof, but keep the tone professional.
If the customer's query is about Airport Transfer change, the outcome should be: 'Confirmation complete. Change is approved/denied based on partner policy.'

Conversation History:
{history_text}

Supervisor Policy Context (If any):
{policy_context}

Target of Call: {target}

Generate the phone call summary (Outcome ONLY):
"""
    if not st.session_state.is_llm_ready:
        return f"❌ LLM Key missing. (Simulated Outcome: The {target} requested the agent to send proof via email.)"

    try:
        summary = run_llm(summary_prompt).strip()
        # 마크다운 제거 (``` 등)
        if summary.startswith("```"):
            lines = summary.split("\n")
            summary = "\n".join(lines[1:-1]) if len(lines) > 2 else summary
        return summary
    except Exception as e:
        return f"❌ Phone call simulation error: {e}"


# ========================================
# 3. Whisper / TTS Helper
# ========================================

def transcribe_bytes_with_whisper(audio_bytes: bytes, mime_type: str = "audio/webm", lang_code: str = None, auto_detect: bool = True) -> str:
    """
    OpenAI Whisper API 또는 Gemini API를 사용하여 오디오 바이트를 텍스트로 전사합니다.
    OpenAI가 실패하면 Gemini로 자동 fallback합니다.
    
    Args:
        audio_bytes: 전사할 오디오 바이트
        mime_type: 오디오 MIME 타입
        lang_code: 언어 코드 (ko, en, ja 등). None이거나 auto_detect=True이면 자동 감지
        auto_detect: True이면 언어를 자동 감지 (lang_code 무시)
    """
    # 언어 키 안전하게 가져오기
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 임시 파일 저장 (API 호환성)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    
    # 1️⃣ OpenAI Whisper API 시도
    client = st.session_state.openai_client
    if client is not None:
        try:
            with open(tmp.name, "rb") as f:
                # 언어 자동 감지 또는 지정된 언어 사용
                if auto_detect or lang_code is None:
                    # language 파라미터를 생략하면 Whisper가 자동으로 언어를 감지합니다
                    res = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="text",
                    )
                else:
                    whisper_lang = {"ko": "ko", "en": "en", "ja": "ja"}.get(lang_code, "en")
                    res = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="text",
                        language=whisper_lang,
                    )
            # res.text 속성이 있는지 확인하고 없으면 res 자체를 문자열로 변환
            result = res.text.strip() if hasattr(res, 'text') else str(res).strip()
            if result:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
                return result
        except Exception as e:
            # OpenAI 실패 시 로그만 남기고 Gemini로 fallback
            print(f"OpenAI Whisper failed: {e}")
    
    # 2️⃣ Gemini API fallback
    gemini_key = get_api_key("gemini")
    if gemini_key:
        try:
            import base64
            genai.configure(api_key=gemini_key)
            
            # Gemini는 오디오 파일을 base64로 인코딩하여 전송
            with open(tmp.name, "rb") as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Gemini 2.0 Flash 모델 사용 (오디오 지원)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            # 프롬프트 구성
            lang_prompt = ""
            if lang_code:
                lang_map = {"ko": "한국어", "en": "English", "ja": "日本語"}
                lang_prompt = f"이 오디오는 {lang_map.get(lang_code, 'English')}로 말하고 있습니다. "
            
            prompt = f"{lang_prompt}이 오디오를 텍스트로 전사해주세요. 오직 전사된 텍스트만 반환하세요."
            
            # Gemini는 파일 업로드 방식 사용 (Gemini 2.0 Flash는 오디오 지원)
            try:
                audio_file = genai.upload_file(path=tmp.name, mime_type=mime_type)
                
                # 파일 업로드 후 잠시 대기 (업로드 완료 대기)
                import time
                time.sleep(1)
                
                response = model.generate_content([prompt, audio_file])
                result = response.text.strip() if response.text else ""
                
                # 파일 삭제
                try:
                    genai.delete_file(audio_file.name)
                except Exception as del_err:
                    print(f"Failed to delete Gemini file: {del_err}")
            except Exception as upload_err:
                # 파일 업로드 실패 시 다른 방법 시도
                print(f"Gemini file upload failed: {upload_err}")
                # 대안: base64 인코딩된 오디오를 직접 전송 (모델이 지원하는 경우)
                raise upload_err
            
            if result:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
                return result
            else:
                raise Exception("Gemini returned empty result")
        except Exception as e:
            print(f"Gemini transcription failed: {e}")
            # Gemini도 실패한 경우 에러 메시지 반환
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            return f"❌ {L.get('whisper_client_error', '전사 실패')}: OpenAI와 Gemini 모두 실패했습니다. ({str(e)[:100]})"
    else:
        # 두 API 모두 사용 불가
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        return f"❌ {L.get('openai_missing', 'OpenAI API Key가 필요합니다.')} 또는 Gemini API Key가 필요합니다."


def transcribe_audio(audio_bytes, filename="audio.wav"):
    client = st.session_state.openai_client

    # 1️⃣ OpenAI Whisper 시도
    if client:
        try:
            import io
            bio = io.BytesIO(audio_bytes)
            bio.name = filename
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
            )
            return resp.text
        except Exception as e:
            print("Whisper OpenAI failed:", e)

    # 2️⃣ Gemini STT fallback
    try:
        genai.configure(api_key=get_api_key("gemini"))
        model = genai.GenerativeModel("gemini-2.5-flash")
        text = model.generate_content("Transcribe this audio:").text
        return text or ""
    except Exception as e:
        print("Gemini STT failed:", e)

    return "❌ STT not available"


# ========================================
# 비디오 동기화 관련 함수
# ========================================

def analyze_text_for_video_selection(text: str, current_lang_key: str, 
                                     agent_last_response: str = None,
                                     conversation_context: List[Dict] = None) -> Dict[str, Any]:
    """
    LLM을 사용하여 텍스트를 분석하고 적절한 감정 상태와 제스처를 판단합니다.
    OpenAI/Gemini API를 활용한 영상 RAG의 핵심 기능입니다.
    
    ⭐ Gemini 제안 적용: 긴급도, 만족도 변화, 에이전트 답변 기반 예측 추가
    
    Args:
        text: 분석할 텍스트 (고객의 질문/응답)
        current_lang_key: 현재 언어 키
        agent_last_response: 에이전트의 마지막 답변 (선택적, 예측 정확도 향상)
        conversation_context: 대화 컨텍스트 (선택적, 만족도 변화 분석용)
    
    Returns:
        {
            "emotion": "NEUTRAL" | "HAPPY" | "ANGRY" | "ASKING" | "SAD",
            "gesture": "NONE" | "HAND_WAVE" | "NOD" | "SHAKE_HEAD" | "POINT",
            "urgency": "LOW" | "MEDIUM" | "HIGH",  # ⭐ 추가: 긴급도
            "satisfaction_delta": -1.0 to 1.0,  # ⭐ 추가: 만족도 변화 (-1: 감소, 0: 유지, 1: 증가)
            "confidence": 0.0-1.0
        }
    """
    if not text or not text.strip():
        return {
            "emotion": "NEUTRAL", 
            "gesture": "NONE", 
            "urgency": "LOW",
            "satisfaction_delta": 0.0,
            "confidence": 0.5
        }
    
    L = LANG.get(current_lang_key, LANG["ko"])
    
    # ⭐ Gemini 제안: 에이전트 답변 기반 예측 컨텍스트 구성
    context_info = ""
    if agent_last_response:
        context_info = f"""
에이전트의 마지막 답변: "{agent_last_response}"

에이전트의 답변을 고려했을 때, 고객이 지금 말하는 내용은 어떤 감정을 수반할 것인지 예측하세요.
예를 들어:
- 에이전트가 솔루션을 제시했다면 → 고객은 HAPPY 또는 ASKING (추가 질문)
- 에이전트가 거절했다면 → 고객은 ANGRY 또는 SAD
- 에이전트가 질문을 했다면 → 고객은 ASKING (답변) 또는 NEUTRAL
"""
    
    # ⭐ Gemini 제안: 만족도 변화 분석 컨텍스트
    satisfaction_context = ""
    if conversation_context and len(conversation_context) > 1:
        # 최근 대화의 감정 변화 추적
        recent_emotions = []
        for msg in conversation_context[-3:]:  # 최근 3개 메시지
            if msg.get("role") == "customer_rebuttal" or msg.get("role") == "customer":
                recent_emotions.append(msg.get("content", ""))
        
        if len(recent_emotions) >= 2:
            satisfaction_context = f"""
최근 대화 흐름:
- 이전 고객 메시지: "{recent_emotions[-2] if len(recent_emotions) >= 2 else ''}"
- 현재 고객 메시지: "{recent_emotions[-1]}"

만족도 변화를 분석하세요:
- 이전보다 더 긍정적이면 satisfaction_delta > 0
- 이전보다 더 부정적이면 satisfaction_delta < 0
- 비슷하면 satisfaction_delta ≈ 0
"""
    
    # ⭐ Gemini 제안: 개선된 LLM 프롬프트 구성
    prompt = f"""다음 고객의 텍스트를 분석하여 적절한 감정 상태, 제스처, 긴급도, 만족도 변화를 판단하세요.

고객 텍스트: "{text}"
{context_info}
{satisfaction_context}

다음 JSON 형식으로만 응답하세요 (다른 설명 없이):
{{
    "emotion": "NEUTRAL" | "HAPPY" | "ANGRY" | "ASKING" | "SAD",
    "gesture": "NONE" | "HAND_WAVE" | "NOD" | "SHAKE_HEAD" | "POINT",
    "urgency": "LOW" | "MEDIUM" | "HIGH",
    "satisfaction_delta": -1.0 to 1.0,
    "confidence": 0.0-1.0
}}

감정 판단 기준 (세분화):
- HAPPY: 긍정적 표현, 감사, 만족, 해결됨 ("감사합니다", "좋아요", "완벽해요", "이제 이해했어요")
- ANGRY: 불만, 화남, 거부, 강한 부정 ("화가 나요", "불가능해요", "거절합니다", "말도 안 돼요")
- ASKING: 질문, 궁금함, 확인 요청, 정보 요구 ("어떻게", "왜", "알려주세요", "주문번호가 뭐예요?")
- SAD: 슬픔, 실망, 좌절 ("슬프네요", "실망했어요", "아쉽습니다", "그렇다면 어쩔 수 없네요")
- NEUTRAL: 중립적 표현, 단순 정보 전달 (기본값)

제스처 판단 기준:
- HAND_WAVE: 인사, 환영 ("안녕하세요", "반갑습니다")
- NOD: 동의, 긍정, 이해 ("네", "맞아요", "그렇습니다", "알겠습니다")
- SHAKE_HEAD: 부정, 거부, 불만족 ("아니요", "안 됩니다", "그건 아니에요")
- POINT: 설명, 지시, 특정 항목 언급 ("여기", "이것", "저것", "주문번호는")
- NONE: 특별한 제스처 없음 (기본값)

긴급도 판단 기준:
- HIGH: 즉시 해결 필요, 긴급한 문제 ("지금 당장", "바로", "긴급", "중요해요")
- MEDIUM: 빠른 해결 선호, 중요하지만 긴급하지 않음
- LOW: 일반적인 문의, 긴급하지 않음 (기본값)

만족도 변화 (satisfaction_delta):
- 1.0: 매우 만족, 문제 해결됨, 감사 표현
- 0.5: 만족, 긍정적 반응
- 0.0: 중립, 변화 없음
- -0.5: 불만족, 부정적 반응
- -1.0: 매우 불만족, 화남, 거부

JSON만 응답하세요:"""

    try:
        # LLM 호출
        if st.session_state.is_llm_ready:
            response_text = run_llm(prompt)
            
            # JSON 파싱 시도
            try:
                # JSON 부분만 추출 (코드 블록 제거)
                import re
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    # 유효성 검사
                    valid_emotions = ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"]
                    valid_gestures = ["NONE", "HAND_WAVE", "NOD", "SHAKE_HEAD", "POINT"]
                    valid_urgencies = ["LOW", "MEDIUM", "HIGH"]
                    
                    emotion = result.get("emotion", "NEUTRAL")
                    gesture = result.get("gesture", "NONE")
                    urgency = result.get("urgency", "LOW")
                    satisfaction_delta = float(result.get("satisfaction_delta", 0.0))
                    confidence = float(result.get("confidence", 0.7))
                    
                    if emotion not in valid_emotions:
                        emotion = "NEUTRAL"
                    if gesture not in valid_gestures:
                        gesture = "NONE"
                    if urgency not in valid_urgencies:
                        urgency = "LOW"
                    
                    # ⭐ Gemini 제안: 상황별 키워드 추출
                    context_keywords = []
                    text_lower_for_context = text.lower()
                    
                    # 주요 상황별 키워드 매핑
                    if any(word in text_lower_for_context for word in ["주문번호", "order number", "주문 번호"]):
                        context_keywords.append("order_number")
                    if any(word in text_lower_for_context for word in ["해결", "완료", "감사", "solution", "resolved"]):
                        if satisfaction_delta > 0.3:
                            context_keywords.append("solution_accepted")
                    if any(word in text_lower_for_context for word in ["거절", "불가", "안 됩니다", "denied", "cannot"]):
                        if emotion == "ANGRY":
                            context_keywords.append("policy_denial")
                    
                    return {
                        "emotion": emotion,
                        "gesture": gesture,
                        "urgency": urgency,
                        "satisfaction_delta": max(-1.0, min(1.0, satisfaction_delta)),
                        "context_keywords": context_keywords,  # ⭐ 추가
                        "confidence": max(0.0, min(1.0, confidence))
                    }
            except json.JSONDecodeError:
                pass
        
        # LLM 호출 실패 시 키워드 기반 간단한 분석
        text_lower = text.lower()
        emotion = "NEUTRAL"
        gesture = "NONE"
        urgency = "LOW"
        satisfaction_delta = 0.0
        
        # 감정 키워드 분석
        if any(word in text_lower for word in ["감사", "좋아", "완벽", "만족", "고마워", "해결"]):
            emotion = "HAPPY"
            satisfaction_delta = 0.5
        elif any(word in text_lower for word in ["화", "불만", "거절", "불가능", "안 됩니다", "말도 안 돼"]):
            emotion = "ANGRY"
            satisfaction_delta = -0.5
        elif any(word in text_lower for word in ["어떻게", "왜", "알려", "질문", "궁금", "주문번호"]):
            emotion = "ASKING"
        elif any(word in text_lower for word in ["슬프", "실망", "아쉽", "그렇다면"]):
            emotion = "SAD"
            satisfaction_delta = -0.3
        
        # 긴급도 키워드 분석
        if any(word in text_lower for word in ["지금 당장", "바로", "긴급", "중요해요", "즉시"]):
            urgency = "HIGH"
        elif any(word in text_lower for word in ["빨리", "가능한 한", "최대한"]):
            urgency = "MEDIUM"
        
        # 제스처 키워드 분석
        if any(word in text_lower for word in ["안녕", "반갑", "인사"]):
            gesture = "HAND_WAVE"
        elif any(word in text_lower for word in ["네", "맞아", "그래", "동의", "알겠습니다"]):
            gesture = "NOD"
            if emotion == "HAPPY":
                satisfaction_delta = 0.3
        elif any(word in text_lower for word in ["아니", "안 됩니다", "거절"]):
            gesture = "SHAKE_HEAD"
            satisfaction_delta = -0.2
        elif any(word in text_lower for word in ["여기", "이것", "저것", "이거", "주문번호"]):
            gesture = "POINT"
        
        # ⭐ Gemini 제안: 상황별 키워드 추출 (키워드 기반 분석)
        context_keywords = []
        if any(word in text_lower for word in ["주문번호", "order number", "주문 번호"]):
            context_keywords.append("order_number")
        if any(word in text_lower for word in ["해결", "완료", "감사", "solution"]):
            if satisfaction_delta > 0.3:
                context_keywords.append("solution_accepted")
        if any(word in text_lower for word in ["거절", "불가", "안 됩니다"]):
            if emotion == "ANGRY":
                context_keywords.append("policy_denial")
        
        return {
            "emotion": emotion,
            "gesture": gesture,
            "urgency": urgency,
            "satisfaction_delta": satisfaction_delta,
            "context_keywords": context_keywords,  # ⭐ 추가
            "confidence": 0.6  # 키워드 기반 분석은 낮은 신뢰도
        }
    
    except Exception as e:
        print(f"텍스트 분석 오류: {e}")
        return {
            "emotion": "NEUTRAL", 
            "gesture": "NONE", 
            "urgency": "LOW",
            "satisfaction_delta": 0.0,
            "context_keywords": [],  # ⭐ 추가
            "confidence": 0.5
        }


def get_video_path_by_avatar(gender: str, emotion: str, is_speaking: bool = False, 
                             gesture: str = "NONE", context_keywords: List[str] = None) -> str:
    """
    고객 아바타 정보(성별, 감정 상태, 제스처, 상황)에 따라 적절한 비디오 경로를 반환합니다.
    OpenAI/Gemini 기반 영상 RAG: LLM이 분석한 감정/제스처에 따라 비디오 클립을 선택합니다.
    
    ⭐ Gemini 제안: 상황별 비디오 클립 패턴 확장 (예: male_asking_order_number.mp4)
    
    Args:
        gender: "male" 또는 "female"
        emotion: "NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD"
        is_speaking: 말하는 중인지 여부
        gesture: "NONE", "HAND_WAVE", "NOD", "SHAKE_HEAD", "POINT"
        context_keywords: 상황별 키워드 리스트 (예: ["order_number", "solution_accepted", "policy_denial"])
    
    Returns:
        비디오 파일 경로 (없으면 None)
    """
    # 비디오 디렉토리 경로 (사용자가 설정한 비디오 파일들이 저장된 위치)
    video_base_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(video_base_dir, exist_ok=True)
    
    # ⭐ Gemini 제안: 우선순위 -1 - 데이터베이스 기반 추천 비디오 (가장 우선)
    if context_keywords:
        db_recommended = get_recommended_video_from_database(emotion, gesture, context_keywords)
        if db_recommended:
            return db_recommended
    else:
        db_recommended = get_recommended_video_from_database(emotion, gesture, [])
        if db_recommended:
            return db_recommended
    
    # ⭐ Gemini 제안: 우선순위 0 - 상황별 비디오 클립 (가장 구체적)
    if context_keywords:
        for keyword in context_keywords:
            # 상황별 파일명 패턴 시도 (예: male_asking_order_number.mp4)
            context_filename = f"{gender}_{emotion.lower()}_{keyword}"
            if is_speaking:
                context_filename += "_speaking"
            context_filename += ".mp4"
            context_path = os.path.join(video_base_dir, context_filename)
            if os.path.exists(context_path):
                return context_path
            
            # 세션 상태에서도 확인
            context_video_key = f"video_{gender}_{emotion.lower()}_{keyword}"
            if context_video_key in st.session_state and st.session_state[context_video_key]:
                video_path = st.session_state[context_video_key]
                if os.path.exists(video_path):
                    return video_path
    
    # 우선순위 1: 제스처가 있는 경우 제스처별 비디오 시도
    if gesture != "NONE" and gesture:
        gesture_video_key = f"video_{gender}_{emotion.lower()}_{gesture.lower()}"
        if gesture_video_key in st.session_state and st.session_state[gesture_video_key]:
            video_path = st.session_state[gesture_video_key]
            if os.path.exists(video_path):
                return video_path
        
        # 제스처별 파일명 패턴 시도
        gesture_filename = f"{gender}_{emotion.lower()}_{gesture.lower()}"
        if is_speaking:
            gesture_filename += "_speaking"
        gesture_filename += ".mp4"
        gesture_path = os.path.join(video_base_dir, gesture_filename)
        if os.path.exists(gesture_path):
            return gesture_path
    
    # 우선순위 2: 감정 상태별 비디오 (제스처 없이)
    video_key = f"video_{gender}_{emotion.lower()}"
    if is_speaking:
        video_key += "_speaking"
    
    # 세션 상태에 저장된 비디오 경로가 있으면 사용
    if video_key in st.session_state and st.session_state[video_key]:
        video_path = st.session_state[video_key]
        if os.path.exists(video_path):
            return video_path
    
    # 기본 비디오 파일명 패턴 시도
    video_filename = f"{gender}_{emotion.lower()}"
    if is_speaking:
        video_filename += "_speaking"
    video_filename += ".mp4"
    
    video_path = os.path.join(video_base_dir, video_filename)
    if os.path.exists(video_path):
        return video_path
    
    # 우선순위 3: 기본 비디오 파일 시도 (중립 상태)
    default_video = os.path.join(video_base_dir, f"{gender}_neutral.mp4")
    if os.path.exists(default_video):
        return default_video
    
    # 우선순위 4: 세션 상태에서 업로드된 비디오 확인
    if "current_customer_video" in st.session_state and st.session_state.current_customer_video:
        return st.session_state.current_customer_video
    
    return None


# ⭐ Gemini 제안: 비디오 매핑 데이터베이스 관리 함수
def load_video_mapping_database() -> Dict[str, Any]:
    """비디오 매핑 데이터베이스를 로드합니다."""
    if os.path.exists(VIDEO_MAPPING_DB_FILE):
        try:
            with open(VIDEO_MAPPING_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"비디오 매핑 데이터베이스 로드 오류: {e}")
            return {"mappings": [], "feedback_history": []}
    return {"mappings": [], "feedback_history": []}


def save_video_mapping_database(db_data: Dict[str, Any]):
    """비디오 매핑 데이터베이스를 저장합니다."""
    try:
        with open(VIDEO_MAPPING_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"비디오 매핑 데이터베이스 저장 오류: {e}")


def add_video_mapping_feedback(
    customer_text: str,
    selected_video_path: str,
    emotion: str,
    gesture: str,
    context_keywords: List[str],
    user_rating: int,  # 1-5 점수
    user_comment: str = ""
) -> None:
    """
    ⭐ Gemini 제안: 사용자 피드백을 비디오 매핑 데이터베이스에 추가합니다.
    
    Args:
        customer_text: 고객의 텍스트
        selected_video_path: 선택된 비디오 경로
        emotion: 분석된 감정
        gesture: 분석된 제스처
        context_keywords: 상황별 키워드
        user_rating: 사용자 평가 (1-5)
        user_comment: 사용자 코멘트 (선택적)
    """
    db_data = load_video_mapping_database()
    
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_text": customer_text[:200],  # 최대 200자
        "selected_video": os.path.basename(selected_video_path) if selected_video_path else None,
        "video_path": selected_video_path,
        "emotion": emotion,
        "gesture": gesture,
        "context_keywords": context_keywords,
        "user_rating": user_rating,
        "user_comment": user_comment[:500] if user_comment else "",  # 최대 500자
        "is_natural_match": user_rating >= 4  # 4점 이상이면 자연스러운 매칭으로 간주
    }
    
    db_data["feedback_history"].append(feedback_entry)
    
    # 매핑 규칙 업데이트 (평가가 높은 경우)
    if user_rating >= 4:
        mapping_key = f"{emotion}_{gesture}_{'_'.join(context_keywords) if context_keywords else 'none'}"
        
        # 기존 매핑 찾기
        existing_mapping = None
        for mapping in db_data["mappings"]:
            if mapping.get("key") == mapping_key:
                existing_mapping = mapping
                break
        
        if existing_mapping:
            # 기존 매핑 업데이트 (평균 점수 계산)
            total_rating = existing_mapping.get("total_rating", 0) + user_rating
            count = existing_mapping.get("count", 0) + 1
            existing_mapping["total_rating"] = total_rating
            existing_mapping["count"] = count
            existing_mapping["avg_rating"] = total_rating / count
            existing_mapping["last_updated"] = datetime.now().isoformat()
        else:
            # 새 매핑 추가
            db_data["mappings"].append({
                "key": mapping_key,
                "emotion": emotion,
                "gesture": gesture,
                "context_keywords": context_keywords,
                "recommended_video": os.path.basename(selected_video_path) if selected_video_path else None,
                "video_path": selected_video_path,
                "total_rating": user_rating,
                "count": 1,
                "avg_rating": float(user_rating),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            })
    
    save_video_mapping_database(db_data)


def get_recommended_video_from_database(
    emotion: str,
    gesture: str,
    context_keywords: List[str]
) -> str:
    """
    ⭐ Gemini 제안: 데이터베이스에서 추천 비디오 경로를 가져옵니다.
    
    Args:
        emotion: 감정 상태
        gesture: 제스처
        context_keywords: 상황별 키워드
    
    Returns:
        추천 비디오 경로 (없으면 None)
    """
    db_data = load_video_mapping_database()
    
    mapping_key = f"{emotion}_{gesture}_{'_'.join(context_keywords) if context_keywords else 'none'}"
    
    # 정확한 매칭 찾기
    for mapping in db_data["mappings"]:
        if mapping.get("key") == mapping_key and mapping.get("avg_rating", 0) >= 4.0:
            video_path = mapping.get("video_path")
            if video_path and os.path.exists(video_path):
                return video_path
    
    # 부분 매칭 시도 (감정과 제스처만)
    partial_key = f"{emotion}_{gesture}_none"
    for mapping in db_data["mappings"]:
        if mapping.get("key") == partial_key and mapping.get("avg_rating", 0) >= 4.0:
            video_path = mapping.get("video_path")
            if video_path and os.path.exists(video_path):
                return video_path
    
    return None


def render_synchronized_video(text: str, audio_bytes: bytes, gender: str, emotion: str, 
                               role: str = "customer", autoplay: bool = True,
                               gesture: str = "NONE", context_keywords: List[str] = None):
    """
    TTS 오디오와 동기화된 비디오를 렌더링합니다.
    
    ⭐ Gemini 제안: 피드백 평가 기능 추가
    
    Args:
        text: 말하는 텍스트 내용
        audio_bytes: TTS로 생성된 오디오 바이트
        gender: 고객 성별 ("male" 또는 "female")
        emotion: 감정 상태 ("NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD")
        role: 역할 ("customer" 또는 "agent")
        autoplay: 자동 재생 여부
        gesture: 제스처 (선택적)
        context_keywords: 상황별 키워드 (선택적)
    """
    if role == "customer":
        is_speaking = True
        if context_keywords is None:
            context_keywords = []
        
        # ⭐ Gemini 제안: 데이터베이스 기반 추천 비디오 우선 사용
        video_path = get_video_path_by_avatar(gender, emotion, is_speaking, gesture, context_keywords)
        
        if video_path and os.path.exists(video_path):
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                
                # 비디오와 오디오를 함께 재생
                # Streamlit의 st.video는 오디오 트랙이 있는 비디오를 지원합니다
                # 여기서는 비디오만 표시하고, 오디오는 별도로 재생합니다
                st.video(video_bytes, format="video/mp4", autoplay=autoplay, loop=False, muted=False)
                
                # 오디오도 함께 재생 (동기화)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
                
                # ⭐ Gemini 제안: 사용자 피드백 평가 UI 추가 (채팅/이메일 탭용)
                if not autoplay:  # 자동 재생이 아닌 경우에만 피드백 UI 표시
                    st.markdown("---")
                    st.markdown("**💬 비디오 매칭 평가**")
                    st.caption("이 비디오가 고객의 텍스트와 감정에 자연스럽게 매칭되었습니까?")
                    
                    feedback_key = f"video_feedback_chat_{st.session_state.get('sim_instance_id', 'default')}_{hash(text) % 10000}"
                    
                    col_rating, col_comment = st.columns([2, 3])
                    with col_rating:
                        rating = st.slider(
                            "평가 점수 (1-5점)",
                            min_value=1,
                            max_value=5,
                            value=3,
                            key=f"{feedback_key}_rating",
                            help="1점: 매우 부자연스러움, 5점: 매우 자연스러움"
                        )
                    
                    with col_comment:
                        comment = st.text_input(
                            "의견 (선택사항)",
                            key=f"{feedback_key}_comment",
                            placeholder="예: 비디오가 텍스트와 잘 맞았습니다"
                        )
                    
                    if st.button("피드백 제출", key=f"{feedback_key}_submit"):
                        # 피드백을 데이터베이스에 저장
                        add_video_mapping_feedback(
                            customer_text=text[:200],
                            selected_video_path=video_path,
                            emotion=emotion,
                            gesture=gesture,
                            context_keywords=context_keywords,
                            user_rating=rating,
                            user_comment=comment
                        )
                        st.success(f"✅ 피드백이 저장되었습니다! (점수: {rating}/5)")
                        st.info("💡 이 피드백은 향후 비디오 선택 정확도를 개선하는 데 사용됩니다.")
                
                return True
            except Exception as e:
                st.warning(f"비디오 재생 오류: {e}")
                # 비디오 재생 실패 시 오디오만 재생
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
                return False
        else:
            # 비디오가 없으면 오디오만 재생
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
            return False
    else:
        # 에이전트는 비디오 없이 오디오만 재생
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay, loop=False)
        return False


def generate_virtual_human_video(text: str, audio_bytes: bytes, gender: str, emotion: str, 
                                 provider: str = "hyperclova") -> bytes:
    """
    가상 휴먼 기술을 사용하여 텍스트와 오디오에 맞는 비디오를 생성합니다.
    
    ⚠️ 주의: OpenAI/Gemini API만으로는 입모양 동기화 비디오 생성이 불가능합니다.
    가상 휴먼 비디오 생성은 별도의 가상 휴먼 API (예: Hyperclova)가 필요합니다.
    
    현재는 미리 준비된 비디오 파일을 사용하는 방식을 권장합니다.
    
    Args:
        text: 말하는 텍스트 내용
        audio_bytes: TTS로 생성된 오디오 바이트
        gender: 고객 성별 ("male" 또는 "female")
        emotion: 감정 상태 ("NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD", "HOLD")
        provider: 가상 휴먼 제공자 ("hyperclova", "other")
    
    Returns:
        생성된 비디오 바이트 (없으면 None)
    """
    # 가상 휴먼 API 키 확인
    if provider == "hyperclova":
        api_key = get_api_key("hyperclova")
        if not api_key:
            return None
        
        # TODO: Hyperclova API 연동 구현 (별도 API 필요)
        # OpenAI/Gemini API만으로는 불가능하므로, 실제 가상 휴먼 API가 필요합니다.
        # 예시 구조:
        # response = requests.post(
        #     "https://api.hyperclova.com/virtual-human/generate",
        #     headers={"Authorization": f"Bearer {api_key}"},
        #     json={
        #         "text": text,
        #         "audio": base64.b64encode(audio_bytes).decode(),
        #         "gender": gender,
        #         "emotion": emotion
        #     }
        # )
        # return response.content
    
    # 다른 제공자도 여기에 추가 가능
    # elif provider == "other":
    #     ...
    
    return None


def get_virtual_human_config() -> Dict[str, Any]:
    """
    가상 휴먼 설정을 반환합니다.
    
    Returns:
        가상 휴먼 설정 딕셔너리
    """
    return {
        "enabled": st.session_state.get("virtual_human_enabled", False),
        "provider": st.session_state.get("virtual_human_provider", "hyperclova"),
        "api_key": get_api_key("hyperclova") if st.session_state.get("virtual_human_provider", "hyperclova") == "hyperclova" else None
    }


# 역할별 TTS 음성 스타일 설정
TTS_VOICES = {
    "customer_male": {
        "gender": "male",
        "voice": "alloy"  # Male voice
    },
    "customer_female": {
        "gender": "female",
        "voice": "nova"  # Female voice
    },
    "customer": {
        "gender": "male",
        "voice": "alloy"  # Default male voice (fallback)
    },
    "agent": {
        "gender": "female",
        "voice": "shimmer"  # Distinct Female, Professional/Agent
    },
    "supervisor": {
        "gender": "female",
        "voice": "nova"  # Another Distinct Female, Informative/Supervisor
    }
}


def synthesize_tts(text: str, lang_key: str, role: str = "agent"):
    # lang_key 검증 및 기본값 처리
    if not lang_key or lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"  # 최종 기본값
    
    L = LANG.get(lang_key, LANG["ko"])  # 안전한 접근
    client = st.session_state.openai_client
    if client is None:
        return None, L.get("openai_missing", "OpenAI API Key가 필요합니다.")

    # ⭐ 수정: 고객 역할인 경우 성별에 따라 음성 선택
    if role == "customer":
        customer_gender = st.session_state.customer_avatar.get("gender", "male")
        if customer_gender == "female":
            voice_key = "customer_female"
        else:
            voice_key = "customer_male"
        
        if voice_key in TTS_VOICES:
            voice_name = TTS_VOICES[voice_key]["voice"]
        else:
            voice_name = TTS_VOICES["customer"]["voice"]  # Fallback
    elif role in TTS_VOICES:
        voice_name = TTS_VOICES[role]["voice"]
    else:
        voice_name = TTS_VOICES["agent"]["voice"]  # Default fallback

    try:
        # ⭐ 수정: 텍스트 길이 제한을 제거하여 전체 문의가 재생되도록 함
        # OpenAI TTS는 최대 4096자를 지원하지만, 실제로는 더 긴 텍스트도 처리 가능
        # 고객의 문의를 끝까지 다 들어야 원활한 응대가 가능하므로 전체 텍스트를 처리
        # 만약 텍스트가 너무 길면 (예: 10000자 이상) 여러 청크로 나눠서 처리할 수 있지만,
        # 일반적인 고객 문의는 4096자 이내이므로 전체를 처리
        
        # tts-1 모델 사용 (안정성)
        resp = client.audio.speech.create(
            model="tts-1",
            voice=voice_name,
            input=text
            # format="mp3"은 기본값입니다.
        )
        return resp.read(), L["tts_status_success"]

    except Exception as e:
        return None, f"{L['tts_status_error']}: {e}"


# ----------------------------------------
# TTS Helper
# ----------------------------------------

def render_tts_button(text, lang_key, role="customer", prefix="", index: int = -1):
    # lang_key 검증 및 기본값 처리
    if not lang_key or lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"  # 최종 기본값
    
    L = LANG.get(lang_key, LANG["ko"])  # 안전한 접근

    # ⭐ 수정: index=-1인 경우, UUID를 사용하여 safe_key 생성
    if index == -1:
        # 이관 요약처럼 인덱스가 고정되지 않는 경우, 텍스트 해시와 세션 인스턴스 ID를 조합
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        session_id_part = st.session_state.get('sim_instance_id', 'default_session')
        # ⭐ 수정: 이관 요약의 경우 안정적인 키를 생성 (time.time_ns() 제거하여 매번 같은 키 생성)
        # 언어 코드도 추가하여 이관 후 언어 변경 시에도 고유성 보장
        lang_code = st.session_state.get('language', lang_key)
        safe_key = f"{prefix}_SUMMARY_{session_id_part}_{lang_code}_{content_hash}"
    else:
        # 대화 로그처럼 인덱스가 존재하는 경우 (기존 로직 유지)
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()
        safe_key = f"{prefix}_{index}_{content_hash}"

    # 재생 버튼을 누를 때만 TTS 요청
    if st.button(L["button_listen_audio"], key=safe_key):
        if not st.session_state.openai_client:
            st.error(L["openai_missing"])
            return  # 키 없으면 종료

        with st.spinner(L["tts_status_generating"]):
            try:
                audio_bytes, msg = synthesize_tts(text, lang_key, role=role)
                if audio_bytes:
                    # ⭐ st.audio 호출 시 성공한 경우에만 재생 시간을 확보
                    # Streamlit 문서: autoplay는 브라우저 정책상 사용자 상호작용 없이는 작동하지 않을 수 있음
                    try:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                        st.success(msg)
                        # ⭐ 수정: 재생이 시작될 충분한 시간을 확보하기 위해 대기 시간을 3초로 늘림
                        time.sleep(3)
                    except Exception as e:
                        st.warning(f"오디오 재생 중 오류: {e}. 오디오 파일은 생성되었지만 자동 재생에 실패했습니다.")
                        st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                        st.success(msg)
                else:
                    st.error(msg)
                    time.sleep(1)  # 에러 발생 시도 잠시 대기
            except Exception as e:
                # TTS API 호출 자체에서 예외 발생 시 (네트워크 등)
                st.error(f"❌ TTS 생성 중 치명적인 오류 발생: {e}")
                time.sleep(1)

            # 버튼 클릭 이벤트 후, 불필요한 재실행을 막기 위해 여기서 함수 종료
            return
        # [중략: TTS Helper 끝]


# ========================================
# 4. 로컬 음성 기록 Helper
# ========================================

def load_voice_records() -> List[Dict[str, Any]]:
    return _load_json(VOICE_META_FILE, [])


def save_voice_records(records: List[Dict[str, Any]]):
    _save_json(VOICE_META_FILE, records)


def save_audio_record_local(
        audio_bytes: bytes,
        filename: str,
        transcript_text: str,
        mime_type: str = "audio/webm",
        meta: Dict[str, Any] = None,
) -> str:
    records = load_voice_records()
    rec_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    ext = filename.split(".")[-1] if "." in filename else "webm"
    audio_filename = f"{rec_id}.{ext}"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    rec = {
        "id": rec_id,
        "created_at": ts,
        "filename": filename,
        "audio_filename": audio_filename,
        "size": len(audio_bytes),
        "transcript": transcript_text,
        "mime_type": mime_type,
        "language": st.session_state.language,
        "meta": meta or {},
    }
    records.insert(0, rec)
    save_voice_records(records)
    return rec_id


def delete_audio_record_local(rec_id: str) -> bool:
    records = load_voice_records()
    idx = next((i for i, r in enumerate(records) if r.get("id") == rec_id), None)
    if idx is None:
        return False
    rec = records.pop(idx)
    audio_filename = rec.get("audio_filename")
    if audio_filename:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        try:
            os.remove(audio_path)
        except FileNotFoundError:
            pass
    save_voice_records(records)
    return True


def get_audio_bytes_local(rec_id: str):
    records = load_voice_records()
    rec = next((r for r in records if r.get("id") == rec_id), None)
    if not rec:
        raise FileNotFoundError("record not found")
    audio_filename = rec["audio_filename"]
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    with open(audio_path, "rb") as f:
        b = f.read()
    return b, rec


# ========================================
# 5. 로컬 시뮬레이션 이력 Helper (요청 4 반영)
# ========================================

def load_simulation_histories_local(lang_key: str) -> List[Dict[str, Any]]:
    histories = _load_json(SIM_META_FILE, [])
    # 현재 언어와 메시지 리스트가 유효한 이력만 필터링
    return [
        h for h in histories
        if h.get("language_key") == lang_key and (isinstance(h.get("messages"), list) or h.get("summary"))
    ]


def generate_chat_summary(messages: List[Dict[str, Any]], initial_query: str, customer_type: str,
                          current_lang_key: str) -> Dict[str, Any]:
    """채팅 내용을 AI로 요약하여 주요 정보와 점수를 추출 (요청 4)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 대화 내용 추출
    conversation_text = f"Initial Query: {initial_query}\n\n"
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["customer", "customer_rebuttal", "phone_exchange"]:
            conversation_text += f"Customer: {content}\n"
        elif role == "agent_response" or role == "agent":
            conversation_text += f"Agent: {content}\n"
        # supervisor 메시지는 LLM에 전달하지 않아 역할 혼동 방지

    # 폰 교환 로그는 이미 "Agent: ... | Customer: ..." 형태로 기록되므로,
    # generate_summary_for_call 함수에서 별도로 처리할 필요 없이,
    # 여기서는 범용 채팅 요약 로직을 따르도록 메시지를 정제합니다.

    summary_prompt = f"""
You are an AI analyst summarizing a customer support conversation.

Analyze the conversation and provide a structured summary in JSON format (ONLY JSON, no markdown).

Extract and score:
1. Main inquiry topic (what the customer asked about)
2. Key responses provided by the agent (list of max 3 core actions/solutions)
3. Customer sentiment score (0-100, where 0=very negative, 50=neutral, 100=very positive)
4. Customer satisfaction score (0-100, based on final response)
5. Customer characteristics:
   - Language preference (if mentioned)
   - Cultural background hints (if any)
   - Location/region (if mentioned, but anonymize specific addresses)
   - Communication style (formal/casual, brief/detailed)
6. Privacy-sensitive information (anonymize: names, emails, phone numbers, specific addresses)
   - Extract patterns only (e.g., "email provided", "phone number provided", "resides in Asia region")

Output format (JSON only):
{{
  "main_inquiry": "brief description of main issue",
  "key_responses": ["response 1", "response 2"],
  "customer_sentiment_score": 75,
  "customer_satisfaction_score": 80,
  "customer_characteristics": {{
    "language": "ko/en/ja or unknown",
    "cultural_hints": "brief description or unknown",
    "region": "general region or unknown",
    "communication_style": "formal/casual/brief/detailed"
  }},
  "privacy_info": {{
    "has_email": true/false,
    "has_phone": true/false,
    "has_address": true/false,
    "region_hint": "general region or unknown"
  }},
  "summary": "overall conversation summary in {lang_name}"
}}

Conversation:
{conversation_text}

JSON Output:
"""

    if not st.session_state.is_llm_ready:
        # Fallback summary
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "region": "unknown",
                "communication_style": "unknown"
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "summary": f"Customer inquiry about: {initial_query[:100]}"
        }

    try:
        summary_text = run_llm(summary_prompt).strip()
        # JSON 추출 (마크다운 코드 블록 제거)
        if "```json" in summary_text:
            summary_text = summary_text.split("```json")[1].split("```")[0].strip()
        elif "```" in summary_text:
            summary_text = summary_text.split("```")[1].split("```")[0].strip()

        import json
        summary_data = json.loads(summary_text)
        return summary_data
    except Exception as e:
        # JSON 파싱 실패 시 fallback
        st.warning(f"요약 생성 중 오류 발생: {e}")
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "region": "unknown",
                "communication_style": "unknown"
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "summary": f"Customer inquiry about: {initial_query[:100]}"
        }
        # Fallback on error
        return {
            "main_inquiry": initial_query[:100],
            "key_responses": [],
            "customer_sentiment_score": 50,
            "customer_satisfaction_score": 50,
            "customer_characteristics": {
                "language": current_lang_key,
                "cultural_hints": "unknown",
                "region": "unknown",
                "communication_style": "unknown"
            },
            "privacy_info": {
                "has_email": False,
                "has_phone": False,
                "has_address": False,
                "region_hint": "unknown"
            },
            "summary": f"Error generating summary: {str(e)}"
        }


def save_simulation_history_local(initial_query: str, customer_type: str, messages: List[Dict[str, Any]],
                                  is_chat_ended: bool, attachment_context: str, is_call: bool = False):
    """AI 요약 데이터를 중심으로 이력을 저장 (요청 4 반영)"""
    histories = _load_json(SIM_META_FILE, [])
    doc_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    # AI 요약 생성 (채팅 종료 시 또는 충분한 대화가 있을 때)
    summary_data = None
    if is_chat_ended or len(messages) > 4 or is_call:  # 전화 통화는 바로 요약 시도
        summary_data = generate_chat_summary(messages, initial_query, customer_type, st.session_state.language)

    # 요약 데이터가 생성된 경우에만 저장 (요약 중심 저장)
    if summary_data:
        # 요약 데이터에 초기 문의와 핵심 정보 포함
        data = {
            "id": doc_id,
            "initial_query": initial_query,  # 초기 문의는 유지
            "customer_type": customer_type,
            "messages": [],  # 전체 메시지는 저장하지 않음 (요약만 저장)
            "summary": summary_data,  # AI 요약 데이터 (주요 저장 내용)
            "language_key": st.session_state.language,
            "timestamp": ts,
            "is_chat_ended": is_chat_ended,
            "attachment_context": attachment_context if attachment_context else "",  # 첨부 파일 컨텍스트
            "is_call": is_call,  # 전화 여부 플래그
        }
    else:
        # 요약이 아직 생성되지 않은 경우 (진행 중인 대화), 최소한의 정보만 저장
        data = {
            "id": doc_id,
            "initial_query": initial_query,
            "customer_type": customer_type,
            "messages": messages[:10] if len(messages) > 10 else messages,  # 최근 10개만 저장
            "summary": None,  # 요약 없음
            "language_key": st.session_state.language,
            "timestamp": ts,
            "is_chat_ended": is_chat_ended,
            "attachment_context": attachment_context if attachment_context else "",
            "is_call": is_call,
        }

    # 기존 이력에 추가 (최신순)
    histories.insert(0, data)
    # 너무 많은 이력 방지 (예: 100개로 증가 - 요약만 저장하므로 용량 부담 적음)
    _save_json(SIM_META_FILE, histories[:100])
    return doc_id


def delete_all_history_local():
    _save_json(SIM_META_FILE, [])


# ========================================
# DB 저장 기능 (Word/PPTX/PDF)
# ========================================
def export_history_to_word(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 Word 파일로 저장"""
    if not IS_DOCX_AVAILABLE:
        raise ImportError("Word 저장을 위해 python-docx가 필요합니다: pip install python-docx")
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    filepath = os.path.join(DATA_DIR, filename)
    
    doc = DocxDocument()
    
    # 제목 추가
    title = doc.add_heading('고객 응대 이력 요약', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 각 이력 추가
    for i, hist in enumerate(histories, 1):
        # 이력 제목
        doc.add_heading(f'이력 #{i}', level=1)
        
        # 기본 정보
        doc.add_paragraph(f'ID: {hist.get("id", "N/A")}')
        doc.add_paragraph(f'날짜: {hist.get("timestamp", "N/A")}')
        doc.add_paragraph(f'초기 문의: {hist.get("initial_query", "N/A")}')
        doc.add_paragraph(f'고객 유형: {hist.get("customer_type", "N/A")}')
        doc.add_paragraph(f'언어: {hist.get("language_key", "N/A")}')
        
        summary = hist.get('summary', {})
        if summary:
            # 요약 섹션
            doc.add_heading('요약', level=2)
            doc.add_paragraph(f'주요 문의: {summary.get("main_inquiry", "N/A")}')
            doc.add_paragraph(f'핵심 응답: {", ".join(summary.get("key_responses", []))}')
            doc.add_paragraph(f'고객 감정 점수: {summary.get("customer_sentiment_score", "N/A")}/100')
            doc.add_paragraph(f'고객 만족도 점수: {summary.get("customer_satisfaction_score", "N/A")}/100')
            
            # 고객 특성
            characteristics = summary.get('customer_characteristics', {})
            doc.add_heading('고객 특성', level=2)
            doc.add_paragraph(f'언어: {characteristics.get("language", "N/A")}')
            doc.add_paragraph(f'문화적 배경: {characteristics.get("cultural_hints", "N/A")}')
            doc.add_paragraph(f'지역: {characteristics.get("region", "N/A")}')
            doc.add_paragraph(f'소통 스타일: {characteristics.get("communication_style", "N/A")}')
            
            # 개인정보 요약
            privacy = summary.get('privacy_info', {})
            doc.add_heading('개인정보 요약', level=2)
            doc.add_paragraph(f'이메일 제공: {"예" if privacy.get("has_email") else "아니오"}')
            doc.add_paragraph(f'전화번호 제공: {"예" if privacy.get("has_phone") else "아니오"}')
            doc.add_paragraph(f'주소 제공: {"예" if privacy.get("has_address") else "아니오"}')
            doc.add_paragraph(f'지역 힌트: {privacy.get("region_hint", "N/A")}')
            
            # 전체 요약
            doc.add_paragraph(f'전체 요약: {summary.get("summary", "N/A")}')
        
        # 구분선
        if i < len(histories):
            doc.add_paragraph('-' * 80)
    
    doc.save(filepath)
    return filepath


def export_history_to_pptx(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 PPTX 파일로 저장"""
    if not IS_PPTX_AVAILABLE:
        raise ImportError("PPTX 저장을 위해 python-pptx가 필요합니다: pip install python-pptx")
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx"
    filepath = os.path.join(DATA_DIR, filename)
    
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # 제목 슬라이드
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "고객 응대 이력 요약"
    subtitle.text = f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # 각 이력에 대해 슬라이드 생성
    for i, hist in enumerate(histories, 1):
        # 제목 및 내용 레이아웃
        bullet_slide_layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(bullet_slide_layout)
        shapes = slide.shapes
        
        title_shape = shapes.title
        body_shape = shapes.placeholders[1]
        
        title_shape.text = f"이력 #{i}"
        
        tf = body_shape.text_frame
        tf.text = f"ID: {hist.get('id', 'N/A')}"
        
        p = tf.add_paragraph()
        p.text = f"날짜: {hist.get('timestamp', 'N/A')}"
        p.level = 0
        
        p = tf.add_paragraph()
        p.text = f"초기 문의: {hist.get('initial_query', 'N/A')}"
        p.level = 0
        
        p = tf.add_paragraph()
        p.text = f"고객 유형: {hist.get('customer_type', 'N/A')}"
        p.level = 0
        
        summary = hist.get('summary', {})
        if summary:
            p = tf.add_paragraph()
            p.text = f"주요 문의: {summary.get('main_inquiry', 'N/A')}"
            p.level = 0
            
            p = tf.add_paragraph()
            p.text = f"고객 감정 점수: {summary.get('customer_sentiment_score', 'N/A')}/100"
            p.level = 0
            
            p = tf.add_paragraph()
            p.text = f"고객 만족도 점수: {summary.get('customer_satisfaction_score', 'N/A')}/100"
            p.level = 0
    
    prs.save(filepath)
    return filepath


def export_history_to_pdf(histories: List[Dict[str, Any]], filename: str = None) -> str:
    """이력을 PDF 파일로 저장 (한글 인코딩 지원)"""
    if not IS_REPORTLAB_AVAILABLE:
        raise ImportError("PDF 저장을 위해 reportlab이 필요합니다: pip install reportlab")
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(DATA_DIR, filename)
    
    # ⭐ 수정: 한글/일본어 폰트 지원을 위한 폰트 설정
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # 한글 폰트 등록 시도 (시스템 폰트 사용)
    korean_font_registered = False
    japanese_font_registered = False
    selected_font_name = None
    
    try:
        # Windows 기본 한글 폰트 경로 시도 (더 많은 경로 추가)
        korean_font_paths = [
            "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕 (TTF)
            "C:/Windows/Fonts/malgunsl.ttf",  # 맑은 고딕 (TTF, 대체)
            "C:/Windows/Fonts/NanumGothic.ttf",  # 나눔고딕
            "C:/Windows/Fonts/NanumBarunGothic.ttf",  # 나눔바른고딕
            "C:/Windows/Fonts/NanumGothicBold.ttf",  # 나눔고딕 볼드
            "C:/Windows/Fonts/gulim.ttc",  # 굴림 (TTC)
            "C:/Windows/Fonts/batang.ttc",  # 바탕 (TTC)
            "C:/Windows/Fonts/malgun.ttc",  # 맑은 고딕 (TTC)
        ]
        
        # 일본어 폰트 경로 (한자 지원 강화)
        japanese_font_paths = [
            "C:/Windows/Fonts/msgothic.ttc",  # MS Gothic (일본어 한자 지원)
            "C:/Windows/Fonts/msmincho.ttc",  # MS Mincho (일본어 한자 지원)
            "C:/Windows/Fonts/meiryo.ttc",  # Meiryo (일본어)
            "C:/Windows/Fonts/yuanti.ttc",  # Microsoft YaHei (중국어/일본어 한자 지원)
        ]
        
        # 한글 폰트 등록
        for font_path in korean_font_paths:
            if os.path.exists(font_path):
                try:
                    if font_path.endswith('.ttf'):
                        # TTF 파일 등록
                        font = TTFont('KoreanFont', font_path)
                        pdfmetrics.registerFont(font)
                        # 등록 확인
                        if 'KoreanFont' in pdfmetrics.getRegisteredFontNames():
                            korean_font_registered = True
                            selected_font_name = 'KoreanFont'
                            print(f"✅ 한글 폰트 등록 성공: {font_path}")
                            break
                    elif font_path.endswith('.ttc'):
                        # TTC 파일 처리 (여러 방법 시도)
                        for subfont_idx in range(4):  # 최대 4개 서브폰트 시도
                            try:
                                font = TTFont('KoreanFont', font_path, subfontIndex=subfont_idx)
                                pdfmetrics.registerFont(font)
                                # 등록 확인
                                if 'KoreanFont' in pdfmetrics.getRegisteredFontNames():
                                    korean_font_registered = True
                                    selected_font_name = 'KoreanFont'
                                    print(f"✅ 한글 폰트 등록 성공 (TTC, subfontIndex={subfont_idx}): {font_path}")
                                    break
                            except Exception as subfont_error:
                                if subfont_idx == 3:  # 마지막 시도
                                    print(f"⚠️ TTC 폰트 등록 실패 (subfontIndex={subfont_idx}): {subfont_error}")
                                continue
                        if korean_font_registered:
                            break
                except Exception as font_error:
                    print(f"⚠️ 폰트 등록 실패 ({font_path}): {font_error}")
                    continue
        
        # 일본어 폰트 등록 (한자 지원 강화) - 한글 폰트가 없을 때만 시도
        if not korean_font_registered:
            for font_path in japanese_font_paths:
                if os.path.exists(font_path):
                    try:
                        if font_path.endswith('.ttf'):
                            font = TTFont('JapaneseFont', font_path)
                            pdfmetrics.registerFont(font)
                            # 등록 확인
                            if 'JapaneseFont' in pdfmetrics.getRegisteredFontNames():
                                japanese_font_registered = True
                                selected_font_name = 'JapaneseFont'
                                print(f"✅ 일본어 폰트 등록 성공: {font_path}")
                                break
                        elif font_path.endswith('.ttc'):
                            # TTC 파일 처리 (여러 서브폰트 시도)
                            for subfont_idx in range(4):  # 최대 4개 서브폰트 시도
                                try:
                                    font = TTFont('JapaneseFont', font_path, subfontIndex=subfont_idx)
                                    pdfmetrics.registerFont(font)
                                    # 등록 확인
                                    if 'JapaneseFont' in pdfmetrics.getRegisteredFontNames():
                                        japanese_font_registered = True
                                        selected_font_name = 'JapaneseFont'
                                        print(f"✅ 일본어 폰트 등록 성공 (TTC, subfontIndex={subfont_idx}): {font_path}")
                                        break
                                except Exception as subfont_error:
                                    if subfont_idx == 3:  # 마지막 시도
                                        print(f"⚠️ 일본어 TTC 폰트 등록 실패 (subfontIndex={subfont_idx}): {subfont_error}")
                                    continue
                            if japanese_font_registered:
                                break
                    except Exception as font_error:
                        print(f"⚠️ 일본어 폰트 등록 실패 ({font_path}): {font_error}")
                        continue
        
        # 폰트 등록 실패 시 경고
        if not korean_font_registered and not japanese_font_registered:
            print("⚠️ 경고: 한글/일본어 폰트를 찾을 수 없습니다. PDF에서 한글이 깨질 수 있습니다.")
            print("   등록된 폰트 목록:", pdfmetrics.getRegisteredFontNames())
            print("   폰트 경로 확인 필요: C:/Windows/Fonts/")
            selected_font_name = None
            
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ 폰트 등록 실패: {error_msg}")
        korean_font_registered = False
        japanese_font_registered = False
        selected_font_name = None
    
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # ⭐ 수정: 한글/영어/일본어 폰트를 사용하는 스타일 생성 (폰트 강제 적용)
    def get_korean_style(base_style_name, **kwargs):
        base_style = styles[base_style_name]
        style_kwargs = {
            'parent': base_style,
            **kwargs
        }
        # 선택된 폰트 사용 (폰트가 등록되어 있는 경우)
        if selected_font_name:
            registered_fonts = pdfmetrics.getRegisteredFontNames()
            if selected_font_name in registered_fonts:
                style_kwargs['fontName'] = selected_font_name
            else:
                print(f"⚠️ 경고: {selected_font_name}가 등록 목록에 없습니다. 등록된 폰트: {registered_fonts}")
                # 폰트가 없으면 기본 폰트 사용 (한글이 깨질 수 있음)
        else:
            # 폰트가 없으면 기본 폰트 사용 (한글이 깨질 수 있음)
            print("⚠️ 경고: 한글 폰트가 없어 기본 폰트를 사용합니다. 한글이 깨질 수 있습니다.")
        return ParagraphStyle(f'Korean{base_style_name}', **style_kwargs)
    
    # 제목 스타일 (한글 폰트 사용)
    title_style = get_korean_style(
        'Heading1',
        fontSize=24,
        textColor=black,
        spaceAfter=30,
        alignment=1  # 중앙 정렬
    )
    
    # 일반 텍스트 스타일 (한글 폰트 사용)
    normal_style = get_korean_style('Normal')
    heading1_style = get_korean_style('Heading1')
    heading2_style = get_korean_style('Heading2')
    
    # ⭐ 수정: 텍스트를 안전하게 처리하는 헬퍼 함수 (UTF-8 인코딩 명시적 처리, 한글/일본어 지원 강화)
    def safe_text(text):
        """텍스트를 안전하게 처리하여 PDF에 표시 (한글/일본어/한자 지원)"""
        if text is None:
            return "N/A"
        
        # 문자열로 변환 (UTF-8 인코딩 명시적 처리)
        text_str = None
        if isinstance(text, bytes):
            # 바이트 문자열인 경우 UTF-8로 디코딩 시도
            try:
                text_str = text.decode('utf-8', errors='replace')
            except:
                try:
                    # UTF-8 실패 시 다른 인코딩 시도
                    text_str = text.decode('cp949', errors='replace')  # 한국어 Windows 인코딩
                except:
                    try:
                        text_str = text.decode('shift_jis', errors='replace')  # 일본어 인코딩
                    except:
                        text_str = text.decode('latin-1', errors='replace')
        else:
            text_str = str(text)
        
        # None 체크
        if text_str is None:
            return "N/A"
        
        # 유니코드 정규화 (NFC 형식으로 통일)
        try:
            import unicodedata
            text_str = unicodedata.normalize('NFC', text_str)
        except:
            pass
        
        # 특수 문자 이스케이프 (HTML 엔티티로 변환) - ReportLab Paragraph는 HTML을 지원
        # 하지만 &는 먼저 처리해야 함
        text_str = text_str.replace('&', '&amp;')
        text_str = text_str.replace('<', '&lt;')
        text_str = text_str.replace('>', '&gt;')
        text_str = text_str.replace('"', '&quot;')
        text_str = text_str.replace("'", '&#39;')
        
        # 한글/일본어 문자가 제대로 있는지 확인
        try:
            # 유니코드 범위 확인 (한글: AC00-D7AF, 일본어 히라가나: 3040-309F, 가타카나: 30A0-30FF, 한자: 4E00-9FFF)
            has_korean = any('\uAC00' <= char <= '\uD7AF' for char in text_str)
            has_japanese = any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF' for char in text_str)
            
            if has_korean or has_japanese:
                # 폰트 등록 상태 확인
                registered_fonts = pdfmetrics.getRegisteredFontNames()
                has_korean_font = 'KoreanFont' in registered_fonts
                has_japanese_font = 'JapaneseFont' in registered_fonts
                
                if not has_korean_font and not has_japanese_font:
                    print(f"⚠️ 경고: 한글/일본어 문자가 포함되어 있지만 폰트가 등록되지 않았습니다.")
                    print(f"   텍스트 샘플: {text_str[:50]}")
                    print(f"   등록된 폰트: {registered_fonts}")
        except Exception as check_error:
            # 확인 중 오류가 발생해도 계속 진행
            pass
        
        return text_str
    
    # 제목 추가
    story.append(Paragraph(safe_text('고객 응대 이력 요약'), title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # 각 이력 추가
    for i, hist in enumerate(histories, 1):
        # 이력 제목
        story.append(Paragraph(safe_text(f'이력 #{i}'), heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        # 기본 정보
        story.append(Paragraph(safe_text(f'ID: {hist.get("id", "N/A")}'), normal_style))
        story.append(Paragraph(safe_text(f'날짜: {hist.get("timestamp", "N/A")}'), normal_style))
        story.append(Paragraph(safe_text(f'초기 문의: {hist.get("initial_query", "N/A")}'), normal_style))
        story.append(Paragraph(safe_text(f'고객 유형: {hist.get("customer_type", "N/A")}'), normal_style))
        story.append(Paragraph(safe_text(f'언어: {hist.get("language_key", "N/A")}'), normal_style))
        
        summary = hist.get('summary', {})
        if summary:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(safe_text('요약'), heading2_style))
            story.append(Paragraph(safe_text(f'주요 문의: {summary.get("main_inquiry", "N/A")}'), normal_style))
            
            key_responses = summary.get("key_responses", [])
            if isinstance(key_responses, list):
                responses_text = ", ".join([safe_text(r) for r in key_responses])
            else:
                responses_text = safe_text(key_responses)
            story.append(Paragraph(safe_text(f'핵심 응답: {responses_text}'), normal_style))
            story.append(Paragraph(safe_text(f'고객 감정 점수: {summary.get("customer_sentiment_score", "N/A")}/100'), normal_style))
            story.append(Paragraph(safe_text(f'고객 만족도 점수: {summary.get("customer_satisfaction_score", "N/A")}/100'), normal_style))
            
            characteristics = summary.get('customer_characteristics', {})
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(safe_text('고객 특성'), heading2_style))
            story.append(Paragraph(safe_text(f'언어: {characteristics.get("language", "N/A")}'), normal_style))
            story.append(Paragraph(safe_text(f'문화적 배경: {characteristics.get("cultural_hints", "N/A")}'), normal_style))
            story.append(Paragraph(safe_text(f'지역: {characteristics.get("region", "N/A")}'), normal_style))
            story.append(Paragraph(safe_text(f'소통 스타일: {characteristics.get("communication_style", "N/A")}'), normal_style))
            
            privacy = summary.get('privacy_info', {})
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(safe_text('개인정보 요약'), heading2_style))
            story.append(Paragraph(safe_text(f'이메일 제공: {"예" if privacy.get("has_email") else "아니오"}'), normal_style))
            story.append(Paragraph(safe_text(f'전화번호 제공: {"예" if privacy.get("has_phone") else "아니오"}'), normal_style))
            story.append(Paragraph(safe_text(f'주소 제공: {"예" if privacy.get("has_address") else "아니오"}'), normal_style))
            story.append(Paragraph(safe_text(f'지역 힌트: {privacy.get("region_hint", "N/A")}'), normal_style))
            
            story.append(Paragraph(safe_text(f'전체 요약: {summary.get("summary", "N/A")}'), normal_style))
        
        # 구분선
        if i < len(histories):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph('-' * 80, normal_style))
            story.append(Spacer(1, 0.2*inch))
    
    # PDF 빌드 (UTF-8 인코딩 명시, 폰트 서브셋팅 강화)
    try:
        # 폰트 등록 상태 확인 및 경고
        registered_fonts = pdfmetrics.getRegisteredFontNames()
        print(f"📋 등록된 폰트 목록: {registered_fonts}")
        
        if not korean_font_registered and not japanese_font_registered:
            print("⚠️ 경고: 한글/일본어 폰트가 등록되지 않았습니다. PDF에서 한글이 깨질 수 있습니다.")
            print("   가능한 해결 방법:")
            print("   1. Windows 폰트 폴더에 한글 폰트가 설치되어 있는지 확인")
            print("   2. 관리자 권한으로 실행")
            print("   3. 폰트 파일 경로 확인")
        else:
            if korean_font_registered:
                print(f"✅ 한글 폰트 등록 확인: KoreanFont in {registered_fonts}")
            if japanese_font_registered:
                print(f"✅ 일본어 폰트 등록 확인: JapaneseFont in {registered_fonts}")
        
        # PDF 빌드 실행
        doc.build(story)
        print(f"✅ PDF 생성 완료: {filepath}")
        
    except Exception as e:
        # 인코딩 오류가 발생하면 에러 메시지와 함께 재시도
        error_msg = str(e)
        print(f"⚠️ PDF 빌드 오류: {error_msg}")
        
        # 폰트 관련 오류인 경우 추가 정보 제공
        if 'font' in error_msg.lower() or 'encoding' in error_msg.lower():
            print("   폰트/인코딩 오류로 보입니다. 폰트 등록 상태를 확인하세요.")
            if korean_font_registered:
                print(f"   - 한글 폰트: 등록됨")
            if japanese_font_registered:
                print(f"   - 일본어 폰트: 등록됨")
        
        # 재시도 (단순 재시도는 위험할 수 있으므로 에러를 다시 발생시킴)
        raise Exception(f"PDF 생성 실패: {error_msg}")
    
    return filepath


# ========================================
# 6. RAG Helper (FAISS)
# ========================================
# RAG 관련 함수는 시뮬레이터와 무관하므로 기존 코드를 유지합니다.

def load_documents(files) -> List[Document]:
    docs: List[Document] = []
    for f in files:
        name = f.name
        lower = name.lower()
        if lower.endswith(".pdf"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.read())
            tmp.flush()
            tmp.close()
            loader = PyPDFLoader(tmp.name)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata["source"] = name
            docs.extend(file_docs)
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        elif lower.endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
        elif lower.endswith(".html") or lower.endswith(".htm"):
            text = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": name}))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embedding_model():
    if get_api_key("openai"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except:
            pass
    if get_api_key("gemini"):
        try:
            return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        except:
            pass
    return None


def get_embedding_function():
    """
    RAG 임베딩에 사용할 임베딩 모델을 결정합니다.
    API 키 유효성 순서: OpenAI (사용자 설정 시) -> Gemini -> NVIDIA -> HuggingFace (fallback)
    API 인증 오류 발생 시 다음 모델로 이동하도록 처리합니다.
    """

    # 1. OpenAI 임베딩 시도 (사용자가 유효한 키를 설정했을 경우)
    openai_key = get_api_key("openai")
    if openai_key:
        try:
            st.info("🔹 RAG: OpenAI Embedding 사용 중")
            return OpenAIEmbeddings(openai_api_key=openai_key)
        except Exception as e:
            st.warning(f"OpenAI 임베딩 실패 → Gemini로 Fallback: {e}")

    # 2. Gemini 임베딩 시도
    gemini_key = get_api_key("gemini")
    if IS_GEMINI_EMBEDDING_AVAILABLE and gemini_key:
        try:
            st.info("🔹 RAG: Gemini Embedding 사용 중")
            # ⭐ 수정: 모델 이름 형식을 'models/model-name'으로 수정
            return GoogleGenerativeAIEmbeddings(google_api_key=gemini_key, model="models/text-embedding-004")
        except Exception as e:
            st.warning(f"Gemini 임베딩 실패 → NVIDIA로 Fallback: {e}")

    # 3. NVIDIA 임베딩 시도
    nvidia_key = get_api_key("nvidia")
    if IS_NVIDIA_EMBEDDING_AVAILABLE and nvidia_key:
        try:
            st.info("🔹 RAG: NVIDIA Embedding 사용 중")
            # NIM 모델 사용 (실제 키가 유효해야 함)
            return NVIDIAEmbeddings(api_key=nvidia_key, model="ai-embed-qa-4")
        except Exception as e:
            st.warning(f"NVIDIA 임베딩 실패 → HuggingFace Fallback: {e}")

    # 4. HuggingFace Embeddings (Local Fallback)
    try:
        st.info("🔹 RAG: Local HuggingFace Embedding 사용 중")
        # 경량 모델 사용
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"최종 Fallback 임베딩 실패: {e}")

    st.error("❌ RAG 임베딩 실패: 사용 가능한 API Key가 없습니다.")
    return None


def build_rag_index(files):
    # 언어 키 안전하게 가져오기
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    if not files: return None, 0

    # 임베딩 함수를 시도하는 과정에서 에러 메시지가 발생할 수 있으므로 try-except로 감쌉니다.
    try:
        embeddings = get_embedding_function()
    except Exception as e:
        st.error(f"RAG 임베딩 함수 초기화 중 치명적인 오류 발생: {e}")
        return None, 0

    if embeddings is None:
        # 어떤 임베딩 모델도 초기화할 수 없음을 알림
        error_msg = L["rag_embed_error_none"]

        # 상세 오류 정보 구성 (실제 사용 가능한 임베딩 모델이 없는 경우)
        if not get_api_key("openai"):
            error_msg += f"\n- {L['rag_embed_error_openai']}"
        if not get_api_key("gemini"):
            error_msg += f"\n- {L['rag_embed_error_gemini']}"
        if not get_api_key("nvidia"):
            error_msg += f"\n- {L['rag_embed_error_nvidia']}"

        st.error(error_msg)
        return None, 0

    # 임베딩 객체 초기화 성공 후, 데이터 로드 및 분할
    docs = load_documents(files)
    if not docs: return None, 0

    chunks = split_documents(docs)
    if not chunks: return None, 0

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # 저장
        vectorstore.save_local(RAG_INDEX_DIR)
    except Exception as e:
        # API 인증 실패 등 실제 API 호출 오류 처리
        st.error(f"RAG 인덱스 생성 중 오류: {e}")
        return None, 0

    return vectorstore, len(chunks)


def load_rag_index():
    # RAG 인덱스 로드 시에도 유효한 임베딩 함수가 필요합니다.
    try:
        embeddings = get_embedding_function()
    except Exception:
        # get_embedding_function 내에서 에러 메시지를 처리하거나 스킵하므로 여기서는 조용히 처리
        return None

    if embeddings is None:
        return None

    try:
        # allow_dangerous_deserialization=True는 필수
        vs = FAISS.load_local(RAG_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        return None


def rag_answer(question: str, vectorstore: FAISS, lang_key: str) -> str:
    # RAG Answer는 LLM 클라이언트 라우팅을 사용하도록 수정
    llm_client, info = get_llm_client()
    if llm_client is None:
        # 언어 키 검증
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
        return LANG.get(lang_key, LANG["ko"]).get("simulation_no_key_warning", "API Key가 필요합니다.")

    # Langchain ChatOpenAI 대신 run_llm을 사용하기 위해 prompt를 직접 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content[:1500] for d in docs)

    # ⭐ RAG 다국어 인식 오류 해결: 답변 생성 모델에게 질문 언어로 일관되게 답하도록 강력히 지시
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(lang_key, "English")

    prompt = (
            f"You are a helpful AI tutor. Answer the question using ONLY the provided context.\n"
            f"The answer MUST be STRICTLY in {lang_name}, which is the language of the question.\n"
            f"If you cannot find the answer in the context, say you don't know in {lang_name}.\n"
            f"Note: The context may be in a different language, but you must still answer in {lang_name}.\n\n"
            "Question:\n" + question + "\n\n"
                                       "Context:\n" + context + "\n\n"
                                                                f"Answer (in {lang_name}):"
    )
    return run_llm(prompt)


# ========================================
# 7. LSTM Helper (간단 Mock + 시각화)
# ========================================

def load_or_train_lstm():
    # 실제 LSTM 대신 랜덤 + sin 파형 기반 Mock
    np.random.seed(42)
    n_points = 50
    ts = 60 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_points)) + np.random.normal(0, 5, n_points)
    ts = np.clip(ts, 50, 100).astype(np.float32)
    return ts





# ========================================
# 8. LLM (ChatOpenAI) for Simulator / Content
# (RAG와 동일하게 run_llm으로 통합)
# ========================================

# ConversationChain 대신 run_llm을 사용하여 메모리 기능을 수동으로 구현
# st.session_state.simulator_memory는 유지하여 대화 기록을 관리합니다.

def get_chat_history_for_prompt(include_attachment=False):
    """메모리에서 대화 기록을 추출하여 프롬프트에 사용할 문자열 형태로 반환 (채팅용)"""
    history_str = ""
    for msg in st.session_state.simulator_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "customer" or role == "customer_rebuttal":
            history_str += f"Customer: {content}\n"
        elif role == "agent_response":
            history_str += f"Agent: {content}\n"
        # supervisor 메시지는 LLM에 전달하지 않아 역할 혼동 방지
    return history_str


def generate_customer_reaction(current_lang_key: str, is_call: bool = False) -> str:
    """
    고객의 다음 반응을 생성하는 LLM 호출 (채팅 전용)
    **수정 사항:** 에이전트 정보 요청 시 필수 정보 (주문번호, eSIM, 자녀 만 나이, 취소 사유) 제공 의무를 강화함.
    """
    history_text = get_chat_history_for_prompt()
    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG.get(current_lang_key, LANG["ko"])

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        # LLM에게 첨부 파일 컨텍스트를 제공하되, 에이전트에게 반복하지 않도록 주의
        attachment_context = f"[INITIAL ATTACHMENT CONTEXT (for customer reference only, do not repeat to agent)]\n{attachment_context}\n\n"
    else:
        attachment_context = ""

    next_prompt = f"""
{attachment_context}
You are now ROLEPLAYING as the CUSTOMER.

Read the following conversation and respond naturally in {lang_name}.

Conversation so far:
{history_text}

RULES:
1. You are only the customer. Do not write as the agent.
2. **[CRITICAL: Mandatory Information Submission for Problem Resolution]** If the agent requested any of the following critical information, you MUST provide it:
    - Order/Booking Number (e.g., ABC123, 123456)
    - eSIM related details (e.g., Host device compatibility, local status/location, time of activation)
    - Child-related product details (e.g., Child's Date of Birth or Current Age)
    - Exception/Refund Reason (e.g., flight cancellation/delay, illness, local natural disaster)
    - **If you are a difficult customer and the agent requests this information, you MUST still provide it, but you may express frustration or impatience while doing so.**
3. **[Crucial Rule for Repetition/New Inquiry]** After the agent has provided an attempt at a solution or answer:
    - If you are still confused or the problem is not fully solved, you MUST state the remaining confusion/problem clearly and briefly. DO NOT REPEAT THE INITIAL QUERY. Focus only on the unresolved aspect or the new inquiry.
4. **[CRITICAL: Solution Acknowledgment]** If the agent provided a clear and accurate solution/confirmation:
    - You MUST respond with appreciation and satisfaction, like "{L_local['customer_positive_response']}" or similar positive acknowledgment. This applies even if you are a difficult customer.
5. If the agent's LAST message was the closing confirmation: "{L_local['customer_closing_confirm']}"
    - If you have NO additional questions: You MUST reply with "{L_local['customer_no_more_inquiries']}".
    - If you DO have additional questions: You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS IMMEDIATELY. DO NOT just repeat that you have an additional question.
6. Do NOT repeat your initial message or previous responses unless necessary.
7. Output ONLY the customer's next message.
"""
    try:
        reaction = run_llm(next_prompt)

        # ⭐ LLM이 응답했지만 내용이 너무 짧거나 비어있을 경우, 긍정 종료 문구를 반환
        if not reaction or len(reaction.strip()) < 5:
            print("LLM returned insufficient response. Using positive closing fallback.")
            return L_local['customer_positive_response']

        return reaction.strip()
    except Exception as e:
        # ⭐ LLM 호출 자체에서 오류 발생 시 (API 키, 할당량) 긍정 종료 문구를 강제 반환
        print(f"LLM Customer Reaction generation failed: {e}. Falling back to positive closing.")
        return L_local['customer_positive_response']  # 강제 안전장치


def summarize_history_with_ai(current_lang_key: str) -> str:
    """전화 통화 로그를 정리하여 LLM에 전달하고 요약 텍스트를 받는 함수."""
    # 전화 로그는 'phone_exchange' 역할을 가지거나, 'initial_query'에 포함되어 있음

    # 1. 로그 추출
    conversation_text = ""
    initial_query = st.session_state.get("call_initial_query", "N/A")
    website_url = st.session_state.get("call_website_url", "").strip()
    if initial_query and initial_query != "N/A":
        conversation_text += f"Initial Query: {initial_query}\n"
    if website_url:
        conversation_text += f"Website URL: {website_url}\n"

    for msg in st.session_state.simulator_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "phone_exchange":
            # phone_exchange는 "Agent: ... | Customer: ..." 형태로 이미 정리되어 있음
            conversation_text += f"{content}\n"

    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    summary_prompt = f"""
You are an AI Analyst specialized in summarizing customer phone calls. 
Analyze the full conversation log below, identify the main issue, the steps taken by the agent, and the customer's sentiment.

Provide a concise, easy-to-read summary of the key exchange STRICTLY in {lang_name}.

--- Conversation Log ---
{conversation_text}
---

Summary:
"""
    if not st.session_state.is_llm_ready:
        return "LLM Key가 없어 요약 생성이 불가합니다."

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ AI 요약 생성 오류: {e}"


def generate_customer_reaction_for_call(current_lang_key: str, last_agent_response: str) -> str:
    """전화 시뮬레이터 전용 고객 반응 생성 (마지막 에이전트 응답 중심)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]
    
    # ⭐ 추가: 고객 성별 및 감정 상태 가져오기
    customer_gender = st.session_state.customer_avatar.get("gender", "male")
    customer_emotion = st.session_state.customer_avatar.get("state", "NEUTRAL")
    
    # 감정 상태에 따른 톤 설정
    emotion_tone_map = {
        "HAPPY": "friendly, positive, and satisfied",
        "ASKING": "slightly frustrated, questioning, and seeking clarification",
        "ANGRY": "angry, frustrated, and demanding",
        "SAD": "sad, depressed, and disappointed",
        "NEUTRAL": "neutral, calm, and polite"
    }
    emotion_tone = emotion_tone_map.get(customer_emotion, "neutral, calm, and polite")
    
    gender_pronoun = "she" if customer_gender == "female" else "he"
    
    # ⭐ 추가: 에이전트가 종료 확인 질문을 했는지 확인
    closing_msg = L_local['customer_closing_confirm']
    is_closing_question = closing_msg in last_agent_response or any(
        phrase in last_agent_response.lower() 
        for phrase in ["다른 문의", "추가 문의", "다른 도움", "anything else", "other questions"]
    )
    
    # ⭐ 수정: 초기 문의를 완전히 제거하고 마지막 에이전트 응답에만 집중
    # 최근 대화 이력만 추출 (최대 3-4개 교환만)
    recent_exchanges = []
    for msg in reversed(st.session_state.simulator_messages):  # 역순으로 최근 것부터
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "phone_exchange":
            recent_exchanges.insert(0, content)  # 앞에 삽입하여 순서 유지
            if len(recent_exchanges) >= 3:  # 최근 3개만
                break
        elif role == "agent":
            recent_exchanges.insert(0, f"Agent: {content}")
            if len(recent_exchanges) >= 3:
                break
    
    # 최근 대화 이력 (있는 경우만)
    recent_history = "\n".join(recent_exchanges) if recent_exchanges else "(No previous exchanges)"
    
    website_url = st.session_state.get("call_website_url", "").strip()
    website_context = f"\nWebsite URL: {website_url}" if website_url else ""
    
    # ⭐ 수정: 마지막 에이전트 응답만 강조 (초기 문의 완전 제거)
    last_agent_text = last_agent_response.strip() if last_agent_response else "None"
    
    history_text = f"""[Recent Conversation Context - For Reference Only]
{recent_history}{website_context}

═══════════════════════════════════════════════════════════════════
🎯 YOUR TASK: Respond ONLY to the Agent's message below
═══════════════════════════════════════════════════════════════════

Agent just said: "{last_agent_text}"

═══════════════════════════════════════════════════════════════════
IMPORTANT: 
- Respond DIRECTLY to what the agent JUST SAID above
- DO NOT repeat your initial query
- DO NOT refer to old conversation unless agent asks
- Keep your response short and conversational
- Your emotional state: {customer_emotion} - respond with {emotion_tone} tone
═══════════════════════════════════════════════════════════════════"""

    # ⭐ 추가: 종료 확인 질문에 대한 특별 처리
    if is_closing_question:
        call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

{history_text}

The agent just asked: "{last_agent_text}"

CRITICAL RULES FOR CLOSING CONFIRMATION:
1. If you have NO additional questions and the conversation is resolved:
   - You MUST reply with: "{L_local['customer_no_more_inquiries']}"
2. If you DO have additional questions or the issue is NOT fully resolved:
   - You MUST reply with: "{L_local['customer_has_additional_inquiries']}" AND immediately state your additional question
3. Your response MUST be ONLY one of the two options above, in {lang_name}.
4. Output ONLY the customer's response (must be one of the two rule options).

Your response (respond to the closing confirmation question):
"""
    else:
        call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

{history_text}

RULES:
1. Respond ONLY to what the agent JUST SAID: "{last_agent_text}"
2. If agent asked a question → Answer it
3. If agent requested information → Provide it
4. If agent gave a solution → Acknowledge based on your emotional state ({customer_emotion})
5. Keep your response short (1-2 sentences max)
6. DO NOT repeat your initial query
7. DO NOT mention old conversation
8. IMPORTANT: Match your tone to your emotional state ({customer_emotion}) - be {emotion_tone}

Your response (respond ONLY to the agent's message above, with {emotion_tone} tone):
"""
    try:
        reaction = run_llm(call_prompt)
        reaction_text = reaction.strip()
        
        # ⭐ 추가: 종료 확인 질문에 대한 응답 검증 및 강제 적용
        if is_closing_question:
            if L_local['customer_no_more_inquiries'] in reaction_text:
                return L_local['customer_no_more_inquiries']
            elif L_local['customer_has_additional_inquiries'] in reaction_text:
                return reaction_text  # 추가 문의 내용 포함 가능
            else:
                # LLM이 규칙을 따르지 않으면, 대화가 해결된 것으로 가정하고 종료 응답 반환
                return L_local['customer_no_more_inquiries']
        
        return reaction_text
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def generate_customer_reaction_for_first_greeting(current_lang_key: str, agent_greeting: str, initial_query: str) -> str:
    """전화 시뮬레이터 전용: 첫 인사말에 대한 고객의 맞춤형 반응 생성 (초기 문의 고려)"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG[current_lang_key]
    
    # ⭐ 추가: 고객 성별 및 감정 상태 가져오기
    customer_gender = st.session_state.customer_avatar.get("gender", "male")
    customer_emotion = st.session_state.customer_avatar.get("state", "NEUTRAL")
    
    # 감정 상태에 따른 톤 설정
    emotion_tone_map = {
        "HAPPY": "friendly, positive, and satisfied",
        "ASKING": "slightly frustrated, questioning, and seeking clarification",
        "ANGRY": "angry, frustrated, and demanding",
        "SAD": "sad, depressed, and disappointed",
        "NEUTRAL": "neutral, calm, and polite"
    }
    emotion_tone = emotion_tone_map.get(customer_emotion, "neutral, calm, and polite")
    
    website_url = st.session_state.get("call_website_url", "").strip()
    website_context = f"\nWebsite URL: {website_url}" if website_url else ""
    
    agent_greeting_text = agent_greeting.strip() if agent_greeting else "None"
    initial_query_text = initial_query.strip() if initial_query else "None"
    
    call_prompt = f"""
You are a CUSTOMER in a phone call. You are a {customer_gender} customer. Respond naturally in {lang_name}.

Your current emotional state: {customer_emotion}
Your response tone should be: {emotion_tone}

═══════════════════════════════════════════════════════════════════
🎯 YOUR SITUATION:
═══════════════════════════════════════════════════════════════════

You called because: "{initial_query_text}"

The agent just greeted you and said: "{agent_greeting_text}"
{website_context}

═══════════════════════════════════════════════════════════════════
YOUR TASK: Respond to the agent's greeting in a way that:
═══════════════════════════════════════════════════════════════════

1. Acknowledge the agent's greeting naturally
2. Briefly mention your inquiry/concern: "{initial_query_text}"
3. Show that you're ready to discuss your issue
4. Keep it conversational and natural (1-2 sentences max)
5. DO NOT be overly formal - this is a phone call, be natural
6. IMPORTANT: Match your tone to your emotional state ({customer_emotion}) - be {emotion_tone}

Example good responses (adjust tone based on your emotional state):
- If {customer_emotion}: [Respond with {emotion_tone} tone]
- "Hello, thank you. I'm calling because [brief mention of issue]..."
- "Hi, yes. I need help with [your issue]..."
- "Thank you. I have a question about [your issue]..."

Your response (respond naturally to the greeting and briefly mention your inquiry, with {emotion_tone} tone):
"""
    try:
        reaction = run_llm(call_prompt)
        return reaction.strip()
    except Exception as e:
        return f"❌ 고객 반응 생성 오류: {e}"


def summarize_history_for_call(call_logs: List[Dict[str, str]], initial_query: str, current_lang_key: str) -> str:
    """전화 통화 로그와 초기 문의를 바탕으로 요약본을 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    # 로그 재구성 (phone_exchange 역할만 사용)
    full_log_text = f"--- Initial Customer Query ---\nCustomer: {initial_query}\n"
    for log in call_logs:
        if log["role"] == "phone_exchange":
            full_log_text += f"{log['content']}\n"
        elif log["role"] == "agent" and "content" in log:
            # 최초 에이전트 인사말은 여기에 포함
            full_log_text += f"Agent (Greeting): {log['content']}\n"

    summary_prompt = f"""
You are an AI Supervisor. Analyze the following telephone support conversation log.
Provide a concise, neutral summary of the key issue, the steps taken by the agent, and the final outcome.
The summary MUST be STRICTLY in {lang_name}.

--- Conversation Log ---
{full_log_text}
---

Summary:
"""
    if not st.session_state.is_llm_ready:
        return f"❌ LLM Key is missing. Cannot generate summary. Log length: {len(full_log_text.splitlines())}"

    try:
        summary = run_llm(summary_prompt)
        return summary.strip()
    except Exception as e:
        return f"❌ Summary Generation Error: {e}"


def generate_customer_closing_response(current_lang_key: str) -> str:
    """에이전트의 마지막 확인 질문에 대한 고객의 최종 답변 생성 (채팅용)"""
    history_text = get_chat_history_for_prompt()
    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]
    L_local = LANG.get(current_lang_key, LANG["ko"])  # ⭐ 수정: 함수 내에서 사용할 언어 팩

    # 마지막 메시지가 에이전트의 종료 확인 메시지인지 확인 (프롬프트에 포함)
    closing_msg = L_local['customer_closing_confirm']

    # 첨부 파일 컨텍스트 추가
    attachment_context = st.session_state.sim_attachment_context_for_llm
    if attachment_context:
        attachment_context = f"[INITIAL ATTACHMENT CONTEXT (for customer reference only, do not repeat to agent)]\n{attachment_context}\n\n"
    else:
        attachment_context = ""

    final_prompt = f"""
{attachment_context}
You are now ROLEPLAYING as the CUSTOMER.

The agent's final message was the closing confirmation: "{closing_msg}".
You MUST respond to this confirmation based on the overall conversation.

Conversation history:
{history_text}

RULES:
1. If the conversation seems resolved and you have NO additional questions:
    - You MUST reply with "{L_local['customer_no_more_inquiries']}".
2. If the conversation is NOT fully resolved and you DO have additional questions (or the agent provided a cancellation denial that you want to appeal):
    - You MUST reply with "{L_local['customer_has_additional_inquiries']}" AND MUST FOLLOW UP WITH THE NEW INQUIRY DETAILS. DO NOT just repeat that you have an additional question.
3. Your reply MUST be ONLY one of the two options above, in {lang_name}.
4. Output ONLY the customer's next message (must be one of the two rule options).
"""
    try:
        reaction = run_llm(final_prompt)
        # LLM의 출력이 규칙을 따르지 않을 경우를 대비하여 강제 적용
        reaction_text = reaction.strip()
        # "추가 문의 사항도 있습니다"가 포함되어 있으면 그대로 반환 (상세 내용 포함 가정)
        if L_local['customer_no_more_inquiries'] in reaction_text:
            return L_local['customer_no_more_inquiries']
        elif L_local['customer_has_additional_inquiries'] in reaction_text:
            return reaction_text
        else:
            # LLM이 규칙을 어겼을 경우, "추가 문의 사항이 있다"고 가정하고 에이전트 턴으로 넘김
            return L_local['customer_has_additional_inquiries']
    except Exception as e:
        st.error(f"고객 최종 반응 생성 오류: {e}")
        return L_local['customer_has_additional_inquiries']  # 오류 시 에이전트 턴으로 유도


# ----------------------------------------
# Initial Advice/Draft Generation (이관 후 재사용) (요청 4 반영)
# ----------------------------------------
def generate_agent_first_greeting(lang_key: str, initial_query: str) -> str:
    """전화 통화 시작 시 에이전트의 첫 인사말을 생성 (임시 함수)"""
    # 언어 키 검증
    if lang_key not in ["ko", "en", "ja"]:
        lang_key = st.session_state.get("language", "ko")
        if lang_key not in ["ko", "en", "ja"]:
            lang_key = "ko"
    L_local = LANG.get(lang_key, LANG["ko"])
    # 문의 내용의 첫 10자만 사용 (too long)
    topic = initial_query.strip()[:15].replace('\n', ' ')
    if len(initial_query.strip()) > 15:
        topic += "..."

    if lang_key == 'ko':
        return f"안녕하세요, {topic} 관련 문의 주셨죠? 상담원 000입니다. 무엇을 도와드릴까요?"
    elif lang_key == 'en':
        return f"Hello, thank you for calling. I see you're calling about {topic}. My name is 000. How may I help you today?"
    elif lang_key == 'ja':
        return f"お電話ありがとうございます。{topic}の件ですね。担当の000と申します。どのようなご用件でしょうか?"
    return "Hello, how may I help you?"


def detect_text_language(text: str) -> str:
    """
    텍스트의 언어를 자동 감지합니다.
    Returns: "ko", "en", "ja" 중 하나 (기본값: "ko")
    """
    if not text or not text.strip():
        return "ko"  # 기본값
    
    try:
        # 간단한 휴리스틱: 일본어 문자(히라가나, 가타카나, 한자)가 많이 포함되어 있으면 일본어
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF')
        if japanese_chars > len(text) * 0.1:  # 10% 이상 일본어 문자
            return "ja"
        
        # 영어 문자 비율이 높으면 영어
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        if english_chars > len(text) * 0.7:  # 70% 이상 영어 문자
            return "en"
        
        # LLM을 사용한 정확한 언어 감지 시도 (오류 발생 시 무시하고 휴리스틱 결과 사용)
        if st.session_state.is_llm_ready:
            try:
                detection_prompt = f"""Detect the language of the following text. Respond with ONLY one word: "ko" (Korean), "en" (English), or "ja" (Japanese).

Text: {text[:200]}

Language:"""
                detected = run_llm(detection_prompt).strip().lower()
                # 오류 메시지가 아닌 경우에만 사용
                if detected and detected not in ["❌", "error", "failed"] and detected in ["ko", "en", "ja"]:
                    return detected
            except Exception as e:
                # LLM 호출 실패 시 휴리스틱 결과 사용
                print(f"Language detection LLM call failed: {e}")
                pass
    except Exception as e:
        # 전체 함수에서 예외 발생 시 기본값 반환
        print(f"Language detection error: {e}")
        return "ko"
    
    # 기본값: 한국어
    return "ko"


def analyze_customer_profile(customer_query: str, current_lang_key: str = None) -> Dict[str, Any]:
    """신규 고객의 문의사항과 말투를 분석하여 고객성향 점수를 실시간으로 계산 (요청 4)"""
    # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
    try:
        detected_lang = detect_text_language(customer_query)
    except Exception as e:
        print(f"Language detection failed in analyze_customer_profile: {e}")
        detected_lang = "ko"  # 기본값 사용
    
    # current_lang_key가 제공되지 않으면 감지된 언어 사용
    lang_key_to_use = current_lang_key if current_lang_key else detected_lang
    # lang_key_to_use가 유효한지 확인
    if lang_key_to_use not in ["ko", "en", "ja"]:
        lang_key_to_use = "ko"  # 기본값으로 폴백
    
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[lang_key_to_use]

    analysis_prompt = f"""
You are an AI analyst analyzing a customer's inquiry to determine their profile and sentiment.

Analyze the following customer inquiry and provide a structured analysis in JSON format (ONLY JSON, no markdown).

Analyze:
1. Customer gender (male/female/unknown - analyze based on name, language patterns, or cultural hints)
2. Customer sentiment score (0-100, where 0=very negative/angry, 50=neutral, 100=very positive/happy)
3. Communication style (formal/casual, brief/detailed, polite/direct)
4. Urgency level (low/medium/high)
5. Customer type prediction (normal/difficult/very_dissatisfied)
6. Language and cultural hints (if any)
7. Key concerns or pain points

Output format (JSON only):
{{
  "gender": "male",
  "sentiment_score": 45,
  "communication_style": "brief, direct, slightly frustrated",
  "urgency_level": "high",
  "predicted_customer_type": "difficult",
  "cultural_hints": "unknown",
  "key_concerns": ["issue 1", "issue 2"],
  "tone_analysis": "brief description of tone"
}}

Customer Inquiry:
{customer_query}

JSON Output:
"""

    if not st.session_state.is_llm_ready:
        return {
            "gender": "unknown",
            "sentiment_score": 50,
            "communication_style": "unknown",
            "urgency_level": "medium",
            "predicted_customer_type": "normal",
            "cultural_hints": "unknown",
            "key_concerns": [],
            "tone_analysis": "Unable to analyze"
        }

    try:
        analysis_text = run_llm(analysis_prompt).strip()
        # JSON 추출
        if "```json" in analysis_text:
            analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
        elif "```" in analysis_text:
            analysis_text = analysis_text.split("```")[1].split("```")[0].strip()

        import json
        analysis_data = json.loads(analysis_text)
        return analysis_data
    except Exception as e:
        return {
            "gender": "unknown",
            "sentiment_score": 50,
            "communication_style": "unknown",
            "urgency_level": "medium",
            "predicted_customer_type": "normal",
            "cultural_hints": "unknown",
            "key_concerns": [],
            "tone_analysis": f"Analysis error: {str(e)}"
        }


def find_similar_cases(customer_query: str, customer_profile: Dict[str, Any], current_lang_key: str,
                       limit: int = 5) -> List[Dict[str, Any]]:
    """저장된 요약 데이터에서 유사한 케이스를 찾아 반환 (요청 4)"""
    histories = load_simulation_histories_local(current_lang_key)

    if not histories:
        return []

    # 요약 데이터가 있는 케이스만 필터링
    cases_with_summary = [
        h for h in histories
        if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
           and not h.get("is_call", False)  # 전화 이력 제외
    ]

    if not cases_with_summary:
        return []

    # 유사도 계산 (간단한 키워드 매칭 + 점수 유사도)
    similar_cases = []
    query_lower = customer_query.lower()
    customer_sentiment = customer_profile.get("sentiment_score", 50)
    customer_style = customer_profile.get("communication_style", "")

    for case in cases_with_summary:
        summary = case.get("summary", {})
        main_inquiry = summary.get("main_inquiry", "").lower()
        case_sentiment = summary.get("customer_sentiment_score", 50)
        case_satisfaction = summary.get("customer_satisfaction_score", 50)

        # 유사도 점수 계산
        similarity_score = 0

        # 1. 문의 내용 유사도 (키워드 매칭)
        query_words = set(query_lower.split())
        inquiry_words = set(main_inquiry.split())
        if query_words and inquiry_words:
            word_overlap = len(query_words & inquiry_words) / len(query_words | inquiry_words)
            similarity_score += word_overlap * 40

        # 2. 감정 점수 유사도
        sentiment_diff = abs(customer_sentiment - case_sentiment)
        sentiment_similarity = max(0, 1 - (sentiment_diff / 100)) * 30
        similarity_score += sentiment_similarity

        # 3. 만족도 점수 (높을수록 좋은 케이스)
        satisfaction_bonus = (case_satisfaction / 100) * 30
        similarity_score += satisfaction_bonus

        if similarity_score > 30:  # 최소 유사도 임계값
            similar_cases.append({
                "case": case,
                "similarity_score": similarity_score,
                "summary": summary
            })

    # 유사도 순으로 정렬
    similar_cases.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similar_cases[:limit]


def visualize_customer_profile_scores(customer_profile: Dict[str, Any], current_lang_key: str):
    """고객 프로필 점수를 시각화 (감정 점수, 긴급도)"""
    if not IS_PLOTLY_AVAILABLE:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    sentiment_score = customer_profile.get("sentiment_score", 50)
    urgency_map = {"low": 25, "medium": 50, "high": 75}
    urgency_level = customer_profile.get("urgency_level", "medium")
    urgency_score = urgency_map.get(urgency_level.lower(), 50)

    # 게이지 차트 생성
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=(
            L.get("sentiment_score_label", "고객 감정 점수"),
            L.get("urgency_score_label", "긴급도 점수")
        )
    )

    # 감정 점수 게이지
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': L.get("sentiment_score_label", "감정 점수")},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )

    # 긴급도 점수 게이지
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=urgency_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': L.get("urgency_score_label", "긴급도")},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
            }
        ),
        row=1, col=2
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def visualize_similarity_cases(similar_cases: List[Dict[str, Any]], current_lang_key: str):
    """유사 케이스 추천을 시각화"""
    if not IS_PLOTLY_AVAILABLE or not similar_cases:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    case_labels = []
    similarity_scores = []
    sentiment_scores = []
    satisfaction_scores = []

    for idx, similar_case in enumerate(similar_cases, 1):
        summary = similar_case["summary"]
        similarity = similar_case["similarity_score"]
        case_labels.append(f"Case {idx}")
        similarity_scores.append(similarity)
        sentiment_scores.append(summary.get("customer_sentiment_score", 50))
        satisfaction_scores.append(summary.get("customer_satisfaction_score", 50))

    # 유사도 차트
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            L.get("similarity_chart_title", "유사 케이스 유사도"),
            L.get("scores_comparison_title",
                  "감정 및 만족도 점수 비교")
        ),
        vertical_spacing=0.15
    )

    # 유사도 바 차트
    fig.add_trace(
        go.Bar(
            x=case_labels,
            y=similarity_scores,
            name=L.get("similarity_score_label", "유사도"),
            marker_color='lightblue',
            text=[f"{s:.1f}%" for s in similarity_scores],
            textposition='outside'
        ),
        row=1, col=1
    )

    # 감정 및 만족도 점수 비교
    fig.add_trace(
        go.Bar(
            x=case_labels,
            y=sentiment_scores,
            name=L.get("sentiment_score_label", "감정 점수"),
            marker_color='lightcoral'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=case_labels,
            y=satisfaction_scores,
            name=L.get("satisfaction_score_label", "만족도"),
            marker_color='lightgreen'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
        barmode='group'
    )
    fig.update_yaxes(title_text="점수", row=2, col=1)
    fig.update_yaxes(title_text="유사도 (%)", row=1, col=1)

    return fig


def visualize_case_trends(histories: List[Dict[str, Any]], current_lang_key: str):
    """과거 성공 사례 트렌드를 시각화"""
    if not IS_PLOTLY_AVAILABLE or not histories:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    # 요약 데이터가 있는 케이스만 필터링
    cases_with_summary = [
        h for h in histories
        if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
    ]

    if not cases_with_summary:
        return None

    # 날짜별로 정렬
    cases_with_summary.sort(key=lambda x: x.get("timestamp", ""))

    dates = []
    sentiment_scores = []
    satisfaction_scores = []

    for case in cases_with_summary:
        summary = case.get("summary", {})
        timestamp = case.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp)
            dates.append(dt)
            sentiment_scores.append(summary.get("customer_sentiment_score", 50))
            satisfaction_scores.append(summary.get("customer_satisfaction_score", 50))
        except Exception:
            continue

    if not dates:
        return None

    # 트렌드 라인 차트
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=sentiment_scores,
        mode='lines+markers',
        name=L.get("sentiment_trend_label", "감정 점수 추이"),
        line=dict(color='lightcoral', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=satisfaction_scores,
        mode='lines+markers',
        name=L.get("satisfaction_trend_label", "만족도 점수 추이"),
        line=dict(color='lightgreen', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=L.get("case_trends_title", "과거 케이스 점수 추이"),
        xaxis_title=L.get("date_label", "날짜"),
        yaxis_title=L.get("score_label", "점수 (0-100)"),
        height=400,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def visualize_customer_characteristics(summary: Dict[str, Any], current_lang_key: str):
    """고객 특성을 시각화 (언어, 문화권, 지역 등)"""
    if not IS_PLOTLY_AVAILABLE or not summary:
        return None

    # 언어 키 검증
    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    characteristics = summary.get("customer_characteristics", {})
    privacy_info = summary.get("privacy_info", {})

    # 특성 데이터 준비
    labels = []
    values = []

    # 언어 정보
    language = characteristics.get("language", "unknown")
    if language != "unknown":
        labels.append(L.get("language_label", "언어"))
        lang_map = {"ko": "한국어", "en": "English", "ja": "日本語"}
        values.append(lang_map.get(language, language))

    # 개인정보 제공 여부
    if privacy_info.get("has_email"):
        labels.append(L.get("email_provided_label", "이메일 제공"))
        values.append("Yes")
    if privacy_info.get("has_phone"):
        labels.append(L.get("phone_provided_label", "전화번호 제공"))
        values.append("Yes")

    # 지역 정보
    region = privacy_info.get("region_hint", characteristics.get("region", "unknown"))
    if region != "unknown":
        labels.append(L.get("region_label", "지역"))
        values.append(region)

    if not labels:
        return None

    # 파이 차트
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=[1] * len(labels),
        hole=0.4,
        marker_colors=px.colors.qualitative.Set3[:len(labels)]
    )])

    fig.update_layout(
        title=L.get("customer_characteristics_title",
                    "고객 특성 분포"),
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def generate_guideline_from_past_cases(customer_query: str, customer_profile: Dict[str, Any],
                                       similar_cases: List[Dict[str, Any]], current_lang_key: str) -> str:
    """과거 유사 케이스의 성공적인 해결 방법을 바탕으로 가이드라인 생성"""
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[current_lang_key]

    if not similar_cases:
        return ""

    # 유사 케이스 요약
    past_cases_text = ""
    for idx, similar_case in enumerate(similar_cases, 1):
        case = similar_case["case"]
        summary = similar_case["summary"]
        similarity = similar_case["similarity_score"]

        past_cases_text += f"""
[Case {idx}] (Similarity: {similarity:.1f}%)
- Inquiry: {summary.get('main_inquiry', 'N/A')}
- Customer Sentiment: {summary.get('customer_sentiment_score', 50)}/100
- Customer Satisfaction: {summary.get('customer_satisfaction_score', 50)}/100
- Key Responses: {', '.join(summary.get('key_responses', [])[:3])}
- Summary: {summary.get('summary', 'N/A')[:200]}
"""

    guideline_prompt = f"""
You are an AI Customer Support Supervisor analyzing past successful cases to provide guidance.

Based on the following similar past cases and their successful resolution strategies, provide actionable guidelines for handling the current customer inquiry.

Current Customer Inquiry:
{customer_query}

Current Customer Profile:
- Gender: {customer_profile.get('gender', 'unknown')}
- Sentiment Score: {customer_profile.get('sentiment_score', 50)}/100
- Communication Style: {customer_profile.get('communication_style', 'unknown')}
- Urgency: {customer_profile.get('urgency_level', 'medium')}
- Predicted Type: {customer_profile.get('predicted_customer_type', 'normal')}

Similar Past Cases (Successful Resolutions):
{past_cases_text}

Provide a concise guideline in {lang_name} that:
1. Identifies what worked well in similar past cases
2. Suggests specific approaches based on successful patterns
3. Warns about potential pitfalls based on past experiences
4. Recommends response strategies that led to high customer satisfaction

Guideline (in {lang_name}):
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        guideline = run_llm(guideline_prompt).strip()
        return guideline
    except Exception as e:
        return f"가이드라인 생성 오류: {str(e)}"


def _generate_initial_advice(customer_query, customer_type_display, customer_email, customer_phone, current_lang_key,
                             customer_attachment_file):
    """Supervisor 가이드라인과 초안을 생성하는 함수 (저장된 데이터 활용)"""
    # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
    try:
        detected_lang = detect_text_language(customer_query)
    except Exception as e:
        print(f"Language detection failed in _generate_initial_advice: {e}")
        detected_lang = current_lang_key if current_lang_key else "ko"
    
    # 감지된 언어를 우선 사용하되, current_lang_key가 명시적으로 제공되면 그것을 사용
    lang_key_to_use = detected_lang if detected_lang else current_lang_key
    # lang_key_to_use가 유효한지 확인
    if lang_key_to_use not in ["ko", "en", "ja"]:
        lang_key_to_use = current_lang_key if current_lang_key else "ko"
    
    # 언어 키 검증
    if lang_key_to_use not in ["ko", "en", "ja"]:
        lang_key_to_use = st.session_state.get("language", "ko")
        if lang_key_to_use not in ["ko", "en", "ja"]:
            lang_key_to_use = "ko"
    L = LANG.get(lang_key_to_use, LANG["ko"])
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[lang_key_to_use]

    contact_info_block = ""
    if customer_email or customer_phone:
        contact_info_block = (
            f"\n\n[Customer contact info for reference (DO NOT use these in your reply draft!)]"
            f"\n- Email: {customer_email or 'N/A'}"
            f"\n- Phone: {customer_phone or 'N/A'}"
        )

    attachment_block = ""
    if customer_attachment_file:
        file_name = customer_attachment_file.name
        attachment_block = f"\n\n[ATTACHMENT NOTE]: {L['attachment_info_llm'].format(filename=file_name)}"

    # 고객 프로필 분석 (감지된 언어 사용)
    customer_profile = analyze_customer_profile(customer_query, lang_key_to_use)

    # 유사 케이스 찾기 (감지된 언어 사용)
    similar_cases = find_similar_cases(customer_query, customer_profile, lang_key_to_use, limit=5)

    # 과거 케이스 기반 가이드라인 생성 (감지된 언어 사용)
    past_cases_guideline = ""
    if similar_cases:
        past_cases_guideline = generate_guideline_from_past_cases(
            customer_query, customer_profile, similar_cases, lang_key_to_use
        )

    # 고객 프로필 정보
    gender_display = customer_profile.get('gender', 'unknown')
    profile_block = f"""
[Customer Profile Analysis]
- Gender: {gender_display}
- Sentiment Score: {customer_profile.get('sentiment_score', 50)}/100
- Communication Style: {customer_profile.get('communication_style', 'unknown')}
- Urgency Level: {customer_profile.get('urgency_level', 'medium')}
- Predicted Type: {customer_profile.get('predicted_customer_type', 'normal')}
- Key Concerns: {', '.join(customer_profile.get('key_concerns', []))}
- Tone: {customer_profile.get('tone_analysis', 'unknown')}
"""

    # 과거 케이스 기반 가이드라인 블록
    past_cases_block = ""
    if past_cases_guideline:
        past_cases_block = f"""
[Guidelines Based on {len(similar_cases)} Similar Past Cases]
{past_cases_guideline}
"""
    elif similar_cases:
        past_cases_block = f"""
[Note: Found {len(similar_cases)} similar past cases, but unable to generate detailed guidelines.
Consider reviewing past cases manually for patterns.]
"""

    # Output ALL text (guidelines and draft) STRICTLY in {lang_name}. <--- 강력한 언어 강제 지시
    initial_prompt = f"""
Output ALL text (guidelines and draft) STRICTLY in {lang_name}.

You are an AI Customer Support Supervisor. Your role is to analyze the following customer inquiry
from a **{st.session_state.customer_type_sim_select}** and provide:

1) A detailed **response guideline for the human agent** (step-by-step).
2) A **ready-to-send draft reply** in {lang_name}.

[FORMAT]
- Use the exact markdown headers:
  - "### {L['simulation_advice_header']}"
  - "### {L['simulation_draft_header']}"

[CRITICAL GUIDELINE RULES]
1. **Initial Information Collection (Req 3):** The first step in the guideline MUST be to request the necessary initial diagnostic information (e.g., device compatibility, local status/location, order number) BEFORE attempting to troubleshoot or solve the problem.
2. **Empathy for Difficult Customers (Req 5):** If the customer type is 'Difficult Customer' or 'Highly Dissatisfied Customer', the guideline MUST emphasize extreme politeness, empathy, and apologies, even if the policy (e.g., no refund) must be enforced.
3. **24-48 Hour Follow-up (Req 6):** If the issue cannot be solved immediately or requires confirmation from a local partner/supervisor, the guideline MUST state the procedure:
   - Acknowledge the issue.
   - Inform the customer they will receive a definite answer within 24 or 48 hours.
   - Request the customer's email or phone number for follow-up contact. (Use provided contact info if available)
4. **Past Cases Learning:** If past cases guidelines are provided, incorporate successful strategies from those cases into your recommendations.

Customer Inquiry:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""
    if not st.session_state.is_llm_ready:
        mock_text = (
            f"### {L['simulation_advice_header']}\n\n"
            f"- (Mock) {st.session_state.customer_type_sim_select} 유형 고객 응대 가이드입니다. (요청 3, 5, 6 반영)\n\n"
            f"### {L['simulation_draft_header']}\n\n"
            f"(Mock) 에이전트 응대 초안이 여기에 들어갑니다。\n\n"
        )
        return mock_text
    else:
        with st.spinner(L["response_generating"]):
            try:
                return run_llm(initial_prompt)
            except Exception as e:
                st.error(f"AI 조언 생성 중 오류 발생: {e}")
                return f"❌ AI Advice Generation Error: {e}"


# ========================================
# 9. 사이드바
# ========================================

with st.sidebar:
    # 언어 키 안전하게 가져오기
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # 회사 목록 초기화 (회사 정보 탭에서 사용)
    if "company_language_priority" not in st.session_state:
        st.session_state.company_language_priority = {
            "default": ["ko", "en", "ja"],
            "companies": {}
        }
    
    st.markdown("---")
    
    # 언어 선택
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    lang_priority = st.session_state.company_language_priority["default"]
    
    selected_lang_key = st.selectbox(
        L["lang_select"],
        options=lang_priority,
        index=lang_priority.index(st.session_state.language) if st.session_state.language in lang_priority else 0,
        format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
    )

    # 🔹 언어 변경 감지
    if selected_lang_key != st.session_state.language:
        st.session_state.language = selected_lang_key
        # 채팅/전화 공통 상태 초기화
        st.session_state.simulator_messages = []
        # ⭐ 안전한 메모리 초기화
        try:
            if hasattr(st.session_state, 'simulator_memory') and st.session_state.simulator_memory is not None:
                st.session_state.simulator_memory.clear()
        except Exception:
            # 메모리 초기화 실패 시 새로 생성
            try:
                st.session_state.simulator_memory = ConversationBufferMemory(memory_key="chat_history")
            except Exception:
                pass  # 초기화 실패해도 계속 진행
        st.session_state.initial_advice_provided = False
        st.session_state.is_chat_ended = False
        # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로 플래그 사용
        st.session_state.reset_agent_response_area = True
        st.session_state.customer_query_text_area = ""
        st.session_state.last_transcript = ""
        st.session_state.sim_audio_bytes = None
        st.session_state.sim_stage = "WAIT_FIRST_QUERY"
        st.session_state.customer_attachment_file = []  # 언어 변경 시 첨부 파일 초기화
        st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
        st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
        # 전화 시뮬레이터 상태 초기화
        st.session_state.call_sim_stage = "WAITING_CALL"
        st.session_state.call_sim_mode = "INBOUND"
        st.session_state.is_on_hold = False
        st.session_state.total_hold_duration = timedelta(0)
        st.session_state.hold_start_time = None
        st.session_state.current_customer_audio_text = ""
        st.session_state.current_agent_audio_text = ""
        st.session_state.agent_response_input_box_widget_call = ""
        st.session_state.call_initial_query = ""
        # 전화 발신 관련 상태 초기화
        st.session_state.sim_call_outbound_summary = ""
        st.session_state.sim_call_outbound_target = None
        # ⭐ 언어 변경 시 재실행 - 무한 루프 방지를 위해 플래그 사용
        if "language_changed" not in st.session_state or not st.session_state.language_changed:
            st.session_state.language_changed = True
        else:
            # 이미 한 번 재실행했으면 플래그 초기화
            st.session_state.language_changed = False

    # 언어 키 안전하게 가져오기
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])

    st.title(L["sidebar_title"])
    st.markdown("---")

    # ⭐ API Key 설정 섹션 추가
    st.subheader("🔑 API Key 설정")
    
    # LLM 선택
    llm_options = {
        "openai_gpt4": "OpenAI GPT-4",
        "openai_gpt35": "OpenAI GPT-3.5",
        "gemini_pro": "Google Gemini Pro",
        "gemini_flash": "Google Gemini Flash",
        "claude": "Anthropic Claude",
        "groq": "Groq",
        "nvidia": "NVIDIA NIM"
    }
    
    current_llm = st.session_state.get("selected_llm", "openai_gpt4")
    selected_llm = st.selectbox(
        "LLM 모델 선택",
        options=list(llm_options.keys()),
        format_func=lambda x: llm_options[x],
        index=list(llm_options.keys()).index(current_llm) if current_llm in llm_options else 0,
        key="sidebar_llm_select"
    )
    if selected_llm != current_llm:
        st.session_state.selected_llm = selected_llm
    
    # API Key 매핑
    api_key_map = {
        "openai_gpt4": "openai",
        "openai_gpt35": "openai",
        "gemini_pro": "gemini",
        "gemini_flash": "gemini",
        "claude": "claude",
        "groq": "groq",
        "nvidia": "nvidia"
    }
    
    api_name = api_key_map.get(selected_llm, "openai")
    api_config = SUPPORTED_APIS.get(api_name, {})
    
    if api_config:
        # 현재 API Key 확인
        current_key = get_api_key(api_name)
        if not current_key:
            # 수동 입력 필드
            session_key = api_config.get("session_key", "")
            manual_key = st.text_input(
                api_config.get("label", "API Key"),
                value=st.session_state.get(session_key, ""),
                type="password",
                placeholder=api_config.get("placeholder", "API Key를 입력하세요"),
                key=f"manual_api_key_{selected_llm}"
            )
            if manual_key and manual_key != st.session_state.get(session_key, ""):
                st.session_state[session_key] = manual_key
        else:
            st.success(f"✅ {api_config.get('label', 'API Key')} 설정됨")
    
    st.markdown("---")

    # ⭐ 기능 선택 - 기본값을 AI 챗 시뮬레이터로 설정
    if "feature_selection" not in st.session_state:
        st.session_state.feature_selection = L["sim_tab_chat_email"]

    # ⭐ 핵심 기능과 더보기 기능 분리 (회사 정보 및 FAQ 추가)
    core_features = [L["sim_tab_chat_email"], L["sim_tab_phone"], L["company_info_tab"]]
    other_features = [L["rag_tab"], L["content_tab"], L["lstm_tab"], L["voice_rec_header"]]
    
    # 모든 기능을 하나의 리스트로 통합 (하나만 선택 가능하도록)
    all_features = core_features + other_features
    
    # 현재 선택된 기능
    current_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
    
    # 현재 선택의 인덱스 찾기
    try:
        current_index = all_features.index(current_selection) if current_selection in all_features else 0
    except (ValueError, AttributeError):
        current_index = 0
    
    # ⭐ 하나의 통합된 선택 로직 (하나만 선택 가능) - 설명 제거
    selected_feature = st.radio(
        "기능 선택",
        all_features,
        index=current_index,
        key="unified_feature_selection",
        label_visibility="hidden"
    )
    
    # 선택된 기능 업데이트
    if selected_feature != current_selection:
        st.session_state.feature_selection = selected_feature
    
    feature_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])

# 메인 타이틀
# ⭐ L 변수가 정의되어 있는지 확인 (사이드바에서 이미 정의됨)
if "language" not in st.session_state:
    st.session_state.language = "ko"
# 언어 키 안전하게 가져오기
current_lang = st.session_state.get("language", "ko")
if current_lang not in ["ko", "en", "ja"]:
    current_lang = "ko"
L = LANG.get(current_lang, LANG["ko"])

# ⭐ 타이틀과 설명을 한 줄로 간결하게 표시
feature_selection = st.session_state.get("feature_selection", L["sim_tab_chat_email"])
if feature_selection == L["sim_tab_chat_email"]:
    st.markdown(f"### 📧 {L['sim_tab_chat_email']}")
    st.caption(L['sim_tab_chat_email_desc'])
elif feature_selection == L["sim_tab_phone"]:
    st.markdown(f"### 📞 {L['sim_tab_phone']}")
    st.caption(L['sim_tab_phone_desc'])
elif feature_selection == L["rag_tab"]:
    st.markdown(f"### 📚 {L['rag_tab']}")
    st.caption(L['rag_tab_desc'])
elif feature_selection == L["content_tab"]:
    st.markdown(f"### 📝 {L['content_tab']}")
    st.caption(L['content_tab_desc'])
elif feature_selection == L["lstm_tab"]:
    st.markdown(f"### 📊 {L['lstm_tab']}")
    st.caption(L['lstm_tab_desc'])
elif feature_selection == L["voice_rec_header"]:
    st.markdown(f"### 🎤 {L['voice_rec_header']}")
    st.caption(L['voice_rec_header_desc'])
elif feature_selection == L["company_info_tab"]:
    # 공백 축소: 제목과 설명을 한 줄로 간결하게 표시
    st.markdown(f"#### 📋 {L['company_info_tab']}")
    st.caption(L['company_info_tab_desc'])

# ========================================
# 10. 기능별 페이지
# ========================================

# -------------------- Company Info & FAQ Tab --------------------
if feature_selection == L["company_info_tab"]:
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    # FAQ 데이터베이스 로드
    faq_data = load_faq_database()
    companies = list(faq_data.get("companies", {}).keys())
    
    # 회사명 검색 입력 (상단에 배치) - 입력란 길이 축소 (회사명은 보통 짧음)
    col_search_header, col_search_input, col_search_btn = st.columns([0.5, 1.2, 0.8])
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
        search_button = st.button(L["company_search_button"], key="company_search_btn", type="primary", use_container_width=True)
    
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
                    "faqs": generated_data.get("faqs", [])
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
                "faqs": display_data.get("faqs", [])
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
            "faqs": company_db_data.get("faqs", [])
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
                            st.markdown(
                                f"""
                                <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 10px; color: white; min-height: 200px; display: flex; flex-direction: column; justify-content: center;'>
                                    <h1 style='font-size: 64px; margin: 0;'>{product_emoji}</h1>
                                    <p style='font-size: 16px; margin-top: 15px; font-weight: bold;'>{product_text[:25]}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
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
                            import tempfile
                            import os
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
                                    import google.generativeai as genai
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
                                    ocr_llm_prompt = f"""{ocr_prompt}

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
            import html as html_escape
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
            import html as html_escape
            hint_escaped = html_escape.escape(hint_text)
            st.markdown(f"""
            <div style="font-size: 18px; line-height: 1.8; padding: 20px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #dee2e6;">
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Malgun Gothic', '맑은 고딕', 'Noto Sans JP', sans-serif; margin: 0; font-size: 18px; color: #212529;">{hint_escaped}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # 다운로드 버튼 추가 (더 안정적인 복사 방법)
            col_copy_hint, col_download_hint = st.columns(2)
            with col_copy_hint:
                st.info(L.get("copy_instruction", "💡 위 텍스트를 선택하고 Ctrl+C (Mac: Cmd+C)로 복사하세요."))
            with col_download_hint:
                st.download_button(
                    label=f"📥 {L.get('button_download_hint', '힌트 다운로드')}",
                    data=hint_text.encode('utf-8'),
                    file_name=f"response_hint_{st.session_state.get('copy_hint_id', 0)}.txt",
                    mime="text/plain",
                    key="download_hint_btn"
                )
        
        # 초기화 버튼
        if st.session_state.get("generated_ai_answer") or st.session_state.get("generated_hint"):
            if st.button(f"🔄 {L.get('button_reset', '새로 시작')}", key="reset_inquiry_btn"):
                st.session_state.generated_ai_answer = None
                st.session_state.generated_hint = None
                st.rerun()

# -------------------- Voice Record Tab --------------------

# -------------------- Voice Record Tab --------------------
if feature_selection == L["voice_rec_header"]:
    # ... (기존 음성 기록 탭 로직 유지)
    st.header(L["voice_rec_header"])
    st.caption(L["record_help"])

    col_rec, col_list = st.columns([1, 1])

    # 녹음/업로드 + 전사 + 저장
    with col_rec:
        st.subheader(L["rec_header"])
        audio_file = st.file_uploader(
            L["uploaded_file"],
            type=["wav", "mp3", "m4a", "webm", "ogg"],
            key="voice_rec_uploader",
        )
        audio_bytes = None
        audio_mime = "audio/webm"

        if audio_file is not None:
            audio_bytes = audio_file.getvalue()
            audio_mime = audio_file.type or "audio/webm"

        # 재생
        # Streamlit 문서: bytes, 파일 경로, URL 모두 지원
        if audio_bytes:
            try:
                # MIME 타입이 올바른지 확인하고 기본값 설정
                valid_formats = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm", "audio/ogg", "audio/m4a"]
                if audio_mime not in valid_formats:
                    # MIME 타입이 유효하지 않으면 파일 확장자로 추정
                    audio_mime = "audio/wav"  # 기본값
                st.audio(audio_bytes, format=audio_mime)
            except Exception as e:
                st.error(f"오디오 재생 오류: {e}")
                # 기본 포맷으로 재시도
                try:
                    st.audio(audio_bytes, format="audio/wav")
                except:
                    st.error("오디오 파일을 재생할 수 없습니다.")

        # 전사 버튼
        if audio_bytes and st.button(L["transcribe_btn"]):
            if st.session_state.openai_client is None:
                st.error(L["openai_missing"])
            else:
                with st.spinner(L["transcribing"]):
                    # 자동 언어 감지 사용 (입력 언어와 관계없이 정확한 전사)
                    text = transcribe_bytes_with_whisper(
                        audio_bytes, audio_mime, lang_code=None, auto_detect=True
                    )
                    st.session_state.last_transcript = text
                    snippet = text[:50].replace("\n", " ")
                    if len(text) > 50:
                        snippet += "..."
                    if text.startswith("❌"):
                        st.error(text)
                    else:
                        st.success(f"{L['transcript_result']} **{snippet}**")

        st.text_area(
            L["transcript_text"],
            value=st.session_state.last_transcript,
            height=150,
            key="voice_rec_transcript_area",
        )

        if audio_bytes and st.button(L["save_btn"]):
            try:
                ext = audio_mime.split("/")[-1] if "/" in audio_mime else "webm"
                filename = f"record_{int(time.time())}.{ext}"
                save_audio_record_local(
                    audio_bytes,
                    filename,
                    st.session_state.last_transcript,
                    mime_type=audio_mime,
                )
                st.success(L["saved_success"])
                st.session_state.last_transcript = ""
                # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
            except Exception as e:
                st.error(f"{L['error']} {e}")

    # 저장된 기록 리스트
    with col_list:
        st.subheader(L["rec_list_title"])
        try:
            records = load_voice_records()
        except Exception as e:
            st.error(f"read error: {e}")
            records = []

        if not records:
            st.info(L["no_records"])
        else:
            for rec in records:
                rec_id = rec["id"]
                created_at = rec.get("created_at")
                try:
                    dt = datetime.fromisoformat(created_at)
                    created_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    created_str = str(created_at)

                transcript_snippet = (rec.get("transcript") or "")[:50].replace("\n", " ")
                if len(rec.get("transcript") or "") > 50:
                    transcript_snippet += "..."

                with st.expander(f"[{created_str}] {transcript_snippet}"):
                    st.write(f"**{L['transcript_text']}:** {rec.get('transcript') or 'N/A'}")
                    st.caption(
                        f"**Size:** {rec.get('size')} bytes | **File:** {rec.get('audio_filename')}"
                    )

                    col_p, col_r, col_d = st.columns([2, 1, 1])

                    if col_p.button(L["playback"], key=f"play_{rec_id}"):
                        try:
                            b, info = get_audio_bytes_local(rec_id)
                            mime = info.get("mime_type", "audio/webm")
                            # Streamlit 문서: bytes 데이터를 직접 전달 가능
                            # MIME 타입 검증
                            valid_formats = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/webm", "audio/ogg", "audio/m4a"]
                            if mime not in valid_formats:
                                mime = "audio/wav"  # 기본값
                            st.audio(b, format=mime, autoplay=False)
                        except Exception as e:
                            st.error(f"{L['gcs_playback_fail']}: {e}")

                    if col_r.button(L["retranscribe"], key=f"re_{rec_id}"):
                        if st.session_state.openai_client is None:
                            st.error(L["openai_missing"])
                        else:
                            with st.spinner(L["transcribing"]):
                                try:
                                    b, info = get_audio_bytes_local(rec_id)
                                    mime = info.get("mime_type", "audio/webm")
                                    # 자동 언어 감지 사용 (입력 언어와 관계없이 정확한 전사)
                                    new_text = transcribe_bytes_with_whisper(
                                        b, mime, lang_code=None, auto_detect=True
                                    )
                                    records = load_voice_records()
                                    for r in records:
                                        if r["id"] == rec_id:
                                            r["transcript"] = new_text
                                            break
                                    save_voice_records(records)
                                    st.success(L["retranscribe"] + " " + L["saved_success"])
                                    # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
                                except Exception as e:
                                    st.error(f"{L['error']} {e}")

                    if col_d.button(L["delete"], key=f"del_{rec_id}"):
                        if st.session_state.get(f"confirm_del_{rec_id}", False):
                            ok = delete_audio_record_local(rec_id)
                            if ok:
                                st.success(L["delete_success"])
                            else:
                                st.error(L["delete_fail"])
                            st.session_state[f"confirm_del_{rec_id}"] = False
                            # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
                        else:
                            st.session_state[f"confirm_del_{rec_id}"] = True
                            st.warning(L["delete_confirm_rec"])
                            st.write("sim_stage:", st.session_state.get("sim_stage"))
                            st.write("is_llm_ready:", st.session_state.get("is_llm_ready"))

# -------------------- Simulator (Chat/Email) Tab --------------------
elif feature_selection == L["sim_tab_chat_email"]:
    # ... (기존 채팅/이메일 시뮬레이터 로직 유지)
    st.header(L["simulator_header"])
    st.markdown(L["simulator_desc"])

    current_lang = st.session_state.language
    L = LANG[current_lang]  # 다시 L 업데이트

    # =========================
    # 0. 전체 이력 삭제
    # =========================
    col_del, _ = st.columns([1, 4])
    with col_del:
        if st.button(L["delete_history_button"], key="trigger_delete_hist"):
            st.session_state.show_delete_confirm = True

    if st.session_state.show_delete_confirm:
        with st.container():
            st.warning(L["delete_confirm_message"])
            c_yes, c_no = st.columns(2)
            if c_yes.button(L["delete_confirm_yes"], key="confirm_del_yes"):
                with st.spinner(L["deleting_history_progress"]):
                    delete_all_history_local()
                    st.session_state.simulator_messages = []
                    st.session_state.simulator_memory.clear()
                    st.session_state.show_delete_confirm = False
                    st.session_state.is_chat_ended = False
                    st.session_state.sim_stage = "WAIT_FIRST_QUERY"
                    st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
                    st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
                    st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
                    st.success(L["delete_success"])
                    # ⭐ 최적화: 버튼 클릭 후 Streamlit이 자동으로 재실행하므로 rerun 제거
            if c_no.button(L["delete_confirm_no"], key="confirm_del_no"):
                st.session_state.show_delete_confirm = False

    # =========================
    # 1. 이전 이력 로드 (검색/필터링 기능 개선)
    # =========================
    with st.expander(L["history_expander_title"]):
        # Always load all available histories for the current language (sorted by recency)
        histories = load_simulation_histories_local(current_lang)

        # 전체 통계 및 트렌드 대시보드 (요약 데이터가 있는 경우만)
        cases_with_summary = [
            h for h in histories
            if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
               and not h.get("is_call", False)  # 전화 이력 제외
        ]

        if cases_with_summary:
            st.markdown("---")
            st.subheader("📈 과거 케이스 트렌드 대시보드")

            # 트렌드 차트 표시
            trend_chart = visualize_case_trends(histories, current_lang)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
                avg_sentiment = np.mean(
                    [h["summary"].get("customer_sentiment_score", 50) for h in cases_with_summary if h.get("summary")])
                avg_satisfaction = np.mean(
                    [h["summary"].get("customer_satisfaction_score", 50) for h in cases_with_summary if
                     h.get("summary")])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균 감정 점수", f"{avg_sentiment:.1f}/100", f"총 {len(cases_with_summary)}건")
                with col2:
                    st.metric("평균 만족도", f"{avg_satisfaction:.1f}/100", f"총 {len(cases_with_summary)}건")

            st.markdown("---")

        # ⭐ 검색 폼 제거 및 독립된 위젯 사용
        col_search, col_btn = st.columns([4, 1])

        with col_search:
            # st.text_input은 Enter 키 입력 시 앱을 재실행합니다.
            search_query = st.text_input(L["search_history_label"], key="sim_hist_search_input_new")

        with col_btn:
            # 검색 버튼: 누르면 앱을 강제 재실행하여 검색/필터링 로직을 다시 타도록 합니다.
            st.markdown("<br>", unsafe_allow_html=True)  # Align button vertically
            search_clicked = st.button(L["history_search_button"], key="apply_search_btn_new")

        # 날짜 범위 필터
        today = datetime.now().date()
        date_range_value = [today - timedelta(days=7), today]
        dr = st.date_input(
            L["date_range_label"],
            value=date_range_value,
            key="sim_hist_date_range_actual",
        )

        # --- Filtering Logic ---
        current_search_query = search_query.strip()

        if histories:
            start_date = min(dr)
            end_date = max(dr)

            filtered = []
            for h in histories:
                # 전화 이력은 제외 (채팅/이메일 탭이므로)
                if h.get("is_call", False):
                    continue

                ok_search = True
                if current_search_query:
                    q = current_search_query.lower()
                    # 검색 대상: 초기 문의, 고객 유형, 요약 데이터
                    text = (h["initial_query"] + " " + h["customer_type"]).lower()

                    # 요약 데이터가 있으면 요약 내용도 검색 대상에 포함
                    summary = h.get("summary")
                    if summary and isinstance(summary, dict):
                        summary_text = summary.get("main_inquiry", "") + " " + summary.get("summary", "")
                        text += " " + summary_text.lower()

                    # Check if query matches in initial query, customer type, or summary
                    if q not in text:
                        ok_search = False

                ok_date = True
                ts = h.get("timestamp")
                if ts:
                    try:
                        d = datetime.fromisoformat(ts).date()
                        # Apply date filtering
                        if not (start_date <= d <= end_date):
                            ok_date = False
                    except Exception:
                        pass  # Ignore histories with invalid timestamp

                if ok_search and ok_date:
                    filtered.append(h)
        else:
            filtered = []

        # Determine the list for display (⭐ 요청 사항: 검색어/필터가 없으면 최근 10건만 표시)
        is_searching_or_filtering = bool(current_search_query) or dr != date_range_value

        if not is_searching_or_filtering:
            # 검색/필터 조건이 없으면, 전체 이력 중 최신 10건만 표시
            filtered_for_display = filtered[:10]  # 필터링된 목록(전화 제외) 중 10개
        else:
            # 검색/필터 조건이 있으면, 필터링된 모든 결과를 표시
            filtered_for_display = filtered

        # --- Display Logic ---

        if filtered_for_display:
            def _label(h):
                try:
                    t = datetime.fromisoformat(h["timestamp"])
                    t_str = t.strftime("%m-%d %H:%M")
                except Exception:
                    t_str = h.get("timestamp", "")

                # 요약 데이터가 있으면 요약 정보 표시, 없으면 초기 문의 표시
                summary = h.get("summary")
                if summary and isinstance(summary, dict):
                    main_inquiry = summary.get("main_inquiry", h["initial_query"][:30])
                    sentiment = summary.get("customer_sentiment_score", 50)
                    satisfaction = summary.get("customer_satisfaction_score", 50)
                    q = main_inquiry[:30].replace("\n", " ")
                    # 첨부 파일 여부 표시 추가
                    attachment_icon = "📎" if h.get("attachment_context") else ""
                    # 요약 데이터 표시 (감정/만족도 점수 포함)
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} | 감정:{sentiment} 만족:{satisfaction} - {q}..."
                else:
                    q = h["initial_query"][:30].replace("\n", " ")
                    attachment_icon = "📎" if h.get("attachment_context") else ""
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} - {q}..."


            options_map = {_label(h): h for h in filtered_for_display}

            # Show a message indicating what is displayed if filters were applied
            if is_searching_or_filtering:
                st.caption(f"🔎 총 {len(filtered_for_display)}개 이력 검색됨 (전화 이력 제외)")
            else:
                st.caption(f"⭐ 최근 {len(filtered_for_display)}개 이력 표시 중 (전화 이력 제외)")

            sel_key = st.selectbox(L["history_selectbox_label"], options=list(options_map.keys()))

            if st.button(L["history_load_button"], key="load_hist_btn"):
                h = options_map[sel_key]
                st.session_state.customer_query_text_area = h["initial_query"]

                # 메시지가 비어있고 요약 데이터가 있는 경우, 요약을 기반으로 최소한의 메시지 재구성
                if not h.get("messages") and h.get("summary"):
                    summary = h["summary"]
                    # 요약 데이터를 기반으로 기본 메시지 구조 생성
                    reconstructed_messages = [
                        {"role": "customer", "content": h["initial_query"]}
                    ]
                    # 요약에서 핵심 응답 추가
                    if summary.get("key_responses"):
                        for response in summary.get("key_responses", [])[:3]:  # 최대 3개만
                            reconstructed_messages.append({"role": "agent_response", "content": response})
                    # 요약 정보를 supervisor 메시지로 추가
                    summary_text = f"**요약된 상담 이력**\n\n"
                    summary_text += f"주요 문의: {summary.get('main_inquiry', 'N/A')}\n"
                    summary_text += f"고객 감정 점수: {summary.get('customer_sentiment_score', 50)}/100\n"
                    summary_text += f"고객 만족도: {summary.get('customer_satisfaction_score', 50)}/100\n"
                    summary_text += f"\n전체 요약:\n{summary.get('summary', 'N/A')}"
                    reconstructed_messages.append({"role": "supervisor", "content": summary_text})
                    st.session_state.simulator_messages = reconstructed_messages

                    # 요약 데이터 시각화
                    st.markdown("---")
                    st.subheader("📊 로드된 케이스 분석")

                    # 요약 데이터를 프로필 형식으로 변환
                    loaded_profile = {
                        "sentiment_score": summary.get("customer_sentiment_score", 50),
                        "urgency_level": "medium",  # 기본값
                        "predicted_customer_type": h.get("customer_type", "normal")
                    }

                    # 프로필 점수 차트
                    profile_chart = visualize_customer_profile_scores(loaded_profile, current_lang)
                    if profile_chart:
                        st.plotly_chart(profile_chart, use_container_width=True)
                    else:
                        # Plotly가 없을 경우 텍스트로 표시
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(L.get("sentiment_score_label", "감정 점수"),
                                      f"{summary.get('customer_sentiment_score', 50)}/100")
                        with col2:
                            st.metric(L.get("urgency_score_label", "긴급도"), f"50/100")
                        with col3:
                            st.metric(L.get("customer_type_label", "고객 유형"), h.get("customer_type", "normal"))

                    # 고객 특성 시각화
                    if summary.get("customer_characteristics") or summary.get("privacy_info"):
                        characteristics_chart = visualize_customer_characteristics(summary, current_lang)
                        if characteristics_chart:
                            st.plotly_chart(characteristics_chart, use_container_width=True)
                else:
                    # 기존 메시지가 있는 경우 그대로 사용
                    st.session_state.simulator_messages = h.get("messages", [])

                st.session_state.initial_advice_provided = True
                st.session_state.is_chat_ended = h.get("is_chat_ended", False)
                st.session_state.sim_attachment_context_for_llm = h.get("attachment_context", "")  # 컨텍스트 로드
                st.session_state.customer_attachment_file = []  # 로드된 이력에는 파일 객체 대신 컨텍스트 문자열만 사용
                st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화

                # 상태 복원
                if st.session_state.is_chat_ended:
                    st.session_state.sim_stage = "CLOSING"
                else:
                    messages = st.session_state.simulator_messages
                    last_role = messages[-1]["role"] if messages else None
                    if last_role == "agent_response":
                        st.session_state.sim_stage = "CUSTOMER_TURN"
                    elif last_role == "customer_rebuttal":
                        st.session_state.sim_stage = "AGENT_TURN"
                    elif last_role == "supervisor" and messages and messages[-1]["content"] == L[
                        "customer_closing_confirm"]:
                        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"

                st.session_state.simulator_memory.clear()  # 메모리 초기화
        else:
            st.info(L["no_history_found"])

    # =========================
    # AHT 타이머 (화면 최상단)
    # =========================
    if st.session_state.sim_stage not in ["WAIT_FIRST_QUERY", "CLOSING", "idle"]:
        elapsed_placeholder = st.empty()

        if st.session_state.start_time is not None:
            # 실시간 업데이트를 위해 페이지 로드 시마다 현재 시간 계산
            elapsed_time = datetime.now() - st.session_state.start_time
            total_seconds = elapsed_time.total_seconds()

            # Hold 시간 제외 (채팅/이메일은 Hold 없음, 전화 탭과 로직 통일 위해 유지)
            # total_seconds -= st.session_state.total_hold_duration.total_seconds()

            # 시간 형식 포맷팅
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 경고 기준
            if total_seconds > 900:  # 15분
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:  # 10분
                delta_str = L["timer_info_warn"]
                delta_color = "off"
            else:
                delta_str = L["timer_info_ok"]
                delta_color = "normal"

            elapsed_placeholder.metric(
                L["timer_metric"],
                time_str,
                delta=delta_str,
                delta_color=delta_color
            )

            # ⭐ 수정: 3초마다 재실행하여 AHT 실시간성 확보
            if seconds % 3 == 0 and total_seconds < 1000:
                time.sleep(1)

        st.markdown("---")

    # =========================
    # 2. LLM 준비 체크 & 채팅 종료 상태
    # =========================
    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.sim_stage == "CLOSING":
        st.success(L["survey_sent_confirm"])
        st.info(L["new_simulation_ready"])
        
        # ⭐ 추가: 현재 세션 이력 다운로드 기능
        st.markdown("---")
        st.markdown("**📥 현재 세션 이력 다운로드**")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        # 현재 세션의 이력을 생성
        current_session_history = None
        if st.session_state.simulator_messages:
            try:
                customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
                current_session_summary = generate_chat_summary(
                    st.session_state.simulator_messages,
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.language
                )
                current_session_history = [{
                    "id": f"session_{st.session_state.sim_instance_id}",
                    "timestamp": datetime.now().isoformat(),
                    "initial_query": st.session_state.customer_query_text_area,
                    "customer_type": customer_type_display,
                    "language_key": st.session_state.language,
                    "messages": st.session_state.simulator_messages,
                    "summary": current_session_summary,
                    "is_chat_ended": True,
                    "attachment_context": st.session_state.sim_attachment_context_for_llm
                }]
            except Exception as e:
                st.warning(f"이력 생성 중 오류 발생: {e}")
        
        # 다운로드 버튼들을 직접 표시
        if current_session_history:
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_word", "📥 이력 다운로드 (Word)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_word_file"
                        )
                except Exception as e:
                    st.error(f"Word 다운로드 오류: {e}")
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pptx", "📥 이력 다운로드 (PPTX)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_pptx_file"
                        )
                except Exception as e:
                    st.error(f"PPTX 다운로드 오류: {e}")
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pdf", "📥 이력 다운로드 (PDF)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pdf),
                            mime="application/pdf",
                            key="download_pdf_file"
                        )
                except Exception as e:
                    st.error(f"PDF 다운로드 오류: {e}")
        else:
            st.warning("다운로드할 이력이 없습니다.")
        
        st.markdown("---")
        
        if st.button(L["new_simulation_button"], key="new_simulation_btn"):
            # 초기화 로직
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.initial_advice_provided = False
            st.session_state.is_chat_ended = False
            st.session_state.agent_response_area_text = ""
            st.session_state.customer_query_text_area = ""
            st.session_state.last_transcript = ""
            st.session_state.sim_audio_bytes = None
            st.session_state.sim_stage = "WAIT_FIRST_QUERY"
            st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
            st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
            st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
            st.session_state.start_time = None
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None
            # ⭐ 재실행
            st.rerun()
        # st.stop()

    # =========================
    # 5-A. 전화 발신 진행 중 (OUTBOUND_CALL_IN_PROGRESS)
    # =========================
    elif st.session_state.sim_stage == "OUTBOUND_CALL_IN_PROGRESS":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        target = st.session_state.get("sim_call_outbound_target", "대상")
        st.warning(L["call_outbound_loading"])

        # LLM 호출 및 요약 생성
        with st.spinner(L["call_outbound_loading"]):
            # 1. LLM 호출하여 통화 요약 생성
            summary = generate_outbound_call_summary(
                st.session_state.customer_query_text_area,
                st.session_state.language,
                target
            )

            # 2. 시스템 메시지 (전화 시도) 추가
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": L["call_outbound_system_msg"].format(target=target)}
            )

            # 3. 요약 메시지 (결과) 추가
            summary_markdown = f"### {L['call_outbound_summary_header']}\n\n{summary}"
            st.session_state.simulator_messages.append(
                {"role": "supervisor", "content": summary_markdown}
            )

            # 4. Agent Turn으로 복귀
            st.session_state.sim_stage = "AGENT_TURN"
            st.session_state.sim_call_outbound_summary = summary_markdown  # Save for display/reference
            st.session_state.sim_call_outbound_target = None  # Reset target

            # 5. 이력 저장 (전화 발신 후 상태 저장)
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display + f" (Outbound Call to {target})",
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.success(f"✅ {L['call_outbound_simulation_header']}가 완료되었습니다. 요약을 확인하고 고객에게 회신하세요.")

    # ========================================
    # 3. 초기 문의 입력 (WAIT_FIRST_QUERY)
    # ========================================
    if st.session_state.sim_stage == "WAIT_FIRST_QUERY":
        customer_query = st.text_area(
            L["customer_query_label"],
            key="customer_query_text_area",
            height=150,
            placeholder=L["initial_query_sample"],
        )

        # --- 필수 입력 필드 (요청 3 반영: UI 텍스트 변경) ---
        customer_email = st.text_input(
            L["customer_email_label"],
            key="customer_email_input",
            value=st.session_state.customer_email,
        )
        customer_phone = st.text_input(
            L["customer_phone_label"],
            key="customer_phone_input",
            value=st.session_state.customer_phone,
        )
        # 세션 상태 업데이트
        st.session_state.customer_email = customer_email
        st.session_state.customer_phone = customer_phone
        # --------------------------------------------------

        customer_type_options = L["customer_type_options"]
        # st.session_state.customer_type_sim_select는 이미 초기화됨
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        # Selectbox는 자체적으로 세션 상태를 업데이트하므로, 여기에 value를 설정할 필요 없음
        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="customer_type_sim_select_widget",
        )

        # --- 첨부 파일 업로더 추가 ---
        customer_attachment_widget = st.file_uploader(
            L["attachment_label"],
            type=["png", "jpg", "jpeg", "pdf"],
            key="customer_attachment_file_uploader",
            help=L["attachment_placeholder"],
            accept_multiple_files=False  # 채팅/이메일은 단일 파일만 허용
        )

        # 파일 정보 저장 및 LLM 컨텍스트 생성
        if customer_attachment_widget:
            st.session_state.customer_attachment_file = customer_attachment_widget
            st.session_state.sim_attachment_context_for_llm = L["attachment_status_llm"].format(
                filename=customer_attachment_widget.name, filetype=customer_attachment_widget.type
            )
        else:
            st.session_state.customer_attachment_file = None
            st.session_state.sim_attachment_context_for_llm = ""
        # --------------------------

        if st.button(L["button_simulate"], key=f"btn_simulate_initial_{st.session_state.sim_instance_id}"):  # 고유 키 사용
            if not customer_query.strip():
                st.warning(L["simulation_warning_query"])
                # st.stop()

            # --- 필수 입력 필드 검증 (요청 3 반영: 검증 로직 추가) ---
            if not st.session_state.customer_email.strip() or not st.session_state.customer_phone.strip():
                st.error(L["error_mandatory_contact"])
                # st.stop()
            # ------------------------------------------

            # 초기 상태 리셋
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.is_chat_ended = False
            st.session_state.initial_advice_provided = False
            st.session_state.is_solution_provided = False  # 솔루션 플래그 리셋
            st.session_state.language_transfer_requested = False  # 언어 요청 플래그 리셋
            st.session_state.transfer_summary_text = ""  # 이관 요약 리셋
            st.session_state.start_time = None  # AHT 타이머 초기화 (첫 고객 반응 후 시작)
            st.session_state.sim_instance_id = str(uuid.uuid4())  # 새 시뮬레이션 ID 할당
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None

            # 1) 고객 첫 메시지 추가
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_query}
            )

            # 2) Supervisor 가이드 + 초안 생성
            # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
            try:
                detected_lang = detect_text_language(customer_query)
                # 감지된 언어가 유효한지 확인
                if detected_lang not in ["ko", "en", "ja"]:
                    detected_lang = current_lang
                else:
                    # 언어가 감지되었고 현재 언어와 다르면 자동으로 언어 설정 업데이트
                    if detected_lang != current_lang:
                        st.session_state.language = detected_lang
                        st.info(f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
            except Exception as e:
                print(f"Language detection failed: {e}")
                detected_lang = current_lang  # 기본값으로 폴백
            
            # 고객 프로필 분석 (시각화를 위해 먼저 수행, 감지된 언어 사용)
            customer_profile = analyze_customer_profile(customer_query, detected_lang)
            similar_cases = find_similar_cases(customer_query, customer_profile, detected_lang, limit=5)

            # 시각화 차트 표시
            st.markdown("---")
            st.subheader("📊 고객 프로필 분석")

            # 고객 프로필 점수 차트 (감지된 언어 사용)
            profile_chart = visualize_customer_profile_scores(customer_profile, detected_lang)
            if profile_chart:
                st.plotly_chart(profile_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    gender_display = customer_profile.get("gender", "unknown")
                    if gender_display == "male":
                        gender_display = "남자"
                    elif gender_display == "female":
                        gender_display = "여자"
                    else:
                        gender_display = "알 수 없음"
                    st.metric(
                        "성별",
                        gender_display
                    )
                with col2:
                    st.metric(
                        L.get("sentiment_score_label", "감정 점수"),
                        f"{customer_profile.get('sentiment_score', 50)}/100"
                    )
                with col3:
                    urgency_map = {"low": 25, "medium": 50, "high": 75}
                    urgency_score = urgency_map.get(customer_profile.get("urgency_level", "medium").lower(), 50)
                    st.metric(
                        L.get("urgency_score_label", "긴급도"),
                        f"{urgency_score}/100"
                    )
                with col4:
                    st.metric(
                        L.get("customer_type_label", "고객 유형"),
                        customer_profile.get("predicted_customer_type", "normal")
                    )

            # 유사 케이스 시각화
            if similar_cases:
                st.markdown("---")
                st.subheader("🔍 유사 케이스 추천")
                similarity_chart = visualize_similarity_cases(similar_cases, detected_lang)
                if similarity_chart:
                    st.plotly_chart(similarity_chart, use_container_width=True)

                # 유사 케이스 요약 표시
                with st.expander(f"💡 {len(similar_cases)}개 유사 케이스 상세 정보"):
                    for idx, similar_case in enumerate(similar_cases, 1):
                        case = similar_case["case"]
                        summary = similar_case["summary"]
                        similarity = similar_case["similarity_score"]
                        st.markdown(f"### 케이스 {idx} (유사도: {similarity:.1f}%)")
                        st.markdown(f"**문의 내용:** {summary.get('main_inquiry', 'N/A')}")
                        st.markdown(f"**감정 점수:** {summary.get('customer_sentiment_score', 50)}/100")
                        st.markdown(f"**만족도 점수:** {summary.get('customer_satisfaction_score', 50)}/100")
                        if summary.get("key_responses"):
                            st.markdown("**핵심 응답:**")
                            for response in summary.get("key_responses", [])[:3]:
                                st.markdown(f"- {response[:100]}...")
                        st.markdown("---")

            # 초기 조언 생성 (감지된 언어 사용)
            text = _generate_initial_advice(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.customer_email,
                st.session_state.customer_phone,
                detected_lang,  # 감지된 언어 사용
                st.session_state.customer_attachment_file
            )
            st.session_state.simulator_messages.append({"role": "supervisor", "content": text})

            st.session_state.initial_advice_provided = True
            save_simulation_history_local(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.simulator_messages,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
                is_chat_ended=False,
            )
            st.session_state.sim_stage = "AGENT_TURN"

    # =========================
    # 4. 대화 로그 표시 (공통)
    # =========================
    
    # 피드백 저장 콜백 함수
    def save_feedback(index):
        """에이전트 응답에 대한 고객 피드백을 저장"""
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{index}"
        if feedback_key in st.session_state:
            feedback_value = st.session_state[feedback_key]
            # 메시지에 피드백 정보 저장
            if index < len(st.session_state.simulator_messages):
                st.session_state.simulator_messages[index]["feedback"] = feedback_value
    
    for idx, msg in enumerate(st.session_state.simulator_messages):
        role = msg["role"]
        content = msg["content"]
        avatar = {"customer": "🙋", "supervisor": "🤖", "agent_response": "🧑‍💻", "customer_rebuttal": "✨",
                  "system_end": "📌", "system_transfer": "📌"}.get(role, "💬")
        tts_role = "customer" if role.startswith("customer") or role == "customer_rebuttal" else (
            "agent" if role == "agent_response" else "supervisor")

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            # 인덱스를 render_tts_button에 전달하여 고유 키 생성에 사용
            render_tts_button(content, st.session_state.language, role=tts_role, prefix=f"{role}_", index=idx)
            
            # ⭐ 에이전트 응답에 대한 피드백 위젯 추가
            if role == "agent_response":
                feedback_key = f"feedback_{st.session_state.sim_instance_id}_{idx}"
                # 기존 피드백 값 가져오기
                existing_feedback = msg.get("feedback", None)
                if existing_feedback is not None:
                    st.session_state[feedback_key] = existing_feedback
                
                # 피드백 위젯 표시
                st.feedback(
                    "thumbs",
                    key=feedback_key,
                    disabled=existing_feedback is not None,
                    on_change=save_feedback,
                    args=[idx],
                )

            # ⭐ [새로운 로직] 고객 첨부 파일 렌더링 (첫 번째 메시지인 경우)
            if idx == 0 and role == "customer" and st.session_state.customer_attachment_b64:
                mime = st.session_state.customer_attachment_mime or "image/png"
                data_url = f"data:{mime};base64,{st.session_state.customer_attachment_b64}"

                # 이미지 파일만 표시 (PDF 등은 아직 처리하지 않음)
                if mime.startswith("image/"):
                    st.image(data_url, caption=f"첨부된 증거물 ({st.session_state.customer_attachment_file.name})",
                             use_column_width=True)
                elif mime == "application/pdf":
                    # PDF 파일일 경우, 파일 이름과 함께 다운로드 링크 또는 경고 표시
                    st.warning(
                        f"첨부된 PDF 파일 ({st.session_state.customer_attachment_file.name})은 현재 인라인 미리보기가 지원되지 않습니다.")

    # 이관 요약 표시 (이관 후에만) - 루프 밖으로 이동하여 한 번만 표시
    if st.session_state.transfer_summary_text or (st.session_state.language != st.session_state.language_at_transfer_start and st.session_state.language_at_transfer_start):
                st.markdown("---")
                st.markdown(f"**{L['transfer_summary_header']}**")
                st.info(L["transfer_summary_intro"])

                # 번역이 실패했을 경우 확인 (번역 성공 여부 플래그 사용)
                is_translation_failed = not st.session_state.get("translation_success", True) or not st.session_state.transfer_summary_text

                if is_translation_failed:
                    # 번역 실패 시에도 원본 텍스트가 표시되므로 오류 메시지 없이 원본 텍스트만 표시
                    # (오류 메시지를 표시하지 않아도 원본 텍스트로 계속 진행 가능)
                    if st.session_state.transfer_summary_text:
                        st.info(st.session_state.transfer_summary_text)
                    # 번역 재시도 버튼 추가 (선택적)
                    if st.button(L.get("button_retry_translation", "번역 다시 시도"),
                                 key=f"btn_retry_translation_{st.session_state.sim_instance_id}"):  # 고유 키 사용
                        # 재시도 로직 실행
                        with st.spinner(L.get("transfer_loading", "번역 중...")):
                            source_lang = st.session_state.language_at_transfer_start
                            target_lang = st.session_state.language

                            # 이전 대화 내용 재가공
                            history_text = get_chat_history_for_prompt(include_attachment=False)
                            for msg in st.session_state.simulator_messages:
                                role = "Customer" if msg["role"].startswith("customer") or msg[
                                    "role"] == "initial_query" else "Agent"
                                if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                                   "customer_closing_response"]:
                                    history_text += f"{role}: {msg['content']}\n"

                            # ⭐ 수정: 먼저 핵심 포인트만 요약한 후 번역
                            lang_name_source = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(source_lang, "Korean")
                            summary_prompt = f"""
You are an AI assistant that summarizes customer service conversations. 
Extract ONLY the key points from the conversation below. Keep it concise and focused on:
1. Customer's main inquiry/question
2. Key information provided by the agent
3. Important decisions or outcomes
4. Any unresolved issues

Write the summary in {lang_name_source}. Maximum 200 words. Be brief and to the point.

--- Conversation ---
{history_text}
---

Key Points Summary:
"""
                            
                            # 요약 생성
                            summarized_text = ""
                            if st.session_state.is_llm_ready:
                                try:
                                    summarized_text = run_llm(summary_prompt).strip()
                                except Exception as e:
                                    print(f"요약 생성 실패, 전체 대화 사용: {e}")
                                    summarized_text = history_text  # 요약 실패 시 전체 대화 사용
                            else:
                                summarized_text = history_text  # LLM이 없으면 전체 대화 사용
                            
                            translated_summary, is_success = translate_text_with_llm(summarized_text, target_lang, source_lang)
                            st.session_state.transfer_summary_text = translated_summary
                            st.session_state.translation_success = is_success
                            st.session_state.transfer_retry_count += 1


                else:
                    # [수정 2] 번역 성공 시 내용 표시 및 TTS 버튼 추가
                    st.markdown(st.session_state.transfer_summary_text)
            # ⭐ 수정: 이관 요약의 경우 안정적인 키를 생성하도록 수정 (세션 ID와 언어 코드 조합)
                    render_tts_button(
                        st.session_state.transfer_summary_text,
                        st.session_state.language,
                        role="agent",
                        prefix="trans_summary_tts",
                        index=-1  # 고유 세션 ID 기반의 키를 생성하도록 지시
                    )
                st.markdown("---")

    # =========================
    # 5. 에이전트 입력 단계 (AGENT_TURN)
    # =========================
    if st.session_state.sim_stage == "AGENT_TURN":
        st.markdown(f"### {L['agent_response_header']}")

        # --- 실시간 응대 힌트 영역 ---
        hint_cols = st.columns([4, 1])
        with hint_cols[0]:
            st.info(L["hint_placeholder"] + st.session_state.realtime_hint_text)

        with hint_cols[1]:
            # 힌트 요청 버튼
            if st.button(L["button_request_hint"], key=f"btn_request_hint_{st.session_state.sim_instance_id}"):
                with st.spinner(L["response_generating"]):
                    # 채팅/이메일 탭이므로 is_call=False
                    hint = generate_realtime_hint(current_lang, is_call=False)
                    st.session_state.realtime_hint_text = hint

        # --- 언어 이관 요청 강조 표시 ---
        if st.session_state.language_transfer_requested:
            st.error("🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。")

        # --- 고객 첨부 파일 정보 재표시 ---
        if st.session_state.sim_attachment_context_for_llm:
            st.info(
                f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")

        # --- AI 응답 초안 생성 버튼 (요청 1 반영) ---
        if st.button(L["button_generate_draft"], key=f"btn_generate_ai_draft_{st.session_state.sim_instance_id}"):
            if not st.session_state.is_llm_ready:
                st.warning(L["simulation_no_key_warning"])
            else:
                with st.spinner(L["draft_generating"]):
                    # 초안 생성 함수 호출
                    ai_draft = generate_agent_response_draft(current_lang)
                    if ai_draft and not ai_draft.startswith("❌"):
                        st.session_state.agent_response_area_text = ai_draft
                        st.success(L["draft_success"])
                        # ⭐ 재실행하여 텍스트 영역 업데이트
                        st.rerun()
                    else:
                        st.error(ai_draft if ai_draft else L.get("draft_error", "응답 초안 생성에 실패했습니다."))

        # --- 전화 발신 버튼 추가 (요청 2 반영) ---
        st.markdown("---")
        st.subheader(L["button_call_outbound"])
        call_cols = st.columns(2)

        with call_cols[0]:
            if st.button(L["button_call_outbound_to_provider"], key="btn_call_outbound_partner", use_container_width=True):
                # 전화 발신 시뮬레이션: 현지 업체
                st.session_state.sim_call_outbound_target = "현지 업체/파트너"
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                st.rerun()

        with call_cols[1]:
            if st.button(L["button_call_outbound_to_customer"], key="btn_call_outbound_customer", use_container_width=True):
                # 전화 발신 시뮬레이션: 고객
                st.session_state.sim_call_outbound_target = "고객"
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                st.rerun()

        st.markdown("---")
        # --- 전화 발신 버튼 추가 끝 ---

        st.markdown("### 🚨 Supervisor 정책/지시 사항 업로드 (예외 처리 방침)")

        # --- Supervisor 정책 업로더 추가 ---
        supervisor_attachment_widget = st.file_uploader(
            "Supervisor 지시 사항/스크린샷 업로드 (예외 정책 포함)",
            type=["png", "jpg", "jpeg", "pdf", "txt"],
            key="supervisor_policy_uploader",
            help="비행기 지연, 질병 등 예외적 상황에 대한 Supervisor의 최신 지시 사항을 업로드하세요。",
            accept_multiple_files=False
        )

        # 파일 정보 저장 및 LLM 컨텍스트 생성
        if supervisor_attachment_widget:
            # 텍스트 파일 또는 PDF/이미지 파일의 텍스트 컨텐츠를 추출하여 policy_context에 저장해야 함
            # 여기서는 파일 이름과 타입만 컨텍스트로 전달하고, LLM이 이것이 '예외 정책'임을 알도록 유도
            file_name = supervisor_attachment_widget.name
            st.session_state.supervisor_policy_context = f"[Supervisor Policy Attached] Filename: {file_name}, Filetype: {supervisor_attachment_widget.type}. This file contains a CRITICAL, temporary policy update regarding exceptions (e.g., flight delays, illness, natural disasters). Analyze and prioritize this policy in the response."
            st.success(f"✅ Supervisor 정책 파일: **{file_name}**이(가) 응대 가이드에 반영됩니다.")
        elif st.session_state.supervisor_policy_context:
            st.info("⭐ 현재 적용 중인 Supervisor 정책이 있습니다.")
        else:
            st.session_state.supervisor_policy_context = ""

        # --- 에이전트 첨부 파일 업로더 (다중 파일 허용) ---
        agent_attachment_files = st.file_uploader(
            L["agent_attachment_label"],
            type=["png", "jpg", "jpeg", "pdf"],
            key="agent_attachment_file_uploader",
            help=L["agent_attachment_placeholder"],
            accept_multiple_files=True
        )

        if agent_attachment_files:
            st.session_state.agent_attachment_file = [
                {"name": f.name, "type": f.type, "size": f.size} for f in agent_attachment_files
            ]
            file_names = ", ".join([f["name"] for f in
                                    st.session_state.agent_attachment_file])  # 수정: file_infos 대신 st.session_state.agent_attachment_file 사용
            st.info(f"✅ {len(agent_attachment_files)}개 에이전트 첨부 파일 준비 완료: {file_names}")
        else:
            st.session_state.agent_attachment_file = []

        # --- 입력 필드 및 버튼 ---
        col_mic, col_text = st.columns([1, 2])

        # --- 마이크 녹음 ---
        with col_mic:
            mic_audio = mic_recorder(
                start_prompt=L["button_mic_input"],
                stop_prompt=L["button_mic_stop"],
                just_once=False,
                format="wav",
                use_container_width=True,
                key="sim_mic_recorder",
            )

        if mic_audio and mic_audio.get("bytes"):
            st.session_state.sim_audio_bytes = mic_audio["bytes"]
            st.info("✅ 녹음 완료! 아래 전사 버튼을 눌러 텍스트로 변환하세요.")

        if st.session_state.sim_audio_bytes:
            col_audio, col_transcribe, col_del = st.columns([3, 1, 1])

            # 1. 오디오 플레이어
            # Streamlit 문서: bytes 데이터를 직접 전달 가능
            with col_audio:
                try:
                    st.audio(st.session_state.sim_audio_bytes, format="audio/wav", autoplay=False)
                except Exception as e:
                    st.error(f"오디오 재생 오류: {e}")

            # 2. 녹음 삭제 버튼 (추가 요청 반영)
            with col_del:
                st.markdown("<br>", unsafe_allow_html=True)  # 버튼 수직 정렬
                if st.button(L["delete_mic_record"], key="btn_delete_sim_audio_call"):
                    # 오디오 및 관련 상태 초기화
                    st.session_state.sim_audio_bytes = None
                    st.session_state.last_transcript = ""
                    # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로 플래그 사용
                    st.session_state.reset_agent_response_area = True
                    st.success("녹음이 삭제되었습니다. 다시 녹음해 주세요.")
                    st.rerun()

            # 3. 전사(Whisper) 버튼 (기존 로직 대체)
            col_tr, _ = st.columns([1, 2])
            if col_tr.button(L["transcribe_btn"], key="sim_transcribe_btn"):
                if st.session_state.sim_audio_bytes is None:
                    st.warning("먼저 마이크로 녹음을 완료하세요.")
                else:
                    # ⭐ 수정: OpenAI 또는 Gemini API 키 체크
                    has_openai = st.session_state.openai_client is not None
                    has_gemini = bool(get_api_key("gemini"))
                    
                    if not has_openai and not has_gemini:
                        st.error(L["whisper_client_error"] + " (OpenAI 또는 Gemini API Key 필요)")
                    else:
                        with st.spinner(L["whisper_processing"]):
                            # transcribe_bytes_with_whisper 함수를 사용하도록 수정
                            # 자동 언어 감지 사용 (입력 언어와 관계없이 정확한 전사)
                            transcribed_text = transcribe_bytes_with_whisper(
                                st.session_state.sim_audio_bytes,
                                "audio/wav",
                                lang_code=None,
                                auto_detect=True,
                            )
                            if transcribed_text.startswith("❌"):
                                st.error(transcribed_text)
                                st.session_state.last_transcript = ""
                            else:
                                st.session_state.last_transcript = transcribed_text.strip()
                                # ⭐ 수정: 전사된 텍스트를 입력창의 세션 상태 변수에 반영
                                st.session_state.agent_response_area_text = transcribed_text.strip()
                                st.session_state.agent_response_input_box_widget = transcribed_text.strip()

                                snippet = transcribed_text[:50].replace("\n", " ")
                                if len(transcribed_text) > 50:
                                    snippet += "..."
                                st.success(L["whisper_success"] + f"\n\n**인식 내용:** *{snippet}*")
                                # ⭐ 수정: 전사 결과가 입력창에 반영되도록 UI 업데이트
                                # 전사 완료 후 다음 단계로 진행할 수 있도록 rerun
                                st.rerun()  # UI 업데이트

        col_text, col_button = st.columns([4, 1])

        # --- 입력 필드 및 버튼 ---
        with col_text:
            # ⭐ 수정: 위젯 생성 전에 초기화 플래그를 확인하여 값을 초기화합니다.
            if st.session_state.get("reset_agent_response_area", False):
                st.session_state.agent_response_area_text = ""
                st.session_state.reset_agent_response_area = False
            
            # st.text_area의 값을 읽어 세션 상태를 직접 업데이트하는 on_change를 제거하고
            # st.text_area 위젯 자체의 키를 사용하여 send_clicked 시 최신 값을 읽도록 합니다.
            # (Streamlit 기본 동작: 버튼 클릭 시 위젯의 최종 값이 세션 상태에 반영됨)
            # ⭐ 수정: key를 agent_response_area_text로 통일하여 세션 상태와 동기화
            agent_response_input = st.text_area(
                L["agent_response_placeholder"],
                value=st.session_state.agent_response_area_text,
                key="agent_response_area_text",  # 세션 상태 키와 동일하게 설정하여 동기화 보장
                height=150,
            )

            # 솔루션 제공 체크박스
            st.session_state.is_solution_provided = st.checkbox(
                L["solution_check_label"],
                value=st.session_state.is_solution_provided,
                key="solution_checkbox_widget",
            )

        with col_button:
            send_clicked = st.button(L["send_response_button"], key="send_agent_response_btn")

        if send_clicked:
            # ⭐ 수정: st.session_state.agent_response_area_text에서 최신 입력값을 가져옴 (key와 동일)
            agent_response = st.session_state.agent_response_area_text.strip()

            if not agent_response:
                st.warning(L["empty_response_warning"])
                # st.stop()

            # AHT 타이머 시작
            if st.session_state.start_time is None and len(st.session_state.simulator_messages) >= 1:
                st.session_state.start_time = datetime.now()

            # --- 에이전트 첨부 파일 처리 (다중 파일 처리) ---
            final_response_content = agent_response
            if st.session_state.agent_attachment_file:
                file_infos = st.session_state.agent_attachment_file
                file_names = ", ".join([f["name"] for f in file_infos])
                attachment_msg = L["agent_attachment_status"].format(
                    filename=file_names, filetype=f"총 {len(file_infos)}개 파일"
                )
                final_response_content = f"{agent_response}\n\n---\n{attachment_msg}"

            # 로그 업데이트
            st.session_state.simulator_messages.append(
                {"role": "agent_response", "content": final_response_content}
            )

            # 입력창/오디오/첨부 파일 초기화
            # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로,
            # rerun 후 위젯이 다시 생성될 때 초기값이 적용되도록 플래그를 사용합니다.
            st.session_state.sim_audio_bytes = None
            st.session_state.agent_attachment_file = []  # 첨부 파일 초기화
            st.session_state.language_transfer_requested = False
            st.session_state.realtime_hint_text = ""  # 힌트 초기화
            st.session_state.sim_call_outbound_summary = ""  # 전화 발신 요약 초기화

            # ⭐ 수정: 고객 반응 생성 로직을 다음 단계에서 처리하도록 sim_stage 변경
            st.session_state.sim_stage = "CUSTOMER_TURN"
            
            # ⭐ 수정: agent_response_area_text는 rerun 후 위젯이 다시 생성될 때 초기화되도록
            # 플래그를 설정합니다. 위젯 생성 전에 이 플래그를 확인하여 값을 초기화합니다.
            st.session_state.reset_agent_response_area = True
            
            # ⭐ 재실행: 이 부분이 즉시 고객 반응을 생성하도록 유도합니다.
            st.rerun()

        # --- 언어 이관 버튼 ---
        st.markdown("---")
        st.markdown(f"**{L['transfer_header']}**")
        transfer_cols = st.columns(len(LANG) - 1)

        languages = list(LANG.keys())
        languages.remove(current_lang)


        def transfer_session(target_lang: str, current_messages: List[Dict[str, str]]):
            """언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다."""

            # API 키 체크는 run_llm 내부에서 처리되지만, 명시적으로 Gemini 키를 요구함
            if not get_api_key("gemini"):
                st.error(LANG[current_lang]["simulation_no_key_warning"].replace('API Key', 'Gemini API Key'))
                # st.stop()
                return

            current_lang_at_start = st.session_state.language  # Source language

            # AHT 타이머 중지
            st.session_state.start_time = None

            # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
            with st.spinner(L["transfer_loading"]):
                # 실제 대기 시간 5~10초 (3~10분 시뮬레이션)
                time.sleep(np.random.uniform(5, 10))

                # 2. 대화 기록을 번역할 텍스트로 가공
                history_text = ""
                for msg in current_messages:
                    role = "Customer" if msg["role"].startswith("customer") or msg[
                        "role"] == "initial_query" else "Agent"
                    if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                       "customer_closing_response"]:
                        history_text += f"{role}: {msg['content']}\n"

                # ⭐ 수정: 먼저 핵심 포인트만 요약한 후 번역
                # 요약 프롬프트 생성
                lang_name_source = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(current_lang_at_start, "Korean")
                summary_prompt = f"""
You are an AI assistant that summarizes customer service conversations. 
Extract ONLY the key points from the conversation below. Keep it concise and focused on:
1. Customer's main inquiry/question
2. Key information provided by the agent
3. Important decisions or outcomes
4. Any unresolved issues

Write the summary in {lang_name_source}. Maximum 200 words. Be brief and to the point.

--- Conversation ---
{history_text}
---

Key Points Summary:
"""
                
                # 요약 생성
                summarized_text = ""
                if st.session_state.is_llm_ready:
                    try:
                        summarized_text = run_llm(summary_prompt).strip()
                    except Exception as e:
                        print(f"요약 생성 실패, 전체 대화 사용: {e}")
                        summarized_text = history_text  # 요약 실패 시 전체 대화 사용
                else:
                    summarized_text = history_text  # LLM이 없으면 전체 대화 사용

                # 3. LLM 번역 실행 (요약된 텍스트를 번역)
                translated_summary, is_success = translate_text_with_llm(summarized_text, target_lang,
                                                             current_lang_at_start)  # Use current_lang_at_start as source

                # 4. 세션 상태 업데이트
                st.session_state.transfer_summary_text = translated_summary
                st.session_state.translation_success = is_success
                st.session_state.language_at_transfer = target_lang  # Save destination language
                st.session_state.language_at_transfer_start = current_lang_at_start  # Save source language for retry
                st.session_state.language = target_lang  # Language switch

                # --- 기존 가이드라인 삭제 및 새 가이드라인 생성 (언어 통일성 확보) ---
                # 1. 기존 Supervisor Advice 메시지 삭제
                st.session_state.simulator_messages = [
                    msg for msg in st.session_state.simulator_messages
                    if msg['role'] != 'supervisor'
                ]

                # 2. 새로운 언어로 가이드라인/초안 재생성
                new_advice = _generate_initial_advice(
                    st.session_state.customer_query_text_area,
                    st.session_state.customer_type_sim_select,
                    st.session_state.customer_email,
                    st.session_state.customer_phone,
                    target_lang,  # 새로운 언어로 생성
                    st.session_state.customer_attachment_file
                )
                st.session_state.simulator_messages.append({"role": "supervisor", "content": new_advice})
                # -------------------------------------------------------------------

                st.session_state.is_solution_provided = False  # 새로운 응대를 위해 플래그 리셋
                st.session_state.language_transfer_requested = False  # 플래그 리셋
                st.session_state.sim_stage = "AGENT_TURN"

                # 5. 이력 저장
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display + f" (Transferred from {current_lang_at_start} to {target_lang})",
                    st.session_state.simulator_messages,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                    is_chat_ended=False,
                )

            # 6. UI 재실행 (언어 변경 적용)
            st.success(f"✅ {LANG[target_lang]['transfer_summary_header']}가 준비되었습니다. 새로운 응대를 시작하세요.")
            st.rerun()


        for i, target_lang in enumerate(languages):
            button_label_key = f"transfer_to_{target_lang}"
            button_label = L.get(button_label_key, f"Transfer to {target_lang.capitalize()} Team")

            if transfer_cols[i].button(button_label, key=f"btn_transfer_{target_lang}"):
                transfer_session(target_lang, st.session_state.simulator_messages)

        st.markdown("---")

    # --- Language Transfer Buttons End ---

    # =========================
    # 6. 고객 반응 생성 단계 (CUSTOMER_TURN)
    # =========================
    elif st.session_state.sim_stage == "CUSTOMER_TURN":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        st.info(L["customer_turn_info"])

        # 1. 고객 반응 생성
        with st.spinner(L["generating_customer_response"]):
            customer_response = generate_customer_reaction(st.session_state.language, is_call=False)

        # 2. 대화 로그 업데이트
        st.session_state.simulator_messages.append(
            {"role": "customer", "content": customer_response}
        )

        # 3. 종료 조건 검토
        positive_closing_phrases = [L["customer_positive_response"], L["customer_no_more_inquiries"]]
        is_positive_closing = any(phrase in customer_response for phrase in positive_closing_phrases)

        # ⭐ 수정: 고객이 "알겠습니다. 감사합니다"라고 답변했을 때, 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
        # 정확한 문자열 비교가 아닌 포함 여부로 확인 (LLM 응답이 약간 다를 수 있음)
        if L["customer_positive_response"] in customer_response:
            # 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # 솔루션이 제공되지 않은 경우 에이전트 턴으로 유지
                st.session_state.sim_stage = "AGENT_TURN"
        elif is_positive_closing:
            # 긍정 종료 응답 처리
            if L['customer_no_more_inquiries'] in customer_response:
                # ⭐ 수정: "없습니다. 감사합니다" 답변 시 에이전트가 감사 인사를 한 후 종료하도록 변경
                # 바로 종료하지 않고 WAIT_CLOSING_CONFIRMATION_FROM_AGENT 단계로 이동하여 에이전트가 감사 인사 후 종료
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # "알겠습니다. 감사합니다"와 유사한 긍정 응답인 경우, 솔루션 제공 여부 확인
                if st.session_state.is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    st.session_state.sim_stage = "AGENT_TURN"


        # ⭐ 수정: 고객이 아직 솔루션에 만족하지 않거나 추가 질문을 한 경우 (일반적인 턴)
        elif customer_response.startswith(L["customer_escalation_start"]):
            st.session_state.sim_stage = "ESCALATION_REQUIRED"  # 에스컬레이션 필요
        else:
            # 에이전트 턴으로 유지 (고객이 추가 질문하거나 정보 제공)
            st.session_state.sim_stage = "AGENT_TURN"

            # 4. 재실행

            st.session_state.is_solution_provided = False  # 종료 단계 진입 후 플래그 리셋

            # 이력 저장 (종료되지 않은 경우에만 저장)
        # ⭐ 수정: "없습니다. 감사합니다" 답변 시에는 이미 이력 저장을 했으므로 중복 저장 방지
        if st.session_state.sim_stage != "CLOSING":
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.session_state.realtime_hint_text = ""  # 힌트 초기화
        # ⭐ 재실행: 고객 반응이 추가되었으므로 상태 변경 반영
        st.rerun()


    # =========================
    # 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
        st.success("고객이 솔루션에 긍정적으로 반응했습니다. 추가 문의 여부를 확인해 주세요.")

        col_chat_end, col_email_end = st.columns(2)  # 버튼을 나란히 배치

        # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼
        with col_chat_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L["send_closing_confirm_button"],
                         key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}"):
                # ⭐ 수정: 에이전트가 감사 인사를 포함한 종료 메시지 전송
                # 언어별 감사 인사 메시지 생성
                agent_name = st.session_state.get("agent_name", "000")
                if current_lang == "ko":
                    closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. {L['customer_closing_confirm']} 즐거운 하루 되세요."
                elif current_lang == "en":
                    closing_msg = f"Thank you for contacting us. This was {agent_name}. {L['customer_closing_confirm']} Have a great day!"
                else:  # ja
                    closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。{L['customer_closing_confirm']} 良い一日をお過ごしください。"

                # 에이전트 응답으로 로그 기록
                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": closing_msg}
                )

                # [추가] TTS 버튼 렌더링을 위해 sleep/rerun 강제
                time.sleep(0.1)
                st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                st.rerun()

        # [2] 이메일 - 상담 종료 버튼 (즉시 종료)
        with col_email_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(L["button_email_end_chat"], key=f"btn_email_end_chat_{st.session_state.sim_instance_id}"):
                # AHT 타이머 정지
                st.session_state.start_time = None

                # [수정 1] 다국어 레이블 사용
                end_msg = L["prompt_survey"]
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": "(시스템: 이메일 상담 종료) " + end_msg}
                )

                # [추가] TTS 버튼 렌더링을 위해 sleep/rerun 강제
                time.sleep(0.1)
                st.session_state.is_chat_ended = True
                st.session_state.sim_stage = "CLOSING"  # 바로 CLOSING으로 전환
                st.rerun()

    # =========================
    # 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        
        # ⭐ 수정: 이미 고객 응답이 생성되어 있는지 확인
        last_customer_message = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "customer_rebuttal":
                last_customer_message = msg.get("content", "")
                break
        
        # 고객 응답이 아직 생성되지 않은 경우에만 생성
        if last_customer_message is None:
            st.info(L["agent_confirmed_inquiry"])

            # 고객 답변 자동 생성 (LLM Key 검증 포함)
            if not st.session_state.is_llm_ready:
                st.warning("LLM Key가 없어 고객 반응 자동 생성이 불가합니다. 수동으로 '고객 반응 생성' 버튼을 클릭하거나 AGENT_TURN으로 돌아가세요。")
                if st.button(L["customer_generate_response_button"], key="btn_generate_final_response"):
                    st.session_state.sim_stage = "AGENT_TURN"
                    st.rerun()
                st.stop()
            
            # LLM이 준비된 경우 고객 응답 생성
            with st.spinner(L["generating_customer_response"]):
                final_customer_reaction = generate_customer_closing_response(st.session_state.language)

            # 로그 기록
            st.session_state.simulator_messages.append(
                {"role": "customer_rebuttal", "content": final_customer_reaction}
            )
            last_customer_message = final_customer_reaction
        
        # 고객 응답에 따라 처리 (생성 직후 또는 이미 있는 경우 모두 처리)
        if last_customer_message is None:
            # 고객 응답이 없는 경우 (이미 생성했는데도 None인 경우는 에러)
            st.warning("고객 응답을 생성할 수 없습니다. 다시 시도해주세요.")
        else:
            final_customer_reaction = last_customer_message
            
            # (A) "없습니다. 감사합니다" 경로 -> 에이전트가 감사 인사 후 버튼 표시
            if L['customer_no_more_inquiries'] in final_customer_reaction:
                # ⭐ 수정: 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        # 이미 에이전트 감사 인사가 있는지 확인
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break
                
                if not agent_closing_added:
                    # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                    
                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )
                
                # ⭐ 수정: 현재 단계에서 바로 버튼 표시 (FINAL_CLOSING_ACTION으로 이동하지 않음)
                st.markdown("---")
                st.success(f"✅ {L['customer_no_more_inquiries_confirmed']}")
                st.markdown(f"### 📋 {L['consultation_end']}")
                st.info(L["consultation_end_instruction"])
                st.markdown("---")
                
                # 버튼을 중앙에 크게 표시
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    end_chat_button = st.button(
                        L["sim_end_chat_button"], 
                        key="btn_final_end_chat_in_wait", 
                        use_container_width=True, 
                        type="primary"
                    )
                
                if end_chat_button:
                    # AHT 타이머 정지
                    st.session_state.start_time = None

                    # 설문 조사 링크 전송 메시지 추가
                    end_msg = L["prompt_survey"]
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": end_msg}
                    )

                    # 채팅 종료 처리
                    st.session_state.is_chat_ended = True
                    st.session_state.sim_stage = "CLOSING"
                    
                    # 이력 저장
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area, customer_type_display,
                        st.session_state.simulator_messages, is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )
                    
                    st.session_state.realtime_hint_text = ""  # 힌트 초기화
                    st.rerun()
            # (B) "추가 문의 사항도 있습니다" 경로 -> AGENT_TURN으로 복귀
            elif L['customer_has_additional_inquiries'] in final_customer_reaction:
                st.session_state.sim_stage = "AGENT_TURN"
                save_simulation_history_local(
                    st.session_state.customer_query_text_area, customer_type_display,
                    st.session_state.simulator_messages, is_chat_ended=False,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                st.session_state.realtime_hint_text = ""
                st.rerun()

    # =========================
    # 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
    # =========================
    elif st.session_state.sim_stage == "FINAL_CLOSING_ACTION":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        
        # ⭐ 수정: 명확한 안내 메시지와 함께 버튼 표시
        st.markdown("---")
        st.success(f"✅ {L['customer_no_more_inquiries_confirmed']}")
        st.markdown(f"### 📋 {L['consultation_end']}")
        st.info(L["consultation_end_instruction"])
        st.markdown("---")
        
        # 버튼을 중앙에 크게 표시
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            end_chat_button = st.button(
                L["sim_end_chat_button"], 
                key="btn_final_end_chat", 
                use_container_width=True, 
                type="primary"
            )
        
        if end_chat_button:
            # AHT 타이머 정지
            st.session_state.start_time = None

            # 설문 조사 링크 전송 메시지 추가
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

            # 채팅 종료 처리
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"
            
            # 이력 저장
            customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
            save_simulation_history_local(
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )
            
            st.session_state.realtime_hint_text = ""  # 힌트 초기화
            st.rerun()

# ========================================
# 전화 시뮬레이터 로직
# ========================================

elif feature_selection == L["sim_tab_phone"]:
    st.header(L["phone_header"])
    st.markdown(L["simulator_desc"])

    current_lang = st.session_state.language
    L = LANG[current_lang]



    # ========================================
    # AHT 타이머 (IN_CALL 상태에서만 동작)
    # ========================================
    if st.session_state.call_sim_stage == "IN_CALL":
        # AHT 타이머 계산 로직
        col_timer, col_duration = st.columns([1, 4])

        if st.session_state.start_time is not None:
            now = datetime.now()

            # Hold 중이라면, Hold 상태가 된 이후의 시간을 현재 total_hold_duration에 더하지 않음 (Resume 시 정산)
            if st.session_state.is_on_hold and st.session_state.hold_start_time:
                # Hold 중이지만 AHT 타이머는 계속 흘러가야 하므로, Hold 시간은 제외하지 않고 최종 AHT 계산에만 사용
                elapsed_time_total = now - st.session_state.start_time
            else:
                elapsed_time_total = now - st.session_state.start_time

            # ⭐ AHT는 통화 시작부터 현재까지의 총 경과 시간입니다.
            total_seconds = elapsed_time_total.total_seconds()
            total_seconds = max(0, total_seconds)  # 음수 방지

            # 시간 형식 포맷팅
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 경고 기준
            if total_seconds > 900:  # 15분
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:  # 10분
                delta_str = L["timer_info_warn"]
                delta_color = "off"
            else:
                delta_str = L["timer_info_ok"]
                delta_color = "normal"

                with col_timer:
                    # AHT 타이머 표시
                    st.metric(L["timer_metric"], time_str, delta=delta_str, delta_color=delta_color)

                # ⭐ 수정: AHT 타이머 실시간 갱신을 위한 강제 재실행 로직 추가
                # 통화 중이고, Hold 상태가 아닐 때만 1초마다 업데이트하여 실시간성을 확보
                if not st.session_state.is_on_hold and total_seconds < 1000:
                    time.sleep(1)

        # ========================================
        # 화면 구분 (애니메이션 / CC)
        # ========================================
    col_video, col_cc = st.columns([1, 2])

    with col_video:
        st.subheader(f"📺 {L['customer_video_simulation']}")

        if st.session_state.call_sim_stage == "WAITING_CALL":
            st.info("통화 수신 대기 중...")

        elif st.session_state.call_sim_stage == "CALL_ENDED":
            st.info("통화 종료")

        else:
            # ⭐ 비디오 파일 업로드 옵션 추가 (로컬 경로 지원)
            # 항상 펼쳐진 상태로 표시하여 비디오를 쉽게 확인할 수 있도록 함
            with st.expander(L["video_upload_expander"], expanded=True):
                # 비디오 동기화 활성화 여부
                st.session_state.is_video_sync_enabled = st.checkbox(
                    L["video_sync_enable"],
                    value=st.session_state.is_video_sync_enabled,
                    key="video_sync_checkbox"
                )
                
                # OpenAI/Gemini 기반 영상 RAG 설명
                st.markdown("---")
                st.markdown(f"**{L['video_rag_title']}**")
                st.success(L["video_rag_desc"])
                
                # 가상 휴먼 기술은 현재 비활성화 (OpenAI/Gemini 기반 영상 RAG 사용)
                st.session_state.virtual_human_enabled = False
                
                # 성별 및 감정 상태별 비디오 업로드
                st.markdown(f"**{L['video_gender_emotion_setting']}**")
                col_gender_video, col_emotion_video = st.columns(2)
                
                with col_gender_video:
                    video_gender = st.radio(L["video_gender_label"], [L["video_gender_male"], L["video_gender_female"]], key="video_gender_select", horizontal=True)
                    gender_key = "male" if video_gender == L["video_gender_male"] else "female"
                
                with col_emotion_video:
                    video_emotion = st.selectbox(
                        L["video_emotion_label"],
                        ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"],
                        key="video_emotion_select"
                    )
                    emotion_key = video_emotion.lower()
                
                # 해당 조합의 비디오 업로드
                video_key = f"video_{gender_key}_{emotion_key}"
                uploaded_video = st.file_uploader(
                    L["video_upload_label"].format(gender=video_gender, emotion=video_emotion),
                    type=["mp4", "webm", "ogg"],
                    key=f"customer_video_uploader_{gender_key}_{emotion_key}"
                )
                
                # ⭐ Gemini 제안: 바이트 데이터를 세션 상태에 직접 저장 (파일 저장은 옵션)
                upload_key = f"last_uploaded_video_{gender_key}_{emotion_key}"
                video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"  # 바이트 데이터 저장 키
                
                if uploaded_video is not None:
                    # 파일이 새로 업로드되었는지 확인 (파일명으로 비교)
                    current_upload_name = uploaded_video.name if hasattr(uploaded_video, 'name') else None
                    last_upload_info = st.session_state.get(upload_key, None)
                    # 딕셔너리인 경우 'name' 키에서 파일명 가져오기
                    if isinstance(last_upload_info, dict):
                        last_upload_name = last_upload_info.get('name', None)
                    else:
                        last_upload_name = last_upload_info
                    
                    # 새 파일이거나 이전과 다른 파일인 경우에만 저장
                    if current_upload_name != last_upload_name:
                        try:
                            # 업로드된 비디오를 즉시 읽기 (rerun 전에 처리)
                            video_bytes = uploaded_video.read()
                            current_upload_size = len(video_bytes)
                            
                            if not video_bytes or len(video_bytes) == 0:
                                st.error(L["video_empty_error"])
                            else:
                                # 파일명 및 확장자 결정
                                uploaded_filename = uploaded_video.name if hasattr(uploaded_video, 'name') else f"{gender_key}_{emotion_key}.mp4"
                                file_ext = os.path.splitext(uploaded_filename)[1].lower() if uploaded_filename else ".mp4"
                                if file_ext not in ['.mp4', '.webm', '.ogg', '.mpeg4']:
                                    file_ext = ".mp4"
                                
                                # MIME 타입 결정
                                mime_type = uploaded_video.type if hasattr(uploaded_video, 'type') else f"video/{file_ext.lstrip('.')}"
                                if not mime_type or mime_type == "application/octet-stream":
                                    mime_type = f"video/{file_ext.lstrip('.')}"
                                
                                # ⭐ 1차 해결책: 바이트 데이터를 세션 상태에 직접 저장 (가장 안정적)
                                st.session_state[video_bytes_key] = video_bytes
                                st.session_state[video_key] = video_bytes_key  # 경로 대신 바이트 키 저장
                                st.session_state[upload_key] = {
                                    'name': current_upload_name,
                                    'size': current_upload_size,
                                    'mime': mime_type,
                                    'ext': file_ext
                                }
                                
                                file_size_mb = current_upload_size / (1024 * 1024)
                                st.success(L["video_bytes_saved"].format(name=current_upload_name, size=f"{file_size_mb:.2f}"))
                                
                                # ⭐ 즉시 미리보기 (바이트 데이터 직접 사용)
                                try:
                                    st.video(video_bytes, format=mime_type, autoplay=False, loop=False, muted=False)
                                except Exception as video_error:
                                    st.warning(f"⚠️ {L.get('video_preview_error', '비디오 미리보기 오류')}: {video_error}")
                                    # MIME 타입을 기본값으로 재시도
                                    try:
                                        st.video(video_bytes, format=f"video/{file_ext.lstrip('.')}", autoplay=False, loop=False, muted=False)
                                    except:
                                        st.error(L["video_playback_error"])
                                
                                # ⭐ 옵션: 파일 저장도 시도 (백업용, 실패해도 바이트는 이미 저장됨)
                                try:
                                    video_dir = os.path.join(DATA_DIR, "videos")
                                    os.makedirs(video_dir, exist_ok=True)
                                    video_filename = f"{gender_key}_{emotion_key}{file_ext}"
                                    video_path = os.path.join(video_dir, video_filename)
                                    
                                    # 파일 저장 시도 (권한 문제가 있어도 바이트는 이미 저장됨)
                                    try:
                                        with open(video_path, "wb") as f:
                                            f.write(video_bytes)
                                            f.flush()
                                        st.info(f"📂 파일도 저장됨: {video_path}")
                                    except Exception as save_error:
                                        st.info(f"💡 파일 저장은 건너뛰었습니다 (바이트 데이터는 메모리에 저장됨): {save_error}")
                                except:
                                    pass  # 파일 저장 실패해도 바이트는 이미 저장됨
                                
                                # rerun 호출 (업로드된 파일 객체가 사라지기 전에 처리 완료)
                                st.rerun()
                                
                        except Exception as e:
                            st.error(L["video_upload_error"].format(error=str(e)))
                            import traceback
                            st.code(traceback.format_exc())
                
                # 업로드된 비디오가 있으면 현재 선택된 조합의 비디오 표시
                st.markdown("---")
                st.markdown(f"**{L['video_current_selection'].format(gender=video_gender, emotion=video_emotion)}**")
                
                # ⭐ Gemini 제안: 세션 상태에서 바이트 데이터 직접 조회
                video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"
                current_video_bytes = st.session_state.get(video_bytes_key, None)
                
                if current_video_bytes:
                    # 바이트 데이터가 있으면 직접 사용 (가장 안정적)
                    upload_info = st.session_state.get(upload_key, {})
                    mime_type = upload_info.get('mime', 'video/mp4')
                    file_ext = upload_info.get('ext', '.mp4')
                    
                    st.success(f"✅ 비디오 바이트 데이터 발견: {upload_info.get('name', '업로드된 비디오')}")
                    try:
                        st.video(current_video_bytes, format=mime_type, autoplay=False, loop=False, muted=False)
                        st.caption(L["video_auto_play_info"].format(gender=video_gender, emotion=video_emotion))
                    except Exception as e:
                        st.warning(f"비디오 재생 오류: {e}")
                        # MIME 타입을 기본값으로 재시도
                        try:
                            st.video(current_video_bytes, format=f"video/{file_ext.lstrip('.')}", autoplay=False, loop=False, muted=False)
                        except:
                            st.error(L["video_playback_error"])
                else:
                    # 바이트 데이터가 없으면 파일 경로로 시도 (하위 호환성)
                    current_video_path = get_video_path_by_avatar(
                        gender_key,
                        video_emotion,
                        is_speaking=False,
                        gesture="NONE"
                    )
                    
                    if current_video_path and os.path.exists(current_video_path):
                        st.success(f"✅ 비디오 파일 발견: {os.path.basename(current_video_path)}")
                        try:
                            with open(current_video_path, "rb") as f:
                                existing_video_bytes = f.read()
                            st.video(existing_video_bytes, format="video/mp4", autoplay=False, loop=False, muted=False)
                            st.caption(L["video_auto_play_info"].format(gender=video_gender, emotion=video_emotion))
                        except Exception as e:
                            st.warning(f"비디오 재생 오류: {e}")
                    else:
                        st.info(L["video_upload_prompt"].format(filename=f"{gender_key}_{emotion_key}.mp4"))
                    
                    # 디버깅 정보: 비디오 디렉토리와 파일 목록 표시
                    video_dir = os.path.join(DATA_DIR, "videos")
                    st.caption(L["video_save_path"] + f" {video_dir}")
                    
                    if os.path.exists(video_dir):
                        all_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.webm', '.ogg'))]
                        if all_videos:
                            st.caption(f"{L['video_uploaded_files']} ({len(all_videos)}개):")
                            for vid in all_videos:
                                st.caption(f"  - {vid}")
                            
                            # 비슷한 비디오 파일이 있는지 확인
                            similar_videos = [
                                f for f in all_videos
                                if f.startswith(f"{gender_key}_") and f.endswith(('.mp4', '.webm', '.ogg'))
                            ]
                            if similar_videos:
                                st.caption(f"📁 {L.get('video_similar_gender', '같은 성별의 다른 비디오')}: {', '.join(similar_videos[:3])}")
                                st.caption(L.get("video_rename_hint", "💡 위 비디오 중 하나를 사용하려면 파일명을 변경하거나 새로 업로드하세요."))
                        else:
                            st.caption(L["video_directory_empty"])
                    else:
                        st.caption(L["video_directory_not_exist"].format(path=video_dir))
                
                # 또는 로컬 파일 경로 입력 및 복사
                video_path_input = st.text_input(
                    L["video_local_path_input"],
                    placeholder=L["video_local_path_placeholder"],
                    key="video_path_input"
                )
                
                if video_path_input:
                    try:
                        # ⭐ Gemini 제안: 절대 경로 검증 강화
                        if not os.path.isabs(video_path_input):
                            st.error("❌ 로컬 경로 입력 시 반드시 **절대 경로**를 사용해주세요 (예: C:\\Users\\...\\video.mp4).")
                            st.error("💡 Streamlit 앱이 실행되는 서버 환경과 파일 시스템이 다르면 접근할 수 없습니다.")
                            st.stop()
                        
                        source_video_path = video_path_input
                        
                        if not os.path.exists(source_video_path):
                            st.error(f"❌ 파일을 찾을 수 없습니다: {source_video_path}")
                            st.error("💡 파일 경로를 확인하고, Streamlit 앱이 실행되는 서버에서 접근 가능한 경로인지 확인해주세요.")
                            st.stop()
                        
                        # 원본 파일 읽기
                        with open(source_video_path, "rb") as f:
                            video_bytes = f.read()
                        
                        if len(video_bytes) == 0:
                            st.error("❌ 파일이 비어있습니다.")
                            st.stop()
                        
                        # 파일명 및 확장자 결정
                        source_filename = os.path.basename(source_video_path)
                        file_ext = os.path.splitext(source_filename)[1].lower()
                        if file_ext not in ['.mp4', '.webm', '.ogg', '.mpeg4']:
                            file_ext = ".mp4"
                        
                        mime_type = f"video/{file_ext.lstrip('.')}"
                        
                        # ⭐ 바이트 데이터를 세션 상태에 직접 저장 (파일 복사는 옵션)
                        video_bytes_key = f"video_bytes_{gender_key}_{emotion_key}"
                        st.session_state[video_bytes_key] = video_bytes
                        st.session_state[video_key] = video_bytes_key
                        st.session_state[upload_key] = {
                            'name': source_filename,
                            'size': len(video_bytes),
                            'mime': mime_type,
                            'ext': file_ext
                        }
                        
                        file_size_mb = len(video_bytes) / (1024 * 1024)
                        st.success(f"✅ 비디오 바이트 로드 완료: {source_filename} ({file_size_mb:.2f} MB)")
                        
                        # 비디오 미리보기 (바이트 데이터 직접 사용)
                        try:
                            st.video(video_bytes, format=mime_type, autoplay=False, loop=False, muted=False)
                        except Exception as video_error:
                            st.warning(f"⚠️ 비디오 미리보기 오류: {video_error}")
                        
                        # ⭐ 옵션: 파일 복사도 시도 (백업용)
                        try:
                            video_dir = os.path.join(DATA_DIR, "videos")
                            os.makedirs(video_dir, exist_ok=True)
                            video_filename = f"{gender_key}_{emotion_key}{file_ext}"
                            target_video_path = os.path.join(video_dir, video_filename)
                            
                            with open(target_video_path, "wb") as f:
                                f.write(video_bytes)
                                f.flush()
                            st.info(f"📂 파일도 복사됨: {target_video_path}")
                        except Exception as copy_error:
                            st.info(f"💡 파일 복사는 건너뛰었습니다 (바이트 데이터는 메모리에 저장됨): {copy_error}")
                        
                        # 입력 필드 초기화 및 rerun
                        st.session_state.video_path_input = ""
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 비디오 파일 로드 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # 상태 선택 및 비디오 표시
            st.markdown("---")
            st.markdown(f"**{L['video_current_avatar']}**")
            
            if st.session_state.is_on_hold:
                avatar_state = "HOLD"
            else:
                avatar_state = st.session_state.customer_avatar.get("state", "NEUTRAL")
            
            customer_gender = st.session_state.customer_avatar.get("gender", "male")
            
            # get_video_path_by_avatar 함수를 사용하여 비디오 경로 찾기
            video_path = get_video_path_by_avatar(
                customer_gender, 
                avatar_state, 
                is_speaking=False,  # 미리보기는 자동 재생하지 않음
                gesture="NONE"
            )
            
            # 비디오 표시
            if video_path and os.path.exists(video_path):
                try:
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    
                    # 비디오 정보 표시
                    avatar_emoji = {
                        "NEUTRAL": "😐",
                        "HAPPY": "😊",
                        "ANGRY": "😠",
                        "ASKING": "🤔",
                        "SAD": "😢",
                        "HOLD": "⏸️"
                    }.get(avatar_state, "😐")
                    
                    st.markdown(f"### {avatar_emoji} {customer_gender.upper()} - {avatar_state}")
                    st.caption(f"비디오: {os.path.basename(video_path)}")
                    
                    # 현재 말하는 중이면 자동 재생, 아니면 수동 재생
                    is_speaking = bool(
                        st.session_state.get("customer_initial_audio_bytes") or 
                        st.session_state.get("current_customer_audio_text")
                    )
                    
                    autoplay_video = st.session_state.is_video_sync_enabled and is_speaking
                    st.video(video_bytes, format="video/mp4", autoplay=autoplay_video, loop=False, muted=False)
                    
                except Exception as e:
                    st.warning(f"비디오 재생 오류: {e}")
                    avatar_emoji = {
                        "NEUTRAL": "😐",
                        "HAPPY": "😊",
                        "ANGRY": "😠",
                        "ASKING": "🤔",
                        "SAD": "😢",
                        "HOLD": "⏸️"
                    }.get(avatar_state, "😐")
                    st.markdown(f"### {avatar_emoji} {L['customer_avatar']}")
                    st.info(L.get("avatar_status_info", "상태: {state} | 성별: {gender}").format(state=avatar_state, gender=customer_gender))
            else:
                # 비디오가 없으면 이모지로 표시
                avatar_emoji = {
                    "NEUTRAL": "😐",
                    "HAPPY": "😊",
                    "ANGRY": "😠",
                    "ASKING": "🤔",
                    "SAD": "😢",
                    "HOLD": "⏸️"
                }.get(avatar_state, "😐")
                
                st.markdown(f"### {avatar_emoji} 고객 아바타")
                st.info(L.get("avatar_status_info", "상태: {state} | 성별: {gender}").format(state=avatar_state, gender=customer_gender))
                st.warning(L["video_avatar_upload_prompt"].format(filename=f"{customer_gender}_{avatar_state.lower()}.mp4"))
                
                # 업로드된 비디오 목록 표시
                video_dir = os.path.join(DATA_DIR, "videos")
                if os.path.exists(video_dir):
                    uploaded_videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.webm', '.ogg'))]
                    if uploaded_videos:
                        st.caption(f"{L['video_uploaded_files']}: {', '.join(uploaded_videos[:5])}")
                        if len(uploaded_videos) > 5:
                            st.caption(L.get("video_more_files", f"... 외 {len(uploaded_videos) - 5}개").format(count=len(uploaded_videos) - 5))

    with col_cc:
        # ⭐ 수정: "전화 수신 중" 메시지는 통화 중일 때만 표시
        if st.session_state.call_sim_stage == "IN_CALL":
            if st.session_state.call_sim_mode == "INBOUND":
                st.markdown(
                    f"## {L['call_status_ringing'].format(number=st.session_state.incoming_phone_number)}"
                )
            else:
                st.markdown(
                    f"## {L['button_call_outbound']} ({st.session_state.incoming_phone_number})"
                )
        st.markdown("---")

    # ========================================
    # WAITING / RINGING 상태
    # ========================================
    if st.session_state.call_sim_stage in ["WAITING_CALL", "RINGING"]:

        if "call_sim_mode" not in st.session_state:
            st.session_state.call_sim_mode = "INBOUND"  # INBOUND or OUTBOUND

        if st.session_state.call_sim_mode == "INBOUND":
            st.subheader(L["call_status_waiting"])
        else:
            st.subheader(L["button_call_outbound"])

        # 홈페이지 웹 주소 입력 (선택사항)
        st.session_state.call_website_url = st.text_input(
            L.get("website_url_label", "홈페이지 웹 주소 (선택사항)"),
            key="call_website_url_input",
            value=st.session_state.call_website_url,
            placeholder=L.get("website_url_placeholder", "https://example.com (홈페이지 주소가 있으면 입력하세요)"),
        )

        # 초기 문의 입력 (고객이 전화로 말할 내용)
        st.session_state.call_initial_query = st.text_area(
            L["customer_query_label"],
            key="call_initial_query_text_area",
            height=100,
            placeholder=L["call_query_placeholder"],
        )

        # 가상 전화번호 표시
        st.session_state.incoming_phone_number = st.text_input(
            "Incoming/Outgoing Phone Number",
            key="incoming_phone_number_input",
            value=st.session_state.incoming_phone_number,
            placeholder=L["call_number_placeholder"],
        )

        # 고객 유형 선택
        customer_type_options = L["customer_type_options"]
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="call_customer_type_sim_select_widget",
        )

        # ⭐ 추가: 고객 성별 및 감정 상태 설정
        col_gender, col_emotion = st.columns(2)
        
        with col_gender:
            # 고객 성별 선택
            if "customer_gender" not in st.session_state:
                st.session_state.customer_gender = "male"
            
            # ⭐ 수정: 번역 키 사용
            gender_options = [L["gender_male"], L["gender_female"]]
            current_gender = st.session_state.customer_avatar.get("gender", "male")
            default_gender_idx = 0 if current_gender == "male" else 1
            
            selected_gender_display = st.radio(
                L["customer_gender_label"],
                gender_options,
                index=default_gender_idx,
                key="call_customer_gender_radio",
                horizontal=True
            )
            # 세션 상태에 저장 (영어로)
            st.session_state.customer_avatar["gender"] = "male" if selected_gender_display == L["gender_male"] else "female"
            st.session_state.customer_gender = st.session_state.customer_avatar["gender"]
        
        with col_emotion:
            # 고객 감정 상태 선택
            # ⭐ 수정: 번역 키 사용
            emotion_options = [
                L["emotion_happy"],
                L["emotion_dissatisfied"],
                L["emotion_angry"],
                L["emotion_sad"],
                L["emotion_neutral"]
            ]
            emotion_mapping = {
                L["emotion_happy"]: "HAPPY",
                L["emotion_dissatisfied"]: "ASKING",
                L["emotion_angry"]: "ANGRY",
                L["emotion_sad"]: "SAD",
                L["emotion_neutral"]: "NEUTRAL"
            }
            
            current_emotion_state = st.session_state.customer_avatar.get("state", "NEUTRAL")
            default_emotion_idx = 4  # 기본값: 중립
            for i, (emotion_display, emotion_state) in enumerate(emotion_mapping.items()):
                if emotion_state == current_emotion_state:
                    default_emotion_idx = i
                    break
            
            selected_emotion = st.selectbox(
                L["customer_emotion_label"],
                emotion_options,
                index=default_emotion_idx,
                key="call_customer_emotion_select",
            )
            # 세션 상태에 저장
            st.session_state.customer_avatar["state"] = emotion_mapping.get(selected_emotion, "NEUTRAL")

        st.markdown("---")

        col_in, col_out = st.columns(2)

        # 전화 응답 (수신)
        with col_in:
            if st.button(L["button_answer"], key=f"answer_call_btn_{st.session_state.sim_instance_id}"):
                # 입력 검증
                if not st.session_state.call_initial_query.strip():
                    st.warning(L["simulation_warning_query"])
                    # st.stop()

                # ⭐ 수정: OpenAI 또는 Gemini API 키 체크
                has_openai = st.session_state.openai_client is not None
                has_gemini = bool(get_api_key("gemini"))
                
                if not st.session_state.is_llm_ready or (not has_openai and not has_gemini):
                    st.error(L["simulation_no_key_warning"] + " (OpenAI 또는 Gemini API Key 필요)")
                    # st.stop()

                # INBOUND 모드 설정
                st.session_state.call_sim_mode = "INBOUND"

                # 시뮬레이션 초기화 및 시작
                st.session_state.call_sim_stage = "IN_CALL"
                st.session_state.is_call_ended = False
                st.session_state.is_on_hold = False
                st.session_state.total_hold_duration = timedelta(0)
                st.session_state.hold_start_time = None
                st.session_state.start_time = datetime.now()  # 통화 시작 시간 (AHT 시작)
                st.session_state.simulator_messages = []
                st.session_state.current_customer_audio_text = ""
                st.session_state.current_agent_audio_text = ""
                st.session_state.agent_response_input_box_widget_call = ""
                st.session_state.sim_instance_id = str(uuid.uuid4())
                st.session_state.call_summary_text = ""  # 요약 초기화
                st.session_state.customer_initial_audio_bytes = None  # 오디오 초기화
                st.session_state.customer_history_summary = ""  # AI 요약 초기화 (추가)
                st.session_state.sim_audio_bytes = None  # 녹음 파일 초기화 (추가)

                # ⭐ 수정: 자동 인사말 생성 제거 - 에이전트가 직접 녹음하도록 변경
                st.session_state.just_entered_call = False
                st.session_state.customer_turn_start = False  # 에이전트 인사말 완료 전까지 False

                # 고객의 첫 번째 음성 메시지 (시뮬레이션 시작 메시지)
                initial_query_text = st.session_state.call_initial_query.strip()
                st.session_state.current_customer_audio_text = initial_query_text

                # ⭐ 입력 텍스트의 언어를 자동 감지 및 언어 설정 업데이트
                try:
                    detected_lang = detect_text_language(initial_query_text)
                    if detected_lang in ["ko", "en", "ja"] and detected_lang != st.session_state.language:
                        st.session_state.language = detected_lang
                        st.info(f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
                except Exception as e:
                    print(f"Language detection failed in call: {e}")
                    detected_lang = st.session_state.language

                # ⭐ 고객의 첫 문의 TTS 음성 생성 및 저장 (감지된 언어 사용)
                with st.spinner(L["tts_status_generating"] + " (Initial Customer Query)"):
                    audio_bytes, msg = synthesize_tts(initial_query_text, st.session_state.language, role="customer")
                    if audio_bytes:
                        st.session_state.customer_initial_audio_bytes = audio_bytes
                    else:
                        st.error(f"❌ {msg}")
                        st.session_state.customer_initial_audio_bytes = None

                # ✅ 상태 변경 후 재실행하여 IN_CALL 상태로 전환
                # 에이전트가 인사말을 녹음할 수 있도록 안내 메시지 표시
                st.info(L["call_started_message"])
                st.rerun()

        # 전화 발신 (새로운 세션 시작)
        with col_out:
            st.markdown(f"### {L['button_call_outbound']}")
            call_targets = [
                L["call_target_customer"],
                L["call_target_partner"]
            ]

            call_target_selection = st.radio(
                L.get("call_target_select_label", "발신 대상 선택"),
                call_targets,
                key="outbound_call_target_radio",
                horizontal=True
            )

            # 선택된 대상에 따라 버튼 텍스트 변경
            if call_target_selection == L["call_target_customer"]:
                button_text = L["button_call_outbound_to_customer"]
            else:
                button_text = L["button_call_outbound_to_provider"]

            if st.button(button_text, key=f"outbound_call_start_btn_{st.session_state.sim_instance_id}", type="secondary", use_container_width=True):
                # 입력 검증
                if not st.session_state.call_initial_query.strip():
                    st.warning("전화 발신 목표 (고객 문의 내용)를 입력해 주세요。")
                    # st.stop()

                # ⭐ 수정: OpenAI 또는 Gemini API 키 체크
                has_openai = st.session_state.openai_client is not None
                has_gemini = bool(get_api_key("gemini"))
                
                if not st.session_state.is_llm_ready or (not has_openai and not has_gemini):
                    st.error(L["simulation_no_key_warning"] + " (OpenAI 또는 Gemini API Key 필요)")
                    # st.stop()

                # OUTBOUND 모드 설정 및 시뮬레이션 시작
                st.session_state.call_sim_mode = "OUTBOUND"

                # 시뮬레이션 초기화 및 시작
                st.session_state.call_sim_stage = "IN_CALL"
                st.session_state.is_call_ended = False
                st.session_state.is_on_hold = False
                st.session_state.total_hold_duration = timedelta(0)
                st.session_state.hold_start_time = None
                st.session_state.start_time = datetime.now()  # 통화 시작 시간 (AHT 시작)
                st.session_state.simulator_messages = []

                # ⭐ 수정: 자동 인사말 생성 제거 - 에이전트가 직접 녹음하도록 변경
                st.session_state.just_entered_call = False
                st.session_state.customer_turn_start = False

                initial_query_text = st.session_state.call_initial_query.strip()

                # 발신 시뮬레이션에서는 에이전트가 먼저 말해야 하므로, 고객 CC 텍스트는 안내 메시지로 설정
                st.session_state.current_customer_audio_text = f"📞 {L['button_call_outbound']} 성공! {call_target_selection}이(가) 받았습니다。 잠시 후 응답이 시작됩니다。 (문의 목표: {initial_query_text[:50]}...)"
                st.session_state.current_agent_audio_text = ""  # Agent speaks first
                st.session_state.agent_response_input_box_widget_call = ""
                st.session_state.sim_instance_id = str(uuid.uuid4())
                st.session_state.call_summary_text = ""
                st.session_state.customer_initial_audio_bytes = None
                st.session_state.customer_history_summary = ""
                st.session_state.sim_audio_bytes = None

                st.success(f"'{call_target_selection}'에게 전화 발신 시뮬레이션이 시작되었습니다. 아래 마이크 버튼을 눌러 인사말을 녹음하세요。")
                st.rerun()

        # ------------------
        # IN_CALL 상태 (통화 중)
        # ------------------
    elif st.session_state.call_sim_stage == "IN_CALL":
        # ⭐ 수정: 자동 인사말 생성 로직 제거 - 에이전트가 직접 녹음하도록 변경
        
        # ------------------------------
        # 전화 통화 제목 (통화 중일 때만 표시)
        # ------------------------------
        # ⭐ 수정: 제목은 이미 위에서 표시되므로 여기서는 제거
        # st.markdown(f"## {title}")
        # st.markdown("---")

        # ------------------------------
        # Hangup / Hold 버튼
        # ------------------------------
        col_hangup, col_hold = st.columns(2)

        with col_hangup:
            if st.button(L["button_hangup"], key="hangup_call_btn"):

                # Hold 정산
                if st.session_state.is_on_hold and st.session_state.hold_start_time:
                    st.session_state.total_hold_duration += datetime.now() - st.session_state.hold_start_time

                # 요약 생성
                with st.spinner("AI 요약 생성 중..."):
                    # ⭐ [수정 9] 함수명 통일: summarize_history_for_call로 변경 및 호출
                    summary = summarize_history_for_call(
                        st.session_state.simulator_messages,
                        st.session_state.call_initial_query,
                        st.session_state.language
                    )
                    st.session_state.call_summary_text = summary

                # 종료
                st.session_state.call_sim_stage = "CALL_ENDED"
                st.session_state.is_call_ended = True

                # ⭐ [수정 10] Hangup 후 UI 갱신을 위해 rerun 추가
                st.rerun()

        # ------------------------------
        # Hold / Resume
        # ------------------------------
        with col_hold:
            if st.session_state.is_on_hold:
                if st.button(L["button_resume"], key="resume_call_btn"):
                    # Hold 상태 해제 및 시간 정산
                    st.session_state.is_on_hold = False
                    if st.session_state.hold_start_time:
                        st.session_state.total_hold_duration += datetime.now() - st.session_state.hold_start_time
                        st.session_state.hold_start_time = None
                    st.rerun()  # ⭐ [수정 10] Resume 후 UI 갱신을 위해 rerun 추가
            else:
                if st.button(L["button_hold"], key="hold_call_btn"):
                    st.session_state.is_on_hold = True
                    st.session_state.hold_start_time = datetime.now()
                    st.rerun()  # ⭐ [수정 10] Hold 후 UI 갱신을 위해 rerun 추가

        # ------------------------------
        # Hold 표시
        # ------------------------------
        if st.session_state.is_on_hold:
            if st.session_state.hold_start_time:
                current_hold = datetime.now() - st.session_state.hold_start_time
            else:
                current_hold = timedelta(0)

            total_hold = st.session_state.total_hold_duration + current_hold
            hold_str = str(total_hold).split('.')[0]

            st.warning(L["hold_status"].format(duration=hold_str))
            time.sleep(1)

        # ------------------------------
        # (중략) - **이관, 힌트, 요약, CC, Whisper 전사, 고객 반응 생성**
        # ------------------------------
        def transfer_session(target_lang: str, current_messages: List[Dict[str, str]]):
            """언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다."""

            current_lang = st.session_state.language  # 현재 언어 확인 (Source language)
            L = LANG[current_lang]

            # API 키 체크
            if not st.session_state.is_llm_ready:
                st.error(L["simulation_no_key_warning"].replace('API Key', 'LLM API Key'))
                return

            current_lang_at_start = st.session_state.language  # Source language

            # AHT 타이머 정지 (실제로 통화가 종료되는 것은 아니므로, AHT는 계속 흐름)
            # st.session_state.start_time = None

            # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
            with st.spinner(L["transfer_loading"]):
                time.sleep(np.random.uniform(5, 10))

                # 2. 대화 기록을 번역할 텍스트로 가공
                history_text = ""
                for msg in current_messages:
                    role = "Customer" if msg["role"].startswith("customer") or msg[
                        "role"] == "initial_query" else "Agent"
                    if msg["role"] in ["initial_query", "customer_rebuttal", "agent_response",
                                       "customer_closing_response", "phone_exchange"]:  # phone_exchange 추가
                        history_text += f"{role}: {msg['content']}\n"

                # ⭐ 수정: 먼저 핵심 포인트만 요약한 후 번역
                # 요약 프롬프트 생성
                lang_name_source = {"ko": "Korean", "en": "English", "ja": "Japanese"}.get(current_lang_at_start, "Korean")
                summary_prompt = f"""
You are an AI assistant that summarizes customer service conversations. 
Extract ONLY the key points from the conversation below. Keep it concise and focused on:
1. Customer's main inquiry/question
2. Key information provided by the agent
3. Important decisions or outcomes
4. Any unresolved issues

Write the summary in {lang_name_source}. Maximum 200 words. Be brief and to the point.

--- Conversation ---
{history_text}
---

Key Points Summary:
"""
                
                # 요약 생성
                summarized_text = ""
                if st.session_state.is_llm_ready:
                    try:
                        summarized_text = run_llm(summary_prompt).strip()
                    except Exception as e:
                        print(f"요약 생성 실패, 전체 대화 사용: {e}")
                        summarized_text = history_text  # 요약 실패 시 전체 대화 사용
                else:
                    summarized_text = history_text  # LLM이 없으면 전체 대화 사용

                # 3. LLM 번역 실행 (요약된 텍스트를 번역)
                translated_summary, is_success = translate_text_with_llm(summarized_text, target_lang,
                                                             current_lang_at_start)  # Use current_lang_at_start as source

                # 4. 세션 상태 업데이트
                st.session_state.transfer_summary_text = translated_summary
                st.session_state.translation_success = is_success
                st.session_state.language_at_transfer = target_lang  # Save destination language
                st.session_state.language_at_transfer_start = current_lang_at_start  # Save source language for retry
                st.session_state.language = target_lang  # Language switch

                # --- 시스템 이관 메시지 추가 ---
                # 전화에서는 별도의 Supervisor 메시지 없이 로그에만 남김
                st.session_state.simulator_messages.append(
                    {"role": "system_transfer",
                     "content": LANG[target_lang]['transfer_system_msg'].format(target_lang=target_lang)})

                st.session_state.is_solution_provided = False
                st.session_state.language_transfer_requested = False

                # 이관 후 상태 전환: 통화 중인 상태는 유지
                st.session_state.call_sim_stage = "IN_CALL"

                # 5. 이력 저장
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.call_initial_query,
                    customer_type_display + f" (Transferred from {current_lang_at_start} to {target_lang})",
                    st.session_state.simulator_messages,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                    is_chat_ended=False,
                    is_call=(st.session_state.call_sim_stage == "IN_CALL")  # 전화 이력임을 표시
                )

            # 6. UI 재실행 (언어 변경 적용)
            st.success(f"✅ {LANG[target_lang]['transfer_summary_header']}가 준비되었습니다. 새로운 응대를 시작하세요.")
            st.rerun()


        st.markdown("---")
        st.markdown(f"**{L['transfer_header']}**")
        transfer_cols = st.columns(len(LANG) - 1)

        languages = list(LANG.keys())
        languages.remove(current_lang)

        # transfer_session 함수를 재정의하지 않고, 기존의 transfer_session 함수를 호출합니다.
        for i, target_lang in enumerate(languages):
            button_label_key = f"transfer_to_{target_lang}"
            button_label = L.get(button_label_key, f"Transfer to {target_lang.capitalize()} Team")

            # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID와 대상 언어를 조합하여 고유 키 생성
            if transfer_cols[i].button(button_label, key=f"btn_transfer_phone_{target_lang}_{st.session_state.sim_instance_id}"):
                # transfer_session 호출 시, 현재 통화 메시지(simulator_messages)를 넘겨줍니다.
                transfer_session(target_lang, st.session_state.simulator_messages)

        # =========================
        # AI 요약 버튼 및 표시 로직 (추가된 기능)
        # =========================
        st.markdown("---")
        # ⭐ history_expander_title에서 괄호 안 내용만 제거 (예: (최근 10건))
        summary_title = L['history_expander_title'].split('(')[0].strip()
        st.markdown(f"### 📑 {summary_title} 요약")

        # 1. 요약/번역 재시도 버튼 영역
        col_sum_btn, col_trans_btn = st.columns(2)

        with col_sum_btn:
            # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID를 키에 추가
            if st.button(L["btn_request_phone_summary"], key=f"btn_request_phone_summary_{st.session_state.sim_instance_id}"):
                # 요약 함수 호출
                st.session_state.customer_history_summary = summarize_history_with_ai(st.session_state.language)
                # ⭐ 최적화: 요약 생성 후 UI 업데이트를 위해 rerun 필요 (유지)
                st.rerun()

        # 2. 이관 번역 재시도 버튼 (이관 후 번역이 실패했을 경우)
        if st.session_state.language != st.session_state.language_at_transfer_start and not st.session_state.transfer_summary_text:
            with col_trans_btn:
                # ⭐ [수정 FIX] 키 중복 오류 해결: 세션 ID와 언어 코드를 조합하여 고유 키 생성
                retry_key = f"btn_retry_translation_{st.session_state.language_at_transfer_start}_{st.session_state.language}_{st.session_state.sim_instance_id}"
                if st.button(L["button_retry_translation"], key=retry_key):
                    with st.spinner(L["transfer_loading"]):
                        # 이관 번역 로직 재실행 (기존 로직 유지)
                        translated_summary, is_success = translate_text_with_llm(
                            get_chat_history_for_prompt(include_attachment=False),
                            st.session_state.language,
                            st.session_state.language_at_transfer_start
                        )
                        st.session_state.transfer_summary_text = translated_summary
                        st.session_state.translation_success = is_success
                        # ⭐ 최적화: 번역 재시도 후 UI 업데이트를 위해 rerun 필요 (유지)
                        st.rerun()

        # 3. 요약 내용 표시
        if st.session_state.transfer_summary_text:
            st.subheader(f"🔍 {L['transfer_summary_header']}")
            st.info(st.session_state.transfer_summary_text)
            # ⭐ 이관 요약에 TTS 버튼 추가
            render_tts_button(
                st.session_state.transfer_summary_text,
                st.session_state.language,
                role="agent",
                prefix="trans_summary_tts_call",
                index=-1  # 고유 세션 ID 기반의 키를 생성하도록 지시
            )
        elif st.session_state.customer_history_summary:
            st.subheader("💡 AI 요약")
            st.info(st.session_state.customer_history_summary)

        st.markdown("---")

        # --- 실시간 응대 힌트 영역 ---
        hint_cols = st.columns([4, 1])
        with hint_cols[0]:
            st.info(L["hint_placeholder"] + st.session_state.realtime_hint_text)

        with hint_cols[1]:
            # 힌트 요청 버튼
            if st.button(L["button_request_hint"], key=f"btn_request_hint_call_{st.session_state.sim_instance_id}"):
                with st.spinner(L["response_generating"]):
                    # 전화 탭이므로 is_call=True
                    hint = generate_realtime_hint(current_lang, is_call=True)
                    st.session_state.realtime_hint_text = hint
                    # ⭐ 최적화: 힌트 생성 후 UI 업데이트를 위해 rerun 필요 (유지)
                    st.rerun()

        # =========================
        # CC 자막 / 음성 입력 및 제어 로직 (기존 로직)
        # =========================================

        # --- 실시간 CC 자막 / 전사 영역 ---
        st.subheader(L["cc_live_transcript"])

        if st.session_state.is_on_hold:
            st.text_area("Customer", value=L["customer_waiting_hold"], height=50, disabled=True, key="customer_live_cc_area")
            st.text_area("Agent", value=L["agent_hold_message"], height=50, disabled=True,
                         key="agent_live_cc_area")
        else:
            # 고객 CC (LLM 생성 텍스트 또는 초기 문의)
            # ⭐ 수정: 고객 문의가 비어있지 않으면 초기 문의를 표시
            customer_cc_text = st.session_state.current_customer_audio_text
            if not customer_cc_text and st.session_state.call_initial_query:
                customer_cc_text = st.session_state.call_initial_query
            st.text_area(
                "Customer",
                value=customer_cc_text,
                height=50,
                disabled=True,
                key="customer_live_cc_area",
            )

            # 에이전트 CC (마이크 전사)
            st.text_area(
                "Agent",
                value=st.session_state.current_agent_audio_text,
                height=50,
                disabled=True,
                key="agent_live_cc_area",
            )

        st.markdown("---")

        # --- 에이전트 음성 입력 / 녹음 ---
        st.subheader(L["mic_input_status"])

        # 음성 입력: 짧은 청크로 끊어서 전사해야 실시간 CC 모방 가능
        if st.session_state.is_on_hold:
            st.info(L["call_on_hold_message"])
            mic_audio = None
        else:
            # ✅ 마이크 위젯을 항상 렌더링하여 활성화 상태를 유지
            mic_audio = mic_recorder(
                start_prompt=L["agent_response_prompt"],
                stop_prompt=L["agent_response_stop_and_send"],
                just_once=True,
                format="wav",
                use_container_width=True,
                key="call_sim_mic_recorder",
            )

            # 녹음 완료 (mic_audio.get("bytes")가 채워짐) 시, 바이트를 저장하고 재실행
            # ⭐ 수정: 채팅/이메일 탭과 동일한 패턴으로 수정 - 조건 단순화
            if mic_audio and mic_audio.get("bytes"):
                # ⭐ 수정: 이미 처리 중인 경우 중복 처리 방지
                if "bytes_to_process" not in st.session_state or st.session_state.bytes_to_process is None:
                    st.session_state.bytes_to_process = mic_audio["bytes"]
                    st.session_state.current_agent_audio_text = L["recording_complete_transcribing"]
                    # ✅ 재실행하여 다음 실행 주기에서 전사 로직을 처리
                    st.rerun()

        # ⭐ 수정: 전사 로직을 마이크 위젯 렌더링 블록 밖으로 이동하여 실행 순서 보장
        # 전사 로직: bytes_to_process에 데이터가 있을 때만 실행
        if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process is not None:
            # ⭐ 수정: OpenAI 또는 Gemini API 키가 있는지 확인
            has_openai = st.session_state.openai_client is not None
            has_gemini = bool(get_api_key("gemini"))
            
            if not has_openai and not has_gemini:
                st.error(L["openai_missing"] + " 또는 Gemini API Key가 필요합니다.")
                st.session_state.bytes_to_process = None
                # ⭐ 최적화: 에러 메시지 표시 후 불필요한 rerun 제거 (사용자가 API 키를 설정하면 자동으로 재실행됨)
            else:
                # ⭐ 전사 결과를 저장할 변수 초기화
                agent_response_transcript = None

                # ⭐ [수정]: Whisper 전사 로직 (채팅/이메일 탭과 동일한 패턴)
                # 전사 후 바이트 데이터 백업 (전사 전에 백업)
                audio_bytes_backup = st.session_state.bytes_to_process
                
                # 전사 후 바이트 데이터 즉시 삭제 (조건문 재평가 방지)
                st.session_state.bytes_to_process = None
                
                with st.spinner(L["whisper_processing"]):
                    try:
                        # 1) Whisper 전사 (자동 언어 감지 사용) - 채팅/이메일과 동일한 방식
                        agent_response_transcript = transcribe_bytes_with_whisper(
                            audio_bytes_backup,
                            "audio/wav",
                            lang_code=None,
                            auto_detect=True
                        )
                    except Exception as e:
                        agent_response_transcript = f"❌ 전사 오류: {e}"

                # 2) 전사 실패 처리 (채팅/이메일과 동일한 패턴)
                if not agent_response_transcript or agent_response_transcript.startswith("❌"):
                    error_msg = agent_response_transcript if agent_response_transcript else L["transcription_no_result"]
                    st.error(error_msg)
                    st.session_state.current_agent_audio_text = L["transcription_error"]
                    # ⭐ 최적화: 전사 실패 시에도 CC에 반영되지만 불필요한 rerun 제거 (Streamlit이 자동으로 재실행)
                elif not agent_response_transcript.strip(): # ⭐ 수정: 전사 결과가 비어 있거나 (공백만 있는 경우) 다음 단계로 진행하지 못하는 문제 해결
                    st.warning(L["transcription_empty_warning"])
                    st.session_state.current_agent_audio_text = ""
                    # ⭐ 최적화: 불필요한 rerun 제거
                elif agent_response_transcript.strip():
                    # 3) 전사 성공 - CC에 반영 (전사 결과를 먼저 CC 영역에 표시)
                    agent_response_transcript = agent_response_transcript.strip()
                    st.session_state.current_agent_audio_text = agent_response_transcript
                    
                    # 성공 메시지 표시 (채팅/이메일과 유사)
                    snippet = agent_response_transcript[:50].replace("\n", " ")
                    if len(agent_response_transcript) > 50:
                        snippet += "..."
                    st.success(L["whisper_success"] + f" **인식 내용:** *{snippet}*")

                    # ⭐ 수정: 첫 인사말인지 확인 (simulator_messages에 phone_exchange가 없으면 첫 인사말)
                    is_first_greeting = not any(
                        msg.get("role") == "phone_exchange" 
                        for msg in st.session_state.simulator_messages
                    )
                    
                    # ⭐ 수정: 전화 발신 모드 확인
                    is_outbound_call = st.session_state.get("call_sim_mode", "INBOUND") == "OUTBOUND"

                    if is_first_greeting:
                        # 첫 인사말인 경우: 로그에 기록하고 고객 문의 재생 준비
                        st.session_state.simulator_messages.append(
                            {"role": "agent", "content": agent_response_transcript}
                        )
                        # 아바타 표정 초기화
                        st.session_state.customer_avatar["state"] = "NEUTRAL"
                        
                        # ⭐ 수정: 전화 발신 모드에서 customer_initial_audio_bytes가 없으면 바로 고객 응답 생성
                        if is_outbound_call and not st.session_state.get("customer_initial_audio_bytes"):
                            # 전화 발신 모드이고 고객 문의 오디오가 없으면 바로 고객 응답 생성
                            st.session_state.current_agent_audio_text = agent_response_transcript
                            st.session_state.process_customer_reaction = True
                            st.session_state.pending_agent_transcript = agent_response_transcript
                            st.rerun()
                        else:
                            # ⭐ 수정: 고객 문의를 CC 자막에 미리 반영 (재생 전에 반영)
                            if st.session_state.call_initial_query:
                                st.session_state.current_customer_audio_text = st.session_state.call_initial_query
                            # ⭐ 수정: 고객 문의 재생을 바로 실행 (같은 실행 주기에서 처리)
                            # 고객 문의 재생 로직이 아래에 있으므로 플래그만 설정
                            st.session_state.customer_turn_start = True
                            # ⭐ 최적화: 플래그 설정 후 재실행하여 고객 문의 재생 로직 실행
                            st.rerun()
                    else:
                        # 이후 응답인 경우: 기존 로직대로 고객 반응 생성
                        # ⭐ 수정: 전화 발신 모드에서도 고객 반응이 생성되도록 보장
                        # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐
                        # 🎯 아바타 표정 업데이트 (LLM 기반 영상 RAG)
                        # LLM이 에이전트 응답을 분석하여 고객의 예상 반응(감정)을 판단
                        # 이는 고객이 다음에 말할 때 어떤 비디오를 보여줄지 결정하는 데 사용됩니다.
                        try:
                            # LLM 기반 분석 (에이전트 응답에 대한 고객의 예상 반응)
                            # 에이전트가 "환불"을 언급하면 고객은 기쁠 것이고,
                            # "기다려"를 요청하면 고객은 질문할 것이고,
                            # "불가"를 말하면 고객은 화날 것입니다.
                            # ⭐ Gemini 제안: 에이전트 답변과 대화 컨텍스트를 전달하여 예측 정확도 향상
                            analysis_result = analyze_text_for_video_selection(
                                agent_response_transcript,
                                st.session_state.language,
                                agent_last_response=agent_response_transcript,
                                conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                            )
                            # 고객의 예상 감정 상태 업데이트 (다음 고객 반응에 사용)
                            predicted_emotion = analysis_result.get("emotion", "NEUTRAL")
                            st.session_state.customer_avatar["state"] = predicted_emotion
                        except Exception as e:
                            # LLM 분석 실패 시 키워드 기반 폴백
                            print(f"LLM 분석 실패, 키워드 기반으로 폴백: {e}")
                            response_text = agent_response_transcript.lower()
                            if "refund" in response_text or "환불" in response_text:
                                st.session_state.customer_avatar["state"] = "HAPPY"
                            elif ("wait" in response_text or "기다려" in response_text or "잠시만" in response_text):
                                st.session_state.customer_avatar["state"] = "ASKING"
                            elif ("no" in response_text or "불가" in response_text or "안 됩니다" in response_text or "cannot" in response_text):
                                st.session_state.customer_avatar["state"] = "ANGRY"
                            else:
                                st.session_state.customer_avatar["state"] = "NEUTRAL"
                        # ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐

                        # ⭐ 수정: 전사 결과를 CC에 먼저 반영
                        st.session_state.current_agent_audio_text = agent_response_transcript

                        # ⭐ 수정: 전사 결과가 CC에 반영되도록 먼저 재실행
                        # 채팅과 동일하게 전사 결과를 먼저 화면에 표시한 후 고객 반응 생성
                        # 다음 실행 주기에서 고객 반응을 생성하도록 플래그 설정
                        st.session_state.process_customer_reaction = True
                        st.session_state.pending_agent_transcript = agent_response_transcript
                        # ⭐ 수정: 전사 완료 후 즉시 재실행하여 고객 반응 생성 단계로 진행
                        st.rerun()
                # ⭐ 수정: else 블록 제거 (이미 위에서 처리됨)

        # ⭐ 수정: 첫 인사말 후 고객 문의 재생 처리
        # customer_turn_start 플래그가 True일 때 고객 문의를 재생
        if st.session_state.get("customer_turn_start", False) and st.session_state.customer_initial_audio_bytes:
            # ⭐ 수정: 고객 문의 텍스트를 즉시 CC 영역에 반영 (재생 시작 전, 확실히 반영)
            st.session_state.current_customer_audio_text = st.session_state.call_initial_query
            
            # 고객 문의 재생 (비디오와 동기화) - LLM 기반 영상 RAG
            try:
                # 비디오 동기화가 활성화되어 있으면 비디오와 함께 재생
                if st.session_state.is_video_sync_enabled:
                    customer_gender = st.session_state.customer_avatar.get("gender", "male")
                    # ⭐ LLM 기반 텍스트 분석으로 감정/제스처 판단
                    # ⭐ Gemini 제안: 대화 컨텍스트 전달
                    agent_last_msg = None
                    if st.session_state.simulator_messages:
                        for msg in reversed(st.session_state.simulator_messages):
                            if msg.get("role") == "phone_exchange" and "Agent:" in msg.get("content", ""):
                                agent_last_msg = msg.get("content", "").split("Agent:")[-1].strip()
                                break
                    
                    analysis_result = analyze_text_for_video_selection(
                        st.session_state.call_initial_query,
                        st.session_state.language,
                        agent_last_response=agent_last_msg,
                        conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                    )
                    avatar_state = analysis_result.get("emotion", st.session_state.customer_avatar.get("state", "NEUTRAL"))
                    gesture = analysis_result.get("gesture", "NONE")
                    context_keywords = analysis_result.get("context_keywords", [])  # ⭐ Gemini 제안
                    
                    # 분석 결과를 아바타 상태에 반영
                    st.session_state.customer_avatar["state"] = avatar_state
                    
                    # ⭐ Gemini 제안: 상황별 키워드를 고려한 비디오 선택
                    video_path = get_video_path_by_avatar(
                        customer_gender, 
                        avatar_state, 
                        is_speaking=True,
                        gesture=gesture,
                        context_keywords=context_keywords
                    )
                    
                    if video_path and os.path.exists(video_path):
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        # 비디오와 오디오를 함께 재생
                        st.video(video_bytes, format="video/mp4", autoplay=True, loop=False, muted=False)
                        st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                    else:
                        # 비디오가 없으면 오디오만 재생
                        st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                else:
                    # 비디오 동기화가 비활성화되어 있으면 오디오만 재생
                    st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                
                st.success(L["customer_query_playing"])
                st.info(f"{L['query_content_label']} {st.session_state.call_initial_query}")
                
                # ⭐ 수정: 재생 완료 대기 로직 완전 제거
                # 브라우저에서 자동으로 재생되므로 서버에서 기다릴 필요 없음
                # 재생은 백그라운드에서 계속 진행되며, CC 자막은 이미 반영됨
                
            except Exception as e:
                st.warning(L["auto_play_failed"].format(error=str(e)))
                st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=False)
                st.info(f"{L['query_content_label']} {st.session_state.call_initial_query}")
            
            # 플래그 초기화
            st.session_state.customer_turn_start = False
            
            # ⭐ 수정: 맞춤형 반응 생성을 같은 실행 주기에서 처리하되, 재생은 계속 진행되도록 함
            # 에이전트의 첫 인사말 가져오기
            agent_greeting = ""
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") == "agent":
                    agent_greeting = msg.get("content", "")
                    break
            
            if agent_greeting:
                # 맞춤형 고객 반응 생성 (재생과 동시에 진행)
                with st.spinner(L["generating_customized_response"]):
                    customer_reaction = generate_customer_reaction_for_first_greeting(
                        st.session_state.language,
                        agent_greeting,
                        st.session_state.call_initial_query
                    )
                    
                    # 고객 반응을 TTS로 재생 및 CC에 반영 (비디오와 동기화) - LLM 기반 영상 RAG
                    if not customer_reaction.startswith("❌"):
                        audio_bytes, msg = synthesize_tts(customer_reaction, st.session_state.language, role="customer")
                        if audio_bytes:
                            try:
                                # 비디오 동기화가 활성화되어 있으면 비디오와 함께 재생
                                if st.session_state.is_video_sync_enabled:
                                    customer_gender = st.session_state.customer_avatar.get("gender", "male")
                                    # ⭐ LLM 기반 텍스트 분석으로 감정/제스처 판단
                                    # ⭐ Gemini 제안: 에이전트 답변과 대화 컨텍스트 전달
                                    agent_last_msg = st.session_state.current_agent_audio_text if hasattr(st.session_state, 'current_agent_audio_text') else None
                                    analysis_result = analyze_text_for_video_selection(
                                        customer_reaction,
                                        st.session_state.language,
                                        agent_last_response=agent_last_msg,
                                        conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                                    )
                                    avatar_state = analysis_result.get("emotion", st.session_state.customer_avatar.get("state", "NEUTRAL"))
                                    gesture = analysis_result.get("gesture", "NONE")
                                    context_keywords = analysis_result.get("context_keywords", [])  # ⭐ Gemini 제안
                                    
                                    # 분석 결과를 아바타 상태에 반영
                                    st.session_state.customer_avatar["state"] = avatar_state
                                    
                                    # ⭐ Gemini 제안: 상황별 키워드를 고려한 비디오 선택
                                    video_path = get_video_path_by_avatar(
                                        customer_gender, 
                                        avatar_state, 
                                        is_speaking=True,
                                        gesture=gesture,
                                        context_keywords=context_keywords
                                    )
                                    
                                    if video_path and os.path.exists(video_path):
                                        with open(video_path, "rb") as f:
                                            video_bytes = f.read()
                                        # 비디오와 오디오를 함께 재생
                                        st.video(video_bytes, format="video/mp4", autoplay=True, loop=False, muted=False)
                                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                        
                                        # ⭐ Gemini 제안: 사용자 피드백 평가 UI 추가
                                        st.markdown("---")
                                        st.markdown("**💬 비디오 매칭 평가**")
                                        st.caption("이 비디오가 고객의 텍스트와 감정에 자연스럽게 매칭되었습니까?")
                                        
                                        feedback_key = f"video_feedback_call_{st.session_state.sim_instance_id}_{len(st.session_state.simulator_messages)}"
                                        
                                        col_rating, col_comment = st.columns([2, 3])
                                        with col_rating:
                                            rating = st.slider(
                                                "평가 점수 (1-5점)",
                                                min_value=1,
                                                max_value=5,
                                                value=3,
                                                key=f"{feedback_key}_rating",
                                                help="1점: 매우 부자연스러움, 5점: 매우 자연스러움"
                                            )
                                        
                                        with col_comment:
                                            comment = st.text_input(
                                                "의견 (선택사항)",
                                                key=f"{feedback_key}_comment",
                                                placeholder="예: 비디오가 텍스트와 잘 맞았습니다"
                                            )
                                        
                                        if st.button("피드백 제출", key=f"{feedback_key}_submit"):
                                            # 피드백을 데이터베이스에 저장
                                            add_video_mapping_feedback(
                                                customer_text=customer_reaction,
                                                selected_video_path=video_path,
                                                emotion=avatar_state,
                                                gesture=gesture,
                                                context_keywords=context_keywords,
                                                user_rating=rating,
                                                user_comment=comment
                                            )
                                            st.success(f"✅ 피드백이 저장되었습니다! (점수: {rating}/5)")
                                            st.info("💡 이 피드백은 향후 비디오 선택 정확도를 개선하는 데 사용됩니다.")
                                    else:
                                        # 비디오가 없으면 오디오만 재생
                                        st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                else:
                                    # 비디오 동기화가 비활성화되어 있으면 오디오만 재생
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                
                                st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                            except Exception as e:
                                st.warning(L["auto_play_failed"].format(error=str(e)))
                                st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                                st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                        else:
                            st.error(L["customer_voice_generation_error"].format(error=msg))
                        
                        # ⭐ 수정: 고객 반응을 CC 영역에 추가 (고객 문의는 유지)
                        # 고객 문의와 반응을 모두 표시
                        if st.session_state.current_customer_audio_text == st.session_state.call_initial_query:
                            # 고객 문의만 있는 경우 반응 추가
                            st.session_state.current_customer_audio_text = f"{st.session_state.call_initial_query}\n\n→ {customer_reaction.strip()}"
                        else:
                            # 이미 반응이 있는 경우 업데이트
                            st.session_state.current_customer_audio_text = customer_reaction.strip()
                        
                        # 이력 저장
                        log_entry = f"Agent: {agent_greeting} | Customer: {customer_reaction.strip()}"
                        st.session_state.simulator_messages.append(
                            {"role": "phone_exchange", "content": log_entry})
                    else:
                        st.error(customer_reaction)
            
            # ⭐ 수정: rerun 완전 제거 - 재생은 브라우저에서 자동으로 진행되므로 서버에서 기다릴 필요 없음

        # ⭐ 수정: 전사 후 고객 반응 생성 처리 (마이크 위젯 렌더링 이후에 위치)
        # 전사 결과가 CC에 먼저 표시된 후 고객 반응을 생성하도록 분리
        if st.session_state.get("process_customer_reaction") and st.session_state.get("pending_agent_transcript"):
            pending_transcript = st.session_state.pending_agent_transcript
            # 플래그 초기화
            st.session_state.process_customer_reaction = False
            del st.session_state.pending_agent_transcript

            # ⭐ 수정: 에이전트 응답을 먼저 CC에 반영
            if hasattr(st.session_state, 'current_agent_audio_text'):
                st.session_state.current_agent_audio_text = pending_transcript
            else:
                st.session_state.current_agent_audio_text = pending_transcript

            # 고객 반응 생성
            with st.spinner(L["generating_customer_response"]):
                customer_reaction = generate_customer_reaction_for_call(
                    st.session_state.language,
                    pending_transcript
                )

                # 고객 반응을 TTS로 재생 및 CC에 반영 (비디오와 동기화) - LLM 기반 영상 RAG
                if not customer_reaction.startswith("❌"):
                    audio_bytes, msg = synthesize_tts(customer_reaction, st.session_state.language, role="customer")
                    if audio_bytes:
                        # Streamlit 문서: autoplay는 브라우저 정책상 제한될 수 있음
                        try:
                            # 비디오 동기화가 활성화되어 있으면 비디오와 함께 재생
                            if st.session_state.is_video_sync_enabled:
                                customer_gender = st.session_state.customer_avatar.get("gender", "male")
                                # ⭐ LLM 기반 텍스트 분석으로 감정/제스처 판단
                                # ⭐ Gemini 제안: 에이전트 답변과 대화 컨텍스트 전달
                                agent_last_msg = st.session_state.current_agent_audio_text if hasattr(st.session_state, 'current_agent_audio_text') else None
                                analysis_result = analyze_text_for_video_selection(
                                    customer_reaction,
                                    st.session_state.language,
                                    agent_last_response=agent_last_msg,
                                    conversation_context=st.session_state.simulator_messages[-5:] if st.session_state.simulator_messages else None
                                )
                                avatar_state = analysis_result.get("emotion", st.session_state.customer_avatar.get("state", "NEUTRAL"))
                                gesture = analysis_result.get("gesture", "NONE")
                                context_keywords = analysis_result.get("context_keywords", [])  # ⭐ Gemini 제안
                                
                                # 분석 결과를 아바타 상태에 반영
                                st.session_state.customer_avatar["state"] = avatar_state
                                
                                # ⭐ Gemini 제안: 상황별 키워드를 고려한 비디오 선택
                                video_path = get_video_path_by_avatar(
                                    customer_gender, 
                                    avatar_state, 
                                    is_speaking=True,
                                    gesture=gesture,
                                    context_keywords=context_keywords
                                )
                                
                                if video_path and os.path.exists(video_path):
                                    with open(video_path, "rb") as f:
                                        video_bytes = f.read()
                                    # 비디오와 오디오를 함께 재생
                                    st.video(video_bytes, format="video/mp4", autoplay=True, loop=False, muted=False)
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                                    
                                    # ⭐ Gemini 제안: 사용자 피드백 평가 UI 추가
                                    st.markdown("---")
                                    st.markdown("**💬 비디오 매칭 평가**")
                                    st.caption("이 비디오가 고객의 텍스트와 감정에 자연스럽게 매칭되었습니까?")
                                    
                                    feedback_key = f"video_feedback_{st.session_state.sim_instance_id}_{len(st.session_state.simulator_messages)}"
                                    
                                    col_rating, col_comment = st.columns([2, 3])
                                    with col_rating:
                                        rating = st.slider(
                                            "평가 점수 (1-5점)",
                                            min_value=1,
                                            max_value=5,
                                            value=3,
                                            key=f"{feedback_key}_rating",
                                            help="1점: 매우 부자연스러움, 5점: 매우 자연스러움"
                                        )
                                    
                                    with col_comment:
                                        comment = st.text_input(
                                            "의견 (선택사항)",
                                            key=f"{feedback_key}_comment",
                                            placeholder="예: 비디오가 텍스트와 잘 맞았습니다"
                                        )
                                    
                                    if st.button("피드백 제출", key=f"{feedback_key}_submit"):
                                        # 피드백을 데이터베이스에 저장
                                        add_video_mapping_feedback(
                                            customer_text=customer_reaction,
                                            selected_video_path=video_path,
                                            emotion=avatar_state,
                                            gesture=gesture,
                                            context_keywords=context_keywords,
                                            user_rating=rating,
                                            user_comment=comment
                                        )
                                        st.success(f"✅ 피드백이 저장되었습니다! (점수: {rating}/5)")
                                        st.info("💡 이 피드백은 향후 비디오 선택 정확도를 개선하는 데 사용됩니다.")
                                else:
                                    # 비디오가 없으면 오디오만 재생
                                    st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                            else:
                                # 비디오 동기화가 비활성화되어 있으면 오디오만 재생
                                st.audio(audio_bytes, format="audio/mp3", autoplay=True, loop=False)
                            
                            st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                            # ⭐ 수정: 고객 반응 재생 시간 확보를 위해 짧은 대기
                            time.sleep(0.5)
                        except Exception as e:
                            st.warning(L["auto_play_failed"].format(error=str(e)))
                            st.audio(audio_bytes, format="audio/mp3", autoplay=False)
                            st.success(L["customer_responded"].format(reaction=customer_reaction.strip()[:50] + "..."))
                    else:
                        st.error(L["customer_voice_generation_error"].format(error=msg))

                    # 고객 반응 텍스트를 CC 영역에 반영
                    st.session_state.current_customer_audio_text = customer_reaction.strip()
                    
                    # ⭐ 수정: 고객 반응을 이력에 저장 (전화 발신 모드에서도 작동)
                    agent_response_text = st.session_state.get("current_agent_audio_text", pending_transcript)
                    log_entry = f"Agent: {agent_response_text} | Customer: {customer_reaction.strip()}"
                    st.session_state.simulator_messages.append(
                        {"role": "phone_exchange", "content": log_entry}
                    )

                    # ⭐ 수정: "없습니다. 감사합니다" 응답 처리 - 에이전트가 감사 인사 후 종료
                    if L['customer_no_more_inquiries'] in customer_reaction:
                        # ⭐ 수정: 이력 저장은 이미 위에서 처리되었으므로 중복 저장 방지
                        
                        # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                        agent_name = st.session_state.get("agent_name", "000")
                        current_lang_call = st.session_state.get("language", "ko")
                        if current_lang_call == "ko":
                            agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                        elif current_lang_call == "en":
                            agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                        else:  # ja
                            agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"
                        
                        st.session_state.simulator_messages.append(
                            {"role": "phone_exchange", "content": f"Agent: {agent_closing_msg}"}
                        )
                        
                        # 통화 요약 생성
                        with st.spinner("AI 요약 생성 중..."):
                            summary = summarize_history_for_call(
                                st.session_state.simulator_messages,
                                st.session_state.call_initial_query,
                                st.session_state.language
                            )
                            st.session_state.call_summary_text = summary
                        
                        # 통화 종료
                        st.session_state.call_sim_stage = "CALL_ENDED"
                        st.session_state.is_call_ended = True
                        
                        # 에이전트 입력 영역 초기화
                        st.session_state.current_agent_audio_text = ""
                        st.session_state.realtime_hint_text = ""
                        if "bytes_to_process" in st.session_state:
                            st.session_state.bytes_to_process = None
                        
                        st.success("✅ 고객이 추가 문의 사항이 없다고 확인했습니다. 에이전트가 감사 인사를 전송한 후 통화가 종료되었습니다.")
                        st.rerun()
                    # ⭐ 추가: "추가 문의 사항도 있습니다" 응답 처리 (통화 계속)
                    elif L['customer_has_additional_inquiries'] in customer_reaction:
                        # ⭐ 수정: 이력 저장은 이미 위에서 처리되었으므로 중복 저장 방지
                        
                        # 에이전트 입력 영역 초기화 (다음 녹음을 위해)
                        st.session_state.current_agent_audio_text = ""
                        st.session_state.realtime_hint_text = ""
                        if "bytes_to_process" in st.session_state:
                            st.session_state.bytes_to_process = None
                        
                        st.info("💡 고객이 추가 문의 사항이 있다고 했습니다. 다음 응답을 녹음하세요.")
                    else:
                        # 일반 고객 반응 처리
                        # ⭐ 수정: 이력 저장은 이미 위에서 처리되었으므로 중복 저장 방지

                        # 에이전트 입력 영역 초기화 (다음 녹음을 위해)
                        st.session_state.current_agent_audio_text = ""
                        st.session_state.realtime_hint_text = ""
                        # ⭐ 최적화: bytes_to_process도 초기화하여 다음 녹음을 준비
                        if "bytes_to_process" in st.session_state:
                            st.session_state.bytes_to_process = None

                    # ⭐ 수정: rerun 제거 - 재생은 브라우저에서 자동으로 진행되므로 서버에서 기다릴 필요 없음
                    # 첫 문의와 동일하게 rerun을 제거하여 재생이 끝까지 진행되도록 함


    # ========================================
    # CALL_ENDED 상태
    # ========================================
    elif st.session_state.call_sim_stage == "CALL_ENDED":
        st.success(L["call_end_message"])

        # AHT
        if st.session_state.start_time is not None:
            final_aht_seconds = max(0, (datetime.now() - st.session_state.start_time).total_seconds())
            final_aht_str = str(timedelta(seconds=final_aht_seconds)).split('.')[0]
            st.metric("Final AHT", final_aht_str)

            hold_str = str(st.session_state.total_hold_duration).split('.')[0]
            st.metric("Total Hold Time", hold_str)
        else:
            st.warning(L["aht_not_recorded"])

        st.markdown("---")

        # ⭐ 추가: 현재 세션 이력 다운로드 기능 (채팅/이메일과 동일)
        st.markdown("**📥 현재 세션 이력 다운로드**")
        download_col1, download_col2, download_col3 = st.columns(3)
        
        # 현재 세션의 이력을 생성
        current_session_history = None
        if st.session_state.simulator_messages:
            try:
                customer_type_display = st.session_state.get("customer_type_sim_select", "")
                # 전화 요약이 있으면 사용, 없으면 생성
                if st.session_state.call_summary_text:
                    # call_summary_text를 summary 형식으로 변환
                    summary_data = {
                        "main_inquiry": st.session_state.call_initial_query,
                        "key_responses": [],
                        "customer_sentiment_score": 50,  # 기본값
                        "customer_satisfaction_score": 50,  # 기본값
                        "customer_characteristics": {},
                        "privacy_info": {},
                        "summary": st.session_state.call_summary_text
                    }
                else:
                    # 요약 생성
                    summary_data = generate_chat_summary(
                        st.session_state.simulator_messages,
                        st.session_state.call_initial_query,
                        customer_type_display,
                        st.session_state.language
                    )
                
                current_session_history = [{
                    "id": f"call_session_{st.session_state.sim_instance_id}",
                    "timestamp": datetime.now().isoformat(),
                    "initial_query": st.session_state.call_initial_query,
                    "customer_type": customer_type_display,
                    "language_key": st.session_state.language,
                    "messages": st.session_state.simulator_messages,
                    "summary": summary_data,
                    "is_chat_ended": True,
                    "attachment_context": st.session_state.get("sim_attachment_context_for_llm", ""),
                    "is_call": True
                }]
            except Exception as e:
                st.warning(f"이력 생성 중 오류 발생: {e}")
        
        # 다운로드 버튼들을 직접 표시
        if current_session_history:
            with download_col1:
                try:
                    filepath_word = export_history_to_word(current_session_history)
                    with open(filepath_word, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_word", "📥 이력 다운로드 (Word)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_word),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="download_call_word_file"
                        )
                except Exception as e:
                    st.error(f"Word 다운로드 오류: {e}")
            
            with download_col2:
                try:
                    filepath_pptx = export_history_to_pptx(current_session_history)
                    with open(filepath_pptx, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pptx", "📥 이력 다운로드 (PPTX)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pptx),
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key="download_call_pptx_file"
                        )
                except Exception as e:
                    st.error(f"PPTX 다운로드 오류: {e}")
            
            with download_col3:
                try:
                    filepath_pdf = export_history_to_pdf(current_session_history)
                    with open(filepath_pdf, "rb") as f:
                        st.download_button(
                            label=L.get("download_history_pdf", "📥 이력 다운로드 (PDF)"),
                            data=f.read(),
                            file_name=os.path.basename(filepath_pdf),
                            mime="application/pdf",
                            key="download_call_pdf_file"
                        )
                except Exception as e:
                    st.error(f"PDF 다운로드 오류: {e}")
        else:
            st.warning("다운로드할 이력이 없습니다.")

        st.markdown("---")

        with st.expander("통화 기록 요약"):
            st.subheader("AI 통화 요약")

            if st.session_state.call_summary_text:
                st.info(st.session_state.call_summary_text)
            else:
                st.error("❌ 통화 요약 생성 실패")

            st.markdown("---")

            st.subheader("고객 최초 문의 (음성)")
            if st.session_state.customer_initial_audio_bytes:
                # Streamlit 문서: bytes 데이터를 직접 전달 가능
                try:
                    st.audio(st.session_state.customer_initial_audio_bytes, format="audio/mp3", autoplay=False)
                except Exception as e:
                    st.error(f"오디오 재생 오류: {e}")
                st.caption(f"전사: {st.session_state.call_initial_query}")
            else:
                st.info("고객 최초 음성 없음")

            st.markdown("---")
            st.subheader("전체 교환 로그")
            for log in st.session_state.simulator_messages:
                st.write(log["content"])

        # 새 시뮬레이션
        if st.button(L["new_simulation_button"]):
            st.session_state.call_sim_stage = "WAITING_CALL"
            st.session_state.call_sim_mode = "INBOUND"
            st.session_state.is_on_hold = False
            st.session_state.total_hold_duration = timedelta(0)
            st.session_state.hold_start_time = None
            st.session_state.start_time = None
            st.session_state.current_customer_audio_text = ""
            st.session_state.current_agent_audio_text = ""
            st.session_state.agent_response_input_box_widget_call = ""
            st.session_state.call_initial_query = ""
            st.session_state.call_website_url = ""  # 홈페이지 주소 초기화
            st.session_state.simulator_messages = []
            st.session_state.call_summary_text = ""
            st.session_state.customer_initial_audio_bytes = None
            st.session_state.customer_history_summary = ""
            st.session_state.sim_audio_bytes = None
            st.rerun() # 새 시뮬레이션 시작 시 rerun


# -------------------- RAG Tab --------------------
elif feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])
    st.markdown("---")

    # ⭐ RAG 데이터 학습 기능 추가 - AI 고객 응대 시뮬레이터 데이터를 일일 파일로 학습
    st.subheader("📚 고객 가이드 자동 생성 (일일 학습)")
    
    if st.button("오늘 날짜 고객 가이드 생성", key="generate_daily_guide"):
        # 오늘 날짜로 파일명 생성 (예: 251130_고객가이드.TXT)
        today_str = datetime.now().strftime("%y%m%d")
        guide_filename = f"{today_str}_고객가이드.TXT"
        guide_filepath = os.path.join(DATA_DIR, guide_filename)
        
        # 최근 이력 로드
        all_histories = load_simulation_histories_local(st.session_state.language)
        recent_histories = all_histories[:50]  # 최근 50개 이력 사용
        
        if recent_histories:
            # LLM을 사용하여 고객 가이드 생성
            guide_prompt = f"""
당신은 CS 센터 교육 전문가입니다. 다음 고객 응대 이력 데이터를 분석하여 종합적인 고객 응대 가이드라인을 작성하세요.

분석할 이력 데이터:
{json.dumps([h.get('summary', {}) for h in recent_histories if h.get('summary')], ensure_ascii=False, indent=2)}

다음 내용을 포함하여 가이드라인을 작성하세요:
1. 고객 유형별 응대 전략 (일반/까다로운/매우 불만족)
2. 문화권별 응대 가이드 (언어, 문화적 배경 고려)
3. 주요 문의 유형별 해결 방법
4. 고객 감정 점수에 따른 응대 전략
5. 개인정보 처리 가이드
6. 효과적인 소통 스타일 권장사항

가이드라인을 한국어로 작성하세요.
"""
            
            if st.session_state.is_llm_ready:
                with st.spinner("고객 가이드 생성 중..."):
                    guide_content = run_llm(guide_prompt)
                    
                    # 파일 저장
                    with open(guide_filepath, "w", encoding="utf-8") as f:
                        f.write(f"고객 응대 가이드라인\n")
                        f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"분석 이력 수: {len(recent_histories)}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(guide_content)
                    
                    st.success(f"✅ 고객 가이드가 생성되었습니다: {guide_filename}")
                    st.info(f"파일 위치: {guide_filepath}")
                    
                    # 생성된 파일을 자동으로 RAG에 추가할지 선택
                    if st.button("생성된 가이드를 RAG에 추가", key="add_guide_to_rag"):
                        # 파일을 업로드된 파일처럼 처리하여 RAG에 추가
                        st.info("RAG 인덱스 업데이트 중...")
                        # 실제로는 파일을 읽어서 RAG 인덱스에 추가하는 로직 필요
            else:
                st.error("LLM이 준비되지 않았습니다. API Key를 설정해주세요.")
        else:
            st.warning("분석할 이력이 없습니다. 먼저 고객 응대 시뮬레이션을 실행하세요.")
    
    st.markdown("---")

    # --- 파일 업로드 섹션 ---
    # ⭐ 수정된 부분: RAG 탭 전용 키 사용
    uploaded_files = st.file_uploader(
        L["file_uploader"],
        type=["pdf", "txt", "html"],
        key="rag_file_uploader", # RAG 전용 키
        accept_multiple_files=True
    )

    if uploaded_files:
        if uploaded_files != st.session_state.uploaded_files_state:
            # 파일이 변경되면 RAG 상태 초기화
            st.session_state.is_rag_ready = False
            st.session_state.rag_vectorstore = None
            st.session_state.uploaded_files_state = uploaded_files

        if not st.session_state.is_rag_ready:
            if st.button(L["button_start_analysis"]):
                if not st.session_state.is_llm_ready:
                    st.error(L["simulation_no_key_warning"])
                    # st.stop()

                with st.spinner(L["data_analysis_progress"]):
                    vectorstore, count = build_rag_index(uploaded_files)

                if vectorstore:
                    st.session_state.rag_vectorstore = vectorstore
                    st.session_state.is_rag_ready = True
                    st.success(L["embed_success"].format(count=count))
                    st.session_state.rag_messages = [
                        {"role": "assistant", "content": f"✅ {len(uploaded_files)}개 파일 분석 완료. 질문해 주세요."}
                    ]
                else:
                    st.error(L["embed_fail"])
                    st.session_state.is_rag_ready = False
    else:
        st.info(L["warning_no_files"])
        st.session_state.is_rag_ready = False
        st.session_state.rag_vectorstore = None
        st.session_state.rag_messages = []

    st.markdown("---")

    # --- 챗봇 섹션 ---
    if st.session_state.is_rag_ready and st.session_state.rag_vectorstore:
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [{"role": "assistant", "content": "분석된 자료에 대해 질문해 주세요."}]

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(L["rag_input_placeholder"]):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(L["response_generating"]):
                    response = rag_answer(
                        prompt,
                        st.session_state.rag_vectorstore,
                        st.session_state.language
                    )
                    st.markdown(response)

            st.session_state.rag_messages.append({"role": "assistant", "content": response})
    else:
        st.warning(L["warning_rag_not_ready"])

# -------------------- Content Tab --------------------
elif feature_selection == L["content_tab"]:
    st.header(L["content_header"])
    st.markdown(L["content_desc"])
    st.markdown("---")

    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])
        st.info("💡 API Key를 설정하면 콘텐츠 생성 기능을 사용할 수 있습니다.")
        # st.stop() 제거: UI는 표시하되 기능만 비활성화

    # 다국어 맵핑 변수는 그대로 사용
    level_map = {
        "초급": "Beginner",
        "중급": "Intermediate",
        "고급": "Advanced",
        "Beginner": "Beginner",
        "Intermediate": "Intermediate",
        "Advanced": "Advanced",
        "初級": "Beginner",
        "中級": "Intermediate",
        "上級": "Advanced",
    }
    content_map = {
        "핵심 요약 노트": "summary",
        "객관식 퀴즈 10문항": "quiz",
        "실습 예제 아이디어": "example",
        "Key Summary Note": "summary",
        "10 MCQ Questions": "quiz",
        "Practical Example Idea": "example",
        "核心要約ノート": "summary",
        "選択式クイズ10問": "quiz",
        "実践例のアイデア": "example",
    }

    topic = st.text_input(L["topic_label"])
    level_display = st.selectbox(L["level_label"], L["level_options"])
    content_display = st.selectbox(L["content_type_label"], L["content_options"])

    level = level_map.get(level_display, "Beginner")
    content_type = content_map.get(content_display, "summary")

    if st.button(L["button_generate"]):
        if not topic.strip():
            st.warning(L["warning_topic"])
            # st.stop() 제거: 경고만 표시하고 계속 진행
        elif not st.session_state.is_llm_ready:
            st.error("❌ LLM이 준비되지 않았습니다. API Key를 설정해주세요.")
            # st.stop() 제거: 에러만 표시하고 계속 진행
        else:
            target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]

            # 공통 프롬프트 설정 (퀴즈 형식을 포함하지 않는 기본 템플릿)
            system_prompt = f"""
            You are a professional AI coach. Generate learning content in {target_lang} for the topic '{topic}' at the '{level}' difficulty.
            The content format requested is: {content_display}.
            Output ONLY the raw content.
            """

            if content_type == "quiz":
                # 퀴즈 전용 프롬프트 및 JSON 구조 강제 (로직 유지)
                lang_instruction = {"ko": "한국어로", "en": "in English", "ja": "日本語で"}.get(st.session_state.language, "in Korean")
                quiz_prompt = f"""
                You are an expert quiz generator. Based on the topic '{topic}' and difficulty '{level}', generate 10 multiple-choice questions.
                IMPORTANT: All questions, options, and explanations must be written {lang_instruction}.
                Your output MUST be a **raw JSON object** containing a single key "quiz_questions" which holds an array of 10 questions.
                Each object in the array must strictly follow the required keys: 
                - "question" (string): The question text in {lang_instruction}
                - "options" (array of 4 strings): Four answer choices in {lang_instruction}
                - "answer" (integer): The correct answer index starting from 1 (1-4)
                - "explanation" (string): A DETAILED and COMPREHENSIVE explanation (at least 2-3 sentences, preferably 50-100 words) explaining:
                  * Why the correct answer is right
                  * Why other options are incorrect (briefly mention key differences)
                  * Additional context or background information that helps understanding
                  * Real-world examples or applications if relevant
                  Write the explanation in {lang_instruction} with clear, educational content.
                DO NOT include any explanation, introductory text, or markdown code blocks (e.g., ```json).
                Output ONLY the raw JSON object, starting with '{{' and ending with '}}'.
                Example structure:
                {{
                  "quiz_questions": [
                    {{
                      "question": "질문 내용",
                      "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
                      "answer": 1,
                      "explanation": "정답인 이유를 상세히 설명하고, 다른 선택지가 왜 틀렸는지 간단히 언급하며, 관련 배경 지식이나 실제 사례를 포함한 충분히 긴 해설 내용 (최소 2-3문장, 50-100단어 정도)"
                    }}
                  ]
                }}
                """

            # JSON 추출 헬퍼 함수
            def extract_json_from_text(text):
                """텍스트에서 JSON 객체를 추출하는 함수"""
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

            generated_json_text = None
            raw_response_text = None
            llm_attempts = []

            # 1순위: OpenAI (JSON mode가 가장 안정적)
            if get_api_key("openai"):
                llm_attempts.append(("openai", get_api_key("openai"), "gpt-4o"))
            # 2순위: Gemini (Fallback)
            if get_api_key("gemini"):
                llm_attempts.append(("gemini", get_api_key("gemini"), "gemini-2.5-flash"))

            with st.spinner(L["response_generating"]):
                for provider, api_key, model_name in llm_attempts:
                    try:
                        if provider == "openai":
                            client = OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": quiz_prompt}],
                                # JSON Mode 강제
                                response_format={"type": "json_object"},
                            )
                            raw_response_text = response.choices[0].message.content.strip()
                            # OpenAI는 JSON 객체를 반환하므로, 직접 사용 시도
                            generated_json_text = extract_json_from_text(raw_response_text) or raw_response_text
                            break

                        elif provider == "gemini":
                            # Gemini는 response_format을 지원하지 않으므로, run_llm을 통해 일반 텍스트로 호출
                            raw_response_text = run_llm(quiz_prompt)
                            generated_json_text = extract_json_from_text(raw_response_text)
                            
                            # JSON 추출 성공 시 시도 종료
                            if generated_json_text:
                                break

                    except Exception as e:
                        print(f"JSON generation failed with {provider}: {e}")
                        continue

            # --- START: JSON Parsing and Error Handling Logic ---
            parsed_obj = None
            quiz_data = None
            
            if generated_json_text:
                try:
                    # JSON 객체 파싱 시도
                    parsed_obj = json.loads(generated_json_text)

                    # 'quiz_questions' 키에서 배열 추출
                    quiz_data = parsed_obj.get("quiz_questions")

                    if not isinstance(quiz_data, list) or len(quiz_data) < 1:
                        raise ValueError("Missing 'quiz_questions' key or empty array.")

                    # 데이터 유효성 검사: 각 문제에 필수 필드가 있는지 확인
                    for i, q in enumerate(quiz_data):
                        if not isinstance(q, dict):
                            raise ValueError(f"Question {i+1} is not a valid object.")
                        if "question" not in q or "options" not in q or "answer" not in q:
                            raise ValueError(f"Question {i+1} is missing required fields (question, options, or answer).")
                        if not isinstance(q["options"], list) or len(q["options"]) != 4:
                            raise ValueError(f"Question {i+1} must have exactly 4 options.")
                        if not isinstance(q["answer"], int) or q["answer"] < 1 or q["answer"] > 4:
                            raise ValueError(f"Question {i+1} answer must be an integer between 1 and 4.")
                        # explanation 필드가 없으면 기본값 추가
                        if "explanation" not in q or not q.get("explanation"):
                            q["explanation"] = f"정답은 {q['options'][q['answer']-1]}입니다. 이 문제에 대한 상세한 해설이 제공되지 않았습니다."

                    # 파싱 성공 및 데이터 유효성 검사 후 상태 저장
                    st.session_state.quiz_data = quiz_data
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = [1] * len(quiz_data)
                    st.session_state.show_explanation = False
                    st.session_state.is_quiz_active = True
                    st.session_state.quiz_type_key = str(uuid.uuid4())

                    st.success(f"**{topic}** - {content_display} 생성 완료")

                except json.JSONDecodeError as e:
                    # JSON 파싱 오류
                    st.error(L["quiz_error_llm"])
                    st.caption(f"JSON 파싱 오류: {str(e)}")
                    st.subheader(L["quiz_original_response"])
                    st.code(raw_response_text or generated_json_text, language="text")
                    if generated_json_text:
                        st.caption("추출된 JSON 텍스트:")
                        st.code(generated_json_text, language="text")
                    
                except ValueError as e:
                    # 데이터 구조 오류
                    st.error(L["quiz_error_llm"])
                    st.caption(f"데이터 구조 오류: {str(e)}")
                    st.subheader(L["quiz_original_response"])
                    st.code(raw_response_text or generated_json_text, language="text")
                    if parsed_obj:
                        st.caption("파싱된 객체:")
                        st.json(parsed_obj)
                        
            else:
                # JSON 추출 실패
                st.error(L["quiz_error_llm"])
                st.caption("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")
                if raw_response_text:
                    st.subheader(L["quiz_original_response"])
                    st.text_area("", raw_response_text, height=300)
                elif generated_json_text:
                    st.subheader(L["quiz_original_response"])
                    st.text_area("", generated_json_text, height=300)
                # --- END: JSON Parsing and Error Handling Logic ---

                else:  # 일반 텍스트 생성
                    st.session_state.is_quiz_active = False
                with st.spinner(L["response_generating"]):
                    content = run_llm(system_prompt)
                st.session_state.generated_content = content

                st.markdown("---")
                st.markdown(f"### {content_display}")
                st.markdown(st.session_state.generated_content)

    # --- 퀴즈/일반 콘텐츠 출력 로직 ---
    if st.session_state.get("is_quiz_active", False) and st.session_state.get("quiz_data"):
        # 퀴즈 진행 로직 (생략 - 기존 로직 유지)
        quiz_data = st.session_state.quiz_data
        idx = st.session_state.current_question_index

        # ⭐ 퀴즈 완료 시 IndexError 방지 로직 (idx >= len(quiz_data))
        if idx >= len(quiz_data):
            # 퀴즈 완료 시 최종 점수 표시
            st.success(L["quiz_complete"])
            total_questions = len(quiz_data)
            score = st.session_state.quiz_score
            incorrect_count = total_questions - score
            st.subheader(f"{L['score']}: {score} / {total_questions} ({(score / total_questions) * 100:.1f}%)")

            # 원형 차트로 맞은 문제/틀린 문제 표시
            if IS_PLOTLY_AVAILABLE:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # 원형 차트 생성
                    fig = go.Figure(data=[go.Pie(
                        labels=[L["correct_questions"], L["incorrect_questions"]],
                        values=[score, incorrect_count],
                        hole=0.4,
                        marker_colors=['#28a745', '#dc3545'],
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig.update_layout(
                        title=L["question_result"],
                        showlegend=True,
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### " + L["question_result"])
                    # 문제별 정오 리스트 표시
                    for i, question_item in enumerate(quiz_data):
                        user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                        is_correct = user_answer == 'Correctly Scored'
                        correct_answer_idx = question_item.get('answer', 1)
                        correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                        
                        # 사용자 답안 텍스트 가져오기
                        if is_correct:
                            user_answer_text = correct_answer_text
                            status_icon = "✅"
                            status_color = "green"
                        else:
                            if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                                user_answer_text = question_item['options'][user_answer - 1]
                            else:
                                user_answer_text = "미응답"
                            status_icon = "❌"
                            status_color = "red"
                        
                        # 문제별 결과 표시
                        with st.container():
                            st.markdown(f"""
                            <div style="border-left: 4px solid {status_color}; padding-left: 10px; margin-bottom: 15px;">
                                <strong>{status_icon} 문항 {i+1}:</strong> {question_item['question']}<br>
                                <span style="color: {status_color};">{L['your_answer']}: {user_answer_text}</span><br>
                                <span style="color: green;">{L['correct_answer_label']}: {correct_answer_text}</span>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                # Plotly가 없는 경우 텍스트로만 표시
                st.markdown(f"**{L['correct_questions']}:** {score}개")
                st.markdown(f"**{L['incorrect_questions']}:** {incorrect_count}개")
                st.markdown("### " + L["question_result"])
                for i, question_item in enumerate(quiz_data):
                    user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                    is_correct = user_answer == 'Correctly Scored'
                    correct_answer_idx = question_item.get('answer', 1)
                    correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                    
                    if is_correct:
                        user_answer_text = correct_answer_text
                        status_icon = "✅"
                    else:
                        if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                            user_answer_text = question_item['options'][user_answer - 1]
                        else:
                            user_answer_text = "미응답"
                        status_icon = "❌"
                    
                    st.markdown(f"**{status_icon} 문항 {i+1}:** {question_item['question']}")
                    st.markdown(f"- {L['your_answer']}: {user_answer_text}")
                    st.markdown(f"- {L['correct_answer_label']}: {correct_answer_text}")
                    st.markdown("---")

            if st.button(L["retake_quiz"], key="retake_quiz_btn"):
                # 퀴즈 상태만 초기화 (퀴즈 데이터는 유지하여 같은 퀴즈를 다시 풀 수 있도록)
                st.session_state.current_question_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = [1] * len(quiz_data)  # 기본값으로 초기화
                st.session_state.show_explanation = False
                st.rerun()  # 페이지 새로고침하여 첫 번째 문제로 이동
            # st.stop() 제거: 퀴즈 완료 후에도 UI는 계속 표시
        else:
            # 퀴즈 진행 (현재 문항)
            question_data = quiz_data[idx]
            st.subheader(f"{L.get('question_label', '문항')} {idx + 1}/{len(quiz_data)}")
            st.markdown(f"**{question_data['question']}**")

            # 기존 퀴즈 진행 및 채점 로직 (변화 없음)
            current_selection_index = st.session_state.quiz_answers[idx]

            options = question_data['options']
            current_answer = st.session_state.quiz_answers[idx]

            if current_answer is None or not isinstance(current_answer, int) or current_answer <= 0:
                radio_index = 0
            else:
                radio_index = min(current_answer - 1, len(options) - 1)

            selected_option = st.radio(
                L["select_answer"],
                options,
                index=radio_index,
                key=f"quiz_radio_{st.session_state.quiz_type_key}_{idx}"
            )

            selected_option_index = options.index(selected_option) + 1 if selected_option in options else None

            check_col, next_col = st.columns([1, 1])

            if check_col.button(L["check_answer"], key=f"check_answer_btn_{idx}"):
                if selected_option_index is None:
                    st.warning("선택지를 선택해 주세요.")
                else:
                    # 점수 계산 로직
                    if st.session_state.quiz_answers[idx] != 'Correctly Scored':
                        correct_answer = question_data.get('answer')  # answer 키가 없을 경우 대비
                        if selected_option_index == correct_answer:
                            st.session_state.quiz_score += 1
                            st.session_state.quiz_answers[idx] = 'Correctly Scored'
                            st.success(L["correct_answer"])
                        else:
                            st.session_state.quiz_answers[idx] = selected_option_index  # 오답은 선택지 인덱스 저장
                            st.error(L["incorrect_answer"])

                    st.session_state.show_explanation = True

            # 정답 및 해설 표시
            if st.session_state.show_explanation:
                correct_index = question_data.get('answer', 1)
                correct_answer_text = question_data['options'][correct_index - 1] if 0 < correct_index <= len(
                    question_data['options']) else "N/A"

                st.markdown("---")
                st.markdown(f"**{L['correct_is']}:** {correct_answer_text}")
                with st.expander(f"**{L['explanation']}**", expanded=True):
                    # 해설이 없거나 비어있을 경우 기본 해설 생성
                    explanation = question_data.get('explanation', '')
                    if not explanation or explanation.strip() == '' or explanation == '해설이 제공되지 않았습니다.':
                        # 기본 해설 생성
                        correct_idx = question_data.get('answer', 1)
                        correct_option = question_data['options'][correct_idx - 1] if 0 < correct_idx <= len(question_data['options']) else "N/A"
                        explanation = f"정답은 **{correct_option}**입니다.\n\n이 선택지가 정답인 이유를 설명하면, 문제에서 요구하는 핵심 개념과 가장 일치하는 답입니다. 다른 선택지들은 문제의 요구사항과 완전히 일치하지 않거나 관련이 적은 내용입니다."
                    st.info(explanation)

                # 다음 문항 버튼
                if next_col.button(L["next_question"], key=f"next_question_btn_{idx}"):
                    st.session_state.current_question_index += 1
                    st.session_state.show_explanation = False

            else:
                # 사용자가 이미 정답을 체크했고 (다시 로드된 경우), 다음 버튼을 바로 표시
                if st.session_state.quiz_answers[idx] == 'Correctly Scored' or (
                        isinstance(st.session_state.quiz_answers[idx], int) and st.session_state.quiz_answers[idx] > 0):
                    if next_col.button(L["next_question"], key=f"next_question_btn_after_check_{idx}"):
                        st.session_state.current_question_index += 1
                        st.session_state.show_explanation = False

    else:
        # 일반 콘텐츠 (핵심 요약 노트, 실습 예제 아이디어) 출력
        if st.session_state.get("generated_content"):
            content = st.session_state.generated_content  # Content를 다시 가져옴
            content_lines = content.split('\n')

            st.markdown("---")
            st.markdown(f"### {content_display}")

            # --- START: 효율성 개선 (상단 분석/하단 본문) ---

            st.subheader("💡 콘텐츠 분석 (Plotly 시각화)")

            if IS_PLOTLY_AVAILABLE:
                # 1. 키워드 빈도 시각화 (모의 데이터)

                # 콘텐츠를 텍스트 줄로 분할하여 모의 키워드 및 주요 문장 생성
                content = st.session_state.generated_content
                content_lines = content.split('\n')
                all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()

                # 모의 키워드 빈도 데이터 생성
                words = ['AI', '기술혁신', '고객경험', '데이터분석', '효율성', '여행산업']
                np.random.seed(42)
                counts = np.random.randint(5, 30, size=len(words))

                # 난이도에 따라 점수 가중치 (모의 감성 점수 변화)
                difficulty_score = {'Beginner': 60, 'Intermediate': 75, 'Advanced': 90}.get(level, 70)

                # --- 차트 1: 키워드 빈도 (Plotly Bar Chart) ---
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=words,
                        y=counts,
                        marker_color=px.colors.sequential.Plotly3,
                        name="키워드 빈도"
                    )
                ])
                fig_bar.update_layout(
                    title_text=f"주요 키워드 빈도 분석",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # --- 차트 2: 콘텐츠 감성 및 복잡도 추이 (Plotly Line Chart) ---
                # 모의 감성/복잡도 점수 추이 (5개 문단 모의)
                sections = ['도입부', '핵심1', '핵심2', '해결책', '결론']
                sentiment_scores = [difficulty_score - 10, difficulty_score + 5, difficulty_score,
                                    difficulty_score + 10, difficulty_score + 2]

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=sections,
                    y=sentiment_scores,
                    mode='lines+markers',
                    name='감성/복잡도 점수',
                    line=dict(color='orange', width=2),
                    marker=dict(size=8)
                ))
                fig_line.update_layout(
                    title_text="콘텐츠 섹션별 감성 및 복잡도 추이 (모의)",
                    yaxis_range=[50, 100],
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_line, use_container_width=True)

            else:  # Plotly가 없을 경우 기존 텍스트 분석 모의 유지
                st.info("Plotly 라이브러리가 없어 시각화를 표시할 수 없습니다. 텍스트 분석 모의를 표시합니다.")
                all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()
                unique_words = sorted(set(all_words), key=len, reverse=True)[:5] if all_words else ["N/A"]
                key_sentences = [
                    content_lines[0].strip() if content_lines else "N/A",
                    content_lines[len(content_lines) // 2].strip() if len(content_lines) > 1 else "",
                    content_lines[-1].strip() if len(content_lines) > 1 else ""
                ]
                key_sentences = [s for s in key_sentences if s and s != "N/A"]

                col_keyword, col_sentences = st.columns([1, 1])

                with col_keyword:
                    st.markdown("**핵심 키워드/개념 (모의)**")
                    st.info(f"[{', '.join(unique_words)}...]")

                with col_sentences:
                    st.markdown("**주요 문장 요약 (모의)**")
                    for sentence in key_sentences[:2]:
                        st.write(f"• {sentence[:50]}...")

            st.markdown("---")

            # 2. 하단 본문 출력
            st.markdown(f"### 📝 원본 콘텐츠")
            st.markdown(content)

            # --- END: 효율성 개선 ---

            # --- START: 아이콘 버튼 활성화 ---
            st.markdown("---")

            # 1. 복사할 내용 정리 및 이스케이프
            content_for_js = json.dumps(content)

            # JavaScript 코드는 이스케이프된 중괄호 {{}}를 사용
            js_copy_script = """
               function copyToClipboard(text) {{
                   navigator.clipboard.writeText(text).then(function() {{
                       // Streamlit toast 호출 (모의)
                       const elements = window.parent.document.querySelectorAll('[data-testid="stToast"]');
                       if (elements.length === 0) {{
                           // Fallback UI update (use Streamlit's native mechanism if possible, or simple alert)
                           console.log("복사 완료: " + text.substring(0, 50) + "...");
                           }}
                       }}, function(err) {{
                           // Fallback: Copy via execCommand (deprecated but often works in Streamlit's iframe)
                           const textarea = document.createElement('textarea');
                           textarea.value = text;
                           document.body.appendChild(textarea);
                           textarea.select();
                           document.execCommand('copy');
                           document.body.removeChild(textarea);
                           alert("복사 완료!"); 
                       }});
                   }}
                   // f-string 대신 .format을 사용하여 JavaScript 코드에 주입
                   // content_for_js는 이미 Python에서 JSON 문자열로 안전하게 이스케이프됨
                   copyToClipboard(JSON.parse('{content_json_safe}'));
               """.format(content_json_safe=content_for_js)

            # --- JavaScript for SHARE Menu (Messenger Mock) ---
            # Streamlit은 현재 소셜 미디어 API를 직접 호출할 수 없으므로, URL 복사를 사용하고 UI에 메시지 옵션을 모의합니다.
            js_share_url_copy = """
               function copyShareUrl() {{
                   const url = window.location.href;
                   navigator.clipboard.writeText(url).then(function() {{
                       console.log('App URL copied');
                   }}, function(err) {{
                       // Fallback
                       const textarea = document.createElement('textarea');
                       textarea.value = url;
                       document.body.appendChild(textarea);
                       textarea.select();
                       document.execCommand('copy');
                       document.body.removeChild(textarea);
                   }});
               }}
            """

            # --- JavaScript for SHARE Menu (Messenger Mock) ---
            # Streamlit은 현재 소셜 미디어 API를 직접 호출할 수 없으므로, URL 복사를 사용하고 UI에 메시지 옵션을 모의합니다.
            js_native_share = """
               function triggerNativeShare(title, text, url) {{
                   if (navigator.share) {{
                       // 1. 네이티브 공유 API 지원 시 사용
                       navigator.share({{
                           title: title,
                           text: text,
                           url: url,
                       }}).then(() => {{
                           console.log('Successful share');
                       }}).catch((error) => {{
                           console.log('Error sharing', error);
                       }});
                       return true;
                   }} else {{
                      // 2. 네이티브 공유 API 미지원 시 (PC 환경 등)
                      return false;
                   }}
               }}
            """


            # --- 더 보기 메뉴 (파일 다운로드/열기 모의) ---

            def mock_download(file_type: str, file_name: str):
                """모의 다운로드 기능: 파일명과 함께 성공 토스트 메시지를 출력합니다."""
                st.toast(f"📥 {file_type} 파일을 생성하여 다운로드를 시작합니다: {file_name}")
                # 실제 다운로드 로직은 Streamlit 컴포넌트 환경에서는 복잡하여 생략합니다.


            col_like, col_dislike, col_share, col_copy, col_more = st.columns([1, 1, 1, 1, 6])
            current_content_id = str(uuid.uuid4())  # 동적 ID 생성

            # 1. 좋아요 버튼 (기능 활성화)
            if col_like.button("👍", key=f"content_like_{current_content_id}"):
                st.toast(L["toast_like"])

            # 2. 싫어요 버튼 (기능 활성화)
            if col_dislike.button("👎", key=f"content_dislike_{current_content_id}"):
                st.toast(L["toast_dislike"])

            # 3. 공유 버튼 (Web Share API 호출 통합)
            with col_share:
                share_clicked = st.button("🔗", key=f"content_share_{current_content_id}")

            if share_clicked:
                # 1단계: 네이티브 공유 API 호출 시도 (모바일 환경 대상)
                share_title = f"{content_display} ({topic})"
                share_text = content[:150] + "..."
                share_url = "https://utility-convenience-salmonyeonwoo.streamlit.app/"  # 실제 배포 URL로 가정

                # JavaScript 실행: 네이티브 공유 호출
                st.components.v1.html(
                    f"""
                    <script>{js_native_share}
                        const shared = triggerNativeShare('{share_title}', '{share_text}', '{share_url}');
                        if (shared) {{
                           // 네이티브 공유 성공 시 (토스트 메시지는 브라우저가 관리)
                            console.log("Native Share Attempted.");
                        }} else {{
                           // 네이티브 공유 미지원 시, 대신 URL 복사
                           const url = window.location.href;
                           const textarea = document.createElement('textarea');
                           textarea.value = url;
                           document.body.appendChild(textarea);
                           textarea.select();
                           document.execCommand('copy');
                           document.body.removeChild(textarea);
                           // PC 환경에서 URL 복사 완료 토스트 메시지 출력
                           const toastElement = window.parent.document.querySelector('[data-testid="stToast"]');
                           if (toastElement) {{
                               // 이미 토스트 메시지가 열려 있다면 갱신 (Streamlit의 toast 기능을 가정)
                           }} else {{
                              alert('URL이 클립보드에 복사되었습니다.');
                           }}
                        }}
                    </script>
                    """,
                    height=0,
                )

                # Streamlit의 toast 메시지는 네이티브 공유 성공 여부를 알 수 없으므로 URL 복사 완료를 알림
                st.toast(L["toast_share"])


            # 4. 복사 버튼 (기능 활성화 - 콘텐츠 텍스트 복사)
            if col_copy.button("📋", key=f"content_copy_{current_content_id}"):
                # JavaScript를 실행하여 복사 (execCommand 사용으로 안정화)
                st.components.v1.html(
                    f"""<script>{js_copy_script}</script>""",
                    height=0,
                )
                st.toast(L["toast_copy"])

            # 5. 더보기 버튼 (기능 활성화 - 파일 옵션 모의)
            with col_more:
                more_clicked = st.button("•••", key=f"content_more_{current_content_id}")

            if more_clicked:
                st.toast(L["toast_more"])

                # 파일 옵션 모의 출력 (버튼 배치)
                st.markdown("**문서 옵션 (모의):**")
                col_doc1, col_doc2, col_doc3 = st.columns(3)

                # 다국어 레이블 적용
                if col_doc1.button(L["mock_pdf_save"], key=f"mock_pdf_save_{current_content_id}"):  # 동적 ID 적용
                    mock_download("PDF", f"{topic}_summary.pdf")
                if col_doc2.button(L["mock_word_open"], key=f"mock_word_open_{current_content_id}"):  # 동적 ID 적용
                    mock_download("Word", f"{topic}_summary.docx")
                if col_doc3.button(L["mock_print"], key=f"mock_print_{current_content_id}"):  # 동적 ID 적용
                    st.toast("🖨 브라우저 인쇄 창이 열립니다.")

            # --- END: 효율성 개선 ---

            # --- END: 아이콘 버튼 추가 ---

# -------------------- LSTM Tab --------------------
elif feature_selection == L["lstm_tab"]:
    # ... (기존 LSTM 탭 로직 유지)
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])

    # ⭐ 최적화: 버튼 자체가 rerun을 유도하므로 명시적 rerun 제거 (버튼 클릭 시 자동 재실행)
    if st.button(L["lstm_rerun_button"]):
        # 버튼 클릭 시 Streamlit이 자동으로 재실행
        pass

    try:
        data = load_or_train_lstm()
        predicted_score = float(np.clip(data[-1] + np.random.uniform(-3, 5), 50, 100))

        st.markdown("---")
        st.subheader(L["lstm_result_header"])

        col_score, col_chart = st.columns([1, 2])

        with col_score:
            suffix = "점" if st.session_state.language == "ko" else ""
            st.metric(L["lstm_score_metric"], f"{predicted_score:.1f}{suffix}")
            st.info(L["lstm_score_info"].format(predicted_score=predicted_score))
        with col_chart:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(data, label="Past Scores", marker="o")
            ax.plot(len(data), predicted_score, marker="*", markersize=10)
            ax.set_title(L["lstm_header"])
            ax.set_xlabel("Time (attempts)")
            ax.set_ylabel("Score (0-100)")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.info(f"LSTM 기능 에러: {e}")
