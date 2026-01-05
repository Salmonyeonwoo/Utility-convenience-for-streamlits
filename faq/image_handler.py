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
제품 이미지 처리 모듈
"""

import os
import hashlib
import requests
from typing import Dict
from openai import OpenAI

from config import PRODUCT_IMAGE_CACHE_FILE, PRODUCT_IMAGE_DIR
from utils import _load_json, _save_json


def load_product_image_cache() -> Dict[str, str]:
    """제품 이미지 캐시 로드"""
    return _load_json(PRODUCT_IMAGE_CACHE_FILE, {})


def save_product_image_cache(cache_data: Dict[str, str]):
    """제품 이미지 캐시 저장"""
    _save_json(PRODUCT_IMAGE_CACHE_FILE, cache_data)


def generate_product_image_prompt(product_name: str) -> str:
    """제품명을 기반으로 이미지 생성 프롬프트 생성"""
    product_lower = product_name.lower()
    
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
        
        # 디즈니랜드 관련 상품
        if ("디즈니" in product_name or "disney" in product_lower or "disneyland" in product_lower or 
            "tokyo disneyland" in product_lower or "hong kong disneyland" in product_lower or
            "ディズニー" in product_name or "ディズニーランド" in product_name):
            return "https://images.unsplash.com/photo-1606813907291-d86efa9b94db?w=400&h=300&fit=crop&q=80"
        
        # 유니버셜 스튜디오 관련 상품
        if ("유니버셜" in product_name or "universal" in product_lower or "universal studio" in product_lower or
            "universal studios" in product_lower or "ユニバーサル" in product_name or "ユニバーサルスタジオ" in product_name):
            return "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?w=400&h=300&fit=crop&q=80"
        
        # 도쿄 스카이트리 관련 상품
        if ("스카이트리" in product_name or "skytree" in product_lower or "도쿄 타워" in product_name or 
            "tokyo tower" in product_lower or "tokyo skytree" in product_lower or
            "スカイツリー" in product_name or "東京タワー" in product_name or "東京スカイツリー" in product_name):
            return "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400&h=300&fit=crop&q=80"
        
        # 홍콩 관련 상품
        if ("홍콩" in product_name or "hong kong" in product_lower or "香港" in product_name):
            if "disney" not in product_lower and "디즈니" not in product_name:
                return "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?w=400&h=300&fit=crop&q=80"
        
        # 방콕 관련 상품
        if ("방콕" in product_name or "bangkok" in product_lower or "バンコク" in product_name):
            return "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?w=400&h=300&fit=crop&q=80"
        
        # 삼성 갤럭시 S 시리즈 관련 상품 (최신 모델 포함)
        if ("갤럭시 s" in product_lower or "galaxy s" in product_lower or 
            "galaxy s25" in product_lower or "galaxy s24" in product_lower or
            "galaxy s23" in product_lower or "galaxy s22" in product_lower or "galaxy s21" in product_lower or
            "galaxy s20" in product_lower or "samsung galaxy s" in product_lower or
            "ギャラクシー s" in product_name.lower() or "ガラクシー s" in product_name.lower()):
            return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
        
        # 삼성 갤럭시 노트 시리즈 관련 상품
        if ("갤럭시 노트" in product_lower or "galaxy note" in product_lower or 
            "galaxy note24" in product_lower or "galaxy note23" in product_lower or 
            "galaxy note22" in product_lower or "galaxy note21" in product_lower or
            "galaxy note20" in product_lower or "samsung galaxy note" in product_lower or
            "ギャラクシー ノート" in product_name.lower() or "ガラクシー ノート" in product_name.lower()):
            return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
        
        # 삼성 QLED TV 관련 상품
        if ("qled" in product_lower or "삼성 qled" in product_lower or "samsung qled" in product_lower or
            "삼성 tv" in product_lower or "samsung tv" in product_lower):
            return "https://images.unsplash.com/photo-1593359677879-a4b92c0a3b8b?w=400&h=300&fit=crop&q=80"
        
        # 삼성 제품 일반
        if ("삼성" in product_name or "samsung" in product_lower):
            if ("스마트폰" in product_name or "smartphone" in product_lower or "phone" in product_lower or
                "갤럭시" in product_name or "galaxy" in product_lower):
                return "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=300&fit=crop&q=80"
            elif ("tv" in product_lower or "티비" in product_name or "텔레비전" in product_name):
                return "https://images.unsplash.com/photo-1593359677879-a4b92c0a3b8b?w=400&h=300&fit=crop&q=80"
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
        
        # 기본값: 해시 기반 이미지 선택
        hash_obj = hashlib.md5(product_name.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        image_seed = hash_int % 1000
        
        category_images = [
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&q=80",
            "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&h=300&fit=crop&q=80",
            "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop&q=80",
            "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=300&fit=crop&q=80",
            "https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=400&h=300&fit=crop&q=80",
        ]
        
        return category_images[image_seed % len(category_images)]
    except Exception:
        return ""

