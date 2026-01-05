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
탭1: 인기 상품 표시 모듈
"""

import streamlit as st
import os
import requests
from faq.image_handler import get_product_image_url


def render_products_section(popular_products: list, current_lang: str, L: dict):
    """인기 상품 목록 렌더링 (이미지 포함)"""
    st.markdown(f"#### {L['popular_products']}")
    product_cols = st.columns(min(3, len(popular_products)))
    
    for idx, product in enumerate(popular_products):
        product_text = product.get(f"text_{current_lang}", product.get("text_ko", ""))
        product_score = product.get("score", 0)
        product_image_url = product.get("image_url", "")

        with product_cols[idx % len(product_cols)]:
            # 이미지 URL 찾기
            if not product_image_url:
                image_found = False
                for lang_key in [current_lang, "ko", "en", "ja"]:
                    check_text = product.get(f"text_{lang_key}", "")
                    if check_text:
                        check_url = get_product_image_url(check_text)
                        if check_url:
                            product_image_url = check_url
                            image_found = True
                            break
                
                if not image_found:
                    product_image_url = get_product_image_url(product_text)

            # 이미지 표시
            image_displayed = _display_product_image(product_image_url, product_text)
            
            # 이미지 URL이 없으면 생성 시도
            if not product_image_url:
                product_image_url = get_product_image_url(product_text)
                if product_image_url:
                    image_displayed = _display_product_image(product_image_url, product_text)

            # 이미지 표시 실패 시 이모지 카드 표시
            if not image_displayed:
                product_emoji = _get_product_emoji(product_text)
                product_html = f"""<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; color: white; min-height: 200px; display: flex; flex-direction: column; justify-content: center;'>
                        <h1 style='font-size: 64px; margin: 0;'>{product_emoji}</h1>
                        <p style='font-size: 16px; margin-top: 15px; font-weight: bold;'>{product_text[:25]}</p>
                    </div>"""
                st.markdown(product_html, unsafe_allow_html=True)

            st.write(f"**{product_text}**")
            st.caption(f"{L.get('popularity', '인기도')}: {product_score}")
            st.markdown("---")


def _display_product_image(product_image_url: str, product_text: str) -> bool:
    """제품 이미지 표시 시도"""
    if not product_image_url:
        return False
    
    try:
        # 로컬 파일 경로인 경우
        if os.path.exists(product_image_url):
            st.image(product_image_url, caption=product_text[:30], use_container_width=True)
            return True
        
        # URL인 경우
        elif product_image_url.startswith(("http://", "https://")):
            try:
                response = requests.get(
                    product_image_url, timeout=5, allow_redirects=True, stream=True)
                if response.status_code == 200:
                    try:
                        from PIL import Image
                        import io
                        img_data = response.content
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, caption=product_text[:30], use_container_width=True)
                        return True
                    except ImportError:
                        st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                        return True
                else:
                    try:
                        st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                        return True
                    except Exception:
                        return False
            except Exception:
                try:
                    st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                    return True
                except Exception:
                    return False
        else:
            try:
                st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                return True
            except Exception:
                return False
    except Exception:
        try:
            if product_image_url.startswith(("http://", "https://")):
                st.image(product_image_url, caption=product_text[:30], use_container_width=True)
                return True
        except Exception:
            pass
    
    return False


def _get_product_emoji(product_text: str) -> str:
    """제품 텍스트에 따른 이모지 반환"""
    text_lower = product_text.lower()
    if "티켓" in product_text or "ticket" in text_lower:
        return "🎫"
    elif "테마파크" in product_text or "theme" in text_lower or "디즈니" in product_text or "유니버셜" in product_text or "스튜디오" in product_text:
        return "🎢"
    elif "항공" in product_text or "flight" in text_lower:
        return "✈️"
    elif "호텔" in product_text or "hotel" in text_lower:
        return "🏨"
    elif "음식" in product_text or "food" in text_lower:
        return "🍔"
    elif "여행" in product_text or "travel" in text_lower or "사파리" in product_text:
        return "🌏"
    else:
        return "📦"

