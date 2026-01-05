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
FAQ 관리 모듈 (메인)
회사 정보, FAQ 검색, 제품 이미지 생성 등의 기능을 제공합니다.
"""

# 데이터베이스 관련
from faq.database import (
    load_faq_database,
    save_faq_database,
    get_company_info_faq
)

# 시각화 관련
from faq.visualization import visualize_company_data

# 이미지 처리 관련
from faq.image_handler import (
    load_product_image_cache,
    save_product_image_cache,
    generate_product_image_prompt,
    generate_product_image_with_ai,
    get_product_image_url
)

# 검색 관련
from faq.search import search_faq

# 공통 FAQ 관련
from faq.common_faqs import get_common_product_faqs

# LLM 생성 관련
from faq.llm_generator import generate_company_info_with_llm

# 하위 호환성을 위한 별칭
DEFAULT_LANG = "ko"
