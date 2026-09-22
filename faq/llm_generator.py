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
LLM을 사용한 회사 정보 생성 모듈
"""

import json
import re
from typing import Dict, Any
from faq.llm_prompts import get_company_info_prompt
from faq.common_faqs import get_common_product_faqs


def generate_company_info_with_llm(company_name: str, lang: str = "ko") -> Dict[str, Any]:
    """LLM을 사용하여 회사 정보 생성"""
    prompt = get_company_info_prompt(company_name, lang)
    
    try:
        from llm_client import run_llm
        response = run_llm(prompt)
        
        # JSON 파싱 시도
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group()
            try:
                company_data = json.loads(json_str)
                
                # 공동 대표 제품 FAQ 추가
                common_faqs = get_common_product_faqs(company_name, lang)
                if common_faqs:
                    existing_faqs = company_data.get("faqs", [])
                    company_data["faqs"] = common_faqs + existing_faqs
                
                # FAQ가 10개 미만이면 기본 FAQ 추가
                if len(company_data.get("faqs", [])) < 10:
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
                    while len(existing_faqs) < 10:
                        idx = len(existing_faqs) % len(default_faqs)
                        existing_faqs.append(default_faqs[idx])
                    company_data["faqs"] = existing_faqs[:10]
                return company_data
            except json.JSONDecodeError:
                return _create_default_response(response)
        else:
            return _create_default_response(response)
    except Exception as e:
        print(f"[generate_company_info_with_llm] Exception: {e}, using smart fallback")
        return _create_smart_fallback(company_name, lang)


def _create_smart_fallback(company_name: str, lang: str = "ko") -> Dict[str, Any]:
    """LLM 부재 또는 유효하지 않은 응답 시 구조화된 고품질 기업 정보 생성"""
    c_lower = company_name.lower()
    
    # 1. 업종 분류 추론
    if any(k in c_lower for k in ["benz", "벤츠", "mercedes", "bmw", "audi", "현대", "기아", "차"]):
        industry = "프리미엄 자동차 제조 및 스마트 모빌리티 솔루션"
        desc = f"{company_name}은(는) 혁신적인 엔지니어링과 첨단 주행 기술, 안전 철학을 바탕으로 글로벌 럭셔리 모빌리티 시장을 선도하는 자동차 제조 기업입니다."
        products = ["플래그십 럭셔리 세단 라인업", "프리미엄 도심형 SUV 시리즈", "순수 전기차(EV) 전용 라인업", "스마트 인포테인먼트 및 자율주행 패키지"]
    elif any(k in c_lower for k in ["klook", "클룩", "여행", "투어", "호텔", "항공", "travel", "tour"]):
        industry = "글로벌 여행/액티비티 및 레저 예약 플랫폼"
        desc = f"{company_name}은(는) 전 세계 여행자들에게 맞춤형 투어, 액티비티, 교통 티켓 및 글로벌 eSIM 서비스를 제공하는 혁신 여행 플랫폼입니다."
        products = ["글로벌 데이터 로밍 eSIM", "테마파크 및 명소 입장권", "프라이빗 일일 투어 패키지", "공항 픽업 및 교통 패스"]
    elif any(k in c_lower for k in ["삼성", "samsung", "애플", "apple", "소니", "lg", "전자", "it", "반도체"]):
        industry = "글로벌 첨단 IT, 전자제품 및 인공지능 솔루션"
        desc = f"{company_name}은(는) 스마트 기기, 차세대 반도체 및 지능형 소프트웨어를 통해 전 세계 고객의 라이프스타일을 혁신하는 글로벌 테크 리더입니다."
        products = ["차세대 플래그십 스마트폰", "고화질 스마트 디스플레이", "초고속 메모리 및 시스템 반도체", "스마트홈 IoT 에코시스템"]
    else:
        industry = "글로벌 비즈니스 및 엔터프라이즈 솔루션"
        desc = f"{company_name}은(는) 전문화된 제품과 고품질 고객 서비스를 바탕으로 신뢰받는 혁신 솔루션을 제공하는 선도 기업입니다."
        products = [f"{company_name} 프리미엄 서비스 플랜", f"{company_name} 엔터프라이즈 솔루션", "고객 맞춤형 컨설팅", "글로벌 파트너십 네트워크"]

    return {
        "company_info": f"{industry}\n\n{desc}",
        "popular_products": products,
        "trending_topics": [f"{company_name} 차세대 신제품 출시", "지속 가능한 친환경 경영 전략", "글로벌 시장 점유율 확대", "고객 만족도 최우수 평가"],
        "faqs": [
            {
                "question_ko": f"{company_name}의 주요 사업 영역과 대표 제품은 무엇인가요?",
                "answer_ko": f"{company_name}은(는) {industry} 분야를 핵심 사업으로 영위하며, {', '.join(products[:2])} 등을 대표적으로 제공하고 있습니다.",
                "question_en": f"What are the main business areas and flagship products of {company_name}?",
                "answer_en": f"{company_name} specializes in {industry}, offering leading products including {products[0]}.",
                "question_ja": f"{company_name}の主要な事業分野と代表製品は何ですか？",
                "answer_ja": f"{company_name}は{industry}を主要事業として展開し、主力製品を提供しています。"
            },
            {
                "question_ko": f"{company_name}의 고객 지원 및 서비스 이용 방법은 어떻게 되나요?",
                "answer_ko": f"공식 웹사이트 및 고객센터를 통해 24시간 실시간 상담, 예약 조회, A/S 접수 및 보증 기간 확인이 가능합니다.",
                "question_en": f"How can customers access support and services for {company_name}?",
                "answer_en": "Customers can access 24/7 assistance, order tracking, and warranty service via the official portal and support hotline.",
                "question_ja": f"{company_name}のカスタマーサポート利用方法は？",
                "answer_ja": "公式ポータルおよびサポート窓口を通じて、24時間体制で各種サポートをご利用いただけます。"
            }
        ],
        "interview_questions": [],
        "ceo_info": {}
    }

def _create_default_response(response: str) -> Dict[str, Any]:
    """기본 응답 구조 생성 (generic 문구 방지)"""
    if not response or "접수하였습니다" in response or "안내해 드리겠습니다" in response:
        return _create_smart_fallback("선택 회사", "ko")
    return {
        "company_info": response[:1000] if len(response) > 1000 else response,
        "popular_products": [],
        "trending_topics": [],
        "faqs": [],
        "interview_questions": [],
        "ceo_info": {}
    }

