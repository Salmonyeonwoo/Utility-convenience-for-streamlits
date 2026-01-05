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
공통 제품 FAQ 모듈
"""

from typing import List, Dict


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
                }
            ]
        elif lang == "en":
            common_faqs = [
                {
                    "question_en": "Which countries can I use eSIM in?",
                    "answer_en": "eSIM can be used in most countries around the world. It is available in over 190 countries and regions including major travel destinations in Europe, Asia, Americas, and Oceania.",
                    "question_ko": "eSIM은 어떤 국가에서 사용할 수 있나요?",
                    "answer_ko": "eSIM은 전 세계 대부분의 국가에서 사용 가능합니다.",
                    "question_ja": "eSIMはどの国で使用できますか？",
                    "answer_ja": "eSIMは世界中のほとんどの国で使用できます。"
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
                }
            ]
    
    # 삼성 공동 제품 FAQ
    elif "samsung" in company_lower or "삼성" in company_name:
        if lang == "ko":
            common_faqs = [
                {
                    "question_ko": "Galaxy S25 Ultra의 주요 특징은 무엇인가요?",
                    "answer_ko": "Galaxy S25 Ultra는 삼성의 최신 플래그십 스마트폰으로, 고성능 프로세서(Snapdragon 8 Gen 4 또는 Exynos), 고해상도 카메라 시스템(200MP 메인 카메라, 10배 광학 줌), 긴 배터리 수명(5000mAh), 초고속 충전 기능(45W)을 제공합니다. 특히 AI 기능이 강화되어 사진 촬영, 생산성 향상, 일상 작업 자동화에 도움이 됩니다. 방수 방진(IP68) 기능도 탑재되어 있습니다.",
                    "question_en": "What are the main features of Galaxy S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra is Samsung's latest flagship smartphone, offering high-performance processor (Snapdragon 8 Gen 4 or Exynos), high-resolution camera system (200MP main camera, 10x optical zoom), long battery life (5000mAh), and ultra-fast charging (45W). AI features are particularly enhanced to help with photography, productivity, and daily task automation. It also features IP68 water and dust resistance.",
                    "question_ja": "Galaxy S25 Ultraの主な特徴は何ですか？",
                    "answer_ja": "Galaxy S25 Ultraはサムスンの最新フラグシップスマートフォンで、高性能プロセッサー（Snapdragon 8 Gen 4またはExynos）、高解像度カメラシステム（200MPメインカメラ、10倍光学ズーム）、長いバッテリー寿命（5000mAh）、超高速充電機能（45W）を提供します。特にAI機能が強化され、写真撮影、生産性向上、日常作業の自動化に役立ちます。防水防塵（IP68）機能も搭載されています。"
                },
                {
                    "question_ko": "Galaxy S24 Ultra와 S25 Ultra의 차이점은 무엇인가요?",
                    "answer_ko": "Galaxy S25 Ultra는 S24 Ultra 대비 AI 성능이 크게 향상되었고, 프로세서 성능이 더욱 강화되었습니다. 카메라 시스템도 개선되어 더 나은 저조도 촬영 성능과 비디오 녹화 기능을 제공합니다. 배터리 효율성과 충전 속도도 개선되었습니다.",
                    "question_en": "What is the difference between Galaxy S24 Ultra and S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra has significantly improved AI performance and enhanced processor performance compared to S24 Ultra. The camera system has also been improved, offering better low-light photography and video recording capabilities. Battery efficiency and charging speed have also been improved.",
                    "question_ja": "Galaxy S24 UltraとS25 Ultraの違いは何ですか？",
                    "answer_ja": "Galaxy S25 Ultraは、S24 Ultraと比較してAI性能が大幅に向上し、プロセッサー性能もさらに強化されました。カメラシステムも改善され、より優れた低照度撮影性能とビデオ録画機能を提供します。バッテリー効率と充電速度も改善されました。"
                }
            ]
        elif lang == "en":
            common_faqs = [
                {
                    "question_en": "What are the main features of Galaxy S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra is Samsung's latest flagship smartphone, offering high-performance processor (Snapdragon 8 Gen 4 or Exynos), high-resolution camera system (200MP main camera, 10x optical zoom), long battery life (5000mAh), and ultra-fast charging (45W). AI features are particularly enhanced to help with photography, productivity, and daily task automation. It also features IP68 water and dust resistance.",
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
                    "answer_ja": "Galaxy S25 Ultraはサムスンの最新フラグシップスマートフォンで、高性能プロセッサー（Snapdragon 8 Gen 4またはExynos）、高解像度カメラシステム（200MPメインカメラ、10倍光学ズーム）、長いバッテリー寿命（5000mAh）、超高速充電機能（45W）を提供します。特にAI機能が強化され、写真撮影、生産性向上、日常作業の自動化に役立ちます。防水防塵（IP68）機能も搭載されています。",
                    "question_ko": "Galaxy S25 Ultra의 주요 특징은 무엇인가요?",
                    "answer_ko": "Galaxy S25 Ultra는 삼성의 최신 플래그십 스마트폰입니다.",
                    "question_en": "What are the main features of Galaxy S25 Ultra?",
                    "answer_en": "Galaxy S25 Ultra is Samsung's latest flagship smartphone."
                }
            ]
    
    return common_faqs

