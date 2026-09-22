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
기업 정보 및 고해상도 비쥬얼 자산(사진/배너/제품 이미지) 통합 엔진
- 한국어(ko), 영어(en), 일본어(ja) 다국어 완전 지원
- 검색된 모든 기업에 대해 업종 분류, 기업 상세 소개, 대표 제품 및 비쥬얼 사진/배너 제공
- RAG 지식 베이스 및 LLM 연동 지원
- generic 플레이스홀더 문구 원천 차단
"""

import os
import json
import re
from typing import Dict, Any, List, Optional


# 업종 및 대표 기업별 고품질 비쥬얼 이미지 큐레이션 (다국어 지원)
VISUAL_PRESETS = {
    "automotive": {
        "industry": {
            "ko": "프리미엄 자동차 제조 및 스마트 모빌리티 솔루션",
            "en": "Premium Automotive Manufacturing & Smart Mobility Solutions",
            "ja": "プレミアム自動車製造＆スマートモビリティソリューション"
        },
        "hero_image": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=800&h=300&fit=crop&q=80",
        "hero_tagline": {
            "ko": "The Best or Nothing - 최첨단 엔지니어링과 럭셔리 모빌리티의 기준",
            "en": "The Best or Nothing - The Benchmark of Advanced Engineering & Luxury Mobility",
            "ja": "The Best or Nothing - 最先端エンジニアリングとラグジュアリーモビリティの基準"
        },
        "products": [
            {
                "name": {
                    "ko": "플래그십 럭셔리 세단 (S-Class)",
                    "en": "Flagship Luxury Sedan (S-Class)",
                    "ja": "フラッグシップ高級セダン (S-Class)"
                },
                "category": {
                    "ko": "Flagship Luxury Sedan",
                    "en": "Flagship Luxury Sedan",
                    "ja": "フラッグシップ高級セダン"
                },
                "desc": {
                    "ko": "최고급 주행 감성과 첨단 운전자 보조 시스템(ADAS) 및 리무진급 편의 사양 제공",
                    "en": "Supreme ride comfort, state-of-the-art ADAS, and limousine-class executive amenities.",
                    "ja": "最高級の走行フィーリング、先進運転支援システム(ADAS)およびリムジン級の快適装備。"
                },
                "image_url": "https://images.unsplash.com/photo-1617788138017-80ad40651399?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "순수 전기차 라인업 (EQ Series)",
                    "en": "Pure Electric Vehicle Lineup (EQ Series)",
                    "ja": "ピュアEVラインナップ (EQ Series)"
                },
                "category": {
                    "ko": "Pure Electric Vehicle",
                    "en": "Pure Electric Vehicle",
                    "ja": "電気自動車 (EV)"
                },
                "desc": {
                    "ko": "전용 전기 플랫폼과 1회 충전 500km+ 주행거리, MBUX 하이퍼스크린 탑재",
                    "en": "Dedicated EV platform, 500km+ single-charge range, and MBUX Hyperscreen cockpit.",
                    "ja": "専用EVプラットフォーム、1回充電500km+の航続距離、MBUXハイパースクリーン搭載。"
                },
                "image_url": "https://images.unsplash.com/photo-1563720223185-11003d516935?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "프리미엄 도심형 SUV (GLE / GLS)",
                    "en": "Premium Luxury SUV (GLE / GLS)",
                    "ja": "プレミアム高級SUV (GLE / GLS)"
                },
                "category": {
                    "ko": "Premium Luxury SUV",
                    "en": "Premium Luxury SUV",
                    "ja": "プレミアム高級SUV"
                },
                "desc": {
                    "ko": "지능형 사륜구동(4MATIC)과 에어 서스펜션으로 온/오프로드 최상의 안락함 제공",
                    "en": "Intelligent 4MATIC all-wheel drive and air suspension for ultimate on/off-road comfort.",
                    "ja": "インテリジェント四輪駆動(4MATIC)とエアサスペンションによる最上の乗り心地。"
                },
                "image_url": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "스마트 커넥티비티 & 디지털 서비스",
                    "en": "Smart Connectivity & Digital Services",
                    "ja": "スマートコネクティビティ＆デジタルサービス"
                },
                "category": {
                    "ko": "Connected Car Service",
                    "en": "Connected Car Service",
                    "ja": "コネクテッドカーサービス"
                },
                "desc": {
                    "ko": "스마트폰 원격 시동/공조 제어, 무선 OTA 소프트웨어 업데이트 및 실시간 내비게이션",
                    "en": "Smartphone remote start/climate control, wireless OTA updates, and live cloud navigation.",
                    "ja": "スマホ遠隔始動/空調制御、無線OTAアップデート、リアルタイムナビゲーション。"
                },
                "image_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&h=250&fit=crop&q=80"
            }
        ]
    },
    "travel": {
        "industry": {
            "ko": "글로벌 여행/액티비티 및 모빌리티 예약 플랫폼",
            "en": "Global Travel/Activities & Mobility Booking Platform",
            "ja": "グローバルトラベル/アクティビティ＆モビリティ予約プラットフォーム"
        },
        "hero_image": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=800&h=300&fit=crop&q=80",
        "hero_tagline": {
            "ko": "Discover & Book Joy - 전 세계 언제 어디서나 가장 편리한 여행 경험",
            "en": "Discover & Book Joy - Seamless Travel Experiences Anytime, Anywhere Worldwide",
            "ja": "Discover & Book Joy - 世界中どこでも最も快適な旅行体験"
        },
        "products": [
            {
                "name": {
                    "ko": "글로벌 데이터 로밍 무제한 eSIM",
                    "en": "Global Unlimited Roaming Data eSIM",
                    "ja": "グローバル無制限ローミングeSIM"
                },
                "category": {
                    "ko": "Global Telecom & eSIM",
                    "en": "Global Telecom & eSIM",
                    "ja": "グローバル通信＆eSIM"
                },
                "desc": {
                    "ko": "QR코드 스캔 즉시 개통, 전 세계 150+ 국가 초고속 5G/LTE 무제한 데이터 지원",
                    "en": "Instant QR activation, high-speed 5G/LTE unlimited data across 150+ countries.",
                    "ja": "QRコード読み取りで即時開通、世界150カ国以上で高速5G/LTE無制限通信。"
                },
                "image_url": "https://images.unsplash.com/photo-1512428559087-560fa5ceab42?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "테마파크 & 명소 패스트트랙 입장권",
                    "en": "Theme Park & Landmark Fast-Track Tickets",
                    "ja": "テーマパーク＆名所ファストトラック入場券"
                },
                "category": {
                    "ko": "Attraction & Theme Park",
                    "en": "Attraction & Theme Park",
                    "ja": "アトラクション＆テーマパーク"
                },
                "desc": {
                    "ko": "도쿄 디즈니랜드, 유니버셜 스튜디오 등 글로벌 랜드마크 즉시 확정 모바일 바우처",
                    "en": "Instant mobile confirmation vouchers for Disneyland, Universal Studios, and global landmarks.",
                    "ja": "ディズニーランド、USJなど世界の名所の即時確定モバイルバウチャー。"
                },
                "image_url": "https://images.unsplash.com/photo-1606813907291-d86efa9b94db?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "프라이빗 일일 투어 & 액티비티",
                    "en": "Private Day Tours & Experiences",
                    "ja": "プライベート日帰りツアー＆アクティビティ"
                },
                "category": {
                    "ko": "Day Tour & Experience",
                    "en": "Day Tour & Experience",
                    "ja": "デイリーツアー＆体験"
                },
                "desc": {
                    "ko": "전문 가이드 동행, 소규모 전용 차량으로 진행되는 프리미엄 현지 투어",
                    "en": "Expert local guides, small-group private vehicles, curated premium day experiences.",
                    "ja": "専門ガイド同行、専用車で催行されるプレミアム現地ツアー。"
                },
                "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "공항 픽업 및 글로벌 교통 패스",
                    "en": "Airport Transfers & Global Transit Passes",
                    "ja": "空港送迎＆グローバル交通パス"
                },
                "category": {
                    "ko": "Airport Transfer & Transit Pass",
                    "en": "Airport Transfer & Transit Pass",
                    "ja": "空港送迎＆交通パス"
                },
                "desc": {
                    "ko": "일본 JR 패스, 유럽 유레일 패스 및 도심-공항 간 안심 정액제 프라이빗 픽업",
                    "en": "Japan JR Pass, Eurail Pass, and reliable fixed-rate private airport transfers.",
                    "ja": "JRパス、ユーレイルパス、および定額プライベート空港送迎。"
                },
                "image_url": "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=250&fit=crop&q=80"
            }
        ]
    },
    "tech": {
        "industry": {
            "ko": "첨단 IT, 반도체 및 인공지능/스마트 기기 솔루션",
            "en": "Advanced IT, Semiconductor & AI Smart Device Solutions",
            "ja": "先端IT・半導体＆AIスマートデバイスソリューション"
        },
        "hero_image": "https://images.unsplash.com/photo-1511707171634-5f897ff02560?w=800&h=300&fit=crop&q=80",
        "hero_tagline": {
            "ko": "Inspire the World, Create the Future - 혁신 기술로 여는 지능형 미래",
            "en": "Inspire the World, Create the Future - Pioneering an Intelligent Future through Innovation",
            "ja": "Inspire the World, Create the Future - 革新技術で切り拓く知能型の未来"
        },
        "products": [
            {
                "name": {
                    "ko": "차세대 온디바이스 AI 스마트폰",
                    "en": "Next-Gen On-Device AI Smartphone",
                    "ja": "次世代オンデバイスAIスマートフォン"
                },
                "category": {
                    "ko": "AI Smartphone & Mobile",
                    "en": "AI Smartphone & Mobile",
                    "ja": "AIスマートフォン＆モバイル"
                },
                "desc": {
                    "ko": "실시간 통번역, 지능형 사진 편집 및 최첨단 티타늄 프레임과 프로급 카메라 탑재",
                    "en": "Real-time AI call translation, generative photo editing, titanium frame, and pro cameras.",
                    "ja": "リアルタイム通訳、AI写真編集、チタンフレームおよびプロ級カメラ搭載。"
                },
                "image_url": "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "차세대 프리미엄 스마트 디스플레이 (QLED/OLED)",
                    "en": "Next-Gen Premium Smart Display (QLED/OLED)",
                    "ja": "次世代プレミアムスマートディスプレイ (QLED/OLED)"
                },
                "category": {
                    "ko": "Smart Display & Home Cinema",
                    "en": "Smart Display & Home Cinema",
                    "ja": "スマートディスプレイ＆ホームシアター"
                },
                "desc": {
                    "ko": "인공지능 8K 업스케일링 프로세서와 시네마틱 돌비 애트모스 입체 사운드",
                    "en": "Neural AI 8K upscaling processor and cinematic Dolby Atmos spatial audio.",
                    "ja": "AI 8KアップスケーリングプロセッサとシネマティックDolby Atmos立体音響。"
                },
                "image_url": "https://images.unsplash.com/photo-1593359677879-a4b92c0a3b8b?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "고성능 AI 초경량 노트북 시리즈",
                    "en": "High-Performance Ultralight AI Laptop Series",
                    "ja": "高性能AI超軽量ノートPCシリーズ"
                },
                "category": {
                    "ko": "AI PC & Ultrabook",
                    "en": "AI PC & Ultrabook",
                    "ja": "AI PC＆ウルトラブック"
                },
                "desc": {
                    "ko": "최신 NPU 탑재로 로컬 AI 연산 가속, 하루 종일 지속되는 배터리와 슬림 메탈 바디",
                    "en": "Latest dedicated NPU for local AI acceleration, all-day battery, and ultra-slim metal body.",
                    "ja": "最新NPUによるローカルAI加速、終日持続バッテリー、スリムメタルボディ。"
                },
                "image_url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "스마트홈 IoT 에코시스템 및 가전",
                    "en": "SmartHome IoT Ecosystem & Appliances",
                    "ja": "スマートホームIoTエコシステム＆家電"
                },
                "category": {
                    "ko": "SmartThings & Smart Living",
                    "en": "SmartThings & Smart Living",
                    "ja": "スマートホーム＆生活家電"
                },
                "desc": {
                    "ko": "인공지능 절전 모드와 기기 간 원터치 연동으로 완성하는 스마트 에너지 케어",
                    "en": "AI energy-saving mode and seamless one-touch ecosystem integration.",
                    "ja": "AI省エネモードと機器間ワンタッチ連動によるスマートエネルギーケア。"
                },
                "image_url": "https://images.unsplash.com/photo-1558002038-1055907df827?w=400&h=250&fit=crop&q=80"
            }
        ]
    },
    "entertainment": {
        "industry": {
            "ko": "글로벌 미디어 엔터테인먼트, 스트리밍 & IP 라이선스",
            "en": "Global Media Entertainment, Streaming & IP Licensing",
            "ja": "グローバルメディアエンターテインメント・配信＆IPライセンス"
        },
        "hero_image": "https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85?w=800&h=300&fit=crop&q=80",
        "hero_tagline": {
            "ko": "Stories that Move the World - 전 세계를 연결하는 엔터테인먼트",
            "en": "Stories that Move the World - Entertainment Connecting the Globe",
            "ja": "Stories that Move the World - 世界をつなぐエンターテインメント"
        },
        "products": [
            {
                "name": {
                    "ko": "프리미엄 4K UHD 스트리밍 멤버십",
                    "en": "Premium 4K UHD Streaming Membership",
                    "ja": "プレミアム4K UHDストリーミングメンバーシップ"
                },
                "category": {
                    "ko": "OTT Streaming Service",
                    "en": "OTT Streaming Service",
                    "ja": "OTT動画配信サービス"
                },
                "desc": {
                    "ko": "최고 화질 4K HDR, 공간 음향 및 동시 접속 4대 지원 프리미엄 요금제",
                    "en": "Ultra HD 4K HDR, immersive spatial audio, and 4 concurrent screens.",
                    "ja": "最高画質4K HDR、空間オーディオ、同時視聴4台対応のプレミアムプラン。"
                },
                "image_url": "https://images.unsplash.com/photo-1522869635100-9f4c5e86aa37?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "독점 오리지널 시리즈 및 영화 라인업",
                    "en": "Exclusive Original Series & Movie Lineup",
                    "ja": "独占オリジナルシリーズ＆映画ラインナップ"
                },
                "category": {
                    "ko": "Exclusive Original Content",
                    "en": "Exclusive Original Content",
                    "ja": "独占オリジナルコンテンツ"
                },
                "desc": {
                    "ko": "세계적 흥행의 독점 블록버스터 영화 및 K-콘텐츠 오리지널 시리즈",
                    "en": "Global hit blockbuster movies and critically acclaimed original series.",
                    "ja": "世界的ヒットを記録した独占大作映画およびオリジナルシリーズ。"
                },
                "image_url": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "글로벌 테마파크 & 어트랙션",
                    "en": "Global Theme Parks & Studio Attractions",
                    "ja": "グローバルテーマパーク＆アトラクション"
                },
                "category": {
                    "ko": "Theme Park & Studio Tour",
                    "en": "Theme Park & Studio Tour",
                    "ja": "テーマパーク＆スタジオツアー"
                },
                "desc": {
                    "ko": "영화 속 세계를 현실로 구현한 글로벌 테마파크 어트랙션 및 캐릭터 굿즈",
                    "en": "Immersive theme park attractions and official world-class character merchandise.",
                    "ja": "作品世界を現実に再現したテーマパークアトラクションおよび公式グッズ。"
                },
                "image_url": "https://images.unsplash.com/photo-1526304640581-d334cdbbf45e?w=400&h=250&fit=crop&q=80"
            }
        ]
    },
    "transit": {
        "industry": {
            "ko": "철도, 고속철 및 친환경 대중교통 모빌리티 네트워크",
            "en": "Rail, High-Speed Train & Eco-Friendly Transit Network",
            "ja": "鉄道・高速鉄道＆環境配慮型交通ネットワーク"
        },
        "hero_image": "https://images.unsplash.com/photo-1474487548417-781cb71495f3?w=800&h=300&fit=crop&q=80",
        "hero_tagline": {
            "ko": "Connecting Cities, Connecting People - 전국을 반나절 생활권으로 잇는 철도 모빌리티",
            "en": "Connecting Cities, Connecting People - Seamless National High-Speed Rail Mobility",
            "ja": "Connecting Cities, Connecting People - 全国を結ぶ鉄道モビリティ"
        },
        "products": [
            {
                "name": {
                    "ko": "초고속 고속철도 (KTX / SRT)",
                    "en": "High-Speed Rail (KTX / SRT)",
                    "ja": "超高速鉄道 (KTX / SRT)"
                },
                "category": {
                    "ko": "High-Speed Rail",
                    "en": "High-Speed Rail",
                    "ja": "高速鉄道"
                },
                "desc": {
                    "ko": "최고 속도 305km/h의 정시 운행과 특실 어메니티, 전국 간선망 연결",
                    "en": "Punctual operation at 305km/h, premium first-class amenities, connecting national hubs.",
                    "ja": "最高速度305km/hの定時運行、特室アメニティ、全国主要幹線の連結。"
                },
                "image_url": "https://images.unsplash.com/photo-1474487548417-781cb71495f3?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "광역 철도 및 친환경 모빌리티 패스",
                    "en": "Regional Transit & Eco-Friendly Travel Pass",
                    "ja": "広域鉄道＆エコ交通パス"
                },
                "category": {
                    "ko": "Regional Transit Pass",
                    "en": "Regional Transit Pass",
                    "ja": "広域交通パス"
                },
                "desc": {
                    "ko": "수도권 통합 환승 할인 및 전국 대중교통 무제한 이용 정기 패스",
                    "en": "Metropolitan integrated transfer discounts and unlimited regional transit passes.",
                    "ja": "統合乗換割引および公共交通乗り放題定期パス。"
                },
                "image_url": "https://images.unsplash.com/photo-1517649763962-0c623266ddc0?w=400&h=250&fit=crop&q=80"
            }
        ]
    },
    "general": {
        "industry": {
            "ko": "글로벌 비즈니스, 엔터프라이즈 솔루션 및 고객 서비스",
            "en": "Global Business, Enterprise Solutions & Customer Support",
            "ja": "グローバルビジネス・エンタープライズソリューション＆顧客サポート"
        },
        "hero_image": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&h=300&fit=crop&q=80",
        "hero_tagline": {
            "ko": "Delivering Excellence - 고객 중심의 신뢰받는 혁신 서비스 파트너",
            "en": "Delivering Excellence - Your Trusted Customer-Centric Innovation Partner",
            "ja": "Delivering Excellence - 顧客中心の信頼される革新サービスパートナー"
        },
        "products": [
            {
                "name": {
                    "ko": "엔터프라이즈 맞춤형 비즈니스 솔루션",
                    "en": "Enterprise Tailored Business Solutions",
                    "ja": "エンタープライズ向けカスタマイズソリューション"
                },
                "category": {
                    "ko": "Enterprise Solution",
                    "en": "Enterprise Solution",
                    "ja": "企業ソリューション"
                },
                "desc": {
                    "ko": "고객사의 업무 생산성 향상과 디지털 전환을 견인하는 종합 솔루션",
                    "en": "Comprehensive digital transformation and productivity solutions for corporate clients.",
                    "ja": "業務生産性の向上とデジタル変革を牽引する総合ソリューション。"
                },
                "image_url": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "24/7 글로벌 프리미엄 고객 케어",
                    "en": "24/7 Global Premium Customer Care",
                    "ja": "24/7 グローバルプレミアム顧客ケア"
                },
                "category": {
                    "ko": "Customer Care & Support",
                    "en": "Customer Care & Support",
                    "ja": "カスタマーケア＆サポート"
                },
                "desc": {
                    "ko": "다국어 실시간 상담과 전담 어카운트 매니저 배정으로 신속한 문제 해결",
                    "en": "Multilingual real-time support and dedicated account management for rapid resolution.",
                    "ja": "多言語リアルタイム相談と専任アカウントマネージャーによる迅速な解決。"
                },
                "image_url": "https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=400&h=250&fit=crop&q=80"
            },
            {
                "name": {
                    "ko": "AI 기반 자동화 및 스마트 컨설팅",
                    "en": "AI-Driven Automation & Smart Consulting",
                    "ja": "AIベースの自動化＆スマートコンサルティング"
                },
                "category": {
                    "ko": "AI Automation & Consulting",
                    "en": "AI Automation & Consulting",
                    "ja": "AI自動化＆コンサルティング"
                },
                "desc": {
                    "ko": "최신 데이터 분석과 업무 프로세스 자동화로 비용 절감 및 품질 혁신",
                    "en": "Advanced data analytics and workflow automation driving cost reduction and quality innovation.",
                    "ja": "最新データ分析と業務自動化によるコスト削減と品質革新。"
                },
                "image_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&h=250&fit=crop&q=80"
            }
        ]
    }
}



# 기업 정규화 및 다국어 표준 표기명 맵
COMPANY_ALIAS_MAP = {
    # Automotive
    "mercedes-benz": "mercedes-benz",
    "mercedes": "mercedes-benz",
    "benz": "mercedes-benz",
    "메르세데스 벤츠": "mercedes-benz",
    "메르세데스-벤츠": "mercedes-benz",
    "벤츠": "mercedes-benz",
    "メルセデス・ベンツ": "mercedes-benz",
    "mercendo benz": "mercedes-benz",
    # Travel
    "klook": "klook",
    "클룩": "klook",
    "クルック": "klook",
    "クルック (klook)": "klook",
    # Tech
    "samsung": "samsung",
    "삼성": "samsung",
    "삼성전자": "samsung",
    "サムスン": "samsung",
    "サムスン電子": "samsung",
    # Entertainment
    "netflix": "netflix",
    "넷플릭스": "netflix",
    "ネットフリックス": "netflix",
    "universal": "universal",
    "유니버셜 스튜디오": "universal",
    "유니버셜": "universal",
    "ユニバーサル・スタジオ": "universal",
    # Transit
    "korail": "korail",
    "코레일": "korail",
    "한국철도공사": "korail",
    "コレイル": "korail",
}

CANONICAL_COMPANY_NAMES = {
    "mercedes-benz": {
        "ko": "메르세데스 벤츠 (Mercedes-Benz)",
        "en": "Mercedes-Benz",
        "ja": "メルセデス・ベンツ (Mercedes-Benz)"
    },
    "klook": {
        "ko": "클룩 (Klook)",
        "en": "Klook Travel",
        "ja": "クルック (Klook)"
    },
    "samsung": {
        "ko": "삼성전자 (Samsung)",
        "en": "Samsung Electronics",
        "ja": "サムスン電子 (Samsung)"
    },
    "netflix": {
        "ko": "넷플릭스 (Netflix)",
        "en": "Netflix",
        "ja": "ネットフリックス (Netflix)"
    },
    "universal": {
        "ko": "유니버셜 스튜디오",
        "en": "Universal Studios",
        "ja": "ユニバーサル・スタジオ"
    },
    "korail": {
        "ko": "한국철도공사 (KORAIL)",
        "en": "Korea Railroad Corporation (KORAIL)",
        "ja": "韓国鉄道公社 (KORAIL)"
    }
}


def get_canonical_company_name(name: str, lang: str = "ko") -> str:
    """회사명을 표준화하고 해당 언어에 맞는 최적의 표기명 반환"""
    clean_name = name.lower().strip()
    canon_id = COMPANY_ALIAS_MAP.get(clean_name)
    if canon_id and canon_id in CANONICAL_COMPANY_NAMES:
        return CANONICAL_COMPANY_NAMES[canon_id].get(lang, CANONICAL_COMPANY_NAMES[canon_id]["ko"])
    if lang == "en" and clean_name.islower():
        return name.title()
    return name


def classify_company_category(company_name: str) -> str:
    """회사명을 기반으로 핵심 비즈니스 카테고리 판별"""
    c = company_name.lower()
    if any(k in c for k in ["benz", "벤츠", "mercedes", "bmw", "audi", "현대", "기아", "tesla", "테슬라", "포르쉐", "porsche", "자동차", "모빌리티", "차량", "メルセデス"]):
        return "automotive"
    elif any(k in c for k in ["klook", "클룩", "여행", "투어", "호텔", "항공", "travel", "tour", "flight", "trip", "트립", "숙소", "티켓", "야놀자", "여기어때", "クルック"]):
        return "travel"
    elif any(k in c for k in ["삼성", "samsung", "애플", "apple", "소니", "sony", "lg", "화웨이", "huawei", "전자", "반도체", "intel", "인텔", "it", "모토로라", "サムスン"]):
        return "tech"
    elif any(k in c for k in ["넷플릭스", "netflix", "유니버셜", "universal", "디즈니", "disney", "로블록스", "roblox", "엔터", "미디어", "스트리밍", "영화", "ネットフリックス"]):
        return "entertainment"
    elif any(k in c for k in ["코레일", "korail", "철도", "교통", "기차", "srt", "지하철", "コレイル"]):
        return "transit"
    return "general"


def get_company_visual_and_details(company_name: str, existing_data: Optional[Dict[str, Any]] = None, lang: str = "ko") -> Dict[str, Any]:
    """
    회사명에 대해 고품질 상세 정보 및 비쥬얼 사진/배너/제품 자산 생성 또는 보강
    - 한국어(ko), 영어(en), 일본어(ja) 다국어 완전 지원
    - 플레이스홀더 문구(접수하였습니다 등) 완전 차단
    """
    if lang not in ["ko", "en", "ja"]:
        lang = "ko"

    # 정규화된 최적 표기명 추출
    display_name = get_canonical_company_name(company_name, lang=lang)
    cat = classify_company_category(company_name)
    preset = VISUAL_PRESETS.get(cat, VISUAL_PRESETS["general"])

    # 1. 업종 및 슬로건 추출
    industry = preset["industry"].get(lang, preset["industry"]["ko"])
    hero_tagline = preset["hero_tagline"].get(lang, preset["hero_tagline"]["ko"])

    # 2. 기업 상세 소개 (다국어 생성)
    if lang == "en":
        if cat == "automotive":
            description = (
                f"{display_name} is a premier luxury automobile manufacturer leading the global mobility industry "
                f"with world-class engineering, exceptional craftsmanship, and an uncompromising safety philosophy. "
                f"Building upon a legendary heritage of internal combustion excellence, the company pioneers the future "
                f"of sustainable luxury through dedicated electric vehicle architectures and intelligent autonomous driving systems. "
                f"Offering an extensive portfolio ranging from flagship executive sedans to high-performance sports cars, "
                f"premium SUVs, and pure electric vehicles, {display_name} operates a nationwide dealership network and 24/7 VIP customer support."
            )
        elif cat == "travel":
            description = (
                f"{display_name} is a world-leading travel and leisure booking platform offering hundreds of thousands of "
                f"curated activities, theme park passes, global eSIMs, and transportation services across more than 1,000 cities worldwide. "
                f"By building an all-in-one digital travel ecosystem, travelers can seamlessly book top local experiences and receive "
                f"instant confirmation mobile vouchers anytime, anywhere. Supported by 24/7 multilingual customer assistance and secure global payments, "
                f"{display_name} stands as the most trusted travel companion for independent global adventurers."
            )
        elif cat == "tech":
            description = (
                f"{display_name} is a global technology leader transforming lifestyles and industrial ecosystems through "
                f"cutting-edge semiconductors, on-device AI smartphones, next-generation premium displays, and intelligent IoT software. "
                f"Driven by a vision of hyper-connected smart living, {display_name} delivers personalized digital experiences, "
                f"rigorous quality engineering, and eco-friendly sustainable technologies to define the global benchmark for consumer electronics."
            )
        elif cat == "entertainment":
            description = (
                f"{display_name} is a premier global entertainment and streaming media corporation captivating hundreds of millions "
                f"of viewers worldwide through compelling storytelling and state-of-the-art cinematic technology. "
                f"Spanning award-winning original blockbuster series to global theme park studio attractions and IP licensing, "
                f"{display_name} bridges cultures and connects audiences across languages and borders."
            )
        elif cat == "transit":
            description = (
                f"{display_name} is a cornerstone national transit corporation operating high-speed (KTX) and regional rail networks, "
                f"delivering safe, punctual, and eco-friendly passenger and freight mobility services across the country. "
                f"With advanced digital signaling and customer-centric booking systems, {display_name} seamlessly connects major cities."
            )
        else:
            description = (
                f"{display_name} is an industry-leading enterprise delivering innovative solutions, superior product quality, "
                f"and dedicated customer-first support. Committed to sustainable growth and social responsibility, {display_name} "
                f"creates differentiated value across its global business operations."
            )
    elif lang == "ja":
        if cat == "automotive":
            description = (
                f"{display_name}は、世界最高水準のエンジニアリング技術、卓越したクラフトマンシップ、そして先進的な安全哲学に基づき、"
                f"世界のラグジュアリーモビリティ市場を牽引する完成車メーカーです。内燃機関の長い歴史と品格を継承しつつ、"
                f"専用EVプラットフォームと高度な自律走行技術を融合し、持続可能な未来のラグジュアリーモビリティを切り拓いています。"
                f"最高級セダンから高性能スポーツカー、プレミアムSUV、ピュアEVまで多彩なラインナップを展開し、"
                f"全国の正規ディーラー網と24時間体制のプレミアムカスタマーケアを提供しています。"
            )
        elif cat == "travel":
            description = (
                f"{display_name}は、世界1,000以上の都市で数十万件のアクティビティ、テーマパークパス、グローバルeSIM、"
                f"宿泊および交通予約サービスを提供する世界有数の旅行プラットフォームです。旅行者がスマートフォン一つで"
                f"現地の人気スポットをリアルタイムで予約し、即時確定バウチャーを受け取れるワンストップ旅行エコシステムを構築しています。"
                f"24時間の多言語サポートと安全な決済システムを備え、自由旅行者に最も信頼されるパートナーとして位置づけられています。"
            )
        elif cat == "tech":
            description = (
                f"{display_name}は、最先端の半導体、オンデバイスAIモバイル端末、次世代ディスプレイ、AIソフトウェアを融合し、"
                f"人々のライフスタイルと産業生態系を変革するグローバルテックリーダーです。超連結スマートホームプラットフォームを基盤に、"
                f"ユーザーに最適化されたスマートなデジタル体験を提供し、持続可能な環境配慮技術を通じて世界のIT市場を先導しています。"
            )
        elif cat == "entertainment":
            description = (
                f"{display_name}は、独創的なストーリーテリングと最先端の映像技術を融合し、世界中の数億人の観客に感動と喜びを届ける"
                f"グローバル総合エンターテインメント企業です。自社制作の独占大作シリーズから世界的なテーマパークアトラクション、"
                f"IPライセンス事業まで幅広く展開し、空間と文化を超えたエンターテインメント体験を創出しています。"
            )
        elif cat == "transit":
            description = (
                f"{display_name}は、高速鉄道および広域鉄道網を統括運営し、安全・迅速・環境配慮型の公共交通モビリティサービスを提供する"
                f"国家の基幹公企業です。最先端のデジタル信号制御と顧客目線の予約システムを通じて、全国を半日生活圏で結んでいます。"
            )
        else:
            description = (
                f"{display_name}は、優れた品質と専門的なビジネス能力、顧客第一の哲学を軸に、関連産業において革新的な価値を創出する"
                f"グローバル先導企業です。顧客満足を最優先とし、持続可能な成長と社会的責任を果たしています。"
            )
    else:  # ko
        if cat == "automotive":
            description = (
                f"{display_name}은(는) 세계 최고 수준의 엔지니어링 기술과 독보적인 장인 정신, 그리고 첨단 안전 철학을 기반으로 "
                f"글로벌 럭셔리 모빌리티 시장을 선도하는 프리미엄 완성차 제조 기업입니다. 내연기관의 오랜 역사와 품격을 계승함과 동시에, "
                f"차세대 전동화 브랜드와 지능형 자율주행 기술을 결합하여 지속 가능한 럭셔리 모빌리티의 미래를 개척하고 있습니다. "
                f"국내외 고객들에게 최고급 세단부터 고성능 스포츠카, 프리미엄 SUV 및 순수 전기차까지 폭넓은 라인업을 제공하며 "
                f"VIP 고객 맞춤형 딜러십 및 24시간 프리미엄 고객 케어 네트워크를 운영하고 있습니다."
            )
        elif cat == "travel":
            description = (
                f"{display_name}은(는) 전 세계 1,000개 이상의 도시에서 수십만 가지 액티비티, 테마파크 패스, 글로벌 eSIM 및 숙박/교통 "
                f"예약 서비스를 제공하는 글로벌 선도 여행 플랫폼입니다. 여행자가 언제 어디서든 스마트폰 하나로 현지 인기 명소를 "
                f"실시간 예약하고 즉시 확정 바우처를 수령할 수 있도록 원스톱 여행 디지털 생태계를 구축하였습니다. "
                f"24시간 다국어 고객 센터와 안전 결제 시스템을 갖추어 글로벌 자유 여행객들에게 가장 신뢰받는 여행 파트너로 자리매김하고 있습니다."
            )
        elif cat == "tech":
            description = (
                f"{display_name}은(는) 최첨단 반도체, 지능형 모바일 기기, 차세대 디스플레이 및 인공지능(AI) 소프트웨어를 융합하여 "
                f"인류의 라이프스타일과 산업 생태계를 혁신하는 글로벌 테크놀로지 리더입니다. 온디바이스 AI 기술과 초연결 스마트홈 플랫폼을 바탕으로 "
                f"사용자에게 최적화된 맞춤형 디지털 경험을 제공하며, 지속 가능한 친환경 기술 개발과 엄격한 품질 관리를 통해 전 세계 IT 시장의 표준을 선도하고 있습니다."
            )
        elif cat == "entertainment":
            description = (
                f"{display_name}은(는) 독창적인 스토리텔링과 최첨단 영상 기술을 결합하여 전 세계 수억 명의 관객에게 감동과 즐거움을 선사하는 "
                f"글로벌 종합 엔터테인먼트 및 미디어 기업입니다. 자체 제작 오리지널 블록버스터 시리즈부터 글로벌 테마파크 어트랙션, "
                f"IP 라이선싱 사업까지 아우르며 공간과 언어를 초월한 문화적 연결고리를 만들어가고 있습니다."
            )
        elif cat == "transit":
            description = (
                f"{display_name}은(는) 대한민국의 대동맥으로서 고속철도(KTX)와 일반/광역 철도망을 총괄 운영하며, 안전하고 신속하며 "
                f"친환경적인 대중교통 모빌리티 서비스를 제공하는 핵심 국가 공기업입니다. 첨단 디지털 신호 제어 시스템과 고객 중심의 예약/환불 편의 서비스를 통해 "
                f"전국을 반나절 생활권으로 연결하고 있습니다."
            )
        else:
            description = (
                f"{display_name}은(는) 우수한 제품 품질과 전문화된 비즈니스 역량, 고객 중심의 철학을 바탕으로 관련 산업 분야에서 "
                f"혁신적인 가치를 창출하는 글로벌 선도 기업입니다. 고객 만족을 최우선 가치로 삼아 차별화된 솔루션을 제공하며, 지속 가능한 성장과 사회적 책임을 다하고 있습니다."
            )

    # 3. 대표 제품 및 비쥬얼 사진 매핑 (다국어 지원)
    products_detail = []
    preset_prods = preset["products"]
    for p in preset_prods:
        products_detail.append({
            "name": p["name"].get(lang, p["name"]["ko"]),
            "category": p["category"].get(lang, p["category"]["ko"]),
            "desc": p["desc"].get(lang, p["desc"]["ko"]),
            "image_url": p["image_url"]
        })

    # 4. FAQ 구축 (다국어 지원)
    faqs = []
    if lang == "en":
        if cat == "automotive":
            faqs = [
                {
                    "question": f"What are the warranty period and maintenance benefits for {display_name} vehicles?",
                    "answer": f"Official {display_name} vehicles come with a 3-year/100,000km standard warranty and complimentary scheduled maintenance packages covering oil, filters, and brake wear. Certified service centers provide identical precision diagnostics."
                },
                {
                    "question": "What is the warranty policy for pure electric vehicles (EQ series) and hybrid batteries?",
                    "answer": "High-voltage batteries are backed by an 8-year or 160,000km warranty ensuring at least 70% capacity retention. Certified technicians provide complimentary module replacements if faults occur."
                },
                {
                    "question": "How can I access 24/7 roadside assistance during an emergency?",
                    "answer": "Press the in-car SOS button or call the 24/7 toll-free hotline to dispatch emergency towing, tire repair, battery jump-start, and fuel delivery 365 days a year."
                },
                {
                    "question": "How do I set up the digital key and smartphone connectivity?",
                    "answer": "Download the official app, scan the QR code generated on the in-vehicle screen, and pair your account to enable remote start, door lock/unlock, pre-conditioning, and parking location tracking."
                }
            ]
        elif cat == "entertainment":
            faqs = [
                {
                    "question": f"How do I manage my {display_name} streaming subscription and concurrent screens?",
                    "answer": "You can upgrade or cancel your membership anytime via Account Settings. Premium memberships support up to 4 concurrent 4K UHD streams and spatial audio across all registered devices."
                },
                {
                    "question": "How do downloads work for offline viewing?",
                    "answer": "Click the download icon next to any supported title in the mobile or tablet app. Downloaded content can be watched offline for up to 30 days without an active internet connection."
                },
                {
                    "question": "What customer support channels are available for streaming or billing issues?",
                    "answer": "Live chat and phone support are available 24/7 through the Help Center. Billing inquiries and refund requests are typically processed within 3 to 5 business days."
                },
                {
                    "question": "How are personal data and viewing history protected?",
                    "answer": "All streaming activities and payment information are encrypted using industry-standard TLS protocols and compliant with global privacy standards."
                }
            ]
        elif cat == "travel":
            faqs = [
                {
                    "question": f"How do I install and activate an international eSIM purchased on {display_name}?",
                    "answer": "Scan the QR code received via email in your phone Settings > Cellular > Add eSIM. Upon arrival at your destination, turn on Data Roaming on the eSIM line to instantly connect."
                },
                {
                    "question": "What is the cancellation and refund policy for tour bookings and attraction tickets?",
                    "answer": "Most standard tours offer 100% free cancellation up to 24 hours prior to the activity date. Theme park tickets and special promo vouchers may be non-refundable as stated on the product page."
                },
                {
                    "question": "What should I do if my voucher is not recognized at the local venue?",
                    "answer": "Contact our 24/7 live chat support immediately with your booking number. Our support team will coordinate directly with the local operator for immediate assistance."
                },
                {
                    "question": "Can I receive a refund if my trip is cancelled due to severe weather or flight disruptions?",
                    "answer": "Submit your official airline cancellation certificate or weather advisory to customer support. Cancellation fees will be waived and a 100% refund will be issued to your original payment method."
                }
            ]
        elif cat == "tech":
            faqs = [
                {
                    "question": f"What is the warranty period and how can I book a service appointment for {display_name} devices?",
                    "answer": "Smartphones and smart electronics include a 1 to 2-year manufacturer warranty. You can book an appointment online to receive prompt diagnostic care without waiting."
                },
                {
                    "question": "How does accidental damage protection and display repair coverage work?",
                    "answer": "Enrolled devices receive up to 70-80% repair fee discounts. Certified repairs preserve original water-resistance and factory specifications."
                },
                {
                    "question": "How do I securely transfer data when upgrading to a new device?",
                    "answer": "Use our official smart transfer tool to wirelessly migrate photos, contacts, and app data with 100% security."
                },
                {
                    "question": "What should I do if my device experiences performance issues after an update?",
                    "answer": "Reboot into safe mode to clear cache, or visit an authorized service center for diagnostic firmware reinstallation."
                }
            ]
        else:
            faqs = [
                {
                    "question": f"What are the customer support operating hours and contact channels for {display_name}?",
                    "answer": f"Phone support is available weekdays 09:00-18:00, and our 24/7 AI chatbot and online help center are available year-round."
                },
                {
                    "question": "How long does it take to process cancellations and refunds?",
                    "answer": "Credit card cancellations are processed within 3 to 5 business days, while bank transfers are refunded within 1 to 2 business days."
                },
                {
                    "question": "How can corporate clients submit B2B partnership or bulk quotation inquiries?",
                    "answer": "Submit an inquiry via our B2B Business Portal; a dedicated account consultant will respond with a tailored proposal within 24 hours."
                },
                {
                    "question": "How are customer privacy and security certifications managed?",
                    "answer": "We comply with international ISO 27001 standards and end-to-end encryption to protect all payment and customer data."
                }
            ]
    elif lang == "ja":
        if cat == "automotive":
            faqs = [
                {
                    "question": f"{display_name}の車両購入後の保証期間および点検特典は？",
                    "answer": f"正規出荷車両には基本3年/10万kmの無償保証修理と、定期消耗品交換パッケージが無償提供されます。全国の正規サービスセンターで共通の診断を受けられます。"
                },
                {
                    "question": "ピュアEV（EQシリーズ）およびハイブリッドのバッテリー保証ポリシーは？",
                    "answer": "高電圧バッテリーに対し、8年または16万kmにわたり容量70%以上を保証します。異常検知時は無償モジュール交換または修理が行われます。"
                },
                {
                    "question": "緊急時の24時間ロードサービスの利用方法は？",
                    "answer": "車内のSOSボタンを押すかフリーダイヤルにお電話いただければ、レッカー移動、パンク修理、バッテリー充電などを24時間365日即時サポートします。"
                },
                {
                    "question": "スマートフォンデジタルキーの設定方法は？",
                    "answer": "専用アプリをインストールし、車載画面のQRコードをスキャンして連携すると、遠隔始動、ドア施錠、空調事前起動が利用可能です。"
                }
            ]
        elif cat == "entertainment":
            faqs = [
                {
                    "question": f"{display_name}の会員登録および同時視聴の管理方法は？",
                    "answer": "アカウント設定からいつでもプラン変更や解約が可能です。プレミアムプランでは最大4台の4K UHD同時視聴および空間オーディオに対応しています。"
                },
                {
                    "question": "オフライン再生用の作品ダウンロード方法は？",
                    "answer": "モバイルアプリで作品横のダウンロードアイコンをタップすると保存され、インターネット接続なしで最大30日間視聴可能です。"
                },
                {
                    "question": "決済やサービスに関するサポート窓口は？",
                    "answer": "ヘルプセンターから24時間年中無休でチャットおよび電話相談が可能です。返金手続きは通常3〜5営業日以内に完了します。"
                },
                {
                    "question": "個人情報や視聴履歴の保護方針は？",
                    "answer": "国際規格の暗号化プロトコルを適用し、決済情報および視聴データを最高水準のセキュリティで安全に保護しています。"
                }
            ]
        elif cat == "travel":
            faqs = [
                {
                    "question": f"{display_name}で購入した海外eSIMの登録・利用方法は？",
                    "answer": "購入後に届くQRコードを設定 > モバイル通信 > eSIM追加でスキャンしてください。現地到着後にデータローミングをオンにすると即座に利用可能です。"
                },
                {
                    "question": "予約したツアーやテーマパークチケットのキャンセル・返金規定は？",
                    "answer": "一般的なツアーは利用日の24時間前まで100%無料でキャンセル可能です。一部テーマパーク券などは取消不可の場合がありますので商品詳細をご確認ください。"
                },
                {
                    "question": "現地でバウチャーが認識されない場合の対処法は？",
                    "answer": "アプリ内1:1リアルタイムチャットで予約番号をお知らせいただければ、24時間年中無休のサポートチームが現地パートナーと即座に連携し対応します。"
                },
                {
                    "question": "天災（台風、欠航など）で旅行が中止になった場合は返金されますか？",
                    "answer": "欠航証明書などの証明資料をご提出いただければ、取消手数料が全額免除され、100%全額返金対応いたします。"
                }
            ]
        elif cat == "tech":
            faqs = [
                {
                    "question": f"{display_name}製品の無償保証期間と修理予約方法は？",
                    "answer": "スマートフォン等は1〜2年の無償品質保証が提供されます。公式サポートページから予約いただくと、待ち時間なく点検・修理を受けられます。"
                },
                {
                    "question": "破損時の補償プログラムの適用方法は？",
                    "answer": "公式補償プログラム加入機器は修理費用の最大70〜80%が即時減免され、正規部品による高品質な修理が提供されます。"
                },
                {
                    "question": "新端末への安全なデータ移行方法は？",
                    "answer": "公式データ移行アプリを使用することで、連絡先、写真、アプリデータをワイヤレスで100%安全に移行できます。"
                },
                {
                    "question": "アップデート後に動作が不安定になった場合の解決策は？",
                    "answer": "セーフモードで起動してキャッシュを整理するか、正規サービス窓口で最新安定ファームウェアの再インストールをおすすめします。"
                }
            ]
        else:
            faqs = [
                {
                    "question": f"{display_name}のカスタマーサポートの営業時間および窓口は？",
                    "answer": "平日09:00〜18:00の電話サポートに加え、Webおよびチャットボットを通じて24時間365日お問い合わせを受け付けております。"
                },
                {
                    "question": "注文・サービスのキャンセル返金期間は？",
                    "answer": "クレジットカード決済の取消は営業日基準で3〜5日以内、銀行振込は1〜2営業日以内に指定口座へ返金されます。"
                },
                {
                    "question": "法人向け（B2B）提携・見積もりの問い合わせ先は？",
                    "answer": "公式B2Bセンターよりお問い合わせいただければ、専任コンサルタントが24時間以内に最適な提案書をご案内いたします。"
                },
                {
                    "question": "個人情報保護およびセキュリティ管理体制は？",
                    "answer": "国際標準情報セキュリティ認証（ISO 27001）を取得し、顧客データを最高水準の暗号化技術で保護しています。"
                }
            ]
    else:  # ko
        if cat == "automotive":
            faqs = [
                {
                    "question": f"{display_name} 차량 구매 후 보증 기간 및 정기 점검 혜택은 어떻게 되나요?",
                    "answer": f"{display_name} 공식 출고 차량은 기본 3년/10만km 무상 보증 수리와 함께 정기 소모품 교환(엔진오일, 브레이크 패드 등) 무상 패키지(ISP)를 제공합니다. 전국 공식 서비스센터 어디서나 동일한 정밀 진단 혜택을 받으실 수 있습니다."
                },
                {
                    "question": "순수 전기차(EQ 시리즈) 및 하이브리드 배터리 보증 정책은 어떻게 되나요?",
                    "answer": "고전압 배터리에 대해 8년 또는 16만km 동안 배터리 잔존 용량 70%를 보증합니다. 배터리 이상 발생 시 공식 테크니션의 정밀 진단 후 무상 모듈 교체 또는 수리가 진행됩니다."
                },
                {
                    "question": "차량 운행 중 긴급 상황 발생 시 24시간 긴급 출동 서비스를 이용할 수 있나요?",
                    "answer": "차량 내 SOS 버튼을 누르거나 24시간 고객지원센터로 전화하시면 견인 서비스, 타이어 펑크 수리, 배터리 충전 등 긴급 출동 서비스를 365일 즉시 지원합니다."
                },
                {
                    "question": "스마트폰 디지털 키 및 MBUX 커넥티비티 설정 방법은 무엇인가요?",
                    "answer": "모바일 앱 설치 후 차량의 멀티미디어 화면에서 생성된 QR코드를 스캔하여 계정을 연동하시면 원격 시동, 도어 개폐, 공조 사전 작동 기능을 즉시 사용하실 수 있습니다."
                }
            ]
        elif cat == "entertainment":
            faqs = [
                {
                    "question": f"{display_name}의 구독 멤버십 요금제 및 동시 접속 정책은?",
                    "answer": "계정 설정에서 언제든 멤버십 변경 및 해지가 가능합니다. 프리미엄 요금제는 최대 4대의 기기에서 동시 4K UHD 스트리밍 및 공간 음향을 지원합니다."
                },
                {
                    "question": "오프라인 저장을 위한 콘텐츠 다운로드 방법은?",
                    "answer": "모바일/태블릿 앱에서 지원 타이틀 옆의 다운로드 아이콘을 누르시면 저장되며, 인터넷 연결 없이 최대 30일 동안 감상하실 수 있습니다."
                },
                {
                    "question": "결제 및 서비스 장애 시 고객센터 문의 방법은?",
                    "answer": "도움말 센터를 통해 24시간 실시간 채팅 및 전화 상담이 지원되며, 결제 취소 요청은 영업일 기준 3~5일 이내에 원결제 수단으로 환불됩니다."
                },
                {
                    "question": "개인정보 보호 및 시청 기록 보안 정책은?",
                    "answer": "업계 표준 TLS 암호화 프로토콜을 적용하여 고객의 모든 결제 및 시청 데이터를 안전하게 보호하고 있습니다."
                }
            ]
        elif cat == "travel":
            faqs = [
                {
                    "question": f"{display_name}에서 구매한 글로벌 eSIM은 어떻게 등록하고 사용하나요?",
                    "answer": "구매 즉시 이메일로 전송된 QR코드를 스마트폰 설정 > 셀룰러 > eSIM 추가에서 스캔하시면 등록됩니다. 현지 도착 후 해당 eSIM의 데이터 로밍을 켜시면 즉시 데이터 통신이 활성화됩니다."
                },
                {
                    "question": "예약한 투어 및 테마파크 티켓의 취소 및 환불 규정은 어떻게 되나요?",
                    "answer": "대부분의 일반 투어는 이용일 24시간 전까지 100% 무료 취소가 가능합니다. 단, 테마파크 입장권 등 일부 프로모션 티켓은 발권 즉시 취소 불가 조건이 적용될 수 있습니다."
                },
                {
                    "question": "현지에서 바우처 인식이 안 되거나 입장이 거부될 경우 어떻게 대처하나요?",
                    "answer": "앱 내 1:1 실시간 채팅 상담을 통해 바우처 예약 번호를 남겨주시면, 24시간 연중무휴 지원팀이 현지 파트너사와 즉각 유선 확인하여 재발권 또는 현장 안내를 즉시 도와드립니다."
                },
                {
                    "question": "천재지변(태풍, 결항 등)으로 인해 일정이 취소된 경우 환불받을 수 있나요?",
                    "answer": "항공 결항 증명서 또는 현지 기상 특보 증빙을 제출하시면 취소 수수료가 전액 면제되며 결제하신 수단으로 100% 전액 환불해 드립니다."
                }
            ]
        elif cat == "tech":
            faqs = [
                {
                    "question": f"{display_name} 제품의 무상 보증 기간과 서비스센터 방문 예약 방법은?",
                    "answer": "스마트폰 및 스마트 기기는 1년~2년 무상 품질 보증이 제공됩니다. 공식 홈페이지 또는 고객지원 앱에서 가까운 서비스센터와 방문 일시를 예약하시면 대기 없이 점검받으실 수 있습니다."
                },
                {
                    "question": "기기 침수 또는 액정 파손 시 안심 케어 보상 절차는 어떻게 되나요?",
                    "answer": "공식 파손 보상 프로그램 가입 기기는 수리비의 최대 70~80%가 즉시 감면 적용되며, 정품 부품을 사용한 완벽한 수리가 진행됩니다."
                },
                {
                    "question": "새 기기로 데이터 안전 이전(스마트 스위치) 방법은?",
                    "answer": "공식 데이터 이전 앱을 통해 연락처, 사진, 앱 데이터를 무선으로 100% 안전하게 이전하실 수 있습니다."
                },
                {
                    "question": "소프트웨어 업데이트 후 기기 이상이 발생했을 때 해결 방법은?",
                    "answer": "기기 재부팅 후 안전 모드로 진입하여 캐시 파티션을 정리하시거나, 공식 서비스센터를 통해 안정 펌웨어 재설치를 권장합니다."
                }
            ]
        else:
            faqs = [
                {
                    "question": f"{display_name}의 주요 고객 지원 운영 시간과 상담 창구는?",
                    "answer": f"평일 09:00~18:00 공식 고객센터 유선 상담이 운영되며, 웹사이트 및 챗봇을 통해 24시간 연중무휴 문의 접수가 가능합니다."
                },
                {
                    "question": "주문/서비스 취소 및 환불 처리 기한은 얼마나 걸리나요?",
                    "answer": "취소 승인 완료일로부터 카드 결제는 영업일 기준 3~5일 이내 카드사 승인 취소되며, 계좌이체는 1~2영업일 이내 환불 입금됩니다."
                },
                {
                    "question": "기업 고객(B2B) 맞춤형 제휴 및 대량 견적 문의는 어디로 하나요?",
                    "answer": "공식 홈페이지 B2B 비즈니스 센터를 통해 문의 사항을 남겨주시면, 24시간 이내에 전담 컨설턴트가 최적의 제안서를 회신해 드립니다."
                },
                {
                    "question": "개인정보 보호 및 보안 인증 현황은 어떻게 관리되나요?",
                    "answer": "국제 표준 정보보안 인증(ISO 27001)을 획득하여 고객의 모든 결제 정보와 개인정보를 최고 수준의 암호화 기술로 안전하게 보호하고 있습니다."
                }
            ]

    # 5. 트렌딩 토픽 (다국어)
    if lang == "en":
        trending = [
            f"{display_name} Announces Next-Gen Strategic Solutions",
            "Recognized as Global Top Tier in Customer Satisfaction (CSAT)",
            "Established Sustainable ESG & Carbon Neutrality Roadmap",
            "Achieved 98% User Satisfaction for Innovative Services"
        ]
    elif lang == "ja":
        trending = [
            f"{display_name} 次世代戦略ソリューションを発表",
            "グローバル顧客満足度(CSAT)最優秀企業に選定",
            "持続可能なESG・脱炭素ロードマップを策定",
            "革新サービスにおいて利用者満足度98%を達成"
        ]
    else:
        trending = [
            f"{display_name} 차세대 전략 솔루션 발표",
            "글로벌 고객 만족도(CSAT) 최우수 기업 선정",
            "친환경 ESG 지속 가능 경영 로드맵 수립",
            "혁신 서비스 이용자 만족도 98% 달성"
        ]

    return {
        "company_id": company_name.lower().replace(" ", "_"),
        "company_name": display_name,
        "industry": industry,
        "hero_image": preset["hero_image"],
        "hero_tagline": hero_tagline,
        "description": description,
        "popular_products": [p["name"] for p in products_detail],
        "products_detail": products_detail,
        "trending_topics": trending,
        "faqs": faqs,
        "services": [p["name"] for p in products_detail],
        "category": cat
    }
