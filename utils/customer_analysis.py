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
고객 분석 모듈
고객 프로필 분석, 유사 케이스 찾기, 시각화, 가이드라인 생성 등의 기능을 제공합니다.
"""

import json
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

from utils.history_handler import load_simulation_histories_local
from llm_client import run_llm
from lang_pack import LANG

# 시각화 함수들은 visualization.py에서 제공됩니다.


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
        import re
        
        # JSON 추출 (더 강력한 방법)
        if "```" in analysis_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
            if json_match:
                analysis_text = json_match.group(1)
            else:
                analysis_text = re.sub(r'```(?:json)?\s*', '', analysis_text)
                analysis_text = re.sub(r'\s*```', '', analysis_text)
        
        # JSON 객체 찾기
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            analysis_text = json_match.group(0)
        
        analysis_text = analysis_text.strip()
        
        # JSON 파싱 시도 (오류 처리 강화)
        try:
            analysis_data = json.loads(analysis_text)
        except json.JSONDecodeError as json_err:
            # 파싱 실패 시 기본값 반환
            print(f"고객 분석 JSON 파싱 오류: {json_err}")
            analysis_data = {
                "gender": "unknown",
                "sentiment_score": 50,
                "communication_style": "unknown",
                "urgency_level": "medium",
                "predicted_customer_type": "normal",
                "cultural_hints": "unknown",
                "key_concerns": [],
                "tone_analysis": f"Analysis error: {str(json_err)}"
            }
        
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


# 시각화 함수들은 visualization.py에서 제공됩니다.
# (중복 방지를 위해 여기서는 제거)
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

    # 언어별 프롬프트 생성
    if current_lang_key == "ko":
        guideline_prompt = f"""
당신은 과거 성공 사례를 분석하여 가이드를 제공하는 AI 고객 지원 슈퍼바이저입니다.

다음 유사한 과거 사례와 그들의 성공적인 해결 전략을 바탕으로 현재 고객 문의를 처리하기 위한 실행 가능한 가이드라인을 제공하세요.

현재 고객 문의:
{customer_query}

현재 고객 프로필:
- 성별: {customer_profile.get('gender', 'unknown')}
- 감정 점수: {customer_profile.get('sentiment_score', 50)}/100
- 커뮤니케이션 스타일: {customer_profile.get('communication_style', 'unknown')}
- 긴급도: {customer_profile.get('urgency_level', 'medium')}
- 예측 유형: {customer_profile.get('predicted_customer_type', 'normal')}

유사한 과거 사례 (성공적인 해결):
{past_cases_text}

다음 내용을 포함한 간결한 가이드라인을 한국어로 제공하세요:
1. 유사한 과거 사례에서 잘 작동한 것 식별
2. 성공적인 패턴을 기반으로 한 구체적인 접근 방법 제안
3. 과거 경험을 바탕으로 한 잠재적 함정 경고
4. 높은 고객 만족도로 이어진 응답 전략 권장

가이드라인 (한국어로):
"""
    elif current_lang_key == "en":
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

Provide a concise guideline in English that:
1. Identifies what worked well in similar past cases
2. Suggests specific approaches based on successful patterns
3. Warns about potential pitfalls based on past experiences
4. Recommends response strategies that led to high customer satisfaction

Guideline (in English):
"""
    else:  # ja
        guideline_prompt = f"""
あなたは過去の成功事例を分析してガイドを提供するAIカスタマーサポートスーパーバイザーです。

以下の類似した過去の事例とその成功した解決戦略に基づいて、現在の顧客問い合わせを処理するための実行可能なガイドラインを提供してください。

現在の顧客問い合わせ:
{customer_query}

現在の顧客プロファイル:
- 性別: {customer_profile.get('gender', 'unknown')}
- 感情スコア: {customer_profile.get('sentiment_score', 50)}/100
- コミュニケーションスタイル: {customer_profile.get('communication_style', 'unknown')}
- 緊急度: {customer_profile.get('urgency_level', 'medium')}
- 予測タイプ: {customer_profile.get('predicted_customer_type', 'normal')}

類似した過去の事例（成功した解決）:
{past_cases_text}

以下の内容を含む簡潔なガイドラインを日本語で提供してください:
1. 類似した過去の事例でうまくいったことを特定
2. 成功したパターンに基づく具体的なアプローチ方法の提案
3. 過去の経験に基づく潜在的な落とし穴の警告
4. 高い顧客満足度につながった応答戦略の推奨

ガイドライン（日本語で）:
"""

    if not st.session_state.is_llm_ready:
        return ""

    try:
        guideline = run_llm(guideline_prompt).strip()
        return guideline
    except Exception as e:
        error_msg = {
            "ko": f"가이드라인 생성 오류: {str(e)}",
            "en": f"Guideline generation error: {str(e)}",
            "ja": f"ガイドライン生成エラー: {str(e)}"
        }.get(current_lang_key, f"Error: {str(e)}")
        return error_msg


def generate_initial_advice(customer_query, customer_type_display, customer_email, customer_phone, current_lang_key,
                             customer_attachment_file):
    """Supervisor 가이드라인과 초안을 생성하는 함수 (저장된 데이터 활용)"""
    # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
    try:
        detected_lang = detect_text_language(customer_query)
    except Exception as e:
        print(f"Language detection failed in generate_initial_advice: {e}")
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

    # 언어별 contact_info_block
    if lang_key_to_use == "ko":
        contact_info_text = "고객 연락처 정보 (참고용, 응대 초안에는 사용하지 마세요!)"
    elif lang_key_to_use == "en":
        contact_info_text = "Customer contact info for reference (DO NOT use these in your reply draft!)"
    else:  # ja
        contact_info_text = "顧客連絡先情報（参考用、対応草案には使用しないでください！）"
    
    contact_info_block = ""
    if customer_email or customer_phone:
        email_label = {"ko": "이메일", "en": "Email", "ja": "メール"}[lang_key_to_use]
        phone_label = {"ko": "전화번호", "en": "Phone", "ja": "電話番号"}[lang_key_to_use]
        na_text = {"ko": "없음", "en": "N/A", "ja": "なし"}[lang_key_to_use]
        contact_info_block = (
            f"\n\n[{contact_info_text}]\n"
            f"- {email_label}: {customer_email or na_text}\n"
            f"- {phone_label}: {customer_phone or na_text}"
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

    # 언어별 고객 프로필 정보
    gender_display = customer_profile.get('gender', 'unknown')
    if lang_key_to_use == "ko":
        profile_block = f"""
[고객 프로필 분석]
- 성별: {gender_display}
- 감정 점수: {customer_profile.get('sentiment_score', 50)}/100
- 커뮤니케이션 스타일: {customer_profile.get('communication_style', 'unknown')}
- 긴급도: {customer_profile.get('urgency_level', 'medium')}
- 예측 유형: {customer_profile.get('predicted_customer_type', 'normal')}
- 주요 관심사: {', '.join(customer_profile.get('key_concerns', []))}
- 톤: {customer_profile.get('tone_analysis', 'unknown')}
"""
    elif lang_key_to_use == "en":
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
    else:  # ja
        profile_block = f"""
[顧客プロファイル分析]
- 性別: {gender_display}
- 感情スコア: {customer_profile.get('sentiment_score', 50)}/100
- コミュニケーションスタイル: {customer_profile.get('communication_style', 'unknown')}
- 緊急度: {customer_profile.get('urgency_level', 'medium')}
- 予測タイプ: {customer_profile.get('predicted_customer_type', 'normal')}
- 主要な関心事: {', '.join(customer_profile.get('key_concerns', []))}
- トーン: {customer_profile.get('tone_analysis', 'unknown')}
"""

    # 언어별 과거 케이스 기반 가이드라인 블록
    past_cases_block = ""
    if past_cases_guideline:
        if lang_key_to_use == "ko":
            past_cases_block = f"""
[유사한 과거 {len(similar_cases)}개 사례 기반 가이드라인]
{past_cases_guideline}
"""
        elif lang_key_to_use == "en":
            past_cases_block = f"""
[Guidelines Based on {len(similar_cases)} Similar Past Cases]
{past_cases_guideline}
"""
        else:  # ja
            past_cases_block = f"""
[類似した過去{len(similar_cases)}件の事例に基づくガイドライン]
{past_cases_guideline}
"""
    elif similar_cases:
        if lang_key_to_use == "ko":
            past_cases_block = f"""
[참고: 유사한 과거 사례 {len(similar_cases)}개를 찾았지만 상세 가이드라인을 생성할 수 없습니다.
패턴을 확인하기 위해 과거 사례를 수동으로 검토하는 것을 고려하세요.]
"""
        elif lang_key_to_use == "en":
            past_cases_block = f"""
[Note: Found {len(similar_cases)} similar past cases, but unable to generate detailed guidelines.
Consider reviewing past cases manually for patterns.]
"""
        else:  # ja
            past_cases_block = f"""
[注: 類似した過去の事例{len(similar_cases)}件が見つかりましたが、詳細なガイドラインを生成できませんでした。
パターンを確認するために、過去の事例を手動で確認することを検討してください。]
"""

    # 언어별 프롬프트 생성 (각 언어로 명확하게 작성)
    if lang_key_to_use == "ko":
        initial_prompt = f"""
🔴 🔴 🔴 극도로 중요 🔴 🔴 🔴
당신의 모든 응답(가이드라인과 초안 포함)은 반드시 100% 한국어로 작성되어야 합니다.
영어나 일본어를 절대 사용하지 마세요. 모든 텍스트는 한국어여야 합니다.
🔴 🔴 🔴 극도로 중요 🔴 🔴 🔴

당신은 AI 고객 지원 슈퍼바이저입니다. 다음 고객 문의를 분석하여 제공하세요:
고객 유형: **{st.session_state.customer_type_sim_select}**

1) 상담원을 위한 상세한 **응대 가이드라인** (단계별, 반드시 한국어로)
2) **전송 가능한 응대 초안** (반드시 한국어로)

[응답 형식 - 반드시 이 형식으로 작성하세요]
### {L['simulation_advice_header']}

(여기에 한국어로 가이드라인 작성)

### {L['simulation_draft_header']}

(여기에 한국어로 초안 작성)

[중요 가이드라인 규칙]
1. **초기 정보 수집 (요청 3):** 가이드라인의 첫 번째 단계는 문제 해결을 시도하기 전에 필수적인 초기 진단 정보(예: 기기 호환성, 현지 상태/위치, 주문 번호)를 요청해야 합니다.
2. **어려운 고객에 대한 공감 (요청 5):** 고객 유형이 '어려운 고객' 또는 '매우 불만족 고객'인 경우, 정책(예: 환불 불가)을 강제해야 하더라도 가이드라인은 극도의 정중함, 공감, 사과를 강조해야 합니다.
3. **24-48시간 후속 조치 (요청 6):** 문제를 즉시 해결할 수 없거나 현지 파트너/슈퍼바이저의 확인이 필요한 경우, 가이드라인은 다음 절차를 명시해야 합니다:
   - 문제를 인정합니다.
   - 고객에게 24시간 또는 48시간 내에 명확한 답변을 받을 것임을 알립니다.
   - 후속 연락을 위해 고객의 이메일 또는 전화번호를 요청합니다. (제공된 연락처 정보가 있으면 사용)
4. **과거 사례 학습:** 과거 사례 가이드라인이 제공된 경우, 해당 사례의 성공적인 전략을 권장사항에 통합하세요.

⚠️ 언어 요구사항: 모든 텍스트(가이드라인, 초안, 설명 등)는 반드시 한국어로만 작성하세요. 영어나 일본어를 사용하면 안 됩니다.

고객 문의:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""
    elif lang_key_to_use == "en":
        initial_prompt = f"""
🔴 🔴 🔴 EXTREMELY IMPORTANT 🔴 🔴 🔴
ALL your responses (including guidelines and draft) MUST be written 100% in English.
Do NOT use Korean or Japanese. All text must be in English.
🔴 🔴 🔴 EXTREMELY IMPORTANT 🔴 🔴 🔴

You are an AI Customer Support Supervisor. Your role is to analyze the following customer inquiry
from a **{st.session_state.customer_type_sim_select}** and provide:

1) A detailed **response guideline for the human agent** (step-by-step, must be in English).
2) A **ready-to-send draft reply** (must be in English).

[RESPONSE FORMAT - You MUST write in this format]
### {L['simulation_advice_header']}

(Write guidelines here in English)

### {L['simulation_draft_header']}

(Write draft here in English)

[CRITICAL GUIDELINE RULES]
1. **Initial Information Collection (Req 3):** The first step in the guideline MUST be to request the necessary initial diagnostic information (e.g., device compatibility, local status/location, order number) BEFORE attempting to troubleshoot or solve the problem.
2. **Empathy for Difficult Customers (Req 5):** If the customer type is 'Difficult Customer' or 'Highly Dissatisfied Customer', the guideline MUST emphasize extreme politeness, empathy, and apologies, even if the policy (e.g., no refund) must be enforced.
3. **24-48 Hour Follow-up (Req 6):** If the issue cannot be solved immediately or requires confirmation from a local partner/supervisor, the guideline MUST state the procedure:
   - Acknowledge the issue.
   - Inform the customer they will receive a definite answer within 24 or 48 hours.
   - Request the customer's email or phone number for follow-up contact. (Use provided contact info if available)
4. **Past Cases Learning:** If past cases guidelines are provided, incorporate successful strategies from those cases into your recommendations.

⚠️ LANGUAGE REQUIREMENT: All text (guidelines, draft, descriptions, etc.) MUST be written ONLY in English. Do NOT use Korean or Japanese.

Customer Inquiry:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""
    else:  # ja
        initial_prompt = f"""
すべてのテキスト（ガイドラインと草案）を必ず日本語で作成してください。

あなたはAIカスタマーサポートスーパーバイザーです。以下の顧客問い合わせを分析し、提供してください：
顧客タイプ: **{st.session_state.customer_type_sim_select}**

1) 人間のエージェントのための詳細な**対応ガイドライン**（ステップバイステップ）
2) **送信可能な対応草案**（日本語で）

[形式]
- 以下のマークダウンヘッダーを正確に使用してください：
  - "### {L['simulation_advice_header']}"
  - "### {L['simulation_draft_header']}"

[重要なガイドライン規則]
1. **初期情報収集（要件3）：** ガイドラインの最初のステップは、問題のトラブルシューティングや解決を試みる前に、必要な初期診断情報（例：デバイスの互換性、現地の状態/場所、注文番号）を要求することです。
2. **困難な顧客への共感（要件5）：** 顧客タイプが「困難な顧客」または「非常に不満足な顧客」の場合、ポリシー（例：返金不可）を強制する必要がある場合でも、ガイドラインは極度の丁寧さ、共感、謝罪を強調する必要があります。
3. **24-48時間のフォローアップ（要件6）：** 問題を即座に解決できない場合、または現地パートナー/スーパーバイザーの確認が必要な場合、ガイドラインは次の手順を記載する必要があります：
   - 問題を認識します。
   - 顧客に24時間または48時間以内に明確な回答を受けることを通知します。
   - フォローアップ連絡のために顧客のメールまたは電話番号を要求します。（提供された連絡先情報がある場合は使用）
4. **過去の事例学習：** 過去の事例ガイドラインが提供されている場合、それらの事例の成功戦略を推奨事項に組み込んでください。

顧客問い合わせ:
{customer_query}
{contact_info_block}
{attachment_block}
{profile_block}
{past_cases_block}
"""
    if not st.session_state.is_llm_ready:
        # 언어별 Mock 텍스트
        if lang_key_to_use == "ko":
            mock_text = (
                f"### {L['simulation_advice_header']}\n\n"
                f"- (Mock) {st.session_state.customer_type_sim_select} 유형 고객 응대 가이드입니다. (요청 3, 5, 6 반영)\n\n"
                f"### {L['simulation_draft_header']}\n\n"
                f"(Mock) 에이전트 응대 초안이 여기에 들어갑니다.\n\n"
            )
        elif lang_key_to_use == "en":
            mock_text = (
                f"### {L['simulation_advice_header']}\n\n"
                f"- (Mock) Response guide for {st.session_state.customer_type_sim_select} type customer. (Reflects Req 3, 5, 6)\n\n"
                f"### {L['simulation_draft_header']}\n\n"
                f"(Mock) Agent response draft will appear here.\n\n"
            )
        else:  # ja
            mock_text = (
                f"### {L['simulation_advice_header']}\n\n"
                f"- (Mock) {st.session_state.customer_type_sim_select}タイプの顧客対応ガイドです。（要件3、5、6を反映）\n\n"
                f"### {L['simulation_draft_header']}\n\n"
                f"(Mock) エージェント対応草案がここに表示されます。\n\n"
            )
        return mock_text
    else:
        with st.spinner(L["response_generating"]):
            try:
                return run_llm(initial_prompt)
            except Exception as e:
                error_msg = {
                    "ko": f"AI 조언 생성 중 오류 발생: {e}",
                    "en": f"Error occurred while generating AI advice: {e}",
                    "ja": f"AIアドバイス生成中にエラーが発生しました: {e}"
                }.get(lang_key_to_use, f"Error: {e}")
                st.error(error_msg)
                return f"❌ {error_msg}"


# 별칭 추가 (하위 호환성)
_generate_initial_advice = generate_initial_advice

