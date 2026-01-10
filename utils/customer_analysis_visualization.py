# ========================================
# utils/customer_analysis_visualization.py
# 고객 분석 - 시각화 모듈
# ========================================

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
from lang_pack import LANG

# Plotly 시각화
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False


def visualize_customer_profile(customer_profile: Dict[str, Any], current_lang_key: str):
    """고객 프로필을 시각화"""
    if not IS_PLOTLY_AVAILABLE:
        return None

    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    sentiment_score = customer_profile.get("sentiment_score", 50)
    urgency_map = {"low": 25, "medium": 50, "high": 75}
    urgency_level = customer_profile.get("urgency_level", "medium")
    urgency_score = urgency_map.get(urgency_level.lower(), 50)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=(
            L.get("sentiment_score_label", "고객 감정 점수"),
            L.get("urgency_score_label", "긴급도 점수")
        )
    )

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

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            L.get("similarity_chart_title", "유사 케이스 유사도"),
            L.get("scores_comparison_title", "감정 및 만족도 점수 비교")
        ),
        vertical_spacing=0.15
    )

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
    fig.update_yaxes(title_text=L.get("score_label_short", "점수"), row=2, col=1)
    fig.update_yaxes(title_text=L.get("similarity_percent", "유사도 (%)"), row=1, col=1)

    return fig


def visualize_case_trends(histories: List[Dict[str, Any]], current_lang_key: str):
    """과거 성공 사례 트렌드를 시각화"""
    if not IS_PLOTLY_AVAILABLE or not histories:
        return None

    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    cases_with_summary = [
        h for h in histories
        if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
    ]

    if not cases_with_summary:
        return None

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
    """고객 특성을 시각화"""
    if not IS_PLOTLY_AVAILABLE or not summary:
        return None

    if current_lang_key not in ["ko", "en", "ja"]:
        current_lang_key = st.session_state.get("language", "ko")
        if current_lang_key not in ["ko", "en", "ja"]:
            current_lang_key = "ko"
    L = LANG.get(current_lang_key, LANG["ko"])

    characteristics = summary.get("customer_characteristics", {})
    privacy_info = summary.get("privacy_info", {})

    labels = []
    values = []

    language = characteristics.get("language", "unknown")
    if language != "unknown":
        labels.append(L.get("language_label", "언어"))
        lang_map = {"ko": "한국어", "en": "English", "ja": "日本語"}
        values.append(lang_map.get(language, language))

    if privacy_info.get("has_email"):
        labels.append(L.get("email_provided_label", "이메일 제공"))
        values.append("Yes")
    if privacy_info.get("has_phone"):
        labels.append(L.get("phone_provided_label", "전화번호 제공"))
        values.append("Yes")

    region = privacy_info.get("region_hint", characteristics.get("region", "unknown"))
    if region != "unknown":
        labels.append(L.get("region_label", "지역"))
        values.append(region)

    if not labels:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=[1] * len(labels),
        hole=0.4,
        marker_colors=px.colors.qualitative.Set3[:len(labels)]
    )])

    fig.update_layout(
        title=L.get("customer_characteristics_title", "고객 특성 분포"),
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig
