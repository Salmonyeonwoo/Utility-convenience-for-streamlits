# ========================================
# _pages/_content_visualization.py
# 콘텐츠 시각화 모듈
# ========================================

import streamlit as st
import numpy as np

# Plotly 시각화
try:
    import plotly.graph_objects as go
    import plotly.express as px
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False


def render_content_visualization(content, level, L):
    """콘텐츠 시각화 렌더링"""
    st.subheader("💡 콘텐츠 분석 (Plotly 시각화)")

    if IS_PLOTLY_AVAILABLE:
        _render_plotly_visualization(content, level)
    else:
        _render_text_visualization(content)


def _render_plotly_visualization(content, level):
    """Plotly를 사용한 시각화"""
    content_lines = content.split('\n')
    all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()

    # 모의 키워드 빈도 데이터
    words = ['AI', '기술혁신', '고객경험', '데이터분석', '효율성', '여행산업']
    np.random.seed(42)
    counts = np.random.randint(5, 30, size=len(words))

    difficulty_score = {
        'Beginner': 60,
        'Intermediate': 75,
        'Advanced': 90
    }.get(level, 70)

    # 키워드 빈도 차트
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

    # 감성/복잡도 추이 차트
    sections = ['도입부', '핵심1', '핵심2', '해결책', '결론']
    sentiment_scores = [
        difficulty_score - 10,
        difficulty_score + 5,
        difficulty_score,
        difficulty_score + 10,
        difficulty_score + 2
    ]

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


def _render_text_visualization(content):
    """텍스트 기반 시각화 (Plotly 없을 때)"""
    st.info("Plotly 라이브러리가 없어 시각화를 표시할 수 없습니다. 텍스트 분석 모의를 표시합니다.")
    content_lines = content.split('\n')
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
