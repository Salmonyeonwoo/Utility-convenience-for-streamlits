import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. 페이지 기본 설정 및 가상 데이터 생성 ---
st.set_page_config(page_title="프랜차이즈 VoC 스마트 관리", layout="wide")

# 데모를 위한 가상 데이터 생성 함수
def get_dummy_data():
    """발표용 가상 데이터를 생성합니다."""
    branches = ['부산 서면점', '부산 해운대점', '부산대점', '광안리점']
    categories = ['서비스 태도', '맛/품질', '매장 청결', '주문 실수']
    comments_bad = ['직원이 불친절해요', '음식이 너무 식어서 나왔어요', '테이블이 끈적거립니다', '주문한 거랑 다른 게 나왔어요']
    comments_good = ['너무 친절하셔서 감동', '역시 맛있습니다', '매장이 깨끗해요', '빠른 응대 감사합니다']

    data = []
    # 지난 7일간의 랜덤 데이터 생성
    for i in range(50):
        date = datetime.now() - timedelta(days=np.random.randint(0, 7))
        branch = np.random.choice(branches)
        rating = np.random.randint(1, 6)  # 1~5점

        if rating <= 2:
            category = np.random.choice(categories)
            comment = np.random.choice(comments_bad)
            status = "미해결"
        else:
            category = "기타"
            comment = np.random.choice(comments_good)
            status = "완료"

        data.append([date.date(), branch, rating, category, comment, status])

    df = pd.DataFrame(data, columns=['날짜', '지점', '평점', '불만유형', '고객 코멘트', '처리상태'])
    return df

# 데이터 로드
df = get_dummy_data()

# --- 2. 사이드바 (가맹점주용 필터) ---
st.sidebar.header("🏪 가맹점 관리 모드")
st.sidebar.write("점주님, 환영합니다.")
selected_branch = st.sidebar.selectbox("관리할 지점을 선택하세요:", df['지점'].unique())
filtered_df = df[df['지점'] == selected_branch]

# --- 3. 메인 화면 구성 ---
st.title(f"📢 [{selected_branch}] 스마트 VoC 관리 시스템")
st.markdown("---")

# 3-1. 핵심 지표 (KPI)
col1, col2, col3 = st.columns(3)
with col1:
    avg_rating = filtered_df['평점'].mean()
    st.metric(label="이번 주 평균 평점", value=f"{avg_rating:.1f}점", delta=f"{avg_rating - 3.0:.1f}")
with col2:
    total_voc = len(filtered_df)
    st.metric(label="총 접수된 고객 의견", value=f"{total_voc}건")
with col3:
    urgent_df = filtered_df[filtered_df['평점'] <= 2].sort_values(by='날짜', ascending=False)
    urgent_cases = len(urgent_df)
    st.metric(label="🚨 긴급 조치 필요 고객", value=f"{urgent_cases}명", delta_color="inverse")

st.markdown("---")

# 3-2. 데이터 시각화
col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    st.subheader("📊 불만 유형별 분석")
    bad_counts = urgent_df['불만유형'].value_counts()
    if not bad_counts.empty:
        st.bar_chart(bad_counts)
    else:
        st.write("분석할 불만 데이터가 없습니다.")

with col_chart2:
    st.subheader("📈 최근 평점 추이")
    daily_rating = filtered_df.groupby('날짜')['평점'].mean()
    st.line_chart(daily_rating)

st.markdown("---")

# ==============================================================================
# [NEW] 4. AI 기반 맞춤형 교육 추천 (스마트 코칭 섹션) - 인터넷 이미지 주소 사용
# ==============================================================================
st.header("🎓 AI 기반 맞춤형 교육 추천 (스마트 코칭)")
st.write("데이터 분석 결과, 현재 우리 매장에 가장 필요한 직원 교육을 추천합니다.")

if urgent_cases > 0:
    top_issue = bad_counts.idxmax()
    st.error(f"💡 분석 결과, [{selected_branch}]은 현재 **'{top_issue}'** 관련 클레임이 가장 많습니다.")

    col_edu1, col_edu2 = st.columns([1, 1])

    with col_edu1:
        st.subheader("📹 추천 영상 교육 자료")
        # [수정됨] 다시 인터넷 임시 이미지 주소(placehold.co) 사용
        if top_issue == '서비스 태도':
            st.image(r"C:\Users\Admin\Downloads\Basic_Service_Manner.jpg", caption="▶️ [필수 시청] 고객 감동을 부르는 기본 응대 매뉴얼 (3분)")
        elif top_issue == '매장 청결':
            st.image(r"C:\Users\Admin\Downloads\table_cleaning.png", caption="▶️ [필수 시청] 피크타임 테이블 정리 표준 요령 (2분)")
        elif top_issue == '주문 실수':
            st.image(r"C:\Users\Admin\Downloads\pos_training.png", caption="▶️ [필수 시청] 주문 실수 '0'를 위한 포스기 조작법 (4분)")
        else:
            st.image(r"C:\Users\Admin\Downloads\General_Guide.jpg", caption="▶️ 기본 매장 운영 가이드")
        st.info("👆 클릭하면 본사 교육 서버의 영상으로 연결됩니다. (프로토타입)")

    with col_edu2:
        st.subheader("📸 현장 부착용 사진 매뉴얼")
        st.write("직원들이 수시로 볼 수 있도록 출력하여 비치해주세요.")
        with st.expander("🖨️ 사진 매뉴얼 보기 및 인쇄 (클릭)"):
            # [수정됨] placehold.co 서비스 사용
            if top_issue == '매장 청결':
                st.image(r"C:\Users\Admin\Downloads\Other_cleaning.jpg", caption="[청결 표준] 테이블 정리 3단계")
            elif top_issue == '서비스 태도':
                st.image(r"C:\Users\Admin\Downloads\Other_service_manner.jpg", caption="[서비스 표준] 고객 응대 3대 요소")
            else:
                 st.write("해당 유형의 표준 사진 매뉴얼을 준비 중입니다.")
        st.success("✅ 점주님! 이 자료를 이번 주 스태프 미팅 때 활용해 보세요.")

else:
    st.success("🎉 축하합니다! 현재 긴급한 고객 불만이 없습니다. 최고의 서비스를 유지하고 계시네요!")

st.markdown("---")
st.info("""
**💡 아이디어의 핵심 경쟁력 (심사위원 어필 포인트)**
1.  **데이터의 시각화:** 막연한 감이 아닌, 실제 데이터로 매장의 문제점을 진단합니다.
2.  **Actionable Insight (실행 가능한 통찰):** 단순히 문제만 보여주는 것이 아니라, **'그래서 무엇을 교육해야 하는지'** 시청각 자료로 즉시 해결책을 제시합니다.
3.  **프랜차이즈 표준화:** 본사의 우수한 교육 자료(영상/사진)를 데이터 기반으로 적재적소에 배포하여, 전 지점의 서비스 품질을 상향 평준화합니다.
""")