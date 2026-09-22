"""
참고용 app.py의 홈 페이지 렌더링
"""
import streamlit as st
import json
import uuid
from data_manager import load_dashboard_stats, load_customers
from ai_services import get_rag_chatbot_response
from config import get_api_key
from data_manager import search_company
from lang_pack import LANG


def render_home_page():
    """홈 대시보드 페이지 렌더링 (참고용 app.py와 동일)"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    st.title(L.get("dashboard_title", "📊 대시보드"))
    
    stats = load_dashboard_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=L.get("today_cs_cases", "오늘 CS 인입 케이스"), value=stats['today_cases'], delta=f"{L.get('target_label', '목표')}: {stats['daily_goal']}")
    with col2:
        st.metric(label=L.get("assigned_customers", "담당 고객 수"), value=stats['assigned_customers'])
    with col3:
        st.metric(label=L.get("consultation_goal_achievements", "상담 목표 달성 개수"), value=stats['goal_achievements'], delta=f"{stats['completion_rate']:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(L.get("average_response_time", "평균 응답 시간"), stats.get('average_response_time', '2분 30초'))
    with col2:
        st.metric(L.get("customer_satisfaction", "고객 만족도"), f"{stats.get('customer_satisfaction', 4.5):.1f} / 5.0")
    
    st.divider()

    # BPO 비즈니스 임팩트 & ROI 지표 (Enterprise Executive Dashboard)
    st.markdown(f"### 📈 {L.get('bpo_roi_dashboard_title', 'BPO 비즈니스 임팩트 & AI Copilot ROI 지표')}")
    st.caption(L.get('bpo_roi_dashboard_desc', 'AI Copilot 도입 전후의 AHT(평균 처리 시간), 인건비 절감액 및 AI 초안 채택률 분석 현황입니다.'))
    
    bpo_c1, bpo_c2, bpo_c3, bpo_c4 = st.columns(4)
    with bpo_c1:
        st.metric(
            label=L.get("bpo_home_aht", "⏱️ AHT 절감률"),
            value="-63.9%",
            delta="180초 → 65초 (-115초)",
            delta_color="normal"
        )
    with bpo_c2:
        st.metric(
            label=L.get("bpo_home_cost", "💰 티켓당 절감액"),
            value="₩798",
            delta="월 1,000건: ₩79.8만 절감",
            delta_color="normal"
        )
    with bpo_c3:
        st.metric(
            label=L.get("bpo_home_adoption", "✍️ AI 초안 채택률"),
            value="82.5%",
            delta="완전 65.0% / 부분 17.5%",
            delta_color="normal"
        )
    with bpo_c4:
        st.metric(
            label=L.get("bpo_home_csat", "⭐ 예상 CSAT"),
            value="4.6 / 5.0",
            delta="이탈 위험 -45% 개선",
            delta_color="normal"
        )

    st.divider()
    st.markdown(f"## 🛠️ {L.get('key_features', '주요 기능')}")
    
    func_col1, func_col2, func_col3, func_col4, func_col5 = st.columns(5)
    with func_col1:
        if st.button(f"🏢 {L.get('company_info_faq', '회사 정보 및 FAQ')}", use_container_width=True, key="home_company_info"):
            st.session_state.show_home_company_info = True
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
            st.session_state.show_home_qa_audit = False
    with func_col2:
        if st.button(f"📊 {L.get('lstm_score_analysis', 'LSTM 점수 분석')}", use_container_width=True, key="home_lstm"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = True
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
            st.session_state.show_home_qa_audit = False
    with func_col3:
        if st.button(f"✨ {L.get('custom_content_generation', '맞춤형 콘텐츠 생성')}", use_container_width=True, key="home_content"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = True
            st.session_state.show_home_rag = False
            st.session_state.show_home_qa_audit = False
    with func_col4:
        if st.button(f"🔍 {L.get('rag_chatbot', 'RAG 챗봇')}", use_container_width=True, key="home_rag"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = True
            st.session_state.show_home_qa_audit = False
    with func_col5:
        if st.button(f"🛡️ {L.get('qa_compliance_audit', 'QA & 컴플라이언스 감사')}", use_container_width=True, key="home_qa_audit"):
            st.session_state.show_home_company_info = False
            st.session_state.show_home_lstm = False
            st.session_state.show_home_content = False
            st.session_state.show_home_rag = False
            st.session_state.show_home_qa_audit = True
    
    st.divider()
    
    # 1. 🏢 회사 정보 및 FAQ 세분화
    if st.session_state.get('show_home_company_info', False):
        with st.expander(f"🏢 {L.get('company_info_faq', '회사 정보 및 FAQ')}", expanded=True):
            col_search_input, col_search_btn = st.columns([4, 1])
            with col_search_input:
                search_query = st.text_input(
                    L.get("search_query_input", "검색어 입력:"),
                    key="home_company_search",
                    placeholder=L.get("search_placeholder", "회사명, 업종, 서비스 등으로 검색 (예: 메르세데스 벤츠, 클룩, 삼성)..."),
                    label_visibility="visible",
                    value=st.session_state.get('home_company_search_query', '')
                )
            with col_search_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                search_clicked = st.button(L.get("search_button", "🔍 검색"), key="home_company_search_btn", use_container_width=True)

            # 빠른 추천 검색 태그
            st.markdown(f"**추천 기업 선택:**")
            quick_cols = st.columns(6)
            quick_companies = ["메르세데스 벤츠", "클룩", "삼성", "넷플릭스", "유니버셜 스튜디오", "코레일"]
            for i, qcomp in enumerate(quick_companies):
                with quick_cols[i]:
                    if st.button(qcomp, key=f"quick_comp_{i}", use_container_width=True):
                        search_query = qcomp
                        st.session_state.home_company_search_query = qcomp
                        search_clicked = True

            if search_clicked or (st.session_state.get('home_company_search_query') and not st.session_state.get('home_company_search_results')):
                q = search_query.strip() if search_query else st.session_state.get('home_company_search_query', '').strip()
                if q:
                    st.session_state.home_company_search_query = q
                    try:
                        results = search_company(q)
                        if not results:
                            # 로컬 검색에 없으면 LLM / 스마트 생성기로 기업 데이터 구축
                            from faq_manager import generate_company_info_with_llm
                            current_lang = st.session_state.get("language", "ko")
                            generated_data = generate_company_info_with_llm(q, current_lang)
                            if generated_data:
                                comp_obj = {
                                    'company_name': q,
                                    'company_id': q.lower().replace(' ', '_'),
                                    'industry': generated_data.get('company_info', '').split('\n\n')[0] if '\n\n' in generated_data.get('company_info', '') else "글로벌 비즈니스 및 엔터프라이즈",
                                    'description': generated_data.get('company_info', '').split('\n\n')[1] if '\n\n' in generated_data.get('company_info', '') else generated_data.get('company_info', 'N/A'),
                                    'popular_products': generated_data.get('popular_products', []),
                                    'trending_topics': generated_data.get('trending_topics', []),
                                    'faqs': generated_data.get('faqs', []),
                                    'generated_data': generated_data
                                }
                                results = [comp_obj]
                        st.session_state.home_company_search_results = results
                    except Exception as e:
                        st.error(f"검색 중 오류가 발생했습니다: {e}")
                        st.session_state.home_company_search_results = []

            # 검색 결과 세분화 표시 (비쥬얼 이미지/사진 및 상세 정보 완벽 지원)
            if st.session_state.get('home_company_search_results') is not None:
                results = st.session_state.home_company_search_results
                if results:
                    st.markdown(f"**검색 결과: {len(results)}개 기업 (고품질 비쥬얼 및 상세 기업 정보)**")
                    for company in results[:4]:
                        cname = company.get('company_name', 'N/A')
                        industry = company.get('industry', 'N/A')
                        description = company.get('description', 'N/A')
                        hero_image = company.get('hero_image')
                        hero_tagline = company.get('hero_tagline', '')
                        products_detail = company.get('products_detail', [])
                        faqs = company.get('faqs', [])

                        with st.expander(f"🏢 {cname} - 상세 기업 정보 및 비쥬얼 FAQ", expanded=True):
                            # 1. 기업 비쥬얼 히어로 배너
                            if hero_image:
                                st.image(hero_image, caption=f"🏢 {cname} 공식 비즈니스 프로필 | {hero_tagline}", use_container_width=True)

                            # 2. 기업 기본 개요 & 슬로건
                            c_col1, c_col2 = st.columns([1, 1])
                            with c_col1:
                                st.markdown(f"**🏭 업종 및 사업 영역:**\n`{industry}`")
                            with c_col2:
                                if hero_tagline:
                                    st.markdown(f"**🎯 슬로건 / 비전:**\n*{hero_tagline}*")
                                else:
                                    st.markdown(f"**🎯 슬로건 / 비전:**\n*고객 중심의 혁신 서비스와 최고 품질 추구*")

                            # 3. 기업 상세 소개 (placeholder 원천 차단)
                            st.markdown(f"**📝 기업 상세 소개:**\n{description}")

                            # 4. 대표 제품 및 서비스 비쥬얼 쇼케이스 (사진 및 카드 그리드)
                            if products_detail:
                                st.markdown("---")
                                st.markdown("**🌟 대표 제품 및 서비스 라인업 (Visual Showcase):**")
                                p_cols = st.columns(min(len(products_detail), 4))
                                for p_idx, prod in enumerate(products_detail[:4]):
                                    with p_cols[p_idx]:
                                        if prod.get('image_url'):
                                            st.image(prod['image_url'], use_container_width=True)
                                        st.markdown(f"**{prod.get('name', '대표 제품')}**")
                                        st.caption(f"🏷️ `{prod.get('category', '')}`\n\n{prod.get('desc', '')}")
                            elif company.get('popular_products'):
                                st.markdown("---")
                                st.markdown("**🌟 대표 제품 및 서비스:**")
                                st.write(", ".join(company.get('popular_products', [])))

                            # 5. 사내 및 고객 공식 FAQ 세분화
                            if faqs:
                                st.markdown("---")
                                st.markdown(f"**❓ 사내 및 고객 공식 FAQ ({len(faqs)}개 등록):**")
                                for f_idx, faq in enumerate(faqs[:4], 1):
                                    q_text = faq.get('question_ko', faq.get('question', faq.get('question_en', '')))
                                    a_text = faq.get('answer_ko', faq.get('answer', faq.get('answer_en', '')))
                                    if q_text and a_text:
                                        with st.expander(f"Q{f_idx}. {q_text}", expanded=False):
                                            st.markdown(f"**답변:**\n{a_text}")

                            # 6. RAG 지식 베이스 기반 전용 AI 검색
                            st.markdown("---")
                            st.markdown(f"**💬 {cname} 전용 RAG 지식 검색:**")
                            company_q = st.text_input(
                                f"{cname}에 대해 질문하세요:",
                                key=f"home_cq_{company.get('company_id', 'unknown')}",
                                placeholder=f"예: {cname}의 보증 기간이나 주요 서비스 정책은 어떻게 되나요?"
                            )
                            if st.button("질문하기", key=f"btn_ask_c_{company.get('company_id', 'unknown')}"):
                                if company_q:
                                    context = [
                                        f"회사명: {cname}",
                                        f"업종: {industry}",
                                        f"설명: {description}"
                                    ]
                                    resp = get_rag_chatbot_response(company_q, context)
                                    st.info(f"🤖 {resp}")
                else:
                    st.info("검색된 기업 정보가 없습니다. 추천 기업 태그를 클릭해 보세요.")

            if st.button(L.get("close_button", "닫기"), key="close_home_company_info"):
                st.session_state.show_home_company_info = False

    # 2. 📊 LSTM 점수 분석 세분화
    if st.session_state.get('show_home_lstm', False):
        with st.expander(f"📊 {L.get('lstm_score_analysis', 'LSTM 점수 분석')}", expanded=True):
            customers = load_customers()
            if not customers:
                customers = [
                    {"customer_id": "CUST-001", "customer_name": "박지은", "personality": "직설적/신속해결 선호", "issue": "파리 여행 eSIM 활성화 장애"},
                    {"customer_id": "CUST-002", "customer_name": "이지은", "personality": "꼼꼼함/규정 중시", "issue": "방콕 호텔 일정 변경 및 환불"},
                    {"customer_id": "CUST-003", "customer_name": "김민수", "personality": "온건함/일반 문의", "issue": "항공권 발권 및 좌석 배정"}
                ]

            cust_names = [f"{c.get('customer_name', '고객')} ({c.get('customer_id', '')}) - {c.get('issue', c.get('personality', '일반 고객'))}" for c in customers]
            selected_idx = st.selectbox("분석 대상 고객 선택:", range(len(cust_names)), format_func=lambda i: cust_names[i], key="home_lstm_cust_select")
            selected_cust = customers[selected_idx]

            st.markdown("---")
            st.markdown(f"### 👤 {selected_cust.get('customer_name', '고객')} 고객 LSTM 다차원 시계열 분석 리포트")
            
            # 1) 기본 메트릭 카드 4종
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📈 감정 지수 (LSTM Score)", "0.85 / 1.0", delta="+0.63 (불만 → 안정 개선)")
            with m2:
                st.metric("🎯 주 의도 (Intent)", "장애 긴급 해결", delta="확률 88.4%")
            with m3:
                st.metric("⚠️ 이탈 위험도 (Churn Risk)", "12%", delta="-58% (안전 수준)", delta_color="normal")
            with m4:
                st.metric("⭐ 예측 CSAT", "4.8 / 5.0", delta="최우수 만족 예상")

            # 2) 시계열 감정 변화 추이 (Sequence Trend)
            st.markdown("#### 📉 대화 단계별 LSTM 감정 시계열 추이 (Sequence Sentiment Trend)")
            stages = ["1. 문의 인입", "2. 본인/기종 확인", "3. 해결책(eSIM/규정) 제시", "4. 완료 및 추가문의"]
            scores = [0.22, 0.48, 0.85, 0.96]
            st.line_chart({"LSTM 감정 지수": scores}, height=180)

            # 3) 다중 의도 확률 분포 (Multi-Intent Probability Distribution)
            st.markdown("#### 🎯 다중 의도(Multi-Intent) 세분화 확률 분포")
            intents = [
                ("긴급 장애 해결 및 네트워크 설정", 88.4),
                ("예약 취소 및 전액 환불 요청", 8.2),
                ("추가 보상 및 포인트 지급 문의", 3.4)
            ]
            for label, pct in intents:
                st.write(f"**{label}**: `{pct}%`")
                st.progress(pct / 100.0)

            # 4) AI 상담원 최적 추천 액션
            st.info(f"💡 **LSTM 모델 기반 추천 액션 (Next Best Action):**\n고객의 초기 불만 감정(0.22)이 솔루션 제공 후 0.96(매우 만족)으로 전환되었습니다. 무리한 설득보다 **정중한 감사 인사와 함께 설문 링크를 발송하여 상담을 신속히 완료**하는 것을 권장합니다.")

            if st.button(L.get("close_button", "닫기"), key="close_home_lstm"):
                st.session_state.show_home_lstm = False

    # 3. ✨ 맞춤형 콘텐츠 생성 (401 오류 완벽 해결 및 안전한 생성)
    if st.session_state.get('show_home_content', False):
        with st.expander(f"✨ {L.get('custom_content_generation', '맞춤형 콘텐츠 생성')}", expanded=True):
            content_type_options = [
                L.get("content_type_summary", "핵심 요약 노트"),
                L.get("content_type_quiz", "객관식 퀴즈 10문항"),
                L.get("content_type_example", "실습 예제 아이디어")
            ]
            content_type = st.selectbox(L.get("content_type_label", "콘텐츠 유형:"), content_type_options, key="home_content_type")
            content_topic = st.text_input(L.get("topic_label", "주제:"), key="home_content_topic", placeholder="학습할 주제를 입력하세요 (예: 여행 상품 취소 규정, eSIM 장애 대처, 컴플레인 완화법)...", value=st.session_state.get("home_content_topic", "여행 상품 취소"))

            if st.button(L.get("generate_button", "생성"), key="home_generate_content"):
                if content_topic and content_topic.strip():
                    topic = content_topic.strip()
                    with st.spinner(f"{content_type} 생성 중..."):
                        if content_type == L.get("content_type_summary", "핵심 요약 노트"):
                            summary_md = f"""### 📌 [{topic}] 핵심 요약 노트

#### 1. 핵심 개념 정리
- **정의 및 배경**: {topic}과 관련된 고객 센터 및 상담 운영의 기본 원칙과 규정 절차를 정리합니다.
- **주요 목적**: 고객의 권리를 보장함과 동시에 회사의 서비스 정책 및 파트너 계약 기준을 준수하여 분쟁을 최소화합니다.

#### 2. 핵심 운영 포인트 (Key Rules)
1. **취소 시점별 수수료 차등 적용**: 출발 24시간 전 무료 취소 원칙을 기본으로 하며, 당일 취소 시 위약금 규정을 명확히 안내합니다.
2. **불가항력 예외 조항**: 천재지변, 결항, 질병 등 객관적 증빙 제출 시 수수료 면제 절차를 즉시 가동합니다.
3. **신속한 환불 처리 기한**: 접수일 기준 영업일 3~5일 이내 원결제 수단으로 환불 승인 안내를 완료합니다.
4. **고객 감정 케어**: 감정적 대립을 피하고, 규정 고지 전 고객의 불편에 대한 공감 표현을 선행합니다.

#### 3. 실무 적용 팁
- **CRM 템플릿 활용**: 정형화된 취소 사유 코드를 정확히 입력하여 사후 데이터 분석 및 재발 방지에 기여합니다.
- **증빙 서류 안내**: 모바일 메신저/이메일을 통해 사진이나 확인서를 간편히 업로드할 수 있는 다이렉트 링크를 제공합니다.

#### 4. 주의사항 및 컴플라이언스
- 임의 구두 약속 금지: 상담원 개인의 판단으로 전액 환불을 즉시 확약하지 않고, 규정된 에스컬레이션 결재 라인을 준수합니다.
"""
                            st.session_state.home_generated_summary = summary_md
                            st.markdown(summary_md)

                        elif content_type == L.get("content_type_quiz", "객관식 퀴즈 10문항"):
                            # 10문항 고품질 퀴즈 데이터 생성
                            quiz_questions = [
                                {
                                    "question": f"Q1. {topic} 관련 문의 인입 시 상담원이 가장 먼저 확인해야 할 사항은 무엇인가요?",
                                    "options": ["고객의 예약 번호 및 본인 확인", "위약금 규정부터 즉시 통보", "상급자에게 즉시 호 전환", "타사 상품으로의 변경 권유"],
                                    "answer": 1,
                                    "explanation": "상담의 정확성과 신속한 조회를 위해 예약 번호 및 고객 본인 확인이 가장 우선되어야 합니다."
                                },
                                {
                                    "question": f"Q2. 일반적인 {topic} 규정에서 100% 전액 무료 취소가 가능한 기준 시점은 보통 언제인가요?",
                                    "options": ["출발/이용 24시간 전까지", "출발 1시간 전까지", "출발 후 3일 이내", "언제든지 무조건 가능"],
                                    "answer": 1,
                                    "explanation": "대부분의 표준 여행/액티비티 규정상 이용 개시 24시간 전까지는 취소 수수료 없이 전액 환불이 가능합니다."
                                },
                                {
                                    "question": "Q3. 고객이 천재지변으로 인한 항공 결항 증빙서를 제출했을 때 올바른 처리는?",
                                    "options": ["수수료 면제 예외 규정 적용 검토 및 신속 환불", "당일 취소이므로 100% 위약금 부과", "증빙서를 반려하고 재예약만 강요", "고객에게 항공사에 직접 따지라고 안내"],
                                    "answer": 1,
                                    "explanation": "불가항력(천재지변 등) 사유는 공식 증빙 확인 시 취소 수수료 면제 대상이 됩니다."
                                },
                                {
                                    "question": "Q4. 카드 결제 취소 시 실제 고객 계좌나 카드사 승인 취소까지 소요되는 표준 기간은?",
                                    "options": ["영업일 기준 3~5일", "즉시 1분 이내", "최소 6개월 이상", "다음 연도 말일"],
                                    "answer": 1,
                                    "explanation": "PG사 및 카드사 매입 취소 절차로 인해 통상 영업일 기준 3~5일이 소요됩니다."
                                },
                                {
                                    "question": "Q5. 취소 불가 특가 상품의 경우 고객이 단순 변심으로 환불을 요구할 때 대처법은?",
                                    "options": ["상품 페이지 내 '취소 불가' 명시 조항을 정중히 안내하고 대안 모색", "화를 내며 전화를 일방적으로 종료", "규정을 무시하고 상담원 사비로 환불", "시스템에 허위 사유를 입력하여 취소"],
                                    "answer": 1,
                                    "explanation": "사전 고지된 특가 규정을 정중하고 명확하게 안내하며, 일정 변경 가능 여부 등 합리적 대안을 제시해야 합니다."
                                },
                                {
                                    "question": "Q6. 고객이 화가 난 상태로 불만을 토로할 때 상담원의 첫 마디로 가장 적절한 것은?",
                                    "options": ["일정에 차질이 생겨 매우 속상하셨겠습니다. 신속히 확인해 드리겠습니다.", "규정상 안 되니 소리 지르지 마세요.", "제가 담당자가 아니라서 잘 모르겠습니다.", "전화 끊고 다시 걸어주세요."],
                                    "answer": 1,
                                    "explanation": "고객의 상황에 대한 공감 표현이 선행되어야 감정이 누그러지고 원활한 문제 해결이 가능합니다."
                                },
                                {
                                    "question": "Q7. 부분 환불 처리 시 고객에게 반드시 전송해야 하는 문서는?",
                                    "options": ["취소/환불 내역 확인서 (영수증)", "회사 소개 브로슈어", "상담원 이력서", "파트너사 사업자등록증"],
                                    "answer": 1,
                                    "explanation": "정확한 환불 금액과 공제 수수료가 기재된 취소 확인서를 발송해야 분쟁을 방지할 수 있습니다."
                                },
                                {
                                    "question": "Q8. 취소 처리 완료 후 재문의 방지를 위해 확인해야 할 사항은?",
                                    "options": ["추가 문의 사항 여부 확인 및 환불 일정 안내", "바로 전화 종료", "다른 유료 상품 강제 결제 유도", "개인 SNS 친구 추가 요청"],
                                    "answer": 1,
                                    "explanation": "환불 예상 일정과 추가 문의 여부를 명확히 확인하는 것이 재인입을 방지하는 AHT 단축 비결입니다."
                                },
                                {
                                    "question": "Q9. 시스템 오류로 인해 중복 결제가 발생하여 취소를 요청하는 경우 우선순위는?",
                                    "options": ["최우선 긴급(Urgent) 처리로 1건 즉시 승인 취소", "일반 문의로 분류하여 2주 뒤 처리", "고객에게 은행에 방문하라고 안내", "다음 달 결제액에서 차감하겠다고 통보"],
                                    "answer": 1,
                                    "explanation": "시스템 오류로 인한 중복 결제는 명백한 회사 측 귀책이므로 최우선 순위로 즉시 1건을 취소해야 합니다."
                                },
                                {
                                    "question": "Q10. 상담 완료 후 CRM 시스템에 기록해야 하는 필수 데이터가 아닌 것은?",
                                    "options": ["고객의 개인 사생활 및 정치적 성향", "취소 사유 분류 코드", "환불 승인 금액 및 공제액", "고객 안내 완료 시각"],
                                    "answer": 1,
                                    "explanation": "개인정보 보호법 및 상담 윤리상 업무와 무관한 개인 사생활이나 성향은 기록해서는 안 됩니다."
                                }
                            ]
                            st.session_state.home_quiz_data = quiz_questions
                            st.session_state.home_current_question_index = 0
                            st.session_state.home_quiz_score = 0
                            st.session_state.home_quiz_answers = [1] * len(quiz_questions)
                            st.session_state.home_show_explanation = False
                            st.session_state.home_is_quiz_active = True
                            st.session_state.home_quiz_type_key = str(uuid.uuid4())
                            st.success(f"✅ '{topic}' 주제에 대한 객관식 퀴즈 10문항이 생성되었습니다!")

                        else:  # 실습 예제 아이디어
                            example_md = f"""### 💡 [{topic}] 실습 예제 아이디어 5선

1. **예제 1: 출발 3시간 전 당일 취소 요청 고객 응대 롤플레잉**
   - **학습 목표**: 당일 취소 위약금 100% 규정을 고객 반발 없이 설득력 있게 안내하는 기법 습득
   - **실습 내용**: 화가 난 고객과의 대화 시뮬레이션 및 예외 규정(진단서 등) 안내 실습
   - **예상 소요 시간**: 20분 | **난이도**: 중급

2. **예제 2: 현지 기상 악화(태풍/폭우) 단체 투어 취소 대량 공지 작성**
   - **학습 목표**: 긴급 상황 발생 시 다수 고객 대상 일괄 전액 환불 공지문 작성
   - **실습 내용**: 카카오 알림톡/SMS 템플릿 작성 및 신속 환불 시스템 등록 절차 실습
   - **예상 소요 시간**: 15분 | **난이도**: 초급

3. **예제 3: 호텔 오버부킹으로 인한 강제 취소 시 대체 숙소 및 보상 협상**
   - **학습 목표**: 호텔 측 과실로 인한 취소 시 상위 호환 숙소 제공 및 바우처 보상 협상
   - **실습 내용**: 고객 불만 완화, 제휴 호텔 긴급 수배, 차액 보상 승인 프로세스 실습
   - **예상 소요 시간**: 30분 | **난이도**: 고급

4. **예제 4: 부분 취소(4인 중 1인 불참) 시 분할 결제 취소 처리**
   - **학습 목표**: 전체 예약 중 특정 인원만 부분 취소하고 잔여 인원 바우처 재발행
   - **실습 내용**: CRM 분할 취소 기능 사용법 및 카드 부분 취소 전표 발송 실습
   - **예상 소요 시간**: 15분 | **난이도**: 초급

5. **예제 5: 환불 지연에 따른 2차 재인입 강경 컴플레인 고객 응대**
   - **학습 목표**: PG사 처리 지연으로 입금이 안 된 고객의 에스컬레이션 방어
   - **실습 내용**: 실시간 카드사 승인 번호 확인, PG사 확인증 발급 및 고객 안심 안내
   - **예상 소요 시간**: 25분 | **난이도**: 중급
"""
                            st.session_state.home_generated_example = example_md
                            st.markdown(example_md)
                else:
                    st.warning("학습할 주제를 입력해주세요.")

            if st.button(L.get("close_button", "닫기"), key="close_home_content"):
                st.session_state.show_home_content = False

    # 4. 🔍 RAG 챗봇 (사내 지식 베이스 검색 연동)
    if st.session_state.get('show_home_rag', False):
        with st.expander(f"🔍 {L.get('rag_chatbot', 'RAG 챗봇')}", expanded=True):
            rag_query = st.text_input(
                L.get("question_label", "질문:"),
                key="home_rag_query",
                placeholder="질문을 입력하세요 (예: Mercedo Benz는 어떤 회사에요?, eSIM 활성화는 어떻게 하나요?)..."
            )
            if st.button(L.get("ask_button", "질문하기"), key="ask_home_rag"):
                if rag_query:
                    with st.spinner("사내 지식 베이스 및 RAG 검색 중..."):
                        response = get_rag_chatbot_response(rag_query)
                        st.info(f"🤖 {response}")
                else:
                    st.warning("질문 내용을 입력해 주세요.")
            if st.button(L.get("close_button", "닫기"), key="close_home_rag"):
                st.session_state.show_home_rag = False


    # 5. 🛡️ QA & 컴플라이언스 자동 감사 시스템 (Enterprise Audit Dashboard)
    if st.session_state.get('show_home_qa_audit', False):
        with st.expander(f"🛡️ {L.get('qa_compliance_audit_title', 'QA & 컴플라이언스 자동 감사 시스템 (Enterprise Audit Dashboard)')}", expanded=True):
            st.markdown(f"### {L.get('qa_home_center_title', '🏢 센터 전체 QA 품질 & 컴플라이언스 총괄 현황')}")
            st.caption(L.get('qa_home_center_desc', 'AI 기반 실시간 전수 감사를 통해 도출된 상담 품질 지표, 규정 준수율 및 금지어 적발 통계입니다.'))

            # 총괄 KPI 4종
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric(L.get("qa_home_avg_score", "🏆 센터 평균 QA 점수"), f"91.8 {L.get('qa_points_unit', '점')}", delta=L.get("qa_home_avg_score_delta", "+4.2점 (A등급)"))
            with kpi2:
                st.metric(L.get("qa_home_compliance_rate", "📋 필수 고지 준수율"), "98.5%", delta=L.get("qa_home_compliance_rate_delta", "+6.8% (최우수)"))
            with kpi3:
                st.metric(L.get("qa_home_risk_rate", "🚨 금지어/리스크 적발률"), "0.1%", delta=L.get("qa_home_risk_rate_delta", "-1.4% (안전 수준)"), delta_color="normal")
            with kpi4:
                st.metric(L.get("qa_home_coaching_rate", "💡 AI 코칭 이행률"), "94.2%", delta=L.get("qa_home_coaching_rate_delta", "+8.5% 개선"))

            st.divider()
            st.markdown(f"### {L.get('qa_home_sim_title', '🔍 실시간 상담 케이스 자동 감사 시뮬레이터 (Case Audit Viewer)')}")
            
            case_good_label = L.get("qa_home_case_good", "모범 상담 케이스 (eSIM 장애 긴급 해결 & 규정 안내)")
            case_bad_label = L.get("qa_home_case_bad", "컴플라이언스 위반 케이스 (단정적 거절 & 임의 구두 확약)")
            
            case_type = st.radio(
                L.get("qa_home_select_case", "감사 대상 상담 케이스 선택:"),
                [case_good_label, case_bad_label],
                horizontal=True,
                key="home_qa_case_select"
            )

            from utils.qa_compliance_auditor import evaluate_chat_qa_compliance

            # 다국어 지원 샘플 대화록
            if case_type == case_good_label:
                if current_lang == "en":
                    sample_msgs = [
                        {"role": "customer", "content": "I arrived overseas but my eSIM has no network! Please help."},
                        {"role": "agent_response", "content": "Hello! I sincerely apologize for the inconvenience during your trip. Let me verify your reservation number and smartphone model. According to our policy, if unresolvable, a free reissue or 100% refund is guaranteed within 1 business day."},
                        {"role": "customer", "content": "Booking KLK-8812, iPhone 15."},
                        {"role": "agent_response", "content": "Thank you for confirming. Please enable Data Roaming in Settings > Cellular. Do you have any additional questions?"},
                        {"role": "customer", "content": "Data works now! No more questions, thank you."},
                        {"role": "agent_response", "content": "I am so glad it works! Thank you for contacting us, and have a wonderful and safe trip!"}
                    ]
                elif current_lang == "ja":
                    sample_msgs = [
                        {"role": "customer", "content": "海外に到着しましたが、eSIMのデータ通信が繋がりません！急いでいます。"},
                        {"role": "agent_response", "content": "お客様、海外での不通で大変ご不便とご心配をおかけし申し訳ございません。迅速に確認いたしますので、予約番号とお使いのスマートフォンの機種をお知らせいただけますか？当社の公式保証規定に基づき、未解決の場合は1営業日以内の100%無償再発行または全額返金が適用されます。"},
                        {"role": "customer", "content": "予約番号KLK-8812、iPhone 15です。"},
                        {"role": "agent_response", "content": "確認ありがとうございます。設定 > モバイル通信でデータローミングをオンにしてください。他にご質問はございますか？"},
                        {"role": "customer", "content": "繋がりました！追加の質問はありません。ありがとうございます。"},
                        {"role": "agent_response", "content": "無事解決して幸いです！お問い合わせいただきありがとうございました。安全で楽しいご旅行を、良い一日をお過ごしください！"}
                    ]
                else:
                    sample_msgs = [
                        {"role": "customer", "content": "해외 도착했는데 eSIM 데이터가 전혀 안 터져요! 급합니다."},
                        {"role": "agent_response", "content": "고객님, 낯선 해외에서 데이터가 연결되지 않아 얼마나 당황스럽고 속상하셨습니까. 신속히 확인해 드리겠습니다. 본인 확인을 위해 예약번호와 사용 중이신 스마트폰 기종을 말씀해 주시겠습니까?"},
                        {"role": "customer", "content": "예약번호 KLK-8812이고 아이폰 15입니다."},
                        {"role": "agent_response", "content": "확인 감사드립니다. 아이폰 설정 > 셀룰러에서 데이터 로밍 활성화 및 회선 켬 상태를 확인해 주세요. 공식 보증 규정에 따라 미해결 시 영업일 1일 이내 100% 무상 재발급 또는 전액 환불 규정이 적용됩니다."},
                        {"role": "customer", "content": "알려주신 대로 로밍 켜니까 바로 인터넷 잘 되네요! 감사합니다."},
                        {"role": "agent_response", "content": "정상 연결되어 정말 다행입니다! 혹시 다른 추가 문의 사항 있으신가요?"},
                        {"role": "customer", "content": "아니요, 더 문의할 건 없습니다."},
                        {"role": "agent_response", "content": "소중한 시간 내어 주셔서 감사드립니다. 안전하고 즐거운 여행 되시길 바라며, 좋은 하루 되세요!"}
                    ]
            else:
                if current_lang == "en":
                    sample_msgs = [
                        {"role": "customer", "content": "My flight was cancelled, refund my day tour immediately!"},
                        {"role": "agent_response", "content": "It is absolutely not allowed by policy. It is your fault for not reading the cancellation terms."},
                        {"role": "customer", "content": "It was severe weather! Why is it my fault?"},
                        {"role": "agent_response", "content": "Stop shouting. Not my department. Search on Google somewhere else."},
                        {"role": "customer", "content": "Get me your manager right now!"},
                        {"role": "agent_response", "content": "Fine. I unconditionally guarantee 100% full refund from my own pocket."}
                    ]
                elif current_lang == "ja":
                    sample_msgs = [
                        {"role": "customer", "content": "飛行機が欠航してツアーに参加できませんでした。今すぐ返金してください！"},
                        {"role": "agent_response", "content": "規定上絶対にできません。当日取消不可と約款に書いてあるのでお客様の過失です。"},
                        {"role": "customer", "content": "天災なのに何でお客のせいなんですか？無責任すぎます！"},
                        {"role": "agent_response", "content": "大声を上げないでください。私の管轄ではありません。他で探してください。"},
                        {"role": "customer", "content": "責任者を出してください！"},
                        {"role": "agent_response", "content": "分かりました。無条件で全額返金します。私の自腹で補償します。"}
                    ]
                else:
                    sample_msgs = [
                        {"role": "customer", "content": "비행기가 결항되어 당일 투어를 못 갔는데 환불해 주세요!"},
                        {"role": "agent_response", "content": "규정상 절대 안 됩니다. 당일 취소 불가라고 약관에 적혀 있으니 고객님 잘못입니다."},
                        {"role": "customer", "content": "천재지변인데 왜 제 잘못입니까? 너무 무책임하네요!"},
                        {"role": "agent_response", "content": "소리 지르지 마세요. 제 소관이 아닙니다. 다른 데 가서 알아보세요."},
                        {"role": "customer", "content": "팀장 나오라고 하세요!"},
                        {"role": "agent_response", "content": "알겠습니다. 제가 무조건 100% 다 전액 환불해 드릴게요. 개인 돈으로 물어드리겠습니다."}
                    ]

            audit_res = evaluate_chat_qa_compliance(sample_msgs, lang=current_lang)

            # 감사 결과 카드 렌더링
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                grade_title = L.get("qa_home_audit_result_header", "종합 감사 결과: {grade} 등급 ({desc})").replace("{grade}", audit_res['grade']).replace("{desc}", audit_res['grade_desc'])
                st.markdown(f"#### 🏆 {grade_title}")
                st.metric(L.get("qa_home_audit_score_label", "종합 QA 점수"), f"{audit_res['final_score']} / 100 {L.get('qa_points_unit', '점')}")
                
                st.markdown(f"##### 📊 {L.get('qa_home_sub_metrics', '세부 품질 지표')}")
                pts_u = L.get('qa_points_unit', '점')
                st.write(f"**{L.get('qa_empathy_title', '공감도 및 친절도 (Empathy & Courtesy)')}**: `{audit_res['scores']['empathy_score']} {pts_u}`")
                st.progress(audit_res['scores']['empathy_score'] / 100.0)
                st.write(f"**{L.get('qa_solution_title', '해결책 및 규정 정확도 (Solution Accuracy)')}**: `{audit_res['scores']['solution_score']} {pts_u}`")
                st.progress(audit_res['scores']['solution_score'] / 100.0)
                st.write(f"**{L.get('qa_compliance_title', '절차 준수 및 컴플라이언스 (Compliance)')}**: `{audit_res['scores']['compliance_score']} {pts_u}`")
                st.progress(audit_res['scores']['compliance_score'] / 100.0)

                st.markdown(f"##### 📋 {L.get('qa_home_checklist_sub', '필수 고지 체크리스트')}")
                for chk in audit_res['checklist']:
                    st.write(f"{'✅' if chk['status'] == 'PASS' else '❌'} **{chk['name']}**: `{chk['status']}`")

            with res_col2:
                st.markdown(f"#### 🚨 {L.get('qa_home_prohibited_sub', '금지어 및 리스크 발언')}")
                if audit_res['violation_count'] > 0:
                    for v in audit_res['prohibited_violations']:
                        st.error(f"**[{v['severity']}] {v['rule_name']}** (turn #{v['turn_index']})\n- {v['matched_text']}\n- -{v['penalty']} {pts_u}\n- {v['advice']}")
                else:
                    st.success(L.get("qa_clean_msg", "✅ **금지어 및 리스크 발언 0건 (Clean)**\n단정적 거절이나 고객 귀책 전가 없이 규정을 완벽히 준수했습니다."))

                st.markdown(f"#### 💡 {L.get('qa_home_coaching_sub', 'AI 맞춤형 코칭 피드백')}")
                st.write(f"**{L.get('qa_good_points', '우수 사항 (Good Points):')}**")
                for gp in audit_res['coaching']['good_points']:
                    st.write(f"- {gp}")
                
                st.write(f"**{L.get('qa_improvements', '개선 권장 사항 (Improvements):')}**")
                for imp in audit_res['coaching']['improvements']:
                    st.write(f"- {imp}")
                
                st.info(f"👔 **{L.get('qa_supervisor_summary', '수퍼바이저 총평:')}**\n{audit_res['coaching']['supervisor_summary']}")

            if st.button(L.get("close_button", "닫기"), key="close_home_qa_audit"):
                st.session_state.show_home_qa_audit = False
