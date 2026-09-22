# ========================================
# _pages/_chat_history.py
# 채팅 시뮬레이터 - 이력 관리 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import generate_outbound_call_summary
from utils.history_handler import (
    load_simulation_histories_local, delete_all_history_local,
    save_simulation_history_local, generate_chat_summary,
    export_history_to_word, export_history_to_pptx, export_history_to_pdf
)
from visualization import visualize_case_trends, visualize_customer_profile_scores, visualize_customer_characteristics
from llm_client import get_api_key
import numpy as np
from datetime import datetime, timedelta
import os
import json
import csv
import io
import uuid


def render_chat_history(current_lang, L):
    """이력 관리 UI 렌더링"""
    # =========================
    # 0. 전체 이력 삭제 및 세션 초기화
    # =========================
    col_del, col_reset, _ = st.columns([1, 1, 3])
    with col_del:
        if st.button(L["delete_history_button"], key="trigger_delete_hist"):
            st.session_state.show_delete_confirm = True
    with col_reset:
        if st.button("🔄 세션 초기화", key="reset_all_session", help="모든 채팅/통화 응대 기록을 초기화합니다"):
            st.session_state.show_reset_confirm = True
    
    # 세션 초기화 확인
    if st.session_state.get("show_reset_confirm", False):
        with st.container():
            st.warning("⚠️ 모든 채팅/통화 응대 기록이 초기화됩니다. 계속하시겠습니까?")
            c_yes, c_no = st.columns(2)
            if c_yes.button("예, 초기화합니다", key="confirm_reset_yes"):
                # 모든 채팅/통화 관련 상태 초기화
                st.session_state.simulator_messages = []
                st.session_state.call_messages = []
                st.session_state.simulator_memory.clear()
                st.session_state.initial_advice_provided = False
                st.session_state.is_chat_ended = False
                st.session_state.agent_response_area_text = ""
                st.session_state.customer_query_text_area = ""
                st.session_state.last_transcript = ""
                st.session_state.sim_audio_bytes = None
                st.session_state.sim_stage = "WAIT_FIRST_QUERY"
                st.session_state.call_sim_stage = "WAITING_CALL"
                st.session_state.inquiry_text = ""
                st.session_state.call_content = ""
                st.session_state.incoming_phone_number = None
                st.session_state.incoming_call = None
                st.session_state.call_active = False
                st.session_state.start_time = None
                st.session_state.call_duration = None
                st.session_state.transfer_summary_text = ""
                st.session_state.language_at_transfer_start = None
                st.session_state.customer_attachment_file = []
                st.session_state.sim_attachment_context_for_llm = ""
                st.session_state.agent_attachment_file = []
                st.session_state.show_reset_confirm = False
                st.success("✅ 모든 세션이 초기화되었습니다.")
            if c_no.button("취소", key="confirm_reset_no"):
                st.session_state.show_reset_confirm = False

    if st.session_state.show_delete_confirm:
        with st.container():
            st.warning(L["delete_confirm_message"])
            c_yes, c_no = st.columns(2)
            if c_yes.button(L["delete_confirm_yes"], key="confirm_del_yes"):
                with st.spinner(L["deleting_history_progress"]):
                    delete_all_history_local()
                    st.session_state.simulator_messages = []
                    st.session_state.simulator_memory.clear()
                    st.session_state.show_delete_confirm = False
                    st.session_state.is_chat_ended = False
                    st.session_state.sim_stage = "WAIT_FIRST_QUERY"
                    st.session_state.customer_attachment_file = []
                    st.session_state.sim_attachment_context_for_llm = ""
                    st.session_state.agent_attachment_file = []
                    st.success(L["delete_success"])
            if c_no.button(L["delete_confirm_no"], key="confirm_del_no"):
                st.session_state.show_delete_confirm = False

    # =========================
    # 1. 이전 이력 로드 (검색/필터링 기능 개선)
    # =========================
    with st.expander(L["history_expander_title"]):
        histories = load_simulation_histories_local(current_lang)

        # 전체 통계 및 트렌드 대시보드
        cases_with_summary = [
            h for h in histories
            if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
            and not h.get("is_call", False)
        ]

        if cases_with_summary:
            st.markdown("---")
            st.subheader("📈 과거 케이스 트렌드 대시보드")

            trend_chart = visualize_case_trends(histories, current_lang)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                avg_sentiment = np.mean([h["summary"].get(
                    "customer_sentiment_score", 50) for h in cases_with_summary if h.get("summary")])
                avg_satisfaction = np.mean(
                    [h["summary"].get("customer_satisfaction_score", 50) for h in cases_with_summary if
                     h.get("summary")])
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "평균 감정 점수",
                        f"{avg_sentiment:.1f}/100",
                        f"총 {len(cases_with_summary)}건")
                with col2:
                    st.metric(
                        "평균 만족도",
                        f"{avg_satisfaction:.1f}/100",
                        f"총 {len(cases_with_summary)}건")

            st.markdown("---")

        # 검색 폼
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            search_query = st.text_input(
                L["search_history_label"],
                key="sim_hist_search_input_new")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button(
                L["history_search_button"],
                key="apply_search_btn_new")

        # 날짜 범위 필터
        today = datetime.now().date()
        date_range_value = [today - timedelta(days=7), today]
        dr = st.date_input(
            L["date_range_label"],
            value=date_range_value,
            key="sim_hist_date_range_actual",
        )

        # 필터링 로직
        current_search_query = search_query.strip()

        if histories:
            start_date = min(dr)
            end_date = max(dr)

            filtered = []
            for h in histories:
                if h.get("is_call", False):
                    continue

                ok_search = True
                if current_search_query:
                    q = current_search_query.lower()
                    text = (
                        h["initial_query"] +
                        " " +
                        h["customer_type"]).lower()

                    summary = h.get("summary")
                    if summary and isinstance(summary, dict):
                        summary_text = summary.get(
                            "main_inquiry", "") + " " + summary.get("summary", "")
                        text += " " + summary_text.lower()

                    if q not in text:
                        ok_search = False

                ok_date = True
                ts = h.get("timestamp")
                if ts:
                    try:
                        d = datetime.fromisoformat(ts).date()
                        if not (start_date <= d <= end_date):
                            ok_date = False
                    except Exception:
                        pass

                if ok_search and ok_date:
                    filtered.append(h)
        else:
            filtered = []

        # 표시할 목록 결정
        is_searching_or_filtering = bool(
            current_search_query) or dr != date_range_value

        if not is_searching_or_filtering:
            filtered_for_display = filtered[:10]
        else:
            filtered_for_display = filtered

        # 표시 로직
        if filtered_for_display:
            def _label(h):
                try:
                    t = datetime.fromisoformat(h["timestamp"])
                    t_str = t.strftime("%m-%d %H:%M")
                except Exception:
                    t_str = h.get("timestamp", "")

                summary = h.get("summary")
                if summary and isinstance(summary, dict):
                    main_inquiry = summary.get(
                        "main_inquiry", h["initial_query"][:30])
                    sentiment = summary.get("customer_sentiment_score", 50)
                    satisfaction = summary.get(
                        "customer_satisfaction_score", 50)
                    q = main_inquiry[:30].replace("\n", " ")
                    attachment_icon = "📎" if h.get(
                        "attachment_context") else ""
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} | 감정:{sentiment} 만족:{satisfaction} - {q}..."
                else:
                    q = h["initial_query"][:30].replace("\n", " ")
                    attachment_icon = "📎" if h.get(
                        "attachment_context") else ""
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} - {q}..."

            options_map = {_label(h): h for h in filtered_for_display}

            if is_searching_or_filtering:
                st.caption(
                    f"🔎 총 {len(filtered_for_display)}개 이력 검색됨 (전화 이력 제외)")
            else:
                st.caption(
                    f"⭐ 최근 {len(filtered_for_display)}개 이력 표시 중 (전화 이력 제외)")

            sel_key = st.selectbox(
                L["history_selectbox_label"],
                options=list(
                    options_map.keys()))

            if st.button(L["history_load_button"], key="load_hist_btn"):
                h = options_map[sel_key]
                st.session_state.customer_query_text_area = h["initial_query"]

                if not h.get("messages") and h.get("summary"):
                    summary = h["summary"]
                    reconstructed_messages = [
                        {"role": "customer", "content": h["initial_query"]}
                    ]
                    if summary.get("key_responses"):
                        for response in summary.get(
                                "key_responses", [])[:3]:
                            reconstructed_messages.append(
                                {"role": "agent_response", "content": response})
                    summary_text = f"**요약된 상담 이력**\n\n"
                    summary_text += f"주요 문의: {summary.get('main_inquiry', 'N/A')}\n"
                    summary_text += f"고객 감정 점수: {summary.get('customer_sentiment_score', 50)}/100\n"
                    summary_text += f"고객 만족도: {summary.get('customer_satisfaction_score', 50)}/100\n"
                    summary_text += f"\n전체 요약:\n{summary.get('summary', 'N/A')}"
                    reconstructed_messages.append(
                        {"role": "supervisor", "content": summary_text})
                    st.session_state.simulator_messages = reconstructed_messages

                    st.markdown("---")
                    st.subheader("📊 로드된 케이스 분석")

                    loaded_profile = {
                        "sentiment_score": summary.get("customer_sentiment_score", 50),
                        "urgency_level": "medium",
                        "predicted_customer_type": h.get("customer_type", "normal")
                    }

                    profile_chart = visualize_customer_profile_scores(
                        loaded_profile, current_lang)
                    if profile_chart:
                        st.plotly_chart(
                            profile_chart, use_container_width=True)
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                L.get(
                                    "sentiment_score_label",
                                    "감정 점수"),
                                f"{summary.get('customer_sentiment_score', 50)}/100")
                        with col2:
                            st.metric(
                                L.get(
                                    "urgency_score_label",
                                    "긴급도"),
                                f"50/100")
                        with col3:
                            st.metric(
                                L.get(
                                    "customer_type_label", "고객 유형"), h.get(
                                    "customer_type", "normal"))

                    if summary.get("customer_characteristics") or summary.get(
                            "privacy_info"):
                        characteristics_chart = visualize_customer_characteristics(
                            summary, current_lang)
                        if characteristics_chart:
                            st.plotly_chart(
                                characteristics_chart, use_container_width=True)
                else:
                    st.session_state.simulator_messages = h.get("messages", [])

                st.session_state.initial_advice_provided = True
                st.session_state.is_chat_ended = h.get("is_chat_ended", False)
                st.session_state.sim_attachment_context_for_llm = h.get(
                    "attachment_context", "")
                st.session_state.customer_attachment_file = []
                st.session_state.agent_attachment_file = []

                if st.session_state.is_chat_ended:
                    st.session_state.sim_stage = "CLOSING"
                else:
                    messages = st.session_state.simulator_messages
                    last_role = messages[-1]["role"] if messages else None
                    if last_role == "agent_response":
                        st.session_state.sim_stage = "CUSTOMER_TURN"
                    elif last_role == "customer_rebuttal":
                        st.session_state.sim_stage = "AGENT_TURN"
                    elif last_role == "supervisor" and messages and messages[-1]["content"] == L[
                            "customer_closing_confirm"]:
                        st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"

                st.session_state.simulator_memory.clear()
        else:
            st.info(L["no_history_found"])


def render_aht_timer(L):
    """AHT 타이머 렌더링"""
    elapsed_placeholder = st.empty()

    if st.session_state.start_time is not None:
        elapsed_time = datetime.now() - st.session_state.start_time
        total_seconds = elapsed_time.total_seconds()

        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        if total_seconds > 900:
            delta_str = L["timer_info_risk"]
            delta_color = "inverse"
        elif total_seconds > 600:
            delta_str = L["timer_info_warn"]
            delta_color = "off"
        else:
            delta_str = L["timer_info_ok"]
            delta_color = "normal"

        elapsed_placeholder.metric(
            L["timer_metric"],
            time_str,
            delta=delta_str,
            delta_color=delta_color
        )

        if seconds % 3 == 0 and total_seconds < 1000:
            import time
            time.sleep(1)

    st.markdown("---")


def render_closing_downloads(L, current_lang):
    """종료 상태 다운로드 UI 렌더링"""
    st.success(L["survey_sent_confirm"])
    st.info(L["new_simulation_ready"])

    # BPO 비즈니스 임팩트 & ROI 분석
    bpo_summary = None
    try:
        from utils.bpo_analytics import generate_bpo_summary
        actual_aht = st.session_state.get("actual_aht_seconds")
        if not actual_aht or actual_aht < 5.0:
            actual_aht = 65.0
        
        draft_history = st.session_state.get("bpo_draft_history", [])
        messages = st.session_state.get("simulator_messages", [])
        bpo_summary = generate_bpo_summary(
            actual_aht_seconds=actual_aht,
            draft_adoption_history=draft_history,
            messages=messages
        )
    except Exception as e:
        print(f"BPO summary generation error: {e}")

    if bpo_summary:
        st.markdown("---")
        st.markdown(f"### 📊 {L.get('bpo_roi_card_title', 'BPO 비즈니스 임팩트 & AI Copilot ROI 분석')}")
        st.caption(L.get('bpo_roi_card_desc', '상담 전 과정의 AI 초안 채택률, AHT 단축 효과 및 고객 감정/이탈 방지 기여도 정량 분석 리포트입니다.'))

        roi_c1, roi_c2, roi_c3, roi_c4 = st.columns(4)
        with roi_c1:
            aht_pct = bpo_summary['aht_roi']['reduction_rate_pct']
            saved_sec = bpo_summary['aht_roi']['saved_seconds']
            st.metric(
                label=L.get("bpo_metric_aht", "⏱️ AHT 절감률"),
                value=f"-{aht_pct}%",
                delta=f"{bpo_summary['aht_roi']['baseline_aht_seconds']}s → {bpo_summary['aht_roi']['actual_aht_seconds']}s ({saved_sec}s 단축)",
                delta_color="normal"
            )
        with roi_c2:
            cost_ticket = bpo_summary['aht_roi']['saved_cost_per_ticket']
            m1000 = bpo_summary['aht_roi']['monthly_1000_savings']
            st.metric(
                label=L.get("bpo_metric_cost", "💰 티켓당 절감 비용"),
                value=f"₩{cost_ticket:,}",
                delta=f"월 1,000건 시 ₩{m1000:,}",
                delta_color="normal"
            )
        with roi_c3:
            adopt_pct = bpo_summary['draft_stats']['adoption_rate_pct']
            full_c = bpo_summary['draft_stats']['full_adopt_count']
            part_c = bpo_summary['draft_stats']['partial_adopt_count']
            st.metric(
                label=L.get("bpo_metric_adoption", "✍️ AI 초안 채택률"),
                value=f"{adopt_pct}%",
                delta=f"완전 {full_c}건 / 부분 {part_c}건",
                delta_color="normal"
            )
        with roi_c4:
            csat_val = bpo_summary['sentiment']['predicted_csat']
            churn_risk = bpo_summary['sentiment']['churn_risk_pct']
            shift_lbl = bpo_summary['sentiment']['shift_label']
            st.metric(
                label=L.get("bpo_metric_csat", "⭐ 예측 CSAT"),
                value=f"{csat_val} / 5.0",
                delta=f"이탈 위험 {churn_risk}% ({shift_lbl})",
                delta_color="inverse" if churn_risk > 30 else "normal"
            )

        with st.expander(L.get("bpo_detail_expander", "💼 BPO ROI 세부 분석 및 산출 근거"), expanded=False):
            st.markdown(f"""
            - **AHT 기준**: 엔터프라이즈 CS 표준 AHT **{bpo_summary['aht_roi']['baseline_aht_seconds']}초** 대비 **{bpo_summary['aht_roi']['actual_aht_seconds']}초** 소요 (**{bpo_summary['aht_roi']['saved_seconds']}초 절감**)
            - **인건비 환산 기준**: 상담원 시급 **₩{bpo_summary['aht_roi']['hourly_wage']:,}** 기준
            - **월간 예상 비용 절감**:
              - 월 1,000건 처리 센터: **₩{bpo_summary['aht_roi']['monthly_1000_savings']:,}** 절감
              - 월 10,000건 처리 센터: **₩{bpo_summary['aht_roi']['monthly_10000_savings']:,}** 절감
            - **고객 감정 전환(Sentiment Shift)**:
              - 상담 초기: **{bpo_summary['sentiment']['initial_sentiment_score']}점** → 상담 종료: **{bpo_summary['sentiment']['final_sentiment_score']}점** (변화량: **+{bpo_summary['sentiment']['sentiment_shift_delta']}점**)
              - 감정 상태: **{bpo_summary['sentiment']['shift_label']}** | 고객 이탈 위험도: **{bpo_summary['sentiment']['churn_risk_pct']}%**
            """)

    # 🛡️ QA & 컴플라이언스 실시간 자동 감사 리포트 (Automated Audit Report)
    qa_audit_result = None
    try:
        from utils.qa_compliance_auditor import evaluate_chat_qa_compliance
        messages = st.session_state.get("simulator_messages", [])
        qa_audit_result = evaluate_chat_qa_compliance(messages, lang=current_lang)
    except Exception as e:
        print(f"QA audit generation error: {e}")

    if qa_audit_result and qa_audit_result.get("grade") != "N/A":
        st.markdown("---")
        st.markdown(f"### 🛡️ {L.get('qa_audit_report_title', 'QA & 컴플라이언스 실시간 자동 감사 리포트 (Audit Report)')}")
        st.caption(L.get('qa_audit_report_desc', '대화 전 과정을 실시간 분석하여 친절도/공감도, 규정 준수도, 금지어 탐지 및 필수 고지 항목을 다차원 평가한 엔터프라이즈 감사 리포트입니다.'))

        # 1) 메트릭 카드 4종
        qa_c1, qa_c2, qa_c3, qa_c4 = st.columns(4)
        with qa_c1:
            grade_val = f"{qa_audit_result['grade']} {L.get('qa_grade_label', '등급')}" if current_lang != "en" else f"Grade {qa_audit_result['grade']}"
            st.metric(
                label=L.get("qa_metric_grade", "🏆 종합 QA 등급"),
                value=grade_val,
                delta=qa_audit_result['grade_desc'],
                delta_color="normal" if qa_audit_result['grade'] in ['S', 'A', 'B'] else "inverse"
            )
        with qa_c2:
            pts_unit = L.get("qa_points_unit", "점")
            target_status = L.get("qa_pass_target_met", "합격 기준(75점) 달성") if qa_audit_result['final_score'] >= 75 else L.get("qa_pass_target_unmet", "합격 기준(75점) 미달")
            st.metric(
                label=L.get("qa_metric_score", "📊 종합 평가 점수"),
                value=f"{qa_audit_result['final_score']} / 100 {pts_unit}",
                delta=target_status,
                delta_color="normal" if qa_audit_result['final_score'] >= 75 else "inverse"
            )
        with qa_c3:
            comp_n = int(qa_audit_result['compliance_rate'] / 25)
            comp_delta = L.get("qa_items_compliant", "4개 항목 중 {n}개 준수").replace("{n}", str(comp_n))
            st.metric(
                label=L.get("qa_metric_compliance", "📋 필수 고지 준수율"),
                value=f"{qa_audit_result['compliance_rate']}%",
                delta=comp_delta,
                delta_color="normal" if qa_audit_result['compliance_rate'] >= 75 else "inverse"
            )
        with qa_c4:
            viol_count = qa_audit_result['violation_count']
            case_unit = L.get("qa_cases_unit", "건")
            viol_delta = L.get("qa_clean_status", "정상 (Clean)") if viol_count == 0 else L.get("qa_deduction_status", "{n}건 감점 발생").replace("{n}", str(viol_count))
            st.metric(
                label=L.get("qa_metric_violations", "⚠️ 금지어 적발"),
                value=f"{viol_count} {case_unit}",
                delta=viol_delta,
                delta_color="normal" if viol_count == 0 else "inverse"
            )

        # 2) 다차원 점수 세부 분석 및 금지어/체크리스트
        with st.expander(L.get("qa_audit_detail_expander", "🔍 QA 세부 평가 지표 및 AI 코칭 리포트 보기"), expanded=True):
            det_col1, det_col2 = st.columns(2)
            
            with det_col1:
                st.markdown(f"#### {L.get('qa_dim_header', '📈 다차원 품질 지표')}")
                scores = qa_audit_result.get("scores", {})
                pts_u = L.get("qa_points_unit", "점")
                
                emp_s = scores.get("empathy_score", 0)
                st.write(f"**{L.get('qa_empathy_title', '공감도 및 친절도 (Empathy & Courtesy)')}**: `{emp_s} {pts_u}`")
                st.progress(emp_s / 100.0)
                
                sol_s = scores.get("solution_score", 0)
                st.write(f"**{L.get('qa_solution_title', '해결책 및 규정 정확도 (Solution Accuracy)')}**: `{sol_s} {pts_u}`")
                st.progress(sol_s / 100.0)
                
                com_s = scores.get("compliance_score", 0)
                st.write(f"**{L.get('qa_compliance_title', '절차 준수 및 컴플라이언스 (Compliance)')}**: `{com_s} {pts_u}`")
                st.progress(com_s / 100.0)

                st.markdown(f"#### {L.get('qa_chk_header', '📋 필수 고지 4단계 체크리스트')}")
                for item in qa_audit_result.get("checklist", []):
                    icon = "✅" if item["status"] == "PASS" else "❌"
                    st.write(f"{icon} **{item['name']}**: `{item['status']}` ({item['feedback']})")

            with det_col2:
                st.markdown(f"#### {L.get('qa_prohibited_header', '🚨 금지어 및 리스크 발언 탐지 결과')}")
                violations = qa_audit_result.get("prohibited_violations", [])
                if violations:
                    for v in violations:
                        st.error(f"**[{v['severity']}] {v['rule_name']}** (turn #{v['turn_index']})\n- {v['matched_text']}\n- -{v['penalty']} {pts_u}\n- {v['advice']}")
                else:
                    st.success(L.get("qa_clean_msg", "✅ **금지어 및 리스크 발언 0건 (Clean)**\n단정적 거절이나 고객 귀책 전가 없이 규정을 완벽히 준수했습니다."))

                st.markdown(f"#### {L.get('qa_coaching_header', '💡 상담원 맞춤형 AI 코칭 피드백')}")
                coaching = qa_audit_result.get("coaching", {})
                st.write(f"**{L.get('qa_good_points', '🌟 우수 사항 (Good Points):')}**")
                for g in coaching.get("good_points", []):
                    st.write(f"- {g}")
                
                st.write(f"**{L.get('qa_improvements', '⚠️ 개선 권장 사항 (Improvements):')}**")
                for imp in coaching.get("improvements", []):
                    st.write(f"- {imp}")
                
                st.info(f"👔 **{L.get('qa_supervisor_summary', '수퍼바이저 종합 총평:')}**\n{coaching.get('supervisor_summary', '우수한 상담입니다.')}")

    st.markdown("---")
    st.markdown(f"**{L.get('download_current_session', '📥 현재 세션 이력 다운로드')}**")
    download_col1, download_col2, download_col3, download_col4, download_col5 = st.columns(
        5)

    current_session_history = None
    if st.session_state.simulator_messages:
        try:
            customer_type_display = st.session_state.get(
                "customer_type_sim_select", L["customer_type_options"][0])
            current_session_summary = generate_chat_summary(
                st.session_state.simulator_messages,
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.language
            )
            current_session_history = [{
                "id": f"session_{st.session_state.sim_instance_id}",
                "timestamp": datetime.now().isoformat(),
                "initial_query": st.session_state.customer_query_text_area,
                "customer_type": customer_type_display,
                "language_key": st.session_state.language,
                "messages": st.session_state.simulator_messages,
                "summary": current_session_summary,
                "is_chat_ended": True,
                "attachment_context": st.session_state.sim_attachment_context_for_llm,
                "bpo_analytics": bpo_summary,
                "rag_grounding": st.session_state.get("last_rag_citation", None)
            }]
        except Exception as e:
            st.warning(
                L.get(
                    "history_generation_error",
                    "이력 생성 중 오류 발생: {error}").format(
                    error=e))

    if current_session_history:
        with download_col1:
            try:
                filepath_word = export_history_to_word(
                    current_session_history, lang=current_lang)
                with open(filepath_word, "rb") as f:
                    st.download_button(
                        label=L.get(
                            "download_history_word",
                            "📥 이력 다운로드 (Word)"),
                        data=f.read(),
                        file_name=os.path.basename(filepath_word),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_word_file")
            except Exception as e:
                st.error(
                    L.get(
                        "word_download_error",
                        "Word 다운로드 오류: {error}").format(
                        error=e))

        with download_col2:
            try:
                filepath_pptx = export_history_to_pptx(
                    current_session_history, lang=current_lang)
                with open(filepath_pptx, "rb") as f:
                    st.download_button(
                        label=L.get(
                            "download_history_pptx",
                            "📥 이력 다운로드 (PPTX)"),
                        data=f.read(),
                        file_name=os.path.basename(filepath_pptx),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key="download_pptx_file")
            except Exception as e:
                st.error(
                    L.get(
                        "pptx_download_error",
                        "PPTX 다운로드 오류: {error}").format(
                        error=e))

        with download_col3:
            try:
                filepath_pdf = export_history_to_pdf(
                    current_session_history, lang=current_lang)
                with open(filepath_pdf, "rb") as f:
                    st.download_button(
                        label=L.get(
                            "download_history_pdf",
                            "📥 이력 다운로드 (PDF)"),
                        data=f.read(),
                        file_name=os.path.basename(filepath_pdf),
                        mime="application/pdf",
                        key="download_pdf_file")
            except Exception as e:
                st.error(
                    L.get(
                        "pdf_download_error",
                        "PDF 다운로드 오류: {error}").format(
                        error=e))

        with download_col4:
            try:
                json_data = json.dumps(
                    current_session_history, ensure_ascii=False, indent=2)
                st.download_button(
                    label=L.get(
                        "download_history_json",
                        "📥 이력 다운로드 (JSON)"),
                    data=json_data.encode('utf-8'),
                    file_name=f"chat_history_{st.session_state.sim_instance_id}.json",
                    mime="application/json",
                    key="download_chat_json_file")
            except Exception as e:
                st.error(
                    L.get(
                        "json_download_error",
                        "JSON 다운로드 오류: {error}").format(
                        error=e))

        with download_col5:
            try:
                output = io.StringIO()
                writer = csv.writer(output)

                if bpo_summary:
                    writer.writerow(["# BPO ROI Summary", f"AHT Saved: {bpo_summary['aht_roi']['reduction_rate_pct']}%", f"Cost Saved/Ticket: {bpo_summary['aht_roi']['saved_cost_per_ticket']} KRW", f"Draft Adoption: {bpo_summary['draft_stats']['adoption_rate_pct']}%", f"Predicted CSAT: {bpo_summary['sentiment']['predicted_csat']}"])
                    writer.writerow([])
                writer.writerow(["Role", "Content", "Timestamp"])

                for msg in current_session_history[0].get("messages", []):
                    writer.writerow([
                        msg.get("role", ""),
                        msg.get("content", ""),
                        current_session_history[0].get("timestamp", "")
                    ])

                csv_data = output.getvalue()
                st.download_button(
                    label=L.get("download_history_csv", "📥 이력 다운로드 (CSV)"),
                    data=csv_data.encode('utf-8-sig'),
                    file_name=f"chat_history_{st.session_state.sim_instance_id}.csv",
                    mime="text/csv",
                    key="download_chat_csv_file"
                )
            except Exception as e:
                st.error(
                    L.get(
                        "csv_download_error",
                        "CSV 다운로드 오류: {error}").format(
                        error=e))
    else:
        st.warning(L.get("no_history_to_download", "다운로드할 이력이 없습니다."))

    st.markdown("---")

    if st.button(L["new_simulation_button"], key="new_simulation_btn"):
        st.session_state.simulator_messages = []
        st.session_state.simulator_memory.clear()
        st.session_state.initial_advice_provided = False
        st.session_state.is_chat_ended = False
        st.session_state.agent_response_area_text = ""
        st.session_state.customer_query_text_area = ""
        st.session_state.last_transcript = ""
        st.session_state.sim_audio_bytes = None
        st.session_state.sim_stage = "WAIT_FIRST_QUERY"
        st.session_state.customer_attachment_file = []
        st.session_state.sim_attachment_context_for_llm = ""
        st.session_state.agent_attachment_file = []
        st.session_state.start_time = None
        st.session_state.sim_call_outbound_summary = ""
        st.session_state.sim_call_outbound_target = None
        st.session_state.bpo_draft_history = []
        st.session_state.last_agent_draft_text = ""
        st.session_state.actual_aht_seconds = None
        st.session_state.last_rag_citation = None


def render_outbound_call(L, current_lang):
    """전화 발신 진행 중 UI 렌더링"""
    target = st.session_state.get("sim_call_outbound_target", "대상")
    st.warning(L["call_outbound_loading"])

    with st.spinner(L["call_outbound_loading"]):
        # ⭐ 수정: 고객 문의 이력이 메시지로 온 경우 처리
        customer_inquiry_from_message = None
        if st.session_state.simulator_messages:
            # 최근 고객 메시지에서 문의 내용 추출
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") in ["customer", "initial_query", "customer_rebuttal"]:
                    customer_inquiry_from_message = msg.get("content", "")
                    break
        
        # 고객 문의 내용 (메시지에서 추출하거나 초기 문의 사용)
        inquiry_text = customer_inquiry_from_message or st.session_state.customer_query_text_area
        
        summary = generate_outbound_call_summary(
            inquiry_text,
            st.session_state.language,
            target
        )

        st.session_state.simulator_messages.append(
            {"role": "system_end", "content": L["call_outbound_system_msg"].format(target=target)}
        )

        summary_markdown = f"### {L['call_outbound_summary_header']}\n\n{summary}"
        st.session_state.simulator_messages.append(
            {"role": "supervisor", "content": summary_markdown}
        )

        st.session_state.sim_stage = "AGENT_TURN"
        st.session_state.sim_call_outbound_summary = summary_markdown
        st.session_state.sim_call_outbound_target = None

        customer_type_display = st.session_state.get(
            "customer_type_sim_select", "")
        save_simulation_history_local(
            inquiry_text,  # ⭐ 수정: 메시지에서 추출한 문의 내용 사용
            customer_type_display +
            f" (Outbound Call to {target})",
            st.session_state.simulator_messages,
            is_chat_ended=False,
            attachment_context=st.session_state.sim_attachment_context_for_llm,
        )

    st.success(
        f"✅ {L['call_outbound_simulation_header']}가 완료되었습니다. 요약을 확인하고 고객에게 회신하세요.")


