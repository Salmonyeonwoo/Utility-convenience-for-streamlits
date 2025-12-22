if feature_selection == L["sim_tab_chat_email"]:
    # =========================
    # 0-1. 일일 데이터 수집 통계 표시
    # =========================
    daily_stats = get_daily_data_statistics(st.session_state.language)
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("오늘 수집된 케이스", daily_stats["total_cases"])
    with col_stat2:
        st.metric("고유 고객 수", daily_stats["unique_customers"],
                  delta="목표: 5인 이상" if daily_stats["target_met"] else "목표 미달")
    with col_stat3:
        st.metric("요약 완료 케이스", daily_stats["cases_with_summary"])
    with col_stat4:
        status_icon = "✅" if daily_stats["target_met"] else "⚠️"
        st.metric("목표 달성", status_icon,
                  delta="달성" if daily_stats["target_met"] else "미달성")

    st.markdown("---")

    # =========================
    # 0. 전체 이력 삭제
    # =========================
    col_del, _ = st.columns([1, 4])
    with col_del:
        if st.button(L["delete_history_button"], key="trigger_delete_hist"):
            st.session_state.show_delete_confirm = True

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
                    st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
                    st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
                    st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
                    st.success(L["delete_success"])
            if c_no.button(L["delete_confirm_no"], key="confirm_del_no"):
                st.session_state.show_delete_confirm = False

    # =========================
    # 1. 이전 이력 로드 (검색/필터링 기능 개선)
    # =========================
    with st.expander(L["history_expander_title"]):
        # Always load all available histories for the current language (sorted
        # by recency)
        histories = load_simulation_histories_local(current_lang)

        # 전체 통계 및 트렌드 대시보드 (요약 데이터가 있는 경우만)
        cases_with_summary = [
            h for h in histories
            if h.get("summary") and isinstance(h.get("summary"), dict) and h.get("is_chat_ended", False)
            and not h.get("is_call", False)  # 전화 이력 제외
        ]

        if cases_with_summary:
            st.markdown("---")
            st.subheader("📈 과거 케이스 트렌드 대시보드")

            # 트렌드 차트 표시
            trend_chart = visualize_case_trends(histories, current_lang)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
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

        # ⭐ 검색 폼 제거 및 독립된 위젯 사용
        col_search, col_btn = st.columns([4, 1])

        with col_search:
            # st.text_input은 Enter 키 입력 시 앱을 재실행합니다.
            search_query = st.text_input(
                L["search_history_label"],
                key="sim_hist_search_input_new")

        with col_btn:
            # 검색 버튼: 누르면 앱을 강제 재실행하여 검색/필터링 로직을 다시 타도록 합니다.
            # Align button vertically
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

        # --- Filtering Logic ---
        current_search_query = search_query.strip()

        if histories:
            start_date = min(dr)
            end_date = max(dr)

            filtered = []
            for h in histories:
                # 전화 이력은 제외 (채팅/이메일 탭이므로)
                if h.get("is_call", False):
                    continue

                ok_search = True
                if current_search_query:
                    q = current_search_query.lower()
                    # 검색 대상: 초기 문의, 고객 유형, 요약 데이터
                    text = (
                        h["initial_query"] +
                        " " +
                        h["customer_type"]).lower()

                    # 요약 데이터가 있으면 요약 내용도 검색 대상에 포함
                    summary = h.get("summary")
                    if summary and isinstance(summary, dict):
                        summary_text = summary.get(
                            "main_inquiry", "") + " " + summary.get("summary", "")
                        text += " " + summary_text.lower()

                    # Check if query matches in initial query, customer type,
                    # or summary
                    if q not in text:
                        ok_search = False

                ok_date = True
                ts = h.get("timestamp")
                if ts:
                    try:
                        d = datetime.fromisoformat(ts).date()
                        # Apply date filtering
                        if not (start_date <= d <= end_date):
                            ok_date = False
                    except Exception:
                        pass  # Ignore histories with invalid timestamp

                if ok_search and ok_date:
                    filtered.append(h)
        else:
            filtered = []

        # Determine the list for display (⭐ 요청 사항: 검색어/필터가 없으면 최근 10건만 표시)
        is_searching_or_filtering = bool(
            current_search_query) or dr != date_range_value

        if not is_searching_or_filtering:
            # 검색/필터 조건이 없으면, 전체 이력 중 최신 10건만 표시
            filtered_for_display = filtered[:10]  # 필터링된 목록(전화 제외) 중 10개
        else:
            # 검색/필터 조건이 있으면, 필터링된 모든 결과를 표시
            filtered_for_display = filtered

        # --- Display Logic ---

        if filtered_for_display:
            def _label(h):
                try:
                    t = datetime.fromisoformat(h["timestamp"])
                    t_str = t.strftime("%m-%d %H:%M")
                except Exception:
                    t_str = h.get("timestamp", "")

                # 요약 데이터가 있으면 요약 정보 표시, 없으면 초기 문의 표시
                summary = h.get("summary")
                if summary and isinstance(summary, dict):
                    main_inquiry = summary.get(
                        "main_inquiry", h["initial_query"][:30])
                    sentiment = summary.get("customer_sentiment_score", 50)
                    satisfaction = summary.get(
                        "customer_satisfaction_score", 50)
                    q = main_inquiry[:30].replace("\n", " ")
                    # 첨부 파일 여부 표시 추가
                    attachment_icon = "📎" if h.get(
                        "attachment_context") else ""
                    # 요약 데이터 표시 (감정/만족도 점수 포함)
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} | 감정:{sentiment} 만족:{satisfaction} - {q}..."
                else:
                    q = h["initial_query"][:30].replace("\n", " ")
                    attachment_icon = "📎" if h.get(
                        "attachment_context") else ""
                    return f"[{t_str}] {attachment_icon} {h['customer_type']} - {q}..."

            options_map = {_label(h): h for h in filtered_for_display}

            # Show a message indicating what is displayed if filters were
            # applied
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

                # 메시지가 비어있고 요약 데이터가 있는 경우, 요약을 기반으로 최소한의 메시지 재구성
                if not h.get("messages") and h.get("summary"):
                    summary = h["summary"]
                    # 요약 데이터를 기반으로 기본 메시지 구조 생성
                    reconstructed_messages = [
                        {"role": "customer", "content": h["initial_query"]}
                    ]
                    # 요약에서 핵심 응답 추가
                    if summary.get("key_responses"):
                        for response in summary.get(
                                "key_responses", [])[:3]:  # 최대 3개만
                            reconstructed_messages.append(
                                {"role": "agent_response", "content": response})
                    # 요약 정보를 supervisor 메시지로 추가
                    summary_text = f"**요약된 상담 이력**\n\n"
                    summary_text += f"주요 문의: {summary.get('main_inquiry', 'N/A')}\n"
                    summary_text += f"고객 감정 점수: {summary.get('customer_sentiment_score', 50)}/100\n"
                    summary_text += f"고객 만족도: {summary.get('customer_satisfaction_score', 50)}/100\n"
                    summary_text += f"\n전체 요약:\n{summary.get('summary', 'N/A')}"
                    reconstructed_messages.append(
                        {"role": "supervisor", "content": summary_text})
                    st.session_state.simulator_messages = reconstructed_messages

                    # 요약 데이터 시각화
                    st.markdown("---")
                    st.subheader("📊 로드된 케이스 분석")

                    # 요약 데이터를 프로필 형식으로 변환
                    loaded_profile = {
                        "sentiment_score": summary.get("customer_sentiment_score", 50),
                        "urgency_level": "medium",  # 기본값
                        "predicted_customer_type": h.get("customer_type", "normal")
                    }

                    # 프로필 점수 차트
                    profile_chart = visualize_customer_profile_scores(
                        loaded_profile, current_lang)
                    if profile_chart:
                        st.plotly_chart(
                            profile_chart, use_container_width=True)
                    else:
                        # Plotly가 없을 경우 텍스트로 표시
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

                    # 고객 특성 시각화
                    if summary.get("customer_characteristics") or summary.get(
                            "privacy_info"):
                        characteristics_chart = visualize_customer_characteristics(
                            summary, current_lang)
                        if characteristics_chart:
                            st.plotly_chart(
                                characteristics_chart, use_container_width=True)
                else:
                    # 기존 메시지가 있는 경우 그대로 사용
                    st.session_state.simulator_messages = h.get("messages", [])

                st.session_state.initial_advice_provided = True
                st.session_state.is_chat_ended = h.get("is_chat_ended", False)
                st.session_state.sim_attachment_context_for_llm = h.get(
                    "attachment_context", "")  # 컨텍스트 로드
                st.session_state.customer_attachment_file = []  # 로드된 이력에는 파일 객체 대신 컨텍스트 문자열만 사용
                st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화

                # 상태 복원
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

                st.session_state.simulator_memory.clear()  # 메모리 초기화
        else:
            st.info(L["no_history_found"])

    # =========================
    # AHT 타이머 (화면 최상단)
    # =========================
    if st.session_state.sim_stage not in [
            "WAIT_FIRST_QUERY", "CLOSING", "idle"]:
        elapsed_placeholder = st.empty()

        if st.session_state.start_time is not None:
            # 실시간 업데이트를 위해 페이지 로드 시마다 현재 시간 계산
            elapsed_time = datetime.now() - st.session_state.start_time
            total_seconds = elapsed_time.total_seconds()

            # Hold 시간 제외 (채팅/이메일은 Hold 없음, 전화 탭과 로직 통일 위해 유지)
            # total_seconds -= st.session_state.total_hold_duration.total_seconds()

            # 시간 형식 포맷팅
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            # 경고 기준
            if total_seconds > 900:  # 15분
                delta_str = L["timer_info_risk"]
                delta_color = "inverse"
            elif total_seconds > 600:  # 10분
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

            # ⭐ 수정: 3초마다 재실행하여 AHT 실시간성 확보
            if seconds % 3 == 0 and total_seconds < 1000:
                time.sleep(1)

        st.markdown("---")

    # =========================
    # 2. LLM 준비 체크 & 채팅 종료 상태
    # =========================
    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])

    if st.session_state.sim_stage == "CLOSING":
        st.success(L["survey_sent_confirm"])
        st.info(L["new_simulation_ready"])

        # ⭐ 추가: 현재 세션 이력 다운로드 기능 - JSON/CSV 추가
        st.markdown("---")
        st.markdown("**📥 현재 세션 이력 다운로드**")
        download_col1, download_col2, download_col3, download_col4, download_col5 = st.columns(
            5)

        # 현재 세션의 이력을 생성
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
                    "attachment_context": st.session_state.sim_attachment_context_for_llm
                }]
            except Exception as e:
                st.warning(
                    L.get(
                        "history_generation_error",
                        "이력 생성 중 오류 발생: {error}").format(
                        error=e))

        # 다운로드 버튼들을 직접 표시
        if current_session_history:
            # 현재 언어 가져오기
            current_lang = st.session_state.get("language", "ko")
            if current_lang not in ["ko", "en", "ja"]:
                current_lang = "ko"

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

            # ⭐ 추가: JSON 다운로드
            with download_col4:
                try:
                    import json
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

            # ⭐ 추가: CSV 다운로드
            with download_col5:
                try:
                    import csv
                    import io
                    output = io.StringIO()
                    writer = csv.writer(output)

                    # 헤더
                    writer.writerow(["Role", "Content", "Timestamp"])

                    # 메시지 데이터
                    for msg in current_session_history[0].get("messages", []):
                        writer.writerow([
                            msg.get("role", ""),
                            msg.get("content", ""),
                            current_session_history[0].get("timestamp", "")
                        ])

                    csv_data = output.getvalue()
                    st.download_button(
                        label=L.get("download_history_csv", "📥 이력 다운로드 (CSV)"),
                        # BOM 추가로 Excel 호환성 향상
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
            # 초기화 로직
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.initial_advice_provided = False
            st.session_state.is_chat_ended = False
            st.session_state.agent_response_area_text = ""
            st.session_state.customer_query_text_area = ""
            st.session_state.last_transcript = ""
            st.session_state.sim_audio_bytes = None
            st.session_state.sim_stage = "WAIT_FIRST_QUERY"
            st.session_state.customer_attachment_file = []  # 첨부 파일 초기화
            st.session_state.sim_attachment_context_for_llm = ""  # 컨텍스트 초기화
            st.session_state.agent_attachment_file = []  # 에이전트 첨부 파일 초기화
            st.session_state.start_time = None
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None
        # st.stop()

    # =========================
    # 5-A. 전화 발신 진행 중 (OUTBOUND_CALL_IN_PROGRESS)
    # =========================
    elif st.session_state.sim_stage == "OUTBOUND_CALL_IN_PROGRESS":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        target = st.session_state.get("sim_call_outbound_target", "대상")
        st.warning(L["call_outbound_loading"])

        # LLM 호출 및 요약 생성
        with st.spinner(L["call_outbound_loading"]):
            # 1. LLM 호출하여 통화 요약 생성
            summary = generate_outbound_call_summary(
                st.session_state.customer_query_text_area,
                st.session_state.language,
                target
            )

            # 2. 시스템 메시지 (전화 시도) 추가
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": L["call_outbound_system_msg"].format(target=target)}
            )

            # 3. 요약 메시지 (결과) 추가
            summary_markdown = f"### {L['call_outbound_summary_header']}\n\n{summary}"
            st.session_state.simulator_messages.append(
                {"role": "supervisor", "content": summary_markdown}
            )

            # 4. Agent Turn으로 복귀
            st.session_state.sim_stage = "AGENT_TURN"
            # Save for display/reference
            st.session_state.sim_call_outbound_summary = summary_markdown
            st.session_state.sim_call_outbound_target = None  # Reset target

            # 5. 이력 저장 (전화 발신 후 상태 저장)
            customer_type_display = st.session_state.get(
                "customer_type_sim_select", "")
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display +
                f" (Outbound Call to {target})",
                st.session_state.simulator_messages,
                is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.success(
            f"✅ {L['call_outbound_simulation_header']}가 완료되었습니다. 요약을 확인하고 고객에게 회신하세요.")

# ========================================
    # 3. 초기 문의 입력 (WAIT_FIRST_QUERY) - app.py 스타일: 바로 시작
# ========================================
    if st.session_state.sim_stage == "WAIT_FIRST_QUERY":
        # admin.py 스타일: 깔끔한 레이아웃
        customer_query = st.text_area(
            L["customer_query_label"],
            key="customer_query_text_area",
            height=150,
            placeholder=L["initial_query_sample"],
        )

        st.divider()

        # 필수 입력 필드 (admin.py 스타일: 간단한 컬럼 구조)
        col_email, col_phone = st.columns(2)
        with col_email:
            customer_email = st.text_input(
                L["customer_email_label"],
                key="customer_email_input",
                value=st.session_state.customer_email,
            )
        with col_phone:
            customer_phone = st.text_input(
                L["customer_phone_label"],
                key="customer_phone_input",
                value=st.session_state.customer_phone,
            )
        # 세션 상태 업데이트
        st.session_state.customer_email = customer_email
        st.session_state.customer_phone = customer_phone

        # 고객 유형 선택 (admin.py 스타일: 간단한 레이아웃)
        customer_type_options = L["customer_type_options"]
        default_idx = customer_type_options.index(
            st.session_state.customer_type_sim_select) if st.session_state.customer_type_sim_select in customer_type_options else 0

        st.session_state.customer_type_sim_select = st.selectbox(
            L["customer_type_label"],
            customer_type_options,
            index=default_idx,
            key="customer_type_sim_select_widget",
        )

        # ⭐ 수정: 고객 파일 업로드 기능 제거 (채팅/이메일 탭에서)
        # 첨부 파일 관련 상태 초기화
        st.session_state.customer_attachment_file = None
        st.session_state.sim_attachment_context_for_llm = ""

        st.divider()

        # ⭐ 수정: app.py 스타일로 바로 시작 (중복 기능 제거)
        # 채팅 시작 버튼 (간단한 버튼, "응대 조언 요청" 중복 기능 제거)
        if st.button(
                L.get(
                    "button_start_chat",
                    "채팅 시작"),
                key=f"btn_start_chat_{st.session_state.sim_instance_id}",
                use_container_width=True,
                type="primary"):
            if not customer_query.strip():
                st.warning(L["simulation_warning_query"])
                # st.stop()

            # --- 필수 입력 필드 검증 (요청 3 반영: 검증 로직 추가) ---
            if not st.session_state.customer_email.strip(
            ) or not st.session_state.customer_phone.strip():
                st.error(L["error_mandatory_contact"])
                # st.stop()
            # ------------------------------------------

            # 초기 상태 리셋
            st.session_state.simulator_messages = []
            st.session_state.simulator_memory.clear()
            st.session_state.is_chat_ended = False
            st.session_state.initial_advice_provided = False
            st.session_state.is_solution_provided = False  # 솔루션 플래그 리셋
            st.session_state.language_transfer_requested = False  # 언어 요청 플래그 리셋
            st.session_state.transfer_summary_text = ""  # 이관 요약 리셋
            st.session_state.start_time = None  # AHT 타이머 초기화 (첫 고객 반응 후 시작)
            st.session_state.sim_instance_id = str(
                uuid.uuid4())  # 새 시뮬레이션 ID 할당

            # ⭐ 추가: UI 플래그 초기화 (채팅 시작 시 모든 기능 UI 숨김, 에이전트 응답 입력만 표시)
            st.session_state.show_verification_ui = False
            st.session_state.show_draft_ui = False
            st.session_state.show_customer_data_ui = False
            st.session_state.show_agent_response_ui = False

            # 고객 검증 상태 초기화 (로그인/계정 관련 문의인 경우)
            is_login_inquiry = check_if_login_related_inquiry(customer_query)
            if is_login_inquiry:
                # 검증 정보 초기화 및 고객이 제공한 정보를 시스템 검증 정보로 저장 (시뮬레이션용)
                # 실제로는 DB에서 가져와야 하지만, 시뮬레이션에서는 고객이 제공한 정보를 저장
                st.session_state.is_customer_verified = False
                st.session_state.verification_stage = "WAIT_VERIFICATION"

                # ⭐ 수정: 고객 파일 업로드 기능 제거로 인해 첨부 파일 정보 없음
                file_info_for_storage = None

                st.session_state.verification_info = {
                    "receipt_number": "",  # 실제로는 DB에서 가져와야 함
                    "card_last4": "",  # 실제로는 DB에서 가져와야 함
                    "customer_name": "",  # 실제로는 DB에서 가져와야 함
                    "customer_email": st.session_state.customer_email,  # 고객이 제공한 정보
                    "customer_phone": st.session_state.customer_phone,  # 고객이 제공한 정보
                    "file_uploaded": False,  # 채팅/이메일 탭에서는 파일 업로드 기능 제거
                    "file_info": None,  # 첨부 파일 상세 정보 없음
                    "verification_attempts": 0
                }
            else:
                # 로그인 관련 문의가 아닌 경우 검증 불필요
                st.session_state.is_customer_verified = True
                st.session_state.verification_stage = "NOT_REQUIRED"
            # 전화 발신 관련 상태 초기화
            st.session_state.sim_call_outbound_summary = ""
            st.session_state.sim_call_outbound_target = None

            # 1) 고객 첫 메시지 추가
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_query}
            )

            # 2) Supervisor 가이드 + 초안 생성
            # 입력 텍스트의 언어를 자동 감지 (오류 발생 시 안전하게 처리)
            try:
                detected_lang = detect_text_language(customer_query)
                # 감지된 언어가 유효한지 확인
                if detected_lang not in ["ko", "en", "ja"]:
                    detected_lang = current_lang
                else:
                    # 언어가 감지되었고 현재 언어와 다르면 자동으로 언어 설정 업데이트
                    if detected_lang != current_lang:
                        st.session_state.language = detected_lang
                        st.info(
                            f"🌐 입력 언어가 감지되어 언어 설정이 '{detected_lang}'로 자동 변경되었습니다.")
            except Exception as e:
                print(f"Language detection failed: {e}")
                detected_lang = current_lang  # 기본값으로 폴백

            # 고객 프로필 분석 (시각화를 위해 먼저 수행, 감지된 언어 사용)
            customer_profile = analyze_customer_profile(
                customer_query, detected_lang)
            similar_cases = find_similar_cases(
                customer_query, customer_profile, detected_lang, limit=5)

            # 시각화 차트 표시
            st.markdown("---")
            st.subheader("📊 고객 프로필 분석")

            # 고객 프로필 점수 차트 (감지된 언어 사용)
            profile_chart = visualize_customer_profile_scores(
                customer_profile, detected_lang)
            if profile_chart:
                st.plotly_chart(profile_chart, use_container_width=True)
            else:
                # Plotly가 없을 경우 텍스트로 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    gender_display = customer_profile.get("gender", "unknown")
                    if gender_display == "male":
                        gender_display = "남자"
                    elif gender_display == "female":
                        gender_display = "여자"
                    else:
                        gender_display = "알 수 없음"
                    st.metric(
                        "성별",
                        gender_display
                    )
                with col2:
                    st.metric(
                        L.get("sentiment_score_label", "감정 점수"),
                        f"{customer_profile.get('sentiment_score', 50)}/100"
                    )
                with col3:
                    urgency_map = {"low": 25, "medium": 50, "high": 75}
                    urgency_score = urgency_map.get(
                        customer_profile.get(
                            "urgency_level", "medium").lower(), 50)
                    st.metric(
                        L.get("urgency_score_label", "긴급도"),
                        f"{urgency_score}/100"
                    )
                with col4:
                    st.metric(
                        L.get(
                            "customer_type_label", "고객 유형"), customer_profile.get(
                            "predicted_customer_type", "normal"))

            # 유사 케이스 시각화
            if similar_cases:
                st.markdown("---")
                st.subheader("🔍 유사 케이스 추천")
                similarity_chart = visualize_similarity_cases(
                    similar_cases, detected_lang)
                if similarity_chart:
                    st.plotly_chart(similarity_chart, use_container_width=True)

                # 유사 케이스 요약 표시
                with st.expander(f"💡 {len(similar_cases)}개 유사 케이스 상세 정보"):
                    for idx, similar_case in enumerate(similar_cases, 1):
                        case = similar_case["case"]
                        summary = similar_case["summary"]
                        similarity = similar_case["similarity_score"]
                        st.markdown(f"### 케이스 {idx} (유사도: {similarity:.1f}%)")
                        st.markdown(
                            f"**문의 내용:** {summary.get('main_inquiry', 'N/A')}")
                        st.markdown(
                            f"**감정 점수:** {summary.get('customer_sentiment_score', 50)}/100")
                        st.markdown(
                            f"**만족도 점수:** {summary.get('customer_satisfaction_score', 50)}/100")
                        if summary.get("key_responses"):
                            st.markdown("**핵심 응답:**")
                            for response in summary.get(
                                    "key_responses", [])[:3]:
                                st.markdown(f"- {response[:100]}...")
                        st.markdown("---")

            # ⭐ 수정: 자동으로 응대 가이드라인/초안 생성하지 않음 (버튼 클릭 시에만 생성)
            # 초기 조언은 버튼을 통해 수동으로 생성하도록 변경
            # st.session_state.initial_advice_provided는 버튼 클릭 시 설정됨
            st.session_state.initial_advice_provided = False

            # ⭐ 수정: AGENT_TURN으로 자동 변경하지 않음 (응대 가이드라인 버튼 클릭 시에만 변경)
            # 채팅 시작 후 고객 메시지가 표시되고, 버튼을 통해 기능 사용 가능
            save_simulation_history_local(
                customer_query,
                st.session_state.customer_type_sim_select,
                st.session_state.simulator_messages,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
                is_chat_ended=False,
            )
            # sim_stage는 CUSTOMER_TURN으로 유지 (에이전트 응답 UI는 버튼 클릭 시에만 표시)
            st.session_state.sim_stage = "CUSTOMER_TURN"

    # =========================
    # 4. 대화 로그 표시 (공통)
    # =========================

    # 피드백 저장 콜백 함수
    def save_feedback(index):
        # 에이전트 응답에 대한 고객 피드백을 저장
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{index}"
        if feedback_key in st.session_state:
            feedback_value = st.session_state[feedback_key]
            # 메시지에 피드백 정보 저장
            if index < len(st.session_state.simulator_messages):
                st.session_state.simulator_messages[index]["feedback"] = feedback_value

    # 채팅/이메일 탭에서만 메시지 표시
    # ⭐ app.py 스타일로 간소화: 깔끔한 채팅 UI
    if st.session_state.simulator_messages:
        for idx, msg in enumerate(st.session_state.simulator_messages):
            role = msg["role"]
            content = msg["content"]

            # 역할에 따른 표시 이름 및 아바타 설정
            if role == "customer" or role == "customer_rebuttal" or role == "initial_query":
                display_role = "user"
                avatar = "🙋"
            elif role == "agent_response":
                display_role = "assistant"
                avatar = "🧑‍💻"
            elif role == "supervisor":
                display_role = "assistant"
                avatar = "🤖"
            else:
                display_role = "assistant"
                avatar = "💬"

            with st.chat_message(display_role, avatar=avatar):
                st.write(content)

                # ⭐ 가이드라인 메시지는 메시지로만 표시 (에이전트 응답 UI는 AGENT_TURN 섹션에서 항상 표시)
                # 가이드라인 메시지 아래의 UI는 제거됨

                # ⭐ 메시지 말풍선 안에 버튼들 추가 (영상 스타일)
                # 버튼 레이아웃: 역할에 따라 다른 버튼 표시

                # 1. 음성으로 듣기 버튼 (모든 메시지에)
                tts_role = "customer" if role.startswith("customer") or role == "customer_rebuttal" else (
                    "agent" if role == "agent_response" else "supervisor")
                render_tts_button(
                    content,
                    st.session_state.language,
                    role=tts_role,
                    prefix=f"{role}_",
                    index=idx)

                # 2. 에이전트 응답에 피드백 버튼만 표시 (응대 힌트, 전화 버튼은 입력 칸으로 이동)
                if role == "agent_response":
                    # 피드백 버튼 (기존 유지)
                    feedback_key = f"feedback_{st.session_state.sim_instance_id}_{idx}"
                    existing_feedback = msg.get("feedback", None)
                    if existing_feedback is not None:
                        st.session_state[feedback_key] = existing_feedback

                    st.feedback(
                        "thumbs",
                        key=feedback_key,
                        disabled=existing_feedback is not None,
                        on_change=save_feedback,
                        args=[idx],
                    )

                # 3. 고객 메시지에 응대 힌트, 전화 버튼 및 추가 기능 버튼들
                if role == "customer" or role == "customer_rebuttal":
                    # 첫 번째 행: 응대 힌트, 전화 버튼들 (admin.py 스타일: 간단한 컬럼 구조)
                    button_cols_customer_row1 = st.columns(3)

                    # 응대 힌트 버튼
                    with button_cols_customer_row1[0]:
                        if st.button(
                                L.get(
                                    "button_hint",
                                    "💡 응대 힌트"),
                                key=f"hint_btn_customer_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            if st.session_state.is_llm_ready:
                                # 다른 플래그들 초기화 (하나만 보이도록)
                                st.session_state.show_verification_ui = False
                                st.session_state.show_draft_ui = False
                                st.session_state.show_customer_data_ui = False
                                st.session_state.show_agent_response_ui = False

                                # ⭐ 수정: 이전 힌트 메시지 제거 (같은 타입의 supervisor 메시지 제거)
                                hint_label = L.get('hint_label', '응대 힌트')
                                st.session_state.simulator_messages = [
                                    msg for msg in st.session_state.simulator_messages if not (
                                        msg.get("role") == "supervisor" and hint_label in msg.get(
                                            "content", ""))]

                                # ⭐ 수정: 세션 언어 설정을 직접 전달
                                session_lang = st.session_state.get(
                                    "language", "ko")
                                if session_lang not in ["ko", "en", "ja"]:
                                    session_lang = "ko"

                                with st.spinner(L.get("response_generating", "생성 중...")):
                                    hint = generate_realtime_hint(
                                        session_lang, is_call=False)
                                    st.session_state.realtime_hint_text = hint
                                    # 힌트를 supervisor 메시지로 추가하여 표시
                                    st.session_state.simulator_messages.append({
                                        "role": "supervisor",
                                        "content": f"💡 **{L.get('hint_label', '응대 힌트')}**: {hint}"
                                    })
                            else:
                                st.warning(
                                    L.get(
                                        "simulation_no_key_warning",
                                        "LLM이 준비되지 않았습니다."))

                    # 업체에 전화 버튼
                    with button_cols_customer_row1[1]:
                        if st.button(
                                L.get(
                                    "button_call_company",
                                    "📞 업체에 전화"),
                                key=f"call_provider_customer_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            # 다른 플래그들 초기화
                            st.session_state.show_verification_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = False
                            st.session_state.show_agent_response_ui = False
                            st.session_state.sim_call_outbound_target = L.get(
                                "call_target_provider", "현지 업체/파트너")
                            st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                            # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                            # st.rerun()

                    # 고객에게 전화 버튼
                    with button_cols_customer_row1[2]:
                        if st.button(
                                L.get(
                                    "button_call_customer",
                                    "📞 고객에게 전화"),
                                key=f"call_customer_customer_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            # 다른 플래그들 초기화
                            st.session_state.show_verification_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = False
                            st.session_state.show_agent_response_ui = False
                            st.session_state.sim_call_outbound_target = L.get(
                                "call_target_customer", "고객")
                            st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"
                            # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                            # st.rerun()

                    # 두 번째 행: AI 응대 가이드라인, 응대 초안, 고객 검증 버튼들
                    button_cols_customer_row2 = st.columns(4)

                    # AI 응대 가이드라인 버튼 (에이전트 응답 UI 포함)
                    with button_cols_customer_row2[0]:
                        if st.button(
                                L.get(
                                    "button_ai_guideline",
                                    "📋 AI 응대 가이드라인"),
                                key=f"guideline_btn_customer_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            if st.session_state.is_llm_ready:
                                # 다른 플래그들 초기화 (하나만 보이도록)
                                st.session_state.show_verification_ui = False
                                st.session_state.show_draft_ui = False
                                st.session_state.show_customer_data_ui = False
                                st.session_state.show_agent_response_ui = False  # 가이드라인은 메시지만 표시

                                # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                                guideline_label = L.get(
                                    'guideline_label', 'AI 응대 가이드라인')
                                draft_label = L.get('draft_label', '응대 초안')
                                customer_data_label = L.get(
                                    'customer_data_label', '고객 데이터')
                                customer_data_loaded = L.get(
                                    'customer_data_loaded', '고객 데이터 불러옴')
                                st.session_state.simulator_messages = [
                                    msg for msg in st.session_state.simulator_messages if not (
                                        msg.get("role") == "supervisor" and (
                                            guideline_label in msg.get(
                                                "content",
                                                "") or draft_label in msg.get(
                                                "content",
                                                "") or customer_data_label in msg.get(
                                                "content",
                                                "") or customer_data_loaded in msg.get(
                                                "content",
                                                "")))]

                                with st.spinner(L.get("generating_guideline", "AI 응대 가이드라인 생성 중...")):
                                    # 초기 문의 가져오기
                                    initial_query = st.session_state.get(
                                        'customer_query_text_area', content)
                                    customer_type_display = st.session_state.get(
                                        "customer_type_sim_select", "")

                                    # ⭐ 수정: 세션 언어 설정을 직접 전달
                                    session_lang = st.session_state.get(
                                        "language", "ko")
                                    if session_lang not in ["ko", "en", "ja"]:
                                        session_lang = "ko"

                                    # 응대 가이드라인 생성
                                    guideline_text = _generate_initial_advice(
                                        initial_query,
                                        customer_type_display,
                                        st.session_state.customer_email,
                                        st.session_state.customer_phone,
                                        session_lang,
                                        st.session_state.customer_attachment_file
                                    )

                                    # 가이드라인을 supervisor 메시지로 추가하여 표시
                                    st.session_state.simulator_messages.append({
                                        "role": "supervisor",
                                        "content": f"📋 **{L.get('guideline_label', 'AI 응대 가이드라인')}**:\n\n{guideline_text}"
                                    })

                                    # AGENT_TURN 단계로 변경하여 에이전트 응답 UI 표시 (항상
                                    # 표시됨)
                                    st.session_state.sim_stage = "AGENT_TURN"
                            else:
                                st.warning(
                                    L.get(
                                        "simulation_no_key_warning",
                                        "LLM이 준비되지 않았습니다."))

                    # 고객 데이터 가져오기 버튼 (app.py 스타일)
                    with button_cols_customer_row2[1]:
                        if st.button(
                                L.get(
                                    "button_customer_data",
                                    "📋 고객 데이터"),
                                key=f"customer_data_btn_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            # 다른 플래그들 초기화 (하나만 보이도록)
                            st.session_state.show_agent_response_ui = False
                            st.session_state.show_verification_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = True

                            # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                            guideline_label = L.get(
                                'guideline_label', 'AI 응대 가이드라인')
                            draft_label = L.get('draft_label', '응대 초안')
                            customer_data_label = L.get(
                                'customer_data_label', '고객 데이터')
                            customer_data_loaded = L.get(
                                'customer_data_loaded', '고객 데이터 불러옴')
                            st.session_state.simulator_messages = [
                                msg for msg in st.session_state.simulator_messages if not (
                                    msg.get("role") == "supervisor" and (
                                        guideline_label in msg.get(
                                            "content",
                                            "") or draft_label in msg.get(
                                            "content",
                                            "") or customer_data_label in msg.get(
                                            "content",
                                            "") or customer_data_loaded in msg.get(
                                            "content",
                                            "")))]

                            # 고객 ID는 이메일 또는 전화번호 기반으로 생성
                            customer_id = st.session_state.get(
                                "customer_email", "") or st.session_state.get(
                                "customer_phone", "")
                            if not customer_id:
                                customer_id = f"customer_{st.session_state.sim_instance_id}"

                            # 고객 데이터 불러오기
                            customer_data = st.session_state.customer_data_manager.load_customer_data(
                                customer_id)

                            # ⭐ 추가: 누적 데이터 수 자동 확인
                            try:
                                all_customers = st.session_state.customer_data_manager.list_all_customers()
                                total_customers = len(all_customers)
                            except Exception:
                                total_customers = 0

                            if customer_data:
                                st.session_state.customer_data = customer_data
                                customer_info = customer_data.get("data", {})

                                # 고객 데이터를 supervisor 메시지로 추가하여 표시
                                info_message = f"📋 **{L.get('customer_data_loaded', '고객 데이터 불러옴')}**\n\n"
                                info_message += f"**{L.get('basic_info_label', '기본 정보')}:**\n"
                                info_message += f"- {L.get('name_label', '이름')}: {customer_info.get('name', 'N/A')}\n"
                                info_message += f"- {L.get('email_label', '이메일')}: {customer_info.get('email', 'N/A')}\n"
                                info_message += f"- {L.get('phone_label', '전화번호')}: {customer_info.get('phone', 'N/A')}\n"
                                info_message += f"- {L.get('company_label', '회사')}: {customer_info.get('company', 'N/A')}\n"

                                # 누적 데이터 수 표시
                                info_message += f"\n**{L.get('accumulated_data_label', '누적 데이터')}:**\n"
                                info_message += f"- {L.get('total_customers_label', '총 고객 수')}: {total_customers}{L.get('cases_label', '건')}\n"

                                if customer_info.get('purchase_history'):
                                    info_message += f"\n**{L.get('purchase_history_label', '구매 이력')}:** ({len(customer_info.get('purchase_history', []))}{L.get('cases_label', '건')})\n"
                                    for purchase in customer_info.get(
                                            'purchase_history', [])[:5]:
                                        info_message += f"- {purchase.get('date', 'N/A')}: {purchase.get('item', 'N/A')} ({purchase.get('amount', 0):,}{L.get('currency_unit', '원')})\n"
                                if customer_info.get('notes'):
                                    info_message += f"\n**{L.get('notes_label', '메모')}:** {customer_info.get('notes', 'N/A')}"

                                st.session_state.simulator_messages.append({
                                    "role": "supervisor",
                                    "content": info_message
                                })
                            else:
                                # 고객 데이터가 없으면 안내 메시지 (누적 데이터 수 포함)
                                info_message = f"📋 **{L.get('customer_data_label', '고객 데이터')}**: {L.get('no_customer_data', '저장된 고객 데이터가 없습니다.')}\n\n"
                                info_message += f"**{L.get('accumulated_data_label', '누적 데이터')}**: {L.get('total_label', '총')} {total_customers}{L.get('cases_label', '건')}"
                                st.session_state.simulator_messages.append({
                                    "role": "supervisor",
                                    "content": info_message
                                })

                    # 응대 초안 버튼
                    with button_cols_customer_row2[2]:
                        if st.button(
                                L.get(
                                    "button_draft",
                                    "✍️ 응대 초안"),
                                key=f"draft_btn_customer_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            if st.session_state.is_llm_ready:
                                # 다른 플래그들 초기화 (하나만 보이도록)
                                st.session_state.show_agent_response_ui = False
                                st.session_state.show_verification_ui = False
                                st.session_state.show_customer_data_ui = False
                                st.session_state.show_draft_ui = True

                                # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                                guideline_label = L.get(
                                    'guideline_label', 'AI 응대 가이드라인')
                                draft_label = L.get('draft_label', '응대 초안')
                                customer_data_label = L.get(
                                    'customer_data_label', '고객 데이터')
                                customer_data_loaded = L.get(
                                    'customer_data_loaded', '고객 데이터 불러옴')
                                st.session_state.simulator_messages = [
                                    msg for msg in st.session_state.simulator_messages if not (
                                        msg.get("role") == "supervisor" and (
                                            guideline_label in msg.get(
                                                "content",
                                                "") or draft_label in msg.get(
                                                "content",
                                                "") or customer_data_label in msg.get(
                                                "content",
                                                "") or customer_data_loaded in msg.get(
                                                "content",
                                                "")))]

                                with st.spinner(L.get("generating_draft", "응대 초안 생성 중...")):
                                    # 초기 문의 가져오기
                                    initial_query = st.session_state.get(
                                        'customer_query_text_area', content)
                                    customer_type_display = st.session_state.get(
                                        "customer_type_sim_select", "")

                                    # ⭐ 수정: 세션 언어 설정을 직접 전달
                                    session_lang = st.session_state.get(
                                        "language", "ko")
                                    if session_lang not in ["ko", "en", "ja"]:
                                        session_lang = "ko"

                                    # 응대 초안 생성 (가이드라인과 동일한 함수 사용)
                                    draft_text = _generate_initial_advice(
                                        initial_query,
                                        customer_type_display,
                                        st.session_state.customer_email,
                                        st.session_state.customer_phone,
                                        session_lang,
                                        st.session_state.customer_attachment_file
                                    )

                                    # 초안을 supervisor 메시지로 추가하여 표시
                                    st.session_state.simulator_messages.append({
                                        "role": "supervisor",
                                        "content": f"✍️ **{L.get('draft_label', '응대 초안')}**:\n\n{draft_text}"
                                    })
                            else:
                                st.warning(
                                    L.get(
                                        "simulation_no_key_warning",
                                        "LLM이 준비되지 않았습니다."))

                    # 고객 검증 버튼 (검증 전 제한 사항 포함)
                    with button_cols_customer_row2[3]:
                        if st.button(
                                L.get(
                                    "button_verification",
                                    "🔐 고객 검증"),
                                key=f"verification_btn_customer_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            # 다른 플래그들 초기화 (하나만 보이도록)
                            st.session_state.show_agent_response_ui = False
                            st.session_state.show_draft_ui = False
                            st.session_state.show_customer_data_ui = False
                            # 검증 UI 표시를 위한 플래그 설정
                            st.session_state.show_verification_ui = True
                            st.session_state.verification_message_idx = idx

                            # ⭐ 수정: 이전 supervisor 메시지 모두 제거 (가이드라인, 초안, 고객 데이터 등)
                            guideline_label = L.get(
                                'guideline_label', 'AI 응대 가이드라인')
                            draft_label = L.get('draft_label', '응대 초안')
                            customer_data_label = L.get(
                                'customer_data_label', '고객 데이터')
                            customer_data_loaded = L.get(
                                'customer_data_loaded', '고객 데이터 불러옴')
                            st.session_state.simulator_messages = [
                                msg for msg in st.session_state.simulator_messages if not (
                                    msg.get("role") == "supervisor" and (
                                        guideline_label in msg.get(
                                            "content",
                                            "") or draft_label in msg.get(
                                            "content",
                                            "") or customer_data_label in msg.get(
                                            "content",
                                            "") or customer_data_loaded in msg.get(
                                            "content",
                                            "")))]

                            st.session_state.sim_stage = "AGENT_TURN"  # 검증 UI를 표시하기 위해 AGENT_TURN으로 변경

                    # 마지막 에이전트 응답에서 솔루션이 제공되었는지 확인
                    last_agent_response_idx = None
                    for i in range(idx - 1, -1, -1):
                        if i < len(st.session_state.simulator_messages) and st.session_state.simulator_messages[i].get(
                                "role") == "agent_response":
                            last_agent_response_idx = i
                            break

                    # 솔루션 제공 여부 확인
                    solution_provided = False
                    if last_agent_response_idx is not None:
                        agent_msg_content = st.session_state.simulator_messages[last_agent_response_idx].get(
                            "content", "")
                        solution_keywords = [
                            "해결",
                            "도움",
                            "안내",
                            "제공",
                            "solution",
                            "help",
                            "assist",
                            "guide",
                            "안내해드리",
                            "도와드리"]
                        solution_provided = any(
                            keyword in agent_msg_content.lower() for keyword in solution_keywords)

                    # "알겠습니다" 또는 "감사합니다"가 포함된 경우 추가 문의 여부 확인 버튼 표시 (admin.py 스타일)
                    if solution_provided or st.session_state.is_solution_provided:
                        if "알겠습니다" in content or "감사합니다" in content or "ok" in content.lower(
                        ) or "thank" in content.lower():
                            if st.button(
                                    L.get(
                                        "button_additional_inquiry",
                                        "✅ 추가 문의 있나요?"),
                                    key=f"additional_inquiry_{idx}_{st.session_state.sim_instance_id}",
                                    use_container_width=True,
                                    type="secondary"):
                                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"

                    # 4. 고객이 "없습니다. 감사합니다" 답변 시 설문 조사 버튼 (admin.py 스타일)
                    no_more_keywords = [
                        "없습니다",
                        "감사합니다",
                        "No, that will be all",
                        "no more",
                        "추가 문의 사항 없습니다",
                        "추가 문의사항 없습니다",
                        "no additional",
                        "結構です"]
                    # 키워드가 모두 포함되어 있거나 "없습니다"와 "감사합니다"가 함께 있는 경우
                    has_no_more = (
                        any(keyword in content for keyword in no_more_keywords) or
                        ("없습니다" in content and "감사합니다" in content) or
                        ("no" in content.lower() and "more" in content.lower() and "thank" in content.lower())
                    )

                    if has_no_more:
                        if st.button(
                                L.get(
                                    "button_survey_end",
                                    "📋 설문 조사 전송 및 종료"),
                                key=f"survey_end_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="primary"):
                            # AHT 타이머 정지
                            st.session_state.start_time = None

                            # 설문 조사 링크 전송 메시지 추가
                            end_msg = L.get(
                                "prompt_survey", "설문 조사 링크를 전송했습니다.")
                            st.session_state.simulator_messages.append(
                                {"role": "system_end", "content": end_msg}
                            )

                            # 채팅 종료 처리
                            customer_type_display = st.session_state.get(
                                "customer_type_sim_select", "")
                            st.session_state.is_chat_ended = True
                            st.session_state.sim_stage = "CLOSING"

                            # 이력 저장
                            save_simulation_history_local(
                                st.session_state.customer_query_text_area,
                                customer_type_display,
                                st.session_state.simulator_messages,
                                is_chat_ended=True,
                                attachment_context=st.session_state.sim_attachment_context_for_llm,
                            )

                            # ⭐ 재실행 불필요: 이력 저장만으로 충분, 자동 업데이트됨
                            # st.rerun()

                # 고객 첨부 파일 표시 (기능 유지)
                if idx == 0 and role == "customer" and st.session_state.customer_attachment_b64:
                    mime = st.session_state.customer_attachment_mime or "image/png"
                    data_url = f"data:{mime};base64,{st.session_state.customer_attachment_b64}"

                    if mime.startswith("image/"):
                        caption_text = L.get(
                            "attachment_evidence_caption", "첨부된 증거물").format(
                            filename=st.session_state.customer_attachment_file.name)
                        st.image(
                            data_url,
                            caption=caption_text,
                            use_column_width=True)
                    elif mime == "application/pdf":
                        warning_text = L.get(
                            "attachment_pdf_warning",
                            "첨부된 PDF 파일 ({filename})은 현재 인라인 미리보기가 지원되지 않습니다.").format(
                            filename=st.session_state.customer_attachment_file.name)
                        st.warning(warning_text)

    # 이관 요약 표시 (이관 후에만) - ⭐ 수정: AI 응대 가이드라인 위에서는 표시하지 않음
    # AGENT_TURN 단계가 아니거나, 가이드라인/초안/고객데이터 UI가 표시되지 않을 때만 표시
    show_guideline_ui = st.session_state.get(
        "show_draft_ui", False) or st.session_state.get(
        "show_customer_data_ui", False)
    should_show_transfer_summary = (
        (st.session_state.transfer_summary_text or (st.session_state.language != st.session_state.language_at_transfer_start and st.session_state.language_at_transfer_start)) and
        st.session_state.sim_stage != "AGENT_TURN" and not show_guideline_ui
    )
    if should_show_transfer_summary:
        st.markdown("---")
        st.markdown(f"**{L['transfer_summary_header']}**")
        st.info(L["transfer_summary_intro"])

        # ⭐ [수정] 번역 성공 여부 확인 및 요약 표시
        is_translation_failed = not st.session_state.get(
            "translation_success", True) or not st.session_state.transfer_summary_text

        # 번역 성공 시 요약 표시
        if st.session_state.transfer_summary_text and st.session_state.get(
                "translation_success", True):
            st.markdown(st.session_state.transfer_summary_text)

        # 번역 실패 시 처리
        if is_translation_failed:
            # 번역 실패 시에도 원본 텍스트가 표시되므로 오류 메시지 없이 원본 텍스트만 표시
            # (오류 메시지를 표시하지 않아도 원본 텍스트로 계속 진행 가능)
            if st.session_state.transfer_summary_text:
                st.info(st.session_state.transfer_summary_text)
            # 번역 재시도 버튼 추가 (선택적)
            if st.button(
                    L.get(
                        "button_retry_translation",
                        "번역 다시 시도"),
                    key=f"btn_retry_translation_{st.session_state.sim_instance_id}"):  # 고유 키 사용
                # 재시도 로직 실행
                try:
                    source_lang = st.session_state.language_at_transfer_start
                    target_lang = st.session_state.language

                    if not source_lang or not target_lang:
                        st.error(
                            L.get(
                                "invalid_language_info",
                                "언어 정보가 올바르지 않습니다."))
                    else:
                        # 이전 대화 내용 재가공
                        history_text = ""
                        for msg in st.session_state.simulator_messages:
                            role = "Customer" if msg["role"].startswith(
                                "customer") or msg["role"] == "initial_query" else "Agent"
                            if msg["role"] in [
                                "initial_query",
                                "customer_rebuttal",
                                "agent_response",
                                    "customer_closing_response"]:
                                content = msg.get("content", "").strip()
                                if content:
                                    history_text += f"{role}: {content}\n"

                        if not history_text.strip():
                            st.warning(
                                L.get(
                                    "no_content_to_translate",
                                    "번역할 대화 내용이 없습니다."))
                        else:
                            # ⭐ 수정: 원본 대화 내용을 그대로 번역 (요약하지 않고 원문 그대로 번역)
                            lang_name_source = {
                                "ko": "Korean", "en": "English", "ja": "Japanese"}.get(
                                source_lang, "Korean")
                            lang_name_target = {
                                "ko": "Korean", "en": "English", "ja": "Japanese"}.get(
                                target_lang, "Korean")

                            # 원본 대화 내용을 그대로 번역
                            with st.spinner(L.get("transfer_loading", "번역 중...")):
                                # 번역 로직 실행 (요약 없이 원본 그대로 번역)
                                translated_summary, is_success = translate_text_with_llm(
                                    history_text, target_lang, source_lang)

                                if not translated_summary:
                                    st.warning(
                                        L.get(
                                            "translation_empty",
                                            "번역 결과가 비어있습니다. 원본 텍스트를 사용합니다."))
                                    translated_summary = summary_text
                                    is_success = False

                                # ⭐ [수정] 번역 재시도 시에도 모든 메시지 번역
                                translated_messages = []
                                for msg in st.session_state.simulator_messages:
                                    translated_msg = msg.copy()
                                    # 번역할 메시지 역할 필터링 (시스템 메시지 등은 제외)
                                    if msg["role"] in [
                                        "initial_query",
                                        "customer",
                                        "customer_rebuttal",
                                        "agent_response",
                                        "customer_closing_response",
                                            "supervisor"]:
                                        if msg.get("content"):
                                            # 각 메시지 내용을 번역
                                            try:
                                                translated_content, trans_success = translate_text_with_llm(
                                                    msg["content"], target_lang, source_lang)
                                                if trans_success:
                                                    translated_msg["content"] = translated_content
                                            except Exception as e:
                                                # 번역 오류 시 원본 유지
                                                pass
                                    translated_messages.append(translated_msg)

                                # 번역된 메시지로 업데이트
                                st.session_state.simulator_messages = translated_messages

                                # 번역 결과 저장
                                st.session_state.transfer_summary_text = translated_summary
                                st.session_state.translation_success = is_success

                                # ⭐ 재실행 불필요: 결과는 이미 세션 상태에 저장되어 자동 표시됨
                                # st.rerun()
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(
                        L.get(
                            "translation_retry_error",
                            "번역 재시도 중 오류 발생: {error}").format(
                            error=str(e)))
                    st.code(error_details)
                    st.session_state.transfer_summary_text = L.get(
                        "translation_error", "번역 오류: {error}").format(error=str(e))
                    st.session_state.translation_success = False

    # =========================
    # 5. 에이전트 입력 단계 (AGENT_TURN) - ⭐ 수정: 원위치 복원 - 항상 입력 칸 표시
    # =========================
    # ⭐ 수정: AGENT_TURN 단계에서 항상 에이전트 응답 입력 UI를 표시 (원위치 복원)
    # app.py 스타일: AGENT_TURN 단계에서 항상 입력 칸이 보이도록 함
    # 단, 검증 UI나 응대 초안 UI가 표시될 때는 에이전트 응답 UI를 숨김
    if st.session_state.sim_stage == "AGENT_TURN":
        show_verification_from_button = st.session_state.get(
            "show_verification_ui", False)
        show_draft_ui = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui = st.session_state.get(
            "show_customer_data_ui", False)

        # 각 기능이 표시될 때는 해당 기능의 헤더만 표시
        if show_verification_from_button:
            # 고객 검증 헤더는 아래에서 표시됨
            pass
        elif show_draft_ui:
            # 응대 초안은 메시지로 표시되므로 헤더 불필요
            pass
        elif show_customer_data_ui:
            # 데이터 가져오기는 메시지로 표시되므로 헤더 불필요
            pass
        else:
            # 기본 에이전트 응답 헤더 표시
            st.markdown(f"### {L['agent_response_header']}")

        # ⭐ 실시간 응대 힌트 영역 제거 (메시지 말풍선에 버튼으로 이동)
        # 힌트는 에이전트 응답 메시지 말풍선의 '응대 힌트' 버튼을 통해 사용할 수 있습니다.

        # ⭐ 추가: 고객 성향 기반 가이드라인 추천 (신규 고객 문의 시)
        if st.session_state.simulator_messages and len(
                st.session_state.simulator_messages) >= 2:
            # 고객 메시지가 있고 요약이 생성 가능한 경우
            try:
                # 현재 대화를 임시 요약하여 고객 성향 분석
                temp_summary = generate_chat_summary(
                    st.session_state.simulator_messages,
                    st.session_state.customer_query_text_area,
                    st.session_state.get("customer_type_sim_select", ""),
                    st.session_state.language
                )

                if temp_summary and temp_summary.get(
                        "customer_sentiment_score"):
                    # 과거 이력 로드
                    all_histories = load_simulation_histories_local(
                        st.session_state.language)

                    # 가이드라인 추천 생성
                    recommended_guideline = recommend_guideline_for_customer(
                        temp_summary,
                        all_histories,
                        st.session_state.language
                    )

                    if recommended_guideline:
                        with st.expander("💡 고객 성향 기반 응대 가이드라인 추천", expanded=False):
                            st.markdown(recommended_guideline)
                            st.caption(
                                "💡 이 가이드는 유사한 과거 고객 사례를 분석하여 자동 생성되었습니다.")
            except Exception as e:
                # 가이드라인 추천 실패 시 무시 (비차단)
                pass

        # --- 언어 이관 요청 강조 표시 ---
        if st.session_state.language_transfer_requested:
            st.error(
                L.get(
                    "language_transfer_requested_msg",
                    "🚨 고객이 언어 전환(이관)을 요청했습니다. 즉시 응대하거나 이관을 진행하세요。"))

        # --- 고객 첨부 파일 정보 재표시 ---
        if st.session_state.sim_attachment_context_for_llm:
            st.info(
                f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")

        # 고객 첨부 파일이 있는지 확인 (검증 프로세스에서 사용)
        customer_has_attachment = (
            st.session_state.customer_attachment_file is not None or
            (st.session_state.sim_attachment_context_for_llm and
             st.session_state.sim_attachment_context_for_llm.strip())
        )

        # --- 고객 검증 프로세스 (로그인/계정 관련 문의이고 고객이 정보를 제공한 경우) ---
        # 개선: 초기 쿼리뿐만 아니라 모든 고객 메시지에서 로그인 관련 문의 확인
        initial_query = st.session_state.get('customer_query_text_area', '')

        # 모든 고객 메시지 수집 (초기 쿼리 포함)
        all_customer_texts = []
        if initial_query:
            all_customer_texts.append(initial_query)

        if st.session_state.simulator_messages:
            # 디버깅: 메시지 확인
            all_roles = [msg.get("role")
                         for msg in st.session_state.simulator_messages]
            customer_messages = [
                msg for msg in st.session_state.simulator_messages if msg.get("role") in [
                    "customer", "customer_rebuttal", "initial_query"]]

            # 모든 고객 메시지의 내용 수집
            for msg in customer_messages:
                content = msg.get("content", "")
                if content and content not in all_customer_texts:
                    all_customer_texts.append(content)

            # 모든 고객 메시지를 합쳐서 로그인 관련 문의 확인
            combined_customer_text = " ".join(all_customer_texts)
            is_login_inquiry = check_if_login_related_inquiry(
                combined_customer_text)

            # 고객이 검증 정보를 제공했는지 확인
            customer_provided_info = check_if_customer_provided_verification_info(
                st.session_state.simulator_messages)

            # 고객이 첨부 파일을 제공한 경우 검증 정보 제공으로 간주
            if customer_has_attachment and is_login_inquiry:
                customer_provided_info = True
                st.session_state.debug_attachment_detected = True

            # 보조 검증: 함수 결과가 False인 경우에도 직접 패턴 확인 (디버깅 및 보완)
            if not customer_provided_info and is_login_inquiry:
                # 고객 메시지에서 검증 정보 패턴 직접 확인
                verification_keywords = [
                    "영수증",
                    "receipt",
                    "예약번호",
                    "reservation",
                    "결제",
                    "payment",
                    "카드",
                    "card",
                    "계좌",
                    "account",
                    "이메일",
                    "email",
                    "전화",
                    "phone",
                    "성함",
                    "이름",
                    "name",
                    "주문번호",
                    "order",
                    "주문",
                    "결제내역",
                    "스크린샷",
                    "screenshot",
                    "사진",
                    "photo",
                    "첨부",
                    "attachment",
                    "파일",
                    "file"]
                combined_text_lower = combined_customer_text.lower()
                manual_check = any(
                    keyword.lower() in combined_text_lower for keyword in verification_keywords)

                # 이메일이나 전화번호 패턴 확인
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                phone_pattern = r'\b\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4}\b'
                has_email = bool(
                    re.search(
                        email_pattern,
                        combined_customer_text))
                has_phone = bool(
                    re.search(
                        phone_pattern,
                        combined_customer_text))

                # 고객이 첨부 파일을 제공한 경우도 검증 정보 제공으로 간주
                if customer_has_attachment:
                    customer_provided_info = True
                    st.session_state.debug_manual_verification_detected = True
                    st.session_state.debug_attachment_detected = True
                # 수동 확인 결과도 고려 (더 관대한 검증)
                elif manual_check or has_email or has_phone:
                    customer_provided_info = True
                    st.session_state.debug_manual_verification_detected = True
                    st.session_state.debug_attachment_detected = False
                else:
                    st.session_state.debug_manual_verification_detected = False
                    st.session_state.debug_attachment_detected = False

            # 디버깅용: 정보 제공 여부 확인
            if is_login_inquiry:
                st.session_state.debug_verification_info = customer_provided_info
                st.session_state.debug_all_roles = all_roles
                st.session_state.debug_customer_messages_count = len(
                    customer_messages)
                # 처음 200자만 저장
                st.session_state.debug_combined_customer_text = combined_customer_text[:200]
        else:
            # 메시지가 없는 경우 초기 쿼리만 확인
            is_login_inquiry = check_if_login_related_inquiry(initial_query)
            customer_provided_info = False
            all_roles = []
            customer_messages = []

        # ⭐ 수정: 검증 UI는 고객 메시지 버튼 클릭 시에만 표시 (기존 자동 표시 제거)
        # 로그인 관련 문의이고, 고객이 정보를 제공했으며, 아직 검증되지 않은 경우
        # 그리고 고객 메시지에서 검증 버튼을 클릭한 경우에만 검증 UI 표시
        # show_verification_from_button은 위에서 이미 정의됨

        # ⭐ 고객 검증 UI 표시 (버튼 클릭 시에만, 다른 기능이 표시되지 않을 때만)
        show_draft_ui_check = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui_check = st.session_state.get(
            "show_customer_data_ui", False)
        if show_verification_from_button and not show_draft_ui_check and not show_customer_data_ui_check:
            st.markdown("---")
            st.markdown(f"### {L.get('verification_header', '고객 검증')}")
            st.warning(L.get('verification_warning', '고객 검증이 필요합니다.'))

        # 디버깅: 조건 확인 (기존 유지하되, 자동 표시는 제거)
        if is_login_inquiry and show_verification_from_button:
            # 디버깅 정보 표시 (항상 표시)
            with st.expander("🔍 검증 감지 디버깅 정보", expanded=True):
                st.write(f"**조건 확인:**")
                st.write(f"- 로그인 관련 문의: ✅ {is_login_inquiry}")
                st.write(
                    f"- 고객 정보 제공 감지: {'✅' if customer_provided_info else '❌'} {customer_provided_info}")
                st.write(
                    f"- 고객 첨부 파일 존재: {'✅' if customer_has_attachment else '❌'} {customer_has_attachment}")
                if 'debug_manual_verification_detected' in st.session_state:
                    st.write(
                        f"- 수동 검증 패턴 감지: {'✅' if st.session_state.debug_manual_verification_detected else '❌'} {st.session_state.debug_manual_verification_detected}")
                if 'debug_attachment_detected' in st.session_state:
                    st.write(
                        f"- 첨부 파일로 인한 검증 정보 감지: {'✅' if st.session_state.debug_attachment_detected else '❌'} {st.session_state.debug_attachment_detected}")
                st.write(
                    f"- 검증 완료 여부: {'✅' if st.session_state.is_customer_verified else '❌'} {st.session_state.is_customer_verified}")
                st.write(
                    f"- 검증 UI 표시 조건: {is_login_inquiry and customer_provided_info and not st.session_state.is_customer_verified}")

                # 확인한 텍스트 정보 표시
                if 'debug_combined_customer_text' in st.session_state and st.session_state.debug_combined_customer_text:
                    st.write(
                        f"**확인한 고객 텍스트 (처음 200자):** {st.session_state.debug_combined_customer_text}")
                elif all_customer_texts:
                    combined_preview = " ".join(all_customer_texts)[:200]
                    st.write(f"**확인한 고객 텍스트 (처음 200자):** {combined_preview}")

                if st.session_state.simulator_messages:
                    st.write(
                        f"**전체 메시지 수:** {len(st.session_state.simulator_messages)}")
                    st.write(
                        f"**모든 role 목록:** {st.session_state.debug_all_roles if 'debug_all_roles' in st.session_state else [msg.get('role') for msg in st.session_state.simulator_messages]}")
                    st.write(
                        f"**고객 메시지 수:** {st.session_state.debug_customer_messages_count if 'debug_customer_messages_count' in st.session_state else len([m for m in st.session_state.simulator_messages if m.get('role') in ['customer', 'customer_rebuttal', 'initial_query']])}")

                    # ⭐ 추가: 고객 데이터 정보 표시 (app.py 스타일)
                    if st.session_state.customer_data:
                        customer_info = st.session_state.customer_data.get(
                            "data", {})
                        st.write(
                            f"**{L.get('customer_data_label', '고객 데이터')}:** ✅ {L.get('loaded', '불러옴')}")
                        st.write(
                            f"- {L.get('name_label', '이름')}: {customer_info.get('name', 'N/A')}")
                        st.write(
                            f"- {L.get('email_label', '이메일')}: {customer_info.get('email', 'N/A')}")
                        st.write(
                            f"- {L.get('phone_label', '전화번호')}: {customer_info.get('phone', 'N/A')}")
                        if customer_info.get('purchase_history'):
                            st.write(
                                f"- {L.get('purchase_history_label', '구매 이력')}: {len(customer_info.get('purchase_history', []))}{L.get('cases_label', '건')}")
                    else:
                        st.write(
                            f"**{L.get('customer_data_label', '고객 데이터')}:** ❌ {L.get('none', '없음')}")

                    # ⭐ 추가: 누적 데이터 수 자동 확인 (고객 데이터 매니저에서)
                    try:
                        all_customers = st.session_state.customer_data_manager.list_all_customers()
                        st.write(
                            f"**{L.get('accumulated_customer_data_label', '누적 고객 데이터 수')}:** {len(all_customers)}{L.get('cases_label', '건')}")
                    except Exception:
                        st.write(
                            f"**{L.get('accumulated_customer_data_label', '누적 고객 데이터 수')}:** {L.get('unavailable', '확인 불가')}")

                    # 모든 메시지 표시 (최근 10개)
                    st.write(f"**최근 모든 메시지 (최근 10개):**")
                    for i, msg in enumerate(
                            st.session_state.simulator_messages[-10:], 1):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")[:300]
                        st.write(f"{i}. [{role}] {content}")

                    # 고객 메시지만 필터링하여 표시
                    customer_messages = [
                        {"role": msg.get("role"), "content": msg.get("content", "")[:300]}
                        for msg in st.session_state.simulator_messages[-10:]
                        if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]
                    ]
                    st.write(f"**고객 메시지만 (최근 10개):**")
                    if customer_messages:
                        for i, msg in enumerate(customer_messages, 1):
                            st.write(f"{i}. [{msg['role']}] {msg['content']}")
                    else:
                        st.write(L.get("no_customer_messages", "고객 메시지 없음"))
                else:
                    st.write(f"**{L.get('no_messages', '메시지 없음')}**")

            if not customer_provided_info:
                # 정보가 아직 제공되지 않은 경우 안내 메시지 표시
                st.warning(
                    "⚠️ 고객이 검증 정보를 제공하면 검증 UI가 표시됩니다. 위의 디버깅 정보를 확인하세요.")

        # ⭐ 수정: 검증 UI는 고객 메시지 버튼 클릭 시에만 표시
        # 고객 데이터 정보를 디버깅 정보에 포함
        # 다른 기능이 표시되지 않을 때만 검증 UI 표시
        show_draft_ui_check2 = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui_check2 = st.session_state.get(
            "show_customer_data_ui", False)
        if is_login_inquiry and customer_provided_info and not st.session_state.is_customer_verified and show_verification_from_button and not show_draft_ui_check2 and not show_customer_data_ui_check2:
            # 헤더는 위에서 이미 표시했으므로 중복 제거

            # 고객 데이터 표시 (있는 경우)
            if st.session_state.customer_data:
                customer_info = st.session_state.customer_data.get("data", {})
                with st.expander(L.get("customer_data_info_expander", "📋 고객 데이터 정보"), expanded=False):
                    st.json(customer_info)
                    # 누적 데이터 수 표시
                    try:
                        all_customers = st.session_state.customer_data_manager.list_all_customers()
                        st.caption(f"📊 누적 고객 데이터: {len(all_customers)}건")
                    except Exception:
                        pass

            with st.expander(L.get("verification_info_input", "고객 검증 정보 입력"), expanded=True):
                # 고객이 처음에 첨부한 파일 표시
                if customer_has_attachment:
                    if st.session_state.customer_attachment_file:
                        attachment_file = st.session_state.customer_attachment_file
                        st.success(
                            L.get(
                                "customer_initial_attachment",
                                "📎 고객이 처음에 첨부한 파일: **{filename}** ({size} bytes, {type})").format(
                                filename=attachment_file.name,
                                size=attachment_file.size,
                                type=attachment_file.type))
                        # 고객 첨부 파일을 검증 파일로도 사용 가능하도록 설정
                        if 'verification_file_info' not in st.session_state or not st.session_state.verification_file_info:
                            st.session_state.verification_file_info = {
                                "filename": attachment_file.name,
                                "size": attachment_file.size,
                                "type": attachment_file.type,
                                "source": "customer_initial_attachment"
                            }
                    elif st.session_state.sim_attachment_context_for_llm:
                        st.info(
                            L.get(
                                "customer_attachment_info",
                                "📎 고객이 첨부한 파일 정보: {info}").format(
                                info=st.session_state.sim_attachment_context_for_llm.replace(
                                    '[ATTACHMENT STATUS]',
                                    '').strip()))

                st.markdown("---")
                st.write(
                    f"**{L.get('additional_verification_file_upload', '추가 검증 파일 업로드 (선택사항)')}**")
                # 파일 업로더 (스크린샷/사진 스캔용) - 추가 파일 업로드 가능
                verification_file = st.file_uploader(
                    L.get(
                        "verification_file_upload_label",
                        "검증 파일 업로드 (스크린샷/사진)"),
                    type=[
                        "png",
                        "jpg",
                        "jpeg",
                        "pdf"],
                    key="verification_file_uploader",
                    help=L.get(
                        "verification_file_upload_help",
                        "고객이 제공한 영수증, 예약 확인서, 결제 내역 등의 스크린샷/사진을 추가로 업로드하세요. (고객이 처음에 첨부한 파일이 있으면 자동으로 포함됩니다.)"))

                # 검증에 사용할 파일 결정 (고객 첨부 파일 우선, 없으면 새로 업로드한 파일)
                file_to_verify = None
                file_verified = False
                ocr_extracted_info = {}  # OCR로 추출된 정보 저장

                if customer_has_attachment and st.session_state.customer_attachment_file:
                    file_to_verify = st.session_state.customer_attachment_file
                    file_verified = True
                    st.info(
                        L.get(
                            "verification_file_using_customer_attachment",
                            "✅ 검증에 사용할 파일: **{filename}** (고객이 처음에 첨부한 파일)").format(
                            filename=file_to_verify.name))
                elif verification_file:
                    file_to_verify = verification_file
                    file_verified = True
                    st.info(
                        L.get(
                            "file_upload_complete",
                            "✅ 파일 업로드 완료: {filename} ({size} bytes)").format(
                            filename=verification_file.name,
                            size=verification_file.size))
                    # 파일 정보를 세션 상태에 저장
                    st.session_state.verification_file_info = {
                        "filename": verification_file.name,
                        "size": verification_file.size,
                        "type": verification_file.type,
                        "source": "verification_uploader"
                    }
                elif customer_has_attachment:
                    # 첨부 파일 정보만 있고 파일 객체는 없는 경우 (이전 세션에서 업로드)
                    file_verified = True  # 파일이 있었다는 정보만으로도 검증 가능
                    st.info(
                        L.get(
                            "customer_attachment_info_confirmed",
                            "✅ 고객이 첨부한 파일 정보가 확인되었습니다."))

                # OCR 기능: 파일이 업로드되면 자동으로 정보 추출
                if file_to_verify and file_to_verify.name.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.pdf')):
                    if 'ocr_extracted_info' not in st.session_state or st.session_state.get(
                            'ocr_file_name') != file_to_verify.name:
                        with st.spinner(L.get("extracting_info_from_screenshot", "🔍 스크린샷에서 정보 추출 중 (OCR)...")):
                            try:
                                # 파일 읽기
                                file_to_verify.seek(0)
                                file_bytes = file_to_verify.getvalue()
                                file_type = file_to_verify.type

                                # Gemini Vision API를 사용한 OCR
                                gemini_key = get_api_key("gemini")
                                if gemini_key:
                                    import google.generativeai as genai
                                    genai.configure(api_key=gemini_key)
                                    model = genai.GenerativeModel(
                                        'gemini-2.0-flash-exp')

                                    # 검증 정보 추출을 위한 특화 프롬프트
                                    ocr_verification_prompt = """이 이미지는 고객 검증을 위한 스크린샷입니다. 다음 정보를 추출해주세요:

1. 영수증 번호 또는 예약 번호 (Receipt/Reservation Number)
2. 고객 성함 (Customer Name)
3. 고객 이메일 (Customer Email)
4. 고객 전화번호 (Customer Phone)
5. 결제 수단 (Payment Method: 신용카드, 체크카드, 카카오페이, 네이버페이, 온라인뱅킹 등)
6. 카드 뒷자리 4자리 (Card Last 4 Digits) - 있는 경우
7. 계좌번호 (Account Number) - 있는 경우

각 정보를 JSON 형식으로 반환해주세요:
{
  "receipt_number": "추출된 영수증/예약 번호 또는 빈 문자열",
  "customer_name": "추출된 고객 성함 또는 빈 문자열",
  "customer_email": "추출된 이메일 주소 또는 빈 문자열",
  "customer_phone": "추출된 전화번호 또는 빈 문자열",
  "payment_method": "추출된 결제 수단 또는 빈 문자열",
  "card_last4": "추출된 카드 뒷자리 4자리 또는 빈 문자열",
  "account_number": "추출된 계좌번호 또는 빈 문자열"
}

정보가 없으면 빈 문자열("")로 반환하세요. JSON 형식만 반환하고 다른 설명은 추가하지 마세요."""

                                    if file_to_verify.name.lower().endswith('.pdf'):
                                        # PDF는 텍스트 추출 후 OCR
                                        import tempfile
                                        import os
                                        tmp = tempfile.NamedTemporaryFile(
                                            delete=False, suffix=".pdf")
                                        tmp.write(file_bytes)
                                        tmp.flush()
                                        tmp.close()
                                        try:
                                            loader = PyPDFLoader(tmp.name)
                                            file_docs = loader.load()
                                            pdf_text = "\n".join(
                                                [doc.page_content for doc in file_docs])
                                            # PDF 텍스트가 있으면 그대로 사용, 없으면 이미지로 처리
                                            if pdf_text.strip():
                                                response = model.generate_content(
                                                    f"{ocr_verification_prompt}\n\n추출된 텍스트:\n{pdf_text}")
                                            else:
                                                # PDF를 이미지로 변환하여 처리 (간단한 경우
                                                # 텍스트만 사용)
                                                response = model.generate_content([
                                                    {"mime_type": "application/pdf", "data": file_bytes},
                                                    ocr_verification_prompt
                                                ])
                                        finally:
                                            try:
                                                os.remove(tmp.name)
                                            except BaseException:
                                                pass
                                    else:
                                        # 이미지 파일
                                        response = model.generate_content([
                                            {"mime_type": file_type, "data": file_bytes},
                                            ocr_verification_prompt
                                        ])

                                    ocr_result = response.text if response.text else ""

                                    # JSON 파싱 시도
                                    try:
                                        # JSON 부분만 추출 (코드 블록 제거)
                                        import json
                                        ocr_result_clean = ocr_result.strip()
                                        if ocr_result_clean.startswith("```"):
                                            # 코드 블록 제거
                                            lines = ocr_result_clean.split(
                                                "\n")
                                            json_lines = [
                                                l for l in lines if not l.strip().startswith("```")]
                                            ocr_result_clean = "\n".join(
                                                json_lines)

                                        ocr_extracted_info = json.loads(
                                            ocr_result_clean)
                                        st.session_state.ocr_extracted_info = ocr_extracted_info
                                        st.session_state.ocr_file_name = file_to_verify.name

                                        # 추출된 정보 표시
                                        extracted_fields = []
                                        if ocr_extracted_info.get(
                                                "receipt_number"):
                                            extracted_fields.append(
                                                f"영수증/예약 번호: {ocr_extracted_info['receipt_number']}")
                                        if ocr_extracted_info.get(
                                                "customer_name"):
                                            extracted_fields.append(
                                                f"고객 성함: {ocr_extracted_info['customer_name']}")
                                        if ocr_extracted_info.get(
                                                "customer_email"):
                                            extracted_fields.append(
                                                f"이메일: {ocr_extracted_info['customer_email']}")
                                        if ocr_extracted_info.get(
                                                "customer_phone"):
                                            extracted_fields.append(
                                                f"전화번호: {ocr_extracted_info['customer_phone']}")
                                        if ocr_extracted_info.get(
                                                "payment_method"):
                                            extracted_fields.append(
                                                f"결제 수단: {ocr_extracted_info['payment_method']}")
                                        if ocr_extracted_info.get(
                                                "card_last4"):
                                            extracted_fields.append(
                                                f"카드 뒷자리: {ocr_extracted_info['card_last4']}")

                                        if extracted_fields:
                                            st.success(
                                                L.get(
                                                    "ocr_extracted_info",
                                                    "✅ OCR로 다음 정보를 추출했습니다:") +
                                                "\n" +
                                                "\n".join(
                                                    f"- {field}" for field in extracted_fields))
                                        else:
                                            st.info(
                                                L.get(
                                                    "ocr_no_verification_info",
                                                    "ℹ️ OCR로 정보를 추출했지만 검증에 필요한 정보를 찾지 못했습니다."))
                                    except json.JSONDecodeError:
                                        # JSON 파싱 실패 시 텍스트에서 직접 추출 시도
                                        st.warning(
                                            L.get(
                                                "ocr_json_parse_failed",
                                                "⚠️ OCR 결과를 JSON으로 파싱하지 못했습니다. 수동으로 입력해주세요."))
                                        st.text_area(
                                            L.get(
                                                "ocr_raw_result_label",
                                                "OCR 원본 결과:"),
                                            ocr_result,
                                            height=100,
                                            key="ocr_raw_result")
                                        ocr_extracted_info = {}
                                else:
                                    st.warning(
                                        L.get(
                                            "ocr_requires_gemini",
                                            "⚠️ OCR 기능을 사용하려면 Gemini API 키가 필요합니다. 수동으로 정보를 입력해주세요."))
                            except Exception as ocr_error:
                                st.warning(
                                    L.get(
                                        "ocr_error_occurred",
                                        "⚠️ OCR 처리 중 오류가 발생했습니다: {error}").format(
                                        error=str(ocr_error)))
                                ocr_extracted_info = {}
                    else:
                        # 이전에 추출한 정보 재사용
                        ocr_extracted_info = st.session_state.get(
                            'ocr_extracted_info', {})
                        if ocr_extracted_info:
                            extracted_fields = []
                            if ocr_extracted_info.get("receipt_number"):
                                extracted_fields.append(
                                    f"{L.get('receipt_number_label', '영수증/예약 번호')}: {ocr_extracted_info['receipt_number']}")
                            if ocr_extracted_info.get("customer_name"):
                                extracted_fields.append(
                                    f"{L.get('customer_name_label', '고객 성함')}: {ocr_extracted_info['customer_name']}")
                            if ocr_extracted_info.get("customer_email"):
                                extracted_fields.append(
                                    f"{L.get('email_label', '이메일')}: {ocr_extracted_info['customer_email']}")
                            if ocr_extracted_info.get("customer_phone"):
                                extracted_fields.append(
                                    f"{L.get('phone_label', '전화번호')}: {ocr_extracted_info['customer_phone']}")
                            if extracted_fields:
                                st.info(
                                    L.get(
                                        "previous_extracted_info",
                                        "ℹ️ 이전에 추출한 정보:") +
                                    " " +
                                    ", ".join(extracted_fields))

                # OCR로 추출된 정보가 있으면 세션 상태에서 가져오기
                if 'ocr_extracted_info' in st.session_state and st.session_state.ocr_extracted_info:
                    ocr_extracted_info = st.session_state.ocr_extracted_info

                verification_cols = st.columns(2)

                with verification_cols[0]:
                    # OCR로 추출한 정보가 있으면 기본값으로 사용
                    receipt_default = ocr_extracted_info.get(
                        "receipt_number", "") if ocr_extracted_info else ""
                    verification_receipt = st.text_input(
                        L['verification_receipt_label'],
                        value=receipt_default,
                        key="verification_receipt_input",
                        help=L.get(
                            "verification_receipt_help",
                            "고객이 제공한 영수증 번호 또는 예약 번호를 입력하세요. (OCR로 자동 추출됨)"))

                    # 결제 수단 선택
                    payment_method_options = [
                        L.get("payment_method_card", "신용/체크카드"),
                        L.get("payment_method_kakaopay", "카카오페이"),
                        L.get("payment_method_naverpay", "네이버페이"),
                        L.get("payment_method_online_banking", "온라인뱅킹"),
                        L.get("payment_method_grabpay", "GrabPay"),
                        L.get("payment_method_tng", "Touch N Go"),
                        L.get("payment_method_other", "기타")
                    ]

                    # OCR로 추출한 결제 수단이 있으면 매칭 시도
                    ocr_payment_method = ocr_extracted_info.get(
                        "payment_method", "") if ocr_extracted_info else ""
                    payment_method_index = 0
                    if ocr_payment_method:
                        # OCR 추출값과 옵션 매칭
                        ocr_payment_lower = ocr_payment_method.lower()
                        for idx, option in enumerate(payment_method_options):
                            if any(
                                keyword in ocr_payment_lower for keyword in [
                                    "카드", "card", "신용", "credit", "체크", "check"]):
                                if "신용" in option or "체크" in option or "card" in option.lower():
                                    payment_method_index = idx
                                    break
                            elif any(keyword in ocr_payment_lower for keyword in ["카카오", "kakao"]):
                                if "카카오" in option:
                                    payment_method_index = idx
                                    break
                            elif any(keyword in ocr_payment_lower for keyword in ["네이버", "naver"]):
                                if "네이버" in option:
                                    payment_method_index = idx
                                    break
                            elif any(keyword in ocr_payment_lower for keyword in ["계좌", "account", "뱅킹", "banking"]):
                                if "뱅킹" in option or "banking" in option.lower():
                                    payment_method_index = idx
                                    break

                    verification_payment_method = st.selectbox(
                        L['verification_payment_method_label'],
                        options=payment_method_options,
                        index=payment_method_index,
                        key="verification_payment_method_input",
                        help="고객이 사용한 결제 수단을 선택하세요. (OCR로 자동 추출됨)"
                    )

                    # 결제 정보 입력 (카드 뒷자리 또는 계좌번호)
                    if verification_payment_method == L.get(
                            "payment_method_card", "신용/체크카드"):
                        card_default = ocr_extracted_info.get(
                            "card_last4", "") if ocr_extracted_info else ""
                        verification_card = st.text_input(
                            L['verification_card_label'],
                            value=card_default,
                            key="verification_card_input",
                            max_chars=4,
                            help=L.get(
                                "verification_card_help",
                                "고객이 제공한 카드 뒷자리 4자리를 입력하세요. (OCR로 자동 추출됨)"))
                        verification_account = ""
                    elif verification_payment_method == L.get("payment_method_online_banking", "온라인뱅킹"):
                        account_default = ocr_extracted_info.get(
                            "account_number", "") if ocr_extracted_info else ""
                        verification_account = st.text_input(
                            L['verification_account_label'],
                            value=account_default,
                            key="verification_account_input",
                            help="고객이 제공한 계좌번호를 입력하세요. (OCR로 자동 추출됨)"
                        )
                        verification_card = ""
                    else:
                        # 카카오페이, 네이버페이 등은 결제 수단 정보만으로 확인 가능
                        verification_card = ""
                        verification_account = ""

                    name_default = ocr_extracted_info.get(
                        "customer_name", "") if ocr_extracted_info else ""
                    verification_name = st.text_input(
                        L['verification_name_label'],
                        value=name_default,
                        key="verification_name_input",
                        help=L.get(
                            "verification_name_help",
                            "고객이 제공한 성함을 입력하세요. (OCR로 자동 추출됨)"))

                with verification_cols[1]:
                    email_default = ocr_extracted_info.get(
                        "customer_email", "") if ocr_extracted_info else ""
                    verification_email = st.text_input(
                        L['verification_email_label'],
                        value=email_default,
                        key="verification_email_input",
                        help=L.get(
                            "verification_email_help",
                            "고객이 제공한 이메일 주소를 입력하세요. (OCR로 자동 추출됨)"))
                    phone_default = ocr_extracted_info.get(
                        "customer_phone", "") if ocr_extracted_info else ""
                    verification_phone = st.text_input(
                        L['verification_phone_label'],
                        value=phone_default,
                        key="verification_phone_input",
                        help=L.get(
                            "verification_phone_help",
                            "고객이 제공한 연락처를 입력하세요. (OCR로 자동 추출됨)"))

                # 시스템에 저장된 검증 정보 (시뮬레이션용 - 실제로는 DB에서 가져옴)
                stored_verification_info = st.session_state.verification_info.copy()

                # 검증 버튼
                st.markdown("---")
                verify_cols = st.columns([1, 1])
                with verify_cols[0]:
                    if st.button(
                            L['button_verify'],
                            key="btn_verify_customer",
                            use_container_width=True,
                            type="primary"):
                        # 파일 검증 정보 확인 (고객 첨부 파일 또는 새로 업로드한 파일)
                        final_file_verified = False
                        file_info_for_verification = None

                        if file_to_verify:
                            final_file_verified = True
                            file_info_for_verification = {
                                "filename": file_to_verify.name, "size": file_to_verify.size if hasattr(
                                    file_to_verify, 'size') else 0, "type": file_to_verify.type if hasattr(
                                    file_to_verify, 'type') else "unknown"}
                            st.session_state.verification_file_verified = True
                        elif file_verified:  # 파일 정보만 있는 경우
                            final_file_verified = True
                            file_info_for_verification = st.session_state.verification_file_info if 'verification_file_info' in st.session_state else None

                        # 결제 정보 구성 (payment_info 필드 추가)
                        payment_info = ""
                        if verification_payment_method == L.get(
                                "payment_method_card", "신용/체크카드"):
                            payment_info = f"{verification_payment_method} {verification_card}" if verification_card else verification_payment_method
                        elif verification_payment_method == L.get("payment_method_online_banking", "온라인뱅킹"):
                            payment_info = f"{verification_payment_method} {verification_account}" if verification_account else verification_payment_method
                        else:
                            payment_info = verification_payment_method

                        # OCR로 추출한 정보가 있으면 우선 사용 (수동 입력값이 있으면 수동 입력값 우선)
                        final_receipt = verification_receipt if verification_receipt else (
                            ocr_extracted_info.get("receipt_number", "") if ocr_extracted_info else "")
                        final_name = verification_name if verification_name else (
                            ocr_extracted_info.get("customer_name", "") if ocr_extracted_info else "")
                        final_email = verification_email if verification_email else (
                            ocr_extracted_info.get("customer_email", "") if ocr_extracted_info else "")
                        final_phone = verification_phone if verification_phone else (
                            ocr_extracted_info.get("customer_phone", "") if ocr_extracted_info else "")
                        final_card = verification_card if verification_card else (
                            ocr_extracted_info.get("card_last4", "") if ocr_extracted_info else "")
                        final_account = verification_account if verification_account else (
                            ocr_extracted_info.get("account_number", "") if ocr_extracted_info else "")

                        provided_info = {
                            "receipt_number": final_receipt,
                            "card_last4": final_card if verification_payment_method == L.get("payment_method_card", "신용/체크카드") else "",
                            "account_number": final_account if verification_payment_method == L.get("payment_method_online_banking", "온라인뱅킹") else "",
                            "payment_method": verification_payment_method,
                            "payment_info": payment_info,  # 결제 정보 통합 필드 추가
                            "customer_name": final_name,
                            "customer_email": final_email,
                            "customer_phone": final_phone,
                            "file_uploaded": final_file_verified,
                            "file_info": file_info_for_verification,  # 파일 상세 정보 추가
                            "ocr_extracted": ocr_extracted_info if ocr_extracted_info else {}  # OCR 추출 정보도 포함
                        }

                        # 시스템에 저장된 검증 정보에도 파일 정보 추가 (시뮬레이션용)
                        stored_verification_info_with_file = stored_verification_info.copy()
                        if customer_has_attachment and st.session_state.customer_attachment_file:
                            stored_verification_info_with_file["file_uploaded"] = True
                            stored_verification_info_with_file["file_info"] = {
                                "filename": st.session_state.customer_attachment_file.name,
                                "size": st.session_state.customer_attachment_file.size if hasattr(
                                    st.session_state.customer_attachment_file,
                                    'size') else 0,
                                "type": st.session_state.customer_attachment_file.type if hasattr(
                                    st.session_state.customer_attachment_file,
                                    'type') else "unknown"}

                        # 검증 실행 (시스템 내부에서만 실행)
                        is_verified, verification_results = verify_customer_info(
                            provided_info, stored_verification_info_with_file)

                        if is_verified:
                            st.session_state.is_customer_verified = True
                            st.session_state.verification_stage = "VERIFIED"
                            st.session_state.verification_info["verification_attempts"] += 1
                            st.success(L['verification_success'])
                        else:
                            st.session_state.verification_stage = "VERIFICATION_FAILED"
                            st.session_state.verification_info["verification_attempts"] += 1
                            failed_fields = [
                                k for k, v in verification_results.items() if not v]

                            # 검증 실패 필드에 대한 상세 정보 제공 (보안: 시스템 저장값은 노출하지 않음)
                            failed_details = []
                            for field in failed_fields:
                                provided_value = provided_info.get(field, "")

                                # 보안: 민감한 정보 마스킹 및 시스템 저장값은 노출하지 않음
                                if field == "file_uploaded":
                                    failed_details.append(
                                        f"{field}: 제공됨={provided_info.get('file_uploaded', False)}")
                                elif field == "file_info":
                                    provided_file = provided_info.get(
                                        'file_info', {})
                                    failed_details.append(
                                        f"{field}: 제공된 파일={provided_file.get('filename', '없음')}")
                                elif field == "customer_email":
                                    # 이메일 마스킹
                                    masked_email = mask_email(
                                        provided_value) if provided_value else "없음"
                                    failed_details.append(
                                        f"{field}: 제공값='{masked_email}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "customer_phone":
                                    # 전화번호 마스킹 (뒷자리만 표시)
                                    if provided_value and len(
                                            provided_value) > 4:
                                        masked_phone = "***-" + \
                                            provided_value[-4:]
                                    else:
                                        masked_phone = provided_value if provided_value else "없음"
                                    failed_details.append(
                                        f"{field}: 제공값='{masked_phone}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "card_last4":
                                    # 카드 번호는 이미 뒷자리 4자리만 있으므로 마스킹
                                    if provided_value:
                                        masked_card = "****" if len(
                                            provided_value) == 4 else provided_value
                                    else:
                                        masked_card = "없음"
                                    failed_details.append(
                                        f"{field}: 제공값='{masked_card}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "account_number":
                                    # 계좌번호 마스킹
                                    if provided_value and len(
                                            provided_value) > 4:
                                        masked_account = "***-" + \
                                            provided_value[-4:]
                                    else:
                                        masked_account = provided_value if provided_value else "없음"
                                    failed_details.append(
                                        f"{field}: 제공값='{masked_account}' (시스템 저장값은 보안상 표시하지 않음)")
                                elif field == "customer_name":
                                    # 이름은 부분 마스킹
                                    if provided_value and len(
                                            provided_value) > 1:
                                        masked_name = provided_value[0] + \
                                            "*" * (len(provided_value) - 1)
                                    else:
                                        masked_name = provided_value if provided_value else "없음"
                                    failed_details.append(
                                        f"{field}: 제공값='{masked_name}' (시스템 저장값은 보안상 표시하지 않음)")
                                else:
                                    # 기타 필드는 값의 일부만 표시 (보안)
                                    if provided_value:
                                        if len(provided_value) > 8:
                                            masked_value = provided_value[:4] + \
                                                "***" + provided_value[-2:]
                                        else:
                                            masked_value = "*" * \
                                                len(provided_value)
                                    else:
                                        masked_value = "없음"
                                    failed_details.append(
                                        f"{field}: 제공값='{masked_value}' (시스템 저장값은 보안상 표시하지 않음)")

                            error_message = L['verification_failed'].format(
                                failed_fields=', '.join(failed_fields))
                            error_message += "\n\n⚠️ **보안 정책**: 시스템에 저장된 실제 검증 정보는 보안상 표시하지 않습니다."
                            if failed_details:
                                error_message += f"\n\n**제공된 정보 (일부 마스킹):**\n" + "\n".join(
                                    f"- {detail}" for detail in failed_details)

                            st.error(error_message)

                with verify_cols[1]:
                    if st.button(
                            L['button_retry_verification'],
                            key="btn_retry_verification",
                            use_container_width=True):
                        st.session_state.verification_stage = "WAIT_VERIFICATION"
                        st.session_state.verification_info["verification_attempts"] = 0
                        # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                        # st.rerun()

                # 검증 시도 횟수 표시
                if st.session_state.verification_info.get(
                        "verification_attempts", 0) > 0:
                    st.info(
                        L['verification_attempts'].format(
                            count=st.session_state.verification_info['verification_attempts']))

            # ⭐ 수정: 검증 전 제한 사항도 버튼 클릭 시에만 표시 (고객 검증 버튼에 포함)
            # 검증되지 않은 상태에서는 힌트 및 초안 생성 제한
            st.markdown("---")
            st.markdown(
                f"### {L.get('verification_restrictions', '검증 전 제한 사항')}")
            st.info(
                L.get(
                    'verification_restrictions_text',
                    '검증이 완료되기 전까지 일부 기능이 제한됩니다.'))

        elif is_login_inquiry and st.session_state.is_customer_verified:
            st.success(L.get('verification_completed', '고객 검증이 완료되었습니다.'))

        # ⭐ 검증 UI가 표시될 때는 에이전트 응답 UI를 숨김
        # ⭐ AI 응답 초안 생성 기능 제거 (회사 정보 & FAQ 탭에 이미 있음)
        # 이 기능은 '회사 정보 & FAQ' > '고객 문의 재확인' 탭에서 사용할 수 있습니다.

        # ⭐ 전화 발신 버튼 제거 (메시지 말풍선에 버튼으로 이동)
        # 전화 발신 기능은 에이전트 응답 메시지 말풍선의 '업체에 전화' / '고객에게 전화' 버튼을 통해 사용할 수 있습니다.

        # Supervisor 정책 업로더 제거됨

        # --- 에이전트 첨부 파일 업로더는 숨김 처리 (버튼으로 대체) ---
        # 파일 업로더는 버튼 클릭 시에만 표시되도록 처리
        agent_attachment_files = None
        if st.session_state.get("show_agent_file_uploader", False):
            agent_attachment_files = st.file_uploader(
                L["agent_attachment_label"],
                type=["png", "jpg", "jpeg", "pdf"],
                key="agent_attachment_file_uploader",
                help=L["agent_attachment_placeholder"],
                accept_multiple_files=True
            )
            if agent_attachment_files:
                st.session_state.agent_attachment_file = [
                    {"name": f.name, "type": f.type, "size": f.size} for f in agent_attachment_files
                ]
                file_names = ", ".join(
                    [f["name"] for f in st.session_state.agent_attachment_file])
                st.info(
                    L.get(
                        "agent_attachment_files_ready",
                        "✅ {count}개 에이전트 첨부 파일 준비 완료: {files}").format(
                        count=len(agent_attachment_files),
                        files=file_names))
                st.session_state.show_agent_file_uploader = False  # 파일 선택 후 숨김
            else:
                st.session_state.agent_attachment_file = []
        else:
            st.session_state.agent_attachment_file = []

        # 마이크 녹음 처리 (전화 부분과 동일한 패턴: 종료 시 자동 전사)
        # 전사 로직: bytes_to_process에 데이터가 있을 때만 실행 (전화 부분과 동일)
        if "bytes_to_process" in st.session_state and st.session_state.bytes_to_process is not None:
            # ⭐ 수정: OpenAI 또는 Gemini API 키가 있는지 확인
            has_openai = st.session_state.openai_client is not None
            has_gemini = bool(get_api_key("gemini"))

            if not has_openai and not has_gemini:
                st.error(
                    L.get(
                        "whisper_client_error",
                        "Whisper 클라이언트 오류") +
                    " (OpenAI 또는 Gemini API Key 필요)")
                st.session_state.bytes_to_process = None
            else:
                # ⭐ 전사 결과를 저장할 변수 초기화
                agent_response_transcript = None

                # 전사 후 바이트 데이터 백업 (전사 전에 백업)
                audio_bytes_backup = st.session_state.bytes_to_process

                # 전사 후 바이트 데이터 즉시 삭제 (조건문 재평가 방지)
                st.session_state.bytes_to_process = None

                with st.spinner(L.get("whisper_processing", "전사 중...")):
                    try:
                        # Whisper 전사 (자동 언어 감지 사용)
                        agent_response_transcript = transcribe_bytes_with_whisper(
                            audio_bytes_backup, "audio/wav", lang_code=None, auto_detect=True)
                    except Exception as e:
                        agent_response_transcript = L.get(
                            "transcription_error_with_error",
                            "❌ 전사 오류: {error}").format(
                            error=str(e))

                # 2) 전사 실패 처리 (채팅/이메일과 동일한 패턴)
                if not agent_response_transcript or agent_response_transcript.startswith(
                        "❌"):
                    error_msg = agent_response_transcript if agent_response_transcript else L.get(
                        "transcription_no_result", "전사 결과가 없습니다.")
                    st.error(error_msg)

                    # ⭐ [수정 4] 채팅/메일 탭에서 에러 발생 시 입력 필드를 비움
                    if st.session_state.get(
                            "feature_selection") == L["sim_tab_chat_email"]:
                        st.session_state.agent_response_area_text = ""
                        st.session_state.last_transcript = ""  # 전사 실패 시 last_transcript 초기화
                    else:
                        # 전화 탭의 경우
                        st.session_state.current_agent_audio_text = L.get(
                            "transcription_error", "전사 오류")
                        if "agent_response_input_box_widget_call" in st.session_state:
                            st.session_state.agent_response_input_box_widget_call = ""  # 전화 탭 입력 필드도 초기화
                        st.session_state.last_transcript = ""  # 전사 실패 시 last_transcript 초기화

                elif not agent_response_transcript.strip():  # ⭐ 수정: 전사 결과가 비어 있거나 (공백만 있는 경우) 다음 단계로 진행하지 못하는 문제 해결
                    st.warning(
                        L.get(
                            "transcription_empty_warning",
                            "전사 결과가 비어 있습니다."))
                    if st.session_state.get(
                            "feature_selection") == L["sim_tab_chat_email"]:
                        st.session_state.agent_response_area_text = ""  # 채팅/메일 탭도 초기화
                    else:
                        st.session_state.current_agent_audio_text = ""
                        if "agent_response_input_box_widget_call" in st.session_state:
                            st.session_state.agent_response_input_box_widget_call = ""
                    st.session_state.last_transcript = ""
                    # ⭐ 재실행 불필요: 전사 결과가 비어있어도 사용자가 다시 녹음할 수 있음
                    # st.rerun()

                elif agent_response_transcript.strip():
                    # 3) 전사 성공 - CC/입력창에 반영
                    agent_response_transcript = agent_response_transcript.strip()

                    # ⭐ [핵심 수정 5] 전사 결과를 last_transcript에 저장하고, AGENT_TURN 상태의 입력 필드에도 반영
                    st.session_state.last_transcript = agent_response_transcript

                    # A. 채팅/메일 탭 처리
                    if st.session_state.get(
                            "feature_selection") == L["sim_tab_chat_email"]:
                        # AGENT_TURN 섹션의 st.text_area value로 사용되는 세션 상태 변수에 반영
                        st.session_state.agent_response_area_text = agent_response_transcript

                    # B. 전화 탭 처리
                    else:
                        st.session_state.current_agent_audio_text = agent_response_transcript
                        # ⭐ [수정 3: 핵심 수정] 전화 탭 입력 칸에도 전사 결과 전달
                        if "agent_response_input_box_widget_call" in st.session_state:
                            st.session_state.agent_response_input_box_widget_call = agent_response_transcript

                    # 성공 메시지 표시 (채팅/이메일과 유사)
                    snippet = agent_response_transcript[:50].replace("\n", " ")
                    if len(agent_response_transcript) > 50:
                        snippet += "..."
                    st.success(
                        L.get(
                            "whisper_success",
                            "전사 완료") +
                        f" **{L.get('recognized_content', '인식 내용')}:** *{snippet}*")
                    st.info(
                        L.get(
                            "transcription_auto_filled",
                            "💡 전사된 텍스트가 CC 자막 및 입력창에 자동으로 입력되었습니다."))

        # ⭐ 검증 UI나 응대 초안 UI가 표시되지 않을 때만 솔루션 체크박스 표시
        show_draft_ui = st.session_state.get("show_draft_ui", False)
        show_customer_data_ui = st.session_state.get(
            "show_customer_data_ui", False)
        if not show_verification_from_button and not show_draft_ui and not show_customer_data_ui:
            # ⭐ admin.py 스타일로 간소화: 깔끔한 레이아웃
            # 솔루션 제공 체크박스 (기능 유지)
            st.session_state.is_solution_provided = st.checkbox(
                L["solution_check_label"],
                value=st.session_state.is_solution_provided,
                key="solution_checkbox_widget",
            )

        # ⭐ 메시지 입력 칸은 항상 표시 (어떤 기능 버튼을 클릭해도 항상 표시)
        # 위젯 생성 전에 초기화 플래그 확인 및 처리
        # ⭐ [핵심 수정 1] 전사 결과가 있으면 초기화하지 않도록 보장
        if st.session_state.get("reset_agent_response_area", False):
            # 전사 결과가 없거나 (last_transcript가 비어 있거나, 전사 중이 아닐 때)만 초기화
            if not st.session_state.get(
                    "last_transcript") or not st.session_state.last_transcript:
                st.session_state.agent_response_area_text = ""
            st.session_state.reset_agent_response_area = False

        # ⭐ 마이크 전사 결과가 있으면 text_area에 표시 (호환성 유지)
        # 위젯 생성 전에만 값을 설정할 수 있으므로 여기서 처리
        # ⭐ [수정 1] 전사 결과가 입력 칸에 확실히 반영되도록 보장 (최우선 처리)
        if st.session_state.get(
                "last_transcript") and st.session_state.last_transcript:
            # 전사 결과를 text_area의 value로 사용되는 세션 상태 변수에 반영
            st.session_state.agent_response_area_text = st.session_state.last_transcript
            # 전사 결과를 반영했으므로, last_transcript는 전송 시점에 초기화하도록 유지
            # st.session_state.last_transcript = "" # *주의: 전송 로직에서 필요할 수 있으므로, 전송 시점에 초기화 고려
        # ⭐ [추가 수정] agent_response_area_text가 비어있고 last_transcript가 있으면 반영
        elif not st.session_state.get("agent_response_area_text") and st.session_state.get("last_transcript") and st.session_state.last_transcript:
            st.session_state.agent_response_area_text = st.session_state.last_transcript

        # --- UI 개선: app.py 스타일로 자연스러운 채팅 입력 (st.chat_input 사용) ---
        # ⭐ 메시지 입력 칸은 항상 표시 (어떤 기능 버튼을 클릭해도 항상 표시)

        # ⭐ [수정] 전사 결과가 있으면 자동으로 메시지로 전송되도록 처리
        if st.session_state.get(
                "last_transcript") and st.session_state.last_transcript:
            # 전사 결과를 자동으로 메시지로 전송
            agent_response_auto = st.session_state.last_transcript.strip()
            if agent_response_auto:
                # 전사 결과를 메시지로 추가
                st.session_state.simulator_messages.append({
                    "role": "agent_response",
                    "content": agent_response_auto
                })
                # 전사 결과 초기화
                st.session_state.last_transcript = ""
                st.session_state.agent_response_area_text = ""
                # 자동으로 고객 반응 생성
                if st.session_state.is_llm_ready:
                    with st.spinner(L["generating_customer_response"]):
                        customer_response = generate_customer_reaction(
                            st.session_state.language, is_call=False)
                        st.session_state.simulator_messages.append({
                            "role": "customer",
                            "content": customer_response
                        })

        # st.chat_input으로 입력 받기 (app.py 스타일)
        agent_response_input = st.chat_input(
            L.get("agent_response_placeholder", "고객에게 응답하세요..."))

        # 추가 기능 버튼들 (파일 첨부만) - 입력 영역 아래에 배치
        col_extra_features = st.columns([1, 1])

        with col_extra_features[0]:
            # (+) 파일 첨부 버튼
            if st.button(
                    L.get(
                        "button_add_attachment",
                        "➕ 파일 첨부"),
                    key="btn_add_attachment_unified",
                    use_container_width=True,
                    type="secondary"):
                st.session_state.show_agent_file_uploader = True

        with col_extra_features[1]:
            # 전사 결과 표시 (있는 경우)
            if st.session_state.get(
                    "agent_response_area_text") and st.session_state.agent_response_area_text:
                transcript_preview = st.session_state.agent_response_area_text[:30]
                st.caption(
                    L.get(
                        "transcription_label",
                        "💬 전사: {text}...").format(
                        text=transcript_preview))

        # 전송 로직 실행 (st.chat_input은 Enter 키 또는 전송 버튼으로 자동 전송됨)
        agent_response = None
        if agent_response_input:
            agent_response = agent_response_input.strip()

        # --- End of Unified Input UI ---

        if agent_response:
            if not agent_response.strip():
                st.warning(L["empty_response_warning"])
                # st.stop()
            else:
                # AHT 타이머 시작
                if st.session_state.start_time is None and len(
                        st.session_state.simulator_messages) >= 1:
                    st.session_state.start_time = datetime.now()

                # --- 에이전트 첨부 파일 처리 (다중 파일 처리) ---
                final_response_content = agent_response
                if st.session_state.agent_attachment_file:
                    file_infos = st.session_state.agent_attachment_file
                    file_names = ", ".join([f["name"] for f in file_infos])
                    attachment_msg = L["agent_attachment_status"].format(
                        filename=file_names, filetype=f"총 {len(file_infos)}개 파일"
                    )
                    final_response_content = f"{agent_response}\n\n---\n{attachment_msg}"

                # 로그 업데이트
                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": final_response_content}
                )

                # ⭐ 추가: 에이전트 응답에 메일 끝인사가 포함되어 있는지 확인
                email_closing_patterns = [
                    "추가 문의사항이 있으면 언제든지 연락",
                    "추가 문의 사항이 있으면 언제든지 연락",
                    "추가 문의사항이 있으시면",
                    "추가 문의 사항이 있으시면",
                    "언제든지 연락",
                    "언제든지 연락 주세요",
                    "additional inquiries",
                    "any additional questions",
                    "any further questions",
                    "feel free to contact",
                    "please feel free to contact",
                    "please don't hesitate to contact",
                    "don't hesitate to contact",
                    "please let me know",
                    "let me know",
                    "let me know if",
                    "please let me know so",
                    "let me know so",
                    "if you have any questions",
                    "if you have any further questions",
                    "if you need any assistance",
                    "if you need further assistance",
                    "if you encounter any issues",
                    "if you still have",
                    "if you remain unclear",
                    "I can assist further",
                    "I can help further",
                    "I can assist",
                    "so I can assist",
                    "so I can help",
                    "so I can assist further",
                    "追加のご質問",
                    "追加のお問い合わせ",
                    "ご質問がございましたら",
                    "お問い合わせがございましたら"]
                is_email_closing_in_response = any(pattern.lower(
                ) in final_response_content.lower() for pattern in email_closing_patterns)
                if is_email_closing_in_response:
                    st.session_state.has_email_closing = True  # 플래그 설정

                # 입력창/오디오/첨부 파일 초기화
                # ⭐ 수정: 위젯이 생성된 후에는 session_state를 직접 수정할 수 없으므로,
                # 플래그를 사용하여 위젯이 다시 생성될 때 초기값이 적용되도록 합니다.
                st.session_state.sim_audio_bytes = None
                st.session_state.agent_attachment_file = []  # 첨부 파일 초기화
                st.session_state.language_transfer_requested = False
                st.session_state.realtime_hint_text = ""  # 힌트 초기화
                st.session_state.sim_call_outbound_summary = ""  # 전화 발신 요약 초기화
                st.session_state.last_transcript = ""  # 전사 결과 초기화

                # ⭐ 수정: agent_response_area_text는 위젯이 다시 생성될 때 초기화되도록
                # 플래그만 설정합니다. 위젯 생성 전에 이 플래그를 확인하여 값을 초기화합니다.
                # 위젯이 생성된 후에는 직접 수정할 수 없으므로 플래그만 사용합니다.
                st.session_state.reset_agent_response_area = True

                # ⭐ 수정: 응답 전송 시 바로 고객 반응 자동 생성
                if st.session_state.is_llm_ready:
                    # LLM이 준비된 경우 바로 고객 반응 생성
                    with st.spinner(L["generating_customer_response"]):
                        customer_response = generate_customer_reaction(
                            st.session_state.language, is_call=False)

                    # 고객 반응을 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "customer", "content": customer_response}
                    )

                    # ⭐ 추가: 메일 끝인사가 포함된 경우 고객 응답 확인 및 설문 조사 버튼 활성화
                    if st.session_state.get("has_email_closing", False):
                        # 고객의 긍정 반응 확인
                        positive_keywords = [
                            "No, that will be all",
                            "no more",
                            "없습니다",
                            "감사합니다",
                            "Thank you",
                            "ありがとう",
                            "추가 문의 사항 없습니다",
                            "추가 문의사항 없습니다",
                            "no additional",
                            "追加の質問はありません",
                            "알겠습니다",
                            "알겠어요",
                            "ok",
                            "okay",
                            "네",
                            "yes",
                            "좋습니다",
                            "good",
                            "fine",
                            "괜찮습니다"]
                        is_positive = any(
                            keyword.lower() in customer_response.lower() for keyword in positive_keywords)

                        # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
                        import re
                        escaped = re.escape(
                            L.get('customer_no_more_inquiries', ''))
                        no_more_pattern = escaped.replace(
                            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                        no_more_regex = re.compile(
                            no_more_pattern, re.IGNORECASE)
                        if is_positive or no_more_regex.search(
                                customer_response):
                            # 설문 조사 버튼 활성화를 위해 WAIT_CUSTOMER_CLOSING_RESPONSE
                            # 단계로 이동
                            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                        else:
                            # 메일 끝인사가 있지만 고객이 추가 질문을 한 경우
                            st.session_state.sim_stage = "AGENT_TURN"
                    else:
                        # ⭐ 고객 응답에 따라 다음 단계 결정 (CUSTOMER_TURN 단계의 로직과 동일)
                        import re
                        escaped_no_more = re.escape(
                            L.get("customer_no_more_inquiries", ""))
                        no_more_pattern = escaped_no_more.replace(
                            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                        no_more_regex = re.compile(
                            no_more_pattern, re.IGNORECASE)
                        escaped_positive = re.escape(
                            L.get("customer_positive_response", ""))
                        positive_pattern = escaped_positive.replace(
                            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                        positive_regex = re.compile(
                            positive_pattern, re.IGNORECASE)
                        is_positive_closing = no_more_regex.search(
                            customer_response) is not None or positive_regex.search(customer_response) is not None

                        # 다음 단계 결정
                        if L.get(
                            "customer_positive_response",
                                "") in customer_response:
                            if st.session_state.get(
                                    "is_solution_provided", False):
                                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                            else:
                                st.session_state.sim_stage = "AGENT_TURN"
                        elif is_positive_closing:
                            if no_more_regex.search(customer_response):
                                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                            else:
                                if st.session_state.get(
                                        "is_solution_provided", False):
                                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                                else:
                                    st.session_state.sim_stage = "AGENT_TURN"
                        elif customer_response.startswith(L.get("customer_escalation_start", "")):
                            st.session_state.sim_stage = "ESCALATION_REQUIRED"
                        else:
                            # 고객이 추가 질문하거나 정보 제공한 경우 -> 에이전트 턴으로 이동
                            st.session_state.sim_stage = "AGENT_TURN"
                else:
                    # LLM이 없는 경우 플래그 설정하여 CUSTOMER_TURN 단계에서 수동 생성 가능하도록
                    st.session_state.need_customer_response = True
                    st.session_state.sim_stage = "CUSTOMER_TURN"

        # --- 언어 이관 버튼 ---
        st.markdown("---")
        st.markdown(f"**{L['transfer_header']}**")
        transfer_cols = st.columns(len(LANG) - 1)

        languages = list(LANG.keys())
        languages.remove(current_lang)

        def transfer_session(
                target_lang: str, current_messages: List[Dict[str, str]]):
            # 언어 이관 시스템 메시지를 추가하고 세션 언어를 변경합니다.
            # ⭐ 채팅도 전화와 마찬가지로 이전 대화가 끊어지고 새 언어 팀원으로 넘어갑니다.

            # 현재 언어 확인 및 L 변수 정의
            current_lang_at_start = st.session_state.language  # Source language
            L = LANG.get(current_lang_at_start, LANG["ko"])  # L 변수 정의 추가

            # API 키 체크는 run_llm 내부에서 처리되지만, 명시적으로 Gemini 키를 요구함
            if not get_api_key("gemini"):
                st.error(
                    L["simulation_no_key_warning"].replace(
                        'API Key', 'Gemini API Key'))
                # st.stop()
            else:
                # ⭐ [수정] 채팅도 이전 대화를 끊고 새 언어 팀원으로 넘어감
                # 대화 기록은 번역되어 유지되지만, 이전 에이전트의 응답은 종료됨

                # AHT 타이머 중지 및 초기화
                st.session_state.start_time = None

                # 1. 로딩 시작 (시간 양해 메시지 시뮬레이션)
                with st.spinner(L["transfer_loading"]):
                    # 실제 대기 시간 5~10초 (3~10분 시뮬레이션)
                    time.sleep(np.random.uniform(5, 10))

                    # ⭐ [수정] 원본 언어로 핵심 요약 생성 후 번역
                    try:
                        # 원본 언어로 핵심 요약 생성
                        original_summary = summarize_history_with_ai(
                            current_lang_at_start)

                        if not original_summary or original_summary.startswith(
                                "❌"):
                            # 요약 생성 실패 시 대화 기록을 번역할 텍스트로 가공
                            history_text = ""
                            for msg in current_messages:
                                role = "Customer" if msg["role"].startswith(
                                    "customer") or msg["role"] == "initial_query" else "Agent"
                                if msg["role"] in [
                                    "initial_query",
                                    "customer_rebuttal",
                                    "agent_response",
                                        "customer_closing_response"]:
                                    history_text += f"{role}: {msg['content']}\n"
                            original_summary = history_text

                        # 핵심 요약을 번역 대상 언어로 번역
                        translated_summary, is_success = translate_text_with_llm(
                            original_summary,
                            target_lang,
                            current_lang_at_start
                        )

                        if not translated_summary:
                            # 번역 실패 시 번역 대상 언어로 요약 재생성
                            translated_summary = summarize_history_with_ai(
                                target_lang)
                            is_success = True if translated_summary and not translated_summary.startswith(
                                "❌") else False

                        # ⭐ [핵심 수정] 모든 메시지를 이관된 언어로 번역
                        translated_messages = []
                        for msg in current_messages:
                            translated_msg = msg.copy()
                            # 번역할 메시지 역할 필터링 (시스템 메시지 등은 제외)
                            if msg["role"] in [
                                "initial_query",
                                "customer",
                                "customer_rebuttal",
                                "agent_response",
                                "customer_closing_response",
                                    "supervisor"]:
                                if msg.get("content"):
                                    # 각 메시지 내용을 이관된 언어로 번역
                                    try:
                                        translated_content, trans_success = translate_text_with_llm(
                                            msg["content"],
                                            target_lang,  # 이관된 언어로 번역
                                            current_lang_at_start  # 원본 언어
                                        )
                                        if trans_success:
                                            translated_msg["content"] = translated_content
                                        else:
                                            # 번역 실패 시 원본 유지
                                            pass
                                    except Exception as e:
                                        # 번역 오류 시 원본 유지
                                        pass
                            translated_messages.append(translated_msg)

                        # 번역된 메시지로 업데이트
                        st.session_state.simulator_messages = translated_messages

                        # 이관 요약 저장
                        st.session_state.transfer_summary_text = translated_summary
                        st.session_state.translation_success = is_success
                        st.session_state.language_at_transfer_start = current_lang_at_start

                        # 언어 변경
                        st.session_state.language = target_lang
                        L = LANG.get(target_lang, LANG["ko"])

                        # 언어 이름 가져오기
                        lang_name_target = {
                            "ko": "Korean",
                            "en": "English",
                            "ja": "Japanese"}.get(
                            target_lang,
                            "Korean")

                        # 시스템 메시지 추가
                        system_msg = L["transfer_system_msg"].format(
                            target_lang=lang_name_target)
                        st.session_state.simulator_messages.append(
                            {"role": "system_transfer", "content": system_msg}
                        )

                        # 이관 요약을 supervisor 메시지로 추가
                        summary_msg = f"### {L['transfer_summary_header']}\n\n{translated_summary}"
                        st.session_state.simulator_messages.append(
                            {"role": "supervisor", "content": summary_msg}
                        )

                        # 이력 저장
                        customer_type_display = st.session_state.get(
                            "customer_type_sim_select", "")
                        save_simulation_history_local(
                            st.session_state.customer_query_text_area,
                            customer_type_display,
                            st.session_state.simulator_messages,
                            is_chat_ended=False,
                            attachment_context=st.session_state.sim_attachment_context_for_llm,
                        )

                        # ⭐ [수정] 채팅도 새 언어 팀원으로 넘어가므로 WAIT_FIRST_QUERY로 초기화
                        # 번역된 메시지는 유지되지만, 새 에이전트가 응답을 시작할 수 있도록 초기화
                        st.session_state.sim_stage = "AGENT_TURN"
                        # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                        # st.rerun()
                    except Exception as e:
                        error_msg = L.get(
                            "transfer_error",
                            "이관 처리 중 오류 발생: {error}").format(
                            error=str(e))
                        st.error(error_msg)
                        summary_text = L.get(
                            "summary_generation_error",
                            "요약 생성 오류: {error}").format(
                            error=str(e))

        # 이관 버튼 렌더링
        for idx, lang_code in enumerate(languages):
            lang_name = {
                "ko": "Korean",
                "en": "English",
                "ja": "Japanese"}.get(
                lang_code,
                lang_code)
            transfer_label = L.get(
                f"transfer_to_{lang_code}",
                f"Transfer to {lang_name} Team")

            with transfer_cols[idx]:
                if st.button(
                        transfer_label,
                        key=f"btn_transfer_{lang_code}_{st.session_state.sim_instance_id}",
                        use_container_width=True):
                    transfer_session(
                        lang_code, st.session_state.simulator_messages)

    # =========================
    # 5-B. 에스컬레이션 요청 단계 (ESCALATION_REQUIRED)
    # =========================
    elif st.session_state.sim_stage == "ESCALATION_REQUIRED":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])

        st.warning(
            L.get(
                "escalation_required_msg",
                "🚨 고객이 에스컬레이션을 요청했습니다. 상급자나 전문 팀으로 이관이 필요합니다."))

        # 에스컬레이션 처리 옵션
        col_escalate, col_continue = st.columns(2)

        with col_escalate:
            if st.button(
                    L.get(
                        "button_escalate",
                        "에스컬레이션 처리"),
                    key=f"btn_escalate_{st.session_state.sim_instance_id}"):
                # 에스컬레이션 시스템 메시지 추가
                escalation_msg = L.get(
                    "escalation_system_msg",
                    "📌 시스템 메시지: 고객 요청에 따라 상급자/전문 팀으로 이관되었습니다.")
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": escalation_msg}
                )

                # 이력 저장
                customer_type_display = st.session_state.get(
                    "customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=True,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )

                # 종료 단계로 이동
                st.session_state.sim_stage = "CLOSING"

        with col_continue:
            if st.button(
                    L.get(
                        "button_continue",
                        "계속 응대"),
                    key=f"btn_continue_{st.session_state.sim_instance_id}"):
                # 계속 응대하는 경우 AGENT_TURN으로 이동
                st.session_state.sim_stage = "AGENT_TURN"

    # =========================
    # 6. 고객 반응 생성 단계 (CUSTOMER_TURN)
    # =========================
    elif st.session_state.sim_stage == "CUSTOMER_TURN":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get(
            "customer_type_sim_select", L["customer_type_options"][0])
        st.info(L["customer_turn_info"])

        # 1. 고객 반응 생성
        # 이미 고객 반응이 생성되어 있는지 확인
        last_customer_message = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "customer" and msg.get("content"):
                last_customer_message = msg.get("content", "")
                break

        if last_customer_message is None:
            # 고객 반응이 없는 경우에만 생성
            with st.spinner(L["generating_customer_response"]):
                customer_response = generate_customer_reaction(
                    st.session_state.language, is_call=False)

            # 2. 대화 로그 업데이트
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_response}
            )

            # 3. 생성 직후 바로 다음 단계 결정
            # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
            import re
            escaped_no_more = re.escape(L["customer_no_more_inquiries"])
            no_more_pattern = escaped_no_more.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
            escaped_positive = re.escape(L["customer_positive_response"])
            positive_pattern = escaped_positive.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            positive_regex = re.compile(positive_pattern, re.IGNORECASE)
            is_positive_closing = no_more_regex.search(
                customer_response) is not None or positive_regex.search(customer_response) is not None

            # 다음 단계 결정
            if L["customer_positive_response"] in customer_response:
                if st.session_state.is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    st.session_state.sim_stage = "AGENT_TURN"
            elif is_positive_closing:
                # ⭐ 정규표현식으로 종료 키워드 인식
                import re
                escaped = re.escape(L['customer_no_more_inquiries'])
                no_more_pattern = escaped.replace(
                    r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
                if no_more_regex.search(customer_response):
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    if st.session_state.is_solution_provided:
                        st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                    else:
                        st.session_state.sim_stage = "AGENT_TURN"
            elif customer_response.startswith(L["customer_escalation_start"]):
                st.session_state.sim_stage = "ESCALATION_REQUIRED"
            else:
                # 고객이 추가 질문하거나 정보 제공한 경우 -> 에이전트 턴으로 이동
                st.session_state.sim_stage = "AGENT_TURN"
        else:
            customer_response = last_customer_message

        # 3. 종료 조건 검토 (이미 고객 반응이 있는 경우)
        # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
        import re
        escaped_no_more = re.escape(L["customer_no_more_inquiries"])
        no_more_pattern = escaped_no_more.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
        escaped_positive = re.escape(L["customer_positive_response"])
        positive_pattern = escaped_positive.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        positive_regex = re.compile(positive_pattern, re.IGNORECASE)
        is_positive_closing = no_more_regex.search(
            customer_response) is not None or positive_regex.search(customer_response) is not None

        # ⭐ 추가: 메일 응대 종료 문구 확인 (플래그 또는 에이전트의 마지막 응답 확인)
        # 먼저 플래그 확인 (에이전트 응답 전송 시 설정됨)
        is_email_closing = st.session_state.get("has_email_closing", False)

        # 플래그가 없으면 에이전트의 마지막 응답에서 직접 확인
        if not is_email_closing:
            last_agent_response = None
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") == "agent_response" and msg.get("content"):
                    last_agent_response = msg.get("content", "")
                    break

            # 메일 끝인사 문구 패턴 (다국어 지원) - 더 포괄적인 패턴 추가
            email_closing_patterns = [
                "추가 문의사항이 있으면 언제든지 연락",
                "추가 문의 사항이 있으면 언제든지 연락",
                "추가 문의사항이 있으시면",
                "추가 문의 사항이 있으시면",
                "언제든지 연락",
                "언제든지 연락 주세요",
                "additional inquiries",
                "any additional questions",
                "any further questions",
                "feel free to contact",
                "please feel free to contact",
                "please don't hesitate to contact",
                "don't hesitate to contact",
                "please let me know",
                "let me know",
                "let me know if",
                "please let me know so",
                "let me know so",
                "if you have any questions",
                "if you have any further questions",
                "if you need any assistance",
                "if you need further assistance",
                "if you encounter any issues",
                "if you still have",
                "if you remain unclear",
                "I can assist further",
                "I can help further",
                "I can assist",
                "so I can assist",
                "so I can help",
                "so I can assist further",
                "追加のご質問",
                "追加のお問い合わせ",
                "ご質問がございましたら",
                "お問い合わせがございましたら"]

            if last_agent_response:
                is_email_closing = any(pattern.lower() in last_agent_response.lower(
                ) for pattern in email_closing_patterns)
                if is_email_closing:
                    st.session_state.has_email_closing = True  # 플래그 업데이트

        # ⭐ 수정: 메일 끝인사가 포함된 경우, 고객의 긍정 반응이나 "추가 문의 사항 없습니다" 답변을 인식하면 설문 조사 링크 전송 버튼 자동 활성화
        if is_email_closing:
            # 고객의 긍정 반응 또는 "추가 문의 사항 없습니다" 답변 확인
            # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
            import re
            no_more_keywords = [
                L['customer_no_more_inquiries'],
                "No, that will be all",
                "no more",
                "없습니다",
                "감사합니다",
                "Thank you",
                "ありがとう",
                "추가 문의 사항 없습니다",
                "추가 문의사항 없습니다",
                "no additional",
                "追加の質問はありません",
                "알겠습니다",
                "알겠어요",
                "ok",
                "okay",
                "네",
                "yes"
            ]
            # 각 키워드를 정규표현식으로 변환하여 검색
            has_no_more_inquiry = False
            for keyword in no_more_keywords:
                escaped = re.escape(keyword)
                pattern = escaped.replace(
                    r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                regex = re.compile(pattern, re.IGNORECASE)
                if regex.search(customer_response):
                    has_no_more_inquiry = True
                    break
            # "없습니다"와 "감사합니다"가 함께 있는 경우도 인식
            if "없습니다" in customer_response and "감사합니다" in customer_response:
                has_no_more_inquiry = True

            # 긍정 반응 키워드 추가 (더 포괄적인 인식)
            positive_keywords = [
                "알겠습니다",
                "알겠어요",
                "네",
                "yes",
                "ok",
                "okay",
                "감사합니다",
                "thank you",
                "ありがとう",
                "좋습니다",
                "good",
                "fine",
                "괜찮습니다",
                "알겠습니다 감사합니다"]
            is_positive_response = any(
                keyword.lower() in customer_response.lower() for keyword in positive_keywords)

            # 긍정 반응이 있거나 "추가 문의 사항 없습니다" 답변이 있으면 설문 조사 링크 전송 버튼 활성화
            # ⭐ 정규표현식으로 종료 키워드 인식
            escaped_check = re.escape(L['customer_no_more_inquiries'])
            no_more_pattern_check = escaped_check.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex_check = re.compile(
                no_more_pattern_check, re.IGNORECASE)
            if is_positive_closing or has_no_more_inquiry or no_more_regex_check.search(
                    customer_response) or is_positive_response:
                # 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break

                if not agent_closing_added:
                    # 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )

                # 설문 조사 링크 전송 버튼 활성화를 위해 WAIT_CUSTOMER_CLOSING_RESPONSE 단계로 이동
                # (실제로는 고객 응답이 이미 있으므로 바로 설문 조사 버튼 표시)
                st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
            else:
                # 메일 끝인사가 있지만 고객이 추가 질문을 한 경우
                st.session_state.sim_stage = "AGENT_TURN"
        # ⭐ 수정: 고객이 "알겠습니다. 감사합니다"라고 답변했을 때, 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
        # 정확한 문자열 비교가 아닌 포함 여부로 확인 (LLM 응답이 약간 다를 수 있음)
        # "알겠습니다"와 "감사합니다"가 함께 있는 경우를 더 명확하게 인식
        elif L["customer_positive_response"] in customer_response or ("알겠습니다" in customer_response and "감사합니다" in customer_response):
            # 솔루션이 제공된 경우에만 추가 문의 여부 확인 단계로 이동
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # 솔루션이 제공되지 않은 경우 에이전트 턴으로 유지
                st.session_state.sim_stage = "AGENT_TURN"
        elif is_positive_closing:
            # 긍정 종료 응답 처리
            # ⭐ 정규표현식으로 종료 키워드 인식
            import re
            escaped = re.escape(L['customer_no_more_inquiries'])
            no_more_pattern = escaped.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
            if no_more_regex.search(customer_response):
                # ⭐ 수정: "없습니다. 감사합니다" 답변 시 에이전트가 감사 인사를 한 후 종료하도록 변경
                # 바로 종료하지 않고 WAIT_CLOSING_CONFIRMATION_FROM_AGENT 단계로 이동하여
                # 에이전트가 감사 인사 후 종료
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                # "알겠습니다. 감사합니다"와 유사한 긍정 응답인 경우, 솔루션 제공 여부 확인
                if st.session_state.is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    st.session_state.sim_stage = "AGENT_TURN"

        # ⭐ 수정: 고객이 아직 솔루션에 만족하지 않거나 추가 질문을 한 경우 (일반적인 턴)
        elif customer_response.startswith(L["customer_escalation_start"]):
            st.session_state.sim_stage = "ESCALATION_REQUIRED"  # 에스컬레이션 필요
        else:
            # 에이전트 턴으로 유지 (고객이 추가 질문하거나 정보 제공)
            st.session_state.sim_stage = "AGENT_TURN"

        st.session_state.is_solution_provided = False  # 종료 단계 진입 후 플래그 리셋

        # 이력 저장 (종료되지 않은 경우에만 저장)
        # ⭐ 수정: "없습니다. 감사합니다" 답변 시에는 이미 이력 저장을 했으므로 중복 저장 방지
        if st.session_state.sim_stage != "CLOSING":
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.session_state.realtime_hint_text = ""  # 힌트 초기화
        # ⭐ 재실행 불필요: 힌트 초기화만으로 충분, 자동 업데이트됨
        # st.rerun()

    # =========================
    # 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        st.success(
            L.get(
                "customer_positive_solution_reaction",
                "고객이 솔루션에 만족했습니다."))

        # ⭐ 버튼들을 메시지 말풍선 스타일로 표시 (간소화)
        st.info(
            L.get(
                "info_use_buttons",
                "💡 아래 버튼을 사용하여 추가 문의 여부를 확인하거나 상담을 종료하세요."))

        col_chat_end, col_email_end = st.columns(2)  # 버튼을 나란히 배치

        # [1] 채팅 - 추가 문의 확인 메시지 보내기 버튼
        with col_chat_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(
                    L.get(
                        "send_closing_confirm_button",
                        "✅ 추가 문의 있나요?"),
                    key=f"btn_send_closing_confirm_{st.session_state.sim_instance_id}",
                    use_container_width=True):
                # ⭐ 수정: 에이전트가 감사 인사를 포함한 종료 메시지 전송
                # 언어별 감사 인사 메시지 생성
                agent_name = st.session_state.get("agent_name", "000")
                if current_lang == "ko":
                    closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. {L.get('customer_closing_confirm', '추가 문의사항이 있으시면 언제든지 연락 주세요.')} 즐거운 하루 되세요."
                elif current_lang == "en":
                    closing_msg = f"Thank you for contacting us. This was {agent_name}. {L.get('customer_closing_confirm', 'Please feel free to contact us if you have any additional questions.')} Have a great day!"
                else:  # ja
                    closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。{L.get('customer_closing_confirm', '追加のご質問がございましたら、お気軽にお問い合わせください。')} 良い一日をお過ごしください。"

                # 에이전트 응답으로 로그 기록
                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": closing_msg}
                )

                # ⭐ time.sleep 제거: 불필요한 지연
                st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
                # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                # st.rerun()

        # [2] 이메일 - 상담 종료 버튼 (즉시 종료)
        with col_email_end:
            # [수정 1] 다국어 레이블 사용
            if st.button(
                    L.get(
                        "button_email_end_chat",
                        "📋 설문 조사 전송 및 종료"),
                    key=f"btn_email_end_chat_{st.session_state.sim_instance_id}",
                    use_container_width=True,
                    type="primary"):
                # AHT 타이머 정지
                st.session_state.start_time = None

                # [수정 1] 다국어 레이블 사용
                end_msg = L.get("prompt_survey", "설문 조사 링크를 전송했습니다.")
                st.session_state.simulator_messages.append(
                    {"role": "system_end", "content": "(시스템: 이메일 상담 종료) " + end_msg}
                )

                time.sleep(0.1)
                st.session_state.is_chat_ended = True
                st.session_state.sim_stage = "CLOSING"  # 바로 CLOSING으로 전환

                # 이력 저장
                customer_type_display = st.session_state.get(
                    "customer_type_sim_select", "")
                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=True,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                # ⭐ 재실행 불필요: 채팅 종료 상태는 이미 반영됨, 다음 렌더링에서 자동 표시됨
                # st.rerun()

    # =========================
    # 8. 고객 최종 응답 생성 및 처리 (WAIT_CUSTOMER_CLOSING_RESPONSE)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CUSTOMER_CLOSING_RESPONSE":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get(
            "customer_type_sim_select", L["customer_type_options"][0])

        # ⭐ 추가: 메일 응대 종료 문구 확인 (에이전트의 마지막 응답에 "추가 문의사항이 있으면 언제든지 연락 주세요" 같은 문구가 포함되어 있는지 확인)
        last_agent_response = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "agent_response" and msg.get("content"):
                last_agent_response = msg.get("content", "")
                break

        # 메일 끝인사 문구 패턴 (다국어 지원) - 더 포괄적인 패턴 추가
        email_closing_patterns = [
            "추가 문의사항이 있으면 언제든지 연락",
            "추가 문의 사항이 있으면 언제든지 연락",
            "추가 문의사항이 있으시면 언제든지 연락",
            "추가 문의 사항이 있으시면 언제든지 연락",
            "추가 문의사항이 있으시면",
            "추가 문의 사항이 있으시면",
            "추가 문의사항이 있으면",
            "추가 문의 사항이 있으면",
            "언제든지 연락",
            "언제든지 연락 주세요",
            "언제든지 연락 주시기 바랍니다",
            "additional inquiries",
            "any additional questions",
            "any further questions",
            "feel free to contact",
            "please feel free to contact",
            "please don't hesitate to contact",
            "don't hesitate to contact",
            "追加のご質問",
            "追加のお問い合わせ",
            "ご質問がございましたら",
            "お問い合わせがございましたら"
        ]

        is_email_closing = False
        if last_agent_response:
            is_email_closing = any(pattern.lower() in last_agent_response.lower(
            ) for pattern in email_closing_patterns)

        # ⭐ 수정: 이미 고객 응답이 생성되어 있는지 확인
        last_customer_message = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "customer_rebuttal":
                last_customer_message = msg.get("content", "")
                break
            # ⭐ 추가: customer 역할의 메시지도 확인 (메일 끝인사가 포함된 경우 CUSTOMER_TURN에서 이미 고객 응답이 생성되었을 수 있음)
            elif msg.get("role") == "customer" and is_email_closing:
                last_customer_message = msg.get("content", "")
                break

        # 고객 응답이 아직 생성되지 않은 경우에만 생성
        if last_customer_message is None:
            # 고객 답변 자동 생성 (LLM Key 검증 포함)
            if not st.session_state.is_llm_ready:
                st.warning(L["llm_key_missing_customer_response"])
                if st.button(
                        L["customer_generate_response_button"],
                        key="btn_generate_final_response"):
                    st.session_state.sim_stage = "AGENT_TURN"
                    # ⭐ 재실행 불필요: 버튼 클릭 시 자동으로 상태 전환됨
                    # st.rerun()
                st.stop()

            # LLM이 준비된 경우 고객 응답 생성
            st.info(L["agent_confirmed_additional_inquiry"])
            with st.spinner(L["generating_customer_response"]):
                final_customer_reaction = generate_customer_closing_response(
                    st.session_state.language)

            # 로그 기록
            st.session_state.simulator_messages.append(
                {"role": "customer_rebuttal", "content": final_customer_reaction}
            )
            last_customer_message = final_customer_reaction

        # 고객 응답에 따라 처리 (생성 직후 또는 이미 있는 경우 모두 처리)
        if last_customer_message is None:
            # 고객 응답이 없는 경우 (이미 생성했는데도 None인 경우는 에러)
            st.warning(L["customer_response_generation_failed"])
        else:
            final_customer_reaction = last_customer_message

            # (A) "없습니다. 감사합니다" 경로 -> 에이전트가 감사 인사 후 버튼 표시
            # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
            import re
            no_more_keywords = [
                L['customer_no_more_inquiries'],
                "No, that will be all",
                "no more",
                "없습니다",
                "감사합니다",
                "結構です",
                "ありがとう",
                "추가 문의 사항 없습니다",
                "추가 문의사항 없습니다",
                "no additional",
                "追加の質問はありません"
            ]
            # 각 키워드를 정규표현식으로 변환하여 검색
            has_no_more_inquiry = False
            for keyword in no_more_keywords:
                escaped = re.escape(keyword)
                pattern = escaped.replace(
                    r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                regex = re.compile(pattern, re.IGNORECASE)
                if regex.search(final_customer_reaction):
                    has_no_more_inquiry = True
                    break

            # ⭐ 추가: 메일 끝인사가 포함된 경우, 고객의 긍정 반응이나 "추가 문의 사항 없습니다" 답변을 인식하면 설문 조사 링크 전송 버튼 자동 활성화
            # 긍정 반응 키워드 추가
            positive_keywords = [
                "알겠습니다",
                "알겠어요",
                "네",
                "yes",
                "ok",
                "okay",
                "감사합니다",
                "thank you",
                "ありがとう"]
            is_positive_response = any(keyword.lower(
            ) in final_customer_reaction.lower() for keyword in positive_keywords)

            # ⭐ 정규표현식으로 종료 키워드 인식
            escaped_check = re.escape(L['customer_no_more_inquiries'])
            no_more_pattern_check = escaped_check.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex_check = re.compile(
                no_more_pattern_check, re.IGNORECASE)
            if is_email_closing and (has_no_more_inquiry or no_more_regex_check.search(
                    final_customer_reaction) or is_positive_response):
                # 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break

                if not agent_closing_added:
                    # 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )

                # 설문 조사 링크 전송 버튼 표시
                st.markdown("---")
                st.success(L["no_more_inquiries_confirmed"])
                st.markdown(f"### {L['consultation_end_header']}")
                st.info(L["click_survey_button_to_end"])
                st.markdown("---")

                # 버튼을 중앙에 크게 표시
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    end_chat_button = st.button(
                        L["sim_end_chat_button"],
                        key="btn_final_end_chat_email_closing",
                        use_container_width=True,
                        type="primary"
                    )

                if end_chat_button:
                    # AHT 타이머 정지
                    st.session_state.start_time = None

                    # 설문 조사 링크 전송 메시지 추가
                    end_msg = L["prompt_survey"]
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": end_msg}
                    )

                    # 채팅 종료 처리
                    st.session_state.is_chat_ended = True
                    st.session_state.sim_stage = "CLOSING"

                    # 이력 저장
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )

                    st.session_state.realtime_hint_text = ""  # 힌트 초기화
            # 메일 끝인사가 포함된 경우 여기서 처리 완료, 다른 로직은 실행하지 않음
            # ⭐ 정규표현식으로 종료 키워드 인식 (메일 끝인사가 아닌 경우)
            elif not is_email_closing:
                import re
                escaped_final = re.escape(L['customer_no_more_inquiries'])
                no_more_pattern_final = escaped_final.replace(
                    r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                no_more_regex_final = re.compile(
                    no_more_pattern_final, re.IGNORECASE)
                if no_more_regex_final.search(
                        final_customer_reaction) or has_no_more_inquiry:
                    # ⭐ 수정: 에이전트 감사 인사가 아직 추가되지 않은 경우에만 추가
                    agent_closing_added = False
                    for msg in reversed(st.session_state.simulator_messages):
                        if msg.get("role") == "agent_response":
                            # 이미 에이전트 감사 인사가 있는지 확인
                            agent_msg_content = msg.get("content", "")
                            if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                                agent_closing_added = True
                            break

                    if not agent_closing_added:
                        # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                        agent_name = st.session_state.get("agent_name", "000")
                        if current_lang == "ko":
                            agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                        elif current_lang == "en":
                            agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                        else:  # ja
                            agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

                        # 에이전트 감사 인사를 메시지에 추가
                        st.session_state.simulator_messages.append(
                            {"role": "agent_response", "content": agent_closing_msg}
                        )

                    # ⭐ 수정: 현재 단계에서 바로 버튼 표시 (FINAL_CLOSING_ACTION으로 이동하지 않음)
                    st.markdown("---")
                    st.success(L["no_more_inquiries_confirmed"])
                    st.markdown(f"### {L['consultation_end_header']}")
                    st.info(L["click_survey_button_to_end"])
                    st.markdown("---")

                    # 버튼을 중앙에 크게 표시
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        end_chat_button = st.button(
                            L["sim_end_chat_button"],
                            key="btn_final_end_chat_in_wait",
                            use_container_width=True,
                            type="primary"
                        )

                    if end_chat_button:
                        # AHT 타이머 정지
                        st.session_state.start_time = None

                        # 설문 조사 링크 전송 메시지 추가
                        end_msg = L["prompt_survey"]
                        st.session_state.simulator_messages.append(
                            {"role": "system_end", "content": end_msg}
                        )

                        # 채팅 종료 처리
                        st.session_state.is_chat_ended = True
                        st.session_state.sim_stage = "CLOSING"

                        # 이력 저장
                        save_simulation_history_local(
                            st.session_state.customer_query_text_area,
                            customer_type_display,
                            st.session_state.simulator_messages,
                            is_chat_ended=True,
                            attachment_context=st.session_state.sim_attachment_context_for_llm,
                        )

                        st.session_state.realtime_hint_text = ""  # 힌트 초기화
            # (B) "추가 문의 사항도 있습니다" 경로 -> AGENT_TURN으로 복귀
            elif L['customer_has_additional_inquiries'] in final_customer_reaction:
                st.session_state.sim_stage = "AGENT_TURN"
                save_simulation_history_local(
                    st.session_state.customer_query_text_area,
                    customer_type_display,
                    st.session_state.simulator_messages,
                    is_chat_ended=False,
                    attachment_context=st.session_state.sim_attachment_context_for_llm,
                )
                st.session_state.realtime_hint_text = ""
            else:
                # 고객 응답이 생성되었지만 조건에 맞지 않는 경우에도 버튼 표시
                # (기본적으로 "없습니다. 감사합니다"로 간주)
                # ⭐ 수정: fallback 경로에서도 에이전트 감사 인사 메시지 추가
                agent_closing_added = False
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") == "agent_response":
                        # 이미 에이전트 감사 인사가 있는지 확인
                        agent_msg_content = msg.get("content", "")
                        if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                            agent_closing_added = True
                        break

                if not agent_closing_added:
                    # ⭐ 추가: 에이전트가 감사 인사 메시지 전송
                    agent_name = st.session_state.get("agent_name", "000")
                    if current_lang == "ko":
                        agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                    elif current_lang == "en":
                        agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                    else:  # ja
                        agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

                    # 에이전트 감사 인사를 메시지에 추가
                    st.session_state.simulator_messages.append(
                        {"role": "agent_response", "content": agent_closing_msg}
                    )

                st.markdown("---")
                st.success(L["no_more_inquiries_confirmed"])
                st.markdown(f"### {L['consultation_end_header']}")
                st.info(L["click_survey_button_to_end"])
                st.markdown("---")

                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    end_chat_button = st.button(
                        L["sim_end_chat_button"],
                        key="btn_final_end_chat_fallback",
                        use_container_width=True,
                        type="primary"
                    )

                if end_chat_button:
                    # AHT 타이머 정지
                    st.session_state.start_time = None

                    # 설문 조사 링크 전송 메시지 추가
                    end_msg = L["prompt_survey"]
                    st.session_state.simulator_messages.append(
                        {"role": "system_end", "content": end_msg}
                    )

                    # 채팅 종료 처리
                    st.session_state.is_chat_ended = True
                    st.session_state.sim_stage = "CLOSING"

                    # 이력 저장
                    save_simulation_history_local(
                        st.session_state.customer_query_text_area,
                        customer_type_display,
                        st.session_state.simulator_messages,
                        is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )

                    st.session_state.realtime_hint_text = ""  # 힌트 초기화

    # =========================
    # 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
    # =========================
    elif st.session_state.sim_stage == "FINAL_CLOSING_ACTION":
        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])

        # ⭐ 수정: 명확한 안내 메시지와 함께 버튼 표시
        st.markdown("---")
        st.success(L["no_more_inquiries_confirmed"])
        st.markdown(f"### {L['consultation_end_header']}")
        st.info(L["click_survey_button_to_end"])
        st.markdown("---")

        # 버튼을 중앙에 크게 표시
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            end_chat_button = st.button(
                L["sim_end_chat_button"],
                key="btn_final_end_chat",
                use_container_width=True,
                type="primary"
            )

        if end_chat_button:
            # AHT 타이머 정지
            st.session_state.start_time = None

            # 설문 조사 링크 전송 메시지 추가
            end_msg = L["prompt_survey"]
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

            # 채팅 종료 처리
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"

            # 이력 저장
            customer_type_display = st.session_state.get(
                "customer_type_sim_select", L["customer_type_options"][0])
            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

            st.session_state.realtime_hint_text = ""  # 힌트 초기화

# ========================================
# 전화 시뮬레이터 로직
# ========================================

