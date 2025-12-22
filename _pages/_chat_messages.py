# ========================================
# _pages/_chat_messages.py
# 채팅 시뮬레이터 - 대화 로그 표시 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import (
    generate_realtime_hint, _generate_initial_advice,
    generate_customer_reaction, save_simulation_history_local,
    translate_text_with_llm
)
from simulation_handler import render_tts_button
import re


def render_chat_messages(L, current_lang):
    """대화 로그 표시 및 메시지 렌더링 (카카오톡 스타일)"""
    # 피드백 저장 콜백 함수
    def save_feedback(index):
        feedback_key = f"feedback_{st.session_state.sim_instance_id}_{index}"
        if feedback_key in st.session_state:
            feedback_value = st.session_state[feedback_key]
            if index < len(st.session_state.simulator_messages):
                st.session_state.simulator_messages[index]["feedback"] = feedback_value

    # 카카오톡 스타일 채팅 컨테이너
    st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 10px;
    }
    .message-bubble {
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 70%;
        word-wrap: break-word;
    }
    .message-customer {
        background-color: #FEE500;
        margin-left: auto;
        text-align: right;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .customer-bubble-expanded {
        min-width: 320px !important;
        max-width: 80% !important;
        padding: 15px !important;
        padding-bottom: 70px !important;
        position: relative !important;
    }
    .customer-message-content {
        text-align: right;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(0,0,0,0.2);
        line-height: 1.5;
    }
    .customer-button-area {
        position: absolute;
        bottom: 10px;
        right: 25px;
        left: 10px;
        display: flex;
        justify-content: flex-end;
        gap: 4px;
        flex-wrap: wrap;
        padding-top: 8px;
        min-height: 50px;
    }
    .message-agent {
        background-color: #FFFFFF;
        margin-right: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .message-supervisor {
        background-color: #E8F5E9;
        margin: 10px auto;
        max-width: 90%;
        font-size: 0.9em;
    }
    .icon-button {
        background: none;
        border: none;
        font-size: 1.2em;
        cursor: pointer;
        padding: 5px;
        margin: 0 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 메시지 표시 (카카오톡 스타일)
    if st.session_state.simulator_messages:
        chat_container = st.container()
        with chat_container:
            for idx, msg in enumerate(st.session_state.simulator_messages):
                role = msg["role"]
                content = msg["content"]

                # 시스템 메시지는 제외
                if role in ["system_end", "system_transfer"]:
                    continue

                # 카카오톡 스타일 말풍선
                if role == "customer" or role == "customer_rebuttal" or role == "initial_query":
                    # 고객 메시지 (오른쪽 정렬, 노란색) - 버튼을 말풍선 안에 통합
                    _render_customer_message_with_icons(L, idx, content, current_lang)

                elif role == "agent_response":
                    # 에이전트 메시지 (왼쪽 정렬, 흰색)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin: 5px 0;">
                        <div class="message-bubble message-agent">
                            <div>{content.replace(chr(10), '<br>')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 에이전트 응답 아이콘 버튼들
                    col_icons = st.columns([1, 1, 1, 1, 1])
                    with col_icons[0]:
                        tts_role = "agent"
                        render_tts_button(
                            content,
                            st.session_state.language,
                            role=tts_role,
                            prefix=f"{role}_",
                            index=idx)
                    
                    with col_icons[1]:
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

                elif role == "supervisor":
                    # Supervisor 메시지 (중앙, 연한 초록색)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center; margin: 10px 0;">
                        <div class="message-bubble message-supervisor">
                            <div>{content.replace(chr(10), '<br>')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 고객 첨부 파일 표시
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

    # 이관 요약 표시
    # 이관 후에는 현재 언어로 다시 설정 (번역 반영을 위해)
    actual_current_lang = st.session_state.get("language", current_lang)
    if actual_current_lang not in ["ko", "en", "ja"]:
        actual_current_lang = "ko"
    actual_L = LANG.get(actual_current_lang, LANG["ko"])
    
    show_guideline_ui = st.session_state.get(
        "show_draft_ui", False) or st.session_state.get(
        "show_customer_data_ui", False)
    should_show_transfer_summary = (
        (st.session_state.transfer_summary_text or (st.session_state.language != st.session_state.language_at_transfer_start and st.session_state.language_at_transfer_start)) and
        st.session_state.sim_stage != "AGENT_TURN" and not show_guideline_ui
    )
    if should_show_transfer_summary:
        _render_transfer_summary(actual_L, actual_current_lang)


def _render_customer_message_with_icons(L, idx, content, current_lang):
    """고객 메시지와 아이콘 버튼들을 말풍선 안에 통합하여 렌더링"""
    # 이관 후에는 현재 언어로 다시 설정 (번역 반영을 위해)
    actual_current_lang = st.session_state.get("language", current_lang)
    if actual_current_lang not in ["ko", "en", "ja"]:
        actual_current_lang = "ko"
    actual_L = LANG.get(actual_current_lang, LANG["ko"])
    # 함수 내에서 L을 actual_L로 재할당하여 모든 곳에서 사용
    L = actual_L
    current_lang = actual_current_lang
    
    # 말풍선과 버튼을 하나의 컨테이너로 통합
    message_wrapper = st.container()
    with message_wrapper:
        # 고객 메시지 말풍선 (확장된 크기, 버튼 영역 포함)
        # 말풍선을 더 크게 만들고 버튼 공간 확보
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 5px 0;">
            <div class="message-bubble message-customer customer-bubble-expanded" style="position: relative;">
                <div class="customer-message-content">
                    {content.replace(chr(10), '<br>')}
                </div>
                <div class="customer-button-area" id="button-area-{idx}">
                    <span style="font-size: 0.75em; color: #666; margin-right: 8px; align-self: center; font-weight: 500;">기능:</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 버튼들을 말풍선 안에 배치하기 위해 음수 마진 사용
        # 말풍선과 같은 위치에 버튼 배치하여 시각적으로 말풍선 안에 있게 함
        st.markdown(f"""
        <style>
        .button-wrapper-{idx} {{
            margin-top: -60px;
            margin-right: 10%;
            display: flex;
            justify-content: flex-end;
            position: relative;
            z-index: 100;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # 버튼들을 말풍선 안에 배치 (음수 마진으로 말풍선 위에 올림)
        btn_wrapper = st.container()
        with btn_wrapper:
            # 오른쪽 정렬을 위한 빈 공간 + 버튼 그리드
            # 말풍선과 같은 너비로 맞추기 (약 80% 너비, 오른쪽 정렬)
            btn_cols = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
            
            # 첫 번째 줄 버튼들 (오른쪽 정렬)
            btn_grid_row1 = [btn_cols[1], btn_cols[2], btn_cols[3], btn_cols[4]]
            # 두 번째 줄 버튼들
            btn_grid_row2 = [btn_cols[5], btn_cols[6], btn_cols[7]]

    # 첫 번째 줄 버튼들
        # 응대 힌트 아이콘 버튼
        with btn_grid_row1[0]:
            if st.button(
                    "💡",
                    key=f"hint_icon_{idx}_{st.session_state.sim_instance_id}",
                    help=L.get("button_hint", "응대 힌트"),
                    use_container_width=False):
                if st.session_state.is_llm_ready:
                    st.session_state.show_verification_ui = False
                    st.session_state.show_draft_ui = False
                    st.session_state.show_customer_data_ui = False
                    st.session_state.show_agent_response_ui = False

                    hint_label = L.get('hint_label', '응대 힌트')
                    st.session_state.simulator_messages = [
                        msg for msg in st.session_state.simulator_messages if not (
                            msg.get("role") == "supervisor" and hint_label in msg.get(
                                "content", ""))]

                    session_lang = st.session_state.get("language", current_lang)
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = current_lang

                    with st.spinner(L.get("response_generating", "생성 중...")):
                        hint = generate_realtime_hint(
                            session_lang, is_call=False)
                        st.session_state.realtime_hint_text = hint
                        st.session_state.simulator_messages.append({
                            "role": "supervisor",
                            "content": f"💡 **{L.get('hint_label', '응대 힌트')}**: {hint}"
                        })
                else:
                    st.warning(
                        L.get(
                            "simulation_no_key_warning",
                            "LLM이 준비되지 않았습니다."))

        # 업체에 전화 아이콘 버튼
        with btn_grid_row1[1]:
            if st.button(
                    "📞",
                    key=f"call_provider_icon_{idx}_{st.session_state.sim_instance_id}",
                    help=L.get("button_call_company", "업체에 전화"),
                    use_container_width=False):
                st.session_state.show_verification_ui = False
                st.session_state.show_draft_ui = False
                st.session_state.show_customer_data_ui = False
                st.session_state.show_agent_response_ui = False
                st.session_state.sim_call_outbound_target = L.get(
                    "call_target_provider", "현지 업체/파트너")
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"

        # 고객에게 전화 아이콘 버튼
        with btn_grid_row1[2]:
            if st.button(
                    "📱",
                    key=f"call_customer_icon_{idx}_{st.session_state.sim_instance_id}",
                    help=L.get("button_call_customer", "고객에게 전화"),
                    use_container_width=False):
                st.session_state.show_verification_ui = False
                st.session_state.show_draft_ui = False
                st.session_state.show_customer_data_ui = False
                st.session_state.show_agent_response_ui = False
                st.session_state.sim_call_outbound_target = L.get(
                    "call_target_customer", "고객")
                st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"

        # AI 응대 가이드라인 아이콘 버튼
        with btn_grid_row1[3]:
            if st.button(
                    "📋",
                    key=f"guideline_icon_{idx}_{st.session_state.sim_instance_id}",
                    help=L.get("button_ai_guideline", "AI 응대 가이드라인"),
                    use_container_width=False):
                if st.session_state.is_llm_ready:
                    st.session_state.show_verification_ui = False
                    st.session_state.show_draft_ui = False
                    st.session_state.show_customer_data_ui = False
                    st.session_state.show_agent_response_ui = False

                    guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
                    draft_label = L.get('draft_label', '응대 초안')
                    customer_data_label = L.get('customer_data_label', '고객 데이터')
                    customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
                    st.session_state.simulator_messages = [
                        msg for msg in st.session_state.simulator_messages if not (
                            msg.get("role") == "supervisor" and (
                                guideline_label in msg.get("content", "") or 
                                draft_label in msg.get("content", "") or 
                                customer_data_label in msg.get("content", "") or 
                                customer_data_loaded in msg.get("content", "")))] 

                with st.spinner(L.get("generating_guideline", "AI 응대 가이드라인 생성 중...")):
                    initial_query = st.session_state.get(
                        'customer_query_text_area', content)
                    customer_type_display = st.session_state.get(
                        "customer_type_sim_select", "")

                    session_lang = st.session_state.get("language", current_lang)
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = current_lang

                    guideline_text = _generate_initial_advice(
                        initial_query,
                        customer_type_display,
                        st.session_state.customer_email,
                        st.session_state.customer_phone,
                        session_lang,
                        st.session_state.customer_attachment_file
                    )

                    st.session_state.simulator_messages.append({
                        "role": "supervisor",
                        "content": f"📋 **{L.get('guideline_label', 'AI 응대 가이드라인')}**:\n\n{guideline_text}"
                    })

                    st.session_state.sim_stage = "AGENT_TURN"
            else:
                st.warning(
                    L.get(
                        "simulation_no_key_warning",
                        "LLM이 준비되지 않았습니다."))

    # 두 번째 줄 버튼들
    # 고객 데이터 아이콘 버튼
    with btn_grid_row2[0]:
        if st.button(
                "👤",
                key=f"customer_data_icon_{idx}_{st.session_state.sim_instance_id}",
                help=L.get("button_customer_data", "고객 데이터"),
                use_container_width=False):
            st.session_state.show_agent_response_ui = False
            st.session_state.show_verification_ui = False
            st.session_state.show_draft_ui = False
            st.session_state.show_customer_data_ui = True

            guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
            draft_label = L.get('draft_label', '응대 초안')
            customer_data_label = L.get('customer_data_label', '고객 데이터')
            customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
            st.session_state.simulator_messages = [
                msg for msg in st.session_state.simulator_messages if not (
                    msg.get("role") == "supervisor" and (
                        guideline_label in msg.get("content", "") or 
                        draft_label in msg.get("content", "") or 
                        customer_data_label in msg.get("content", "") or 
                        customer_data_loaded in msg.get("content", "")))]

            customer_id = st.session_state.get(
                "customer_email", "") or st.session_state.get("customer_phone", "")
            if not customer_id:
                customer_id = f"customer_{st.session_state.sim_instance_id}"

            customer_data = st.session_state.customer_data_manager.load_customer_data(
                customer_id)

            try:
                all_customers = st.session_state.customer_data_manager.list_all_customers()
                total_customers = len(all_customers)
            except Exception:
                total_customers = 0

            if customer_data:
                st.session_state.customer_data = customer_data
                customer_info = customer_data.get("data", {})

                info_message = f"📋 **{L.get('customer_data_loaded', '고객 데이터 불러옴')}**\n\n"
                info_message += f"**{L.get('basic_info_label', '기본 정보')}:**\n"
                info_message += f"- {L.get('name_label', '이름')}: {customer_info.get('name', 'N/A')}\n"
                info_message += f"- {L.get('email_label', '이메일')}: {customer_info.get('email', 'N/A')}\n"
                info_message += f"- {L.get('phone_label', '전화번호')}: {customer_info.get('phone', 'N/A')}\n"
                info_message += f"- {L.get('company_label', '회사')}: {customer_info.get('company', 'N/A')}\n"
                info_message += f"\n**{L.get('accumulated_data_label', '누적 데이터')}:**\n"
                info_message += f"- {L.get('total_customers_label', '총 고객 수')}: {total_customers}{L.get('cases_label', '건')}\n"

                if customer_info.get('purchase_history'):
                    info_message += f"\n**{L.get('purchase_history_label', '구매 이력')}:** ({len(customer_info.get('purchase_history', []))}{L.get('cases_label', '건')})\n"
                    for purchase in customer_info.get('purchase_history', [])[:5]:
                        info_message += f"- {purchase.get('date', 'N/A')}: {purchase.get('item', 'N/A')} ({purchase.get('amount', 0):,}{L.get('currency_unit', '원')})\n"
                if customer_info.get('notes'):
                    info_message += f"\n**{L.get('notes_label', '메모')}:** {customer_info.get('notes', 'N/A')}"

                st.session_state.simulator_messages.append({
                    "role": "supervisor",
                    "content": info_message
                })
            else:
                info_message = f"📋 **{L.get('customer_data_label', '고객 데이터')}**: {L.get('no_customer_data', '저장된 고객 데이터가 없습니다.')}\n\n"
                info_message += f"**{L.get('accumulated_data_label', '누적 데이터')}**: {L.get('total_label', '총')} {total_customers}{L.get('cases_label', '건')}"
                st.session_state.simulator_messages.append({
                    "role": "supervisor",
                    "content": info_message
                })

    # 응대 초안 아이콘 버튼 (수동 생성용 - 자동 생성은 별도 처리)
    with btn_grid_row2[1]:
        if st.button(
                "✍️",
                key=f"draft_icon_{idx}_{st.session_state.sim_instance_id}",
                help=L.get("button_draft", "응대 초안"),
                use_container_width=False):
            if st.session_state.is_llm_ready:
                st.session_state.show_agent_response_ui = False
                st.session_state.show_verification_ui = False
                st.session_state.show_customer_data_ui = False
                st.session_state.show_draft_ui = True

                guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
                draft_label = L.get('draft_label', '응대 초안')
                customer_data_label = L.get('customer_data_label', '고객 데이터')
                customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
                st.session_state.simulator_messages = [
                    msg for msg in st.session_state.simulator_messages if not (
                        msg.get("role") == "supervisor" and (
                            guideline_label in msg.get("content", "") or 
                            draft_label in msg.get("content", "") or 
                            customer_data_label in msg.get("content", "") or 
                            customer_data_loaded in msg.get("content", "")))]

                with st.spinner(L.get("generating_draft", "응대 초안 생성 중...")):
                    initial_query = st.session_state.get(
                        'customer_query_text_area', content)
                    customer_type_display = st.session_state.get(
                        "customer_type_sim_select", "")

                    session_lang = st.session_state.get("language", "ko")
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = "ko"

                    draft_text = _generate_initial_advice(
                        initial_query,
                        customer_type_display,
                        st.session_state.customer_email,
                        st.session_state.customer_phone,
                        session_lang,
                        st.session_state.customer_attachment_file
                    )

                    st.session_state.simulator_messages.append({
                        "role": "supervisor",
                        "content": f"✍️ **{L.get('draft_label', '응대 초안')}**:\n\n{draft_text}"
                    })
            else:
                st.warning(
                    L.get(
                        "simulation_no_key_warning",
                        "LLM이 준비되지 않았습니다."))

    # 고객 검증 아이콘 버튼
    with btn_grid_row2[2]:
        if st.button(
                "🔐",
                key=f"verification_icon_{idx}_{st.session_state.sim_instance_id}",
                help=L.get("button_verification", "고객 검증"),
                use_container_width=False):
            st.session_state.show_agent_response_ui = False
            st.session_state.show_draft_ui = False
            st.session_state.show_customer_data_ui = False
            st.session_state.show_verification_ui = True
            st.session_state.verification_message_idx = idx

            guideline_label = L.get('guideline_label', 'AI 응대 가이드라인')
            draft_label = L.get('draft_label', '응대 초안')
            customer_data_label = L.get('customer_data_label', '고객 데이터')
            customer_data_loaded = L.get('customer_data_loaded', '고객 데이터 불러옴')
            st.session_state.simulator_messages = [
                msg for msg in st.session_state.simulator_messages if not (
                    msg.get("role") == "supervisor" and (
                        guideline_label in msg.get("content", "") or 
                        draft_label in msg.get("content", "") or 
                        customer_data_label in msg.get("content", "") or 
                        customer_data_loaded in msg.get("content", "")))]

            st.session_state.sim_stage = "AGENT_TURN"

    # 솔루션 제공 여부 확인 및 추가 문의 버튼
    last_agent_response_idx = None
    for i in range(idx - 1, -1, -1):
        if i < len(st.session_state.simulator_messages) and st.session_state.simulator_messages[i].get(
                "role") == "agent_response":
            last_agent_response_idx = i
            break

    solution_provided = False
    if last_agent_response_idx is not None:
        agent_msg_content = st.session_state.simulator_messages[last_agent_response_idx].get(
            "content", "")
        solution_keywords = [
            "해결", "도움", "안내", "제공", "solution", "help", "assist", "guide",
            "안내해드리", "도와드리", "확인", "처리", "진행", "완료"]
        solution_provided = any(
            keyword in agent_msg_content.lower() for keyword in solution_keywords)

    # 솔루션이 제공되었고, 고객이 긍정적으로 응답한 경우 추가 문의 버튼 표시
    is_solution_given = solution_provided or st.session_state.get("is_solution_provided", False)
    
    if is_solution_given:
        # 고객의 긍정적 응답 패턴 확인 (더 포괄적으로)
        positive_response_keywords = [
            "알겠습니다", "알겠어요", "알겠습니다", "감사합니다", "감사해요", "감사드립니다",
            "ok", "okay", "yes", "thank", "thanks", "ありがとう", "承知しました",
            "네", "예", "좋습니다", "좋아요", "괜찮습니다", "괜찮아요",
            "이해했습니다", "이해했어요", "확인했습니다", "확인했어요"
        ]
        
        # 고객 메시지에서 긍정적 응답 확인
        content_lower = content.lower()
        has_positive_response = any(
            keyword in content_lower for keyword in positive_response_keywords
        ) or (
            "알겠" in content and "감사" in content
        ) or (
            "ok" in content_lower and "thank" in content_lower
        )
        
        # 추가 문의 버튼 표시
        if has_positive_response:
            if st.button(
                    L.get("button_additional_inquiry", "✅ 추가 문의 있나요?"),
                    key=f"additional_inquiry_{idx}_{st.session_state.sim_instance_id}",
                    use_container_width=True,
                    type="secondary"):
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                # 상태 업데이트 트리거
                st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)

    # 설문 조사 버튼
    no_more_keywords = [
        "없습니다", "감사합니다", "No, that will be all", "no more",
        "추가 문의 사항 없습니다", "추가 문의사항 없습니다", "no additional", "結構です"]
    has_no_more = (
        any(keyword in content for keyword in no_more_keywords) or
        ("없습니다" in content and "감사합니다" in content) or
        ("no" in content.lower() and "more" in content.lower() and "thank" in content.lower())
    )

    if has_no_more:
        if st.button(
                L.get("button_survey_end", "📋 설문 조사 전송 및 종료"),
                key=f"survey_end_{idx}_{st.session_state.sim_instance_id}",
                use_container_width=True,
                type="primary"):
            st.session_state.start_time = None

            end_msg = L.get("prompt_survey", "설문 조사 링크를 전송했습니다.")
            st.session_state.simulator_messages.append(
                {"role": "system_end", "content": end_msg}
            )

            customer_type_display = st.session_state.get(
                "customer_type_sim_select", "")
            st.session_state.is_chat_ended = True
            st.session_state.sim_stage = "CLOSING"

            save_simulation_history_local(
                st.session_state.customer_query_text_area,
                customer_type_display,
                st.session_state.simulator_messages,
                is_chat_ended=True,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )


def _render_transfer_summary(L, current_lang):
    """이관 요약 표시"""
    st.markdown("---")
    st.markdown(f"**{L['transfer_summary_header']}**")
    st.info(L["transfer_summary_intro"])

    is_translation_failed = not st.session_state.get(
        "translation_success", True) or not st.session_state.transfer_summary_text

    if st.session_state.transfer_summary_text and st.session_state.get(
            "translation_success", True):
        st.markdown(st.session_state.transfer_summary_text)

    if is_translation_failed:
        if st.session_state.transfer_summary_text:
            st.info(st.session_state.transfer_summary_text)
        if st.button(
                L.get("button_retry_translation", "번역 다시 시도"),
                key=f"btn_retry_translation_{st.session_state.sim_instance_id}"):
            try:
                source_lang = st.session_state.language_at_transfer_start
                target_lang = st.session_state.language

                if not source_lang or not target_lang:
                    st.error(
                        L.get(
                            "invalid_language_info",
                            "언어 정보가 올바르지 않습니다."))
                else:
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
                        lang_name_source = {
                            "ko": "Korean", "en": "English", "ja": "Japanese"}.get(
                            source_lang, "Korean")
                        lang_name_target = {
                            "ko": "Korean", "en": "English", "ja": "Japanese"}.get(
                            target_lang, "Korean")

                        with st.spinner(L.get("transfer_loading", "번역 중...")):
                            translated_summary, is_success = translate_text_with_llm(
                                history_text, target_lang, source_lang)

                            if not translated_summary:
                                st.warning(
                                    L.get(
                                        "translation_empty",
                                        "번역 결과가 비어있습니다. 원본 텍스트를 사용합니다."))
                                translated_summary = history_text
                                is_success = False

                            translated_messages = []
                            for msg in st.session_state.simulator_messages:
                                translated_msg = msg.copy()
                                if msg["role"] in [
                                    "initial_query",
                                    "customer",
                                    "customer_rebuttal",
                                    "agent_response",
                                    "customer_closing_response",
                                        "supervisor"]:
                                    if msg.get("content"):
                                        try:
                                            translated_content, trans_success = translate_text_with_llm(
                                                msg["content"], target_lang, source_lang)
                                            if trans_success:
                                                translated_msg["content"] = translated_content
                                        except Exception:
                                            pass
                                translated_messages.append(translated_msg)

                            st.session_state.simulator_messages = translated_messages
                            st.session_state.transfer_summary_text = translated_summary
                            st.session_state.translation_success = is_success
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