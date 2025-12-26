# ========================================
# _pages/_chat_customer_message.py
# 채팅 시뮬레이터 - 고객 메시지 렌더링 및 버튼 처리
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import generate_realtime_hint
from utils.customer_analysis import _generate_initial_advice
from utils.history_handler import save_simulation_history_local


def render_customer_message_with_icons(L, idx, content, current_lang):
    """고객 메시지와 아이콘 버튼들을 깔끔하게 렌더링"""
    # 이관 후에는 현재 언어로 다시 설정 (번역 반영을 위해)
    actual_current_lang = st.session_state.get("language", current_lang)
    if actual_current_lang not in ["ko", "en", "ja"]:
        actual_current_lang = "ko"
    actual_L = LANG.get(actual_current_lang, LANG["ko"])
    L = actual_L
    current_lang = actual_current_lang
    
    # 사용자 모드 확인
    perspective = st.session_state.get("sim_perspective", "AGENT")
    user_role = st.session_state.get("user_role_selected", None)
    is_customer_mode = (user_role == "CUSTOMER" or perspective == "CUSTOMER")
    
    # 고객 모드일 때: 고객 메시지는 오른쪽 (노란색)
    # 상담원 모드일 때: 고객 메시지는 왼쪽 (회색)
    if is_customer_mode:
        justify_content = "flex-end"  # 오른쪽
        message_class = "message-customer"  # 오른쪽 노란색
        animation = "slideInRight"
    else:
        justify_content = "flex-start"  # 왼쪽
        message_class = "message-customer-left"  # 왼쪽 회색
        animation = "slideInLeft"
    
    # 타임스탬프 추가 (스크린샷 스타일)
    from datetime import datetime
    timestamp = ""
    if idx < len(st.session_state.simulator_messages):
        msg = st.session_state.simulator_messages[idx]
        if "timestamp" in msg:
            timestamp = msg["timestamp"]
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 고객 이름 가져오기
    customer_name = st.session_state.get("customer_name", L.get("customer_label", "고객")) or L.get("customer_label", "고객")
    if st.session_state.get("customer_data"):
        customer_name = st.session_state.customer_data.get("basic_info", {}).get("customer_name", customer_name)
    
    st.markdown(f"""
    <div style="display: flex; justify-content: {justify_content}; margin: 8px 0; animation: {animation} 0.4s ease-out;">
        <div class="message-bubble {message_class}" style="max-width: 70%;">
            <div style="font-weight: 600; margin-bottom: 4px; font-size: 14px;">{customer_name}</div>
            <div style="line-height: 1.5; margin-bottom: 4px;">{content.replace(chr(10), '<br>')}</div>
            <div style="font-size: 11px; color: #666; text-align: left; margin-top: 4px;">{timestamp}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 아이콘 버튼들을 말풍선 아래에 깔끔하게 배치 (한 줄)
    st.markdown('<div style="display: flex; justify-content: center; gap: 8px; margin: 10px 0; flex-wrap: wrap;">', unsafe_allow_html=True)
    
    # 7개 아이콘 버튼을 한 줄로 배치
    icon_cols = st.columns(7)
    with icon_cols[0]:
        if st.button("💡", key=f"hint_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_hint", "응대 힌트"), use_container_width=True):
            _handle_hint_button(L, idx, current_lang)
    with icon_cols[1]:
        if st.button("📞", key=f"call_provider_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_call_company", "업체에 전화"), use_container_width=True):
            _handle_call_provider_button(L)
    with icon_cols[2]:
        if st.button("📱", key=f"call_customer_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_call_customer", "고객에게 전화"), use_container_width=True):
            _handle_call_customer_button(L)
    with icon_cols[3]:
        if st.button("📋", key=f"guideline_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_ai_guideline", "AI 응대 가이드라인"), use_container_width=True):
            _handle_guideline_button(L, idx, content, current_lang)
    with icon_cols[4]:
        if st.button("👤", key=f"customer_data_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_customer_data", "고객 데이터"), use_container_width=True):
            _handle_customer_data_button(L)
    with icon_cols[5]:
        if st.button("✍️", key=f"draft_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_draft", "응대 초안"), use_container_width=True):
            _handle_draft_button(L, idx, content, current_lang)
    with icon_cols[6]:
        if st.button("🔐", key=f"verification_icon_{idx}_{st.session_state.sim_instance_id}", help=L.get("button_verification", "고객 검증"), use_container_width=True):
            _handle_verification_button(L, idx)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 추가 문의 및 설문 조사 버튼
    _render_additional_buttons(L, idx, content)




def _render_additional_buttons(L, idx, content):
    """추가 문의 및 설문 조사 버튼 렌더링"""
    # ⭐ 고객 모드일 때는 추가 문의 사항 버튼 제외
    user_role = st.session_state.get("user_role_selected", None)
    perspective = st.session_state.get("sim_perspective", "AGENT")
    is_customer_mode = (user_role == "CUSTOMER" or perspective == "CUSTOMER")
    
    # 솔루션 제공 여부 확인
    last_agent_response_idx = None
    for i in range(idx - 1, -1, -1):
        if i < len(st.session_state.simulator_messages) and st.session_state.simulator_messages[i].get("role") == "agent_response":
            last_agent_response_idx = i
            break

    solution_provided = False
    if last_agent_response_idx is not None:
        agent_msg_content = st.session_state.simulator_messages[last_agent_response_idx].get("content", "")
        solution_keywords = ["해결", "도움", "안내", "제공", "solution", "help", "assist", "guide", "안내해드리", "도와드리", "확인", "처리", "진행", "완료"]
        solution_provided = any(keyword in agent_msg_content.lower() for keyword in solution_keywords)

    is_solution_given = solution_provided or st.session_state.get("is_solution_provided", False)
    
    if is_solution_given:
        positive_response_keywords = [
            "알겠습니다", "알겠어요", "감사합니다", "감사해요", "감사드립니다",
            "ok", "okay", "yes", "thank", "thanks", "ありがとうございます", "承知致しました", "承知いたしました", "了解しました",
            "네", "예", "좋습니다", "좋아요", "괜찮습니다", "괜찮아요",
            "이해했습니다", "이해했어요", "확인했습니다", "확인했어요"
        ]
        
        content_lower = content.lower()
        has_positive_response = any(keyword in content_lower for keyword in positive_response_keywords) or ("알겠" in content and "감사" in content) or ("ok" in content_lower and "thank" in content_lower)
        
        no_more_keywords = ["없습니다", "감사합니다", "No, that will be all", "no more", "추가 문의 사항 없습니다", "추가 문의사항 없습니다", "no additional", "結構です"]
        has_no_more = any(keyword in content for keyword in no_more_keywords) or ("없습니다" in content and "감사합니다" in content) or ("no" in content.lower() and "more" in content.lower() and "thank" in content.lower())
        
        if has_positive_response or has_no_more:
            # ⭐ 고객 모드일 때는 추가 문의 사항 버튼 제외, 종료 버튼만 표시
            if is_customer_mode:
                if has_no_more:
                    btn_col1, btn_col2 = st.columns([1, 5])
                    with btn_col1:
                        if st.button(
                                "📋 종료",
                                key=f"survey_end_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="primary"):
                            _handle_survey_end(L)
            else:
                # 에이전트 모드: 기존 로직 유지
                btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 4])
                
                if has_positive_response:
                    with btn_col1:
                        if st.button(
                                "✅ 추가",
                                key=f"additional_inquiry_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="secondary"):
                            st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                            st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
                
                if has_no_more:
                    with btn_col2:
                        if st.button(
                                "📋 종료",
                                key=f"survey_end_{idx}_{st.session_state.sim_instance_id}",
                                use_container_width=True,
                                type="primary"):
                            _handle_survey_end(L)


def _handle_hint_button(L, idx, current_lang):
    """응대 힌트 버튼 처리"""
    if st.session_state.is_llm_ready:
        st.session_state.show_verification_ui = False
        st.session_state.show_draft_ui = False
        st.session_state.show_customer_data_ui = False
        st.session_state.show_agent_response_ui = False

        hint_label = L.get('hint_label', '응대 힌트')
        st.session_state.simulator_messages = [
            msg for msg in st.session_state.simulator_messages if not (
                msg.get("role") == "supervisor" and hint_label in msg.get("content", ""))]

        session_lang = st.session_state.get("language", current_lang)
        if session_lang not in ["ko", "en", "ja"]:
            session_lang = current_lang

        with st.spinner(L.get("response_generating", "생성 중...")):
            from simulation_handler import generate_realtime_hint
            hint = generate_realtime_hint(session_lang, is_call=False)
            st.session_state.realtime_hint_text = hint
            st.session_state.simulator_messages.append({
                "role": "supervisor",
                "content": f"💡 **{L.get('hint_label', '응대 힌트')}**: {hint}"
            })
    else:
        from llm_client import get_api_key
        has_api_key = any([
            bool(get_api_key("openai")),
            bool(get_api_key("gemini")),
            bool(get_api_key("claude")),
            bool(get_api_key("groq"))
        ])
        if not has_api_key:
            st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
        else:
            st.session_state.is_llm_ready = True


def _handle_call_provider_button(L):
    """업체에 전화 버튼 처리"""
    st.session_state.show_verification_ui = False
    st.session_state.show_draft_ui = False
    st.session_state.show_customer_data_ui = False
    st.session_state.show_agent_response_ui = False
    st.session_state.sim_call_outbound_target = L.get("call_target_provider", "현지 업체/파트너")
    st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"


def _handle_call_customer_button(L):
    """고객에게 전화 버튼 처리"""
    st.session_state.show_verification_ui = False
    st.session_state.show_draft_ui = False
    st.session_state.show_customer_data_ui = False
    st.session_state.show_agent_response_ui = False
    st.session_state.sim_call_outbound_target = L.get("call_target_customer", "고객")
    st.session_state.sim_stage = "OUTBOUND_CALL_IN_PROGRESS"


def _handle_guideline_button(L, idx, content, current_lang):
    """AI 응대 가이드라인 버튼 처리"""
    if st.session_state.is_llm_ready:
        st.session_state.show_verification_ui = False
        st.session_state.show_draft_ui = False
        st.session_state.show_customer_data_ui = False
        st.session_state.show_agent_response_ui = False

        _clear_supervisor_messages(L)
        
        with st.spinner(L.get("generating_guideline", "AI 응대 가이드라인 생성 중...")):
            initial_query = st.session_state.get('customer_query_text_area', content)
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
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
        from llm_client import get_api_key
        has_api_key = any([
            bool(get_api_key("openai")),
            bool(get_api_key("gemini")),
            bool(get_api_key("claude")),
            bool(get_api_key("groq"))
        ])
        if not has_api_key:
            st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
        else:
            st.session_state.is_llm_ready = True


def _handle_customer_data_button(L):
    """고객 데이터 버튼 처리"""
    st.session_state.show_agent_response_ui = False
    st.session_state.show_verification_ui = False
    st.session_state.show_draft_ui = False
    st.session_state.show_customer_data_ui = True

    _clear_supervisor_messages(L)

    customer_id = st.session_state.get("customer_email", "") or st.session_state.get("customer_phone", "")
    if not customer_id:
        customer_id = f"customer_{st.session_state.sim_instance_id}"

    customer_data = st.session_state.customer_data_manager.load_customer_data(customer_id)

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


def _handle_draft_button(L, idx, content, current_lang):
    """응대 초안 버튼 처리"""
    if st.session_state.is_llm_ready:
        st.session_state.show_agent_response_ui = False
        st.session_state.show_verification_ui = False
        st.session_state.show_customer_data_ui = False
        st.session_state.show_draft_ui = True

        _clear_supervisor_messages(L)

        with st.spinner(L.get("generating_draft", "응대 초안 생성 중...")):
            initial_query = st.session_state.get('customer_query_text_area', content)
            customer_type_display = st.session_state.get("customer_type_sim_select", "")
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
        from llm_client import get_api_key
        has_api_key = any([
            bool(get_api_key("openai")),
            bool(get_api_key("gemini")),
            bool(get_api_key("claude")),
            bool(get_api_key("groq"))
        ])
        if not has_api_key:
            st.warning(L.get("simulation_no_key_warning", "LLM이 준비되지 않았습니다."))
        else:
            st.session_state.is_llm_ready = True


def _handle_verification_button(L, idx):
    """고객 검증 버튼 처리"""
    st.session_state.show_agent_response_ui = False
    st.session_state.show_draft_ui = False
    st.session_state.show_customer_data_ui = False
    st.session_state.show_verification_ui = True
    st.session_state.verification_message_idx = idx

    _clear_supervisor_messages(L)
    st.session_state.sim_stage = "AGENT_TURN"


def _handle_survey_end(L):
    """설문 조사 종료 버튼 처리"""
    st.session_state.start_time = None
    end_msg = L.get("prompt_survey", "설문 조사 링크를 전송했습니다.")
    st.session_state.simulator_messages.append({"role": "system_end", "content": end_msg})
    customer_type_display = st.session_state.get("customer_type_sim_select", "")
    st.session_state.is_chat_ended = True
    st.session_state.sim_stage = "CLOSING"

    save_simulation_history_local(
        st.session_state.customer_query_text_area,
        customer_type_display,
        st.session_state.simulator_messages,
        is_chat_ended=True,
        attachment_context=st.session_state.sim_attachment_context_for_llm,
    )


def _clear_supervisor_messages(L):
    """Supervisor 메시지 정리"""
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

