# ========================================
# _pages/chat_modules/agent_turn_guidelines.py
# 에이전트 턴 - 가이드라인, 호텔/BPO 실무 매크로, CRM 요약
# ========================================

import streamlit as st
from utils.history_handler import (
    generate_chat_summary, load_simulation_histories_local,
    recommend_guideline_for_customer
)

def render_guidelines_and_info(L):
    """고객 성향 기반 가이드라인 추천, 호텔/BPO CS 매크로 및 CRM 연동"""
    
    # 1. 고객 성향 기반 가이드라인 추천
    if st.session_state.simulator_messages and len(st.session_state.simulator_messages) >= 2:
        try:
            temp_summary = generate_chat_summary(
                st.session_state.simulator_messages,
                st.session_state.customer_query_text_area,
                st.session_state.get("customer_type_sim_select", ""),
                st.session_state.language
            )

            if temp_summary and temp_summary.get("customer_sentiment_score"):
                all_histories = load_simulation_histories_local(st.session_state.language)
                recommended_guideline = recommend_guideline_for_customer(
                    temp_summary, all_histories, st.session_state.language
                )

                if recommended_guideline:
                    with st.expander("💡 고객 성향 기반 응대 가이드라인 추천", expanded=False):
                        st.markdown(recommended_guideline)
                        st.caption("ℹ️ 이 가이드라인은 유사한 과거 고객 응대 건을 분석하여 자동 생성되었습니다.")
        except Exception:
            pass

    # 2. 실무 특화: 호텔 & 콜센터 BPO 원클릭 응대 매크로 템플릿
    with st.expander("⚡ 호텔 & 콜센터 BPO 실무 원클릭 응대 매크로", expanded=False):
        tab_hotel, tab_bpo, tab_crm = st.tabs(["🏨 호텔 CS 매크로", "📞 콜센터 BPO 매크로", "📋 CRM 티켓 복사"])
        
        # --- 호텔 CS 매크로 ---
        with tab_hotel:
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                if st.button("🛎️ 체크인/체크아웃 안내", key="macro_hotel_checkin", use_container_width=True):
                    text = "고객님, 저희 호텔 기본 체크인 시간은 오후 3시, 체크아웃은 오전 11시입니다. 얼리 체크인/레이트 체크아웃은 당일 객실 상황에 따라 유·무료로 가능하오니 확인해 드릴까요?"
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()
                
                if st.button("🔄 예약 변경 및 취소 규정", key="macro_hotel_cancel", use_container_width=True):
                    text = "예약 변경 및 취소 안내드립니다. 예약번호와 투숙자 성함을 알려주시면 위약금 규정 및 잔여 객실 조회를 거쳐 최우선으로 변경/환불 접수해 드리겠습니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

            with col_h2:
                if st.button("🍷 룸서비스/어메니티 요청", key="macro_hotel_amenity", use_container_width=True):
                    text = "요청하신 추가 어메니티(타월/가운/세면도구) 및 룸서비스는 하우스키핑 부서에 즉시 전달하였으며, 15분 이내로 객실로 정성껏 전달해 드리겠습니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

                if st.button("🍳 조식 뷔페 및 부대시설 안내", key="macro_hotel_facility", use_container_width=True):
                    text = "조식 뷔페는 2층 레스토랑에서 오전 6시 30분부터 10시까지 운영됩니다. 투숙객 전용 수영장 및 피트니스 센터는 3층에서 룸키 태그 후 무료로 이용 가능합니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

        # --- 콜센터 BPO 매크로 ---
        with tab_bpo:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("📦 배송 지연 정중한 안내", key="macro_bpo_delivery", use_container_width=True):
                    text = "배송 지연으로 불편을 드려 진심으로 사과드립니다. 물류센터 출고 물량 급증으로 1~2일 지연되고 있으나, 오늘 중 최우선 출고 등록하여 송장 번호를 신속히 안내해 드리겠습니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

                if st.button("🔁 교환/반품 무상 수거 접수", key="macro_bpo_exchange", use_container_width=True):
                    text = "교환/반품 접수를 신속히 진행해 드리겠습니다. 기사님이 2~3 영업일 내로 방문 수거 예정이며, 제품 하자로 인한 건이므로 회수 및 재배송 비용은 당사 전액 부담입니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

            with col_b2:
                if st.button("👑 VIP 고객 최우선 케어", key="macro_bpo_vip", use_container_width=True):
                    text = "VIP 고객님, 문의 주셔서 감사드립니다. 해당 안건은 전담 시니어 매니저에게 긴급 건으로 배정되었으며, 30분 내로 최선의 맞춤 솔루션을 정리하여 연락드리겠습니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

                if st.button("🛡️ 컴플레인 진정 프로토콜", key="macro_bpo_complaint", use_container_width=True):
                    text = "고객님께서 겪으신 큰 불편과 실망에 대해 깊이 공감하며 머리 숙여 사과드립니다. 말씀해 주신 내용은 품질책임자에게 즉시 에스컬레이션되었으며, 재발 방지와 함께 납득하실 수 있는 보상안을 마련하겠습니다."
                    st.session_state.auto_generated_draft_text = text
                    st.session_state.agent_response_area_text = text
                    st.rerun()

        # --- CRM 티켓 복사 ---
        with tab_crm:
            c_name = st.session_state.get('customer_name', '고객')
            c_id = st.session_state.get('customer_id', 'CUST-TEMP')
            c_phone = st.session_state.get('customer_phone', 'N/A')
            c_email = st.session_state.get('customer_email', 'N/A')
            lang = st.session_state.get('language', 'ko').upper()
            
            crm_summary = (
                f"[호텔 / BPO CRM 상담 티켓]\n"
                f"• 고객명: {c_name} (ID: {c_id})\n"
                f"• 연락처: {c_phone} | 이메일: {c_email}\n"
                f"• 인입 채널: 채팅/이메일 CS | 상담 언어: {lang}\n"
                f"• 최근 문의: {st.session_state.get('customer_query_text_area', '일반 상담')[:80]}\n"
                f"• 상담 상태: 진행 중 (In-Progress)\n"
                f"• 처리 내역: 표준 CS 가이드라인 및 실무 매크로 기반 즉시 응대 완료"
            )
            st.text_area("📋 CRM 등록용 표준 텍스트 (오페라/세일즈포스/제네시스)", crm_summary, height=140)
            st.caption("위 텍스트 박스 내용을 복사하여 사내 CRM 시스템에 바로 붙여넣을 수 있습니다.")

    # 3. 언어 이관 요청 강조 표시
    if st.session_state.get("language_transfer_requested", False):
        st.error(L.get("language_transfer_requested_msg", "🌐 고객이 언어 전환(이관)을 요청했습니다. 아래 언어 이관 버튼을 눌러 담당 팀으로 즉시 이관하세요."))

    # 4. 고객 첨부 파일 정보
    if st.session_state.get("sim_attachment_context_for_llm"):
        st.info(f"📎 최초 문의 시 첨부된 파일 정보:\n\n{st.session_state.sim_attachment_context_for_llm.replace('[ATTACHMENT STATUS]', '').strip()}")
