# ========================================
# chat_modules/agent_message_handler.py
# 에이전트 메시지 처리 모듈
# ========================================

import streamlit as st
from datetime import datetime


def handle_agent_message_send(agent_response, L):
    """에이전트 메시지 전송 처리"""
    if not agent_response or not agent_response.strip():
        st.warning(L["empty_response_warning"])
        return False
    
    # AHT 타이머 시작
    if st.session_state.start_time is None and len(st.session_state.simulator_messages) >= 1:
        st.session_state.start_time = datetime.now()
    
    # 에이전트 첨부 파일 처리
    final_response_content = agent_response
    if st.session_state.agent_attachment_file:
        file_infos = st.session_state.agent_attachment_file
        file_names = ", ".join([f["name"] for f in file_infos])
        attachment_msg = L["agent_attachment_status"].format(
            filename=file_names, filetype=f"총 {len(file_infos)}개 파일"
        )
        final_response_content = f"{agent_response}\n\n---\n{attachment_msg}"
    
    # 메시지 추가
    new_message = {
        "role": "agent_response", 
        "content": final_response_content,
        "is_manual_response": True,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
    st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
    
    # 에이전트 응답 이력 기록
    _record_agent_response_history(final_response_content)
    
    # 메일 끝인사 확인
    _check_email_closing(final_response_content)
    
    # 입력창 초기화
    _reset_input_state()
    
    # 고객 반응 자동 생성
    if st.session_state.is_llm_ready:
        st.session_state.pending_customer_reaction = True
        st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1
        st.session_state.sim_stage = "CUSTOMER_TURN"
    else:
        st.session_state.need_customer_response = True
        st.session_state.sim_stage = "CUSTOMER_TURN"
    
    return True


def _record_agent_response_history(final_response_content):
    """에이전트 응답 이력 기록"""
    try:
        from customer_data_management import AdvancedCustomerManager
        manager = AdvancedCustomerManager()
        customer_id = st.session_state.get("customer_email", "") or st.session_state.get("customer_phone", "")
        
        if not customer_id:
            return
        
        all_custs = manager.list_all_customers()
        target_customer_id = None
        customer_hash = manager.generate_identity_hash(
            st.session_state.get("customer_phone", ""),
            st.session_state.get("customer_email", "")
        )
        
        for c in all_custs:
            if c["basic_info"]["identity_hash"] == customer_hash:
                target_customer_id = c["basic_info"]["customer_id"]
                break
        
        if not target_customer_id:
            target_customer_id = manager.create_customer(
                st.session_state.get("customer_email", "고객").split("@")[0],
                st.session_state.get("customer_phone", ""),
                st.session_state.get("customer_email", ""),
                "일반"
            )
        
        customer_data = manager.load_customer(target_customer_id)
        if customer_data:
            if "agent_manual_responses" not in customer_data:
                customer_data["agent_manual_responses"] = []
            
            reason = "일반 응대"
            if st.session_state.get("requires_agent_response", False):
                last_customer_msg = ""
                for msg in reversed(st.session_state.simulator_messages):
                    if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]:
                        last_customer_msg = msg.get("content", "").lower()
                        break
                
                cancellation_keywords = ["취소", "환불", "cancel", "refund"]
                has_cancellation = any(kw in last_customer_msg for kw in cancellation_keywords)
                
                if has_cancellation:
                    reason = "취소/환불 요청 - 에이전트 직접 응대"
                else:
                    reason = "고객 불만/직접 응대 요청"
            
            customer_data["agent_manual_responses"].append({
                "response": final_response_content,
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "is_cancellation_refund": any(kw in final_response_content.lower() for kw in ["취소", "환불", "cancel", "refund"])
            })
            manager._save_to_file(target_customer_id, customer_data)
            print(f"✅ 에이전트 직접 응답 이력 기록 완료 (고객 ID: {target_customer_id}, 사유: {reason})")
    except Exception as e:
        print(f"에이전트 응답 이력 기록 오류: {e}")


def _check_email_closing(final_response_content):
    """메일 끝인사 확인"""
    email_closing_patterns = [
        "추가 문의사항이 있으면 언제든지 연락",
        "추가 문의 사항이 있으면 언제든지 연락",
        "additional inquiries", "any additional questions",
        "feel free to contact", "please feel free to contact",
        "追加のご質問", "追加のお問い合わせ"]
    is_email_closing_in_response = any(pattern.lower() in final_response_content.lower() 
                                      for pattern in email_closing_patterns)
    if is_email_closing_in_response:
        st.session_state.has_email_closing = True


def _reset_input_state():
    """입력 상태 초기화"""
    st.session_state.sim_audio_bytes = None
    st.session_state.agent_attachment_file = []
    st.session_state.language_transfer_requested = False
    st.session_state.realtime_hint_text = ""
    st.session_state.sim_call_outbound_summary = ""
    st.session_state.last_transcript = ""
    st.session_state.reset_agent_response_area = True
    st.session_state.auto_draft_generated = False
    st.session_state.auto_generated_draft_text = ""
    st.session_state.auto_draft_auto_sent = False

