# ========================================
# _pages/chat_modules/agent_turn_draft_generation.py
# 에이전트 턴 - 응대 초안 자동 생성
# ========================================

import streamlit as st
from llm_client import get_api_key

def handle_auto_draft_generation(L):
    """응대 초안 자동 생성 처리"""
    # API Key 확인 및 is_llm_ready 설정
    has_api_key = any([
        bool(get_api_key("openai")),
        bool(get_api_key("gemini")),
        bool(get_api_key("claude")),
        bool(get_api_key("groq"))
    ])
    
    # API Key가 있으면 is_llm_ready를 True로 설정
    if has_api_key:
        st.session_state.is_llm_ready = True
    
    # ⭐ AGENT_TURN 단계에서 응대 초안 확인 및 생성
    if st.session_state.is_llm_ready and st.session_state.sim_stage == "AGENT_TURN":
        # ⭐ 자동 응답이 비활성화되지 않은 경우에만 자동 응답 생성
        auto_response_disabled = st.session_state.get("auto_response_disabled", False)
        
        # ⭐ 초기 문의 입력 시 자동 응답 즉시 생성 및 전송
        if not auto_response_disabled and st.session_state.get("need_auto_response_on_agent_turn", False):
            st.session_state.need_auto_response_on_agent_turn = False
            try:
                from simulation_handler import generate_agent_response_draft
                session_lang = st.session_state.get("language", "ko")
                if session_lang not in ["ko", "en", "ja"]:
                    session_lang = "ko"
                
                # ⭐ 응대 초안 즉시 생성 및 자동 전송
                draft_text = generate_agent_response_draft(session_lang)
                
                if draft_text and draft_text.strip():
                    # 마크다운 헤더 제거
                    draft_text_clean = draft_text
                    if "###" in draft_text_clean:
                        lines = draft_text_clean.split("\n")
                        draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                    draft_text_clean = draft_text_clean.strip()
                    
                    if draft_text_clean:
                        # ⭐ 즉시 자동 전송
                        new_message = {"role": "agent_response", "content": draft_text_clean}
                        st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
                        st.session_state.auto_draft_auto_sent = True
                        
                        # 고객 반응 자동 생성 플래그 설정
                        if st.session_state.is_llm_ready:
                            st.session_state.pending_customer_reaction = True
                            st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1
                        
                        # ⭐ rerun 제거: 같은 렌더링 사이클에서 render_chat_messages가 호출되므로 메시지가 즉시 표시됨

            except Exception as e:
                print(f"초기 자동 응답 생성 오류: {e}")
        
        # 마지막 고객 메시지 확인
        last_customer_msg = None
        last_customer_msg_idx = -1
        for idx, msg in enumerate(reversed(st.session_state.simulator_messages)):
            if msg.get("role") in ["customer", "customer_rebuttal", "initial_query"]:
                last_customer_msg = msg.get("content", "")
                last_customer_msg_idx = len(st.session_state.simulator_messages) - 1 - idx
                break
        
        # 응대 초안이 이미 생성되었는지 확인
        last_draft_for_idx = st.session_state.get("last_draft_for_message_idx", -1)
        auto_draft_exists = (
            st.session_state.get("auto_draft_generated", False) and 
            st.session_state.get("auto_generated_draft_text", "") and
            last_draft_for_idx == last_customer_msg_idx
        )
        
        # ⭐ 새로운 고객 메시지가 들어왔고, 응대 초안이 없거나 다른 메시지용이면 생성
        if not auto_response_disabled and last_customer_msg and not auto_draft_exists:
            # ⭐ 응대 초안 생성 중 플래그로 중복 생성 방지
            if not st.session_state.get("draft_generation_in_progress", False):
                st.session_state.draft_generation_in_progress = True
                try:
                    from simulation_handler import generate_agent_response_draft
                    session_lang = st.session_state.get("language", "ko")
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = "ko"
                    
                    # ⭐ 응대 초안 즉시 생성
                    draft_text = generate_agent_response_draft(session_lang)
                    
                    # 입력창에 자동으로 표시
                    if draft_text and draft_text.strip():
                        # 마크다운 헤더 제거
                        draft_text_clean = draft_text
                        if "###" in draft_text_clean:
                            lines = draft_text_clean.split("\n")
                            draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                        draft_text_clean = draft_text_clean.strip()
                        
                        if draft_text_clean:
                            # ⭐ 응대 초안 저장 및 자동 전송
                            st.session_state.agent_response_area_text = draft_text_clean
                            st.session_state.auto_draft_generated = True
                            st.session_state.auto_generated_draft_text = draft_text_clean
                            st.session_state.last_draft_for_message_idx = last_customer_msg_idx
                            
                            # ⭐ 응대 초안 자동 전송
                            auto_sent_key = f"auto_sent_for_msg_{last_customer_msg_idx}"
                            if not st.session_state.get(auto_sent_key, False):
                                new_message = {"role": "agent_response", "content": draft_text_clean}
                                st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
                                st.session_state[auto_sent_key] = True
                                st.session_state.auto_draft_auto_sent = True
                                
                                # 입력창 초기화
                                st.session_state.agent_response_area_text = ""
                                st.session_state.auto_generated_draft_text = ""
                                
                                # 고객 반응은 다음 렌더링 사이클에서 생성
                                st.session_state.pending_customer_reaction = True
                                st.session_state.pending_customer_reaction_for_msg_idx = last_customer_msg_idx
                                
                                # ⭐ rerun 제거: 같은 렌더링 사이클에서 render_chat_messages가 호출되므로 메시지가 즉시 표시됨
                            
                            print(f"✅ 응대 초안 생성 및 자동 전송 완료 (메시지 인덱스: {last_customer_msg_idx})")
                except Exception as e:
                    st.session_state.auto_draft_generated = False
                    print(f"❌ 응대 초안 자동 생성 오류: {e}")
                finally:
                    st.session_state.draft_generation_in_progress = False

def handle_transcript_auto_send(L):
    """전사 결과 반영 및 자동 전송"""
    if st.session_state.get("last_transcript") and st.session_state.last_transcript:
        agent_response_auto = st.session_state.last_transcript.strip()
        if agent_response_auto:
            # ⭐ 전사 결과 자동 전송
            new_message = {"role": "agent_response", "content": agent_response_auto}
            st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
            
            # 전사 결과 초기화
            st.session_state.last_transcript = ""
            st.session_state.agent_response_area_text = ""
            st.session_state.auto_draft_generated = False
            
            # 고객 반응은 다음 렌더링 사이클에서 생성
            if st.session_state.is_llm_ready:
                st.session_state.pending_customer_reaction = True
                st.session_state.pending_customer_reaction_for_msg_idx = len(st.session_state.simulator_messages) - 1
            
            # ⭐ rerun 제거: 같은 렌더링 사이클에서 render_chat_messages가 호출되므로 메시지가 즉시 표시됨

