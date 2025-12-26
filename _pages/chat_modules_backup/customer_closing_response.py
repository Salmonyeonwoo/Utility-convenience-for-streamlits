# 이 모듈은 _chat_simulator.py에서 분리된 부분입니다
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
from simulation_handler import *
from visualization import *
from audio_handler import *
from llm_client import get_api_key
from typing import List, Dict, Any
import uuid
import time
import os

        # 언어 키 안전하게 가져오기
        current_lang = st.session_state.get("language", "ko")
        if current_lang not in ["ko", "en", "ja"]:
            current_lang = "ko"
        L = LANG.get(current_lang, LANG["ko"])
        customer_type_display = st.session_state.get("customer_type_sim_select", L["customer_type_options"][0])
        
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
            is_email_closing = any(pattern.lower() in last_agent_response.lower() for pattern in email_closing_patterns)
        
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
                if st.button(L["customer_generate_response_button"], key="btn_generate_final_response"):
                    st.session_state.sim_stage = "AGENT_TURN"
                    # ⭐ 재실행 불필요: 버튼 클릭 시 자동으로 상태 전환됨
                    # st.rerun()
                st.stop()
            
            # LLM이 준비된 경우 고객 응답 생성
            st.info(L["agent_confirmed_additional_inquiry"])
            with st.spinner(L["generating_customer_response"]):
                final_customer_reaction = generate_customer_closing_response(st.session_state.language)

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
                pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                regex = re.compile(pattern, re.IGNORECASE)
                if regex.search(final_customer_reaction):
                    has_no_more_inquiry = True
                    break
            
            # ⭐ 추가: 메일 끝인사가 포함된 경우, 고객의 긍정 반응이나 "추가 문의 사항 없습니다" 답변을 인식하면 설문 조사 링크 전송 버튼 자동 활성화
            # 긍정 반응 키워드 추가
            positive_keywords = [
                "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", "감사합니다", "thank you", "ありがとう"
            ]
            is_positive_response = any(keyword.lower() in final_customer_reaction.lower() for keyword in positive_keywords)
            
            # ⭐ 정규표현식으로 종료 키워드 인식
            escaped_check = re.escape(L['customer_no_more_inquiries'])
            no_more_pattern_check = escaped_check.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)
            if is_email_closing and (has_no_more_inquiry or no_more_regex_check.search(final_customer_reaction) or is_positive_response):
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
                        st.session_state.customer_query_text_area, customer_type_display,
                        st.session_state.simulator_messages, is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )
                    
                    st.session_state.realtime_hint_text = ""  # 힌트 초기화
            # 메일 끝인사가 포함된 경우 여기서 처리 완료, 다른 로직은 실행하지 않음
            # ⭐ 정규표현식으로 종료 키워드 인식 (메일 끝인사가 아닌 경우)
            elif not is_email_closing:
                import re
                escaped_final = re.escape(L['customer_no_more_inquiries'])
                no_more_pattern_final = escaped_final.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                no_more_regex_final = re.compile(no_more_pattern_final, re.IGNORECASE)
                if no_more_regex_final.search(final_customer_reaction) or has_no_more_inquiry:
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
                            st.session_state.customer_query_text_area, customer_type_display,
                            st.session_state.simulator_messages, is_chat_ended=True,
                            attachment_context=st.session_state.sim_attachment_context_for_llm,
                        )
                        
                        st.session_state.realtime_hint_text = ""  # 힌트 초기화
            # (B) "추가 문의 사항도 있습니다" 경로 -> AGENT_TURN으로 복귀
            elif L['customer_has_additional_inquiries'] in final_customer_reaction:
                st.session_state.sim_stage = "AGENT_TURN"
                save_simulation_history_local(
                    st.session_state.customer_query_text_area, customer_type_display,
                    st.session_state.simulator_messages, is_chat_ended=False,
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
                        st.session_state.customer_query_text_area, customer_type_display,
                        st.session_state.simulator_messages, is_chat_ended=True,
                        attachment_context=st.session_state.sim_attachment_context_for_llm,
                    )
                    
                    st.session_state.realtime_hint_text = ""  # 힌트 초기화

    # =========================
    # 9. 최종 종료 행동 (FINAL_CLOSING_ACTION)
    # =========================
    elif st.session_state.sim_stage == "FINAL_CLOSING_ACTION":