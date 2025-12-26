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
        st.info(L["customer_turn_info"])

        # ⭐ 수정: 에이전트 인사말 자동 생성 제거 - 에이전트가 직접 입력하도록 변경
        # 에이전트 인사말이 없는 경우에도 자동 생성하지 않고, 에이전트가 직접 입력하도록 함

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
                customer_response = generate_customer_reaction(st.session_state.language, is_call=False)

            # 2. 대화 로그 업데이트
            st.session_state.simulator_messages.append(
                {"role": "customer", "content": customer_response}
            )
            
            # 3. 생성 직후 바로 다음 단계 결정
            # ⭐ 정규표현식으로 종료 키워드 인식 (띄어쓰기, 마침표 무시)
            import re
            escaped_no_more = re.escape(L["customer_no_more_inquiries"])
            no_more_pattern = escaped_no_more.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
            escaped_positive = re.escape(L["customer_positive_response"])
            positive_pattern = escaped_positive.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            positive_regex = re.compile(positive_pattern, re.IGNORECASE)
            is_positive_closing = no_more_regex.search(customer_response) is not None or positive_regex.search(customer_response) is not None
            
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
                no_more_pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
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
        no_more_pattern = escaped_no_more.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
        escaped_positive = re.escape(L["customer_positive_response"])
        positive_pattern = escaped_positive.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        positive_regex = re.compile(positive_pattern, re.IGNORECASE)
        is_positive_closing = no_more_regex.search(customer_response) is not None or positive_regex.search(customer_response) is not None

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
                "추가 문의사항이 있으면 언제든지 연락", "추가 문의 사항이 있으면 언제든지 연락",
                "추가 문의사항이 있으시면", "추가 문의 사항이 있으시면",
                "언제든지 연락", "언제든지 연락 주세요",
                "additional inquiries", "any additional questions", "any further questions",
                "feel free to contact", "please feel free to contact",
                "please don't hesitate to contact", "don't hesitate to contact",
                "please let me know", "let me know", "let me know if",
                "please let me know so", "let me know so",
                "if you have any questions", "if you have any further questions",
                "if you need any assistance", "if you need further assistance",
                "if you encounter any issues", "if you still have", "if you remain unclear",
                "I can assist further", "I can help further", "I can assist",
                "so I can assist", "so I can help", "so I can assist further",
                "追加のご質問", "追加のお問い合わせ", "ご質問がございましたら", "お問い合わせがございましたら"
            ]
            
            if last_agent_response:
                is_email_closing = any(pattern.lower() in last_agent_response.lower() for pattern in email_closing_patterns)
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
                pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
                regex = re.compile(pattern, re.IGNORECASE)
                if regex.search(customer_response):
                    has_no_more_inquiry = True
                    break
            # "없습니다"와 "감사합니다"가 함께 있는 경우도 인식
            if "없습니다" in customer_response and "감사합니다" in customer_response:
                has_no_more_inquiry = True
            
            # 긍정 반응 키워드 추가 (더 포괄적인 인식)
            positive_keywords = [
                "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", "감사합니다", "thank you", "ありがとう",
                "좋습니다", "good", "fine", "괜찮습니다", "알겠습니다 감사합니다"
            ]
            is_positive_response = any(keyword.lower() in customer_response.lower() for keyword in positive_keywords)
            
            # 긍정 반응이 있거나 "추가 문의 사항 없습니다" 답변이 있으면 설문 조사 링크 전송 버튼 활성화
            # ⭐ 정규표현식으로 종료 키워드 인식
            escaped_check = re.escape(L['customer_no_more_inquiries'])
            no_more_pattern_check = escaped_check.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)
            if is_positive_closing or has_no_more_inquiry or no_more_regex_check.search(customer_response) or is_positive_response:
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
            no_more_pattern = escaped.replace(r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            no_more_regex = re.compile(no_more_pattern, re.IGNORECASE)
            if no_more_regex.search(customer_response):
                # ⭐ 수정: "없습니다. 감사합니다" 답변 시 에이전트가 감사 인사를 한 후 종료하도록 변경
                # 바로 종료하지 않고 WAIT_CLOSING_CONFIRMATION_FROM_AGENT 단계로 이동하여 에이전트가 감사 인사 후 종료
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
                st.session_state.customer_query_text_area, customer_type_display,
                st.session_state.simulator_messages, is_chat_ended=False,
                attachment_context=st.session_state.sim_attachment_context_for_llm,
            )

        st.session_state.realtime_hint_text = ""  # 힌트 초기화
        # ⭐ 재실행 불필요: 힌트 초기화만으로 충분, 자동 업데이트됨
        # st.rerun()


    # =========================
    # 7. 종료 확인 메시지 대기 (WAIT_CLOSING_CONFIRMATION_FROM_AGENT)
    # =========================
    elif st.session_state.sim_stage == "WAIT_CLOSING_CONFIRMATION_FROM_AGENT":