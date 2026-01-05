# ========================================
# _pages/_chat_customer_turn.py
# 채팅 시뮬레이터 - 고객 반응 생성 모듈
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import generate_customer_reaction
from utils.history_handler import save_simulation_history_local

# 응대 초안 자동 생성을 위한 플래그 초기화
# 고객 메시지가 생성되면 다음 AGENT_TURN에서 자동으로 응대 초안 생성
import re


def render_customer_turn(L, current_lang):
    """고객 반응 생성 단계 UI 렌더링"""
    # ⭐ 고객 체험 모드일 때 사용자가 직접 입력
    perspective = st.session_state.get("sim_perspective", "AGENT")
    if perspective == "CUSTOMER" and st.session_state.sim_stage == "CUSTOMER_TURN":
        st.info(L.get("customer_mode_info", "👤 고객 입장에서 AI 상담원에게 응답을 입력하세요."))
        user_customer_input = st.chat_input(L.get("customer_inquiry_input_placeholder", "문의 사항을 입력하세요 (고객 입장)..."))
        
        if user_customer_input:
            # ⭐ 언어 전환 요청 감지 및 처리
            customer_response = user_customer_input
            language_change_requested = False
            requested_lang = None
            
            # 영어 전환 요청 감지
            english_requests = [
                "can we speak english", "speak english", "english please", 
                "english, please", "in english", "use english",
                "영어로", "영어로 말씀해주세요", "영어로 해주세요", "영어로 부탁합니다"
            ]
            
            # 일본어 전환 요청 감지
            japanese_requests = [
                "日本語で", "日本語でお願いします", "日本語で話してください",
                "speak japanese", "japanese please", "in japanese", "use japanese"
            ]
            
            # 한국어 전환 요청 감지
            korean_requests = [
                "한국어로", "한국어로 말씀해주세요", "한국어로 해주세요",
                "speak korean", "korean please", "in korean", "use korean"
            ]
            
            customer_response_lower = customer_response.lower()
            
            # 언어 전환 요청 확인
            if any(req.lower() in customer_response_lower for req in english_requests):
                requested_lang = "en"
                language_change_requested = True
            elif any(req.lower() in customer_response_lower for req in japanese_requests):
                requested_lang = "ja"
                language_change_requested = True
            elif any(req.lower() in customer_response_lower for req in korean_requests):
                requested_lang = "ko"
                language_change_requested = True
            
            # 언어 전환 처리
            if language_change_requested and requested_lang:
                current_lang = st.session_state.get("language", "ko")
                if requested_lang != current_lang:
                    st.session_state.language = requested_lang
                    lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
                    st.info(f"🌐 언어가 {lang_names[requested_lang]}로 자동 변경되었습니다.")
                    # 언어 변경 후 L 업데이트
                    L = LANG.get(requested_lang, LANG["ko"])
                    current_lang = requested_lang
            
            # 메시지 추가
            new_msg = {"role": "customer", "content": user_customer_input}
            st.session_state.simulator_messages.append(new_msg)
            
            # ⭐ 고객 모드일 때도 closing 단계 전환 로직 적용
            
            # 다국어 지원: 고객의 긍정적 종료 응답 감지 (존경어 표현 포함)
            positive_response_keywords = [
                L["customer_positive_response"],  # 다국어 키 사용
                "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", 
                "承知致しました", "承知いたしました", "了解しました",
                "감사합니다", "thank you", "ありがとうございます", "thanks", "thank"
            ]
            has_positive_response = any(
                keyword.lower() in customer_response.lower() 
                for keyword in positive_response_keywords
            )
            # "알겠습니다" + "감사합니다" 조합도 감지 (다국어, 일본어는 존경어 표현)
            # 마침표, 공백 등을 고려하여 감지
            has_positive_combination = (
                ("알겠습니다" in customer_response or "알겠어요" in customer_response or 
                 "承知致しました" in customer_response or "承知いたしました" in customer_response or
                 "了解しました" in customer_response or
                 "yes" in customer_response.lower() or "ok" in customer_response.lower() or "okay" in customer_response.lower() or
                 "承知" in customer_response or "了解" in customer_response) and
                ("감사합니다" in customer_response or "ありがとうございます" in customer_response or 
                 "ありがとう" in customer_response or
                 "thank you" in customer_response.lower() or "thanks" in customer_response.lower() or "thank" in customer_response.lower())
            ) or (
                # 단독으로도 감지: "ありがとうございます"만 있어도 감지
                "ありがとうございます" in customer_response or 
                "ありがとう" in customer_response or
                "감사합니다" in customer_response or
                "thank you" in customer_response.lower() or "thanks" in customer_response.lower()
            )
            
            # 종료 조건 검토
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
            
            # 솔루션이 제공되었고 고객이 긍정적으로 응답한 경우 closing 단계로 전환
            # ⭐ 고객 모드에서는 에이전트가 응답을 했다면 솔루션이 제공된 것으로 간주
            has_agent_response = any(
                msg.get("role") == "agent_response" 
                for msg in st.session_state.simulator_messages
            )
            is_solution_provided = st.session_state.get("is_solution_provided", False) or has_agent_response
            
            if (L["customer_positive_response"] in customer_response or 
                has_positive_response or has_positive_combination or is_positive_closing):
                if is_solution_provided:
                    st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
                else:
                    # 솔루션이 제공되지 않았으면 일반 응대 계속
                    st.session_state.sim_stage = "AGENT_TURN"
                    st.session_state.ai_agent_response_generated = False
            elif customer_response.startswith(L["customer_escalation_start"]):
                st.session_state.sim_stage = "ESCALATION_REQUIRED"
            else:
                # 다음 단계로 이동 (AI 상담원이 답변할 차례)
                st.session_state.sim_stage = "AGENT_TURN"
                st.session_state.ai_agent_response_generated = False  # AI 응답 생성 플래그 리셋
        return  # 고객 모드일 때는 기존 AI 고객 반응 생성 로직을 실행하지 않음
    
    # ⭐ 상담원 테스트 모드: 기존 로직 (AI가 고객 반응 자동 생성)
    customer_type_display = st.session_state.get(
        "customer_type_sim_select", L["customer_type_options"][0])
    st.info(L["customer_turn_info"])

    # 고객 반응 생성
    last_customer_message = None
    for msg in reversed(st.session_state.simulator_messages):
        if msg.get("role") == "customer" and msg.get("content"):
            last_customer_message = msg.get("content", "")
            break

    if last_customer_message is None:
        # 고객 반응 즉시 생성 (5초 이내 빠른 응답)
        customer_response = generate_customer_reaction(
            st.session_state.language, is_call=False)

        # ⭐ 언어 전환 요청 감지 및 처리 (AI 생성 메시지도 포함)
        language_change_requested = False
        requested_lang = None
        
        # 영어 전환 요청 감지
        english_requests = [
            "can we speak english", "speak english", "english please", 
            "english, please", "in english", "use english",
            "영어로", "영어로 말씀해주세요", "영어로 해주세요", "영어로 부탁합니다"
        ]
        
        # 일본어 전환 요청 감지
        japanese_requests = [
            "日本語で", "日本語でお願いします", "日本語で話してください",
            "speak japanese", "japanese please", "in japanese", "use japanese"
        ]
        
        # 한국어 전환 요청 감지
        korean_requests = [
            "한국어로", "한국어로 말씀해주세요", "한국어로 해주세요",
            "speak korean", "korean please", "in korean", "use korean"
        ]
        
        customer_response_lower = customer_response.lower()
        
        # 언어 전환 요청 확인
        if any(req.lower() in customer_response_lower for req in english_requests):
            requested_lang = "en"
            language_change_requested = True
        elif any(req.lower() in customer_response_lower for req in japanese_requests):
            requested_lang = "ja"
            language_change_requested = True
        elif any(req.lower() in customer_response_lower for req in korean_requests):
            requested_lang = "ko"
            language_change_requested = True
        
        # 언어 전환 처리
        if language_change_requested and requested_lang:
            current_lang_state = st.session_state.get("language", "ko")
            if requested_lang != current_lang_state:
                st.session_state.language = requested_lang
                lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
                st.info(f"🌐 고객의 요청에 따라 언어가 {lang_names[requested_lang]}로 자동 변경되었습니다.")
                # L 업데이트
                L = LANG.get(requested_lang, LANG["ko"])
                current_lang = requested_lang
        else:
            # 언어 전환 요청이 없으면 메시지 언어 자동 감지
            try:
                from utils.customer_analysis import detect_text_language
                detected_lang = detect_text_language(customer_response)
                if detected_lang in ["ko", "en", "ja"]:
                    current_lang_state = st.session_state.get("language", "ko")
                    if detected_lang != current_lang_state:
                        # 감지된 언어가 현재 언어와 다르고, 메시지가 해당 언어로 작성된 경우
                        # (단, 언어 전환 요청이 명확하지 않은 경우만 자동 감지)
                        st.session_state.language = detected_lang
                        lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
                        st.info(f"🌐 입력 언어가 감지되어 언어 설정이 {lang_names[detected_lang]}로 자동 변경되었습니다.")
                        L = LANG.get(detected_lang, LANG["ko"])
                        current_lang = detected_lang
            except Exception as e:
                # 언어 감지 실패 시 현재 언어 유지
                print(f"Language detection failed: {e}")

        # 메시지 추가 및 즉시 화면 반영을 위한 상태 업데이트
        new_message = {"role": "customer", "content": customer_response}
        st.session_state.simulator_messages = st.session_state.simulator_messages + [new_message]
        
        # ⭐ 상태 변경을 명시적으로 트리거하여 즉시 화면 업데이트
        st.session_state._message_update_trigger = not st.session_state.get("_message_update_trigger", False)
        
        # ⭐ 응대초안 즉시 자동 생성 (고객 메시지 수신 시 - 백그라운드에서 조용히)
        # ⭐ API Key 확인
        from llm_client import get_api_key
        has_api_key = any([
            bool(get_api_key("openai")),
            bool(get_api_key("gemini")),
            bool(get_api_key("claude")),
            bool(get_api_key("groq"))
        ])
        
        if has_api_key:
            st.session_state.is_llm_ready = True
            
            # ⭐ 응대 초안 생성 중 플래그로 중복 생성 방지
            if not st.session_state.get("draft_generation_in_progress", False):
                st.session_state.draft_generation_in_progress = True
                try:
                    from simulation_handler import generate_agent_response_draft
                    session_lang = st.session_state.get("language", "ko")
                    if session_lang not in ["ko", "en", "ja"]:
                        session_lang = "ko"
                    
                    # ⭐ 응대 초안 즉시 생성 (백그라운드에서 조용히 - spinner 없이)
                    draft_text = generate_agent_response_draft(session_lang)
                    
                    if draft_text and draft_text.strip():
                        # 마크다운 헤더 제거
                        draft_text_clean = draft_text
                        if "###" in draft_text_clean:
                            lines = draft_text_clean.split("\n")
                            draft_text_clean = "\n".join([line for line in lines if not line.strip().startswith("###")])
                        draft_text_clean = draft_text_clean.strip()
                        
                        if draft_text_clean:
                            # ⭐ 응대 초안 즉시 저장 (AGENT_TURN에서 바로 사용)
                            st.session_state.agent_response_area_text = draft_text_clean
                            st.session_state.auto_draft_generated = True
                            st.session_state.auto_generated_draft_text = draft_text_clean
                            st.session_state.last_draft_for_message_idx = len(st.session_state.simulator_messages) - 1
                            
                            # ⭐ 응대 초안은 입력창에만 표시 (자동 전송하지 않음)
                            # 사용자가 수정 후 직접 전송하도록 함
                            
                            # 디버깅: 응대 초안 생성 확인
                            print(f"✅ 고객 메시지 수신 시 응대 초안 생성 완료 (메시지 인덱스: {len(st.session_state.simulator_messages) - 1})")
                except Exception as e:
                    # 오류 발생 시에도 계속 진행 (조용히)
                    print(f"❌ 응대 초안 자동 생성 오류: {e}")
                    st.session_state.auto_draft_generated = False
                finally:
                    st.session_state.draft_generation_in_progress = False
        else:
            # 응대초안 자동 생성을 위한 플래그 리셋
            st.session_state.auto_draft_generated = False
            st.session_state.auto_generated_draft_text = ""
            st.session_state.last_draft_for_message_idx = -1

        # 다음 단계 결정
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

        if L["customer_positive_response"] in customer_response:
            if st.session_state.is_solution_provided:
                st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
            else:
                st.session_state.sim_stage = "AGENT_TURN"
        elif is_positive_closing:
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
            st.session_state.sim_stage = "AGENT_TURN"
    else:
        # 기존 고객 메시지가 있는 경우
        customer_response = last_customer_message
        
        # ⭐ 기존 고객 메시지에서도 언어 전환 요청 확인
        if customer_response:
            language_change_requested = False
            requested_lang = None
            
            # 영어 전환 요청 감지
            english_requests = [
                "can we speak english", "speak english", "english please", 
                "english, please", "in english", "use english",
                "영어로", "영어로 말씀해주세요", "영어로 해주세요", "영어로 부탁합니다"
            ]
            
            # 일본어 전환 요청 감지
            japanese_requests = [
                "日本語で", "日本語でお願いします", "日本語で話してください",
                "speak japanese", "japanese please", "in japanese", "use japanese"
            ]
            
            # 한국어 전환 요청 감지
            korean_requests = [
                "한국어로", "한국어로 말씀해주세요", "한국어로 해주세요",
                "speak korean", "korean please", "in korean", "use korean"
            ]
            
            customer_response_lower = customer_response.lower()
            
            # 언어 전환 요청 확인
            if any(req.lower() in customer_response_lower for req in english_requests):
                requested_lang = "en"
                language_change_requested = True
            elif any(req.lower() in customer_response_lower for req in japanese_requests):
                requested_lang = "ja"
                language_change_requested = True
            elif any(req.lower() in customer_response_lower for req in korean_requests):
                requested_lang = "ko"
                language_change_requested = True
            
            # 언어 전환 처리
            if language_change_requested and requested_lang:
                current_lang_state = st.session_state.get("language", "ko")
                if requested_lang != current_lang_state:
                    st.session_state.language = requested_lang
                    lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
                    st.info(f"🌐 고객의 요청에 따라 언어가 {lang_names[requested_lang]}로 자동 변경되었습니다.")
                    # L 업데이트
                    L = LANG.get(requested_lang, LANG["ko"])
                    current_lang = requested_lang
            else:
                # 언어 전환 요청이 없으면 메시지 언어 자동 감지
                try:
                    from utils.customer_analysis import detect_text_language
                    detected_lang = detect_text_language(customer_response)
                    if detected_lang in ["ko", "en", "ja"]:
                        current_lang_state = st.session_state.get("language", "ko")
                        if detected_lang != current_lang_state:
                            st.session_state.language = detected_lang
                            lang_names = {"ko": "한국어", "en": "English", "ja": "日本語"}
                            st.info(f"🌐 입력 언어가 감지되어 언어 설정이 {lang_names[detected_lang]}로 자동 변경되었습니다.")
                            L = LANG.get(detected_lang, LANG["ko"])
                            current_lang = detected_lang
                except Exception as e:
                    # 언어 감지 실패 시 현재 언어 유지
                    print(f"Language detection failed: {e}")

    # 종료 조건 검토
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

    # 메일 응대 종료 문구 확인
    is_email_closing = st.session_state.get("has_email_closing", False)

    if not is_email_closing:
        last_agent_response = None
        for msg in reversed(st.session_state.simulator_messages):
            if msg.get("role") == "agent_response" and msg.get("content"):
                last_agent_response = msg.get("content", "")
                break

        email_closing_patterns = [
            "추가 문의사항이 있으면 언제든지 연락",
            "추가 문의 사항이 있으면 언제든지 연락",
            "additional inquiries", "any additional questions",
            "feel free to contact", "please feel free to contact",
            "追加のご質問", "追加のお問い合わせ"]
        if last_agent_response:
            is_email_closing = any(pattern.lower() in last_agent_response.lower(
            ) for pattern in email_closing_patterns)
            if is_email_closing:
                st.session_state.has_email_closing = True

    # 메일 끝인사 처리
    if is_email_closing:
        no_more_keywords = [
            L['customer_no_more_inquiries'],
            "No, that will be all", "no more", "없습니다", "감사합니다",
            "Thank you", "ありがとうございます", "추가 문의 사항 없습니다",
            "no additional", "追加の質問はありません", "알겠습니다", "ok", "네", "yes"]
        has_no_more_inquiry = False
        for keyword in no_more_keywords:
            escaped = re.escape(keyword)
            pattern = escaped.replace(
                r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
            regex = re.compile(pattern, re.IGNORECASE)
            if regex.search(customer_response):
                has_no_more_inquiry = True
                break
        if "없습니다" in customer_response and "감사합니다" in customer_response:
            has_no_more_inquiry = True

        positive_keywords = [
            "알겠습니다", "알겠어요", "네", "yes", "ok", "okay",
            "감사합니다", "thank you", "ありがとうございます", "좋습니다", "good", "fine", "괜찮습니다"]
        is_positive_response = any(
            keyword.lower() in customer_response.lower() for keyword in positive_keywords)

        escaped_check = re.escape(L['customer_no_more_inquiries'])
        no_more_pattern_check = escaped_check.replace(
            r'\.', r'[.\\s]*').replace(r'\ ', r'[.\\s]*')
        no_more_regex_check = re.compile(no_more_pattern_check, re.IGNORECASE)
        if is_positive_closing or has_no_more_inquiry or no_more_regex_check.search(
                customer_response) or is_positive_response:
            agent_closing_added = False
            for msg in reversed(st.session_state.simulator_messages):
                if msg.get("role") == "agent_response":
                    agent_msg_content = msg.get("content", "")
                    if "감사" in agent_msg_content or "Thank you" in agent_msg_content or "ありがとう" in agent_msg_content:
                        agent_closing_added = True
                    break

            if not agent_closing_added:
                agent_name = st.session_state.get("agent_name", "000")
                if current_lang == "ko":
                    agent_closing_msg = f"연락 주셔서 감사드립니다. 지금까지 상담원 {agent_name}였습니다. 즐거운 하루 되세요."
                elif current_lang == "en":
                    agent_closing_msg = f"Thank you for contacting us. This was {agent_name}. Have a great day!"
                else:
                    agent_closing_msg = f"お問い合わせいただき、ありがとうございました。担当は{agent_name}でした。良い一日をお過ごしください。"

                st.session_state.simulator_messages.append(
                    {"role": "agent_response", "content": agent_closing_msg}
                )

            st.session_state.sim_stage = "WAIT_CUSTOMER_CLOSING_RESPONSE"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
    # 다국어 지원: 고객의 긍정적 종료 응답 감지 (일본어는 존경어 표현 사용)
    positive_response_keywords = [
        L["customer_positive_response"],  # 다국어 키 사용
        "알겠습니다", "알겠어요", "네", "yes", "ok", "okay", 
        "承知致しました", "承知いたしました", "了解しました",
        "감사합니다", "thank you", "ありがとうございます", "thanks", "thank"
    ]
    # 마침표, 공백 등을 제거하여 정확한 매칭
    customer_response_clean = customer_response.replace("。", "").replace(".", "").replace(" ", "").strip()
    has_positive_response = any(
        keyword.lower() in customer_response.lower() or 
        keyword.replace("。", "").replace(".", "").replace(" ", "").lower() in customer_response_clean.lower()
        for keyword in positive_response_keywords
    )
    # "알겠습니다" + "감사합니다" 조합도 감지 (다국어, 일본어는 존경어 표현)
    # 단독으로도 감지: "ありがとうございます"만 있어도 감지
    has_positive_combination = (
        ("알겠습니다" in customer_response or "알겠어요" in customer_response or 
         "承知致しました" in customer_response or "承知いたしました" in customer_response or
         "了解しました" in customer_response or
         "yes" in customer_response.lower() or "ok" in customer_response.lower() or "okay" in customer_response.lower() or
         "承知" in customer_response or "了解" in customer_response) and
        ("감사합니다" in customer_response or "ありがとうございます" in customer_response or 
         "ありがとう" in customer_response or
         "thank you" in customer_response.lower() or "thanks" in customer_response.lower() or "thank" in customer_response.lower())
    ) or (
        # 단독 감사 표현도 감지
        "ありがとうございます" in customer_response or 
        "ありがとう" in customer_response or
        "감사합니다" in customer_response or
        "thank you" in customer_response.lower() or "thanks" in customer_response.lower()
    )
    
    if L["customer_positive_response"] in customer_response or has_positive_response or has_positive_combination:
        if st.session_state.is_solution_provided:
            st.session_state.sim_stage = "WAIT_CLOSING_CONFIRMATION_FROM_AGENT"
        else:
            st.session_state.sim_stage = "AGENT_TURN"
    elif is_positive_closing:
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
        st.session_state.sim_stage = "AGENT_TURN"
        # 응대 초안 자동 생성을 위한 플래그 리셋
        st.session_state.auto_draft_generated = False

    st.session_state.is_solution_provided = False

    # 이력 저장
    if st.session_state.sim_stage != "CLOSING":
        save_simulation_history_local(
            st.session_state.customer_query_text_area,
            customer_type_display,
            st.session_state.simulator_messages,
            is_chat_ended=False,
            attachment_context=st.session_state.sim_attachment_context_for_llm,
        )

    st.session_state.realtime_hint_text = ""

