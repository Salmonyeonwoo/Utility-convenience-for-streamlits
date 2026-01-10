# ========================================
# _pages/_content.py
# 콘텐츠 생성 모듈 (메인)
# ========================================

import streamlit as st
from lang_pack import LANG
from simulation_handler import run_llm
from _pages._content_utils import get_level_map, get_content_map
from _pages._content_quiz import generate_quiz, parse_quiz_json, render_quiz_completion, render_quiz_question
from _pages._content_visualization import render_content_visualization
from _pages._content_share import render_share_buttons


def render_content():
    """콘텐츠 생성 렌더링 함수"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    
    st.header(L.get("content_header", "콘텐츠 생성"))
    st.markdown(L.get("content_desc", "학습 콘텐츠를 생성합니다."))
    st.markdown("---")

    if not st.session_state.is_llm_ready:
        st.warning(L.get("simulation_no_key_warning", "API Key가 설정되지 않았습니다."))
        st.info("💡 API Key를 설정하면 콘텐츠 생성 기능을 사용할 수 있습니다.")
        # st.stop() 제거: UI는 표시하되 기능만 비활성화

    # 다국어 맵핑 변수
    level_map = get_level_map()
    content_map = get_content_map()

    topic = st.text_input(L.get("topic_label", "주제"))
    level_display = st.selectbox(L.get("level_label", "난이도"), L.get("level_options", ["초급", "중급", "고급"]))
    content_display = st.selectbox(
        L.get("content_type_label", "콘텐츠 유형"),
        L.get("content_options", ["핵심 요약 노트", "객관식 퀴즈 10문항", "실습 예제 아이디어"]))

    level = level_map.get(level_display, "Beginner")
    content_type = content_map.get(content_display, "summary")

    if st.button(L.get("button_generate", "생성")):
        if not topic.strip():
            st.warning(L.get("warning_topic", "주제를 입력해주세요."))
            # st.stop() 제거: 경고만 표시하고 계속 진행
        elif not st.session_state.is_llm_ready:
            st.error("❌ LLM이 준비되지 않았습니다. API Key를 설정해주세요.")
            # st.stop() 제거: 에러만 표시하고 계속 진행
        else:
            target_lang = {
                "ko": "Korean",
                "en": "English",
                "ja": "Japanese"}.get(
                st.session_state.language, "Korean")

            # 공통 프롬프트 설정 (퀴즈 형식을 포함하지 않는 기본 템플릿)
            system_prompt = (
                f"You are a professional AI coach. Generate learning content in {target_lang} "
                f"for the topic '{topic}' at the '{level}' difficulty. "
                f"The content format requested is: {content_display}. "
                f"Output ONLY the raw content.")

            if content_type == "quiz":
                lang_instruction = {
                    "ko": "한국어로", "en": "in English", "ja": "日本語で"
                }.get(st.session_state.language, "in Korean")
                
                generated_json_text, raw_response_text = generate_quiz(topic, level, lang_instruction, L)
                quiz_data, error = parse_quiz_json(generated_json_text, raw_response_text, L)
                
                if quiz_data:
                    st.success(f"**{topic}** - {content_display} 생성 완료")
                elif error:
                    error_type, error_msg = error
                    st.error(L.get("quiz_error_llm", "퀴즈 생성에 실패했습니다."))
                    st.caption(f"{error_type}: {error_msg}")
                    if raw_response_text or generated_json_text:
                        st.subheader(L.get("quiz_original_response", "원본 응답"))
                        st.text_area("", raw_response_text or generated_json_text or "", height=300)
            else:  # 일반 텍스트 생성
                st.session_state.is_quiz_active = False
                with st.spinner(L.get("response_generating", "생성 중...")):
                    content = run_llm(system_prompt)
                st.session_state.generated_content = content

                st.markdown("---")
                st.markdown(f"### {content_display}")
                st.markdown(st.session_state.generated_content)

    # --- 퀴즈/일반 콘텐츠 출력 로직 ---
    if st.session_state.get("is_quiz_active", False) and st.session_state.get("quiz_data"):
        quiz_data = st.session_state.quiz_data
        idx = st.session_state.current_question_index

        if idx >= len(quiz_data):
            # 퀴즈 완료
            render_quiz_completion(quiz_data, L)
        else:
            # 퀴즈 진행 중
            render_quiz_question(quiz_data, idx, L)

    else:
        # 일반 콘텐츠 (핵심 요약 노트, 실습 예제 아이디어) 출력
        if st.session_state.get("generated_content"):
            content = st.session_state.generated_content
            st.markdown("---")
            st.markdown(f"### {content_display}")
            
            # 콘텐츠 시각화
            render_content_visualization(content, level, L)
            
            st.markdown("---")
            st.markdown(f"### 📝 원본 콘텐츠")
            st.markdown(content)
            
            # 공유 버튼
            st.markdown("---")
            render_share_buttons(content, content_display, topic, L)

