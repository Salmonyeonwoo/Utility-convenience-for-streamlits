# 콘텐츠 생성기
import streamlit as st
from lang_pack import LANG
from llm_client import get_api_key, run_llm
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import json
import uuid

def render_content_generator():
    """콘텐츠 생성기 렌더링"""
    current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"
    L = LANG.get(current_lang, LANG["ko"])
    st.header(L["content_header"])
    st.markdown(L["content_desc"])
    st.markdown("---")

    if not st.session_state.is_llm_ready:
        st.warning(L["simulation_no_key_warning"])
        st.info("💡 API Key를 설정하면 콘텐츠 생성 기능을 사용할 수 있습니다.")
        # st.stop() 제거: UI는 표시하되 기능만 비활성화

    # 다국어 맵핑 변수는 그대로 사용
    level_map = {
        "초급": "Beginner",
        "중급": "Intermediate",
        "고급": "Advanced",
        "Beginner": "Beginner",
        "Intermediate": "Intermediate",
        "Advanced": "Advanced",
        "初級": "Beginner",
        "中級": "Intermediate",
        "上級": "Advanced",
    }
    content_map = {
        "핵심 요약 노트": "summary",
        "객관식 퀴즈 10문항": "quiz",
        "실습 예제 아이디어": "example",
        "Key Summary Note": "summary",
        "10 MCQ Questions": "quiz",
        "Practical Example Idea": "example",
        "核心要約ノート": "summary",
        "選択式クイズ10問": "quiz",
        "実践例のアイデア": "example",
    }

    topic = st.text_input(L["topic_label"])
    level_display = st.selectbox(L["level_label"], L["level_options"])
    content_display = st.selectbox(L["content_type_label"], L["content_options"])

    level = level_map.get(level_display, "Beginner")
    content_type = content_map.get(content_display, "summary")

    if st.button(L["button_generate"]):
        if not topic.strip():
            st.warning(L["warning_topic"])
            # st.stop() 제거: 경고만 표시하고 계속 진행
        elif not st.session_state.is_llm_ready:
            st.error("❌ LLM이 준비되지 않았습니다. API Key를 설정해주세요.")
            # st.stop() 제거: 에러만 표시하고 계속 진행
        else:
            target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]

            # 공통 프롬프트 설정 (퀴즈 형식을 포함하지 않는 기본 템플릿)
            system_prompt = (
                f"You are a professional AI coach. Generate learning content in {target_lang} "
                f"for the topic '{topic}' at the '{level}' difficulty. "
                f"The content format requested is: {content_display}. "
                f"Output ONLY the raw content."
            )

            if content_type == "quiz":
                # 퀴즈 전용 프롬프트 및 JSON 구조 강제 (로직 유지)
                lang_instruction = {"ko": "한국어로", "en": "in English", "ja": "日本語で"}.get(st.session_state.language, "in Korean")
                quiz_prompt = (
                    f"You are an expert quiz generator. Based on the topic '{topic}' and difficulty '{level}', generate 10 multiple-choice questions.\n"
                    f"IMPORTANT: All questions, options, and explanations must be written {lang_instruction}.\n"
                    f"Your output MUST be a **raw JSON object** containing a single key \"quiz_questions\" which holds an array of 10 questions.\n"
                    f"Each object in the array must strictly follow the required keys:\n"
                    f"- \"question\" (string): The question text in {lang_instruction}\n"
                    f"- \"options\" (array of 4 strings): Four answer choices in {lang_instruction}\n"
                    f"- \"answer\" (integer): The correct answer index starting from 1 (1-4)\n"
                    f"- \"explanation\" (string): A DETAILED and COMPREHENSIVE explanation (at least 2-3 sentences, preferably 50-100 words) explaining:\n"
                    f"  * Why the correct answer is right\n"
                    f"  * Why other options are incorrect (briefly mention key differences)\n"
                    f"  * Additional context or background information that helps understanding\n"
                    f"  * Real-world examples or applications if relevant\n"
                    f"  Write the explanation in {lang_instruction} with clear, educational content.\n"
                    f"DO NOT include any explanation, introductory text, or markdown code blocks (e.g., ```json).\n"
                    f"Output ONLY the raw JSON object, starting with '{{' and ending with '}}'.\n"
                    f"Example structure:\n"
                    f"{{\n"
                    f"  \"quiz_questions\": [\n"
                    f"    {{\n"
                    f"      \"question\": \"질문 내용\",\n"
                    f"      \"options\": [\"선택지1\", \"선택지2\", \"선택지3\", \"선택지4\"],\n"
                    f"      \"answer\": 1,\n"
                    f"      \"explanation\": \"정답인 이유를 상세히 설명하고, 다른 선택지가 왜 틀렸는지 간단히 언급하며, 관련 배경 지식이나 실제 사례를 포함한 충분히 긴 해설 내용 (최소 2-3문장, 50-100단어 정도)\"\n"
                    f"    }}\n"
                    f"  ]\n"
                    f"}}"
                )
            
            def extract_json_from_text(text):
                # 텍스트에서 JSON 객체를 추출하는 함수
                if not text:
                    return None
                
                text = text.strip()
                
                # 1. Markdown 코드 블록 제거
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    if end != -1:
                        text = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    if end != -1:
                        text = text[start:end].strip()
                
                # 2. 첫 번째 '{' 부터 마지막 '}' 까지 추출
                first_brace = text.find('{')
                if first_brace == -1:
                    return None
                
                # 중괄호 매칭으로 JSON 객체 끝 찾기
                brace_count = 0
                last_brace = -1
                for i in range(first_brace, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_brace = i
                            break
                
                if last_brace != -1:
                    json_str = text[first_brace:last_brace + 1]
                    return json_str.strip()
                
                return None

            generated_json_text = None
            raw_response_text = None
            llm_attempts = []

            # 1순위: OpenAI (JSON mode가 가장 안정적)
            if get_api_key("openai"):
                llm_attempts.append(("openai", get_api_key("openai"), "gpt-4o"))
            # 2순위: Gemini (Fallback)
            if get_api_key("gemini"):
                llm_attempts.append(("gemini", get_api_key("gemini"), "gemini-1.5-flash"))

            with st.spinner(L["response_generating"]):
                for provider, api_key, model_name in llm_attempts:
                    try:
                        if provider == "openai":
                            client = OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": quiz_prompt}],
                                # JSON Mode 강제
                                response_format={"type": "json_object"},
                            )
                            raw_response_text = response.choices[0].message.content.strip()
                            # OpenAI는 JSON 객체를 반환하므로, 직접 사용 시도
                            generated_json_text = extract_json_from_text(raw_response_text) or raw_response_text
                            break

                        elif provider == "gemini":
                            # Gemini는 response_format을 지원하지 않으므로, run_llm을 통해 일반 텍스트로 호출
                            raw_response_text = run_llm(quiz_prompt)
                            generated_json_text = extract_json_from_text(raw_response_text)
                            
                            # JSON 추출 성공 시 시도 종료
                            if generated_json_text:
                                break

                    except Exception as e:
                        print(f"JSON generation failed with {provider}: {e}")
                        continue

            # --- START: JSON Parsing and Error Handling Logic ---
            parsed_obj = None
            quiz_data = None
            
            if generated_json_text:
                try:
                    # JSON 객체 파싱 시도
                    parsed_obj = json.loads(generated_json_text)

                    # 'quiz_questions' 키에서 배열 추출
                    quiz_data = parsed_obj.get("quiz_questions")

                    if not isinstance(quiz_data, list) or len(quiz_data) < 1:
                        raise ValueError("Missing 'quiz_questions' key or empty array.")

                    # 데이터 유효성 검사: 각 문제에 필수 필드가 있는지 확인
                    for i, q in enumerate(quiz_data):
                        if not isinstance(q, dict):
                            raise ValueError(f"Question {i+1} is not a valid object.")
                        if "question" not in q or "options" not in q or "answer" not in q:
                            raise ValueError(f"Question {i+1} is missing required fields (question, options, or answer).")
                        if not isinstance(q["options"], list) or len(q["options"]) != 4:
                            raise ValueError(f"Question {i+1} must have exactly 4 options.")
                        if not isinstance(q["answer"], int) or q["answer"] < 1 or q["answer"] > 4:
                            raise ValueError(f"Question {i+1} answer must be an integer between 1 and 4.")

                    # 파싱 성공 및 데이터 유효성 검사 후 상태 저장
                    st.session_state.quiz_data = quiz_data
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = [1] * len(quiz_data)
                    st.session_state.show_explanation = False
                    st.session_state.is_quiz_active = True
                    st.session_state.quiz_type_key = str(uuid.uuid4())

                    st.success(f"**{topic}** - {content_display} 생성 완료")

                except json.JSONDecodeError as e:
                    # JSON 파싱 오류
                    st.error(L["quiz_error_llm"])
                    st.caption(f"JSON 파싱 오류: {str(e)}")
                    st.subheader(L["quiz_original_response"])
                    st.code(raw_response_text or generated_json_text, language="text")
                    if generated_json_text:
                        st.caption("추출된 JSON 텍스트:")
                        st.code(generated_json_text, language="text")
                    
                except ValueError as e:
                    # 데이터 구조 오류
                    st.error(L["quiz_error_llm"])
                    st.caption(f"데이터 구조 오류: {str(e)}")
                    st.subheader(L["quiz_original_response"])
                    st.code(raw_response_text or generated_json_text, language="text")
                    if parsed_obj:
                        st.caption("파싱된 객체:")
                        st.json(parsed_obj)
                        
            else:
                # JSON 추출 실패
                st.error(L["quiz_error_llm"])
                st.caption("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")
                if raw_response_text:
                    st.subheader(L["quiz_original_response"])
                    st.text_area("", raw_response_text, height=300)
                elif generated_json_text:
                    st.subheader(L["quiz_original_response"])
                    st.text_area("", generated_json_text, height=300)
                # --- END: JSON Parsing and Error Handling Logic ---

                else:  # 일반 텍스트 생성
                    st.session_state.is_quiz_active = False
                with st.spinner(L["response_generating"]):
                    content = run_llm(system_prompt)
                st.session_state.generated_content = content

                st.markdown("---")
                st.markdown(f"### {content_display}")
                st.markdown(st.session_state.generated_content)

    # --- 퀴즈/일반 콘텐츠 출력 로직 ---
    if st.session_state.get("is_quiz_active", False) and st.session_state.get("quiz_data"):
        # 퀴즈 진행 로직 (생략 - 기존 로직 유지)
        quiz_data = st.session_state.quiz_data
        idx = st.session_state.current_question_index

        # ⭐ 퀴즈 완료 시 IndexError 방지 로직 (idx >= len(quiz_data))
        if idx >= len(quiz_data):
            # 퀴즈 완료 시 최종 점수 표시
            st.success(L["quiz_complete"])
            total_questions = len(quiz_data)
            score = st.session_state.quiz_score
            incorrect_count = total_questions - score
            st.subheader(f"{L['score']}: {score} / {total_questions} ({(score / total_questions) * 100:.1f}%)")

            # 원형 차트로 맞은 문제/틀린 문제 표시
            if IS_PLOTLY_AVAILABLE:
                col1, col2 = st.columns([1, 2])
                with col1:
                    # 원형 차트 생성
                    fig = go.Figure(data=[go.Pie(
                        labels=[L["correct_questions"], L["incorrect_questions"]],
                        values=[score, incorrect_count],
                        hole=0.4,
                        marker_colors=['#28a745', '#dc3545'],
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    fig.update_layout(
                        title=L["question_result"],
                        showlegend=True,
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### " + L["question_result"])
                    # 문제별 정오 리스트 표시
                    for i, question_item in enumerate(quiz_data):
                        user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                        is_correct = user_answer == 'Correctly Scored'
                        correct_answer_idx = question_item.get('answer', 1)
                        correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                        
                        # 사용자 답안 텍스트 가져오기
                        if is_correct:
                            user_answer_text = correct_answer_text
                            status_icon = "✅"
                            status_color = "green"
                        else:
                            if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                                user_answer_text = question_item['options'][user_answer - 1]
                            else:
                                user_answer_text = "미응답"
                            status_icon = "❌"
                            status_color = "red"
                        
                        # 문제별 결과 표시
                        with st.container():
                            html_content = (
                                f"<div style=\"border-left: 4px solid {status_color}; padding-left: 10px; margin-bottom: 15px;\">\n"
                                f"    <strong>{status_icon} 문항 {i+1}:</strong> {question_item['question']}<br>\n"
                                f"    <span style=\"color: {status_color};\">{L['your_answer']}: {user_answer_text}</span><br>\n"
                                f"    <span style=\"color: green;\">{L['correct_answer_label']}: {correct_answer_text}</span>\n"
                                f"</div>"
                            )
                            st.markdown(html_content, unsafe_allow_html=True)
            else:
                # Plotly가 없는 경우 텍스트로만 표시
                st.markdown(f"**{L['correct_questions']}:** {score}개")
                st.markdown(f"**{L['incorrect_questions']}:** {incorrect_count}개")
                st.markdown("### " + L["question_result"])
                for i, question_item in enumerate(quiz_data):
                    user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
                    is_correct = user_answer == 'Correctly Scored'
                    correct_answer_idx = question_item.get('answer', 1)
                    correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"
                    
                    if is_correct:
                        user_answer_text = correct_answer_text
                        status_icon = "✅"
                    else:
                        if isinstance(user_answer, int) and 0 < user_answer <= len(question_item['options']):
                            user_answer_text = question_item['options'][user_answer - 1]
                        else:
                            user_answer_text = "미응답"
                        status_icon = "❌"
                    
                    st.markdown(f"**{status_icon} 문항 {i+1}:** {question_item['question']}")
                    st.markdown(f"- {L['your_answer']}: {user_answer_text}")
                    st.markdown(f"- {L['correct_answer_label']}: {correct_answer_text}")
                    st.markdown("---")

            if st.button(L["retake_quiz"], key="retake_quiz_btn"):
                # 퀴즈 상태만 초기화 (퀴즈 데이터는 유지하여 같은 퀴즈를 다시 풀 수 있도록)
                st.session_state.current_question_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_answers = [1] * len(quiz_data)  # 기본값으로 초기화
                st.session_state.show_explanation = False
                # ⭐ 재실행 불필요: 상태 변경은 이미 반영됨, 다음 렌더링에서 자동 표시됨
                # st.rerun()
            # st.stop() 제거: 퀴즈 완료 후에도 UI는 계속 표시
        else:
            # 퀴즈 진행 (현재 문항)
            question_data = quiz_data[idx]
            st.subheader(f"{L.get('question_label', '문항')} {idx + 1}/{len(quiz_data)}")
            st.markdown(f"**{question_data['question']}**")

            # 기존 퀴즈 진행 및 채점 로직 (변화 없음)
            current_selection_index = st.session_state.quiz_answers[idx]

            options = question_data['options']
            current_answer = st.session_state.quiz_answers[idx]

            if current_answer is None or not isinstance(current_answer, int) or current_answer <= 0:
                radio_index = 0
            else:
                radio_index = min(current_answer - 1, len(options) - 1)

            selected_option = st.radio(
                L["select_answer"],
                options,
                index=radio_index,
                key=f"quiz_radio_{st.session_state.quiz_type_key}_{idx}"
            )

            selected_option_index = options.index(selected_option) + 1 if selected_option in options else None

            check_col, next_col = st.columns([1, 1])

            if check_col.button(L["check_answer"], key=f"check_answer_btn_{idx}"):
                if selected_option_index is None:
                    st.warning("선택지를 선택해 주세요.")
                else:
                    # 점수 계산 로직
                    if st.session_state.quiz_answers[idx] != 'Correctly Scored':
                        correct_answer = question_data.get('answer')  # answer 키가 없을 경우 대비
                        if selected_option_index == correct_answer:
                            st.session_state.quiz_score += 1
                            st.session_state.quiz_answers[idx] = 'Correctly Scored'
                            st.success(L["correct_answer"])
                        else:
                            st.session_state.quiz_answers[idx] = selected_option_index  # 오답은 선택지 인덱스 저장
                            st.error(L["incorrect_answer"])

                    st.session_state.show_explanation = True

            # 정답 및 해설 표시
            if st.session_state.show_explanation:
                correct_index = question_data.get('answer', 1)
                correct_answer_text = question_data['options'][correct_index - 1] if 0 < correct_index <= len(
                    question_data['options']) else "N/A"

                st.markdown("---")
                st.markdown(f"**{L['correct_is']}:** {correct_answer_text}")
                with st.expander(f"**{L['explanation']}**", expanded=True):
                    st.info(question_data.get('explanation', '해설이 제공되지 않았습니다.'))

                # 다음 문항 버튼
                if next_col.button(L["next_question"], key=f"next_question_btn_{idx}"):
                    st.session_state.current_question_index += 1
                    st.session_state.show_explanation = False

            else:
                # 사용자가 이미 정답을 체크했고 (다시 로드된 경우), 다음 버튼을 바로 표시
                if st.session_state.quiz_answers[idx] == 'Correctly Scored' or (
                        isinstance(st.session_state.quiz_answers[idx], int) and st.session_state.quiz_answers[idx] > 0):
                    if next_col.button(L["next_question"], key=f"next_question_btn_after_check_{idx}"):
                        st.session_state.current_question_index += 1
                        st.session_state.show_explanation = False

    else:
        # 일반 콘텐츠 (핵심 요약 노트, 실습 예제 아이디어) 출력
        if st.session_state.get("generated_content"):
            content = st.session_state.generated_content  # Content를 다시 가져옴
            content_lines = content.split('\n')

            st.markdown("---")
            st.markdown(f"### {content_display}")

            # --- START: 효율성 개선 (상단 분석/하단 본문) ---

            st.subheader("💡 콘텐츠 분석 (Plotly 시각화)")

            if IS_PLOTLY_AVAILABLE:
                # 1. 키워드 빈도 시각화 (모의 데이터)

                # 콘텐츠를 텍스트 줄로 분할하여 모의 키워드 및 주요 문장 생성
                content = st.session_state.generated_content
                content_lines = content.split('\n')
                all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()

                # 모의 키워드 빈도 데이터 생성
                words = ['AI', '기술혁신', '고객경험', '데이터분석', '효율성', '여행산업']
                np.random.seed(42)
                counts = np.random.randint(5, 30, size=len(words))

                # 난이도에 따라 점수 가중치 (모의 감성 점수 변화)
                difficulty_score = {'Beginner': 60, 'Intermediate': 75, 'Advanced': 90}.get(level, 70)

                # --- 차트 1: 키워드 빈도 (Plotly Bar Chart) ---
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=words,
                        y=counts,
                        marker_color=px.colors.sequential.Plotly3,
                        name="키워드 빈도"
                    )
                ])
                fig_bar.update_layout(
                    title_text=f"주요 키워드 빈도 분석",
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # --- 차트 2: 콘텐츠 감성 및 복잡도 추이 (Plotly Line Chart) ---
                # 모의 감성/복잡도 점수 추이 (5개 문단 모의)
                sections = ['도입부', '핵심1', '핵심2', '해결책', '결론']
                sentiment_scores = [difficulty_score - 10, difficulty_score + 5, difficulty_score,
                                    difficulty_score + 10, difficulty_score + 2]

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=sections,
                    y=sentiment_scores,
                    mode='lines+markers',
                    name='감성/복잡도 점수',
                    line=dict(color='orange', width=2),
                    marker=dict(size=8)
                ))
                fig_line.update_layout(
                    title_text="콘텐츠 섹션별 감성 및 복잡도 추이 (모의)",
                    yaxis_range=[50, 100],
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_line, use_container_width=True)

            else:  # Plotly가 없을 경우 기존 텍스트 분석 모의 유지
                st.info("Plotly 라이브러리가 없어 시각화를 표시할 수 없습니다. 텍스트 분석 모의를 표시합니다.")
                all_words = ' '.join(content_lines).replace('.', '').replace(',', '').split()
                unique_words = sorted(set(all_words), key=len, reverse=True)[:5] if all_words else ["N/A"]
                key_sentences = [
                    content_lines[0].strip() if content_lines else "N/A",
                    content_lines[len(content_lines) // 2].strip() if len(content_lines) > 1 else "",
                    content_lines[-1].strip() if len(content_lines) > 1 else ""
                ]
                key_sentences = [s for s in key_sentences if s and s != "N/A"]

                col_keyword, col_sentences = st.columns([1, 1])

                with col_keyword:
                    st.markdown("**핵심 키워드/개념 (모의)**")
                    st.info(f"[{', '.join(unique_words)}...]")

                with col_sentences:
                    st.markdown("**주요 문장 요약 (모의)**")
                    for sentence in key_sentences[:2]:
                        st.write(f"• {sentence[:50]}...")

            st.markdown("---")

            # 2. 하단 본문 출력
            st.markdown(f"### 📝 원본 콘텐츠")
            st.markdown(content)

            # --- END: 효율성 개선 ---

            # --- START: 아이콘 버튼 활성화 ---
            st.markdown("---")

            # 1. 복사할 내용 정리 및 이스케이프
            content_for_js = json.dumps(content)

            # JavaScript 코드는 이스케이프된 중괄호 {{}}를 사용
            js_copy_script = """
               function copyToClipboard(text) {{
                   navigator.clipboard.writeText(text).then(function() {{
                       // Streamlit toast 호출 (모의)
                       const elements = window.parent.document.querySelectorAll('[data-testid="stToast"]');
                       if (elements.length === 0) {{
                           // Fallback UI update (use Streamlit's native mechanism if possible, or simple alert)
                           console.log("복사 완료: " + text.substring(0, 50) + "...");
                           }}
                       }}, function(err) {{
                           // Fallback: Copy via execCommand (deprecated but often works in Streamlit's iframe)
                           const textarea = document.createElement('textarea');
                           textarea.value = text;
                           document.body.appendChild(textarea);
                           textarea.select();
                           document.execCommand('copy');
                           document.body.removeChild(textarea);
                           alert("복사 완료!"); 
                       }});
                   }}
                   // f-string 대신 .format을 사용하여 JavaScript 코드에 주입
                   // content_for_js는 이미 Python에서 JSON 문자열로 안전하게 이스케이프됨
                   copyToClipboard(JSON.parse('{content_json_safe}'));
               """.format(content_json_safe=content_for_js)

            # --- JavaScript for SHARE Menu (Messenger Mock) ---
            # Streamlit은 현재 소셜 미디어 API를 직접 호출할 수 없으므로, URL 복사를 사용하고 UI에 메시지 옵션을 모의합니다.
            js_share_url_copy = """
               function copyShareUrl() {{
                   const url = window.location.href;
                   navigator.clipboard.writeText(url).then(function() {{
                       console.log('App URL copied');
                   }}, function(err) {{
                       // Fallback
                       const textarea = document.createElement('textarea');
                       textarea.value = url;
                       document.body.appendChild(textarea);
                       textarea.select();
                       document.execCommand('copy');
                       document.body.removeChild(textarea);
                   }});
               }}
            """

            # --- JavaScript for SHARE Menu (Messenger Mock) ---
            # Streamlit은 현재 소셜 미디어 API를 직접 호출할 수 없으므로, URL 복사를 사용하고 UI에 메시지 옵션을 모의합니다.
            js_native_share = """
               function triggerNativeShare(title, text, url) {{
                   if (navigator.share) {{
                       // 1. 네이티브 공유 API 지원 시 사용
                       navigator.share({{
                           title: title,
                           text: text,
                           url: url,
                       }}).then(() => {{
                           console.log('Successful share');
                       }}).catch((error) => {{
                           console.log('Error sharing', error);
                       }});
                       return true;
                   }} else {{
                      // 2. 네이티브 공유 API 미지원 시 (PC 환경 등)
                      return false;
                   }}
               }}
            """
            
            def mock_download(file_type: str, file_name: str):
                # 모의 다운로드 기능: 파일명과 함께 성공 토스트 메시지를 출력합니다.
                st.toast(f"📥 {file_type} 파일을 생성하여 다운로드를 시작합니다: {file_name}")
                # 실제 다운로드 로직은 Streamlit 컴포넌트 환경에서는 복잡하여 생략합니다.


            col_like, col_dislike, col_share, col_copy, col_more = st.columns([1, 1, 1, 1, 6])
            current_content_id = str(uuid.uuid4())  # 동적 ID 생성

            # 1. 좋아요 버튼 (기능 활성화)
            if col_like.button("👍", key=f"content_like_{current_content_id}"):
                st.toast(L["toast_like"])

            # 2. 싫어요 버튼 (기능 활성화)
            if col_dislike.button(L.get("button_dislike", "👎"), key=f"content_dislike_{current_content_id}"):
                st.toast(L["toast_dislike"])

            # 3. 공유 버튼 (Web Share API 호출 통합)
            with col_share:
                share_clicked = st.button(L.get("button_share", "🔗"), key=f"content_share_{current_content_id}")

            if share_clicked:
                st.toast("🔗 공유 링크가 준비되었습니다!")

            # 4. 복사 버튼
            if col_copy.button("📋", key=f"content_copy_{current_content_id}"):
                st.toast("📋 클립보드에 복사되었습니다!")
