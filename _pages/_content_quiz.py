# ========================================
# _pages/_content_quiz.py
# 퀴즈 생성 및 진행 로직
# ========================================

import streamlit as st
import json
import uuid
from lang_pack import LANG
from llm_client import get_api_key
from openai import OpenAI
from simulation_handler import run_llm
from _pages._content_utils import extract_json_from_text

# Plotly 시각화
try:
    import plotly.graph_objects as go
    import plotly.express as px
    IS_PLOTLY_AVAILABLE = True
except ImportError:
    IS_PLOTLY_AVAILABLE = False


def generate_quiz_prompt(topic, level, lang_instruction):
    """퀴즈 생성 프롬프트 생성"""
    return (
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
        f"}}")


def generate_quiz(topic, level, lang_instruction, L):
    """퀴즈 생성"""
    quiz_prompt = generate_quiz_prompt(topic, level, lang_instruction)
    generated_json_text = None
    raw_response_text = None
    llm_attempts = []

    # 1순위: OpenAI (JSON mode가 가장 안정적)
    if get_api_key("openai"):
        llm_attempts.append(("openai", get_api_key("openai"), "gpt-4o"))
    # 2순위: Gemini (Fallback)
    if get_api_key("gemini"):
        llm_attempts.append(("gemini", get_api_key("gemini"), "gemini-2.5-flash"))

    with st.spinner(L.get("response_generating", "생성 중...")):
        for provider, api_key, model_name in llm_attempts:
            try:
                if provider == "openai":
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": quiz_prompt}],
                        response_format={"type": "json_object"},
                    )
                    raw_response_text = response.choices[0].message.content.strip()
                    generated_json_text = extract_json_from_text(raw_response_text) or raw_response_text
                    break

                elif provider == "gemini":
                    raw_response_text = run_llm(quiz_prompt)
                    generated_json_text = extract_json_from_text(raw_response_text)
                    if generated_json_text:
                        break

            except Exception as e:
                print(f"JSON generation failed with {provider}: {e}")
                continue

    return generated_json_text, raw_response_text


def parse_quiz_json(generated_json_text, raw_response_text, L):
    """퀴즈 JSON 파싱 및 검증"""
    parsed_obj = None
    quiz_data = None

    if generated_json_text:
        try:
            parsed_obj = json.loads(generated_json_text)
            quiz_data = parsed_obj.get("quiz_questions")

            if not isinstance(quiz_data, list) or len(quiz_data) < 1:
                raise ValueError("Missing 'quiz_questions' key or empty array.")

            # 데이터 유효성 검사
            for i, q in enumerate(quiz_data):
                if not isinstance(q, dict):
                    raise ValueError(f"Question {i+1} is not a valid object.")
                if "question" not in q or "options" not in q or "answer" not in q:
                    raise ValueError(f"Question {i+1} is missing required fields.")
                if not isinstance(q["options"], list) or len(q["options"]) != 4:
                    raise ValueError(f"Question {i+1} must have exactly 4 options.")
                if not isinstance(q["answer"], int) or q["answer"] < 1 or q["answer"] > 4:
                    raise ValueError(f"Question {i+1} answer must be between 1 and 4.")

            # 퀴즈 상태 저장
            st.session_state.quiz_data = quiz_data
            st.session_state.current_question_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = [1] * len(quiz_data)
            st.session_state.show_explanation = False
            st.session_state.is_quiz_active = True
            st.session_state.quiz_type_key = str(uuid.uuid4())

            return quiz_data, None

        except json.JSONDecodeError as e:
            return None, ("JSON 파싱 오류", str(e))
        except ValueError as e:
            return None, ("데이터 구조 오류", str(e))

    return None, ("JSON 추출 실패", "LLM 응답에서 JSON 객체를 찾을 수 없습니다.")


def render_quiz_completion(quiz_data, L):
    """퀴즈 완료 화면 렌더링"""
    total_questions = len(quiz_data)
    score = st.session_state.quiz_score
    incorrect_count = total_questions - score
    
    st.success(L.get("quiz_complete", "퀴즈 완료!"))
    st.subheader(f"{L.get('score', '점수')}: {score} / {total_questions} ({(score / total_questions) * 100:.1f}%)")

    if IS_PLOTLY_AVAILABLE:
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=[L.get("correct_questions", "맞은 문제"), L.get("incorrect_questions", "틀린 문제")],
                values=[score, incorrect_count],
                hole=0.4,
                marker_colors=['#28a745', '#dc3545'],
                textinfo='label+percent',
                textposition='outside'
            )])
            fig.update_layout(
                title=L.get("question_result", "문제 결과"),
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            _render_quiz_results(quiz_data, L)
    else:
        st.markdown(f"**{L.get('correct_questions', '맞은 문제')}:** {score}개")
        st.markdown(f"**{L.get('incorrect_questions', '틀린 문제')}:** {incorrect_count}개")
        _render_quiz_results(quiz_data, L)

    if st.button(L.get("retake_quiz", "다시 풀기"), key="retake_quiz_btn"):
        st.session_state.current_question_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_answers = [1] * len(quiz_data)
        st.session_state.show_explanation = False


def _render_quiz_results(quiz_data, L):
    """퀴즈 결과 목록 렌더링"""
    st.markdown("### " + L.get("question_result", "문제 결과"))
    for i, question_item in enumerate(quiz_data):
        user_answer = st.session_state.quiz_answers[i] if i < len(st.session_state.quiz_answers) else None
        is_correct = user_answer == 'Correctly Scored'
        correct_answer_idx = question_item.get('answer', 1)
        correct_answer_text = question_item['options'][correct_answer_idx - 1] if 0 < correct_answer_idx <= len(question_item['options']) else "N/A"

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

        if IS_PLOTLY_AVAILABLE:
            html_content = (
                f"<div style=\"border-left: 4px solid {status_color}; padding-left: 10px; margin-bottom: 15px;\">\n"
                f"    <strong>{status_icon} 문항 {i+1}:</strong> {question_item['question']}<br>\n"
                f"    <span style=\"color: {status_color};\">{L.get('your_answer', '내 답안')}: {user_answer_text}</span><br>\n"
                f"    <span style=\"color: green;\">{L.get('correct_answer_label', '정답')}: {correct_answer_text}</span>\n"
                f"</div>")
            st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.markdown(f"**{status_icon} 문항 {i+1}:** {question_item['question']}")
            st.markdown(f"- {L.get('your_answer', '내 답안')}: {user_answer_text}")
            st.markdown(f"- {L.get('correct_answer_label', '정답')}: {correct_answer_text}")
            st.markdown("---")


def render_quiz_question(quiz_data, idx, L):
    """퀴즈 문제 렌더링"""
    question_data = quiz_data[idx]
    st.subheader(f"{L.get('question_label', '문항')} {idx + 1}/{len(quiz_data)}")
    st.markdown(f"**{question_data['question']}**")

    options = question_data['options']
    current_answer = st.session_state.quiz_answers[idx]

    if current_answer is None or not isinstance(current_answer, int) or current_answer <= 0:
        radio_index = 0
    else:
        radio_index = min(current_answer - 1, len(options) - 1)

    selected_option = st.radio(
        L.get("select_answer", "답안 선택"),
        options,
        index=radio_index,
        key=f"quiz_radio_{st.session_state.quiz_type_key}_{idx}"
    )

    selected_option_index = options.index(selected_option) + 1 if selected_option in options else None

    check_col, next_col = st.columns([1, 1])

    if check_col.button(L.get("check_answer", "답안 확인"), key=f"check_answer_btn_{idx}"):
        if selected_option_index is None:
            st.warning("선택지를 선택해 주세요.")
        else:
            if st.session_state.quiz_answers[idx] != 'Correctly Scored':
                correct_answer = question_data.get('answer')
                if selected_option_index == correct_answer:
                    st.session_state.quiz_score += 1
                    st.session_state.quiz_answers[idx] = 'Correctly Scored'
                    st.success(L.get("correct_answer", "정답입니다!"))
                else:
                    st.session_state.quiz_answers[idx] = selected_option_index
                    st.error(L.get("incorrect_answer", "틀렸습니다."))

            st.session_state.show_explanation = True

    if st.session_state.show_explanation:
        correct_index = question_data.get('answer', 1)
        correct_answer_text = question_data['options'][correct_index - 1] if 0 < correct_index <= len(question_data['options']) else "N/A"

        st.markdown("---")
        st.markdown(f"**{L.get('correct_is', '정답')}:** {correct_answer_text}")
        with st.expander(f"**{L.get('explanation', '해설')}**", expanded=True):
            st.info(question_data.get('explanation', '해설이 제공되지 않았습니다.'))

        if next_col.button(L.get("next_question", "다음 문항"), key=f"next_question_btn_{idx}"):
            st.session_state.current_question_index += 1
            st.session_state.show_explanation = False
    else:
        if st.session_state.quiz_answers[idx] == 'Correctly Scored' or (
            isinstance(st.session_state.quiz_answers[idx], int) and st.session_state.quiz_answers[idx] > 0):
            if next_col.button(L.get("next_question", "다음 문항"), key=f"next_question_btn_after_check_{idx}"):
                st.session_state.current_question_index += 1
                st.session_state.show_explanation = False
