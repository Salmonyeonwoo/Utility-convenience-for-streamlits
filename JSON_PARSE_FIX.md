# JSON 파싱 오류 수정

## 문제
LLM 응답에 마크다운 코드 블록(```json ... ```)이 포함되어 있어 JSON 파싱 오류 발생:
```
Json Parse Error: Unexpected token '`', "```json { "... is not valid JSON
```

## 해결 방법

### 1. `extract_json_from_text` 함수 추가
LLM 응답에서 마크다운 코드 블록을 제거하고 순수 JSON만 추출하는 함수 추가

### 2. 수정된 코드 흐름
```python
# 1. LLM 응답 받기
response = llm.invoke([{"role": "user", "content": quiz_prompt}])

# 2. 응답 텍스트 추출
response_text = response.content if hasattr(response, 'content') else str(response)

# 3. 마크다운 코드 블록 제거 및 JSON 추출
extracted_json = extract_json_from_text(response_text) or response_text

# 4. JSON 파싱 시도
try:
    quiz_data = json.loads(extracted_json)
    st.json(quiz_data)  # 파싱된 JSON 표시
except json.JSONDecodeError as e:
    st.error(f"JSON 파싱 오류: {str(e)}")
    st.code(extracted_json, language="text")  # 디버깅을 위한 원본 텍스트 표시
```

### 3. 프롬프트 개선
프롬프트에 "마크다운 코드 블록 없이 순수 JSON만 출력해주세요"라는 지시사항 추가

## 수정된 파일
- `_pages/_reference_home.py`: 맞춤형 콘텐츠 생성 부분의 "객관식 퀴즈 10문항" 생성 로직

## 테스트 방법
1. 홈 페이지에서 "맞춤형 콘텐츠 생성" 버튼 클릭
2. 콘텐츠 유형: "객관식 퀴즈 10문항" 선택
3. 주제 입력 후 "생성" 버튼 클릭
4. JSON이 올바르게 파싱되어 표시되는지 확인


