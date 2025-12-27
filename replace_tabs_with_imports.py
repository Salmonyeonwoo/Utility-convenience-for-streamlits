# 각 탭 코드를 함수 호출로 대체하는 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 각 탭의 시작 라인 찾기
phone_start = None
rag_start = None
content_start = None
lstm_start = None
voice_start = None

for i, line in enumerate(lines):
    if 'elif feature_selection == L["sim_tab_phone"]:' in line and 'st.header' in line:
        phone_start = i
    elif 'elif feature_selection == L["rag_tab"]:' in line and 'st.header' in line:
        rag_start = i
    elif 'elif feature_selection == L["content_tab"]:' in line and 'st.header' in line:
        content_start = i
    elif 'elif feature_selection == L["lstm_tab"]:' in line and 'st.header' in line:
        lstm_start = i
    elif 'elif feature_selection == L["voice_rec_header"]:' in line and 'st.header' in line:
        voice_start = i

# 각 탭의 끝 라인 찾기
def find_end(start_idx, next_start_idx):
    if start_idx is None:
        return None
    if next_start_idx is None:
        # 마지막 탭인 경우, 파일 끝까지
        return len(lines)
    return next_start_idx

phone_end = find_end(phone_start, rag_start)
rag_end = find_end(rag_start, content_start)
content_end = find_end(content_start, lstm_start)
lstm_end = find_end(lstm_start, voice_start)
voice_end = find_end(voice_start, None)

# 각 탭을 함수 호출로 대체
new_lines = lines[:]

# 전화 시뮬레이터 탭 대체
if phone_start is not None and phone_end is not None:
    replacement = ['elif feature_selection == L["sim_tab_phone"]:\n',
                   '    if PHONE_SIMULATOR_AVAILABLE:\n',
                   '        render_phone_simulator()\n',
                   '    else:\n',
                   '        st.error("전화 시뮬레이터 탭 모듈을 찾을 수 없습니다.")\n',
                   '\n']
    new_lines = new_lines[:phone_start] + replacement + new_lines[phone_end:]

# RAG 탭 대체
if rag_start is not None and rag_end is not None:
    # phone_end가 변경되었으므로 다시 찾기
    for i, line in enumerate(new_lines):
        if 'elif feature_selection == L["rag_tab"]:' in line and 'st.header' in line:
            rag_start = i
            break
    # 다음 탭 찾기
    for i in range(rag_start + 1, len(new_lines)):
        if 'elif feature_selection == L["content_tab"]:' in line and 'st.header' in line:
            rag_end = i
            break
    if rag_end is None:
        rag_end = len(new_lines)
    
    replacement = ['elif feature_selection == L["rag_tab"]:\n',
                   '    if RAG_AVAILABLE:\n',
                   '        render_rag()\n',
                   '    else:\n',
                   '        st.error("RAG 탭 모듈을 찾을 수 없습니다.")\n',
                   '\n']
    new_lines = new_lines[:rag_start] + replacement + new_lines[rag_end:]

# 콘텐츠 탭 대체
if content_start is not None and content_end is not None:
    # rag_end가 변경되었으므로 다시 찾기
    for i, line in enumerate(new_lines):
        if 'elif feature_selection == L["content_tab"]:' in line and 'st.header' in line:
            content_start = i
            break
    # 다음 탭 찾기
    for i in range(content_start + 1, len(new_lines)):
        if 'elif feature_selection == L["lstm_tab"]:' in line and 'st.header' in line:
            content_end = i
            break
    if content_end is None:
        content_end = len(new_lines)
    
    replacement = ['elif feature_selection == L["content_tab"]:\n',
                   '    if CONTENT_AVAILABLE:\n',
                   '        render_content()\n',
                   '    else:\n',
                   '        st.error("콘텐츠 탭 모듈을 찾을 수 없습니다.")\n',
                   '\n']
    new_lines = new_lines[:content_start] + replacement + new_lines[content_end:]

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("OK: Replaced all tabs with function calls")











