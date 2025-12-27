# 모든 탭 코드를 제거하고 함수 호출로 대체하는 스크립트
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 각 탭의 시작과 끝 패턴 찾기
import re

# 전화 시뮬레이터 탭 제거
phone_pattern = r'elif feature_selection == L\["sim_tab_phone"\]:.*?st\.header\(L\["phone_header"\]\).*?(?=elif feature_selection == L\["rag_tab"\]|$)'
phone_match = re.search(phone_pattern, content, re.DOTALL)
if phone_match:
    start = phone_match.start()
    # 다음 탭 시작 전까지 찾기
    next_tab = content.find('elif feature_selection == L["rag_tab"]:', start)
    if next_tab != -1:
        # 전화 시뮬레이터 탭 전체 제거하고 함수 호출로 대체
        replacement = '''elif feature_selection == L["sim_tab_phone"]:
    if PHONE_SIMULATOR_AVAILABLE:
        render_phone_simulator()
    else:
        st.error("전화 시뮬레이터 탭 모듈을 찾을 수 없습니다.")

'''
        content = content[:start] + replacement + content[next_tab:]
        print("OK: Replaced phone simulator tab")

# RAG 탭 제거
rag_pattern = r'elif feature_selection == L\["rag_tab"\]:.*?st\.header\(L\["rag_header"\]\).*?(?=elif feature_selection == L\["content_tab"\]|$)'
rag_match = re.search(rag_pattern, content, re.DOTALL)
if rag_match:
    start = rag_match.start()
    next_tab = content.find('elif feature_selection == L["content_tab"]:', start)
    if next_tab != -1:
        replacement = '''elif feature_selection == L["rag_tab"]:
    if RAG_AVAILABLE:
        render_rag()
    else:
        st.error("RAG 탭 모듈을 찾을 수 없습니다.")

'''
        content = content[:start] + replacement + content[next_tab:]
        print("OK: Replaced RAG tab")

# 콘텐츠 탭 제거
content_pattern = r'elif feature_selection == L\["content_tab"\]:.*?st\.header\(L\["content_header"\]\).*?(?=elif feature_selection == L\["lstm_tab"\]|$)'
content_match = re.search(content_pattern, content, re.DOTALL)
if content_match:
    start = content_match.start()
    next_tab = content.find('elif feature_selection == L["lstm_tab"]:', start)
    if next_tab == -1:
        next_tab = content.find('elif feature_selection == L["voice_rec_header"]:', start)
    if next_tab != -1:
        replacement = '''elif feature_selection == L["content_tab"]:
    if CONTENT_AVAILABLE:
        render_content()
    else:
        st.error("콘텐츠 탭 모듈을 찾을 수 없습니다.")

'''
        content = content[:start] + replacement + content[next_tab:]
        print("OK: Replaced content tab")

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("All tabs replaced successfully!")











