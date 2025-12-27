"""Import 테스트 스크립트"""
import sys
import os

# 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Import 테스트 시작")
print("=" * 60)

# 1. _chat_simulator 테스트
try:
    from _pages._chat_simulator import render_chat_simulator
    print("[OK] _pages._chat_simulator import 성공")
except Exception as e:
    print(f"[FAIL] _pages._chat_simulator import 실패: {e}")

# 2. _content 테스트
try:
    from _pages._content import render_content
    print("[OK] _pages._content import 성공")
except Exception as e:
    print(f"[FAIL] _pages._content import 실패: {e}")

# 3. _app_chat_page 테스트
try:
    from _pages._app_chat_page import render_chat_page
    print("[OK] _pages._app_chat_page import 성공")
except Exception as e:
    print(f"[FAIL] _pages._app_chat_page import 실패: {e}")

print("=" * 60)


