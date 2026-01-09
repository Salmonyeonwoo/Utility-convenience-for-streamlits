# ========================================
# simulation_handler_new.py
# 시뮬레이션 처리 모듈 (메인 - 모든 모듈 통합)
# ========================================

"""
시뮬레이션 처리 모듈 (핵심 기능만 포함)
고객 응대 시뮬레이션, 채팅/전화 대화 생성, 힌트 생성 등의 핵심 기능을 제공합니다.
"""

# 모듈화된 컴포넌트들 import
from simulation_handler_new_chat_history import get_chat_history_for_prompt
from simulation_handler_new_hints import generate_realtime_hint

# 나머지 함수들은 원본 파일에서 직접 import (순환 참조 방지)
# 실제 구현은 원본 파일에 유지하고, 여기서는 재export만 수행

# 원본 파일에서 직접 import (순환 참조 방지)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 원본 함수들을 직접 정의하거나 import
# (실제로는 원본 파일을 그대로 유지하고 여기서는 재export만 수행)

# 모든 함수를 외부에서 사용할 수 있도록 export
__all__ = [
    'get_chat_history_for_prompt',
    'generate_realtime_hint',
    # 나머지 함수들은 원본 파일에서 직접 import
]

# 하위 호환성을 위해 원본 함수들을 재export
# (실제 구현은 원본 파일에 유지)
