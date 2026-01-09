# ========================================
# utils/history_handler.py
# 시뮬레이션 이력 관리 모듈 (메인)
# ========================================

"""
시뮬레이션 이력 관리 모듈
로컬 저장, 요약 생성, 가이드 생성 등의 이력 관련 기능을 제공합니다.
"""

# 모듈화된 컴포넌트들 import
from utils.history_handler_load_save import (
    load_simulation_histories_local,
    save_simulation_history_local,
    delete_all_history_local
)
from utils.history_handler_summaries import (
    generate_call_summary,
    generate_chat_summary
)
from utils.history_handler_guides import (
    recommend_guideline_for_customer,
    generate_daily_customer_guide,
    save_daily_customer_guide
)
from utils.history_handler_stats import (
    get_daily_data_statistics
)
from utils.history_handler_exports import (
    export_history_to_word,
    export_history_to_pptx
)
from utils.history_handler_exports_pdf import (
    export_history_to_pdf
)

# 모든 함수를 외부에서 사용할 수 있도록 export
__all__ = [
    'load_simulation_histories_local',
    'generate_call_summary',
    'generate_chat_summary',
    'save_simulation_history_local',
    'delete_all_history_local',
    'get_daily_data_statistics',
    'recommend_guideline_for_customer',
    'generate_daily_customer_guide',
    'save_daily_customer_guide',
    'export_history_to_word',
    'export_history_to_pptx',
    'export_history_to_pdf',
]
