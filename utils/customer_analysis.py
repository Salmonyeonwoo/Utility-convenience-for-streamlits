# ========================================
# utils/customer_analysis.py
# 고객 분석 모듈 (메인 - Export 역할)
# ========================================

# 모든 함수를 별도 모듈에서 import하여 하위 호환성 유지
from utils.customer_analysis_language import detect_text_language
from utils.customer_analysis_profile import analyze_customer_profile
from utils.customer_analysis_similarity import find_similar_cases
from utils.customer_analysis_visualization import (
    visualize_customer_profile,
    visualize_similarity_cases,
    visualize_case_trends,
    visualize_customer_characteristics
)
from utils.customer_analysis_guidelines import generate_guideline_from_past_cases
from utils.customer_analysis_advice import generate_initial_advice

# 하위 호환성을 위한 별칭
_generate_initial_advice = generate_initial_advice

# 모든 함수를 export
__all__ = [
    'detect_text_language',
    'analyze_customer_profile',
    'find_similar_cases',
    'visualize_customer_profile',
    'visualize_similarity_cases',
    'visualize_case_trends',
    'visualize_customer_characteristics',
    'generate_guideline_from_past_cases',
    'generate_initial_advice'
]
