# ========================================
# video_handler.py
# 비디오 처리 모듈 (메인 - Export 역할)
# ========================================

# 모든 함수를 별도 모듈에서 import하여 하위 호환성 유지
from video_handler_analysis import analyze_text_for_video_selection
from video_handler_path import get_video_path_by_avatar
from video_handler_database import (
    load_video_mapping_database,
    save_video_mapping_database,
    add_video_mapping_feedback,
    get_recommended_video_from_database
)
from video_handler_rendering import render_synchronized_video
from video_handler_virtual import generate_virtual_human_video, get_virtual_human_config

# 역할별 TTS 음성 스타일 설정
TTS_VOICES = {
    "customer_male": {
        "gender": "male",
        "voice": "alloy"
    },
    "customer_female": {
        "gender": "female",
        "voice": "nova"
    },
    "customer": {
        "gender": "male",
        "voice": "alloy"
    },
    "agent": {
        "gender": "female",
        "voice": "shimmer"
    },
    "supervisor": {
        "gender": "female",
        "voice": "nova"
    }
}

# 모든 함수를 export
__all__ = [
    'analyze_text_for_video_selection',
    'get_video_path_by_avatar',
    'load_video_mapping_database',
    'save_video_mapping_database',
    'add_video_mapping_feedback',
    'get_recommended_video_from_database',
    'render_synchronized_video',
    'generate_virtual_human_video',
    'get_virtual_human_config',
    'TTS_VOICES'
]
