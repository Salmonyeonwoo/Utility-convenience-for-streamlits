# ========================================
# utils/i18n.py
# 다국어 설정 모듈 (메인 - Export 역할)
# ========================================

from typing import Dict

# 언어별 번역 모듈에서 import
from utils.i18n_lang_ko import LANG_KO
from utils.i18n_lang_en import LANG_EN
from utils.i18n_lang_ja import LANG_JA

# 기본 언어 설정
DEFAULT_LANG = "ko"

# 모든 언어를 하나의 딕셔너리로 결합
LANG: Dict[str, Dict[str, str]] = {
    "ko": LANG_KO,
    "en": LANG_EN,
    "ja": LANG_JA
}

# Export
__all__ = ['LANG', 'DEFAULT_LANG']
