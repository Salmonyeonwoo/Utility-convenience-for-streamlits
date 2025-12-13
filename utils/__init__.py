# utils 패키지 초기화 파일
"""
유틸리티 함수 모듈
JSON 파일 로드/저장, 파일 처리 등의 유틸리티 함수를 제공합니다.
"""

import os
import json
from typing import Any

from config import FAQ_DB_FILE, PRODUCT_IMAGE_CACHE_FILE


# ----------------------------------------
# JSON Helper
# ----------------------------------------
def _load_json(path: str, default: Any):
    """JSON 파일을 로드합니다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_json(path: str, data: Any):
    """JSON 파일을 저장합니다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
