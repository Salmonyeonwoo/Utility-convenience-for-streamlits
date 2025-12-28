# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2025)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
고객 검증 관련 모듈
고객 정보 검증, 로그인 관련 문의 확인 등의 기능을 제공합니다.
"""

from typing import Dict, List, Any, Tuple
import re


def mask_email(email: str, show_chars: int = 2) -> str:
    """이메일 주소를 마스킹합니다."""
    if not email or "@" not in email:
        return email
    
    local_part, domain = email.split("@", 1)
    domain_parts = domain.split(".", 1)
    
    if len(local_part) <= show_chars:
        masked_local = local_part
    else:
        masked_local = local_part[:show_chars] + "*" * (len(local_part) - show_chars)
    
    if len(domain_parts[0]) <= show_chars:
        masked_domain = domain_parts[0]
    else:
        masked_domain = domain_parts[0][:show_chars] + "*" * (len(domain_parts[0]) - show_chars)
    
    if len(domain_parts) > 1:
        return f"{masked_local}@{masked_domain}.{domain_parts[1]}"
    else:
        return f"{masked_local}@{masked_domain}"


def verify_customer_info(provided_info: Dict[str, str], stored_info: Dict[str, str]) -> Tuple[bool, Dict[str, bool]]:
    """고객이 제공한 정보와 저장된 정보를 비교하여 검증합니다."""
    verification_results = {
        "receipt_number": False,
        "payment_info": False,
        "customer_name": False,
        "customer_email": False,
        "customer_phone": False,
        "file_uploaded": False
    }
    
    if provided_info.get("file_uploaded"):
        verification_results["file_uploaded"] = True
    
    if provided_info.get("receipt_number") and stored_info.get("receipt_number"):
        provided_receipt = provided_info["receipt_number"].strip().upper().replace(" ", "")
        stored_receipt = stored_info["receipt_number"].strip().upper().replace(" ", "")
        verification_results["receipt_number"] = provided_receipt == stored_receipt
    
    payment_method = provided_info.get("payment_method", "")
    
    if "카드" in payment_method or "card" in payment_method.lower():
        if provided_info.get("card_last4") and stored_info.get("card_last4"):
            provided_card = "".join(filter(str.isdigit, provided_info["card_last4"]))[-4:]
            stored_card = "".join(filter(str.isdigit, stored_info["card_last4"]))[-4:]
            verification_results["payment_info"] = provided_card == stored_card and len(provided_card) == 4
    elif "온라인뱅킹" in payment_method or "online banking" in payment_method.lower():
        if provided_info.get("account_number") and stored_info.get("account_number"):
            provided_account = "".join(filter(str.isdigit, provided_info["account_number"]))
            stored_account = "".join(filter(str.isdigit, stored_info["account_number"]))
            verification_results["payment_info"] = provided_account[-6:] == stored_account[-6:] or provided_account[-4:] == stored_account[-4:]
    elif payment_method and stored_info.get("payment_method"):
        provided_payment = payment_method.strip().lower()
        stored_payment = stored_info["payment_method"].strip().lower()
        verification_results["payment_info"] = provided_payment == stored_payment
    
    if provided_info.get("customer_name") and stored_info.get("customer_name"):
        provided_name = " ".join(provided_info["customer_name"].strip().split()).upper()
        stored_name = " ".join(stored_info["customer_name"].strip().split()).upper()
        verification_results["customer_name"] = provided_name == stored_name
    
    if provided_info.get("customer_email") and stored_info.get("customer_email"):
        provided_email = provided_info["customer_email"].strip().lower()
        stored_email = stored_info["customer_email"].strip().lower()
        verification_results["customer_email"] = provided_email == stored_email
    
    if provided_info.get("customer_phone") and stored_info.get("customer_phone"):
        provided_phone = "".join(filter(str.isdigit, provided_info["customer_phone"]))
        stored_phone = "".join(filter(str.isdigit, stored_info["customer_phone"]))
        verification_results["customer_phone"] = provided_phone[-10:] == stored_phone[-10:] or provided_phone[-4:] == stored_phone[-4:]
    
    matched_count = sum(verification_results.values())
    if verification_results["file_uploaded"]:
        matched_count += 1
    is_verified = matched_count >= 3
    
    return is_verified, verification_results


def check_if_customer_provided_verification_info(messages: List[Dict[str, Any]]) -> bool:
    """고객이 검증 정보를 제공했는지 확인합니다."""
    if not messages:
        return False
    
    recent_customer_messages = []
    for msg in messages[-10:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ["customer", "customer_rebuttal", "initial_query"] or "customer" in role.lower():
            recent_customer_messages.append(content)
    
    if not recent_customer_messages:
        return False
    
    combined_text_original = " ".join(recent_customer_messages)
    combined_text = combined_text_original.lower()
    
    has_numbers = bool(re.search(r'\d{4,}', combined_text))
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', combined_text_original))
    has_phone = bool(re.search(r'010[-.\s]?\d{3,4}[-.\s]?\d{4}', combined_text_original) or 
                     re.search(r'010\d{8}', combined_text_original))
    
    info_count = 0
    
    if (re.search(r'예약\s*번호', combined_text) or 
        re.search(r'영수증\s*번호', combined_text) or
        re.search(r'예약.*[:：]\s*\d{4,}', combined_text_original) or
        re.search(r'영수증.*[:：]\s*\d{4,}', combined_text_original) or
        re.search(r'예약번호.*[:：]\s*\d{4,}', combined_text_original) or
        re.search(r'예약.*\d{4,}', combined_text) or
        re.search(r'영수증.*\d{4,}', combined_text) or
        re.search(r'booking.*number', combined_text) or
        re.search(r'receipt.*number', combined_text) or
        re.search(r'\d{5,}', combined_text_original)):
        info_count += 1
    
    payment_keywords = [
        "카드", "card", "visa", "master", "amex", "american express",
        "신용카드", "체크카드", "credit card", "debit card",
        "카카오페이", "kakao", "kakaopay",
        "네이버페이", "naver", "naverpay",
        "온라인뱅킹", "online banking", "online",
        "grabpay", "grab pay", "grab",
        "touch n go", "touch n' go", "tng",
        "결제 수단", "payment method", "payment", "결제하", "결제 내역"
    ]
    if (any(kw in combined_text for kw in payment_keywords) or
        re.search(r'결제\s*수단\s*[:：]', combined_text_original)):
        info_count += 1
    
    name_keywords = ["성함", "이름", "name", "제 이름", "고객님의 성함", "my name", "고객님의 이름"]
    korean_name_pattern = (
        re.search(r'성함\s*[:：]\s*[가-힣]{2,4}', combined_text_original) or
        re.search(r'이름\s*[:：]\s*[가-힣]{2,4}', combined_text_original) or
        re.search(r'고객님의\s*성함\s*[:：]\s*[가-힣]{2,4}', combined_text_original) or
        re.search(r'[가-힣]{2,4}', combined_text_original)
    )
    english_name_pattern = re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', combined_text_original)
    
    if (any(kw in combined_text for kw in name_keywords) or 
        korean_name_pattern or 
        english_name_pattern):
        info_count += 1
    
    if has_email:
        info_count += 1
    
    if (has_phone or 
        re.search(r'연락처', combined_text) or 
        re.search(r'전화번호', combined_text) or
        re.search(r'phone', combined_text)):
        info_count += 1
    
    return info_count >= 2 or (has_email and info_count >= 1) or (has_phone and info_count >= 1)


def check_if_login_related_inquiry(customer_query: str) -> bool:
    """고객 문의가 로그인/계정 관련인지 확인합니다."""
    if not customer_query or not customer_query.strip():
        return False
    
    login_keywords = [
        "로그인", "login", "ログイン",
        "계정", "account", "アカウント",
        "비밀번호", "password", "パスワード",
        "아이디", "id", "ID", "ユーザーID",
        "접속", "access", "アクセス",
        "인증", "authentication", "認証",
        "로그인 안됨", "cannot login", "ログインできない",
        "로그인 오류", "login error", "ログインエラー",
        "로그인 안", "로그인 실패", "로그인 문제",
        "계정 문제", "계정 잠금", "계정 오류",
        "account problem", "account error", "account locked",
        "password reset", "비밀번호 재설정", "パスワードリセット",
        "forgot password", "비밀번호 분실", "パスワード忘れ"
    ]
    
    query_lower = customer_query.lower()
    for keyword in login_keywords:
        if keyword.lower() in query_lower:
            return True
    
    return False









