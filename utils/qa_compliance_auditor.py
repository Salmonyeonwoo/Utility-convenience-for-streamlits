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
QA(상담 품질 평가) & 컴플라이언스(규정 준수) 자동 감사 엔진 (Enterprise Audit System)
- 다국어(한국어, 영어, 일본어) 완전 지원
- 다차원 상담 품질 평가 매트릭스 (공감도, 문제해결력, 절차준수도)
- 금지어 및 고위험 발언(컴플레인 유발, 구두 확약) 실시간 탐지
- 필수 고지 항목(본인확인, 규정안내, 추가문의 확인, 종료인사) 4단계 체크리스트
- 종합 QA 등급(S, A, B, C, F) 산출 및 상담원 맞춤형 AI 코칭 피드백 생성
"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime


# 1. 금지어 및 고위험 발언 사전 (다국어 지원)
PROHIBITED_RULES = [
    {
        "pattern": r"(규정상\s*(절대\s*)?(안\s*됩니다|불가합니다|안\s*돼요)|법적으로\s*안\s*됩니다|absolutely not allowed|strictly prohibited by policy|規定上(絶対)?(できません|不可です))",
        "label": {
            "ko": "감정 대립형 단정적 거절",
            "en": "Abrupt Policy Refusal",
            "ja": "感情対立型の断定的拒絶"
        },
        "severity": "HIGH",
        "penalty": 15,
        "advice": {
            "ko": "규정만 앞세우기보다 고객 상황에 공감 후 대안이나 예외 증빙 절차를 먼저 안내하세요.",
            "en": "Instead of abruptly citing policy, empathize first and guide alternative or exception procedures.",
            "ja": "規定のみを前面に出さず、顧客の状況に共感した上で代替案や例外手続きを先に案内してください。"
        }
    },
    {
        "pattern": r"(소리\s*지르지\s*마세요|화내지\s*마세요|진정하세요|stop shouting|don't get angry|calm down|大声を(出さないで|上げないで)|怒らないで|落ち着いて)",
        "label": {
            "ko": "고객 감정 자극 발언",
            "en": "Customer Emotion Provocation",
            "ja": "顧客感情の刺激発言"
        },
        "severity": "CRITICAL",
        "penalty": 25,
        "advice": {
            "ko": "고객의 감정적 반응에 직접 반박하지 말고 경청과 유감 표명으로 온도를 낮추세요.",
            "en": "Do not directly rebuke customer frustration; de-escalate with active listening and sincere empathy.",
            "ja": "顧客の感情的な反応に直接反論せず、傾聴と遺憾の意を示すことで温度を下げてください。"
        }
    },
    {
        "pattern": r"(제가\s*알\s*바\s*아닙니다|제\s*소관이\s*아닙니다|저한테\s*따지지\s*마세요|not my business|not my department|don't blame me|私の知ったことではありません|私の管轄ではありません|私に文句を言わないで)",
        "label": {
            "ko": "책임 회피 및 불성실 응대",
            "en": "Responsibility Evasion",
            "ja": "責任回避・不誠実な応対"
        },
        "severity": "CRITICAL",
        "penalty": 30,
        "advice": {
            "ko": "타 부서 업무이더라도 담당 부서로의 안전한 에스컬레이션 절차를 정중히 안내해야 합니다.",
            "en": "Even if it is outside your scope, politely guide the customer through the proper escalation process.",
            "ja": "他部署の業務であっても、担当部署への安全なエスカレーション手順を丁寧に案内してください。"
        }
    },
    {
        "pattern": r"(고객님(의)?\s*잘못(입니다|이에요)|고객님\s*탓입니다|주의하셨어야죠|your fault|your mistake|you should have been careful|お客様の過失です|お客様のせいです|注意すべきでした)",
        "label": {
            "ko": "고객 귀책 전가 발언",
            "en": "Blaming the Customer",
            "ja": "顧客への責任転嫁発言"
        },
        "severity": "CRITICAL",
        "penalty": 25,
        "advice": {
            "ko": "고객의 과실을 직접 탓하지 말고 객관적 이용 약관과 사실 관계를 완곡히 설명하세요.",
            "en": "Do not blame the customer directly; explain terms and facts gently and objectively.",
            "ja": "顧客の過失を直接責めず、客観的な利用規約と事実関係を穏やかに説明してください。"
        }
    },
    {
        "pattern": r"(무조건\s*(100%|다|전액)?\s*(환불|처리|보상)\s*해\s*드릴게요|제가\s*개인\s*돈으로|장담합니다|unconditionally refund|pay from my own pocket|I guarantee 100%|無条件で(全額)?返金します|私の自腹で|保証します)",
        "label": {
            "ko": "임의 구두 확약 (컴플라이언스 위반)",
            "en": "Unauthorized Oral Promise (Compliance Breach)",
            "ja": "無断口頭確約 (コンプライアンス違反)"
        },
        "severity": "CRITICAL",
        "penalty": 30,
        "advice": {
            "ko": "결재권자 승인 없는 임의 구두 약속은 금지되며, 규정된 에스컬레이션 심사 절차를 안내해야 합니다.",
            "en": "Unauthorized oral commitments are prohibited; guide the customer through the formal approval procedure.",
            "ja": "決裁者の承認のない無断口頭確約は禁止されています。規定のエスカレーション審査手順を案内してください。"
        }
    },
    {
        "pattern": r"(다른\s*데(가서)?\s*알아보세요|인터넷\s*찾아보세요|포털에\s*검색해보세요|look somewhere else|search on google|search the internet|他で(調べて|探して)ください|ネットで検索してください)",
        "label": {
            "ko": "상담 포기 및 무단 전가",
            "en": "Consultation Abandonment",
            "ja": "応対放棄・無断転嫁"
        },
        "severity": "HIGH",
        "penalty": 20,
        "advice": {
            "ko": "고객이 필요한 정보나 링크, 고객센터 경로를 상담원이 직접 확인하여 제공해야 합니다.",
            "en": "The agent must directly check and provide necessary links or customer support paths.",
            "ja": "顧客に必要な情報やリンク、窓口への経路をエージェントが直接確認して提供してください。"
        }
    }
]


# 2. 필수 고지 항목 정의 (다국어 지원)
MANDATORY_ITEMS = [
    {
        "id": "CHECK_ID_VERIFY",
        "name": {
            "ko": "1. 본인 및 이용 정보 확인",
            "en": "1. Identity & Inquiry Verification",
            "ja": "1. 本人確認および照会情報の確認"
        },
        "keywords": ["예약번호", "성함", "연락처", "구매번호", "주문번호", "기종", "차량번호", "확인해", "조회해", "booking", "reservation", "order", "name", "phone", "verify", "check", "予約番号", "お名前", "ご連絡先", "確認", "照会"],
        "weight": 25,
        "fail_feedback": {
            "ko": "상담 시작 시 고객의 예약번호 또는 본인 확인 절차가 누락되었습니다.",
            "en": "Customer identity or booking verification was missing at the start of consultation.",
            "ja": "相談開始時の顧客の予約番号または本人確認手順が抜けています。"
        }
    },
    {
        "id": "CHECK_POLICY_EXPLAIN",
        "name": {
            "ko": "2. 공식 규정 및 소요 기한 고지",
            "en": "2. Official Policy & Timeline Disclosure",
            "ja": "2. 公式規定および所要期間の告知"
        },
        "keywords": ["규정", "영업일", "환불", "취소", "보증", "수수료", "설정", "안내", "절차", "기한", "policy", "business day", "refund", "cancel", "warranty", "fee", "procedure", "timeline", "規定", "営業日", "返金", "キャンセル", "保証", "案内", "手続き"],
        "weight": 25,
        "fail_feedback": {
            "ko": "취소/환불/A/S 처리 시 공식 소요 기한(영업일 3~5일 등)이나 규정 고지가 미흡했습니다.",
            "en": "Official policy or processing timeline disclosure (e.g., 3-5 business days) was missing.",
            "ja": "取消・返金・サポートの所要期間（営業日3〜5日など）や規定の案内が不十分でした。"
        }
    },
    {
        "id": "CHECK_ADDITIONAL_INQUIRY",
        "name": {
            "ko": "3. 추가 문의 사항 확인",
            "en": "3. Additional Inquiries Confirmation",
            "ja": "3. 追加質問の有無確認"
        },
        "keywords": ["다른 문의", "추가 문의", "더 궁금하신", "도움 드릴", "문의 사항 있으신가요", "additional question", "further question", "anything else", "help you with", "他にご質問", "追加の質問", "他にお困り"],
        "weight": 25,
        "fail_feedback": {
            "ko": "솔루션 제공 후 고객에게 '다른 문의 사항 있으신가요?' 추가 문의 확인이 누락되었습니다.",
            "en": "Confirmation of additional questions ('Do you have any other questions?') was missing.",
            "ja": "解決後の「他にご質問はございますか？」という追加質問の確認が抜けています。"
        }
    },
    {
        "id": "CHECK_POLITE_CLOSING",
        "name": {
            "ko": "4. 정중한 종료 및 감사 인사",
            "en": "4. Polite Closing & Gratitude",
            "ja": "4. 丁寧な終了と感謝の挨拶"
        },
        "keywords": ["감사합니다", "좋은 하루", "즐거운 하루", "행복한 하루", "고맙습니다", "이용해 주셔서", "thank you", "great day", "good day", "appreciate", "ありがとう", "良い一日", "ご利用いただき"],
        "weight": 25,
        "fail_feedback": {
            "ko": "상담 종료 시 '감사합니다, 좋은 하루 되세요'와 같은 정중한 감사 인사가 누락되었습니다.",
            "en": "Polite closing and thank you message was missing at the end of consultation.",
            "ja": "終了時の「ありがとうございました、良い一日を」などの感謝の挨拶が抜けています。"
        }
    }
]


def evaluate_chat_qa_compliance(messages: List[Dict[str, Any]], lang: str = "ko") -> Dict[str, Any]:
    """
    상담 대화록을 다차원적으로 평가하여 QA 점수, 금지어 위반, 필수 고지 체크리스트 및 코칭 피드백 산출
    - 한국어(ko), 영어(en), 일본어(ja) 다국어 완전 지원
    """
    if lang not in ["ko", "en", "ja"]:
        lang = "ko"

    if not messages:
        return _create_empty_audit_result(lang)

    # 상담원 발화와 고객 발화 분리
    agent_texts = []
    customer_texts = []
    
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ["agent_response", "agent", "assistant"]:
            agent_texts.append(content)
        elif role in ["customer", "user", "customer_rebuttal"]:
            customer_texts.append(content)

    full_agent_text = " ".join(agent_texts)
    full_customer_text = " ".join(customer_texts)

    # 1. 금지어 및 고위험 발언 탐지
    prohibited_violations = []
    total_penalty = 0

    for rule in PROHIBITED_RULES:
        for idx, text in enumerate(agent_texts, 1):
            matches = re.findall(rule["pattern"], text, re.IGNORECASE)
            if matches:
                matched_str = matches[0][0] if isinstance(matches[0], tuple) else str(matches[0])
                rule_name = rule["label"].get(lang, rule["label"]["ko"])
                advice = rule["advice"].get(lang, rule["advice"]["ko"])
                prohibited_violations.append({
                    "rule_name": rule_name,
                    "severity": rule["severity"],
                    "penalty": rule["penalty"],
                    "matched_text": matched_str,
                    "turn_index": idx,
                    "full_sentence": text.strip()[:100],
                    "advice": advice
                })
                total_penalty += rule["penalty"]

    # 2. 필수 고지 항목 체크리스트 판정
    checklist_results = []
    checklist_score = 0

    for item in MANDATORY_ITEMS:
        passed = any(kw.lower() in full_agent_text.lower() for kw in item["keywords"])
        status = "PASS" if passed else "FAIL"
        if passed:
            checklist_score += item["weight"]
        
        item_name = item["name"].get(lang, item["name"]["ko"])
        feedback_text = {
            "ko": "준수 완료",
            "en": "Complied",
            "ja": "遵守完了"
        }.get(lang, "준수 완료") if passed else item["fail_feedback"].get(lang, item["fail_feedback"]["ko"])
        
        checklist_results.append({
            "id": item["id"],
            "name": item_name,
            "status": status,
            "weight": item["weight"],
            "feedback": feedback_text
        })

    # 3. 다차원 점수 산출
    # 1) 공감도 및 친절도 (Empathy & Courtesy, 0~100)
    courtesy_keywords = [
        "안녕하세요", "감사", "소중한", "도와드리", "안내해 드리", "정성껏", "즐거운", "좋은 하루", "행복한",
        "hello", "thank", "help", "guide", "pleasure", "great day", "appreciate",
        "こんにちは", "ありがとう", "案内", "サポート", "良い一日", "幸い"
    ]
    empathy_keywords = [
        "죄송", "불편", "속상", "이해", "공감", "기다려",
        "sorry", "inconvenience", "understand", "apologize", "patience",
        "申し訳", "不便", "理解", "お待たせ"
    ]
    courtesy_hits = sum(1 for kw in courtesy_keywords if kw.lower() in full_agent_text.lower())
    empathy_hits = sum(1 for kw in empathy_keywords if kw.lower() in full_agent_text.lower())
    empathy_score = min(100, 70 + (courtesy_hits * 5) + (empathy_hits * 6))

    # 2) 문제해결 및 정확도 (Solution Accuracy, 0~100)
    solution_keywords = [
        "설정", "안내", "확인", "진행", "처리", "재발급", "규정", "조회", "링크", "해결", "영업일", "반영",
        "setting", "guide", "confirm", "process", "reissue", "policy", "solution", "business day",
        "設定", "案内", "確認", "進行", "処理", "再発行", "規定", "照会", "解決", "営業日"
    ]
    solution_hits = sum(1 for kw in solution_keywords if kw.lower() in full_agent_text.lower())
    solution_score = min(100, 60 + (solution_hits * 7))

    # 3) 절차 준수도 (Procedural Compliance, 0~100)
    compliance_score = max(0, checklist_score - total_penalty)

    # 4) 종합 QA 점수 산출 (가중 평균)
    raw_final_score = (empathy_score * 0.25) + (solution_score * 0.35) + (compliance_score * 0.40)
    final_score = max(0, min(100, int(round(raw_final_score))))

    # 4. 등급 판정 (S, A, B, C, F)
    critical_violation_count = sum(1 for v in prohibited_violations if v["severity"] == "CRITICAL")
    if critical_violation_count >= 2 or final_score < 60:
        grade = "F"
        grade_desc = {"ko": "재교육 필요 (Fail)", "en": "Failed / Requires Re-training (F)", "ja": "不合格 / 再研修が必要 (F)"}.get(lang)
        grade_color = "red"
    elif final_score >= 95:
        grade = "S"
        grade_desc = {"ko": "최우수 모범 상담 (Excellent)", "en": "Exceptional Exemplary Consultation (S)", "ja": "最優秀模範応対 (S)"}.get(lang)
        grade_color = "green"
    elif final_score >= 85:
        grade = "A"
        grade_desc = {"ko": "우수 상담 (Good)", "en": "Excellent Consultation (A)", "ja": "優秀応対 (A)"}.get(lang)
        grade_color = "blue"
    elif final_score >= 75:
        grade = "B"
        grade_desc = {"ko": "양호 (Pass)", "en": "Good / Passed (B)", "ja": "合格 / 良好 (B)"}.get(lang)
        grade_color = "orange"
    else:
        grade = "C"
        grade_desc = {"ko": "주의 필요 (Conditional Pass)", "en": "Conditional Pass / Needs Attention (C)", "ja": "注意が必要 / 条件付き合格 (C)"}.get(lang)
        grade_color = "orange"

    # 5. 맞춤형 AI 코칭 피드백 생성 (다국어)
    good_points = []
    improvements = []

    if lang == "en":
        if empathy_score >= 80:
            good_points.append("Consistently maintained polite honorifics and sincere empathy for customer inconvenience.")
        if checklist_results[0]["status"] == "PASS":
            good_points.append("Accurately verified customer identity and inquiry details early in the consultation.")
        if checklist_results[2]["status"] == "PASS":
            good_points.append("Proactively confirmed additional questions without rushing to close.")
        if not prohibited_violations:
            good_points.append("Complied with company policy without using abrupt refusals or blaming language.")
        if not good_points:
            good_points.append("Faithfully addressed customer inquiries throughout the session.")

        for v in prohibited_violations:
            improvements.append(f"[{v['rule_name']}] '{v['matched_text']}' (-{v['penalty']} pts): {v['advice']}")
        for item in checklist_results:
            if item["status"] == "FAIL":
                improvements.append(f"[{item['name']}] Missing: {item['feedback']}")
        if not improvements:
            improvements.append("Maintaining excellent customer service quality; no specific improvements required.")

        supervisor_summary = (
            f"This consultation is evaluated at {final_score} points (Grade {grade}). "
            f"Mandatory disclosure compliance is {int((checklist_score / 100) * 100)}%, "
            f"with {len(prohibited_violations)} prohibited word violation(s) detected. "
        )
        if grade in ["S", "A"]:
            supervisor_summary += "An exemplary model of outstanding customer care and policy compliance."
        elif grade == "B":
            supervisor_summary += "Overall procedure is solid, but supplementary disclosures will further elevate quality."
        else:
            supervisor_summary += "Compliance breach or prohibited language detected; managerial coaching is recommended."

    elif lang == "ja":
        if empathy_score >= 80:
            good_points.append("顧客の不便に対して誠実な共感表現と丁寧な敬語を一貫して維持しました。")
        if checklist_results[0]["status"] == "PASS":
            good_points.append("相談初期に顧客の基本情報および注文・機器情報を正確に確認しました。")
        if checklist_results[2]["status"] == "PASS":
            good_points.append("解決提供後に急いで終了せず、追加の質問事項を能動的に確認しました。")
        if not prohibited_violations:
            good_points.append("断定的な拒絶や禁止用語を使用せず、規定を遵守しました。")
        if not good_points:
            good_points.append("基本的な顧客の質問に対して誠実に対応しました。")

        for v in prohibited_violations:
            improvements.append(f"[{v['rule_name']}] '{v['matched_text']}' 減点 (-{v['penalty']}点): {v['advice']}")
        for item in checklist_results:
            if item["status"] == "FAIL":
                improvements.append(f"[{item['name']}] 未遵守: {item['feedback']}")
        if not improvements:
            improvements.append("現在優れた応対品質を維持しており、特段の改善事項はありません。")

        supervisor_summary = (
            f"本相談は総合スコア {final_score}点（{grade}等級）と判定されました。 "
            f"必須告知遵守率は {int((checklist_score / 100) * 100)}% で、"
            f"禁止用語違反は計 {len(prohibited_violations)}件 検知されました。 "
        )
        if grade in ["S", "A"]:
            supervisor_summary += "優れた顧客ケアと規定遵守を示した模範的な事例です。"
        elif grade == "B":
            supervisor_summary += "全般的な手順は良好ですが、追加案内の補強によりさらに完成度が高まります。"
        else:
            supervisor_summary += "規定違反または禁止用語が検知されたため、管理者によるコーチングを推奨します。"

    else:  # ko
        if empathy_score >= 80:
            good_points.append("고객의 불편 사항에 대해 적극적인 공감 표현과 정중한 경어를 일관되게 유지했습니다.")
        if checklist_results[0]["status"] == "PASS":
            good_points.append("상담 초기 고객의 기본 정보 및 기종/주문 내역을 정확히 확인하여 효율적인 상담을 이끌었습니다.")
        if checklist_results[2]["status"] == "PASS":
            good_points.append("솔루션 제공 후 성급히 종료하지 않고 추가 문의 사항을 능동적으로 확인했습니다.")
        if not prohibited_violations:
            good_points.append("단정적 거절이나 컴플레인 유발 금지어를 전혀 사용하지 않고 규정을 준수했습니다.")
        if not good_points:
            good_points.append("기본적인 고객 문의에 대해 성실하게 답변을 시도했습니다.")

        for v in prohibited_violations:
            improvements.append(f"[{v['rule_name']}] '{v['matched_text']}' 표현 감점 (-{v['penalty']}점): {v['advice']}")
        for item in checklist_results:
            if item["status"] == "FAIL":
                improvements.append(f"[{item['name']}] 누락: {item['feedback']}")
        if not improvements:
            improvements.append("현재 우수한 응대 수준을 유지하고 있으며, 특이 개선 사항이 없습니다.")

        supervisor_summary = (
            f"본 상담은 종합 점수 {final_score}점({grade}등급)으로 판정되었습니다. "
            f"필수 고지 준수율은 {int((checklist_score / 100) * 100)}%이며, "
            f"금지어 위반은 총 {len(prohibited_violations)}건 탐지되었습니다. "
        )
        if grade in ["S", "A"]:
            supervisor_summary += "우수한 고객 케어와 규정 준수를 보여준 모범 사례입니다."
        elif grade == "B":
            supervisor_summary += "전반적인 절차는 양호하나 추가 안내 사항을 보강하면 더욱 완성도 높은 상담이 됩니다."
        else:
            supervisor_summary += "규정 미준수 또는 금지어 사용이 감지되어 매니저 피드백 및 재교육 코칭을 권장합니다."

    return {
        "timestamp": datetime.now().isoformat(),
        "final_score": final_score,
        "grade": grade,
        "grade_desc": grade_desc,
        "grade_color": grade_color,
        "scores": {
            "empathy_score": empathy_score,
            "solution_score": solution_score,
            "compliance_score": compliance_score
        },
        "prohibited_violations": prohibited_violations,
        "violation_count": len(prohibited_violations),
        "checklist": checklist_results,
        "compliance_rate": int((checklist_score / 100) * 100),
        "coaching": {
            "good_points": good_points[:3],
            "improvements": improvements[:4],
            "supervisor_summary": supervisor_summary
        }
    }


def _create_empty_audit_result(lang: str = "ko") -> Dict[str, Any]:
    """대화록이 없을 때 기본 빈 감사 결과"""
    grade_desc = {"ko": "평가 데이터 없음", "en": "No Evaluation Data", "ja": "評価データなし"}.get(lang, "평가 데이터 없음")
    good_pt = {"ko": "대화 내용이 존재하지 않습니다.", "en": "No conversation history found.", "ja": "対話内容が存在しません。"}.get(lang)
    imp_pt = {"ko": "상담을 진행한 후 감사 리포트를 생성해 주세요.", "en": "Please conduct a consultation before generating an audit report.", "ja": "相談を実施した後に監査レポートを生成してください。"}.get(lang)
    summary = {"ko": "상담 대화 내역이 없어 평가를 진행할 수 없습니다.", "en": "Cannot conduct evaluation as conversation history is empty.", "ja": "対話履歴がないため評価を実施できません。"}.get(lang)

    return {
        "timestamp": datetime.now().isoformat(),
        "final_score": 0,
        "grade": "N/A",
        "grade_desc": grade_desc,
        "grade_color": "gray",
        "scores": {
            "empathy_score": 0,
            "solution_score": 0,
            "compliance_score": 0
        },
        "prohibited_violations": [],
        "violation_count": 0,
        "checklist": [
            {"name": item["name"].get(lang, item["name"]["ko"]), "status": "NOT_EVALUATED", "feedback": good_pt}
            for item in MANDATORY_ITEMS
        ],
        "compliance_rate": 0,
        "coaching": {
            "good_points": [good_pt],
            "improvements": [imp_pt],
            "supervisor_summary": summary
        }
    }
