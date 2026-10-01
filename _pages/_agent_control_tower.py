# ========================================
# _pages/_agent_control_tower.py
# AI 에이전트 관제탑 (Autonomous Agent Control Tower - OpenAI Dots / Operator Architecture)
# 다국어(ko, en, ja) 100% 지원 및 실시간 Gemini 추론 & 채팅 탭 양방향 데이터 브릿지
# ========================================
import streamlit as st
import json
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from core.agent_engine import AutonomousAgentEngine, AgentMission


# ----------------------------------------------------
# 다국어 번역 사전 (ko, en, ja)
# ----------------------------------------------------
TRANSLATIONS = {
    "ko": {
        "header_title": "🤖 AI 에이전트 관제탑 (Agent Control Tower)",
        "header_subtitle": "자율형 AI 백그라운드 에이전트(OpenAI Dots / Operator)의 업무를 지시하고 실시간 추론 과정을 관제합니다.",
        "badge_live": "● 에이전트 런타임: 정상 작동 중 (Live Connected)",
        "mission_cfg_title": "🎯 에이전트 업무 지시 (Mission Configuration)",
        "mission_cfg_desc": "OpenAI Dots 방식으로 자율 수행할 업무 목표를 프리셋에서 선택하거나 직접 프롬프트를 입력하세요.",
        "task_type_label": "📋 자동화 업무 유형 선택",
        "task_type_help": "에이전트가 백그라운드에서 자율적으로 완수할 목표 워크플로우를 선택합니다.",
        "task_options": [
            "고객 CS 민원 티켓 자동 분석 및 솔루션 도출",
            "불만/이탈 위험 고객 긴급 대응 워크플로우",
            "사내 FAQ 및 RAG 지식베이스 자동 최신화",
            "상담 품질(QA) & 컴플라이언스 규정 준수 감사"
        ],
        "custom_goal_label": "✨ Dots AI 자유 목표 직접 입력 (Optional - 자연어 지시)",
        "custom_goal_placeholder": "예: 고객 불만 및 보상 규정을 검토하고 규정 위반 없는 맞춤 솔루션 초안 및 CRM 업데이트를 자율 수행해 줘",
        "target_customer_label": "👤 대상 고객 선택",
        "customer_presets": [
            {"id": "CUST001", "name": "김민수", "label": "김민수 [CUST001] (해외 데이터 로밍 장애 긴급 보상)"},
            {"id": "CUST002", "name": "박은찬", "label": "박은찬 [CUST002] (투어 취소 및 100% 전액 환불 요청)"},
            {"id": "CUST003", "name": "이영희", "label": "이영희 [CUST003] (항공권 수수료 감면 및 일정 변경 문의)"}
        ],
        "execution_mode_label": "⚙️ 에이전트 실행 모드",
        "execution_mode_options": [
            "로컬 실시간 관제 모드 (Local Realtime)",
            "원격 백그라운드 에이전트(Dots) 디스패치"
        ],
        "execution_mode_help": "로컬 워커에서 직접 스트리밍 관제하거나 외부 백그라운드 에이전트 서버로 디스패치합니다.",
        "btn_run_workflow": "🚀 자율 업무 자동화 시작 (Run Autonomous Workflow)",
        "pipeline_title": "📡 에이전트 실시간 처리 상태 (Live Pipeline)",
        "status_running": "🤖 AI 에이전트가 자율 워크플로우를 수행 중입니다...",
        "status_complete": "✅ AI 에이전트 자율 업무 자동화 완료!",
        "step1_title": "1단계: 데이터 수집",
        "step1_desc": "고객 이력 및 RAG 규정 조회",
        "step2_title": "2단계: 심층 분석",
        "step2_desc": "감정/의도 분석 및 컴플라이언스 검증",
        "step3_title": "3단계: 결과 보고",
        "step3_desc": "맞춤 솔루션 생성 및 CRM 연동",
        "badge_completed": "✔ 완료",
        "badge_running": "⚡ 처리 중...",
        "badge_waiting": "⏳ 대기",
        "progress_step1": "🔍 1단계 데이터 수집 및 RAG 검색 진행 중...",
        "progress_step2": "🧠 2단계 심층 분석 및 최적 해결책 추론 중...",
        "progress_step3": "💾 3단계 결과 보고서 및 CRM 페이로드 생성 중...",
        "progress_finished": "🎉 자율 업무 완료! (100%)",
        "dashboard_title": "📊 에이전트 자율 수행 결과 (Execution Dashboard)",
        "metric_duration": "⏱️ 처리 소요 시간",
        "metric_duration_delta": "수동 15분 대비 -98.2%",
        "metric_compliance": "🛡️ BPO 규정 준수율",
        "metric_compliance_delta": "보상 약관 100% 충족",
        "metric_sentiment": "💖 분석 고객 감정",
        "metric_sentiment_val": "불만 ➔ 안심 전환",
        "metric_sentiment_delta": "이탈 위험 긴급 방어",
        "metric_automation": "⚙️ 자율화 완결도",
        "metric_automation_val": "100%",
        "metric_automation_delta": "원클릭 승인 준비 완료",
        "tab_summary": "📋 종합 업무 보고서 (Executive Summary)",
        "tab_solution": "💡 AI 자동 생성 맞춤 솔루션",
        "tab_crm": "💾 CRM 자동 동기화 데이터",
        "tab_telemetry": "🔍 Dots 도구 실행 추적 (Tool Telemetry)",
        "tab_remote": "📡 원격 백그라운드 에이전트 디스패치 규격",
        "root_cause_title": "🔍 분석된 핵심 원인",
        "reasoning_title": "🧠 AI 추론 상세",
        "rag_kb_title": "📚 참조된 RAG 지식베이스",
        "solution_draft_title": "✉️ 고객 1:1 발송용 솔루션 초안",
        "solution_draft_caption": "상담원 검토 및 즉시 발송 메시지 (필요 시 수정 가능)",
        "rec_channel": "권장 발송 채널",
        "rec_action": "추천 후속 조치",
        "btn_send_customer": "🚀 [채팅 탭]에 자동 승인 답변으로 전송",
        "send_success": "✅ 고객 맞춤 솔루션이 채팅 탭으로 안전하게 전송되었습니다!",
        "crm_payload_title": "📋 CRM 자동 업데이트 페이로드",
        "btn_commit_crm": "💾 CRM 시스템 데이터베이스 즉시 커밋",
        "commit_success": "✅ 고객 CRM 데이터베이스에 자율 처리 이력이 성공적으로 저장되었습니다.",
        "telemetry_title": "🛠️ OpenAI Dots 도구 실행 및 추론 로그",
        "telemetry_desc": "에이전트가 자율적으로 계획(Planning)하고 호출한 개별 도구 실행 내역입니다.",
        "remote_spec_title": "📡 OpenAI Dots / 백그라운드 에이전트 원격 디스패치 규격",
        "remote_spec_desc": "외부 자율형 에이전트 서버(REST/Webhook)와 상호 운용하기 위한 표준 JSON 통신 스펙입니다.",
        "btn_download_report": "📥 종합 업무 보고서 다운로드 (.md)",
        "btn_new_mission": "🔄 새로운 자율 에이전트 미션 실행하기",
        "report_filename": "Dots_AI_업무_보고서",
        "report_download_label": "Dots AI 자율 수행 보고서 (.md)"
    },
    "en": {
        "header_title": "🤖 AI Agent Control Tower (OpenAI Dots Engine)",
        "header_subtitle": "Command autonomous background AI agents and monitor real-time multi-step reasoning workflows.",
        "badge_live": "● Agent Runtime: Active & Connected (Live)",
        "mission_cfg_title": "🎯 Mission Configuration (Dots Operator)",
        "mission_cfg_desc": "Select an autonomous workflow template or provide a custom high-level goal in natural language.",
        "task_type_label": "📋 Select Autonomous Workflow Task",
        "task_type_help": "Select the high-level objective for the autonomous agent to execute in the background.",
        "task_options": [
            "Customer CS Ticket Auto-Analysis & Resolution",
            "Urgent Escalation & Churn Defense Workflow",
            "FAQ & Internal RAG Knowledge Base Auto-Sync",
            "QA Service Quality & Compliance Policy Audit"
        ],
        "custom_goal_label": "✨ Dots AI Custom Goal (Optional - Natural Language Prompt)",
        "custom_goal_placeholder": "e.g., Investigate customer dispute under policy, generate empathetic response, and prepare CRM sync payload.",
        "target_customer_label": "👤 Target Customer",
        "customer_presets": [
            {"id": "CUST001", "name": "Min-soo Kim", "label": "Min-soo Kim [CUST001] (Data Roaming Outage Compensation)"},
            {"id": "CUST002", "name": "Eun-chan Park", "label": "Eun-chan Park [CUST002] (Tour Cancellation & Full Refund Request)"},
            {"id": "CUST003", "name": "Young-hee Lee", "label": "Young-hee Lee [CUST003] (Flight Fee Waiver & Date Change)"}
        ],
        "execution_mode_label": "⚙️ Execution Mode",
        "execution_mode_options": [
            "Local Realtime Control Mode",
            "Remote Background Agent (Dots) Dispatch"
        ],
        "execution_mode_help": "Stream real-time telemetry on the local worker or dispatch to an external autonomous background agent server.",
        "btn_run_workflow": "🚀 Run Autonomous Workflow",
        "pipeline_title": "📡 Live Agent Execution Pipeline",
        "status_running": "🤖 AI Agent is autonomously executing the mission workflow...",
        "status_complete": "✅ Autonomous Agent Workflow Completed Successfully!",
        "step1_title": "Step 1: Data Ingestion",
        "step1_desc": "Retrieve customer profile & RAG policy docs",
        "step2_title": "Step 2: Deep Reasoning",
        "step2_desc": "Analyze sentiment, intent & compliance checks",
        "step3_title": "Step 3: Action & Synthesis",
        "step3_desc": "Generate custom solution & package CRM sync",
        "badge_completed": "✔ Completed",
        "badge_running": "⚡ Running...",
        "badge_waiting": "⏳ Pending",
        "progress_step1": "🔍 Step 1: Ingesting customer data & running RAG retrieval...",
        "progress_step2": "🧠 Step 2: Analyzing intent & deducing policy resolution...",
        "progress_step3": "💾 Step 3: Synthesizing executive report & CRM commit payload...",
        "progress_finished": "🎉 Autonomous Workflow Completed! (100%)",
        "dashboard_title": "📊 Execution Dashboard & Artifacts",
        "metric_duration": "⏱️ Total Processing Time",
        "metric_duration_delta": "vs Manual 15m: -98.2%",
        "metric_compliance": "🛡️ Compliance Adherence",
        "metric_compliance_delta": "100% Policy Compliant",
        "metric_sentiment": "💖 Customer Sentiment",
        "metric_sentiment_val": "Frustrated ➔ Relieved",
        "metric_sentiment_delta": "Churn Risk Mitigated",
        "metric_automation": "⚙️ Automation Degree",
        "metric_automation_val": "100%",
        "metric_automation_delta": "Ready for 1-Click Approval",
        "tab_summary": "📋 Executive Summary",
        "tab_solution": "💡 AI Solution Draft",
        "tab_crm": "💾 CRM Auto-Sync Data",
        "tab_telemetry": "🔍 Dots Tool Telemetry",
        "tab_remote": "📡 Remote Agent Dispatch Spec",
        "root_cause_title": "🔍 Root Cause Analysis",
        "reasoning_title": "🧠 AI Reasoning & Guardrails",
        "rag_kb_title": "📚 Retrieved RAG Knowledge Base",
        "solution_draft_title": "✉️ 1:1 Customer Response Draft",
        "solution_draft_caption": "Agent review & instant dispatch message (editable as needed)",
        "rec_channel": "Recommended Channel",
        "rec_action": "Recommended Next Action",
        "btn_send_customer": "🚀 Dispatch to Chat Simulator Now",
        "send_success": "✅ Tailored solution has been safely dispatched to the chat tab!",
        "crm_payload_title": "📋 CRM Update Payload (Structured JSON)",
        "btn_commit_crm": "💾 Commit to CRM Database Now",
        "commit_success": "✅ Autonomous resolution records successfully committed to CRM database.",
        "telemetry_title": "🛠️ OpenAI Dots Tool Calling & Execution Telemetry",
        "telemetry_desc": "Individual tools planned and invoked autonomously by the Dots agent during execution.",
        "remote_spec_title": "📡 OpenAI Dots / Background Agent Dispatch Specification",
        "remote_spec_desc": "Standard JSON communication spec for interoperability with external autonomous agent endpoints (REST/Webhook).",
        "btn_download_report": "📥 Download Mission Report (.md)",
        "btn_new_mission": "🔄 Run New Autonomous Mission",
        "report_filename": "Dots_AI_Mission_Report",
        "report_download_label": "Dots AI Autonomous Report (.md)"
    },
    "ja": {
        "header_title": "🤖 AI管制塔 (OpenAI Dots自律業務エンジン)",
        "header_subtitle": "自律型AIバックグラウンドエージェント(OpenAI Dots / Operator)に業務を指示し、リアルタイムの推論プロセスを管制します。",
        "badge_live": "● エージェントランタイム: 正常稼働中 (Live Connected)",
        "mission_cfg_title": "🎯 エージェント業務指示 (Mission Configuration)",
        "mission_cfg_desc": "OpenAI Dots方式で自律遂行する業務目標をプリセットから選択するか、自然言語で直接目標を入力してください。",
        "task_type_label": "📋 自動化業務タイプの選択",
        "task_type_help": "エージェントがバックグラウンドで自律的に完遂する目標ワークフローを選択します。",
        "task_options": [
            "顧客CSチケット自動分析および解決策の導出",
            "不満・解約リスク顧客の緊急対応ワークフロー",
            "社内FAQおよびRAGナレッジベース自動最新化",
            "応対品質(QA)＆コンプライアンス規約遵守監査"
        ],
        "custom_goal_label": "✨ Dots AI 自由目標の直接入力 (任意 - 自然言語プロンプト)",
        "custom_goal_placeholder": "例: 顧客の不満および補償規約を検証し、規約違反なくパーソナライズ対応文案およびCRM更新を自律実行してください",
        "target_customer_label": "👤 対象顧客の選択",
        "customer_presets": [
            {"id": "CUST001", "name": "キム・ミンス", "label": "キム・ミンス [CUST001] (データローミング障害緊急補償)"},
            {"id": "CUST002", "name": "パク・ウンチャン", "label": "パク・ウンチャン [CUST002] (ツアーキャンセル・100%返金要請)"},
            {"id": "CUST003", "name": "イ・ヨンヒ", "label": "イ・ヨンヒ [CUST003] (航空券手数料免除・日程変更照会)"}
        ],
        "execution_mode_label": "⚙️ エージェント実行モード",
        "execution_mode_options": [
            "ローカル・リアルタイム管制モード",
            "リモート・バックグラウンドエージェント(Dots)ディスパッチ"
        ],
        "execution_mode_help": "ローカルワーカーで直接ストリーミング管制するか、外部バックグラウンドエージェントサーバーへディスパッチします。",
        "btn_run_workflow": "🚀 自律業務の自動化開始 (Run Autonomous Workflow)",
        "pipeline_title": "📡 エージェント・リアルタイム処理パイプライン",
        "status_running": "🤖 AIエージェントが自律ワークフローを実行中です...",
        "status_complete": "✅ AIエージェント自律業務の自動化が完了しました！",
        "step1_title": "第1段階: データ収集",
        "step1_desc": "顧客履歴およびRAG規約の照会",
        "step2_title": "第2段階: 深層分析",
        "step2_desc": "感情・意図分析およびコンプライアンス検証",
        "step3_title": "第3段階: 結果報告",
        "step3_desc": "パーソナライズ解決策生成＆CRM連携",
        "badge_completed": "✔ 完了",
        "badge_running": "⚡ 処理中...",
        "badge_waiting": "⏳ 待機",
        "progress_step1": "🔍 第1段階: データ収集およびRAG検索を実行中...",
        "progress_step2": "🧠 第2段階: 感情分析および最適解決策を推論中...",
        "progress_step3": "💾 第3段階: 総合報告書およびCRMペイロードを生成中...",
        "progress_finished": "🎉 自律業務完了！ (100%)",
        "dashboard_title": "📊 エージェント自律遂行結果 (Execution Dashboard)",
        "metric_duration": "⏱️ 処理所要時間",
        "metric_duration_delta": "手動15分比 -98.2%",
        "metric_compliance": "🛡️ 規約遵守率",
        "metric_compliance_delta": "補償約款100%充足",
        "metric_sentiment": "💖 分析顧客感情",
        "metric_sentiment_val": "不満 ➔ 安心転換",
        "metric_sentiment_delta": "解約リスク緊急防御",
        "metric_automation": "⚙️ 自律化完結度",
        "metric_automation_val": "100%",
        "metric_automation_delta": "ワンクリック承認準備完了",
        "tab_summary": "📋 総合業務報告書 (Executive Summary)",
        "tab_solution": "💡 AI自動生成ソリューション",
        "tab_crm": "💾 CRM自動同期データ",
        "tab_telemetry": "🔍 Dotsツール実行追跡 (Tool Telemetry)",
        "tab_remote": "📡 リモートエージェント配信規格",
        "root_cause_title": "🔍 分析された根本原因",
        "reasoning_title": "🧠 AI推論詳細",
        "rag_kb_title": "📚 参照されたRAGナレッジベース",
        "solution_draft_title": "✉️ 顧客向け1:1送信ソリューション下書き",
        "solution_draft_caption": "オペレーター確認および即時送信メッセージ（必要に応じて編集可能）",
        "rec_channel": "推奨送信チャネル",
        "rec_action": "推奨フォローアクション",
        "btn_send_customer": "🚀 チャットタブへ即時送信",
        "send_success": "✅ 顧客向けソリューションがチャットタブに安全に送信されました！",
        "crm_payload_title": "📋 CRM自動更新ペイロード",
        "btn_commit_crm": "💾 CRMデータベースへ即時コミット",
        "commit_success": "✅ 顧客CRMデータベースに自律処理履歴が正常に保存されました。",
        "telemetry_title": "🛠️ OpenAI Dots ツール実行および推論ログ",
        "telemetry_desc": "エージェントが自律的に計画(Planning)し呼び出した個別ツールの実行履歴です。",
        "remote_spec_title": "📡 OpenAI Dots / バックグラウンドエージェント規格",
        "remote_spec_desc": "外部の自律型エージェントサーバー(REST/Webhook)と相互連携するための標準JSON仕様です。",
        "btn_download_report": "📥 業務報告書のダウンロード (.md)",
        "btn_new_mission": "🔄 新しい自律エージェントミッションを実行する",
        "report_filename": "Dots_AI_業務報告書",
        "report_download_label": "Dots AI 自律業務報告書 (.md)"
    }
}


def _build_markdown_report(result: Any, current_lang: str) -> str:
    """다운로드용 마크다운 종합 업무 보고서 생성 (모든 필드 방어적 접근)"""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mission = getattr(result, "mission", None)
    mission_id = getattr(mission, "mission_id", f"MSN-{int(time.time())}") if mission else "MSN-UNKNOWN"
    cust_name = getattr(mission, "customer_name", "Customer") if mission else "Customer"
    cust_id = getattr(mission, "customer_id", "CUST001") if mission else "CUST001"
    task_type = getattr(mission, "task_type", "Autonomous CS Workflow") if mission else "Autonomous CS Workflow"
    custom_goal = getattr(mission, "custom_goal", "") if mission else ""
    if not custom_goal:
        custom_goal = "None (Default Template)"

    tools_str = ""
    tool_list = getattr(result, "tool_telemetry", []) or []
    for t in tool_list:
        tools_str += f"- **Tool**: `{t.get('tool', 'Unknown')}` | Status: `{t.get('status', 'SUCCESS')}` | Input: `{json.dumps(t.get('input', {}), ensure_ascii=False)}`\n"
    if not tools_str:
        tools_str = "- No tool telemetry recorded.\n"

    metrics = getattr(result, "metrics", {}) or {}
    action_plan = getattr(result, "action_plan", {}) or {}
    steps = getattr(result, "steps", []) or []
    step2_out = steps[1].output if len(steps) > 1 and hasattr(steps[1], "output") else {}
    crm_payload = getattr(result, "crm_payload", {}) or {}

    return f"""# 🤖 OpenAI Dots Autonomous Agent Execution Report
**Generated At**: {created_at}  
**Mission ID**: `{mission_id}`  
**Language Runtime**: `{current_lang}`  
**Target Customer**: {cust_name} ({cust_id})  
**Workflow Type**: {task_type}  
**Custom Dots Goal**: {custom_goal}  

---

## 📊 1. Executive Summary & KPIs
{getattr(result, 'executive_summary', 'N/A')}

- **Total Execution Time**: {metrics.get('total_duration_sec', 2.5)}s
- **Compliance Score**: {metrics.get('compliance_score', 98.6)}%
- **AHT Reduction**: {metrics.get('aht_reduction_pct', 83.5)}%
- **Automation Degree**: {metrics.get('automation_degree', 100)}%

---

## 🔍 2. Root Cause & Reasoning
- **Identified Root Cause**:  
  {step2_out.get('root_cause', 'N/A')}
- **Autonomous Reasoning & Guardrails**:  
  {step2_out.get('reasoning_summary', 'N/A')}

---

## ✉️ 3. Synthesized 1:1 Customer Response Draft
```text
{action_plan.get('solution_draft', '')}
```

- **Recommended Delivery Channel**: {action_plan.get('channel', 'Chat & SMS')}
- **Follow-up Action**: {action_plan.get('recommendation', 'N/A')}

---

## 🛠️ 4. OpenAI Dots Tool Execution Telemetry
{tools_str}

---

## 📋 5. Structured CRM Commit Payload
```json
{json.dumps(crm_payload, indent=2, ensure_ascii=False)}
```

---
*Report autonomously generated by AI Agent Control Tower (OpenAI Dots / Operator Emulation System)*
"""


def render_agent_control_tower_page(current_lang: Optional[str] = None):
    """AI 에이전트 관제탑 대시보드 메인 렌더링 함수 (다국어 & Dots AI 아키텍처)"""
    if not current_lang:
        current_lang = st.session_state.get("language", "ko")
    if current_lang not in ["ko", "en", "ja"]:
        current_lang = "ko"

    T = TRANSLATIONS.get(current_lang, TRANSLATIONS["ko"])

    # ----------------------------------------------------
    # 1. 관제탑 헤더 및 상태 배너 (다국어)
    # ----------------------------------------------------
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 22px 26px; border-radius: 12px; margin-bottom: 24px; border: 1px solid #334155; color: white;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 style="color: #38bdf8; margin: 0; font-size: 1.8rem; font-weight: 700;">
                    {T["header_title"]}
                </h1>
                <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 0.95rem;">
                    {T["header_subtitle"]}
                </p>
            </div>
            <div style="margin-top: 8px;">
                <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.4);">
                    {T["badge_live"]}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # 2. 에이전트 미션 설정 패널 (동적 고객 풀 & 실시간 채팅 연동)
    # ----------------------------------------------------
    st.markdown(f"### {T['mission_cfg_title']}")
    st.caption(T["mission_cfg_desc"])

    # [핵심] 고객 목록 구성: 기본 프리셋 3명 + 채팅 탭의 실시간 대화 자동 결합
    customer_presets = list(T["customer_presets"])

    # 채팅 시뮬레이터 세션(simulator_messages)에서 실시간 고객 발화 감지
    sim_msgs = st.session_state.get("simulator_messages", [])
    cust_inquiries = [
        m.get("content", "") for m in sim_msgs 
        if isinstance(m, dict) and m.get("role") in ["customer", "customer_rebuttal", "initial_query", "user"]
    ]
    if cust_inquiries:
        latest_cust_msg = cust_inquiries[-1]
        live_cust_name = st.session_state.get("current_customer_name", "실시간 채팅 고객")
        snippet = (latest_cust_msg[:24] + "...") if len(latest_cust_msg) > 24 else latest_cust_msg
        
        live_entry = {
            "id": "LIVE_CHAT",
            "name": live_cust_name,
            "label": f"🔥 [실시간 채팅 연동] {live_cust_name} ({snippet})"
        }
        # 맨 앞에 추가하여 사용자가 채팅 후 넘어왔을 때 바로 보이도록 함
        customer_presets.insert(0, live_entry)

    customer_dict = {p["label"]: p for p in customer_presets}

    with st.container():
        col1, col2, col3 = st.columns([5, 3, 3])

        with col1:
            selected_task = st.selectbox(
                T["task_type_label"],
                options=T["task_options"],
                index=0,
                help=T["task_type_help"]
            )

        with col2:
            selected_cust_label = st.selectbox(
                T["target_customer_label"],
                options=list(customer_dict.keys()),
                index=0
            )
            selected_customer = customer_dict[selected_cust_label]

        with col3:
            execution_mode = st.selectbox(
                T["execution_mode_label"],
                options=T["execution_mode_options"],
                index=0,
                help=T["execution_mode_help"]
            )

        # OpenAI Dots 자유 목표 프롬프트 입력창
        custom_goal = st.text_input(
            T["custom_goal_label"],
            placeholder=T["custom_goal_placeholder"],
            key="custom_dots_goal_input"
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 실행 버튼
    start_clicked = st.button(
        T["btn_run_workflow"],
        type="primary",
        use_container_width=True,
        key="btn_run_autonomous_agent"
    )

    # ----------------------------------------------------
    # 3. 자율 업무 자동화 실행 및 실시간 파이프라인 (Live Pipeline)
    # ----------------------------------------------------
    if start_clicked:
        st.session_state.agent_running = True
        st.session_state.agent_workflow_result = None

    if st.session_state.get("agent_running", False):
        st.markdown("---")
        st.markdown(f"### {T['pipeline_title']}")

        # 시각화 3단계 상태카드 컨테이너
        step_cols = st.columns(3)
        with step_cols[0]:
            card_step1 = st.empty()
        with step_cols[1]:
            card_step2 = st.empty()
        with step_cols[2]:
            card_step3 = st.empty()

        progress_bar = st.progress(0, text=T["progress_step1"])
        status_box = st.status(T["status_running"], expanded=True)

        engine = AutonomousAgentEngine()
        mission = AgentMission(
            mission_id=f"MSN-{int(time.time())}",
            task_type=selected_task,
            customer_id=selected_customer.get("id", "CUST001"),
            customer_name=selected_customer.get("name", "고객"),
            custom_goal=custom_goal.strip() if custom_goal else "",
            params={"execution_mode": execution_mode}
        )

        final_state = None
        for current_state in engine.run_workflow_stream(mission, lang=current_lang):
            final_state = current_state
            pct = int(current_state.overall_progress * 100)

            # 단계 카드 업데이트 (언어별 명칭)
            step_names = [
                (T["step1_title"], T["step1_desc"]),
                (T["step2_title"], T["step2_desc"]),
                (T["step3_title"], T["step3_desc"])
            ]

            cards = [card_step1, card_step2, card_step3]
            for i in range(3):
                s = current_state.steps[i]
                if s.status == "COMPLETED":
                    badge = T["badge_completed"]
                    border_color = "#10b981"
                    bg_color = "rgba(16, 185, 129, 0.08)"
                elif s.status == "RUNNING":
                    badge = T["badge_running"]
                    border_color = "#38bdf8"
                    bg_color = "rgba(56, 189, 248, 0.12)"
                else:
                    badge = T["badge_waiting"]
                    border_color = "#cbd5e1"
                    bg_color = "rgba(241, 245, 249, 0.5)"

                cards[i].markdown(f"""
                <div style="border: 2px solid {border_color}; background-color: {bg_color}; border-radius: 10px; padding: 12px 14px; min-height: 85px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 0.95rem;">{step_names[i][0]}</span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: {border_color};">{badge}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">{step_names[i][1]}</div>
                </div>
                """, unsafe_allow_html=True)

            # 진행률 텍스트
            if pct < 33:
                step_desc = T["progress_step1"]
            elif pct < 66:
                step_desc = T["progress_step2"]
            elif pct < 100:
                step_desc = T["progress_step3"]
            else:
                step_desc = T["progress_finished"]

            progress_bar.progress(pct, text=f"{pct}% - {step_desc}")

            # 실시간 로그 스트리밍 (상태창 내부)
            with status_box:
                for step in current_state.steps:
                    for log_msg in step.logs[-2:]:
                        st.write(log_msg)

        status_box.update(label=T["status_complete"], state="complete", expanded=False)
        st.session_state.agent_running = False
        st.session_state.agent_workflow_result = final_state

    # ----------------------------------------------------
    # 4. 최종 결과 시각화 대시보드 (Executive Summary & Artifacts)
    # ----------------------------------------------------
    result = st.session_state.get("agent_workflow_result")
    if result:
        # 방어적 속성 검사
        if not hasattr(result, "tool_telemetry") or result.tool_telemetry is None:
            result.tool_telemetry = []
        if not hasattr(result, "lang") or not result.lang:
            result.lang = "ko"
        if hasattr(result, "mission") and result.mission:
            if not hasattr(result.mission, "custom_goal") or result.mission.custom_goal is None:
                result.mission.custom_goal = ""

    if result and getattr(result, "is_finished", False):
        st.markdown("---")
        st.markdown(f"### {T['dashboard_title']}")

        metrics = getattr(result, "metrics", {}) or {}
        steps = getattr(result, "steps", []) or []
        step1_out = steps[0].output if len(steps) > 0 and hasattr(steps[0], "output") else {}
        step2_out = steps[1].output if len(steps) > 1 and hasattr(steps[1], "output") else {}
        action_plan = getattr(result, "action_plan", {}) or {}
        crm_payload = getattr(result, "crm_payload", {}) or {}
        tool_telemetry = getattr(result, "tool_telemetry", []) or []

        # 핵심 성과 KPI 메트릭 카드
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label=T["metric_duration"],
                value=f"{metrics.get('total_duration_sec', 2.5)}s",
                delta=T["metric_duration_delta"],
                delta_color="normal"
            )
        with m2:
            st.metric(
                label=T["metric_compliance"],
                value=f"{metrics.get('compliance_score', 98.6)}%",
                delta=T["metric_compliance_delta"],
                delta_color="normal"
            )
        with m3:
            st.metric(
                label=T["metric_sentiment"],
                value=step2_out.get("sentiment_label", T["metric_sentiment_val"])[:18],
                delta=T["metric_sentiment_delta"],
                delta_color="normal"
            )
        with m4:
            st.metric(
                label=T["metric_automation"],
                value=T["metric_automation_val"],
                delta=T["metric_automation_delta"],
                delta_color="normal"
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # 결과 탭 (Executive Summary, AI 솔루션, CRM 연동, Dots 도구 텔레메트리, 원격 디스패치)
        tab_summary, tab_solution, tab_crm, tab_telemetry, tab_remote = st.tabs([
            T["tab_summary"],
            T["tab_solution"],
            T["tab_crm"],
            T["tab_telemetry"],
            T["tab_remote"]
        ])

        with tab_summary:
            st.info(getattr(result, "executive_summary", ""))

            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown(f"#### {T['root_cause_title']}")
                st.write(step2_out.get("root_cause", "Analysis Complete"))
                st.markdown(f"#### {T['reasoning_title']}")
                st.write(step2_out.get("reasoning_summary", "Reasoning Complete"))
            with sub_col2:
                st.markdown(f"#### {T['rag_kb_title']}")
                kb_list = step1_out.get("kb_docs", [])
                for kb in kb_list:
                    with st.expander(f"📄 {kb.get('title')}", expanded=True):
                        st.caption(kb.get("content"))

        with tab_solution:
            st.markdown(f"#### {T['solution_draft_title']}")
            solution_text = action_plan.get("solution_draft", "")
            edited_solution = st.text_area(
                T["solution_draft_caption"],
                value=solution_text,
                height=180
            )
            col_act1, col_act2 = st.columns([3, 1.4])
            with col_act1:
                st.caption(f"{T['rec_channel']}: {action_plan.get('channel', 'Chat & SMS')} | {T['rec_action']}: {action_plan.get('recommendation')}")
            with col_act2:
                # [핵심 역동기화 브릿지] 버튼 클릭 시 채팅 탭의 simulator_messages로 답변 주입
                if st.button(T["btn_send_customer"], type="primary", use_container_width=True):
                    if "simulator_messages" not in st.session_state:
                        st.session_state.simulator_messages = []
                    
                    st.session_state.simulator_messages.append({
                        "role": "agent_response",
                        "content": f"🤖 **[Dots AI 자율 에이전트 승인 답변]**\n\n{edited_solution}",
                        "feedback": None,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.session_state.agent_response_area_text = edited_solution
                    st.session_state.initial_advice_provided = True
                    st.success(f"{T['send_success']}")
                    st.info("💡 이제 상단 또는 사이드바 메뉴에서 **[채팅/이메일]** 탭으로 이동하시면 방금 승인된 에이전트 답변이 대화창에 즉시 등록되어 있는 것을 확인하실 수 있습니다!")

        with tab_crm:
            st.markdown(f"#### {T['crm_payload_title']}")
            st.json(crm_payload)
            if st.button(T["btn_commit_crm"], use_container_width=True):
                if "crm_committed_records" not in st.session_state:
                    st.session_state.crm_committed_records = []
                st.session_state.crm_committed_records.append(crm_payload)
                st.success(T["commit_success"])

        with tab_telemetry:
            st.markdown(f"#### {T['telemetry_title']}")
            st.caption(T["telemetry_desc"])
            if tool_telemetry:
                for idx, t in enumerate(tool_telemetry, 1):
                    with st.container():
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.05); border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; color: #0284c7;">Tool #{idx}: {t.get('tool', 'Unknown')}</span>
                                <span style="background: rgba(16, 185, 129, 0.15); color: #059669; font-weight: 600; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem;">
                                    {t.get('status', 'SUCCESS')}
                                </span>
                            </div>
                            <div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">
                                <code>Input: {json.dumps(t.get('input', {}), ensure_ascii=False)}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("No tool telemetry records.")

        with tab_remote:
            st.markdown(f"#### {T['remote_spec_title']}")
            st.caption(T["remote_spec_desc"])
            st.code(json.dumps(getattr(result, "remote_dispatch_spec", {}), indent=2, ensure_ascii=False), language="json")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 다운로드 및 신규 미션 실행 행
        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:
            report_md = _build_markdown_report(result, current_lang)
            mission_obj = getattr(result, "mission", None)
            mission_id = getattr(mission_obj, "mission_id", "report") if mission_obj else "report"
            st.download_button(
                label=T["btn_download_report"],
                data=report_md,
                file_name=f"{T['report_filename']}_{mission_id}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dcol2:
            if st.button(T["btn_new_mission"], use_container_width=True):
                st.session_state.agent_workflow_result = None
                st.session_state.agent_running = False
                st.rerun()
