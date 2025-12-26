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
시뮬레이션 처리 모듈 (메인 진입점)
고객 응대 시뮬레이션, 채팅/전화 대화 생성, 힌트 생성 등의 핵심 기능을 제공합니다.
"""

# 기본 함수들
from simulation_handler_base import get_chat_history_for_prompt

# 힌트 생성
from simulation_handler_hint import generate_realtime_hint

# 에이전트 응답 생성
from simulation_handler_agent_response import (
    generate_agent_response_draft,
    generate_outbound_call_summary,
    generate_agent_first_greeting
)

# 고객 반응 생성
from simulation_handler_customer_reaction import (
    generate_customer_reaction,
    generate_customer_reaction_for_call,
    generate_customer_reaction_for_first_greeting,
    generate_customer_closing_response
)

# 요약 생성
from simulation_handler_summary import (
    summarize_history_with_ai,
    summarize_history_for_call
)

# 하위 호환성을 위해 모든 함수를 export
__all__ = [
    'get_chat_history_for_prompt',
    'generate_realtime_hint',
    'generate_agent_response_draft',
    'generate_outbound_call_summary',
    'generate_customer_reaction',
    'summarize_history_with_ai',
    'generate_customer_reaction_for_call',
    'generate_customer_reaction_for_first_greeting',
    'summarize_history_for_call',
    'generate_customer_closing_response',
    'generate_agent_first_greeting',
]
