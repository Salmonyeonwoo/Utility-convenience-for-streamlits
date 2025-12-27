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
시뮬레이션 핸들러 기본 함수들
"""

import streamlit as st

def get_chat_history_for_prompt(include_attachment=False):
    """메모리에서 대화 기록을 추출하여 프롬프트에 사용할 문자열 형태로 반환 (채팅용)"""
    history_str = ""
    for msg in st.session_state.simulator_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "customer" or role == "customer_rebuttal":
            history_str += f"Customer: {content}\n"
        elif role == "agent_response":
            history_str += f"Agent: {content}\n"
    return history_str



