# -*- coding: utf-8 -*-
"""
채팅 시뮬레이터 - Import 및 일일 통계 모듈
"""
import streamlit as st
from lang_pack import LANG
from datetime import datetime, timedelta
import numpy as np
from simulation_handler import *
from visualization import *
from audio_handler import *
from llm_client import get_api_key
from typing import List, Dict, Any
import uuid
import time
import os

def render_daily_stats():
    """일일 데이터 수집 통계 표시"""
    daily_stats = get_daily_data_statistics(st.session_state.language)
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("오늘 수집된 케이스", daily_stats["total_cases"])
    with col_stat2:
        st.metric("고유 고객 수", daily_stats["unique_customers"], 
                 delta="목표: 5인 이상" if daily_stats["target_met"] else "목표 미달")
    with col_stat3:
        st.metric("요약 완료 케이스", daily_stats["cases_with_summary"])
    with col_stat4:
        status_icon = "✅" if daily_stats["target_met"] else "⚠️"
        st.metric("목표 달성", status_icon, 
                 delta="달성" if daily_stats["target_met"] else "미달성")
    
    st.markdown("---")



