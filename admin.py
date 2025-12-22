import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

class AdminManager:
    """관리자 모드 관리 클래스"""
    
    def __init__(self):
        self.admin_password = "admin123"
        self.config_file = "admin_config.json"
        self._load_config()
    
    def _load_config(self):
        """관리자 설정 로드"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "password": self.admin_password,
                "users": [],
                "settings": {}
            }
            self._save_config()
    
    def _save_config(self):
        """관리자 설정 저장"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def authenticate(self, password):
        """관리자 인증"""
        return password == self.config.get("password", self.admin_password)
    
    def show_login(self):
        """관리자 로그인 화면"""
        st.title("🔐 관리자 로그인")
        
        password = st.text_input("비밀번호", type="password", key="admin_password_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("로그인", use_container_width=True):
                if self.authenticate(password):
                    st.session_state.admin_logged_in = True
                    st.session_state.is_admin = True
                    st.session_state.show_admin_login = False
                    st.success("관리자 로그인 성공!")
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        
        with col2:
            if st.button("취소", use_container_width=True):
                st.session_state.show_admin_login = False
    
    def show_dashboard(self):
        """관리자 대시보드"""
        st.title("👨‍💼 관리자 대시보드")
        
        if not st.session_state.admin_logged_in:
            st.warning("관리자 권한이 필요합니다.")
            if st.button("로그인 페이지로"):
                st.session_state.show_admin = False
                st.session_state.show_admin_login = True
            return
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 통계", 
            "💬 채팅 로그", 
            "👥 사용자 관리",
            "⚙️ 설정"
        ])
        
        with tab1:
            self._show_statistics()
        
        with tab2:
            self._show_chat_logs()
        
        with tab3:
            self._show_user_management()
        
        with tab4:
            self._show_settings()
        
        st.divider()
        if st.button("채팅으로 돌아가기"):
            st.session_state.show_admin = False
    
    def _show_statistics(self):
        """통계 표시"""
        st.subheader("📊 채팅 통계")
        
        log_dir = "chat_logs"
        if not os.path.exists(log_dir):
            st.info("아직 채팅 로그가 없습니다.")
            return
        
        all_logs = []
        user_stats = {}
        
        for filename in os.listdir(log_dir):
            if filename.endswith('.json'):
                user_id = filename.replace('.json', '')
                filepath = os.path.join(log_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    
                    user_stats[user_id] = {
                        "user_id": user_id[:8],
                        "total_messages": len(logs),
                        "user_messages": len([l for l in logs if l.get("sender") == "user"]),
                        "assistant_messages": len([l for l in logs if l.get("sender") == "assistant"]),
                        "audio_messages": len([l for l in logs if l.get("audio_file")]),
                        "last_activity": max([l.get("timestamp", "") for l in logs]) if logs else ""
                    }
                    
                    all_logs.extend(logs)
                except Exception as e:
                    st.error(f"로그 파일 읽기 오류: {filename} - {str(e)}")
        
        if user_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 사용자 수", len(user_stats))
            with col2:
                st.metric("총 메시지 수", len(all_logs))
            with col3:
                st.metric("오디오 메시지 수", len([l for l in all_logs if l.get("audio_file")]))
            with col4:
                st.metric("활성 사용자", len([u for u in user_stats.values() if u["last_activity"]]))
            
            st.subheader("사용자별 통계")
            df = pd.DataFrame(list(user_stats.values()))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("통계 데이터가 없습니다.")
    
    def _show_chat_logs(self):
        """채팅 로그 표시"""
        st.subheader("💬 채팅 로그")
        
        log_dir = "chat_logs"
        if not os.path.exists(log_dir):
            st.info("채팅 로그가 없습니다.")
            return
        
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if not log_files:
            st.info("로그 파일이 없습니다.")
            return
        
        selected_file = st.selectbox(
            "사용자 선택",
            log_files,
            format_func=lambda x: f"사용자: {x.replace('.json', '')[:8]}"
        )
        
        if selected_file:
            filepath = os.path.join(log_dir, selected_file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                
                st.write(f"**총 {len(logs)}개의 메시지**")
                
                for i, log in enumerate(logs):
                    with st.expander(
                        f"[{log.get('timestamp', 'N/A')}] {log.get('sender', 'unknown')} - {log.get('message', '')[:50]}..."
                    ):
                        st.write(f"**발신자:** {log.get('sender', 'unknown')}")
                        st.write(f"**시간:** {log.get('timestamp', 'N/A')}")
                        st.write(f"**메시지:** {log.get('message', '')}")
                        if log.get('audio_file'):
                            st.write(f"**오디오 파일:** {log.get('audio_file')}")
                            if os.path.exists(log.get('audio_file')):
                                st.audio(log.get('audio_file'), format="audio/wav")
                
                st.download_button(
                    "로그 다운로드 (JSON)",
                    json.dumps(logs, ensure_ascii=False, indent=2),
                    file_name=selected_file,
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"로그 파일 읽기 오류: {str(e)}")
    
    def _show_user_management(self):
        """사용자 관리"""
        st.subheader("👥 사용자 관리")
        
        log_dir = "chat_logs"
        if not os.path.exists(log_dir):
            st.info("사용자 데이터가 없습니다.")
            return
        
        user_files = [f.replace('.json', '') for f in os.listdir(log_dir) if f.endswith('.json')]
        
        if user_files:
            selected_user = st.selectbox("사용자 선택", user_files)
            
            if selected_user:
                filepath = os.path.join(log_dir, f"{selected_user}.json")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("총 메시지 수", len(logs))
                        st.metric("사용자 메시지", len([l for l in logs if l.get("sender") == "user"]))
                    with col2:
                        st.metric("봇 메시지", len([l for l in logs if l.get("sender") == "assistant"]))
                        st.metric("오디오 메시지", len([l for l in logs if l.get("audio_file")]))
                    
                    if st.button("사용자 로그 삭제", type="primary"):
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            st.success("사용자 로그가 삭제되었습니다.")
                except Exception as e:
                    st.error(f"오류: {str(e)}")
        else:
            st.info("등록된 사용자가 없습니다.")
    
    def _show_settings(self):
        """설정"""
        st.subheader("⚙️ 관리자 설정")
        
        st.write("**비밀번호 변경**")
        new_password = st.text_input("새 비밀번호", type="password", key="new_admin_password")
        confirm_password = st.text_input("비밀번호 확인", type="password", key="confirm_admin_password")
        
        if st.button("비밀번호 변경"):
            if new_password == confirm_password:
                if new_password:
                    self.config["password"] = new_password
                    self._save_config()
                    st.success("비밀번호가 변경되었습니다.")
                else:
                    st.error("비밀번호를 입력하세요.")
            else:
                st.error("비밀번호가 일치하지 않습니다.")
        
        st.divider()
        
        st.write("**데이터 관리**")
        if st.button("모든 채팅 로그 삭제", type="primary"):
            log_dir = "chat_logs"
            if os.path.exists(log_dir):
                for filename in os.listdir(log_dir):
                    filepath = os.path.join(log_dir, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                st.success("모든 채팅 로그가 삭제되었습니다.")


