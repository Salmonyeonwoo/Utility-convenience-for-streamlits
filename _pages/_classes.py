# ========================================
# _pages/_classes.py
# 클래스 정의 모듈
# ========================================

import os
import json
from datetime import datetime


class CallHandler:
    """전화 통화 처리 클래스"""

    def __init__(self):
        self.call_dir = "call_logs"
        os.makedirs(self.call_dir, exist_ok=True)
        self.is_call_active = False
        self.call_start_time = None
        self.call_audio_chunks = []

    def start_call(self, user_id, call_type="audio"):
        """통화 시작"""
        self.is_call_active = True
        self.call_start_time = datetime.now()
        self.call_audio_chunks = []
        self.current_call_id = f"{user_id}_{self.call_start_time.strftime('%Y%m%d_%H%M%S')}"
        return self.current_call_id

    def end_call(self, user_id, call_id):
        """통화 종료"""
        self.is_call_active = False
        call_duration = 0
        if self.call_start_time:
            call_duration = (datetime.now() - self.call_start_time).total_seconds()

        # 통화 로그 저장
        call_log = {
            "call_id": call_id,
            "user_id": user_id,
            "start_time": self.call_start_time.isoformat() if self.call_start_time else None,
            "end_time": datetime.now().isoformat(),
            "duration": call_duration,
            "audio_chunks": len(self.call_audio_chunks)
        }

        log_file = os.path.join(self.call_dir, f"{call_id}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(call_log, f, ensure_ascii=False, indent=2)

        self.call_start_time = None
        self.call_audio_chunks = []

        return call_duration

    def add_audio_chunk(self, audio_bytes, sender="user"):
        """오디오 청크 추가"""
        if self.is_call_active:
            size = 0
            if audio_bytes:
                try:
                    if hasattr(audio_bytes, 'getvalue'):
                        try:
                            audio_data = audio_bytes.getvalue()
                            size = len(audio_data) if audio_data else 0
                        except BaseException:
                            pass
                    if size == 0 and hasattr(audio_bytes, 'size'):
                        try:
                            size = audio_bytes.size
                        except BaseException:
                            pass
                    if size == 0 and hasattr(audio_bytes, 'read'):
                        try:
                            if hasattr(audio_bytes, 'tell'):
                                current_pos = audio_bytes.tell()
                            else:
                                current_pos = 0
                            if hasattr(audio_bytes, 'seek'):
                                audio_bytes.seek(0)
                            data = audio_bytes.read()
                            size = len(data) if data else 0
                            if hasattr(audio_bytes, 'seek'):
                                audio_bytes.seek(current_pos)
                        except BaseException:
                            pass
                    if size == 0 and isinstance(audio_bytes, bytes):
                        size = len(audio_bytes)
                except Exception:
                    size = 0

            chunk_info = {
                "timestamp": datetime.now().isoformat(),
                "sender": sender,
                "size": size
            }
            self.call_audio_chunks.append(chunk_info)
            return True
        return False

    def get_call_status(self):
        """통화 상태 반환"""
        if not self.is_call_active:
            return None

        duration = 0
        if self.call_start_time:
            duration = (datetime.now() - self.call_start_time).total_seconds()

        return {
            "is_active": self.is_call_active,
            "duration": duration,
            "chunks_count": len(self.call_audio_chunks)
        }

    def simulate_response(self, user_audio_bytes=None):
        """상대방 응답 시뮬레이션"""
        import time
        time.sleep(0.3)
        return {
            "text": "네, 잘 들립니다. 말씀해주세요.",
            "audio_available": False
        }


class AppAudioHandler:
    """오디오 처리 클래스"""

    def __init__(self):
        self.audio_dir = "audio_files"
        os.makedirs(self.audio_dir, exist_ok=True)

    def save_audio(self, audio_bytes, user_id):
        """오디오 바이트를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}.wav"
        filepath = os.path.join(self.audio_dir, filename)

        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        return filepath


class CustomerDataManager:
    """고객 데이터 관리 클래스"""
    
    def __init__(self):
        self.data_dir = "customer_data"
        os.makedirs(self.data_dir, exist_ok=True)

    def load_customer_data(self, customer_id):
        filepath = os.path.join(self.data_dir, f"{customer_id}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def list_all_customers(self):
        if not os.path.exists(self.data_dir):
            return []
        customers = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                customer_id = filename.replace('.json', '')
                data = self.load_customer_data(customer_id)
                if data:
                    customers.append({
                        "customer_id": customer_id,
                        "updated_at": data.get("updated_at", ""),
                        "has_data": True
                    })
        return sorted(customers, key=lambda x: x.get("updated_at", ""), reverse=True)



