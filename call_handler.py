import os
import time
from datetime import datetime
import json

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
                        except:
                            pass
                    if size == 0 and hasattr(audio_bytes, 'size'):
                        try:
                            size = audio_bytes.size
                        except:
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
                        except:
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
        time.sleep(0.3)
        return {
            "text": "네, 잘 들립니다. 말씀해주세요.",
            "audio_available": False
        }
    
    def receive_call(self, caller_id, caller_phone):
        """전화 수신 처리"""
        call_id = self.start_call(caller_id, call_type="incoming")
        return {
            "call_id": call_id,
            "caller_id": caller_id,
            "caller_phone": caller_phone,
            "status": "received"
        }
    
    def process_inquiry(self, call_id, inquiry_text):
        """문의 입력 처리"""
        if not self.is_call_active:
            return None
        
        inquiry_data = {
            "call_id": call_id,
            "inquiry_text": inquiry_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # 문의 저장
        inquiry_file = os.path.join(self.call_dir, f"{call_id}_inquiry.json")
        with open(inquiry_file, 'w', encoding='utf-8') as f:
            json.dump(inquiry_data, f, ensure_ascii=False, indent=2)
        
        return inquiry_data




