import os
import wave
from datetime import datetime

class AudioHandler:
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
    
    def get_audio_info(self, filepath):
        """오디오 파일 정보 가져오기"""
        if not os.path.exists(filepath):
            return None
        
        try:
            with wave.open(filepath, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                
                return {
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "frames": frames,
                    "channels": wav_file.getnchannels()
                }
        except Exception as e:
            return None
    
    def list_user_audios(self, user_id):
        """사용자의 오디오 파일 목록 가져오기"""
        if not os.path.exists(self.audio_dir):
            return []
        
        user_files = [
            f for f in os.listdir(self.audio_dir)
            if f.startswith(user_id) and f.endswith('.wav')
        ]
        
        return sorted(user_files, reverse=True)
