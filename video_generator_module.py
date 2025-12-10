"""
비디오 생성 API 통합 모듈
D-ID, Synthesia 등의 API를 사용하여 비디오를 생성하는 기능
또는 OpenAI/Gemini API를 사용한 대안 방법 제공
"""

import os
import requests
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import streamlit as st
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class VideoGenerator:
    """비디오 생성 API 클래스"""
    
    def __init__(self, api_provider: str = "d-id"):
        """
        Args:
            api_provider: "d-id", "synthesia", "heygen" 등
        """
        self.api_provider = api_provider.lower()
        self.api_key = self._get_api_key()
        
    def _get_api_key(self) -> Optional[str]:
        """API 키 가져오기"""
        if self.api_provider == "d-id":
            # Streamlit secrets에서 가져오기
            if hasattr(st, 'secrets'):
                return st.secrets.get("D_ID_API_KEY") or os.getenv("D_ID_API_KEY")
            return os.getenv("D_ID_API_KEY")
        elif self.api_provider == "synthesia":
            if hasattr(st, 'secrets'):
                return st.secrets.get("SYNTHESIA_API_KEY") or os.getenv("SYNTHESIA_API_KEY")
            return os.getenv("SYNTHESIA_API_KEY")
        elif self.api_provider == "openai":
            if hasattr(st, 'secrets'):
                return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            return os.getenv("OPENAI_API_KEY")
        elif self.api_provider == "gemini":
            if hasattr(st, 'secrets'):
                return st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
            return os.getenv("GEMINI_API_KEY")
        return None
    
    def generate_static_image_video(
        self,
        gender: str,
        emotion: str,
        script: str,
        use_openai: bool = True
    ) -> Dict[str, any]:
        """
        OpenAI/Gemini API를 사용하여 정적 이미지를 생성하고 비디오로 변환
        (실제 talking head는 아니지만, 정적 이미지 + 텍스트 오버레이로 비디오 생성)
        
        Args:
            gender: "남자" 또는 "여자"
            emotion: "NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"
            script: 표시할 텍스트
            use_openai: True면 OpenAI DALL-E 사용, False면 Gemini 사용
        
        Returns:
            생성 결과 딕셔너리
        """
        try:
            # 프롬프트 생성
            emotion_kr = {
                "NEUTRAL": "중립적인",
                "HAPPY": "행복한, 미소짓는",
                "ANGRY": "화난, 불만스러운",
                "ASKING": "질문하는, 궁금해하는",
                "SAD": "슬픈, 우울한"
            }.get(emotion, "중립적인")
            
            gender_kr = "남성" if gender == "남자" else "여성"
            
            prompt = f"Professional {gender_kr} portrait, {emotion_kr} expression, business casual, clean background, high quality, realistic"
            
            if use_openai and OPENAI_AVAILABLE:
                client = OpenAI(api_key=self.api_key)
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                image_url = response.data[0].url
                
                return {
                    "success": True,
                    "image_url": image_url,
                    "method": "openai_dalle",
                    "note": "정적 이미지가 생성되었습니다. 실제 talking head 비디오를 원하시면 D-ID API를 사용하세요."
                }
            elif GEMINI_AVAILABLE:
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-pro-vision')
                # Gemini는 이미지 생성이 제한적이므로 텍스트만 반환
                return {
                    "success": False,
                    "error": "Gemini API는 이미지 생성 기능이 제한적입니다. DALL-E 또는 D-ID API를 사용하세요."
                }
            else:
                return {
                    "success": False,
                    "error": "필요한 라이브러리가 설치되지 않았습니다."
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"이미지 생성 중 오류: {str(e)}"
            }
    
    def generate_video_with_did(
        self, 
        image_url: str, 
        script: str, 
        voice_id: str = "en-US-JennyNeural",
        gender: str = "남자",
        emotion: str = "NEUTRAL"
    ) -> Dict[str, any]:
        """
        D-ID API를 사용하여 비디오 생성
        
        Args:
            image_url: 아바타 이미지 URL 또는 파일 경로
            script: 비디오에서 말할 텍스트
            voice_id: 음성 ID (기본값: 영어 여성)
            gender: 성별 ("남자" 또는 "여자")
            emotion: 감정 상태 ("NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD")
        
        Returns:
            {
                "success": bool,
                "video_url": str,
                "video_id": str,
                "status": str,
                "error": str
            }
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "D-ID API 키가 설정되지 않았습니다. Streamlit Secrets에 'D_ID_API_KEY'를 설정하거나 환경 변수로 설정하세요."
            }
        
        # D-ID API 엔드포인트
        create_url = "https://api.d-id.com/talks"
        
        # 헤더 설정 (D-ID API는 Bearer 토큰 사용)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 음성 설정 (성별에 따라)
        if gender == "남자":
            voice_id = "en-US-GuyNeural"  # 남성 음성
        else:
            voice_id = "en-US-JennyNeural"  # 여성 음성
        
        # 이미지가 URL이 아닌 경우 처리
        if not image_url.startswith("http"):
            # 로컬 파일인 경우 업로드 필요
            return {
                "success": False,
                "error": "이미지 URL이 필요합니다. 로컬 파일은 먼저 업로드해야 합니다."
            }
        
        # 요청 데이터
        payload = {
            "source_url": image_url,
            "script": {
                "type": "text",
                "input": script,
                "provider": {
                    "type": "microsoft",
                    "voice_id": voice_id
                },
                "ssml": "false"
            },
            "config": {
                "result_format": "mp4"
            }
        }
        
        try:
            # 비디오 생성 요청
            response = requests.post(create_url, json=payload, headers=headers)
            
            if response.status_code == 201:
                result = response.json()
                video_id = result.get("id")
                
                return {
                    "success": True,
                    "video_id": video_id,
                    "status": "created",
                    "message": "비디오 생성이 시작되었습니다. 잠시 후 다운로드할 수 있습니다."
                }
            else:
                return {
                    "success": False,
                    "error": f"API 오류: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"요청 중 오류 발생: {str(e)}"
            }
    
    def get_video_status(self, video_id: str) -> Dict[str, any]:
        """비디오 생성 상태 확인"""
        if not self.api_key:
            return {
                "success": False,
                "error": "API 키가 설정되지 않았습니다."
            }
        
        url = f"https://api.d-id.com/talks/{video_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "status": data.get("status"),
                    "video_url": data.get("result_url"),
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "error": f"상태 확인 실패: {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"오류 발생: {str(e)}"
            }
    
    def download_video(self, video_url: str, save_path: str) -> bool:
        """비디오 다운로드"""
        try:
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            return False
        except Exception as e:
            print(f"다운로드 오류: {str(e)}")
            return False


def generate_videos_batch(
    genders: list,
    emotions: list,
    scripts: Dict[str, str],
    image_urls: Dict[str, str],
    output_dir: str = "generated_videos"
) -> Dict[str, any]:
    """
    여러 비디오를 일괄 생성
    
    Args:
        genders: ["남자", "여자"]
        emotions: ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"]
        scripts: {emotion: script} 형식의 딕셔너리
        image_urls: {gender: image_url} 형식의 딕셔너리
        output_dir: 저장할 디렉토리
    
    Returns:
        생성 결과 딕셔너리
    """
    generator = VideoGenerator()
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for gender in genders:
        for emotion in emotions:
            key = f"{gender}_{emotion}"
            script = scripts.get(emotion, f"{emotion} 상태의 인사말입니다.")
            image_url = image_urls.get(gender)
            
            if not image_url:
                results[key] = {
                    "success": False,
                    "error": f"{gender}에 대한 이미지 URL이 없습니다."
                }
                continue
            
            # 비디오 생성
            result = generator.generate_video_with_did(
                image_url=image_url,
                script=script,
                gender=gender,
                emotion=emotion
            )
            
            results[key] = result
            
            # 비디오가 생성되면 다운로드
            if result.get("success") and result.get("video_id"):
                # 상태 확인 및 다운로드 (폴링)
                max_attempts = 30
                for attempt in range(max_attempts):
                    time.sleep(2)  # 2초 대기
                    status_result = generator.get_video_status(result["video_id"])
                    
                    if status_result.get("status") == "done":
                        video_url = status_result.get("video_url")
                        if video_url:
                            save_path = output_path / f"{gender}_{emotion}.mp4"
                            if generator.download_video(video_url, str(save_path)):
                                results[key]["video_path"] = str(save_path)
                                results[key]["status"] = "completed"
                            break
                    elif status_result.get("status") == "error":
                        results[key]["error"] = "비디오 생성 실패"
                        break
            
            # API 제한을 피하기 위해 대기
            time.sleep(1)
    
    return results

