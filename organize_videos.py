"""
비디오 파일 구조화 스크립트
성별(남자/여자)과 감정(NEUTRAL, HAPPY, ANGRY, ASKING, SAD)에 따라 비디오 파일을 정리합니다.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# 설정
GENDERS = ["남자", "여자", "male", "female", "man", "woman"]
EMOTIONS = {
    "NEUTRAL": ["neutral", "중립", "평상", "normal"],
    "HAPPY": ["happy", "행복", "기쁨", "joy", "smile"],
    "ANGRY": ["angry", "화남", "분노", "anger", "mad"],
    "ASKING": ["asking", "질문", "question", "inquire"],
    "SAD": ["sad", "슬픔", "슬프", "sorrow", "cry"]
}
VIDEO_EXTENSIONS = [".mp4", ".webm", ".MP4", ".WEBM"]


class VideoOrganizer:
    """비디오 파일을 성별과 감정별로 구조화하는 클래스"""
    
    def __init__(self, source_dir: str, target_dir: str = None):
        """
        Args:
            source_dir: 비디오 파일이 있는 소스 디렉토리
            target_dir: 정리된 파일을 저장할 타겟 디렉토리 (None이면 source_dir 내에 'organized' 폴더 생성)
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir) if target_dir else self.source_dir / "organized"
        self.mapping_file = self.target_dir / "file_mapping.json"
        self.stats = {
            "total_files": 0,
            "organized": 0,
            "failed": 0,
            "by_gender": {"남자": 0, "여자": 0},
            "by_emotion": {emotion: 0 for emotion in EMOTIONS.keys()}
        }
        
    def normalize_gender(self, text: str) -> Optional[str]:
        """텍스트에서 성별을 추출하고 정규화"""
        text_lower = text.lower()
        if any(gender.lower() in text_lower for gender in ["남자", "male", "man"]):
            return "남자"
        elif any(gender.lower() in text_lower for gender in ["여자", "female", "woman"]):
            return "여자"
        return None
    
    def normalize_emotion(self, text: str) -> Optional[str]:
        """텍스트에서 감정을 추출하고 정규화"""
        text_lower = text.lower()
        for emotion, keywords in EMOTIONS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                return emotion
        return None
    
    def parse_filename(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """파일명에서 성별과 감정을 파싱"""
        # 파일명에서 확장자 제거
        name_without_ext = Path(filename).stem
        
        # 성별 추출
        gender = self.normalize_gender(name_without_ext)
        
        # 감정 추출
        emotion = self.normalize_emotion(name_without_ext)
        
        return gender, emotion
    
    def create_directory_structure(self):
        """디렉토리 구조 생성"""
        for gender in ["남자", "여자"]:
            for emotion in EMOTIONS.keys():
                dir_path = self.target_dir / gender / emotion
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def organize_videos(self, interactive: bool = False) -> Dict:
        """
        비디오 파일들을 구조화
        
        Args:
            interactive: True면 파일명에서 파싱 실패 시 사용자에게 입력 요청
        
        Returns:
            통계 정보 딕셔너리
        """
        # 타겟 디렉토리 생성
        self.create_directory_structure()
        
        # 기존 매핑 파일 로드
        file_mapping = {}
        if self.mapping_file.exists():
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                file_mapping = json.load(f)
        
        # 비디오 파일 찾기 (개선된 검색 로직)
        video_files = []
        
        # 디렉토리 존재 확인
        if not self.source_dir.exists():
            print(f"❌ 오류: 소스 디렉토리가 존재하지 않습니다: {self.source_dir}")
            return self.stats
        
        print(f"📁 소스 디렉토리: {self.source_dir}")
        print(f"🔍 비디오 파일 검색 중... (확장자: {', '.join(VIDEO_EXTENSIONS)})\n")
        
        # 현재 디렉토리의 파일 검색
        for ext in VIDEO_EXTENSIONS:
            found = list(self.source_dir.glob(f"*{ext}"))
            video_files.extend(found)
            if found:
                print(f"  ✓ 현재 폴더에서 {len(found)}개 발견 (*{ext})")
        
        # 하위 디렉토리 재귀 검색
        for ext in VIDEO_EXTENSIONS:
            found = list(self.source_dir.rglob(f"*{ext}"))
            # 중복 제거 (이미 추가된 파일 제외)
            new_files = [f for f in found if f not in video_files]
            video_files.extend(new_files)
            if new_files:
                print(f"  ✓ 하위 폴더에서 {len(new_files)}개 발견 (*{ext})")
        
        # 중복 제거
        video_files = list(set(video_files))
        
        self.stats["total_files"] = len(video_files)
        
        print(f"\n📊 총 {len(video_files)}개의 비디오 파일을 찾았습니다.\n")
        
        # 파일이 없을 경우 안내
        if len(video_files) == 0:
            print("⚠️  비디오 파일을 찾을 수 없습니다.")
            print(f"   확인 사항:")
            print(f"   1. 폴더 경로가 올바른지 확인: {self.source_dir}")
            print(f"   2. 파일 확장자가 .mp4 또는 .webm인지 확인")
            print(f"   3. 파일이 실제로 존재하는지 확인\n")
            return self.stats
        
        # 찾은 파일 목록 출력 (처음 10개만)
        print("찾은 파일 목록:")
        for idx, video_file in enumerate(video_files[:10], 1):
            print(f"  {idx}. {video_file.name}")
        if len(video_files) > 10:
            print(f"  ... 외 {len(video_files) - 10}개 파일\n")
        else:
            print()
        
        for video_file in video_files:
            try:
                # 파일명에서 성별과 감정 파싱
                gender, emotion = self.parse_filename(video_file.name)
                
                # 파싱 실패 시 처리
                if not gender or not emotion:
                    if interactive:
                        print(f"\n파일명에서 정보를 추출할 수 없습니다: {video_file.name}")
                        print("수동으로 입력해주세요.")
                        
                        if not gender:
                            gender_input = input("성별 (남자/여자): ").strip()
                            gender = self.normalize_gender(gender_input) or "남자"
                        
                        if not emotion:
                            print("감정 선택:")
                            for idx, emo in enumerate(EMOTIONS.keys(), 1):
                                print(f"  {idx}. {emo}")
                            emotion_input = input("감정 번호 또는 이름: ").strip()
                            if emotion_input.isdigit():
                                emotion = list(EMOTIONS.keys())[int(emotion_input) - 1]
                            else:
                                emotion = self.normalize_emotion(emotion_input) or "NEUTRAL"
                    else:
                        # 비대화형 모드: 기본값 사용
                        gender = gender or "남자"
                        emotion = emotion or "NEUTRAL"
                        print(f"⚠️  {video_file.name}: 파싱 실패, 기본값 사용 (성별: {gender}, 감정: {emotion})")
                
                # 타겟 경로 생성
                target_path = self.target_dir / gender / emotion / video_file.name
                
                # 파일 복사 (이미 존재하면 건너뛰기)
                if target_path.exists():
                    print(f"⏭️  건너뜀 (이미 존재): {video_file.name} -> {gender}/{emotion}/")
                else:
                    shutil.copy2(video_file, target_path)
                    print(f"✅ 복사 완료: {video_file.name} -> {gender}/{emotion}/")
                
                # 통계 업데이트
                self.stats["organized"] += 1
                self.stats["by_gender"][gender] += 1
                self.stats["by_emotion"][emotion] += 1
                
                # 매핑 저장
                file_mapping[str(video_file)] = {
                    "gender": gender,
                    "emotion": emotion,
                    "target_path": str(target_path)
                }
                
            except Exception as e:
                print(f"❌ 오류 발생 ({video_file.name}): {str(e)}")
                self.stats["failed"] += 1
        
        # 매핑 파일 저장
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(file_mapping, f, ensure_ascii=False, indent=2)
        
        return self.stats
    
    def print_statistics(self):
        """통계 정보 출력"""
        print("\n" + "="*50)
        print("📊 정리 통계")
        print("="*50)
        print(f"총 파일 수: {self.stats['total_files']}")
        print(f"정리 완료: {self.stats['organized']}")
        print(f"실패: {self.stats['failed']}")
        
        print("\n성별별 분포:")
        for gender, count in self.stats['by_gender'].items():
            print(f"  {gender}: {count}개")
        
        print("\n감정별 분포:")
        for emotion, count in self.stats['by_emotion'].items():
            print(f"  {emotion}: {count}개")
        
        print("\n" + "="*50)
        print(f"정리된 파일 위치: {self.target_dir}")
        print("="*50)
    
    def generate_structure_report(self) -> str:
        """구조 리포트 생성"""
        report = []
        report.append("# 비디오 파일 구조 리포트\n")
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"소스 디렉토리: {self.source_dir}\n")
        report.append(f"타겟 디렉토리: {self.target_dir}\n\n")
        
        report.append("## 디렉토리 구조\n")
        report.append("```\n")
        report.append("organized/\n")
        for gender in ["남자", "여자"]:
            report.append(f"  {gender}/\n")
            for emotion in EMOTIONS.keys():
                emotion_dir = self.target_dir / gender / emotion
                file_count = len(list(emotion_dir.glob("*"))) if emotion_dir.exists() else 0
                report.append(f"    {emotion}/ ({file_count}개 파일)\n")
        report.append("```\n\n")
        
        report.append("## 통계\n")
        report.append(f"- 총 파일 수: {self.stats['total_files']}\n")
        report.append(f"- 정리 완료: {self.stats['organized']}\n")
        report.append(f"- 실패: {self.stats['failed']}\n")
        
        return "".join(report)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="비디오 파일을 성별과 감정별로 구조화")
    parser.add_argument("source_dir", help="비디오 파일이 있는 소스 디렉토리")
    parser.add_argument("-t", "--target", help="타겟 디렉토리 (기본값: source_dir/organized)")
    parser.add_argument("-i", "--interactive", action="store_true", 
                       help="대화형 모드 (파일명 파싱 실패 시 사용자 입력 요청)")
    parser.add_argument("-r", "--report", help="리포트 파일 저장 경로")
    
    args = parser.parse_args()
    
    # VideoOrganizer 생성
    organizer = VideoOrganizer(args.source_dir, args.target)
    
    # 비디오 정리 실행
    print("비디오 파일 구조화를 시작합니다...")
    stats = organizer.organize_videos(interactive=args.interactive)
    
    # 통계 출력
    organizer.print_statistics()
    
    # 리포트 생성
    if args.report:
        report = organizer.generate_structure_report()
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n리포트 저장 완료: {args.report}")


if __name__ == "__main__":
    main()

