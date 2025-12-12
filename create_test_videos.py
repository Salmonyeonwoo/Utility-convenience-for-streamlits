"""
테스트용 더미 비디오 파일 생성 스크립트
실제 비디오 파일이 없을 때 테스트를 위한 더미 파일을 생성합니다.
"""

import os
from pathlib import Path

def create_dummy_video(filename: str, size_kb: int = 100):
    """더미 비디오 파일 생성 (실제 비디오가 아닌 더미 파일)"""
    # 간단한 더미 데이터로 파일 생성
    dummy_data = b'\x00' * (size_kb * 1024)
    
    with open(filename, 'wb') as f:
        # MP4 파일 시그니처 추가 (최소한의 유효한 MP4 헤더)
        f.write(b'ftyp')
        f.write(b'mp41')
        f.write(dummy_data)
    
    print(f"  ✓ 생성: {Path(filename).name}")


def create_test_videos(target_folder: str = None):
    """테스트용 비디오 파일 생성"""
    if target_folder is None:
        target_folder = Path(__file__).parent / "videos_to_organize"
    else:
        target_folder = Path(target_folder)
    
    target_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 테스트 비디오 파일 생성 위치: {target_folder.absolute()}\n")
    print("=" * 60)
    print("⚠️  주의: 이것은 더미 파일입니다!")
    print("=" * 60)
    print("실제 비디오 파일이 아닌 테스트용 파일입니다.")
    print("실제 사용을 위해서는 진짜 비디오 파일(.mp4, .webm)이 필요합니다.\n")
    
    # 필요한 파일 목록
    genders = ["남자", "여자"]
    emotions = {
        "NEUTRAL": ["neutral", "중립", "평상"],
        "HAPPY": ["happy", "행복", "기쁨"],
        "ANGRY": ["angry", "화남", "분노"],
        "ASKING": ["asking", "질문"],
        "SAD": ["sad", "슬픔"]
    }
    
    print("생성 중...\n")
    
    created_count = 0
    for gender in genders:
        for emotion, keywords in emotions.items():
            # 여러 파일명 패턴으로 생성
            filename1 = target_folder / f"{gender}_{keywords[0]}.mp4"
            create_dummy_video(filename1)
            created_count += 1
            
            # 영어 버전도 생성
            if gender == "남자":
                filename2 = target_folder / f"male_{keywords[0]}.mp4"
            else:
                filename2 = target_folder / f"female_{keywords[0]}.mp4"
            create_dummy_video(filename2)
            created_count += 1
    
    print(f"\n✅ 총 {created_count}개의 테스트 파일이 생성되었습니다.")
    print(f"\n이제 organize_videos_gui.bat를 실행하여")
    print(f"'{target_folder.name}' 폴더를 선택하세요!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = None
    
    create_test_videos(folder_path)
    print("\n아무 키나 누르면 종료됩니다...")
    input()














