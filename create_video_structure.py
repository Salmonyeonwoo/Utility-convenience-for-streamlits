"""
비디오 파일 구조 준비 스크립트
비디오 파일을 넣을 폴더 구조를 생성하고 가이드를 제공합니다.
"""

import os
from pathlib import Path

def create_video_structure(base_path: str = None):
    """비디오 파일을 넣을 폴더 구조 생성"""
    if base_path is None:
        base_path = Path(__file__).parent / "videos_to_organize"
    else:
        base_path = Path(base_path)
    
    base_path.mkdir(exist_ok=True)
    
    print(f"📁 비디오 파일 준비 폴더 생성: {base_path.absolute()}\n")
    print("=" * 60)
    print("📝 필요한 비디오 파일 목록")
    print("=" * 60)
    print("\n다음 파일들을 이 폴더에 준비하세요:\n")
    
    genders = ["남자", "여자"]
    emotions = ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"]
    
    file_list = []
    for gender in genders:
        for emotion in emotions:
            filename = f"{gender}_{emotion}.mp4"
            file_list.append(filename)
            print(f"  ✓ {filename}")
    
    print(f"\n총 {len(file_list)}개의 비디오 파일이 필요합니다.")
    print("\n" + "=" * 60)
    print("💡 파일명 규칙")
    print("=" * 60)
    print("""
파일명에 성별과 감정 키워드를 포함하면 자동으로 인식됩니다.

성별 키워드:
  - 남자: "남자", "male", "man"
  - 여자: "여자", "female", "woman"

감정 키워드:
  - NEUTRAL: "neutral", "중립", "평상", "normal"
  - HAPPY: "happy", "행복", "기쁨", "joy", "smile"
  - ANGRY: "angry", "화남", "분노", "anger", "mad"
  - ASKING: "asking", "질문", "question", "inquire"
  - SAD: "sad", "슬픔", "슬프", "sorrow", "cry"

예시 파일명:
  ✓ 남자_happy.mp4
  ✓ female_sad.webm
  ✓ 여자_질문.mp4
  ✓ man_angry_video.mp4
    """)
    
    print("=" * 60)
    print("📂 생성된 폴더")
    print("=" * 60)
    print(f"  {base_path.absolute()}\n")
    print("이 폴더에 비디오 파일을 넣은 후,")
    print("organize_videos_gui.bat를 실행하여 이 폴더를 선택하세요!")
    print("=" * 60)
    
    # README 파일 생성
    readme_path = base_path / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("비디오 파일 준비 가이드\n")
        f.write("=" * 60 + "\n\n")
        f.write("이 폴더에 다음 비디오 파일들을 준비하세요:\n\n")
        for filename in file_list:
            f.write(f"  - {filename}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("파일명 규칙:\n")
        f.write("- 파일명에 성별과 감정 키워드를 포함하세요\n")
        f.write("- 예: 남자_happy.mp4, female_sad.webm\n")
        f.write("\n준비가 완료되면 organize_videos_gui.bat를 실행하세요!\n")
    
    print(f"\n✓ 가이드 파일 생성: {readme_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # 현재 스크립트 위치 기준으로 videos_to_organize 폴더 생성
        folder_path = None
    
    create_video_structure(folder_path)
    print("\n아무 키나 누르면 종료됩니다...")
    input()














