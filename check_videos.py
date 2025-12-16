"""
비디오 파일 검색 진단 도구
폴더에 어떤 파일들이 있는지 확인하는 스크립트
"""

import sys
from pathlib import Path

def check_videos(folder_path):
    """폴더 내 비디오 파일 확인"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ 오류: 폴더가 존재하지 않습니다: {folder_path}")
        return
    
    print(f"📁 검색 폴더: {folder_path}")
    print(f"📁 절대 경로: {folder.absolute()}")
    print("=" * 60)
    
    # 모든 파일 목록
    all_files = list(folder.iterdir())
    print(f"\n📋 폴더 내 모든 항목: {len(all_files)}개\n")
    
    # 비디오 확장자
    video_extensions = [".mp4", ".webm", ".MP4", ".WEBM", ".avi", ".mov", ".mkv"]
    
    # 파일 타입별 분류
    video_files = []
    subdirs = []
    other_files = []
    
    for item in all_files:
        if item.is_file():
            ext = item.suffix.lower()
            if ext in [e.lower() for e in video_extensions]:
                video_files.append(item)
            else:
                other_files.append(item)
        elif item.is_dir():
            subdirs.append(item)
    
    # 결과 출력
    print("🎬 비디오 파일:")
    if video_files:
        for idx, video in enumerate(video_files, 1):
            print(f"  {idx}. {video.name} ({video.suffix})")
    else:
        print("  (비디오 파일 없음)")
    
    print(f"\n📁 하위 폴더: {len(subdirs)}개")
    if subdirs:
        for idx, subdir in enumerate(subdirs[:10], 1):
            print(f"  {idx}. {subdir.name}/")
        if len(subdirs) > 10:
            print(f"  ... 외 {len(subdirs) - 10}개 폴더")
    
    print(f"\n📄 기타 파일: {len(other_files)}개")
    if other_files:
        print("  (처음 10개만 표시)")
        for idx, file in enumerate(other_files[:10], 1):
            print(f"  {idx}. {file.name} ({file.suffix})")
        if len(other_files) > 10:
            print(f"  ... 외 {len(other_files) - 10}개 파일")
    
    # 하위 폴더에서도 비디오 파일 검색
    print("\n" + "=" * 60)
    print("🔍 하위 폴더에서 비디오 파일 검색 중...\n")
    
    subdir_videos = []
    for ext in video_extensions:
        found = list(folder.rglob(f"*{ext}"))
        subdir_videos.extend(found)
    
    # 중복 제거
    subdir_videos = list(set(subdir_videos))
    
    if subdir_videos:
        print(f"✓ 하위 폴더에서 {len(subdir_videos)}개의 비디오 파일 발견:")
        for idx, video in enumerate(subdir_videos[:20], 1):
            rel_path = video.relative_to(folder)
            print(f"  {idx}. {rel_path}")
        if len(subdir_videos) > 20:
            print(f"  ... 외 {len(subdir_videos) - 20}개 파일")
    else:
        print("  (하위 폴더에 비디오 파일 없음)")
    
    # 총계
    total_videos = len(video_files) + len([v for v in subdir_videos if v not in video_files])
    print("\n" + "=" * 60)
    print(f"📊 총 비디오 파일: {total_videos}개")
    print("=" * 60)
    
    # 권장 사항
    if total_videos == 0:
        print("\n💡 권장 사항:")
        print("  1. 비디오 파일이 실제로 해당 폴더에 있는지 확인하세요")
        print("  2. 파일 확장자가 .mp4 또는 .webm인지 확인하세요")
        print("  3. 파일명에 한글이나 특수문자가 있는지 확인하세요")
        print("  4. 다른 폴더를 선택해보세요")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("확인할 폴더 경로를 입력하세요: ").strip().strip('"')
    
    check_videos(folder_path)
    print("\n아무 키나 누르면 종료됩니다...")
    input()

























