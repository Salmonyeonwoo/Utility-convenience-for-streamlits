"""
비디오 파일 구조화 스크립트 사용 예제
"""

from organize_videos import VideoOrganizer

# 예제 1: 기본 사용법
def example_basic():
    """기본 사용법 - 소스 디렉토리만 지정"""
    source_dir = "C:/videos"  # 비디오 파일이 있는 폴더
    
    organizer = VideoOrganizer(source_dir)
    stats = organizer.organize_videos()
    organizer.print_statistics()


# 예제 2: 타겟 디렉토리 지정
def example_custom_target():
    """타겟 디렉토리 지정"""
    source_dir = "C:/videos"
    target_dir = "C:/organized_videos"  # 정리된 파일을 저장할 폴더
    
    organizer = VideoOrganizer(source_dir, target_dir)
    stats = organizer.organize_videos()
    organizer.print_statistics()


# 예제 3: 대화형 모드
def example_interactive():
    """대화형 모드 - 파일명 파싱 실패 시 사용자 입력 요청"""
    source_dir = "C:/videos"
    
    organizer = VideoOrganizer(source_dir)
    stats = organizer.organize_videos(interactive=True)
    organizer.print_statistics()


# 예제 4: 리포트 생성
def example_with_report():
    """리포트 생성과 함께 사용"""
    source_dir = "C:/videos"
    
    organizer = VideoOrganizer(source_dir)
    stats = organizer.organize_videos()
    organizer.print_statistics()
    
    # 리포트 생성
    report = organizer.generate_structure_report()
    with open("video_structure_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n리포트가 'video_structure_report.md'에 저장되었습니다.")


if __name__ == "__main__":
    # 사용할 예제를 선택하세요
    print("비디오 파일 구조화 예제")
    print("=" * 50)
    print("1. 기본 사용법")
    print("2. 타겟 디렉토리 지정")
    print("3. 대화형 모드")
    print("4. 리포트 생성")
    
    choice = input("\n예제 번호를 선택하세요 (1-4): ").strip()
    
    if choice == "1":
        example_basic()
    elif choice == "2":
        example_custom_target()
    elif choice == "3":
        example_interactive()
    elif choice == "4":
        example_with_report()
    else:
        print("잘못된 선택입니다.")










































