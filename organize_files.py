"""
파일 타입별로 폴더를 생성하고 파일을 이동하는 스크립트
사용법: python organize_files.py
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List


def organize_files_by_type(source_dir: str, file_types: Dict[str, List[str]]) -> None:
    """
    파일 타입별로 폴더를 생성하고 파일을 이동합니다.
    
    Args:
        source_dir: 파일이 있는 소스 디렉토리
        file_types: {폴더명: [확장자 리스트]} 형식의 딕셔너리
                    예: {"docx": ["docx"], "images": ["png", "jpg", "jpeg"]}
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ 오류: 디렉토리를 찾을 수 없습니다: {source_dir}")
        return
    
    print(f"📁 작업 디렉토리: {source_path.absolute()}\n")
    
    # 폴더 생성
    print("=== 1단계: 폴더 생성 ===")
    for folder_name in file_types.keys():
        folder_path = source_path / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"✓ 폴더 확인: {folder_name}/")
    
    # 파일 이동
    print("\n=== 2단계: 파일 이동 ===")
    moved_count = {folder: 0 for folder in file_types.keys()}
    errors = []
    
    # 모든 파일 검사
    for file_path in source_path.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower().lstrip('.')
            
            # 해당 확장자의 폴더 찾기
            moved = False
            for folder_name, extensions in file_types.items():
                if file_ext in extensions:
                    try:
                        dest_path = source_path / folder_name / file_path.name
                        # 중복 파일 처리: 이름 변경
                        if dest_path.exists():
                            base_name = file_path.stem
                            counter = 1
                            while dest_path.exists():
                                new_name = f"{base_name}_{counter}{file_path.suffix}"
                                dest_path = source_path / folder_name / new_name
                                counter += 1
                        
                        shutil.move(str(file_path), str(dest_path))
                        moved_count[folder_name] += 1
                        moved = True
                        break
                    except Exception as e:
                        errors.append(f"{file_path.name}: {str(e)}")
            
            if not moved and file_ext:
                print(f"⚠️  처리되지 않은 확장자: {file_ext} ({file_path.name})")
    
    # 결과 출력
    print("\n=== 3단계: 이동 결과 ===")
    total = 0
    for folder, count in moved_count.items():
        if count > 0:
            print(f"  {folder}/: {count}개 파일 이동")
            total += count
    
    if total == 0:
        print("  이동할 파일이 없습니다.")
    else:
        print(f"\n✅ 총 {total}개 파일 이동 완료!")
    
    if errors:
        print("\n❌ 오류 발생:")
        for error in errors:
            print(f"  {error}")


def main():
    """메인 함수"""
    # 설정: 소스 디렉토리 경로
    source_directory = r"C:\Users\Admin\Downloads\Updated_streamlit_app_files\local_db"
    
    # 설정: 파일 타입 매핑 (폴더명: [확장자 리스트])
    file_type_mapping = {
        "docx": ["docx"],
        "pptx": ["pptx"],
        "json": ["json"],
        "pdf": ["pdf"],
        "csv": ["csv"],
        "txt": ["txt"],
        "xlsx": ["xlsx", "xls"],
    }
    
    # 사용자 확인
    print("=" * 60)
    print("파일 정리 스크립트")
    print("=" * 60)
    print(f"\n대상 디렉토리: {source_directory}")
    print("\n정리할 파일 타입:")
    for folder, exts in file_type_mapping.items():
        print(f"  {folder}/: {', '.join(exts)}")
    
    response = input("\n계속하시겠습니까? (Y/N): ").strip().upper()
    if response != "Y":
        print("취소되었습니다.")
        return
    
    # 파일 정리 실행
    organize_files_by_type(source_directory, file_type_mapping)
    
    print("\n" + "=" * 60)
    print("작업 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
