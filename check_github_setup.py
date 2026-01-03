# -*- coding: utf-8 -*-
"""
GitHub 실행 환경 체크 스크립트
"""
import os
import sys

print("=" * 60)
print("GitHub 실행 환경 체크")
print("=" * 60)
print()

# 1. 필수 파일 확인
print("1. 필수 파일 확인:")
required_files = [
    "streamlit_app.py",
    "requirements.txt",
    ".gitignore",
    ".streamlit/config.toml"
]

for file in required_files:
    exists = os.path.exists(file)
    status = "[OK]" if exists else "[X]"
    print(f"   {status} {file}")
    
    if not exists and file == ".streamlit/config.toml":
        print("      [WARN] config.toml이 없습니다. Streamlit Cloud에서는 기본 설정으로 동작합니다.")
        print("            (로컬에서만 필요하며 GitHub에 올리지 않아도 됩니다)")

print()

# 2. .gitignore 확인
print("2. .gitignore 확인:")
gitignore_path = ".gitignore"
if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if ".streamlit/config.toml" in content:
            print("   [OK] .streamlit/config.toml이 .gitignore에 포함되어 있습니다.")
        else:
            print("   [WARN] .streamlit/config.toml이 .gitignore에 없습니다.")
            print("          (로컬 설정이므로 추가하는 것을 권장합니다)")
        if ".streamlit/secrets.toml" in content:
            print("   [OK] .streamlit/secrets.toml이 .gitignore에 포함되어 있습니다.")
else:
    print("   [X] .gitignore 파일이 없습니다!")

print()

# 3. Streamlit 설정 확인
print("3. Streamlit 설정 확인:")
try:
    import streamlit as st
    print(f"   [OK] Streamlit 설치됨 (버전: {st.__version__})")
except ImportError:
    print("   [X] Streamlit이 설치되지 않았습니다!")
    print("       설치: pip install streamlit")

print()

# 4. config.toml 내용 확인
print("4. config.toml 내용 확인:")
config_path = ".streamlit/config.toml"
if os.path.exists(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
        if "port = 8501" in config_content:
            print("   [OK] config.toml이 정상적으로 설정되어 있습니다.")
        else:
            print("   [WARN] config.toml 내용을 확인하세요.")
else:
    print("   [INFO] config.toml이 없습니다 (기본 설정 사용)")

print()

# 5. GitHub/Streamlit Cloud 배포 확인
print("5. 배포 환경 확인:")
print("   [INFO] Streamlit Cloud:")
print("          - config.toml은 선택 사항입니다 (없어도 기본값으로 동작)")
print("          - secrets.toml은 Streamlit Cloud 대시보드에서 설정해야 합니다")
print("          - requirements.txt가 필요합니다")
print()
print("   [INFO] GitHub:")
print("          - config.toml은 로컬 설정이므로 .gitignore에 포함 권장")
print("          - secrets.toml은 보안을 위해 반드시 .gitignore에 포함")

print()
print("=" * 60)
print("체크 완료!")
print("=" * 60)
print()
print("결론:")
print("- config.toml: 로컬에서만 사용 (GitHub에 올리지 않아도 됨)")
print("- Streamlit Cloud: config.toml 없이도 정상 동작")
print("- secrets.toml: 반드시 .gitignore에 포함되어야 함")

