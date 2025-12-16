# Git 작업 가이드 (상황별 해결 방법)

## 📋 목차
1. [일반적인 Push 실패 상황](#일반적인-push-실패-상황)
2. [Merge Conflict 해결](#merge-conflict-해결)
3. [Rebase 작업](#rebase-작업)
4. [Conflict 마커 제거](#conflict-마커-제거)
5. [상황별 명령어 체크리스트](#상황별-명령어-체크리스트)

---

## 일반적인 Push 실패 상황

### 상황 1: `! [rejected] main -> main (non-fast-forward)`

**원인**: 원격 저장소에 로컬에 없는 커밋이 있음

**해결 방법 A: Pull 후 Push (권장)**
```bash
git pull origin main
# 충돌이 없으면 자동으로 merge됨
git push origin main
```

**해결 방법 B: 원격 변경사항 무시하고 강제 Push (주의!)**
```bash
git push origin main --force-with-lease
# 또는
git push origin main --force
```

---

## Merge Conflict 해결

### 상황 2: `CONFLICT (content): Merge conflict in streamlit_app.py`

**해결 방법 A: 로컬 버전 유지 (--ours)**
```bash
git checkout --ours streamlit_app.py
git add streamlit_app.py
git commit -m "Resolve conflict - keep local version"
git push origin main
```

**해결 방법 B: 원격 버전으로 통일 (--theirs)**
```bash
git checkout --theirs streamlit_app.py
git add streamlit_app.py
git commit -m "Resolve conflict - use remote version"
git push origin main
```

**해결 방법 C: 수동으로 해결**
```bash
# 1. 파일 열어서 conflict 마커 제거
# <<<<<<< HEAD
# 로컬 코드
# =======
# 원격 코드
# >>>>>>> commit_hash

# 2. 원하는 코드만 남기고 마커 제거

# 3. 저장 후
git add streamlit_app.py
git commit -m "Resolve conflict manually"
git push origin main
```

**주의사항: Rebase 중에는 --ours/--theirs 의미가 반대!**
- 일반 merge: `--ours` = 로컬, `--theirs` = 원격
- Rebase 중: `--ours` = 원격, `--theirs` = 로컬 (반대!)

---

## Rebase 작업

### 상황 3: Interactive Rebase 진행 중

**Rebase 시작**
```bash
git rebase origin/main
# 또는
git rebase -i HEAD~5  # 최근 5개 커밋 rebase
```

**Rebase 중 Conflict 발생**
```bash
# 1. Conflict 해결
git checkout --ours streamlit_app.py  # 또는 --theirs
# 또는 수동으로 해결

# 2. 해결 완료 표시
git add streamlit_app.py

# 3. Rebase 계속
git rebase --continue
```

**Rebase 중단**
```bash
git rebase --abort  # rebase 시작 전 상태로 돌아감
```

**Rebase 완료 후 Push**
```bash
git push origin main --force-with-lease
```

---

## Conflict 마커 제거

### 상황 4: Conflict 마커가 파일에 남아있음

**자동 제거 스크립트 사용**
```bash
# remove_conflicts.py 실행
python remove_conflicts.py

# 또는 fix_merge_conflicts.py 실행
python fix_merge_conflicts.py

# 그 다음
git add streamlit_app.py
git commit -m "Remove conflict markers"
```

**수동 제거**
```bash
# Conflict 마커 찾기
grep -n "<<<<<<< HEAD" streamlit_app.py
grep -n "=======" streamlit_app.py
grep -n ">>>>>>>" streamlit_app.py

# 파일 열어서 수동으로 제거
# <<<<<<< HEAD
# =======
# >>>>>>> commit_hash
# 이 부분들을 모두 제거
```

---

## 상황별 명령어 체크리스트

### ✅ 시나리오 1: 일반적인 Push 실패

```bash
# 1. 상태 확인
git status

# 2. 원격 변경사항 가져오기
git fetch origin

# 3. Pull (merge)
git pull origin main

# 4. 충돌 있으면 해결
git add .
git commit -m "Merge remote changes"

# 5. Push
git push origin main
```

### ✅ 시나리오 2: Merge Conflict 발생

```bash
# 1. Conflict 파일 확인
git status

# 2. 로컬 버전 유지
git checkout --ours streamlit_app.py
git checkout --ours requirements.txt

# 3. 또는 원격 버전으로 통일
git checkout --theirs streamlit_app.py
git checkout --theirs requirements.txt

# 4. 해결 완료 표시
git add streamlit_app.py requirements.txt

# 5. Commit
git commit -m "Resolve merge conflicts"

# 6. Push
git push origin main
```

### ✅ 시나리오 3: Rebase 중 Conflict

```bash
# 1. Rebase 시작
git rebase origin/main

# 2. Conflict 발생 시 해결
# Rebase 중에는 --ours/--theirs 의미가 반대!
git checkout --ours streamlit_app.py  # 원격 버전
# 또는
git checkout --theirs streamlit_app.py  # 로컬 버전

# 3. 해결 완료
git add streamlit_app.py

# 4. Rebase 계속
git rebase --continue

# 5. 모든 conflict 해결 후
git push origin main --force-with-lease
```

### ✅ 시나리오 4: Conflict 마커가 남아있음

```bash
# 1. Conflict 마커 확인
grep -n "<<<<<<< HEAD" streamlit_app.py

# 2. 자동 제거 스크립트 실행
python remove_conflicts.py

# 3. 또는 수동으로 제거 후
git add streamlit_app.py

# 4. Commit
git commit -m "Remove conflict markers"

# 5. Push
git push origin main
```

### ✅ 시나리오 5: Rebase 중단하고 일반 Merge로 전환

```bash
# 1. Rebase 중단
git rebase --abort

# 2. 일반 Merge
git pull origin main --no-rebase

# 3. Conflict 해결
git checkout --ours streamlit_app.py
git add streamlit_app.py

# 4. Commit
git commit -m "Merge origin/main"

# 5. Push
git push origin main
```

---

## 🚨 주의사항

### --ours vs --theirs 의미

**일반 Merge 상황:**
- `--ours`: 현재 브랜치 (로컬)
- `--theirs`: 병합하려는 브랜치 (원격)

**Rebase 상황 (반대!):**
- `--ours`: 원격 브랜치 (onto 브랜치)
- `--theirs`: 현재 rebase 중인 커밋 (로컬 변경사항)

### Force Push 주의

```bash
# 안전한 Force Push
git push origin main --force-with-lease

# 위험한 Force Push (절대 사용 금지!)
git push origin main --force
```

`--force-with-lease`는 원격에 예상치 못한 변경이 있으면 실패하므로 더 안전합니다.

---

## 📝 빠른 참조표

| 상황 | 명령어 |
|------|--------|
| Push 실패 | `git pull origin main` → `git push origin main` |
| Conflict 해결 (로컬 유지) | `git checkout --ours 파일명` → `git add` → `git commit` |
| Conflict 해결 (원격 유지) | `git checkout --theirs 파일명` → `git add` → `git commit` |
| Rebase 계속 | `git add 파일명` → `git rebase --continue` |
| Rebase 중단 | `git rebase --abort` |
| Conflict 마커 제거 | `python remove_conflicts.py` → `git add` → `git commit` |
| 안전한 Force Push | `git push origin main --force-with-lease` |

---

## 💡 팁

1. **항상 상태 확인 먼저**: `git status`로 현재 상황 파악
2. **충돌 해결 전 백업**: 중요한 변경사항은 미리 백업
3. **작은 단위로 Commit**: 큰 변경사항은 여러 커밋으로 나누기
4. **Rebase보다 Merge가 안전**: Rebase는 히스토리를 재작성하므로 주의
5. **Force Push는 팀과 협의 후**: 다른 사람이 작업 중이면 문제 발생 가능

---

## 🔧 유용한 스크립트

### remove_conflicts.py
```python
"""간단한 conflict 마커 제거 스크립트"""
import re

file_path = "streamlit_app.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 모든 conflict 마커 제거 (HEAD 버전 유지)
pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n.*?\n>>>>>>> [^\n]+\n'
content = re.sub(pattern, r'\1\n', content, flags=re.DOTALL)

# 남은 단독 마커들 제거
content = re.sub(r'<<<<<<< HEAD\n', '', content)
content = re.sub(r'=======\n', '', content)
content = re.sub(r'>>>>>>> [^\n]+\n', '', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ 충돌 마커 제거 완료!")
```

---

**마지막 업데이트**: 2025-12-10






















