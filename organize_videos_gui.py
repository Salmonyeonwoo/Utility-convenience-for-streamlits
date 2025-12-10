"""
비디오 파일 구조화 GUI 버전
더블클릭으로 실행 가능한 간단한 GUI 인터페이스
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sys
from pathlib import Path
from organize_videos import VideoOrganizer

class VideoOrganizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("비디오 파일 구조화 도구")
        self.root.geometry("600x500")
        
        # 변수
        self.source_dir = tk.StringVar()
        self.target_dir = tk.StringVar()
        self.interactive_mode = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        # 제목
        title_label = tk.Label(
            self.root, 
            text="비디오 파일 구조화 도구",
            font=("맑은 고딕", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # 소스 디렉토리 선택
        source_frame = tk.Frame(self.root)
        source_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(source_frame, text="비디오 폴더:", font=("맑은 고딕", 10)).pack(anchor=tk.W)
        
        source_path_frame = tk.Frame(source_frame)
        source_path_frame.pack(fill=tk.X, pady=5)
        
        tk.Entry(source_path_frame, textvariable=self.source_dir, font=("맑은 고딕", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            source_path_frame, 
            text="찾아보기", 
            command=self.browse_source,
            font=("맑은 고딕", 9)
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # 타겟 디렉토리 선택 (선택사항)
        target_frame = tk.Frame(self.root)
        target_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(target_frame, text="저장 폴더 (선택사항, 비워두면 자동 생성):", font=("맑은 고딕", 10)).pack(anchor=tk.W)
        
        target_path_frame = tk.Frame(target_frame)
        target_path_frame.pack(fill=tk.X, pady=5)
        
        tk.Entry(target_path_frame, textvariable=self.target_dir, font=("맑은 고딕", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            target_path_frame, 
            text="찾아보기", 
            command=self.browse_target,
            font=("맑은 고딕", 9)
        ).pack(side=tk.LEFT, padx=(5, 0))
        
        # 옵션
        options_frame = tk.Frame(self.root)
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Checkbutton(
            options_frame,
            text="대화형 모드 (파일명 파싱 실패 시 수동 입력)",
            variable=self.interactive_mode,
            font=("맑은 고딕", 9)
        ).pack(anchor=tk.W)
        
        # 실행 버튼
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.run_button = tk.Button(
            button_frame,
            text="비디오 파일 정리 시작",
            command=self.start_organizing,
            font=("맑은 고딕", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10
        )
        self.run_button.pack()
        
        # 로그 영역
        log_label = tk.Label(self.root, text="실행 로그:", font=("맑은 고딕", 10))
        log_label.pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(
            self.root,
            height=10,
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
    def browse_source(self):
        directory = filedialog.askdirectory(title="비디오 파일이 있는 폴더 선택")
        if directory:
            self.source_dir.set(directory)
            
    def browse_target(self):
        directory = filedialog.askdirectory(title="정리된 파일을 저장할 폴더 선택")
        if directory:
            self.target_dir.set(directory)
    
    def log(self, message):
        """로그 메시지 추가"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_organizing(self):
        """비디오 정리 시작"""
        source = self.source_dir.get().strip()
        
        if not source:
            messagebox.showerror("오류", "비디오 폴더를 선택해주세요.")
            return
        
        if not Path(source).exists():
            messagebox.showerror("오류", "선택한 폴더가 존재하지 않습니다.")
            return
        
        # UI 비활성화
        self.run_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)
        
        # 별도 스레드에서 실행
        thread = threading.Thread(target=self.organize_thread, args=(source,))
        thread.daemon = True
        thread.start()
    
    def organize_thread(self, source_dir):
        """비디오 정리 스레드"""
        try:
            target = self.target_dir.get().strip() or None
            
            self.log("비디오 파일 구조화를 시작합니다...")
            self.log(f"소스 폴더: {source_dir}")
            if target:
                self.log(f"타겟 폴더: {target}")
            self.log("")
            
            # VideoOrganizer 생성 및 실행
            organizer = VideoOrganizer(source_dir, target)
            
            # 로그를 위한 커스텀 출력 함수
            import sys
            from io import StringIO
            
            class LogRedirect:
                def __init__(self, log_func):
                    self.log_func = log_func
                    self.buffer = StringIO()
                
                def write(self, text):
                    if text.strip():
                        self.log_func(text.strip())
                    return len(text)
                
                def flush(self):
                    pass
            
            # stdout 리다이렉트
            old_stdout = sys.stdout
            sys.stdout = LogRedirect(self.log)
            
            try:
                stats = organizer.organize_videos(interactive=self.interactive_mode.get())
                
                # 통계 출력
                self.log("\n" + "="*50)
                self.log("📊 정리 통계")
                self.log("="*50)
                self.log(f"총 파일 수: {stats['total_files']}")
                self.log(f"정리 완료: {stats['organized']}")
                self.log(f"실패: {stats['failed']}")
                self.log("\n성별별 분포:")
                for gender, count in stats['by_gender'].items():
                    self.log(f"  {gender}: {count}개")
                self.log("\n감정별 분포:")
                for emotion, count in stats['by_emotion'].items():
                    self.log(f"  {emotion}: {count}개")
                self.log("\n" + "="*50)
                self.log(f"정리된 파일 위치: {organizer.target_dir}")
                self.log("="*50)
                
                messagebox.showinfo("완료", f"비디오 파일 정리가 완료되었습니다!\n\n총 {stats['total_files']}개 파일 중 {stats['organized']}개 정리 완료")
                
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            self.log(f"\n❌ 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"작업 중 오류가 발생했습니다:\n{str(e)}")
        finally:
            # UI 활성화
            self.run_button.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = VideoOrganizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()




