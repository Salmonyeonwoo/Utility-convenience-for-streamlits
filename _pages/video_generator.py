"""
비디오 생성 페이지
Streamlit pages 폴더에 추가하여 자동으로 탭으로 표시됩니다.
"""

import streamlit as st
import os
from pathlib import Path
import sys

# 상위 디렉토리의 모듈 import
sys.path.append(str(Path(__file__).parent.parent))
from video_generator_module import VideoGenerator, generate_videos_batch

st.set_page_config(
    page_title="비디오 생성",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI 비디오 생성 도구")

# API 키는 환경 변수에서 자동으로 읽어옵니다
# 환경 변수 설정: D_ID_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
api_key = os.environ.get("D_ID_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# 안내 메시지
with st.expander("📖 API 비교 및 안내"):
    st.markdown("""
    ### API 비교
    
    | 기능 | D-ID API | OpenAI DALL-E | Gemini API |
    |------|----------|---------------|------------|
    | Talking Head 비디오 | ✅ 가능 | ❌ 불가능 | ❌ 불가능 |
    | 정적 이미지 생성 | ✅ 가능 | ✅ 가능 | ⚠️ 제한적 |
    | 감정 표현 | ✅ 가능 | ✅ 가능 | ❌ 불가능 |
    | 무료 크레딧 | ✅ 제공 | ⚠️ 유료 | ✅ 제공 |
    
    ### 권장 사항
    
    **실제 talking head 비디오가 필요하다면:**
    - ✅ **D-ID API 사용** (가장 현실적이고 효과적)
    - 무료 계정: https://studio.d-id.com
    
    **정적 이미지만 필요하다면:**
    - OpenAI DALL-E 사용 가능
    - 하지만 실제 비디오를 원하시면 D-ID가 필수입니다
    
    **결론:** OpenAI/Gemini API만으로는 talking head 비디오를 생성할 수 없습니다.
    """)

# 탭 생성
tab1, tab2 = st.tabs(["단일 비디오 생성", "일괄 비디오 생성"])

with tab1:
    st.header("단일 비디오 생성")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("기본 설정")
        gender = st.selectbox("성별", ["남자", "여자"], key="single_gender")
        emotion = st.selectbox(
            "감정 상태",
            ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"],
            key="single_emotion"
        )
        
        # 이미지 업로드 또는 URL 입력
        image_source = st.radio(
            "이미지 소스",
            ["URL 입력", "파일 업로드"],
            key="single_image_source"
        )
        
        if image_source == "URL 입력":
            image_url = st.text_input(
                "아바타 이미지 URL",
                placeholder="https://example.com/avatar.jpg",
                key="single_image_url"
            )
        else:
            uploaded_file = st.file_uploader(
                "아바타 이미지 업로드",
                type=["jpg", "jpeg", "png"],
                key="single_image_upload"
            )
            if uploaded_file:
                # 임시 파일로 저장
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_url = str(temp_path.absolute())
                st.success(f"✅ 이미지 업로드 완료: {uploaded_file.name}")
            else:
                image_url = None
    
    with col2:
        st.subheader("스크립트 설정")
        script = st.text_area(
            "비디오에서 말할 텍스트",
            placeholder="안녕하세요. 반갑습니다.",
            height=200,
            key="single_script"
        )
        
        voice_id = st.selectbox(
            "음성 선택",
            ["en-US-GuyNeural", "en-US-JennyNeural", "ko-KR-InJoonNeural", "ko-KR-SunHiNeural"],
            key="single_voice"
        )
    
    if st.button("비디오 생성", type="primary", key="single_generate"):
        if not api_key:
            st.error("❌ API 키가 설정되지 않았습니다. 환경 변수 D_ID_API_KEY, OPENAI_API_KEY, 또는 GEMINI_API_KEY를 설정해주세요.")
        elif not image_url:
            st.error("❌ 이미지 URL 또는 파일을 입력해주세요.")
        elif not script:
            st.error("❌ 스크립트를 입력해주세요.")
        else:
            with st.spinner("비디오를 생성하는 중..."):
                generator = VideoGenerator()
                result = generator.generate_video_with_did(
                    image_url=image_url,
                    script=script,
                    voice_id=voice_id,
                    gender=gender,
                    emotion=emotion
                )
                
                if result.get("success"):
                    st.success("✅ 비디오 생성이 시작되었습니다!")
                    video_id = result.get("video_id")
                    st.info(f"비디오 ID: {video_id}")
                    
                    # 상태 확인
                    with st.spinner("비디오 생성 상태 확인 중..."):
                        import time
                        max_attempts = 30
                        for attempt in range(max_attempts):
                            time.sleep(2)
                            status_result = generator.get_video_status(video_id)
                            
                            if status_result.get("status") == "done":
                                video_url = status_result.get("video_url")
                                if video_url:
                                    st.video(video_url)
                                    st.success("✅ 비디오 생성 완료!")
                                    
                                    # 다운로드 버튼
                                    st.download_button(
                                        "비디오 다운로드",
                                        data=video_url,
                                        file_name=f"{gender}_{emotion}.mp4",
                                        mime="video/mp4"
                                    )
                                break
                            elif status_result.get("status") == "error":
                                st.error("❌ 비디오 생성 실패")
                                break
                else:
                    st.error(f"❌ 오류: {result.get('error')}")

with tab2:
    st.header("일괄 비디오 생성")
    st.markdown("성별과 감정 상태에 맞는 모든 비디오를 한 번에 생성합니다.")
    
    st.subheader("설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**성별 선택**")
        gender_male = st.checkbox("남자", value=True, key="batch_male")
        gender_female = st.checkbox("여자", value=True, key="batch_female")
        
        genders = []
        if gender_male:
            genders.append("남자")
        if gender_female:
            genders.append("여자")
    
    with col2:
        st.write("**감정 선택**")
        emotions_selected = st.multiselect(
            "감정 상태",
            ["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"],
            default=["NEUTRAL", "HAPPY", "ANGRY", "ASKING", "SAD"],
            key="batch_emotions"
        )
    
    # 이미지 URL 설정
    st.subheader("아바타 이미지 설정")
    image_url_male = st.text_input("남자 아바타 이미지 URL", key="batch_image_male")
    image_url_female = st.text_input("여자 아바타 이미지 URL", key="batch_image_female")
    
    # 스크립트 설정
    st.subheader("감정별 스크립트")
    scripts = {}
    for emotion in emotions_selected:
        scripts[emotion] = st.text_input(
            f"{emotion} 스크립트",
            value=f"{emotion} 상태의 인사말입니다.",
            key=f"script_{emotion}"
        )
    
    # 출력 폴더 설정
    output_dir = st.text_input(
        "저장 폴더",
        value="generated_videos",
        key="batch_output_dir"
    )
    
    if st.button("일괄 생성 시작", type="primary", key="batch_generate"):
        if not api_key:
            st.error("❌ API 키가 설정되지 않았습니다. 환경 변수 D_ID_API_KEY, OPENAI_API_KEY, 또는 GEMINI_API_KEY를 설정해주세요.")
        elif not genders:
            st.error("❌ 최소 하나의 성별을 선택해주세요.")
        elif not emotions_selected:
            st.error("❌ 최소 하나의 감정을 선택해주세요.")
        elif not image_url_male or not image_url_female:
            st.error("❌ 남자와 여자 아바타 이미지 URL을 모두 입력해주세요.")
        else:
            image_urls = {
                "남자": image_url_male,
                "여자": image_url_female
            }
            
            with st.spinner("비디오를 일괄 생성하는 중... (시간이 걸릴 수 있습니다)"):
                results = generate_videos_batch(
                    genders=genders,
                    emotions=emotions_selected,
                    scripts=scripts,
                    image_urls=image_urls,
                    output_dir=output_dir
                )
                
                # 결과 표시
                st.subheader("생성 결과")
                success_count = sum(1 for r in results.values() if r.get("success"))
                total_count = len(results)
                
                st.metric("성공", f"{success_count}/{total_count}")
                
                # 결과 상세
                for key, result in results.items():
                    with st.expander(f"{key} - {result.get('status', 'N/A')}"):
                        if result.get("success"):
                            st.success("✅ 생성 성공")
                            if result.get("video_path"):
                                st.video(result["video_path"])
                        else:
                            st.error(f"❌ 오류: {result.get('error')}")

# 사용 가이드
with st.expander("📖 사용 가이드"):
    st.markdown("""
    ### D-ID API 사용 방법
    
    1. **API 키 발급**
       - [D-ID Studio](https://studio.d-id.com)에 가입
       - API 키 발급 (무료 크레딧 제공)
    
    2. **이미지 준비**
       - 아바타로 사용할 사람의 얼굴 사진 준비
       - URL로 접근 가능한 이미지 또는 파일 업로드
    
    3. **비디오 생성**
       - 성별과 감정 상태 선택
       - 스크립트 입력 (비디오에서 말할 내용)
       - 생성 버튼 클릭
    
    4. **결과 확인**
       - 생성 완료 후 비디오 자동 재생
       - 다운로드 버튼으로 저장
    
    ### 주의사항
    - API 사용량에 따라 비용이 발생할 수 있습니다
    - 비디오 생성에는 시간이 걸릴 수 있습니다 (약 30초~2분)
    - 무료 플랜에는 제한이 있을 수 있습니다
    """)

