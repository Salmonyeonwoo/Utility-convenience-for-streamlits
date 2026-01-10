# ========================================
# _pages/_content_share.py
# 콘텐츠 공유 기능 모듈
# ========================================

import streamlit as st
import json
import uuid
import streamlit.components.v1 as components
from lang_pack import LANG


def render_share_buttons(content, content_display, topic, L):
    """공유 버튼 렌더링"""
    col_like, col_dislike, col_share, col_copy, col_more = st.columns([1, 1, 1, 1, 6])
    current_content_id = str(uuid.uuid4())

    # 좋아요 버튼
    if col_like.button("👍", key=f"content_like_{current_content_id}"):
        st.toast(L.get("toast_like", "좋아요!"))

    # 싫어요 버튼
    if col_dislike.button(L.get("button_dislike", "👎"), key=f"content_dislike_{current_content_id}"):
        st.toast(L.get("toast_dislike", "싫어요..."))

    # 공유 버튼
    with col_share:
        share_clicked = st.button(L.get("button_share", "🔗"), key=f"content_share_{current_content_id}")

    if share_clicked:
        _handle_share(content, content_display, topic)

    # 복사 버튼
    if col_copy.button("📋", key=f"content_copy_{current_content_id}"):
        _handle_copy(content, L)


def _handle_share(content, content_display, topic):
    """공유 처리"""
    share_title = f"{content_display} ({topic})"
    share_text = content[:150] + "..."
    share_url = "https://utility-convenience-salmonyeonwoo.streamlit.app/"

    js_native_share = """
       function triggerNativeShare(title, text, url) {
           if (navigator.share) {
               navigator.share({
                   title: title,
                   text: text,
                   url: url,
               }).then(() => {
                   console.log('Successful share');
               }).catch((error) => {
                   console.log('Error sharing', error);
               });
               return true;
           } else {
              return false;
           }
       }
    """

    html_content = (
        f"<script>{js_native_share}\n"
        f"    const shared = triggerNativeShare('{share_title}', '{share_text}', '{share_url}');\n"
        f"    if (shared) {{\n"
        f"        console.log(\"Native Share Attempted.\");\n"
        f"    }} else {{\n"
        f"       const url = window.location.href;\n"
        f"       const textarea = document.createElement('textarea');\n"
        f"       textarea.value = url;\n"
        f"       document.body.appendChild(textarea);\n"
        f"       textarea.select();\n"
        f"       document.execCommand('copy');\n"
        f"       document.body.removeChild(textarea);\n"
        f"       alert('URL이 클립보드에 복사되었습니다.');\n"
        f"    }}\n"
        f"</script>")
    components.v1.html(html_content, height=0)
    st.toast("공유 완료!")


def _handle_copy(content, L):
    """콘텐츠 복사 처리"""
    content_for_js = json.dumps(content)
    js_copy_script = """
       function copyToClipboard(text) {{
           navigator.clipboard.writeText(text).then(function() {{
               console.log("복사 완료: " + text.substring(0, 50) + "...");
           }}, function(err) {{
               const textarea = document.createElement('textarea');
               textarea.value = text;
               document.body.appendChild(textarea);
               textarea.select();
               document.execCommand('copy');
               document.body.removeChild(textarea);
               alert("복사 완료!");
           }});
       }}
       copyToClipboard(JSON.parse('{content_json_safe}'));
    """.format(content_json_safe=content_for_js)
    
    components.v1.html(js_copy_script, height=0)
    st.toast(L.get("toast_copy", "복사 완료!"))
