# ========================================
# utils/history_handler_exports_pdf.py
# 이력 관리 - PDF 내보내기 함수
# ========================================

import os
import platform
from datetime import datetime
from typing import List, Dict, Any
from lang_pack import LANG
from config import DATA_DIR

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import black
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    IS_REPORTLAB_AVAILABLE = True
except ImportError:
    IS_REPORTLAB_AVAILABLE = False

def export_history_to_pdf(histories: List[Dict[str, Any]], filename: str = None, lang: str = "ko") -> str:
    """이력을 PDF 파일로 저장 (한글/일본어 인코딩 지원 강화)"""
    if not IS_REPORTLAB_AVAILABLE:
        raise ImportError("PDF 저장을 위해 reportlab이 필요합니다: pip install reportlab")
    
    if lang not in ["ko", "en", "ja"]:
        lang = "ko"
    L = LANG.get(lang, LANG["ko"])
    
    if filename is None:
        filename = f"customer_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(DATA_DIR, filename)
    
    korean_font_registered = False
    japanese_font_registered = False
    korean_font_name = 'KoreanFont'
    japanese_font_name = 'JapaneseFont'
    
    def register_font(font_name: str, font_path: str) -> bool:
        """폰트를 등록하는 헬퍼 함수"""
        try:
            if font_path.endswith('.ttf'):
                font = TTFont(font_name, font_path)
                pdfmetrics.registerFont(font)
                return font_name in pdfmetrics.getRegisteredFontNames()
            elif font_path.endswith('.ttc'):
                for subfont_idx in range(8):
                    try:
                        font = TTFont(font_name, font_path, subfontIndex=subfont_idx)
                        pdfmetrics.registerFont(font)
                        if font_name in pdfmetrics.getRegisteredFontNames():
                            return True
                    except Exception:
                        continue
            elif font_path.endswith('.otf'):
                try:
                    font = TTFont(font_name, font_path)
                    pdfmetrics.registerFont(font)
                    return font_name in pdfmetrics.getRegisteredFontNames()
                except Exception:
                    pass
            return False
        except Exception:
            return False
    
    def download_font_if_needed(font_dir: str = None) -> str:
        """폰트가 없을 경우 Noto Sans CJK 폰트를 자동 다운로드"""
        if font_dir is None:
            font_dir = os.path.join(DATA_DIR, "fonts")
        os.makedirs(font_dir, exist_ok=True)
        
        font_files = [
            ("NotoSansCJK-Regular.ttf", "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Subset/TTF/NotoSansCJK-Regular.ttf"),
            ("NotoSansCJK-Regular.otf", "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Subset/OTF/NotoSansCJK-Regular.otf"),
        ]
        
        for font_filename, font_url in font_files:
            font_path = os.path.join(font_dir, font_filename)
            if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
                return font_path
        
        try:
            import requests
            for font_filename, font_url in font_files:
                font_path = os.path.join(font_dir, font_filename)
                try:
                    response = requests.get(font_url, timeout=30, stream=True)
                    response.raise_for_status()
                    with open(font_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if os.path.exists(font_path) and os.path.getsize(font_path) > 1000:
                        return font_path
                except Exception:
                    continue
        except ImportError:
            pass
        return None
    
    try:
        system = platform.system()
        
        if system == 'Windows':
            korean_font_paths = [
                "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunsl.ttf",
                "C:/Windows/Fonts/NanumGothic.ttf", "C:/Windows/Fonts/gulim.ttc",
                "C:/Windows/Fonts/batang.ttc", "C:/Windows/Fonts/malgun.ttc",
            ]
            japanese_font_paths = [
                "C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/msmincho.ttc",
                "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/yuanti.ttc",
            ]
        elif system == 'Darwin':
            korean_font_paths = [
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/Library/Fonts/AppleGothic.ttf",
            ]
            japanese_font_paths = [
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
            ]
        else:
            korean_font_paths = [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]
            japanese_font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/takao/TakaoGothic.ttf",
            ]
        
        for font_path in korean_font_paths:
            if os.path.exists(font_path) and register_font(korean_font_name, font_path):
                korean_font_registered = True
                break
        
        for font_path in japanese_font_paths:
            if os.path.exists(font_path) and register_font(japanese_font_name, font_path):
                japanese_font_registered = True
                break
        
        if not korean_font_registered and not japanese_font_registered and system == 'Linux':
            downloaded_font = download_font_if_needed()
            if downloaded_font and os.path.exists(downloaded_font):
                if register_font(korean_font_name, downloaded_font):
                    korean_font_registered = True
                    japanese_font_registered = True
    except Exception:
        pass
    
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    def get_multilingual_style(base_style_name, default_font=None, **kwargs):
        """다국어 지원 스타일 생성"""
        base_style = styles[base_style_name]
        style_kwargs = {'parent': base_style, **kwargs}
        registered_fonts = pdfmetrics.getRegisteredFontNames()
        if default_font and default_font in registered_fonts:
            style_kwargs['fontName'] = default_font
        elif korean_font_registered and korean_font_name in registered_fonts:
            style_kwargs['fontName'] = korean_font_name
        elif japanese_font_registered and japanese_font_name in registered_fonts:
            style_kwargs['fontName'] = japanese_font_name
        return ParagraphStyle(f'Multilingual{base_style_name}', **style_kwargs)
    
    title_style = get_multilingual_style('Heading1', fontSize=24, textColor=black, spaceAfter=30, alignment=1,
                                         default_font=korean_font_name if korean_font_registered else japanese_font_name)
    normal_style = get_multilingual_style('Normal')
    heading1_style = get_multilingual_style('Heading1')
    heading2_style = get_multilingual_style('Heading2')
    
    def safe_text(text, detect_font=True):
        """텍스트를 안전하게 처리"""
        if text is None:
            return ("N/A", None)
        if isinstance(text, bytes):
            try:
                text_str = text.decode('utf-8', errors='replace')
            except:
                text_str = str(text)
        else:
            text_str = str(text)
        if text_str is None:
            return ("N/A", None)
        try:
            import unicodedata
            text_str = unicodedata.normalize('NFC', text_str)
        except:
            pass
        text_str = text_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        text_str = text_str.replace('"', '&quot;').replace("'", '&#39;')
        
        recommended_font = None
        if detect_font:
            try:
                has_korean = any('\uAC00' <= char <= '\uD7AF' or '\u1100' <= char <= '\u11FF' for char in text_str)
                has_japanese = any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF' for char in text_str)
                registered_fonts = pdfmetrics.getRegisteredFontNames()
                if has_korean and korean_font_registered and korean_font_name in registered_fonts:
                    recommended_font = korean_font_name
                elif has_japanese and japanese_font_registered and japanese_font_name in registered_fonts:
                    recommended_font = japanese_font_name
            except Exception:
                pass
        return (text_str, recommended_font)
    
    def create_paragraph(text, style, auto_font=True):
        """Paragraph 생성"""
        text_str, recommended_font = safe_text(text, detect_font=auto_font)
        if recommended_font and auto_font:
            style_with_font = ParagraphStyle(name=f'{style.name}_with_font', parent=style, fontName=recommended_font)
            return Paragraph(text_str, style_with_font)
        return Paragraph(text_str, style)
    
    title_text, _ = safe_text(L.get("download_history_title", "고객 응대 이력 요약"))
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 0.2*inch))
    
    for i, hist in enumerate(histories, 1):
        heading_text, _ = safe_text(f'{L.get("download_history_number", "이력 #")}{i}')
        story.append(Paragraph(heading_text, heading1_style))
        story.append(Spacer(1, 0.1*inch))
        
        id_text, _ = safe_text(f'ID: {hist.get("id", "N/A")}')
        story.append(create_paragraph(id_text, normal_style))
        
        timestamp_text, _ = safe_text(f'{L.get("date_label", "날짜")}: {hist.get("timestamp", "N/A")}')
        story.append(create_paragraph(timestamp_text, normal_style))
        
        query_text, _ = safe_text(f'{L.get("download_initial_inquiry", "초기 문의")}: {hist.get("initial_query", "N/A")}')
        story.append(create_paragraph(query_text, normal_style))
        
        customer_type_text, _ = safe_text(f'{L.get("customer_type_label", "고객 유형")}: {hist.get("customer_type", "N/A")}')
        story.append(create_paragraph(customer_type_text, normal_style))
        
        language_text, _ = safe_text(f'{L.get("language_label", "언어")}: {hist.get("language_key", "N/A")}')
        story.append(create_paragraph(language_text, normal_style))
        
        summary = hist.get('summary', {})
        if summary:
            story.append(Spacer(1, 0.1*inch))
            summary_title, _ = safe_text(L.get("download_summary", "요약"))
            story.append(Paragraph(summary_title, heading2_style))
            
            main_inquiry_text, _ = safe_text(f'{L.get("download_main_inquiry", "주요 문의")}: {summary.get("main_inquiry", "N/A")}')
            story.append(create_paragraph(main_inquiry_text, normal_style))
            
            key_responses = summary.get("key_responses", [])
            if isinstance(key_responses, list):
                responses_list = [safe_text(r)[0] for r in key_responses]
                responses_text = ", ".join(responses_list)
            else:
                responses_text, _ = safe_text(key_responses)
            responses_para_text, _ = safe_text(f'{L.get("download_key_response", "핵심 응답")}: {responses_text}')
            story.append(create_paragraph(responses_para_text, normal_style))
            
            sentiment_text, _ = safe_text(f'{L.get("sentiment_score_label", "고객 감정 점수")}: {summary.get("customer_sentiment_score", "N/A")}/100')
            story.append(create_paragraph(sentiment_text, normal_style))
            
            satisfaction_text, _ = safe_text(f'{L.get("customer_satisfaction_score_label", "고객 만족도 점수")}: {summary.get("customer_satisfaction_score", "N/A")}/100')
            story.append(create_paragraph(satisfaction_text, normal_style))
            
            characteristics = summary.get('customer_characteristics', {})
            story.append(Spacer(1, 0.1*inch))
            char_title, _ = safe_text(L.get("download_customer_characteristics", "고객 특성"))
            story.append(Paragraph(char_title, heading2_style))
            
            lang_char_text, _ = safe_text(f'{L.get("language_label", "언어")}: {characteristics.get("language", "N/A")}')
            story.append(create_paragraph(lang_char_text, normal_style))
            
            cultural_text, _ = safe_text(f'{L.get("download_cultural_background", "문화적 배경")}: {characteristics.get("cultural_hints", "N/A")}')
            story.append(create_paragraph(cultural_text, normal_style))
            
            region_text, _ = safe_text(f'{L.get("region_label", "지역")}: {characteristics.get("region", "N/A")}')
            story.append(create_paragraph(region_text, normal_style))
            
            comm_style_text, _ = safe_text(f'{L.get("download_communication_style", "소통 스타일")}: {characteristics.get("communication_style", "N/A")}')
            story.append(create_paragraph(comm_style_text, normal_style))
            
            privacy = summary.get('privacy_info', {})
            story.append(Spacer(1, 0.1*inch))
            privacy_title, _ = safe_text(L.get("download_privacy_summary", "개인정보 요약"))
            story.append(Paragraph(privacy_title, heading2_style))
            
            email_text, _ = safe_text(f'{L.get("email_provided_label", "이메일 제공")}: {L.get("download_yes", "예") if privacy.get("has_email") else L.get("download_no", "아니오")}')
            story.append(create_paragraph(email_text, normal_style))
            
            phone_text, _ = safe_text(f'{L.get("phone_provided_label", "전화번호 제공")}: {L.get("download_yes", "예") if privacy.get("has_phone") else L.get("download_no", "아니오")}')
            story.append(create_paragraph(phone_text, normal_style))
            
            address_text, _ = safe_text(f'{L.get("download_address_provided", "주소 제공")}: {L.get("download_yes", "예") if privacy.get("has_address") else L.get("download_no", "아니오")}')
            story.append(create_paragraph(address_text, normal_style))
            
            region_hint_text, _ = safe_text(f'{L.get("download_region_hint", "지역 힌트")}: {privacy.get("region_hint", "N/A")}')
            story.append(create_paragraph(region_hint_text, normal_style))
            
            full_summary_text, _ = safe_text(f'{L.get("download_overall_summary", "전체 요약")}: {summary.get("summary", "N/A")}')
            story.append(create_paragraph(full_summary_text, normal_style))
        
        if i < len(histories):
            story.append(Spacer(1, 0.2*inch))
            divider_text, _ = safe_text('-' * 80)
            story.append(Paragraph(divider_text, normal_style))
            story.append(Spacer(1, 0.2*inch))
    
    try:
        doc.build(story)
    except Exception as e:
        raise Exception(f"PDF 생성 실패: {str(e)}")
    
    return filepath

