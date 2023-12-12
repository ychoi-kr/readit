import streamlit as st
import easyocr
import cv2
from PIL import Image
import numpy as np
import urllib


# 색상 스펙트럼을 결정하는 함수
def get_color(confidence):
    match confidence:
        case _ if confidence > 0.9:
            return (0, 255, 0)  # 밝은 녹색
        case _ if confidence > 0.8:
            return (100, 255, 100)  # 중간 녹색
        case _ if confidence > 0.7:
            return (255, 255, 0)  # 밝은 노란색
        case _ if confidence > 0.6:
            return (255, 200, 0)  # 중간 노란색
        case _ if confidence > 0.5:
            return (255, 150, 0)  # 어두운 노란색
        case _:
            # 신뢰도가 낮을수록 빨간색이 진해짐
            red_intensity = int(255 * confidence)  
            return (255, red_intensity, red_intensity)

# Streamlit 페이지 설정
st.set_page_config(page_title="OCR Web App", layout="wide")

# 타이틀
st.title("OCR Web App")

# 언어 선택 (복수 선택 가능)
language_options = ["en", "ja", "ko", "fr", "de"]  # 사용 가능한 언어 목록
selected_languages = st.multiselect("언어 선택", language_options, default=["en", "ja"])

# 이미지 업로드
uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])

# OCR 실행 버튼 및 정확도 임곗값 설정
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("OCR 실행"):
        st.session_state['ocr_clicked'] = True
with col2:
    threshold = st.slider("정확도 임곗값 설정", 0.0, 1.0, 0.5, key='threshold_slider')

# OCR 처리 및 결과 표시
def display_ocr_results(image, result, threshold):
    col1, col2 = st.columns(2)
    with col1:
        for bbox, text, conf in result:
            if conf > threshold:
                top_left = tuple([int(val) for val in bbox[0]])
                bottom_right = tuple([int(val) for val in bbox[2]])
                color = get_color(conf)
                image = cv2.rectangle(image, top_left, bottom_right, color, 2)
        st.image(image, caption='Processed Image', use_column_width=True)

    with col2:
        st.write("OCR 결과:")
        full_text = ""
        for bbox, text, conf in result:
            if conf > threshold:
                full_text += text + "\n"

        # OCR 결과를 코드 블록으로 표시(복사하기 편리하도록)
        if full_text:
            st.code(full_text)

        # 번역 대상 언어 선택 (기본값은 한국어 'ko')
        target_language = st.selectbox("번역할 언어 선택:", ["en", "ko", "ja", "fr", "de"], index=1)

        # 번역 링크 생성
        google_translate_url = f"https://translate.google.com/?sl=auto&tl={target_language}&text={urllib.parse.quote(full_text)}"
        deepl_translate_url = f"https://www.deepl.com/translator#auto/{target_language}/{urllib.parse.quote(full_text)}"
        papago_translate_url = f"https://papago.naver.com/?sk=auto&tk={target_language}&st={urllib.parse.quote(full_text)}"

        st.markdown(f"[Google 번역]({google_translate_url}) | [DeepL]({deepl_translate_url}) | [Papago]({papago_translate_url})", unsafe_allow_html=True)

if uploaded_file is not None and 'ocr_clicked' in st.session_state and st.session_state['ocr_clicked']:
    # 이미지를 OpenCV 형식으로 변환
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # EasyOCR Reader
    reader = easyocr.Reader(selected_languages)
    result = reader.readtext(img)
    st.session_state['ocr_result'] = result
    st.session_state['original_image'] = img.copy()
    st.session_state['ocr_clicked'] = False

if 'ocr_result' in st.session_state:
    display_ocr_results(st.session_state['original_image'].copy(), st.session_state['ocr_result'], st.session_state.threshold_slider)
