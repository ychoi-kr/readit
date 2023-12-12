import streamlit as st
import easyocr
import cv2
from PIL import Image
import numpy as np

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
                color = (0, 255, 0) if conf > 0.8 else (255, 0, 0)
                image = cv2.rectangle(image, top_left, bottom_right, color, 2)
        st.image(image, caption='Processed Image', use_column_width=True)

    with col2:
        st.write("OCR 결과:")
        for bbox, text, conf in result:
            if conf > threshold:
                st.write(text)

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
