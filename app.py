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

# 정확도 임계값 설정
threshold = st.slider("정확도 임곗값 설정", 0.0, 1.0, 0.5)

# 이미지 업로드
uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # OCR 처리
    if st.button("OCR 실행"):
        # EasyOCR Reader
        reader = easyocr.Reader(selected_languages)
        result = reader.readtext(img)

        # 결과 표시를 위한 컬럼 생성
        col1, col2 = st.columns(2)

        # 왼쪽 컬럼: 사각형 그려진 이미지
        with col1:
            for bbox, text, conf in result:
                if conf > threshold:
                    top_left = tuple([int(val) for val in bbox[0]])
                    bottom_right = tuple([int(val) for val in bbox[2]])
                    color = (0, 255, 0) if conf > 0.8 else (255, 0, 0)
                    img = cv2.rectangle(img, top_left, bottom_right, color, 2)
            st.image(img, caption='Processed Image', use_column_width=True)

        # 오른쪽 컬럼: OCR 결과 텍스트
        with col2:
            st.write("OCR 결과:")
            for bbox, text, conf in result:
                if conf > threshold:
                    st.write(text)
    else:
        # OCR 실행 전 원본 이미지 표시
        st.image(image, caption='Uploaded Image', use_column_width=True)
