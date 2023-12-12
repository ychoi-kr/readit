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
selected_languages = st.multiselect("언어 선택", language_options, default=["en"])

# 정확도 임계값 설정
threshold = st.slider("정확도 임곗값 설정", 0.0, 1.0, 0.5)

# 이미지 업로드
uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # OCR 처리
    if st.button("OCR 실행"):
        # 이미지를 OpenCV 형식으로 변환
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # EasyOCR Reader
        reader = easyocr.Reader(selected_languages)
        result = reader.readtext(img)

        # 사각형 그리기 및 결과 표시
        st.write("OCR 결과:")
        for bbox, text, conf in result:
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])
            if conf > threshold:
                # 신뢰도에 따른 색상 설정
                color = (0, 255, 0) if conf > 0.8 else (255, 0, 0)
                # 사각형 그리기
                img = cv2.rectangle(img, top_left, bottom_right, color, 2)
                st.write(text)
        
        # 변환된 이미지 표시
        st.image(img, caption='Processed Image', use_column_width=True)
