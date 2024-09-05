import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = tf.keras.models.load_model('Model/dzongkha_digits_classifier.h5')
    return model

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    
    prediction = model.predict(img_reshape)
    
    return prediction

with st.spinner('Model is being loaded..'):
    model = load_model()

st.title("Dzongkha Digits Classification")

st.markdown("<h5 style='text-align: center;'>Share your image and let us analyze the Dzongkha Digit. <br><br> It's quick and fun give it a try<br><br></h5>", unsafe_allow_html=True)

file = st.file_uploader("Please Upload Your Image (jpg or png)", type=["jpg", "png"])

st.markdown("")

if file is None:
    st.warning("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True, caption="Uploaded Image")
    
if st.button("Analyze", key="classify_button"):
    with st.spinner("Classifying..."):
        predictions = import_and_predict(image, model)
        emotions = ["0 - Zero", "1 - One", "2 - Two", "3 - Three", "4 - Four", "5 - Five", "6 - Six", "7 - Seven", "8 - Eight", "9 - Nine"]
        predicted_emotion = emotions[np.argmax(predictions)]

        # Center the prediction message and predicted emotion
        st.markdown("<h5 style='text-align: center;'>Current Dzongkha Digit:</h5>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color:green;'>{predicted_emotion}</h1>", unsafe_allow_html=True)

# Style the "Classify" button
st.write("<style>div.stButton > button {width: 100%; font-size: 20px; font-weight: bold; background-color: green;}</style>", unsafe_allow_html=True)