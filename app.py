import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=1VunCHlYtRHsWSf_3u8oWelTfz2jFyVQ6"
MODEL_PATH = "face_recognition_model.h5"

CLASS_NAMES = ["Saravana Kumar", "Guru Nagajothi", "Gobinath"]

def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_cnn_model()

_, IMG_H, IMG_W, IMG_C = model.input_shape
st.write("Detected model input shape:", model.input_shape)


st.title("🧠 Face Recognition App")

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)

    # Handle grayscale vs RGB automatically
    color_mode = "rgb" if IMG_C == 3 else "grayscale"

    img = image.load_img(
        uploaded_file,
        target_size=(IMG_H, IMG_W),
        color_mode=color_mode
    )

    img_array = image.img_to_array(img)

    if IMG_C == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.write("Image shape sent to model:", img_array.shape)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
