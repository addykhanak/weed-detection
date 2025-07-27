import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

# Load the trained model
MODEL_PATH = "models/weed_detection_model.keras"
model = load_model(MODEL_PATH)

st.title("Weed Detection App 🌱")
st.write("Upload an image of a plant to check if it contains weed.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Uploaded Image", use_column_width=True)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img_resized = cv2.resize(img, (128, 128)) / 255.0  # Resize and normalize
    img_array = np.expand_dims(img_resized, axis=0)    # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    if class_idx == 1:
        st.success(f"✅ No weed detected with confidence: {confidence:.2f}")
        
    else:
        st.success(f"🌾 Weed detected with confidence: {confidence:.2f}")
