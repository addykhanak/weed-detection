import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Path where model will be stored locally
model_path = "models/weed_detection_model.keras"

# Google Drive file ID extracted from your link
drive_url = "https://drive.google.com/uc?id=17yQXcPjRTA8MlElJI-3OeM7Gjj92tgdd"

# Auto-download model if it doesn't exist
if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    st.info("Downloading model... Please wait.")
    gdown.download(drive_url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Streamlit app interface
st.title("Weed Detection in Plants 🌿")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((128, 128))  # Update size based on your model input
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    class_label = "Weed" if prediction[0][0] > 0.5 else "Not Weed"

    st.subheader("Prediction:")
    st.success(f"This plant is likely: **{class_label}**")
