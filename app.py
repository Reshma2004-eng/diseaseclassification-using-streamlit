import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import random
from skimage.io import imread
from skimage.transform import resize

# ---- CONFIG ----
IMAGE_SIZE = (48,48)
TEST_DIR = r"C:\Users\reshm\OneDrive\Desktop\dataset\data\test"
MODEL_PATH = "bestmodel.pkl"
METRICS_PATH = "model_metrics.csv"

# ---- Load Model & Metrics ----
st.title("🔬 Benign vs Malignant Image Classifier Dashboard")

st.markdown("""
This dashboard shows performance of 5 trained models and allows:
- Uploading an image for prediction
- Auto-prediction of 10 random test images
""")

# Load metrics and best model
df = pd.read_csv(METRICS_PATH)
best_model_name = df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
model = joblib.load(MODEL_PATH)

# Highlight best model
def highlight_best(s):
    return ['background-color: lightgreen' if v == best_model_name else '' for v in s]

st.subheader("📊 Model Metrics")
st.dataframe(df.style.apply(highlight_best, subset=['Model']))

# ---- Upload Image Prediction ----
st.subheader("📤 Upload an Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=250)
    img = imread(uploaded_file)
    img_resized = resize(img, IMAGE_SIZE).flatten().reshape(1, -1)
    prediction = model.predict(img_resized)[0]
    label = "Benign" if prediction == 0 else "Malignant"
    st.success(f"🔍 Prediction: **{label}**")

# ---- Auto Predict from Test Set ----
st.subheader("🎲 Random Predictions from Test Dataset (10 Samples)")

def get_random_test_images(n=10):
    images = []
    for cls_index, cls in enumerate(['benign', 'malignant']):
        folder = os.path.join(TEST_DIR, cls)
        if not os.path.exists(folder):
            st.error(f"❌ Folder not found: {folder}")
            continue
        all_files = os.listdir(folder)
        if not all_files:
            st.warning(f"⚠️ No images in: {folder}")
            continue
        files = random.sample(all_files, min(n // 2, len(all_files)))
        for file in files:
            path = os.path.join(folder, file)
            images.append((path, cls_index))  # (file path, true label)
    random.shuffle(images)
    return images

# Run auto prediction
random_images = get_random_test_images(n=10)

for img_path, true_label in random_images:
    img = imread(img_path)
    img_resized = resize(img, IMAGE_SIZE).flatten().reshape(1, -1)
    pred = model.predict(img_resized)[0]
    pred_label = "Benign" if pred == 0 else "Malignant"
    actual_label = "Benign" if true_label == 0 else "Malignant"

    st.image(img, width=150, caption=f"📌 Predicted: {pred_label} | 🧾 Actual: {actual_label}")
