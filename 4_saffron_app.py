# (paste the full saffron Streamlit code here)
# 4_saffron_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
import shap
import matplotlib.pyplot as plt

# Load model + encoder
model = joblib.load("saffron_quality_model.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="SaffronSense", page_icon="🌺", layout="wide")

st.title("🌺 SaffronSense: AI-Powered Quality Assessment with Explainability")
st.markdown("""
### Upload an image of saffron threads for automatic quality grading
This system uses computer vision, machine learning, and SHAP explainability.
""")

# --------------------------
# Feature extraction
# --------------------------
def extract_features(img):
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (256, 256))

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    hue = np.mean(hsv[:, :, 0])
    sat = np.mean(hsv[:, :, 1])
    val = np.mean(hsv[:, :, 2])

    red_mask = (img_resized[:, :, 0] > 150) & (img_resized[:, :, 1] < 80) & (img_resized[:, :, 2] < 80)
    red_percentage = np.sum(red_mask) / (256 * 256) * 100

    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    brightness = np.mean(gray)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths, widths = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        lengths.append(max(w, h))
        widths.append(min(w, h))
    thread_length = np.mean(lengths) if lengths else 0
    thread_width = np.mean(widths) if widths else 0

    features = {
        "Thread Length (mm)": thread_length,
        "HSV Hue": hue,
        "HSV Saturation": sat,
        "HSV Value": val,
        "Red Pixel Percentage": red_percentage,
        "Texture Contrast": contrast,
        "Texture Homogeneity": homogeneity,
        "Brightness": brightness,
        "Thread Width (mm)": thread_width
    }
    return pd.DataFrame([features])

# --------------------------
# File uploader
# --------------------------
uploaded_file = st.file_uploader("Choose a saffron image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Saffron Image", use_column_width=True)

    if st.button("🔍 Analyze Quality"):
        with st.spinner("Extracting features and analyzing..."):
            features_df = extract_features(image)
            prediction = model.predict(features_df)
            probability = model.predict_proba(features_df)
            predicted_grade = le.inverse_transform(prediction)[0]

            # Results
            st.success(f"**Predicted Quality Grade: {predicted_grade}**")

            st.subheader("Confidence Levels")
            for grade, prob in zip(le.classes_, probability[0]):
                st.write(f"{grade}: {prob:.2%}")

            st.subheader("Extracted Features")
            st.dataframe(features_df.T.rename(columns={0: "Value"}))

            # --------------------------
            # SHAP Explainability
            # --------------------------
            #st.subheader("🔎 Why this prediction? (SHAP Analysis)")

            #explainer = shap.Explainer(model, features_df)
            #shap_values = explainer(features_df)

            # Waterfall plot for single prediction
            #fig, ax = plt.subplots(figsize=(10, 6))
            #shap.plots.waterfall(shap_values[0], max_display=9, show=False)
            #st.pyplot(fig)

            # Bar plot (global importance for this prediction)
            #fig2, ax2 = plt.subplots(figsize=(10, 6))
            #shap.plots.bar(shap_values, show=False)
            #st.pyplot(fig2)

# --------------------------
# Sidebar info
# --------------------------
with st.sidebar:
    st.header("About SaffronSense")
    st.info("""
    This tool grades saffron based on:
    - Color (HSV + red pixel ratio)
    - Thread length & width
    - Texture features
    - Brightness
    - Explainability via SHAP
    """)

    st.header("Quality Standards")
    st.markdown("""
    - **AAAA**: Superior, longest threads, deepest red
    - **AAA**: High quality
    - **AA**: Good commercial grade
    - **A**: Acceptable
    - **B**: Low grade
    """)
