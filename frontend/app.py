"""Streamlit UI. Start: streamlit run frontend/app.py"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
st.set_page_config(page_title="Potato Leaf Classifier", page_icon="🌿")
st.title("🌿 Potato Leaf Disease Classifier")
st.write("Upload a potato leaf photo to see the CNN classification.")
st.caption("Educational demonstration — not a definitive diagnosis.")
image_file = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)
    if st.button("Predict", type="primary"):
        with st.spinner("Running CNN..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (image_file.name, image_file.getvalue(), image_file.type or "application/octet-stream")},
                    timeout=45,
                )
                response.raise_for_status()
                result = response.json()
                st.subheader("Prediction: " + result["label"])
                conf = float(result["confidence"])
                st.write(f"Model score: {conf:.1%} (not a calibrated certainty)")
                st.progress(min(max(conf, 0.0), 1.0))
                st.write("**Class scores**")
                st.bar_chart({"Class": list(result["all_probabilities"]),
                              "Score": list(result["all_probabilities"].values())},
                             x="Class", y="Score")
                st.write("**General precautions:** " + result["advice"])
                st.caption(result["disclaimer"])
            except requests.RequestException as exc:
                st.error("Cannot connect to prediction API. Start FastAPI first. " + str(exc))
