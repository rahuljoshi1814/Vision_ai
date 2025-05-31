import streamlit as st
import os
import sys
import tempfile
import shutil
import json
import uuid
from datetime import datetime

# Ensure src path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your modules
from src.segmentation.segment_image import segment_image
from src.object_analysis.detect_objects import classify_images
from src.text_extraction.extract_text import extract_text_from_images
from src.summarization.summarize_data import summarize_objects
from src.video_mode.video_pipeline import process_video

# Setup paths
segmented_dir = "data/segmented"
results_dir = "data/results"
log_dir = "data/logs"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Streamlit UI
st.set_page_config(page_title="AI Vision System", layout="centered")
st.title("🧠 AI Image & Video Analyzer")
st.markdown("Upload an **image** or **video** for object detection, text extraction, and summarization.")

# ==================== IMAGE MODE ====================
st.header("📷 Image Upload")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_img_path = tmp_file.name

    st.image(tmp_img_path, caption="Original Image", use_column_width=True)

    with st.spinner("🔍 Processing image..."):
        # Clean segmented folder
        if os.path.exists(segmented_dir):
            shutil.rmtree(segmented_dir)
        os.makedirs(segmented_dir, exist_ok=True)

        segmented_paths = segment_image(tmp_img_path, output_dir=segmented_dir)

    if not segmented_paths:
        st.error("❌ No objects detected.")
    else:
        with st.spinner("📦 Classifying & extracting text..."):
            classes = classify_images(segmented_dir)
            texts = extract_text_from_images(segmented_dir)
            summaries = summarize_objects(classes, texts)

        st.success("✅ Done!")

        # Show segmented objects
        st.subheader("🧩 Segmented Objects")
        for file in os.listdir(segmented_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(segmented_dir, file)
                st.image(path, caption=file, width=150)

        # Show summary
        st.subheader("🧠 Summarized Output")
        for img, summary in summaries.items():
            st.markdown(f"**{img}**")
            st.write(summary)

        # Save summary as JSON
        json_name = f"summary_{uuid.uuid4().hex[:6]}.json"
        json_path = os.path.join(results_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(summaries, f, indent=4)
        with open(json_path, "rb") as f:
            st.download_button("⬇️ Download Summary JSON", f, file_name=json_name, mime="application/json")

        # Save session log
        log_name = f"log_image_{uuid.uuid4().hex[:6]}.txt"
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, "w") as log_file:
            log_file.write(f"🕒 Timestamp: {datetime.now()}\n")
            log_file.write(f"📷 Image File: {os.path.basename(tmp_img_path)}\n")
            log_file.write(f"🧩 Total Segmented Objects: {len(segmented_paths)}\n\n")
            for img, summary in summaries.items():
                log_file.write(f"📦 {img}:\n{summary}\n\n")
        with open(log_path, "rb") as f:
            st.download_button("📝 Download Session Log (Image)", f, file_name=log_name, mime="text/plain")

# ==================== VIDEO MODE ====================
st.header("🎥 Video Upload")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="video")

if video_file:
    st.video(video_file)

    if st.button("▶️ Process Video"):
        with st.spinner("⏳ Running YOLOv8 + EasyOCR..."):
            # Save video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
                temp_vid.write(video_file.read())
                video_path = temp_vid.name

            # Clear results folder
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            os.makedirs(results_dir, exist_ok=True)

            # Generate unique video output name
            output_name = f"annotated_video_{uuid.uuid4().hex[:6]}.avi"
            output_path = os.path.join(results_dir, output_name)

            # Generate log file
            log_name = f"log_video_{uuid.uuid4().hex[:6]}.txt"
            log_path = os.path.join(log_dir, log_name)

            # Run pipeline
            process_video(video_path, output_path, log_path=log_path)

        st.success("✅ Video processing complete!")
        st.video(output_path)

        # Download video
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Annotated Video", f, file_name=output_name, mime="video/avi")

        # Download log
        with open(log_path, "rb") as f:
            st.download_button("📝 Download Session Log (Video)", f, file_name=log_name, mime="text/plain")
