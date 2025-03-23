import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(page_title='🚀 YOLO11 Object Detection Dashboard', layout='wide')
st.markdown("""
    <style>
        body {background-color: #121212; color: white;}
        .stApp {background-color: #1E1E1E;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 YOLO11 Enhanced Object Detection Dashboard")
st.sidebar.header("⚙️ Settings & Controls")

# Sidebar options
model_option = st.sidebar.selectbox("Select YOLO11 Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

def process_image(image):
    results = model(image, conf=confidence_threshold)
    detections = results[0].boxes
    output_img = results[0].plot() if show_boxes else np.array(image)
    return output_img, detections

# Load Model
model = YOLO(model_option)
st.sidebar.success(f"✅ Loaded {model_option}")

upload_type = st.radio("📥 Choose Input Type", ("Image", "Video", "Webcam"))

if upload_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Detect Objects"):
            output_img, detections = process_image(image)
            st.image(output_img, caption="Detected Objects", use_container_width=True)

elif upload_type == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st.video(tfile.name)
        stframe = st.empty()
        frame_count = 0
        object_counts = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frame, detections = process_image(frame)
            stframe.image(output_frame, channels="RGB")
            frame_count += 1
            object_counts.append(len(detections))
            time.sleep(0.03)
        
        cap.release()
        
        # Visualization of Object Counts
        df = pd.DataFrame({"Frame": list(range(1, frame_count+1)), "Objects Detected": object_counts})
        fig = px.line(df, x="Frame", y="Objects Detected", title="📈 Object Count Per Frame")
        st.plotly_chart(fig, use_container_width=True)

elif upload_type == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.sidebar.warning("Press 'q' to Stop Webcam")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame, detections = process_image(frame)
        stframe.image(output_frame, channels="RGB")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

st.sidebar.info("💡 Supports Image, Video, and Webcam Inputs!")
