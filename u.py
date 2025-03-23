import cv2
import numpy as np
import torch
import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import time

# Load the YOLOv8 Model (Trained on Disaster Dataset)
model = YOLO("yolov8n.pt")  # Replace with your custom-trained model if available

# Twilio Credentials (For SMS Alerts)
TWILIO_SID = "YOUR_TWILIO_SID"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_PHONE = "YOUR_TWILIO_PHONE_NUMBER"
ALERT_PHONE = "RECIPIENT_PHONE_NUMBER"

# Function to send SMS alerts
def send_alert(message):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(body=message, from_=TWILIO_PHONE, to=ALERT_PHONE)

# Streamlit UI
st.title("🌍 AI-Driven Disaster Management System 🚀")
st.sidebar.title("Settings")

# Select Source: Webcam or Video File
source_option = st.sidebar.radio("Select Source", ("Webcam", "Upload Video"))
if source_option == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

st.sidebar.subheader("Detection Threshold")
threshold = st.sidebar.slider("Set Confidence Threshold", 0.1, 1.0, 0.5)

# Start Detection
start_button = st.sidebar.button("Start Detection")

if start_button:
    if source_option == "Webcam":
        cap = cv2.VideoCapture(0)  # Open Webcam
    else:
        if video_file is not None:
            cap = cv2.VideoCapture(video_file.name)
        else:
            st.warning("Please upload a video file")
            st.stop()

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 on the frame
        results = model(frame)[0]

        # Draw bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = results.names[cls]

            if confidence > threshold:
                color = (0, 255, 0)  # Green
                if label in ["fire", "flood", "collapsed_building"]:
                    color = (0, 0, 255)  # Red for danger alerts
                    send_alert(f"🚨 Disaster Alert: {label.upper()} detected!")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Add a small delay
        time.sleep(0.03)

    cap.release()
