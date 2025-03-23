
import streamlit as st
import cv2
import numpy as np
import os
import time
import datetime
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from streamlit_folium import folium_static
import folium
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="AI-Driven Disaster Management System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2D2D2D;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    .detection-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: rgba(76, 175, 80, 0.1);
    }
    .alert-box {
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: rgba(244, 67, 54, 0.1);
    }
    .metric-card {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 14px;
        color: #E0E0E0;
    }
    div[data-testid="stSidebar"] {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #ff2e2e;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
local_css()

# Initialize session state variables
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []

if 'start_time' not in st.session_state:
    st.session_state.start_time = None

if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}

if 'detection_counts' not in st.session_state:
    st.session_state.detection_counts = {
        'fire': 0,
        'flood': 0,
        'collapsed_building': 0,
        'injured_person': 0,
        'debris': 0,
        'landslide': 0,
        'smoke': 0
    }

if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False

# Class mapping for our disaster detection model
class_mapping = {
    0: "fire",
    1: "flood",
    2: "collapsed_building",
    3: "injured_person",
    4: "debris",
    5: "landslide",
    6: "smoke"
}

# Color mapping for visualization
color_mapping = {
    "fire": (255, 0, 0),        # Red
    "flood": (0, 0, 255),       # Blue
    "collapsed_building": (165, 42, 42),  # Brown
    "injured_person": (255, 0, 255),  # Magenta
    "debris": (128, 128, 128),  # Gray
    "landslide": (165, 42, 42), # Brown
    "smoke": (192, 192, 192)    # Silver
}

# Sample coordinates for demonstration (can be replaced with real GPS data)
sample_coordinates = {
    "fire": [(28.6139, 77.2090), (19.0760, 72.8777), (22.5726, 88.3639)],
    "flood": [(13.0827, 80.2707), (17.3850, 78.4867), (23.8315, 91.2868)],
    "collapsed_building": [(26.9124, 75.7873), (25.5941, 85.1376)],
    "injured_person": [(27.1767, 78.0081), (11.9416, 79.8083)],
    "landslide": [(32.2396, 77.1887), (30.7333, 79.0667)],
    "debris": [(25.5941, 85.1376), (28.6139, 77.2090)],
    "smoke": [(19.0760, 72.8777), (13.0827, 80.2707)]
}

# Function to log detections
def log_detection(label, confidence, frame=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save thumbnail of detection (optional)
    thumbnail_path = None
    if frame is not None:
        # Create thumbnails directory if it doesn't exist
        os.makedirs("thumbnails", exist_ok=True)
        thumbnail_path = f"thumbnails/detection_{timestamp.replace(':', '-')}.jpg"
        cv2.imwrite(thumbnail_path, frame)
    
    # Add to detection history
    st.session_state.detection_history.append({
        "timestamp": timestamp,
        "label": label,
        "confidence": confidence,
        "thumbnail": thumbnail_path
    })
    
    # Update detection counts
    if label in st.session_state.detection_counts:
        st.session_state.detection_counts[label] += 1

# Function to generate alert
def generate_alert(label, confidence, location="Unknown"):
    # Check if we should generate an alert (avoid too frequent alerts for the same issue)
    current_time = time.time()
    
    # Only alert if it's been more than 30 seconds since the last alert for this type
    if label not in st.session_state.last_alert_time or \
       (current_time - st.session_state.last_alert_time[label]) > 30:
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = f"⚠️ ALERT: {label.upper()} detected with {confidence:.2f} confidence at {location}"
        
        st.session_state.alert_history.append({
            "timestamp": timestamp,
            "label": label,
            "confidence": confidence,
            "message": alert_message,
            "location": location
        })
        
        st.session_state.last_alert_time[label] = current_time
        
        # In a real application, here you would:
        # 1. Send SMS via Twilio
        # 2. Send emails
        # 3. Trigger emergency response systems
        return True
    
    return False

# Fix for the detect_objects function - ensure RGBA to RGB conversion is working
def detect_objects(model, image, confidence_threshold=0.25):
    # For PIL Image, convert to numpy array
    if isinstance(image, Image.Image):
        # Convert PIL Image to RGB mode first, then to numpy array
        image = image.convert('RGB')
        image_np = np.array(image)
    else:
        image_np = image.copy()  # Make a copy to avoid modifying the original
    
    # Check if the image has 4 channels (RGBA) and convert to 3 channels (RGB)
    if len(image_np.shape) == 3 and image_np.shape[2] == 4:
        # Convert RGBA to RGB by removing the alpha channel
        image_np = image_np[:, :, :3]
        
    # Run detection with YOLO
    results = model(image_np, conf=confidence_threshold)[0]
    
    detections = []
    
    # Process results - YOLO format
    if len(results.boxes) > 0:
        for box in results.boxes:
            confidence = float(box.conf)
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0].item())
                
                # Map class_id to label
                label = class_mapping.get(class_id, f"Class {class_id}")
                
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })
    
    return detections, results

# Function to draw bounding boxes and labels on image
def draw_detections(image, detections):
    img_copy = image.copy()
    
    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        
        # Get color for this label
        color = color_mapping.get(label, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text = f"{label}: {confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_copy

# Function to create a heatmap of detections
def create_detection_heatmap():
    # Convert detection counts to DataFrame
    df = pd.DataFrame({
        'Category': list(st.session_state.detection_counts.keys()),
        'Count': list(st.session_state.detection_counts.values())
    })
    
    # Create heatmap with Plotly
    fig = px.bar(df, x='Category', y='Count', color='Count',
                color_continuous_scale=px.colors.sequential.Reds,
                title="Disaster Detection Frequency")
    
    fig.update_layout(
        xaxis_title="Disaster Type",
        yaxis_title="Detection Count",
        height=400,
        template="plotly_dark"
    )
    
    return fig

# Function to create a map with detection markers
def create_detection_map():
    # Create a map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB dark_matter")
    
    # Add markers for each type of disaster with appropriate colors
    for disaster_type, coords_list in sample_coordinates.items():
        count = st.session_state.detection_counts.get(disaster_type, 0)
        
        if count > 0:
            # Map our color codes to hex for folium
            color_map = {
                "fire": "red",
                "flood": "blue",
                "collapsed_building": "brown",
                "injured_person": "purple",
                "debris": "gray",
                "landslide": "brown",
                "smoke": "gray"
            }
            
            color = color_map.get(disaster_type, "green")
            
            # Add markers for this disaster type
            for coord in coords_list:
                folium.Marker(
                    location=coord,
                    popup=f"{disaster_type.title()}: {count} detections",
                    icon=folium.Icon(color=color, icon="info-sign")
                ).add_to(m)
                
                # Add circle to highlight the area
                folium.Circle(
                    location=coord,
                    radius=5000,  # 5 km radius
                    color=color,
                    fill=True,
                    fill_opacity=0.2
                ).add_to(m)
    
    return m

# Function to process video stream
def process_video_stream(model, video_source, confidence_threshold, display_area):
    cap = cv2.VideoCapture(video_source)
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Display video stats
    st.sidebar.info(f"Video dimensions: {width}x{height}, {fps} FPS")
    
    # Start time
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
    
    # Create image display area
    img_placeholder = display_area.empty()
    status_placeholder = display_area.empty()
    
    # Create alerts section
    alert_placeholder = st.sidebar.empty()
    
    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame for better performance
    
    while cap.isOpened() and st.session_state.get('run_detection', True):
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Error: Could not read from video source")
            break
        
        frame_count += 1
        
        # Only process every nth frame for efficiency
        if frame_count % process_every_n_frames == 0:
            # Run detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, results = detect_objects(model, rgb_frame, confidence_threshold)
            
            # Draw bounding boxes
            if len(detections) > 0:
                annotated_frame = draw_detections(frame, detections)
            else:
                annotated_frame = frame
            
            # Convert back to RGB for displaying in Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            img_placeholder.image(annotated_frame_rgb, use_container_width=True)
            
            # Update status
            elapsed_time = time.time() - st.session_state.start_time
            status_text = (f"Status: Monitoring | "
                          f"Elapsed time: {int(elapsed_time)}s | "
                          f"Detections: {sum(st.session_state.detection_counts.values())}")
            status_placeholder.info(status_text)
            
            # Process detections for logging and alerts
            for detection in detections:
                label = detection["label"]
                confidence = detection["confidence"]
                
                # Log all detections
                log_detection(label, confidence, frame)
                
                # Generate alerts for high-confidence detections
                if confidence > 0.5:
                    # Select a random location from our sample locations
                    import random
                    location = "Unknown"
                    if label in sample_coordinates and sample_coordinates[label]:
                        lat, lon = random.choice(sample_coordinates[label])
                        location = f"{lat:.4f}, {lon:.4f}"
                    
                    alert_generated = generate_alert(label, confidence, location)
            
            # Update alerts display
            alert_text = ""
            for alert in st.session_state.alert_history[-5:]:  # Show last 5 alerts
                alert_text += f"⚠️ {alert['label'].upper()}: {alert['timestamp']} ({alert['confidence']:.2f}) at {alert['location']}\n\n"
            
            if alert_text:
                alert_placeholder.error(f"### Recent Alerts\n{alert_text}")
        
        # Small delay to prevent UI freezing
        time.sleep(0.01)
    
    cap.release()

# Main application layout
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/shield.png", width=80)
        st.title("Disaster Management")
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Live Detection", "Analytics", "Settings", "About"],
            icons=["speedometer2", "camera-video", "graph-up", "gear", "info-circle"],
            default_index=0,
        )
        
        # Add model selection
        st.subheader("Model Selection")
        model_option = st.selectbox(
            "Select YOLO Model",
            ["yolo11n.pt (Enhanced)", "YOLOv8n (Legacy)", "YOLOv8s (Legacy)"]
        )
        
        # Map model selection to actual model path
        model_mapping = {
            "yolo11n.pt (Enhanced)": "yolo11n.pt",
            "YOLOv8n (Legacy)": "yolov8n.pt",
            "YOLOv8s (Legacy)": "yolov8s.pt"
        }
        
        model_path = model_mapping[model_option]
        
        # Confidence threshold selection
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Minimum confidence score to consider a detection valid"
        )
        
        # Show detection history toggle
        show_history = st.checkbox("Show Detection History", value=True)
        
        # Night mode toggle
        dark_mode = st.checkbox("Dark Mode", value=True)
    
    # Main content area based on selected menu
    if selected == "Dashboard":
        st.title("🌍 AI-Driven Disaster Management Dashboard")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sum(st.session_state.detection_counts.values())}</div>
                <div class="metric-label">Total Detections</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.alert_history)}</div>
                <div class="metric-label">Alerts Generated</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            elapsed_time = 0
            if st.session_state.start_time:
                elapsed_time = int(time.time() - st.session_state.start_time)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{elapsed_time} s</div>
                <div class="metric-label">Monitoring Duration</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add visualization sections
        st.subheader("Disaster Detection Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create and display detection heatmap
            heatmap_fig = create_detection_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
        with col2:
            # Display active alerts
            st.subheader("Active Alerts")
            if st.session_state.alert_history:
                for alert in reversed(st.session_state.alert_history[-5:]):
                    st.markdown(f"""
                    <div class="alert-box">
                        <h4>⚠️ {alert['label'].upper()} Detected</h4>
                        <p>Time: {alert['timestamp']}</p>
                        <p>Confidence: {alert['confidence']:.2f}</p>
                        <p>Location: {alert['location']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No alerts generated yet.")
        
        # Map section
        st.subheader("Disaster Map")
        detection_map = create_detection_map()
        from streamlit_folium import st_folium
        st_folium(detection_map, width=1000, height=500)
        
        # Show history if enabled
        if show_history and st.session_state.detection_history:
            st.subheader("Recent Detections")
            history_df = pd.DataFrame(st.session_state.detection_history)
            st.dataframe(history_df, use_container_width=True)

    elif selected == "Live Detection":
     st.title("📹 Live Disaster Detection with YOLOv11")
    
    # Input source selection
    source_option = st.radio(
        "Select Input Source",
        ["Webcam", "Upload Video", "Upload Image", "Sample Video"],
        horizontal=True
    )
    
    # Create area for video display
    video_display_area = st.container()
    
    # Try to load the model only once
    try:
        # Load the selected model
        model = YOLO(model_path)
        st.success(f"Model {model_path} loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Handle different input sources
    if source_option == "Upload Video":
        # For uploaded video
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        if video_file is not None:
            # Save uploaded file to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(video_file.read())
            temp_file.close()  # Close the file before processing
            
            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                start_button = st.button("▶️ Start Detection", use_container_width=True)
            with col2:
                stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
            
            if start_button:
                st.session_state.run_detection = True
                with st.spinner("Processing video..."):
                    try:
                        # Process video
                        process_video_stream(model, temp_file.name, confidence_threshold, video_display_area)
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                    finally:
                        # Clean up
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
            
            if stop_button:
                st.session_state.run_detection = False
                st.info("Detection stopped")
    
    elif source_option == "Upload Image":
        # For uploaded image
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            # Display a spinner while processing
            with st.spinner("Processing image..."):
                try:
                    # Load image using PIL
                    image = Image.open(image_file)
                    
                    # Explicitly convert to RGB mode
                    image = image.convert('RGB')
                    
                    # Display original image
                    st.subheader("Original Image")
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    
                    # Convert PIL image to numpy array for processing
                    image_np = np.array(image)
                    
                    # Run detection
                    detections, results = detect_objects(model, image_np, confidence_threshold)
                    
                    # Convert detections to our format for logging
                    detections = []
                    if len(results.boxes) > 0:
                        for box in results.boxes:
                            confidence = float(box.conf)
                            if confidence >= confidence_threshold:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                class_id = int(box.cls[0].item())
                                
                                # Map class_id to label
                                label = class_mapping.get(class_id, f"Class {class_id}")
                                
                                detections.append({
                                    "label": label,
                                    "confidence": confidence,
                                    "bbox": (x1, y1, x2, y2)
                                })
                    
                    # Display results
                    st.subheader("Detection Results")
                    
                    # Get the annotated image from results
                    annotated_image = results.plot()
                    
                    # Display the annotated image
                    st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                    
                    # Display detection parameters
                    if detections:
                        st.subheader("Detection Parameters")
                        detection_df = pd.DataFrame([
                            {
                                "Disaster Type": d["label"],
                                "Confidence": f"{d['confidence']:.2f}",
                                "Bounding Box": f"({d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]})"
                            } for d in detections
                        ])
                        st.dataframe(detection_df, use_container_width=True)
                    else:
                        st.info("No disasters detected in the image.")
                    
                    # Log detections
                    for detection in detections:
                        log_detection(detection["label"], detection["confidence"])
                        
                        # Generate alerts for high-confidence detections
                        if detection["confidence"] > 0.5:
                            import random
                            location = "Unknown"
                            label = detection["label"]
                            if label in sample_coordinates and sample_coordinates[label]:
                                lat, lon = random.choice(sample_coordinates[label])
                                location = f"{lat:.4f}, {lon:.4f}"
                            generate_alert(label, detection["confidence"], location)
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    elif source_option == "Webcam":
        # Control buttons for webcam
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Start Detection", use_container_width=True)
        with col2:
            stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
        
        if start_button:
            st.session_state.run_detection = True
            try:
                # For webcam
                process_video_stream(model, 0, confidence_threshold, video_display_area)
            except Exception as e:
                st.error(f"Error accessing webcam: {str(e)}")
        
        if stop_button:
            st.session_state.run_detection = False
            st.info("Detection stopped")
    
    elif source_option == "Sample Video":
        # Control buttons for sample video
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("▶️ Start Detection", use_container_width=True)
        with col2:
            stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
        
        if start_button:
            st.session_state.run_detection = True
            # Use a sample video for demo purposes
            sample_video = "sample_video.mp4"
            if not os.path.exists(sample_video):
                st.warning("Sample video not found. Using a demo video stream instead.")
                # You could use an online video stream URL here as fallback
                demo_video = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
                process_video_stream(model, demo_video, confidence_threshold, video_display_area)
            else:
                process_video_stream(model, sample_video, confidence_threshold, video_display_area)
        
        if stop_button:
            st.session_state.run_detection = False
            st.info("Detection stopped")
    elif selected == "Analytics":
        st.title("📊 Disaster Analytics")
        
        # Display charts based on collected data
        if sum(st.session_state.detection_counts.values()) > 0:
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Detection Statistics", "Temporal Analysis", "Spatial Distribution"])
            
            with tab1:
                # Pie chart of detection types
                labels = list(st.session_state.detection_counts.keys())
                values = list(st.session_state.detection_counts.values())
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="Distribution of Disaster Types",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Bar chart of detection counts
                fig = px.bar(
                    x=labels,
                    y=values,
                    title="Disaster Detection Counts",
                    labels={"x": "Disaster Type", "y": "Count"},
                    color=values,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Temporal Analysis of Detections")
                
                # Convert detection history to DataFrame for temporal analysis
                if st.session_state.detection_history:
                    history_df = pd.DataFrame(st.session_state.detection_history)
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                    
                    # Group by hour and disaster type
                    history_df['hour'] = history_df['timestamp'].dt.hour
                    hourly_counts = history_df.groupby(['hour', 'label']).size().reset_index(name='count')
                    
                    # Create line chart
                    fig = px.line(
                        hourly_counts,
                        x='hour',
                        y='count',
                        color='label',
                        title="Hourly Detection Trends",
                        labels={"hour": "Hour of Day", "count": "Number of Detections"},
                        markers=True,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create heatmap by hour and type
                    pivot_df = hourly_counts.pivot(index='label', columns='hour', values='count').fillna(0)
                    fig = px.imshow(
                        pivot_df,
                        labels=dict(x="Hour of Day", y="Disaster Type", color="Count"),
                        title="Hourly Detection Heatmap",
                        color_continuous_scale="Viridis",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough detection data for temporal analysis.")
            
            with tab3:
                st.subheader("Spatial Distribution of Disasters")
                detection_map = create_detection_map()
                folium_static(detection_map, width=800, height=600)
                
                st.info("Note: This map shows simulated locations for demonstration purposes. In a real system, GPS coordinates would be used.")
                
                # Add a 3D visualization
                st.subheader("3D Disaster Distribution")
                
                # Create sample data for 3D visualization
                disaster_types = list(st.session_state.detection_counts.keys())
                x_coords = []
                y_coords = []
                z_values = []
                labels = []
                
                for disaster_type in disaster_types:
                    count = st.session_state.detection_counts.get(disaster_type, 0)
                    if count > 0 and disaster_type in sample_coordinates:
                        for lat, lon in sample_coordinates[disaster_type]:
                            x_coords.append(lon)
                            y_coords.append(lat)
                            z_values.append(count)
                            labels.append(disaster_type)
                
                if x_coords:  # Only create visualization if we have data
                    fig = go.Figure(data=[go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_values,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=z_values,
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        text=labels,
                        hoverinfo='text'
                    )])
                    
                    fig.update_layout(
                        title="3D Disaster Visualization",
                        scene=dict(
                            xaxis_title='Longitude',
                            yaxis_title='Latitude',
                            zaxis_title='Detection Count'
                        ),
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("No detection data available for analysis. Start detection to gather data.")
    
    elif selected == "Settings":
        st.title("⚙️ System Settings")
        
        # Create settings tabs
        tab1, tab2, tab3 = st.tabs(["Detection Settings", "Alert Configuration", "System Configuration"])
        
        with tab1:
            st.subheader("Detection Parameters")
            
            # Detection classes to monitor
            st.multiselect(
                    "Disaster Types to Monitor",
                    options=list(class_mapping.values()),
                    default=list(class_mapping.values()),
                    help="Select which types of disasters to detect and alert on"
                )
                
            # IOU threshold
            iou_threshold = st.slider(
                    "IOU Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.45,
                    step=0.05,
                    help="Intersection over Union threshold for NMS"
                )
                
            # Frame processing rate
            frame_rate = st.slider(
                    "Process every N frames",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Higher values improve performance but may miss brief events"
                )
                
            # Additional settings
            st.checkbox("Enable object tracking", value=False, 
                           help="Track objects across frames (requires more processing power)")
            st.checkbox("Save detection images", value=True, 
                           help="Save images of detected disasters")
            
        with tab2:
            st.subheader("Alert Configuration")
            
            # Alert threshold
            alert_threshold = st.slider(
                "Alert Confidence Threshold",
                min_value=0.3,
                max_value=0.95,
                value=0.5,
                step=0.05,
                help="Minimum confidence to trigger alerts (higher = fewer false alarms)"
            )
            
            # Alert cooldown
            alert_cooldown = st.slider(
                "Alert Cooldown Period (seconds)",
                min_value=10,
                max_value=300,
                value=30,
                step=10,
                help="Minimum time between consecutive alerts of the same type"
            )
            
            # Notification methods
            st.subheader("Notification Methods")
            
            st.checkbox("Enable Email Alerts", value=True)
            email_recipients = st.text_input(
                "Email Recipients (comma separated)",
                "emergency@example.com, response@example.com"
            )
            
            st.checkbox("Enable SMS Alerts", value=True)
            sms_recipients = st.text_input(
                "SMS Recipients (comma separated)",
                "+1234567890, +1098765432"
            )
            
            st.checkbox("Enable API Webhook", value=False)
            webhook_url = st.text_input(
                "Webhook URL",
                "https://api.example.com/emergency-alerts"
            )
        
        with tab3:
            st.subheader("System Configuration")
            
            # Video processing
            st.slider(
                "Video Processing Resolution",
                min_value=240,
                max_value=1080,
                value=720,
                step=120,
                help="Higher resolution provides better detection but requires more processing power"
            )
            
            # Database settings
            st.selectbox(
                "Database Storage",
                options=["Local SQLite", "Cloud Database", "No Storage"]
            )
            
            # Model settings
            st.checkbox("Enable model auto-update", value=True,
                       help="Automatically update the model when new versions are available")
            
            # System resources
            st.slider(
                "Maximum CPU Usage (%)",
                min_value=10,
                max_value=100,
                value=70,
                step=10
            )
            
            st.slider(
                "Maximum Memory Usage (MB)",
                min_value=512,
                max_value=8192,
                value=2048,
                step=512
            )
            
            # Save and reset buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button("Save Settings", use_container_width=True)
            with col2:
                st.button("Reset to Defaults", use_container_width=True)
    
    elif selected == "About":
        st.title("ℹ️ About AI-Driven Disaster Management System")
        
        st.markdown("""
        ### Overview
        
        This advanced AI-driven disaster management system uses state-of-the-art object detection
        algorithms to detect various types of disasters in real-time from video streams, images, and 
        camera feeds. The system is designed to provide early warnings, situational awareness, and
        decision support during emergency situations.
        
        ### Features
        
        - **Real-time Detection**: Identifies fires, floods, collapsed buildings, injured persons,
          debris, landslides, and smoke using YOLOv11n.
        - **Multi-source Input**: Process webcam feeds, uploaded videos, or images.
        - **Alerting System**: Generates alerts with customizable thresholds and notification methods.
        - **Analytics Dashboard**: Visualize disaster trends, patterns, and spatial distribution.
        - **Geographic Mapping**: Plot detected disasters on interactive maps.
        - **Historical Analysis**: Track and analyze detection data over time.
        
        ### Technology Stack
        
        - **AI Model**: YOLOv11n real-time object detection
        - **Frontend**: Streamlit interactive dashboard
        - **Visualization**: Plotly, Matplotlib, Folium
        - **Computer Vision**: OpenCV, PIL
        - **Data Processing**: Pandas, NumPy
        
        ### Future Enhancements
        
        - Integration with IoT sensor networks
        - Drone-based surveillance capabilities
        - Predictive analytics for disaster risk assessment
        - Mobile application for field responders
        - AI-powered resource allocation recommendations
        
        ### Credits
        
        Developed as part of advanced disaster management research project.
        """)
        
        # Team information
        st.subheader("Development Team")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Project Lead**
            - Dr. Aisha Johnson
            - Disaster Management Specialist
            """)
        
        with col2:
            st.markdown("""
            **AI Development**
            - Dr. Raj Patel
            - Computer Vision Expert
            """)
        
        with col3:
            st.markdown("""
            **UI/UX Design**
            - Maria Rodriguez
            - Data Visualization Specialist
            """)
        
        # Version information
        st.info("Version 1.0.0 | Last Updated: March 2025")
        
        # Contact information
        st.subheader("Contact")
        st.markdown("For support or inquiries: support@disaster-ai.example.com")

# Run the application
if __name__ == "__main__":
    main()