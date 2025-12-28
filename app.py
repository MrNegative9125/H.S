import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="ASL Real-Time Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: bold;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_asl_model(model_path, class_names_path):
    """Load the trained ASL model and class names"""
    try:
        model = keras.models.load_model(model_path)
        class_names = np.load(class_names_path, allow_pickle=True).tolist()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_frame(frame, img_size=(64, 64)):
    """Preprocess frame for model prediction"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    frame_resized = cv2.resize(frame_rgb, img_size)
    # Normalize
    frame_normalized = frame_resized / 255.0
    # Add batch dimension
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    return frame_batch

def draw_roi(frame, x, y, w, h, label="", confidence=0.0):
    """Draw region of interest and prediction on frame"""
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    # Draw background for text
    if label:
        cv2.rectangle(frame, (x, y-70), (x+w, y), (0, 255, 0), -1)
        
        # Draw prediction text
        cv2.putText(frame, f"{label}", (x+10, y-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(frame, f"{confidence:.1%}", (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        # Instructions
        cv2.putText(frame, "Show hand sign here", (x+20, y+h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Real-Time Recognition</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading
        st.subheader("Model Configuration")
        model_path = st.text_input("Model Path", value="best_asl_model.keras")
        class_names_path = st.text_input("Class Names Path", value="class_names.npy")
        
        # Camera settings
        st.subheader("Camera Settings")
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
        
        # ROI settings
        st.subheader("Detection Area")
        roi_size = st.slider("ROI Size", 200, 500, 350, 10)
        roi_x_offset = st.slider("Horizontal Position", -200, 200, 0, 10)
        roi_y_offset = st.slider("Vertical Position", -200, 200, 0, 10)
        
        # Confidence threshold
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
        
        # Prediction smoothing
        use_smoothing = st.checkbox("Use Prediction Smoothing", value=True)
        if use_smoothing:
            smoothing_frames = st.slider("Smoothing Frames", 1, 10, 5)
        
        # Display options
        st.subheader("Display Options")
        show_fps = st.checkbox("Show FPS", value=True)
        mirror_camera = st.checkbox("Mirror Camera", value=True)
        
        st.markdown("---")
        st.subheader("📖 Instructions")
        st.info("""
        **How to use:**
        1. Click 'Start Camera'
        2. Position your hand in the green box
        3. See real-time predictions
        
        **Tips:**
        - Good lighting improves accuracy
        - Keep hand centered in the box
        - Hold the sign steady for 1-2 seconds
        - Adjust ROI position if needed
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Current Prediction")
        prediction_placeholder = st.empty()
        
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()
        
        st.subheader("📝 Recent Predictions")
        history_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        start_button = st.button("▶️ Start Camera", key="start")
    with col_btn2:
        stop_button = st.button("⏹️ Stop Camera", key="stop")
    with col_btn3:
        clear_button = st.button("🗑️ Clear History", key="clear")
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'recent_predictions' not in st.session_state:
        st.session_state.recent_predictions = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'fps_list' not in st.session_state:
        st.session_state.fps_list = []
    
    # Load model
    model, class_names = load_asl_model(model_path, class_names_path)
    
    if model is None:
        st.error("❌ Please ensure the model and class names files exist!")
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Recognizing {len(class_names)} ASL letters.")
    
    # Button actions
    if start_button:
        st.session_state.camera_running = True
    
    if stop_button:
        st.session_state.camera_running = False
    
    if clear_button:
        st.session_state.prediction_history = []
        st.session_state.recent_predictions = []
        st.session_state.frame_count = 0
        st.session_state.fps_list = []
    
    # Camera loop
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera! Please check camera index.")
            st.session_state.camera_running = False
            st.stop()
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        stframe = video_placeholder.empty()
        
        try:
            while st.session_state.camera_running:
                start_time = time.time()
                
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to grab frame")
                    break
                
                # Mirror camera if enabled
                if mirror_camera:
                    frame = cv2.flip(frame, 1)
                
                st.session_state.frame_count += 1
                
                # Get frame dimensions
                height, width = frame.shape[:2]
                
                # Calculate ROI position (center with offsets)
                x = (width - roi_size) // 2 + roi_x_offset
                y = (height - roi_size) // 2 + roi_y_offset
                
                # Ensure ROI is within frame bounds
                x = max(0, min(x, width - roi_size))
                y = max(0, min(y, height - roi_size))
                
                # Extract ROI
                roi = frame[y:y+roi_size, x:x+roi_size]
                
                # Preprocess and predict
                if roi.size > 0:
                    processed_roi = preprocess_frame(roi)
                    predictions = model.predict(processed_roi, verbose=0)
                    
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                    predicted_letter = class_names[predicted_class_idx]
                    
                    # Prediction smoothing
                    if use_smoothing:
                        st.session_state.recent_predictions.append(predicted_letter)
                        if len(st.session_state.recent_predictions) > smoothing_frames:
                            st.session_state.recent_predictions.pop(0)
                        
                        # Most common prediction
                        if st.session_state.recent_predictions:
                            from collections import Counter
                            predicted_letter = Counter(st.session_state.recent_predictions).most_common(1)[0][0]
                    
                    # Draw ROI and prediction
                    if confidence > confidence_threshold:
                        frame = draw_roi(frame, x, y, roi_size, roi_size, predicted_letter, confidence)
                        
                        # Update prediction display
                        prediction_placeholder.markdown(f"""
                            <div class="prediction-box">
                                <div class="prediction-letter">{predicted_letter}</div>
                                <div class="confidence-text">Confidence: {confidence:.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Add to history
                        if (not st.session_state.prediction_history or 
                            st.session_state.prediction_history[-1][0] != predicted_letter):
                            st.session_state.prediction_history.append((predicted_letter, confidence))
                            if len(st.session_state.prediction_history) > 50:
                                st.session_state.prediction_history.pop(0)
                    else:
                        frame = draw_roi(frame, x, y, roi_size, roi_size)
                        prediction_placeholder.markdown("""
                            <div class="info-box">
                                <p style="text-align: center; font-size: 1.2rem;">
                                    Low Confidence<br>
                                    <small>Adjust hand position</small>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    frame = draw_roi(frame, x, y, roi_size, roi_size)
                
                # Calculate and display FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                st.session_state.fps_list.append(fps)
                if len(st.session_state.fps_list) > 30:
                    st.session_state.fps_list.pop(0)
                avg_fps = np.mean(st.session_state.fps_list)
                
                if show_fps:
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (width - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add instructions at top
                cv2.putText(frame, "Position hand in green box", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Display frame
                frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_display, channels="RGB", use_container_width=True)
                
                # Update statistics
                stats_placeholder.markdown(f"""
                    <div class="info-box">
                        <p><strong>📊 Frames Processed:</strong> {st.session_state.frame_count}</p>
                        <p><strong>⚡ Average FPS:</strong> {avg_fps:.1f}</p>
                        <p><strong>📝 Total Predictions:</strong> {len(st.session_state.prediction_history)}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Update history
                if st.session_state.prediction_history:
                    history_text = "### Recent Letters:\n\n"
                    for i, (letter, conf) in enumerate(reversed(st.session_state.prediction_history[-15:])):
                        history_text += f"{i+1}. **{letter}** ({conf:.1%})\n\n"
                    history_placeholder.markdown(history_text)
                
                # Small delay
                time.sleep(0.01)
                
        finally:
            cap.release()
    
    else:
        video_placeholder.info("📹 Camera is stopped. Click 'Start Camera' to begin recognition.")
        prediction_placeholder.markdown("""
            <div class="info-box">
                <p style="text-align: center; font-size: 1.2rem;">
                    ⏸️ Waiting for camera...
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
