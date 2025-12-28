import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="ASL Recognition - Live",
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
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .word-display {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #0d47a1;
        letter-spacing: 0.3rem;
        margin: 1rem 0;
        min-height: 80px;
    }
    </style>
""", unsafe_allow_html=True)


class ASLRecognizer:
    """Real-time ASL Recognition System"""
    
    def __init__(self, model_path='asl_model_final.keras', class_names_path='class_names.npy'):
        """Initialize the recognizer with trained model"""
        self.model = None
        self.class_names = []
        self.img_size = (64, 64)
        self.prediction_history = deque(maxlen=30)
        self.confidence_threshold = 0.7
        
        # Word formation
        self.current_word = []
        self.last_confirmed_letter = None
        self.last_confirmation_time = 0
        self.confirmation_delay = 1.5
        self.hold_time = {}
        
        # Statistics
        self.session_start = time.time()
        
        # Load model
        self.load_model(model_path, class_names_path)
    
    def load_model(self, model_path, class_names_path):
        """Load the trained model and class names"""
        try:
            self.model = keras.models.load_model(model_path)
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        img_resized = cv2.resize(frame, self.img_size)
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    
    def predict(self, frame):
        """Make prediction on a frame"""
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_letter = self.class_names[predicted_class_idx]
        
        self.prediction_history.append({
            'letter': predicted_letter,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        return predicted_letter, confidence, predictions[0]
    
    def get_smoothed_prediction(self, window_size=5):
        """Get smoothed prediction based on recent history"""
        if len(self.prediction_history) < window_size:
            return None, 0.0
        
        recent_predictions = list(self.prediction_history)[-window_size:]
        letter_counts = {}
        confidence_sum = {}
        
        for pred in recent_predictions:
            letter = pred['letter']
            conf = pred['confidence']
            
            if letter not in letter_counts:
                letter_counts[letter] = 0
                confidence_sum[letter] = 0.0
            
            letter_counts[letter] += 1
            confidence_sum[letter] += conf
        
        most_common = max(letter_counts, key=letter_counts.get)
        avg_confidence = confidence_sum[most_common] / letter_counts[most_common]
        
        return most_common, avg_confidence
    
    def update_word_formation(self, letter, confidence):
        """Update word formation based on confirmed letters"""
        current_time = time.time()
        
        if letter not in self.hold_time:
            self.hold_time[letter] = current_time
        
        hold_duration = current_time - self.hold_time.get(letter, current_time)
        
        if (confidence >= self.confidence_threshold and 
            hold_duration >= 1.0 and
            letter != self.last_confirmed_letter and
            (current_time - self.last_confirmation_time) >= self.confirmation_delay):
            
            self.current_word.append(letter)
            self.last_confirmed_letter = letter
            self.last_confirmation_time = current_time
            self.hold_time = {letter: current_time}
            return True
        
        if hold_duration > 2.0:
            self.hold_time.pop(letter, None)
        
        return False
    
    def add_space(self):
        """Add space to current word"""
        if self.current_word and self.current_word[-1] != ' ':
            self.current_word.append(' ')
    
    def clear_word(self):
        """Clear current word"""
        self.current_word = []
        self.last_confirmed_letter = None
    
    def delete_last_letter(self):
        """Delete last letter from word"""
        if self.current_word:
            self.current_word.pop()
    
    def get_word(self):
        """Get current word as string"""
        return ''.join(self.current_word)
    
    def draw_prediction_on_frame(self, frame, letter, confidence):
        """Draw prediction overlay on frame"""
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw prediction box
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (255, 255, 255), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw prediction text
        cv2.putText(frame, f"Prediction: {letter}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (31, 119, 180), 3)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        
        # Draw confidence bar
        bar_width = int((width - 40) * confidence)
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
        cv2.rectangle(frame, (20, 105), (20 + bar_width, 115), color, -1)
        
        # Draw ROI box
        roi_size = 300
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size),
                     (31, 119, 180), 3)
        cv2.putText(frame, "Place hand here", (roi_x + 50, roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (31, 119, 180), 2)
        
        return frame, (roi_x, roi_y, roi_size, roi_size)


def create_confidence_chart(predictions_probs, class_names):
    """Create a bar chart of top predictions using matplotlib"""
    # Get top 5 predictions
    top_indices = np.argsort(predictions_probs)[-5:][::-1]
    top_letters = [class_names[i] for i in top_indices]
    top_probs = [predictions_probs[i] for i in top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(top_letters, top_probs, color='#1f77b4')
    
    # Color bars based on confidence
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        if prob > 0.7:
            bar.set_color('#2ecc71')
        elif prob > 0.5:
            bar.set_color('#3498db')
        else:
            bar.set_color('#e74c3c')
    
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_ylabel('Letter', fontsize=10)
    ax.set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🤟 ASL Alphabet Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Sign Language Detection</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Model Files")
        model_path = st.text_input("Model Path", "asl_model_final.keras")
        class_names_path = st.text_input("Class Names Path", "class_names.npy")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence to display prediction"
        )
        
        smoothing_window = st.slider(
            "Smoothing Window",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of frames to average"
        )
        
        st.subheader("Camera Settings")
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
        
        st.markdown("---")
        
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. Click **'Start Camera'**
        2. Position hand in blue box
        3. Make ASL sign
        4. Hold for 1 second to add to word
        
        **Tips:**
        - Good lighting helps
        - Keep hand centered
        - Hold sign steady
        """)
        
        st.markdown("---")
        st.info("💡 **Word Formation**: Hold a sign for 1 second to add it to the word!")
    
    # Initialize recognizer
    if 'recognizer' not in st.session_state:
        with st.spinner("Loading model..."):
            st.session_state.recognizer = ASLRecognizer(model_path, class_names_path)
            if st.session_state.recognizer.model is not None:
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Check file paths.")
                return
    
    recognizer = st.session_state.recognizer
    recognizer.confidence_threshold = confidence_threshold
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        camera_placeholder = st.empty()
        
    with col2:
        st.subheader("🎯 Current Prediction")
        prediction_placeholder = st.empty()
        
        st.subheader("📈 Top Predictions")
        chart_placeholder = st.empty()
    
    # Word display
    st.markdown("---")
    st.subheader("📝 Formed Word")
    word_placeholder = st.empty()
    
    # Word control buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("➕ Add Space", use_container_width=True):
            recognizer.add_space()
    with btn_col2:
        if st.button("⌫ Delete Letter", use_container_width=True):
            recognizer.delete_last_letter()
    with btn_col3:
        if st.button("🗑️ Clear Word", use_container_width=True):
            recognizer.clear_word()
    with btn_col4:
        word_text = recognizer.get_word()
        if word_text:
            st.download_button(
                "💾 Save Word",
                data=word_text,
                file_name=f"asl_word_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    total_predictions = stats_col1.empty()
    avg_confidence = stats_col2.empty()
    fps_display = stats_col3.empty()
    letters_count = stats_col4.empty()
    
    # Control buttons
    st.markdown("---")
    control_col1, control_col2, control_col3 = st.columns([1, 1, 3])
    
    with control_col1:
        start_button = st.button("🎥 Start Camera", use_container_width=True)
    with control_col2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    if start_button:
        st.session_state.camera_running = True
    
    if stop_button:
        st.session_state.camera_running = False
    
    # Camera loop
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error("❌ Could not open camera. Check camera index.")
            st.session_state.camera_running = False
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while st.session_state.camera_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to read frame")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get ROI for prediction
                height, width = frame_rgb.shape[:2]
                roi_size = 300
                roi_x = (width - roi_size) // 2
                roi_y = (height - roi_size) // 2
                roi = frame_rgb[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                
                # Predict
                letter, confidence, predictions_probs = recognizer.predict(roi)
                
                # Get smoothed prediction
                smoothed_letter, smoothed_confidence = recognizer.get_smoothed_prediction(smoothing_window)
                
                if smoothed_letter is not None:
                    display_letter = smoothed_letter
                    display_confidence = smoothed_confidence
                else:
                    display_letter = letter
                    display_confidence = confidence
                
                # Update word formation
                letter_added = recognizer.update_word_formation(display_letter, display_confidence)
                
                # Draw on frame
                annotated_frame, _ = recognizer.draw_prediction_on_frame(
                    frame_rgb, display_letter, display_confidence
                )
                
                # Display camera feed
                camera_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # Display prediction
                if display_confidence >= confidence_threshold:
                    prediction_placeholder.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-letter">{display_letter}</div>
                            <div class="confidence-text">{display_confidence:.1%} confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    prediction_placeholder.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-letter">-</div>
                            <div class="confidence-text">Low confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display confidence chart
                fig = create_confidence_chart(predictions_probs, recognizer.class_names)
                chart_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Display current word
                current_word = recognizer.get_word()
                if current_word:
                    word_placeholder.markdown(f"""
                        <div class="word-display">{current_word}</div>
                    """, unsafe_allow_html=True)
                else:
                    word_placeholder.markdown(f"""
                        <div class="word-display" style="color: #999;">Hold signs to form words...</div>
                    """, unsafe_allow_html=True)
                
                # Update statistics
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                history_list = list(recognizer.prediction_history)
                
                total_predictions.metric("Total Predictions", len(history_list))
                
                if history_list:
                    avg_conf = np.mean([p['confidence'] for p in history_list])
                    avg_confidence.metric("Avg Confidence", f"{avg_conf:.1%}")
                else:
                    avg_confidence.metric("Avg Confidence", "N/A")
                
                fps_display.metric("FPS", f"{fps:.1f}")
                letters_count.metric("Letters in Word", len(recognizer.current_word))
                
                # Small delay
                time.sleep(0.03)
        
        finally:
            cap.release()
            st.info("ℹ️ Camera stopped")
    
    else:
        st.info("👆 Click 'Start Camera' to begin recognition")
        
        # Show word even when camera is off
        current_word = recognizer.get_word()
        if current_word:
            word_placeholder.markdown(f"""
                <div class="word-display">{current_word}</div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()