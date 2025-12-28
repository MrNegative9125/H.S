import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-letter {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
    }
    .prediction-confidence {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .word-display {
        padding: 2rem;
        border-radius: 10px;
        background: #f8f9fa;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 1rem 0;
    }
    .word-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        font-family: monospace;
        letter-spacing: 0.2rem;
    }
    .stButton>button {
        width: 100%;
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
        
        # Statistics
        self.session_start = time.time()
        self.total_predictions = 0
        
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
        
        self.total_predictions += 1
        
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
        
        if (confidence >= self.confidence_threshold and
            letter != self.last_confirmed_letter and
            (current_time - self.last_confirmation_time) >= self.confirmation_delay):
            
            self.current_word.append(letter)
            self.last_confirmed_letter = letter
            self.last_confirmation_time = current_time
            return True
        
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
    
    def draw_roi_box(self, frame):
        """Draw ROI box on frame"""
        height, width = frame.shape[:2]
        roi_size = min(height, width) - 40
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x + roi_size, roi_y + roi_size), 
                     (31, 119, 180), 3)
        cv2.putText(frame, "Place hand here", 
                   (roi_x + 50, roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (31, 119, 180), 2)
        
        return frame, (roi_x, roi_y, roi_size)

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
        
        auto_add = st.checkbox(
            "Auto-add to Word",
            value=True,
            help="Automatically add high-confidence predictions to word"
        )
        
        st.markdown("---")
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. Click **'Take Photo'** below
        2. Position hand clearly
        3. Make ASL sign
        4. Click to capture
        
        **Tips:**
        - Good lighting helps
        - Keep hand centered
        - Clear background
        """)
        
        st.markdown("---")
        st.info("💡 **Word Formation**: Each photo adds a letter to the word!")
    
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
        st.subheader("📸 Camera Input")
        
        # Streamlit Cloud compatible camera input
        camera_photo = st.camera_input("Take a photo of your ASL sign")
        
        if camera_photo is not None:
            # Convert uploaded image to numpy array
            image = Image.open(camera_photo)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = image_np
            
            # Draw ROI box
            annotated_frame, (roi_x, roi_y, roi_size) = recognizer.draw_roi_box(frame_bgr.copy())
            
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display annotated image
            st.image(annotated_frame_rgb, caption="Captured Image", use_container_width=True)
            
            # Extract ROI for prediction
            roi = frame_bgr[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Make prediction
            with st.spinner("Analyzing..."):
                letter, confidence, predictions_probs = recognizer.predict(roi)
            
            # Get smoothed prediction
            smoothed_letter, smoothed_confidence = recognizer.get_smoothed_prediction(smoothing_window)
            
            if smoothed_letter is not None:
                display_letter = smoothed_letter
                display_confidence = smoothed_confidence
            else:
                display_letter = letter
                display_confidence = confidence
            
            # Auto-add to word if enabled
            if auto_add and display_confidence >= confidence_threshold:
                recognizer.update_word_formation(display_letter, display_confidence)
    
    with col2:
        st.subheader("🎯 Current Prediction")
        
        if camera_photo is not None:
            if display_confidence >= confidence_threshold:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-letter">{display_letter}</div>
                    <div class="prediction-confidence">{display_confidence:.1%} confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Manual add button
                if not auto_add:
                    if st.button("➕ Add to Word", use_container_width=True, type="primary"):
                        recognizer.update_word_formation(display_letter, display_confidence)
                        st.rerun()
            else:
                st.markdown(f"""
                <div class="prediction-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="prediction-letter">-</div>
                    <div class="prediction-confidence">Low confidence</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 Take a photo to see prediction")
        
        st.subheader("📈 Top Predictions")
        
        if camera_photo is not None:
            fig = create_confidence_chart(predictions_probs, recognizer.class_names)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Waiting for photo...")
    
    # Word display
    st.markdown("---")
    st.subheader("📝 Formed Word")
    
    current_word = recognizer.get_word()
    if current_word:
        st.markdown(f"""
        <div class="word-display">
            <div class="word-text">{current_word}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="word-display">
            <div style="color: #999; font-size: 1.2rem;">Capture signs to form words...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Word control buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
    with btn_col1:
        if st.button("➕ Add Space", use_container_width=True):
            recognizer.add_space()
            st.rerun()
    
    with btn_col2:
        if st.button("⌫ Delete Letter", use_container_width=True):
            recognizer.delete_last_letter()
            st.rerun()
    
    with btn_col3:
        if st.button("🗑️ Clear Word", use_container_width=True):
            recognizer.clear_word()
            st.rerun()
    
    with btn_col4:
        if current_word:
            st.download_button(
                "💾 Save Word",
                data=current_word,
                file_name=f"asl_word_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Statistics
    st.markdown("---")
    st.subheader("📊 Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Total Predictions", recognizer.total_predictions)
    
    with stats_col2:
        history_list = list(recognizer.prediction_history)
        if history_list:
            avg_conf = np.mean([p['confidence'] for p in history_list])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with stats_col3:
        st.metric("Letters in Word", len(recognizer.current_word))
    
    with stats_col4:
        session_time = int(time.time() - recognizer.session_start)
        st.metric("Session Time", f"{session_time}s")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit | ASL Recognition System</p>
        <p style="font-size: 0.9rem;">📷 Capture photos to build words • 🎯 AI-powered recognition • 📝 Save your words</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
