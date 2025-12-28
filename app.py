import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="ASL Recognition Live",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #1f77b4 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-confidence {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    .word-display {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: 3px solid #1f77b4;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .word-text {
        font-size: 2.8rem;
        font-weight: bold;
        color: #1f77b4;
        font-family: 'Courier New', monospace;
        letter-spacing: 0.3rem;
        min-height: 3rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

class ASLRecognizer:
    """Real-time ASL Recognition System"""
    
    def __init__(self, model_path='asl_model_final.keras', class_names_path='class_names.npy'):
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
        self.correct_predictions = 0
        
        # Load model
        self.load_model(model_path, class_names_path)
    
    def load_model(self, model_path, class_names_path):
        """Load the trained Keras model and class names"""
        try:
            import os
            
            # Check if files exist
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: '{model_path}'")
                st.info("📁 Please upload your model file to the repository root")
                return False
            
            if not os.path.exists(class_names_path):
                st.error(f"❌ Class names file not found: '{class_names_path}'")
                st.info("📁 Please upload your class_names.npy file to the repository root")
                return False
            
            # Load model with compatibility settings
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False  # Don't compile to avoid optimizer issues
            )
            
            # Recompile the model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load class names
            self.class_names = np.load(class_names_path, allow_pickle=True).tolist()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.exception(e)
            st.info("""
            **Troubleshooting Tips:**
            1. Make sure model file is in Keras format (.keras or .h5)
            2. Verify files are in repository root
            3. Check file names match exactly
            4. Model should be saved with: `model.save('asl_model_final.keras')`
            """)
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        # Ensure frame is in correct format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(frame, self.img_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict(self, frame):
        """Make prediction on a frame"""
        if self.model is None:
            return None, 0.0, None
        
        try:
            processed_frame = self.preprocess_frame(frame)
            predictions = self.model.predict(processed_frame, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_letter = self.class_names[predicted_class_idx]
            
            # Update history
            self.prediction_history.append({
                'letter': predicted_letter,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            self.total_predictions += 1
            
            return predicted_letter, confidence, predictions[0]
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, 0.0, None
    
    def get_smoothed_prediction(self, window_size=5):
        """Get smoothed prediction based on recent history"""
        if len(self.prediction_history) < window_size:
            if len(self.prediction_history) > 0:
                last_pred = list(self.prediction_history)[-1]
                return last_pred['letter'], last_pred['confidence']
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
            self.last_confirmed_letter = None
    
    def get_word(self):
        """Get current word as string"""
        return ''.join(self.current_word)
    
    def draw_roi_box(self, frame):
        """Draw ROI box on frame for hand placement guide"""
        height, width = frame.shape[:2]
        
        # Calculate ROI dimensions (square box)
        roi_size = min(height, width) - 100
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # Draw outer box
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x + roi_size, roi_y + roi_size), 
                     (31, 119, 180), 3)
        
        # Draw corner markers
        corner_length = 30
        thickness = 5
        
        # Top-left
        cv2.line(frame, (roi_x, roi_y), (roi_x + corner_length, roi_y), (31, 119, 180), thickness)
        cv2.line(frame, (roi_x, roi_y), (roi_x, roi_y + corner_length), (31, 119, 180), thickness)
        
        # Top-right
        cv2.line(frame, (roi_x + roi_size, roi_y), (roi_x + roi_size - corner_length, roi_y), (31, 119, 180), thickness)
        cv2.line(frame, (roi_x + roi_size, roi_y), (roi_x + roi_size, roi_y + corner_length), (31, 119, 180), thickness)
        
        # Bottom-left
        cv2.line(frame, (roi_x, roi_y + roi_size), (roi_x + corner_length, roi_y + roi_size), (31, 119, 180), thickness)
        cv2.line(frame, (roi_x, roi_y + roi_size), (roi_x, roi_y + roi_size - corner_length), (31, 119, 180), thickness)
        
        # Bottom-right
        cv2.line(frame, (roi_x + roi_size, roi_y + roi_size), (roi_x + roi_size - corner_length, roi_y + roi_size), (31, 119, 180), thickness)
        cv2.line(frame, (roi_x + roi_size, roi_y + roi_size), (roi_x + roi_size, roi_y + roi_size - corner_length), (31, 119, 180), thickness)
        
        # Add instruction text
        cv2.putText(frame, "Place hand inside the box", 
                   (roi_x + 20, roi_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (31, 119, 180), 2)
        
        return frame, (roi_x, roi_y, roi_size)

def create_confidence_chart(predictions_probs, class_names, top_n=5):
    """Create a horizontal bar chart of top predictions"""
    # Get top N predictions
    top_indices = np.argsort(predictions_probs)[-top_n:][::-1]
    top_letters = [class_names[i] for i in top_indices]
    top_probs = [predictions_probs[i] for i in top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(len(top_letters)), top_probs)
    
    # Color bars based on confidence
    colors = []
    for prob in top_probs:
        if prob > 0.7:
            colors.append('#2ecc71')  # Green
        elif prob > 0.5:
            colors.append('#3498db')  # Blue
        else:
            colors.append('#e74c3c')  # Red
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(range(len(top_letters)))
    ax.set_yticklabels(top_letters, fontsize=11, fontweight='bold')
    ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax.set_title('Top Predictions', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.2%}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🤟 ASL Alphabet Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time American Sign Language Detection with Deep Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📁 Model Files")
        
        # Option to upload files
        upload_mode = st.checkbox("Upload model files", value=False, 
                                  help="Enable if model files are not in repository")
        
        if upload_mode:
            model_file = st.file_uploader("Upload Model (.keras)", type=['keras', 'h5'])
            class_names_file = st.file_uploader("Upload Class Names (.npy)", type=['npy'])
            
            # Save uploaded files temporarily
            if model_file is not None:
                with open("uploaded_model.keras", "wb") as f:
                    f.write(model_file.getbuffer())
                model_path = "uploaded_model.keras"
                st.success("✅ Model uploaded")
            else:
                model_path = None
                
            if class_names_file is not None:
                with open("uploaded_class_names.npy", "wb") as f:
                    f.write(class_names_file.getbuffer())
                class_names_path = "uploaded_class_names.npy"
                st.success("✅ Class names uploaded")
            else:
                class_names_path = None
        else:
            model_path = st.text_input("Model Path", "asl_model_final.keras", 
                                       help="Path to your trained Keras model file")
            class_names_path = st.text_input("Class Names Path", "class_names.npy",
                                             help="Path to class names numpy file")
        
        st.markdown("---")
        st.subheader("🎛️ Prediction Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence required to display and add prediction"
        )
        
        smoothing_window = st.slider(
            "Smoothing Window",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of recent predictions to average for stability"
        )
        
        top_n_predictions = st.slider(
            "Show Top N Predictions",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of top predictions to display"
        )
        
        st.markdown("---")
        st.subheader("🎯 Word Formation")
        
        auto_add = st.checkbox(
            "Auto-add Letters",
            value=True,
            help="Automatically add high-confidence predictions to word"
        )
        
        confirmation_delay = st.slider(
            "Letter Delay (seconds)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.5,
            help="Delay between adding same letter twice"
        )
        
        st.markdown("---")
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        **Steps:**
        1. 📸 Click **'Take a photo'** 
        2. ✋ Make an ASL hand sign
        3. 📷 Capture the image
        4. 🎯 View prediction & confidence
        5. 📝 Build words letter by letter
        
        **Tips for Best Results:**
        - ☀️ Use good lighting
        - 🎯 Center hand in box
        - 🖐️ Clear hand shape
        - 🏠 Plain background
        - 📏 Keep consistent distance
        """)
        
        st.markdown("---")
        st.info("💡 **Pro Tip**: Multiple photos with same sign improves accuracy through smoothing!")
    
    # Initialize recognizer in session state
    if 'recognizer' not in st.session_state or st.sidebar.button("🔄 Reload Model"):
        if upload_mode and (model_path is None or class_names_path is None):
            st.warning("⚠️ Please upload both model and class names files")
            st.stop()
        
        if model_path and class_names_path:
            with st.spinner("🔄 Loading AI model..."):
                st.session_state.recognizer = ASLRecognizer(model_path, class_names_path)
                
            if st.session_state.recognizer.model is not None:
                st.success(f"✅ Model loaded successfully! Ready to recognize {len(st.session_state.recognizer.class_names)} ASL signs.")
            else:
                st.error("❌ Failed to load model. Please check the error messages above.")
                st.stop()
        else:
            st.warning("⚠️ Please provide model file paths")
            st.stop()
    
    recognizer = st.session_state.recognizer
    recognizer.confidence_threshold = confidence_threshold
    recognizer.confirmation_delay = confirmation_delay
    
    # Main layout
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📸 Camera Input")
        
        # Camera input (compatible with Streamlit Cloud)
        camera_photo = st.camera_input("Take a photo of your ASL sign", 
                                       help="Click to activate camera and capture your hand sign")
        
        if camera_photo is not None:
            # Convert uploaded image to numpy array
            image = Image.open(camera_photo)
            image_np = np.array(image)
            
            # Convert to BGR for OpenCV processing
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = image_np
            
            # Draw ROI box
            annotated_frame, (roi_x, roi_y, roi_size) = recognizer.draw_roi_box(frame_bgr.copy())
            
            # Convert back to RGB for display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display annotated image
            st.image(annotated_frame_rgb, caption="📷 Captured Image with ROI", use_container_width=True)
            
            # Extract ROI for prediction
            roi = frame_bgr[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Make prediction
            with st.spinner("🔍 Analyzing hand sign..."):
                letter, confidence, predictions_probs = recognizer.predict(roi)
            
            if letter is not None:
                # Get smoothed prediction
                smoothed_letter, smoothed_confidence = recognizer.get_smoothed_prediction(smoothing_window)
                
                if smoothed_letter is not None:
                    display_letter = smoothed_letter
                    display_confidence = smoothed_confidence
                else:
                    display_letter = letter
                    display_confidence = confidence
                
                # Store in session state for display
                st.session_state.current_prediction = {
                    'letter': display_letter,
                    'confidence': display_confidence,
                    'predictions_probs': predictions_probs
                }
                
                # Auto-add to word if enabled
                if auto_add and display_confidence >= confidence_threshold:
                    if recognizer.update_word_formation(display_letter, display_confidence):
                        st.success(f"✅ Added '{display_letter}' to word!")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction
            display_letter = pred['letter']
            display_confidence = pred['confidence']
            predictions_probs = pred['predictions_probs']
            
            # Prediction box
            if display_confidence >= confidence_threshold:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-letter">{display_letter}</div>
                    <div class="prediction-confidence">Confidence: {display_confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Manual add button if auto-add is off
                if not auto_add:
                    if st.button("➕ Add Letter to Word", use_container_width=True, type="primary"):
                        if recognizer.update_word_formation(display_letter, display_confidence):
                            st.success(f"Added '{display_letter}'!")
                            st.rerun()
            else:
                st.markdown(f"""
                <div class="prediction-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="prediction-letter">{display_letter}</div>
                    <div class="prediction-confidence">⚠️ Low Confidence: {display_confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ Confidence below threshold. Try better lighting or clearer hand position.")
            
            # Top predictions chart
            st.markdown("---")
            st.markdown("**📊 Confidence Distribution**")
            fig = create_confidence_chart(predictions_probs, recognizer.class_names, top_n_predictions)
            st.pyplot(fig)
            plt.close(fig)
            
        else:
            st.info("👆 Take a photo above to see prediction results")
            st.markdown("""
            <div class="prediction-box" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                <div style="font-size: 3rem;">📷</div>
                <div class="prediction-confidence">Waiting for input...</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Word formation section
    st.markdown("---")
    st.subheader("📝 Word Formation")
    
    current_word = recognizer.get_word()
    if current_word:
        st.markdown(f"""
        <div class="word-display">
            <div class="word-text">{current_word}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="word-display">
            <div style="color: #999; font-size: 1.5rem; font-style: italic;">
                📸 Capture signs to start forming words...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Word control buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
    with btn_col1:
        if st.button("⎵ Add Space", use_container_width=True, help="Add space between words"):
            recognizer.add_space()
            st.rerun()
    
    with btn_col2:
        if st.button("⌫ Delete Last", use_container_width=True, help="Remove last letter"):
            recognizer.delete_last_letter()
            st.rerun()
    
    with btn_col3:
        if st.button("🗑️ Clear All", use_container_width=True, help="Clear entire word"):
            recognizer.clear_word()
            st.rerun()
    
    with btn_col4:
        if current_word:
            st.download_button(
                "💾 Save Word",
                data=current_word,
                file_name=f"asl_word_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True,
                help="Download word as text file"
            )
    
    # Statistics section
    st.markdown("---")
    st.subheader("📊 Session Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("🔢 Total Predictions", recognizer.total_predictions)
    
    with stats_col2:
        history_list = list(recognizer.prediction_history)
        if history_list:
            avg_conf = np.mean([p['confidence'] for p in history_list])
            st.metric("📈 Avg Confidence", f"{avg_conf:.1%}")
        else:
            st.metric("📈 Avg Confidence", "N/A")
    
    with stats_col3:
        st.metric("📝 Letters in Word", len(recognizer.current_word))
    
    with stats_col4:
        session_time = int(time.time() - recognizer.session_start)
        minutes, seconds = divmod(session_time, 60)
        st.metric("⏱️ Session Time", f"{minutes}m {seconds}s")
    
    # Recent predictions history
    if len(recognizer.prediction_history) > 0:
        with st.expander("📜 View Recent Predictions History", expanded=False):
            history_df = pd.DataFrame(list(recognizer.prediction_history)[-10:])
            history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
            st.dataframe(history_df[['letter', 'confidence', 'timestamp']], use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
            <strong>ASL Recognition System</strong> • Built with TensorFlow & Streamlit
        </p>
        <p style="font-size: 0.9rem; margin: 0;">
            📷 Capture • 🤖 AI Recognition • 📝 Word Formation • 💾 Export
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
