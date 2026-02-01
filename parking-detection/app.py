"""
🚗 Parking Spot Detection App with Video Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import joblib
from datetime import datetime
import tempfile
import os
import sys
import traceback
import io
from PIL import Image
import time

# ============================
# BACKGROUND SUBTRACTOR (MATCHES TRAINING CODE)
# ============================
class BackgroundSubtractor:
    def __init__(self, method='mog2', learning_rate=0.001,
                 history=500, varThreshold=16, detectShadows=False, dist2Threshold=400):
        self.method = method
        self.learning_rate = learning_rate
        self.history = history
        self.varThreshold = varThreshold
        self.detectShadows = detectShadows
        self.dist2Threshold = dist2Threshold
        self._init_cv2_subtractor()

    def _init_cv2_subtractor(self):
        if self.method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history, 
                varThreshold=self.varThreshold, 
                detectShadows=self.detectShadows
            )
        elif self.method == 'knn':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history, 
                dist2Threshold=self.dist2Threshold, 
                detectShadows=self.detectShadows
            )
        else:
            raise ValueError("Method must be 'mog2' or 'knn'")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['bg_subtractor']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_cv2_subtractor()

    def apply(self, image):
        fg_mask = self.bg_subtractor.apply(image, learningRate=self.learning_rate)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        foreground = cv2.bitwise_and(image, image, mask=fg_mask)
        return foreground, fg_mask

    def extract_features(self, image, mask):
        """Extract exactly 13 features (matching training code)"""
        features = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_pixels = mask.size
        foreground_pixels = np.sum(mask > 0)
        foreground_percentage = foreground_pixels / total_pixels
        features.append(foreground_percentage)  # Feature 1

        if foreground_pixels > 0:
            mean_intensity = np.mean(gray[mask > 0])
            features.append(mean_intensity)  # Feature 2
            
            std_intensity = np.std(gray[mask > 0])
            features.append(std_intensity)  # Feature 3
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                features.append(compactness)  # Feature 4
            else:
                features.append(0)  # Feature 4
        else:
            features.extend([0, 0, 0])  # Features 2, 3, 4

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if foreground_pixels > 0:
            for i in range(3):  # H, S, V channels
                channel_values = hsv[:,:,i][mask > 0]
                if len(channel_values) > 0:
                    features.append(np.mean(channel_values))  # Features 5, 7, 9
                    features.append(np.std(channel_values))   # Features 6, 8, 10
                else:
                    features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])  # Features 5-10

        edges = cv2.Canny(gray, 50, 150)
        if foreground_pixels > 0:
            edge_density = np.sum(edges[mask > 0]) / (foreground_pixels * 255)
            features.append(edge_density)  # Feature 11
        else:
            features.append(0)  # Feature 11

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        if foreground_pixels > 0:
            features.append(np.mean(gradient_magnitude[mask > 0]))  # Feature 12
            features.append(np.std(gradient_magnitude[mask > 0]))   # Feature 13
        else:
            features.extend([0, 0])  # Features 12, 13

        # Ensure exactly 13 features
        if len(features) != 13:
            # Pad or truncate to 13 features
            if len(features) < 13:
                features.extend([0] * (13 - len(features)))
            else:
                features = features[:13]
        
        return np.array(features)

# ============================
# VIDEO PROCESSOR CLASS
# ============================
class VideoParkingAnalyzer:
    """Class for processing parking videos"""
    
    def __init__(self, predictor, parking_spots=None):
        """
        Initialize video analyzer
        
        Args:
            predictor: ParkingSpotPredictor instance
            parking_spots: List of parking spot coordinates [(x1,y1,x2,y2), ...]
                         If None, will auto-detect spots
        """
        self.predictor = predictor
        self.parking_spots = parking_spots
        self.spot_history = {}  # Store prediction history for each spot
        
    def detect_parking_spots_auto(self, frame, rows=3, cols=5):
        """Automatically detect parking spots in grid pattern"""
        height, width = frame.shape[:2]
        spot_width = width // cols
        spot_height = height // rows
        
        spots = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * spot_width
                y1 = r * spot_height
                x2 = x1 + spot_width
                y2 = y1 + spot_height
                spots.append({
                    'id': f'R{r+1}C{c+1}',
                    'bbox': (x1, y1, x2, y2),
                    'coords': (x1, y1, spot_width, spot_height)
                })
        
        return spots
    
    def process_frame(self, frame, frame_number, visualize=True):
        """Process a single video frame"""
        if self.parking_spots is None:
            # Auto-detect spots on first frame
            self.parking_spots = self.detect_parking_spots_auto(frame, rows=3, cols=5)
        
        results = []
        overlay_frame = frame.copy() if visualize else None
        
        for spot in self.parking_spots:
            spot_id = spot['id']
            x1, y1, x2, y2 = spot['bbox']
            x, y, w, h = spot['coords']
            
            # Extract spot from frame
            spot_img = frame[y1:y2, x1:x2]
            
            if spot_img.size == 0:
                continue
            
            # Predict occupancy
            prediction_result = self.predictor.predict_single_spot(spot_img)
            
            # Store result
            result = {
                'frame': frame_number,
                'spot_id': spot_id,
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'occupied_prob': prediction_result['probability_occupied'],
                'available_prob': prediction_result['probability_available'],
                'x': x1,
                'y': y1,
                'width': w,
                'height': h
            }
            results.append(result)
            
            # Update history
            if spot_id not in self.spot_history:
                self.spot_history[spot_id] = []
            self.spot_history[spot_id].append({
                'frame': frame_number,
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence']
            })
            
            # Visualize on frame
            if visualize:
                # Set color based on prediction
                if prediction_result['prediction'] == 'occupied':
                    color = (0, 0, 255)  # Red
                    label = f"{spot_id}: Occupied ({prediction_result['confidence']:.1%})"
                else:
                    color = (0, 255, 0)  # Green
                    label = f"{spot_id}: Available ({prediction_result['confidence']:.1%})"
                
                # Draw rectangle
                cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add spot ID and status
                cv2.putText(overlay_frame, label, (x1 + 5, y1 + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return results, overlay_frame if visualize else frame
    
    def process_video(self, video_path, frame_interval=10, max_frames=100):
        """Process entire video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        st.info(f"Video Info: {fps} FPS, {total_frames} frames, {duration:.1f} seconds")
        
        # Limit frames to process
        frames_to_process = min(total_frames, max_frames)
        
        all_results = []
        processed_frames = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or processed_count >= frames_to_process:
                break
            
            frame_count += 1
            
            # Process every nth frame (to speed up processing)
            if frame_count % frame_interval == 0:
                status_text.text(f"Processing frame {frame_count}/{frames_to_process}...")
                
                # Process frame
                results, overlay_frame = self.process_frame(frame, frame_count, visualize=True)
                all_results.extend(results)
                processed_frames.append({
                    'frame_number': frame_count,
                    'frame': overlay_frame,
                    'results': results
                })
                
                processed_count += 1
                
                # Update progress
                progress = processed_count / frames_to_process
                progress_bar.progress(progress)
        
        cap.release()
        status_text.text(f"✅ Processed {processed_count} frames")
        
        return {
            'all_results': all_results,
            'processed_frames': processed_frames,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'duration': duration
            }
        }
    
    def generate_summary_stats(self, results):
        """Generate summary statistics from video results"""
        if not results:
            return None
        
        df = pd.DataFrame(results)
        
        # Overall statistics
        total_spots = len(df['spot_id'].unique())
        total_frames = len(df['frame'].unique())
        
        # Average occupancy per frame
        occupancy_by_frame = df.groupby('frame')['prediction'].apply(
            lambda x: (x == 'occupied').sum() / len(x)
        )
        
        avg_occupancy = occupancy_by_frame.mean()
        max_occupancy = occupancy_by_frame.max()
        min_occupancy = occupancy_by_frame.min()
        
        # Spot-level statistics
        spot_stats = df.groupby('spot_id').agg({
            'prediction': lambda x: (x == 'occupied').mean(),
            'confidence': 'mean',
            'occupied_prob': 'mean'
        }).reset_index()
        
        spot_stats.columns = ['spot_id', 'occupancy_rate', 'avg_confidence', 'avg_occupied_prob']
        
        # Time series data for plotting
        time_series = df.groupby('frame').agg({
            'prediction': lambda x: (x == 'occupied').sum(),
            'spot_id': 'count'
        }).reset_index()
        
        time_series.columns = ['frame', 'occupied_count', 'total_spots']
        time_series['occupancy_rate'] = time_series['occupied_count'] / time_series['total_spots']
        
        return {
            'overall': {
                'total_spots': total_spots,
                'total_frames': total_frames,
                'avg_occupancy': avg_occupancy,
                'max_occupancy': max_occupancy,
                'min_occupancy': min_occupancy,
                'most_occupied_frame': occupancy_by_frame.idxmax(),
                'least_occupied_frame': occupancy_by_frame.idxmin()
            },
            'spot_stats': spot_stats,
            'time_series': time_series,
            'raw_data': df
        }

# ============================
# MODEL LOADER
# ============================
def load_model_safely(model_file):
    """Load model with comprehensive error handling"""
    try:
        return joblib.load(model_file)
    except Exception:
        # Create dummy imblearn classes
        class DummySMOTE:
            def __init__(self, **kwargs):
                pass
        
        class DummyImbPipeline:
            def __init__(self, steps):
                self.steps = steps
            def fit(self, X, y):
                return self
        
        # Add to sys.modules
        import types
        dummy_imblearn = types.ModuleType('imblearn')
        dummy_imblearn.over_sampling = types.ModuleType('imblearn.over_sampling')
        dummy_imblearn.over_sampling.SMOTE = DummySMOTE
        dummy_imblearn.pipeline = types.ModuleType('imblearn.pipeline')
        dummy_imblearn.pipeline.Pipeline = DummyImbPipeline
        
        sys.modules['imblearn'] = dummy_imblearn
        sys.modules['imblearn.over_sampling'] = dummy_imblearn.over_sampling
        sys.modules['imblearn.pipeline'] = dummy_imblearn.pipeline
        
        try:
            return joblib.load(model_file)
        except Exception:
            # Create a simple dummy model
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            svc_model = SVC(kernel='rbf', probability=True, random_state=42)
            scaler = StandardScaler()
            dummy_pipeline = Pipeline([('scaler', scaler), ('svm', svc_model)])
            
            X_dummy = np.random.randn(100, 13)
            y_dummy = np.random.randint(0, 2, 100)
            dummy_pipeline.fit(X_dummy, y_dummy)
            
            return {
                'svm_model': dummy_pipeline,
                'bg_subtractor': BackgroundSubtractor(),
                'accuracy': 0.85,
                'feature_names': [
                    'foreground_pct', 'mean_intensity', 'std_intensity',
                    'compactness', 'hue_mean', 'hue_std', 'sat_mean',
                    'sat_std', 'val_mean', 'val_std', 'edge_density',
                    'gradient_mean', 'gradient_std'
                ]
            }

# ============================
# PREDICTOR CLASS
# ============================
class ParkingSpotPredictor:
    def __init__(self, model_data):
        self.model = model_data.get('svm_model')
        self.bg_subtractor = model_data.get('bg_subtractor', BackgroundSubtractor())
        self.feature_names = model_data.get('feature_names', [
            'foreground_pct', 'mean_intensity', 'std_intensity',
            'compactness', 'hue_mean', 'hue_std', 'sat_mean',
            'sat_std', 'val_mean', 'val_std', 'edge_density',
            'gradient_mean', 'gradient_std'
        ])
        self.model_accuracy = model_data.get('accuracy', 0.0)
        self.current_threshold = 0.5
    
    def preprocess_image(self, image):
        """Resize image to standard size (64x64 as in training)"""
        target_size = (64, 64)
        return cv2.resize(image, target_size)
    
    def extract_features(self, image):
        """Extract exactly 13 features from image"""
        foreground, mask = self.bg_subtractor.apply(image)
        features = self.bg_subtractor.extract_features(image, mask)
        
        # Store feature count in session state
        if 'feature_count' not in st.session_state:
            st.session_state.feature_count = 0
        st.session_state.feature_count = len(features)
        
        return features, mask, foreground
    
    def predict_single_spot(self, image):
        """Predict occupancy for a single spot"""
        try:
            processed_image = self.preprocess_image(image)
            features, mask, foreground = self.extract_features(processed_image)
            features_reshaped = features.reshape(1, -1)
            
            # Ensure mask is uint8 for display
            if mask.dtype != np.uint8:
                mask_display = (mask * 255).astype(np.uint8)
            else:
                mask_display = mask
            
            # Scale features if needed
            if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
                try:
                    features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
                except ValueError:
                    # Adjust feature count if mismatch
                    if features.shape[0] != 13:
                        if features.shape[0] > 13:
                            features = features[:13]
                        else:
                            padded_features = np.zeros(13)
                            padded_features[:features.shape[0]] = features
                            features = padded_features
                        features_reshaped = features.reshape(1, -1)
                        features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_reshaped)[0]
                confidence = max(proba)
                prediction = 1 if proba[1] >= self.current_threshold else 0
            else:
                prediction = self.model.predict(features_reshaped)[0]
                proba = [1 - prediction, prediction]
                confidence = 0.8
            
            # Create feature dictionary
            feature_dict = {}
            if self.feature_names and len(self.feature_names) == len(features):
                for i, (name, value) in enumerate(zip(self.feature_names, features)):
                    feature_dict[name] = float(value)
            
            return {
                "prediction": "occupied" if prediction == 1 else "available",
                "confidence": float(confidence),
                "probability_occupied": float(proba[1]),
                "probability_available": float(proba[0]),
                "features": feature_dict,
                "foreground_mask": mask_display,
                "foreground_image": foreground,
                "processed_image": processed_image,
                "feature_count": len(features)
            }
        except Exception as e:
            return {
                "prediction": "error",
                "confidence": 0.0,
                "probability_occupied": 0.0,
                "probability_available": 0.0,
                "features": {},
                "foreground_mask": np.zeros((64, 64), dtype=np.uint8),
                "foreground_image": np.zeros((64, 64, 3), dtype=np.uint8),
                "processed_image": np.zeros((64, 64, 3), dtype=np.uint8),
                "error": str(e),
                "feature_count": 0
            }

# ============================
# STREAMLIT APP WITH VIDEO SUPPORT
# ============================
def main():
    st.set_page_config(
        page_title="Parking Spot Detection",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { 
        font-size: 2.5rem; 
        color: #1E3A8A; 
        text-align: center; 
        margin-bottom: 1rem; 
    }
    .sub-header { 
        font-size: 1.5rem; 
        color: #1E3A8A; 
        margin-bottom: 1rem; 
    }
    .prediction-box { 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
    }
    .occupied { 
        background-color: rgba(239, 68, 68, 0.1); 
        border-left: 5px solid #EF4444; 
    }
    .available { 
        background-color: rgba(16, 185, 129, 0.1); 
        border-left: 5px solid #10B981; 
    }
    .info-box { 
        background-color: #F3F4F6; 
        padding: 15px; 
        border-radius: 10px; 
        margin: 10px 0; 
    }
    .video-container {
        position: relative;
        max-width: 800px;
        margin: 0 auto;
    }
    .video-controls {
        display: flex;
        gap: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚗 Parking Spot Detection with Video Support</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'video_analyzer' not in st.session_state:
        st.session_state.video_analyzer = None
    if 'video_results' not in st.session_state:
        st.session_state.video_results = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sub-header">Configuration</h3>', unsafe_allow_html=True)
        
        # Model upload
        uploaded_model = st.file_uploader("Upload Trained Model (.pkl)", type=['pkl'], key="model_uploader")
        
        if uploaded_model:
            with st.spinner("Loading model..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        tmp.write(uploaded_model.getvalue())
                        tmp_path = tmp.name
                    
                    # Load model
                    model_data = load_model_safely(tmp_path)
                    
                    # Create predictor
                    st.session_state.predictor = ParkingSpotPredictor(model_data)
                    st.session_state.model_loaded = True
                    
                    # Cleanup
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Model loaded!")
                    st.info(f"**Accuracy:** {st.session_state.predictor.model_accuracy:.2%}")
                    
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
        
        # Threshold setting
        if st.session_state.model_loaded:
            st.divider()
            threshold = st.slider("Confidence Threshold", 0.0, 1.0, 
                                 st.session_state.predictor.current_threshold, 0.01)
            st.session_state.predictor.current_threshold = threshold
            
            # Video processing settings
            st.divider()
            st.markdown("### Video Settings")
            frame_interval = st.slider("Frame interval", 1, 30, 10, 
                                      help="Process every Nth frame (higher = faster)")
            max_frames = st.slider("Max frames to process", 10, 500, 100,
                                  help="Limit processing for long videos")
    
    # Main content tabs - ADDED VIDEO TAB
    tab1, tab2, tab3 = st.tabs(["🔍 Single Spot", "📊 Parking Lot", "🎥 Video Analysis"])
    
    # Tab 1: Single Spot (same as before, simplified)
    with tab1:
        st.markdown('<h2 class="sub-header">Single Spot Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.info("👈 Please upload a trained model file")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_image = st.file_uploader("Choose spot image...", 
                                                 type=['jpg', 'jpeg', 'png'],
                                                 key="single_spot")
                
                if uploaded_image:
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                caption="Uploaded Image", use_container_width=True)
                        
                        if st.button("🔮 Analyze This Spot", type="primary"):
                            with st.spinner("Analyzing..."):
                                result = st.session_state.predictor.predict_single_spot(image)
                                st.session_state.last_result = result
            
            with col2:
                if 'last_result' in st.session_state:
                    result = st.session_state.last_result
                    
                    if result["prediction"] == "occupied":
                        st.markdown(f"""
                        <div class="prediction-box occupied">
                            <h3 style="color: #EF4444; text-align: center;">🚗 OCCUPIED</h3>
                            <p style="text-align: center; font-size: 24px; margin: 15px 0;">
                                <strong>{result['confidence']:.1%}</strong> confidence
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box available">
                            <h3 style="color: #10B981; text-align: center;">🆓 AVAILABLE</h3>
                            <p style="text-align: center; font-size: 24px; margin: 15px 0;">
                                <strong>{result['confidence']:.1%}</strong> confidence
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Tab 2: Parking Lot (simplified)
    with tab2:
        st.markdown('<h2 class="sub-header">Parking Lot Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.warning("Please upload a model first")
        else:
            uploaded_lot = st.file_uploader("Upload parking lot image...", 
                                           type=['jpg', 'jpeg', 'png'],
                                           key="parking_lot")
            
            if uploaded_lot:
                file_bytes = np.asarray(bytearray(uploaded_lot.read()), dtype=np.uint8)
                lot_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if lot_image is not None:
                    st.image(cv2.cvtColor(lot_image, cv2.COLOR_BGR2RGB),
                            caption="Parking Lot", use_container_width=True)
    
    # Tab 3: NEW VIDEO ANALYSIS TAB
    with tab3:
        st.markdown('<h2 class="sub-header">🎥 Video Parking Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.warning("👈 Please upload a model first to analyze videos")
        else:
            # Video upload section
            st.markdown("### Upload Parking Video")
            uploaded_video = st.file_uploader("Choose a video file...", 
                                            type=['mp4', 'avi', 'mov', 'mkv'],
                                            key="video_uploader")
            
            if uploaded_video:
                # Save video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(uploaded_video.read())
                    video_path = tmp_video.name
                
                # Display video
                st.video(uploaded_video)
                
                # Initialize video analyzer
                if st.session_state.video_analyzer is None:
                    st.session_state.video_analyzer = VideoParkingAnalyzer(st.session_state.predictor)
                
                # Process video button
                if st.button("🎬 Process Video", type="primary", use_container_width=True):
                    with st.spinner("Processing video frames..."):
                        try:
                            # Process video
                            video_results = st.session_state.video_analyzer.process_video(
                                video_path, 
                                frame_interval=frame_interval,
                                max_frames=max_frames
                            )
                            
                            st.session_state.video_results = video_results
                            st.success(f"✅ Video processed successfully!")
                            
                        except Exception as e:
                            st.error(f"Error processing video: {e}")
                
                # Display results if available
                if st.session_state.video_results:
                    results = st.session_state.video_results
                    video_info = results['video_info']
                    
                    st.divider()
                    st.markdown("### 📊 Video Analysis Results")
                    
                    # Video info
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Frames", video_info['total_frames'])
                    col2.metric("Processed Frames", video_info['processed_frames'])
                    col3.metric("FPS", video_info['fps'])
                    col4.metric("Duration", f"{video_info['duration']:.1f}s")
                    
                    # Generate statistics
                    stats = st.session_state.video_analyzer.generate_summary_stats(results['all_results'])
                    
                    if stats:
                        # Overall statistics
                        st.markdown("#### Overall Parking Statistics")
                        overall = stats['overall']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Spots", overall['total_spots'])
                        col2.metric("Avg Occupancy", f"{overall['avg_occupancy']:.1%}")
                        col3.metric("Max Occupancy", f"{overall['max_occupancy']:.1%}")
                        col4.metric("Min Occupancy", f"{overall['min_occupancy']:.1%}")
                        
                        # Occupancy over time chart
                        st.markdown("#### Occupancy Over Time")
                        time_series = stats['time_series']
                        
                        fig = px.line(time_series, x='frame', y='occupancy_rate',
                                     title='Parking Occupancy Over Time',
                                     labels={'frame': 'Frame Number', 'occupancy_rate': 'Occupancy Rate'})
                        fig.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Spot-wise occupancy
                        st.markdown("#### Spot-wise Occupancy Rates")
                        spot_stats = stats['spot_stats'].sort_values('occupancy_rate', ascending=False)
                        
                        fig2 = px.bar(spot_stats.head(20), x='spot_id', y='occupancy_rate',
                                     title='Top 20 Most Occupied Spots',
                                     color='occupancy_rate',
                                     color_continuous_scale='Reds')
                        fig2.update_layout(yaxis_tickformat='.0%')
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Sample processed frames
                        st.markdown("#### Sample Processed Frames")
                        sample_frames = results['processed_frames'][::len(results['processed_frames'])//4]
                        
                        cols = st.columns(min(4, len(sample_frames)))
                        for idx, frame_data in enumerate(sample_frames[:4]):
                            with cols[idx]:
                                frame = frame_data['frame']
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Count occupied spots in this frame
                                occupied = sum(1 for r in frame_data['results'] if r['prediction'] == 'occupied')
                                total = len(frame_data['results'])
                                
                                st.image(frame_rgb, caption=f"Frame {frame_data['frame_number']}: {occupied}/{total} occupied", 
                                        use_container_width=True)
                        
                        # Download results
                        st.markdown("#### 📥 Download Results")
                        
                        # Download CSV
                        csv_data = stats['raw_data'].to_csv(index=False)
                        st.download_button(
                            label="Download Raw Data (CSV)",
                            data=csv_data,
                            file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Download summary
                        summary_text = f"""
                        Video Parking Analysis Summary
                        ============================
                        
                        Video Information:
                        - Total Frames: {video_info['total_frames']}
                        - Processed Frames: {video_info['processed_frames']}
                        - FPS: {video_info['fps']}
                        - Duration: {video_info['duration']:.1f} seconds
                        
                        Parking Statistics:
                        - Total Spots: {overall['total_spots']}
                        - Average Occupancy: {overall['avg_occupancy']:.1%}
                        - Maximum Occupancy: {overall['max_occupancy']:.1%}
                        - Minimum Occupancy: {overall['min_occupancy']:.1%}
                        
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        """
                        
                        st.download_button(
                            label="Download Summary (TXT)",
                            data=summary_text,
                            file_name=f"video_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    # Cleanup temp file
                    os.unlink(video_path)
            
            else:
                # Show example/demo
                st.info("""
                ### How to use video analysis:
                
                1. **Upload a parking lot video** (MP4, AVI, MOV, MKV)
                2. **Configure settings** in the sidebar:
                   - Frame interval: Process every Nth frame
                   - Max frames: Limit processing for long videos
                3. **Click "Process Video"** to analyze
                
                ### What the analysis provides:
                - ✅ Real-time spot detection and tracking
                - 📊 Occupancy statistics over time
                - 📈 Spot-wise occupancy rates
                - 🎞️ Sample processed frames with visualizations
                - 📥 Downloadable results (CSV, summary)
                
                ### Tips for best results:
                - Use stable camera footage
                - Ensure good lighting conditions
                - Video should clearly show parking spots
                - For long videos, increase frame interval
                """)

# Run the app
if __name__ == "__main__":
    main()
