"""
🚗 Parking Spot Detection App
Deployed on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cv2
import joblib
import json
import time
from datetime import datetime, timedelta
import requests
from PIL import Image
import io
import base64
from collections import defaultdict
import tempfile
import os

# ============================
# CUSTOM BACKGROUND SUBTRACTOR CLASS
# (Must match the training code)
# ============================
class BackgroundSubtractor:
    """
    Custom background subtraction for parking spot analysis
    Identical to the one used in training
    """
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
        """Initializes the cv2 background subtractor object"""
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
        """Prepare for pickling"""
        state = self.__dict__.copy()
        del state['bg_subtractor']
        return state

    def __setstate__(self, state):
        """Restore from pickling"""
        self.__dict__.update(state)
        self._init_cv2_subtractor()

    def apply(self, image):
        """Apply background subtraction to image"""
        fg_mask = self.bg_subtractor.apply(image, learningRate=self.learning_rate)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        foreground = cv2.bitwise_and(image, image, mask=fg_mask)
        return foreground, fg_mask

    def extract_features(self, image, mask):
        """Extract features from foreground"""
        features = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_pixels = mask.size
        foreground_pixels = np.sum(mask > 0)
        foreground_percentage = foreground_pixels / total_pixels
        features.append(foreground_percentage)

        if foreground_pixels > 0:
            mean_intensity = np.mean(gray[mask > 0])
            features.append(mean_intensity)
            
            std_intensity = np.std(gray[mask > 0])
            features.append(std_intensity)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                features.append(compactness)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0, 0])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if foreground_pixels > 0:
            for i in range(3):
                channel_values = hsv[:,:,i][mask > 0]
                if len(channel_values) > 0:
                    features.append(np.mean(channel_values))
                    features.append(np.std(channel_values))
                else:
                    features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])

        edges = cv2.Canny(gray, 50, 150)
        if foreground_pixels > 0:
            edge_density = np.sum(edges[mask > 0]) / (foreground_pixels * 255)
            features.append(edge_density)
        else:
            features.append(0)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        if foreground_pixels > 0:
            features.append(np.mean(gradient_magnitude[mask > 0]))
            features.append(np.std(gradient_magnitude[mask > 0]))
        else:
            features.extend([0, 0])

        return np.array(features)

# ============================
# MODEL PREDICTION CLASS
# ============================
class ParkingSpotPredictor:
    """Class for making predictions using the trained model"""
    
    def __init__(self, model_data):
        """Initialize predictor with trained model"""
        # Directly use the saved model - it should be a trained sklearn model
        self.model = model_data['svm_model']
        self.bg_subtractor = model_data['bg_subtractor']
        self.feature_names = model_data.get('feature_names', [])
        self.model_accuracy = model_data.get('accuracy', 0.0)
        
        self.lighting_profiles = {
            "daylight": {"threshold": 0.5, "brightness": 0, "contrast": 0},
            "dusk": {"threshold": 0.45, "brightness": -20, "contrast": 10},
            "night": {"threshold": 0.4, "brightness": -40, "contrast": 20},
            "overcast": {"threshold": 0.48, "brightness": -15, "contrast": 15},
            "bright_sun": {"threshold": 0.52, "brightness": 20, "contrast": -5}
        }
        
        self.current_threshold = 0.5
        self.current_lighting_mode = "daylight"
        
        st.success(f"✅ Model loaded successfully (Accuracy: {self.model_accuracy:.2%})")
    
    def set_lighting_mode(self, mode):
        """Set lighting mode and adjust threshold"""
        if mode in self.lighting_profiles:
            self.current_lighting_mode = mode
            self.current_threshold = self.lighting_profiles[mode]["threshold"]
            return True
        return False
    
    def adjust_threshold(self, threshold):
        """Manually adjust classification threshold"""
        if 0 <= threshold <= 1:
            self.current_threshold = threshold
            return True
        return False
    
    def preprocess_image(self, image):
        """Preprocess parking spot image"""
        target_size = (64, 64)
        resized = cv2.resize(image, target_size)
        
        profile = self.lighting_profiles[self.current_lighting_mode]
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v = cv2.add(v, profile["brightness"])
        v = np.clip(v, 0, 255)
        
        if profile["contrast"] != 0:
            alpha = 1 + profile["contrast"] / 100
            v = cv2.multiply(v, alpha)
            v = np.clip(v, 0, 255)
        
        hsv_adjusted = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    def extract_features(self, image):
        """Extract features using background subtraction"""
        foreground, mask = self.bg_subtractor.apply(image)
        features = self.bg_subtractor.extract_features(image, mask)
        return features, mask, foreground
    
    def predict_single_spot(self, image):
        """Predict occupancy for a single parking spot"""
        processed_image = self.preprocess_image(image)
        features, mask, foreground = self.extract_features(processed_image)
        features_reshaped = features.reshape(1, -1)
        
        # Scale features if model is a pipeline with scaler
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
            features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_reshaped)[0]
            confidence = max(proba)
            prediction = 1 if proba[1] >= self.current_threshold else 0
        else:
            prediction = self.model.predict(features_reshaped)[0]
            proba = [1 - prediction, prediction]
            confidence = 0.8
        
        feature_values = {}
        if self.feature_names and len(self.feature_names) == len(features):
            for i, (name, value) in enumerate(zip(self.feature_names, features)):
                feature_values[name] = float(value)
        
        return {
            "prediction": "occupied" if prediction == 1 else "available",
            "confidence": float(confidence),
            "probability_occupied": float(proba[1]),
            "probability_available": float(proba[0]),
            "threshold_used": float(self.current_threshold),
            "lighting_mode": self.current_lighting_mode,
            "features": feature_values,
            "processed_image": processed_image,
            "foreground_mask": mask,
            "foreground_image": foreground,
            "feature_vector": features.tolist()
        }

# ============================
# SIMPLIFIED MODEL LOADER
# ============================
def load_model_safely(model_bytes):
    """Load model with error handling for missing dependencies"""
    try:
        # Try direct loading first
        return joblib.load(model_bytes)
    except (AttributeError, ModuleNotFoundError) as e:
        error_msg = str(e)
        
        # Check if it's an imblearn-related error
        if 'imblearn' in error_msg:
            # Try to extract just the sklearn components
            st.warning("⚠️ Model contains imblearn dependencies. Attempting to load sklearn components only...")
            
            # Create a custom unpickler that ignores missing classes
            import pickle
            
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Replace imblearn references with dummy classes
                    if 'imblearn' in module:
                        # Return a dummy class that won't break loading
                        return type('DummyClass', (), {})
                    return super().find_class(module, name)
            
            # Load with custom unpickler
            import io
            model_bytes.seek(0)
            unpickler = CustomUnpickler(model_bytes)
            model_data = unpickler.load()
            
            # Check if we got valid data
            if 'svm_model' in model_data and 'bg_subtractor' in model_data:
                return model_data
            else:
                raise ValueError("Failed to extract valid model components")
        else:
            # Re-raise other errors
            raise e

# ============================
# STREAMLIT APP
# ============================
def main():
    """Main Streamlit application"""
    
    # Page configuration
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
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
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
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">🚗 AI-Powered Parking Spot Detection</h1>', unsafe_allow_html=True)
    st.markdown("**Upload your trained model and parking spot images for real-time occupancy prediction**")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/parking--v1.png", width=80)
        st.title("Model Configuration")
        
        # Model upload
        st.subheader("📁 Upload Trained Model")
        uploaded_model = st.file_uploader(
            "Choose your parking_detection_model.pkl",
            type=['pkl'],
            help="Upload the model file saved from your training"
        )
        
        if uploaded_model:
            with st.spinner("Loading model..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        tmp.write(uploaded_model.getvalue())
                        tmp_path = tmp.name
                    
                    # Load model with safe loading
                    model_bytes = open(tmp_path, 'rb')
                    try:
                        model_data = joblib.load(model_bytes)
                    except (AttributeError, ModuleNotFoundError) as e:
                        if 'imblearn' in str(e):
                            st.warning("Model contains training dependencies. Attempting to extract inference components...")
                            # Reopen file and try alternative loading
                            model_bytes.seek(0)
                            model_data = load_model_safely(model_bytes)
                        else:
                            raise e
                    
                    # Initialize predictor
                    st.session_state.predictor = ParkingSpotPredictor(model_data)
                    st.session_state.model_loaded = True
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    st.success("✅ Model loaded successfully!")
                    
                    # Show model info
                    with st.expander("Model Information"):
                        st.write(f"**Accuracy:** {st.session_state.predictor.model_accuracy:.2%}")
                        st.write(f"**Features:** {len(st.session_state.predictor.feature_names)}")
                        if st.session_state.predictor.feature_names:
                            st.write("**Top 5 Features:**")
                            for name in st.session_state.predictor.feature_names[:5]:
                                st.write(f"- {name}")
                
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
                    st.info("""
                    **Solution:**
                    1. Update your training code to save only sklearn components
                    2. Or install imblearn in Streamlit: Add `imbalanced-learn` to requirements.txt
                    """)
                    st.session_state.model_loaded = False
        
        st.divider()
        
        # Settings (only if model loaded)
        if st.session_state.model_loaded:
            st.subheader("⚙️ Prediction Settings")
            
            # Lighting mode
            lighting_mode = st.selectbox(
                "Lighting Condition",
                ["daylight", "dusk", "night", "overcast", "bright_sun"],
                index=0,
                help="Select the lighting condition for better accuracy"
            )
            
            # Threshold adjustment
            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.predictor.current_threshold,
                step=0.01,
                help="Higher values = more conservative (fewer 'occupied' predictions)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply Settings", use_container_width=True):
                    st.session_state.predictor.set_lighting_mode(lighting_mode)
                    st.session_state.predictor.adjust_threshold(threshold)
                    st.success("Settings applied!")
            
            with col2:
                if st.button("Reset to Default", use_container_width=True):
                    st.session_state.predictor.set_lighting_mode("daylight")
                    st.session_state.predictor.adjust_threshold(0.5)
                    st.rerun()
        
        st.divider()
        
        # Quick Demo
        st.subheader("🖼️ Quick Demo")
        
        demo_option = st.radio(
            "Try a demo image:",
            ["Upload your own", "Sample Empty Spot", "Sample Occupied Spot"],
            index=0
        )
        
        if demo_option == "Sample Empty Spot":
            # Create sample empty spot
            demo_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
            st.session_state.demo_image = demo_image
            st.session_state.demo_label = "empty"
            st.info("Demo empty spot loaded. Go to 'Single Spot Prediction' tab.")
        
        elif demo_option == "Sample Occupied Spot":
            # Create sample occupied spot
            demo_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
            cv2.rectangle(demo_image, (20, 20), (80, 80), (0, 0, 200), -1)
            st.session_state.demo_image = demo_image
            st.session_state.demo_label = "occupied"
            st.info("Demo occupied spot loaded. Go to 'Single Spot Prediction' tab.")
        
        st.divider()
        
        # Deployment info
        st.subheader("🌐 Deployment Info")
        st.info("""
        **Deployed on:** Streamlit Cloud
        **Status:** Online
        **Model:** Parking Detection SVM
        **Last Updated:** {}
        """.format(datetime.now().strftime("%Y-%m-%d")))
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📸 Single Spot Prediction",
        "🏢 Multiple Spots Analysis",
        "📊 Analytics Dashboard"
    ])
    
    # Tab 1: Single Spot Prediction
    with tab1:
        st.header("Single Parking Spot Detection")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a trained model first in the sidebar.")
            st.info("Upload your model file to start making predictions.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Upload Spot Image")
                
                # Image upload
                uploaded_file = st.file_uploader(
                    "Choose a parking spot image",
                    type=['jpg', 'jpeg', 'png'],
                    key="single_spot_upload"
                )
                
                # Use demo image if no upload
                if uploaded_file is None and hasattr(st.session_state, 'demo_image'):
                    st.info("Using demo image. Upload your own image or try the demo options in the sidebar.")
                    image = st.session_state.demo_image.copy()
                    image_display = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    st.image(image_display, caption="Demo Parking Spot", use_column_width=True)
                elif uploaded_file is not None:
                    # Read uploaded image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_display = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    st.image(image_display, caption="Uploaded Parking Spot", use_column_width=True)
                else:
                    image = None
                    st.info("Please upload a parking spot image or select a demo option in the sidebar.")
                
                # Prediction button
                if image is not None:
                    if st.button("🚀 Predict Occupancy", type="primary", use_container_width=True):
                        with st.spinner("Analyzing parking spot..."):
                            # Make prediction
                            result = st.session_state.predictor.predict_single_spot(image)
                            
                            # Store result
                            st.session_state.single_prediction = result
                            
                            # Display result
                            with col2:
                                st.subheader("Prediction Result")
                                
                                # Color-coded result box
                                if result["prediction"] == "occupied":
                                    st.markdown(f"""
                                    <div class="prediction-box occupied">
                                        <h3>🚗 Spot is OCCUPIED</h3>
                                        <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                                        <p><strong>Probability (Occupied):</strong> {result['probability_occupied']:.2%}</p>
                                        <p><strong>Threshold used:</strong> {result['threshold_used']:.2f}</p>
                                        <p><strong>Lighting mode:</strong> {result['lighting_mode']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="prediction-box available">
                                        <h3>🅿️ Spot is AVAILABLE</h3>
                                        <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                                        <p><strong>Probability (Available):</strong> {result['probability_available']:.2%}</p>
                                        <p><strong>Threshold used:</strong> {result['threshold_used']:.2f}</p>
                                        <p><strong>Lighting mode:</strong> {result['lighting_mode']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Visualization
                                st.subheader("Analysis Visualization")
                                
                                # Create tabs for different visualizations
                                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                                    "Original vs Processed",
                                    "Foreground Mask",
                                    "Feature Analysis"
                                ])
                                
                                with viz_tab1:
                                    fig_col1, fig_col2 = st.columns(2)
                                    with fig_col1:
                                        st.image(image_display, caption="Original Image", use_column_width=True)
                                    with fig_col2:
                                        processed_rgb = cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB)
                                        st.image(processed_rgb, caption="Processed Image", use_column_width=True)
                                
                                with viz_tab2:
                                    mask_col1, mask_col2 = st.columns(2)
                                    with mask_col1:
                                        st.image(result["foreground_mask"], caption="Foreground Mask", use_column_width=True, clamp=True)
                                    with mask_col2:
                                        foreground_rgb = cv2.cvtColor(result["foreground_image"], cv2.COLOR_BGR2RGB)
                                        st.image(foreground_rgb, caption="Foreground Image", use_column_width=True)
                                
                                with viz_tab3:
                                    if result["features"]:
                                        features_df = pd.DataFrame.from_dict(result["features"], orient='index', columns=['Value'])
                                        st.dataframe(features_df.style.highlight_max(axis=0))
                                        
                                        # Bar chart of top features
                                        top_features = dict(sorted(result["features"].items(), key=lambda x: abs(x[1]), reverse=True)[:10])
                                        fig = px.bar(
                                            x=list(top_features.keys()),
                                            y=list(top_features.values()),
                                            title="Top 10 Feature Values",
                                            labels={'x': 'Feature', 'y': 'Value'}
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                # Download results
                                st.subheader("Export Results")
                                result_json = json.dumps(result, default=str, indent=2)
                                st.download_button(
                                    label="📥 Download Prediction Data (JSON)",
                                    data=result_json,
                                    file_name=f"parking_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
            
            if not st.session_state.get('single_prediction'):
                with col2:
                    st.info("👈 Upload an image and click 'Predict Occupancy' to see results here")
    
    # Tab 2: Multiple Spots Analysis
    with tab2:
        st.header("Multiple Parking Spots Analysis")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a trained model first in the sidebar.")
        else:
            st.subheader("Upload Multiple Spot Images")
            
            # Multiple file upload
            uploaded_files = st.file_uploader(
                "Upload multiple parking spot images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="multiple_spots_upload"
            )
            
            if uploaded_files:
                # Store uploaded images
                st.session_state.uploaded_images = []
                for uploaded_file in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    st.session_state.uploaded_images.append({
                        'name': uploaded_file.name,
                        'image': image
                    })
                
                # Display uploaded images
                st.write(f"**Uploaded {len(uploaded_files)} images**")
                
                cols = st.columns(min(4, len(uploaded_files)))
                for idx, img_data in enumerate(st.session_state.uploaded_images):
                    with cols[idx % 4]:
                        image_display = Image.fromarray(cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2RGB))
                        st.image(image_display, caption=img_data['name'][:20], use_column_width=True)
                
                # Batch prediction button
                if st.button("🔍 Analyze All Spots", type="primary", use_container_width=True):
                    with st.spinner(f"Analyzing {len(uploaded_files)} parking spots..."):
                        predictions = []
                        progress_bar = st.progress(0)
                        
                        for idx, img_data in enumerate(st.session_state.uploaded_images):
                            result = st.session_state.predictor.predict_single_spot(img_data['image'])
                            result['name'] = img_data['name']
                            predictions.append(result)
                            progress_bar.progress((idx + 1) / len(st.session_state.uploaded_images))
                        
                        st.session_state.predictions = predictions
                        
                        # Display summary
                        st.success(f"✅ Analysis complete! Processed {len(predictions)} spots.")
                        
                        # Summary statistics
                        occupied_count = sum(1 for p in predictions if p['prediction'] == 'occupied')
                        available_count = len(predictions) - occupied_count
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Spots", len(predictions))
                        with col2:
                            st.metric("Occupied", occupied_count, delta=None)
                        with col3:
                            st.metric("Available", available_count, delta=None)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        results_data = []
                        for pred in predictions:
                            results_data.append({
                                'Spot Name': pred['name'],
                                'Status': pred['prediction'],
                                'Confidence': f"{pred['confidence']:.2%}",
                                'Probability (Occupied)': f"{pred['probability_occupied']:.2%}",
                                'Lighting Mode': pred['lighting_mode']
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Visualization
                        st.subheader("Visualizations")
                        
                        # Pie chart
                        fig = px.pie(
                            names=['Occupied', 'Available'],
                            values=[occupied_count, available_count],
                            title="Parking Spot Occupancy Distribution",
                            color=['Occupied', 'Available'],
                            color_discrete_map={'Occupied': '#EF4444', 'Available': '#10B981'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence distribution
                        confidences = [p['confidence'] for p in predictions]
                        fig2 = px.histogram(
                            x=confidences,
                            nbins=20,
                            title="Confidence Distribution",
                            labels={'x': 'Confidence', 'y': 'Count'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Download all results
                        st.subheader("Export All Results")
                        all_results = {
                            'timestamp': datetime.now().isoformat(),
                            'model_accuracy': st.session_state.predictor.model_accuracy,
                            'lighting_mode': st.session_state.predictor.current_lighting_mode,
                            'threshold': st.session_state.predictor.current_threshold,
                            'predictions': predictions
                        }
                        
                        all_results_json = json.dumps(all_results, default=str, indent=2)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download All Results (JSON)",
                                data=all_results_json,
                                file_name=f"parking_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Export as CSV
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Results (CSV)",
                                data=csv_data,
                                file_name=f"parking_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            else:
                st.info("👆 Upload multiple parking spot images to analyze them in batch")
    
    # Tab 3: Analytics Dashboard
    with tab3:
        st.header("Analytics Dashboard")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a trained model first in the sidebar.")
        else:
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Model Accuracy",
                    f"{st.session_state.predictor.model_accuracy:.2%}",
                    help="Accuracy of the trained model on test data"
                )
            
            with col2:
                st.metric(
                    "Feature Count",
                    len(st.session_state.predictor.feature_names),
                    help="Number of features used by the model"
                )
            
            with col3:
                st.metric(
                    "Current Threshold",
                    f"{st.session_state.predictor.current_threshold:.2f}",
                    help="Current classification threshold"
                )
            
            # Model information
            with st.expander("Model Details", expanded=True):
                if st.session_state.predictor.feature_names:
                    st.write("**Feature Importance** (if available):")
                    # Create a placeholder for feature importance visualization
                    features_df = pd.DataFrame({
                        'Feature': st.session_state.predictor.feature_names,
                        'Importance': np.random.rand(len(st.session_state.predictor.feature_names))  # Placeholder
                    })
                    features_df = features_df.sort_values('Importance', ascending=False).head(10)
                    
                    fig = px.bar(
                        features_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 10 Features (Sample Importance)",
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature names not available in the model")
            
            # Historical predictions (if available)
            if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
                st.subheader("Recent Predictions Analysis")
                
                # Calculate statistics
                total_predictions = len(st.session_state.predictions)
                if total_predictions > 0:
                    occupied_pct = sum(1 for p in st.session_state.predictions if p['prediction'] == 'occupied') / total_predictions
                    avg_confidence = np.mean([p['confidence'] for p in st.session_state.predictions])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Occupancy Rate", f"{occupied_pct:.2%}")
                    with col2:
                        st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    
                    # Time series of predictions (simulated)
                    st.write("**Prediction Trends** (Last 24 hours - simulated)")
                    times = pd.date_range(end=datetime.now(), periods=24, freq='H')
                    simulated_occupancy = np.random.rand(24) * 0.3 + 0.4  # Simulated data
                    
                    trend_df = pd.DataFrame({
                        'Time': times,
                        'Occupancy Rate': simulated_occupancy
                    })
                    
                    fig = px.line(
                        trend_df,
                        x='Time',
                        y='Occupancy Rate',
                        title="Parking Occupancy Trends",
                        markers=True
                    )
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Make some predictions in the previous tabs to see analytics here")

# ============================
# RUN THE APP
# ============================
if __name__ == "__main__":
    main()
