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
    
    # Main content tabs (same as before, unchanged)
    # ... [Keep all the tab1, tab2, tab3 code exactly as in your original app]

# ============================
# RUN THE APP
# ============================
if __name__ == "__main__":
    main()
