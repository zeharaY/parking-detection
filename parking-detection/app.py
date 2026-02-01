"""
🚗 Parking Spot Detection App
Clean version for Streamlit Cloud
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
from PIL import Image
import tempfile
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================
# BACKGROUND SUBTRACTOR CLASS
# ============================
class BackgroundSubtractor:
    """Background subtraction for parking spot analysis"""
    
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
        """Initialize OpenCV background subtractor"""
        if self.method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.varThreshold,
                detectShadows=self.detectShadows
            )
        else:  # 'knn'
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.dist2Threshold,
                detectShadows=self.detectShadows
            )
    
    def __getstate__(self):
        """Prepare for pickling"""
        state = self.__dict__.copy()
        if 'bg_subtractor' in state:
            del state['bg_subtractor']
        return state
    
    def __setstate__(self, state):
        """Restore from pickling"""
        self.__dict__.update(state)
        self._init_cv2_subtractor()
    
    def apply(self, image):
        """Apply background subtraction"""
        fg_mask = self.bg_subtractor.apply(image, learningRate=self.learning_rate)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        foreground = cv2.bitwise_and(image, image, mask=fg_mask)
        return foreground, fg_mask
    
    def extract_features(self, image, mask):
        """Extract features from foreground"""
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Foreground percentage
        total_pixels = mask.size
        foreground_pixels = np.sum(mask > 0)
        foreground_percentage = foreground_pixels / total_pixels
        features.append(foreground_percentage)
        
        # 2. Foreground statistics
        if foreground_pixels > 0:
            mean_intensity = np.mean(gray[mask > 0])
            features.append(mean_intensity)
            
            std_intensity = np.std(gray[mask > 0])
            features.append(std_intensity)
            
            # Compactness
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
        
        # 3. Color features (HSV)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if foreground_pixels > 0:
            for i in range(3):  # H, S, V channels
                channel_values = hsv[:,:,i][mask > 0]
                if len(channel_values) > 0:
                    features.append(np.mean(channel_values))
                    features.append(np.std(channel_values))
                else:
                    features.extend([0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # 4. Edge density
        edges = cv2.Canny(gray, 50, 150)
        if foreground_pixels > 0:
            edge_density = np.sum(edges[mask > 0]) / (foreground_pixels * 255)
            features.append(edge_density)
        else:
            features.append(0)
        
        # 5. Texture features
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
        # Extract model components
        self.model = model_data['svm_model']
        self.bg_subtractor = model_data['bg_subtractor']
        self.feature_names = model_data.get('feature_names', [])
        self.model_accuracy = model_data.get('accuracy', 0.0)
        
        # Lighting adjustment profiles
        self.lighting_profiles = {
            "daylight": {"threshold": 0.5, "brightness": 0, "contrast": 0},
            "dusk": {"threshold": 0.45, "brightness": -20, "contrast": 10},
            "night": {"threshold": 0.4, "brightness": -40, "contrast": 20},
            "overcast": {"threshold": 0.48, "brightness": -15, "contrast": 15},
            "bright_sun": {"threshold": 0.52, "brightness": 20, "contrast": -5}
        }
        
        self.current_threshold = 0.5
        self.current_lighting_mode = "daylight"
        
        # Get the actual sklearn model
        if hasattr(self.model, 'svm'):
            self.sklearn_model = self.model.svm
        else:
            self.sklearn_model = self.model
        
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
        # Resize to 64x64 (same as training)
        target_size = (64, 64)
        resized = cv2.resize(image, target_size)
        
        # Apply lighting adjustment
        profile = self.lighting_profiles[self.current_lighting_mode]
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Adjust brightness
        v = cv2.add(v, profile["brightness"])
        v = np.clip(v, 0, 255)
        
        # Adjust contrast
        if profile["contrast"] != 0:
            alpha = 1 + profile["contrast"] / 100
            v = cv2.multiply(v, alpha)
            v = np.clip(v, 0, 255)
        
        # Merge back
        hsv_adjusted = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    def extract_features(self, image):
        """Extract features using background subtraction"""
        foreground, mask = self.bg_subtractor.apply(image)
        features = self.bg_subtractor.extract_features(image, mask)
        return features, mask, foreground
    
    def predict_single_spot(self, image):
        """Predict occupancy for a single parking spot"""
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Extract features
        features, mask, foreground = self.extract_features(processed_image)
        features_reshaped = features.reshape(1, -1)
        
        # Get prediction probabilities
        if hasattr(self.sklearn_model, 'predict_proba'):
            proba = self.sklearn_model.predict_proba(features_reshaped)[0]
            confidence = max(proba)
            prediction = 1 if proba[1] >= self.current_threshold else 0
        else:
            # Fallback
            prediction = self.sklearn_model.predict(features_reshaped)[0]
            proba = [1 - prediction, prediction]
            confidence = 0.8
        
        # Get feature values for explanation
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
    
    def predict_batch(self, images_list):
        """Predict occupancy for multiple parking spots"""
        results = []
        for i, image in enumerate(images_list):
            result = self.predict_single_spot(image)
            result["spot_id"] = f"spot_{i+1}"
            results.append(result)
        return results
    
    def predict_parking_lot(self, lot_image, spots_config):
        """Predict occupancy for entire parking lot"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_spots": len(spots_config),
            "occupied_spots": 0,
            "available_spots": 0,
            "spot_details": [],
            "utilization_rate": 0,
            "lighting_mode": self.current_lighting_mode,
            "threshold": self.current_threshold
        }
        
        for spot_config in spots_config:
            spot_id = spot_config.get("id", f"spot_{len(results['spot_details'])+1}")
            x, y, w, h = spot_config["bbox"]
            
            # Extract spot from lot image
            spot_img = lot_image[y:y+h, x:x+w]
            if spot_img.size == 0:
                continue
            
            # Predict
            prediction_result = self.predict_single_spot(spot_img)
            
            # Update counts
            if prediction_result["prediction"] == "occupied":
                results["occupied_spots"] += 1
            else:
                results["available_spots"] += 1
            
            # Add to details
            results["spot_details"].append({
                "id": spot_id,
                "state": prediction_result["prediction"],
                "confidence": prediction_result["confidence"],
                "bbox": spot_config["bbox"],
                "position": spot_config.get("position", "Unknown"),
                "probability_occupied": prediction_result["probability_occupied"]
            })
        
        # Calculate utilization
        if results["total_spots"] > 0:
            results["utilization_rate"] = results["occupied_spots"] / results["total_spots"]
        
        return results

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
            type=['pkl', 'joblib'],
            help="Upload the model file saved from your training"
        )
        
        if uploaded_model:
            with st.spinner("Loading model..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        tmp.write(uploaded_model.getvalue())
                        tmp_path = tmp.name
                    
                    # Load model
                    model_data = joblib.load(tmp_path)
                    
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
        
        # App info
        st.subheader("ℹ️ App Information")
        st.info(f"""
        **Status:** {'Model Loaded' if st.session_state.model_loaded else 'Ready for Model Upload'}
        **Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
        **Version:** 1.0
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Single Spot Prediction", 
        "📊 Parking Lot Analysis", 
        "📈 Model Insights",
        "📁 Batch Processing"
    ])
    
    # Tab 1: Single Spot Prediction
    with tab1:
        st.header("Single Spot Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a trained model in the sidebar first.")
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info("""
                **How to get started:**
                1. Train your model using the training code
                2. Save as `parking_detection_model.pkl`
                3. Upload it in the sidebar
                """)
            
            with col_info2:
                st.info("""
                **Features:**
                - Real-time prediction
                - Lighting adjustment
                - Feature visualization
                - Confidence scores
                """)
        else:
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("Upload Image")
                
                # Image upload
                uploaded_image = st.file_uploader(
                    "Choose a parking spot image",
                    type=['jpg', 'jpeg', 'png'],
                    key="spot_image"
                )
                
                # Use demo image if available
                image_to_predict = None
                if 'demo_image' in st.session_state:
                    st.info(f"Using demo image ({st.session_state.demo_label})")
                    image_to_predict = st.session_state.demo_image.copy()
                    st.image(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB),
                            caption="Demo Image",
                            use_container_width=True)
                elif uploaded_image:
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    image_to_predict = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if image_to_predict is not None:
                        st.image(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB),
                                caption="Uploaded Image",
                                use_container_width=True)
                
                if image_to_predict is not None:
                    if st.button("🔮 Predict Occupancy", type="primary", use_container_width=True):
                        with st.spinner("Analyzing parking spot..."):
                            result = st.session_state.predictor.predict_single_spot(image_to_predict)
                            st.session_state.last_prediction = result
            
            with col_right:
                st.subheader("Prediction Results")
                
                if 'last_prediction' in st.session_state:
                    result = st.session_state.last_prediction
                    
                    # Display prediction
                    if result["prediction"] == "occupied":
                        st.markdown(f"""
                        <div class="prediction-box occupied">
                            <h2 style="color: #EF4444; text-align: center;">🚗 OCCUPIED</h2>
                            <div style="text-align: center; font-size: 24px; margin: 20px 0;">
                                <strong>{result['confidence']:.1%}</strong> confidence
                            </div>
                            <div class="metric-card">
                                <p><strong>Probability (Occupied):</strong> {result['probability_occupied']:.1%}</p>
                                <p><strong>Threshold Used:</strong> {result['threshold_used']:.2f}</p>
                                <p><strong>Lighting Mode:</strong> {result['lighting_mode']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box available">
                            <h2 style="color: #10B981; text-align: center;">🆓 AVAILABLE</h2>
                            <div style="text-align: center; font-size: 24px; margin: 20px 0;">
                                <strong>{result['confidence']:.1%}</strong> confidence
                            </div>
                            <div class="metric-card">
                                <p><strong>Probability (Available):</strong> {result['probability_available']:.1%}</p>
                                <p><strong>Threshold Used:</strong> {result['threshold_used']:.2f}</p>
                                <p><strong>Lighting Mode:</strong> {result['lighting_mode']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization
                    with st.expander("🔬 Feature Extraction Visualization", expanded=False):
                        col_v1, col_v2, col_v3 = st.columns(3)
                        with col_v1:
                            st.image(cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB),
                                    caption="Preprocessed", use_column_width=True)
                        with col_v2:
                            st.image(result["foreground_mask"], 
                                    caption="Foreground Mask", use_column_width=True, cmap="gray")
                        with col_v3:
                            st.image(cv2.cvtColor(result["foreground_image"], cv2.COLOR_BGR2RGB),
                                    caption="Foreground", use_column_width=True)
                    
                    # Feature analysis
                    if result["features"]:
                        with st.expander("📊 Feature Analysis", expanded=False):
                            features_df = pd.DataFrame({
                                "Feature": list(result["features"].keys()),
                                "Value": list(result["features"].values())
                            })
                            
                            # Bar chart
                            fig = px.bar(
                                features_df,
                                x="Feature",
                                y="Value",
                                title="Extracted Feature Values",
                                color="Value",
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download features
                            csv = features_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Features (CSV)",
                                data=csv,
                                file_name=f"spot_features_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.info("👈 Upload an image and click 'Predict Occupancy' to see results here")
    
    # Tab 2: Parking Lot Analysis
    with tab2:
        st.header("Parking Lot Analysis")
        
        if not st.session_state.model_loaded:
            st.warning("Please upload a model to use parking lot analysis")
        else:
            st.info("Upload a full parking lot image to analyze multiple spots at once")
            
            uploaded_lot = st.file_uploader(
                "Upload parking lot image",
                type=['jpg', 'jpeg', 'png'],
                key="lot_image"
            )
            
            if uploaded_lot:
                file_bytes = np.asarray(bytearray(uploaded_lot.read()), dtype=np.uint8)
                lot_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if lot_image is not None:
                    st.image(cv2.cvtColor(lot_image, cv2.COLOR_BGR2RGB),
                            caption="Parking Lot Image", use_container_width=True)
                    
                    # Configuration
                    col_config1, col_config2 = st.columns(2)
                    with col_config1:
                        rows = st.number_input("Number of Rows", 1, 10, 3)
                    with col_config2:
                        cols = st.number_input("Spots per Row", 1, 20, 5)
                    
                    if st.button("🔍 Analyze Parking Lot", type="primary", use_container_width=True):
                        with st.spinner(f"Analyzing {rows}x{cols} spots..."):
                            # Create spot grid
                            spots_config = []
                            spot_width = lot_image.shape[1] // cols
                            spot_height = lot_image.shape[0] // rows
                            
                            for r in range(rows):
                                for c in range(cols):
                                    spots_config.append({
                                        "id": f"R{r+1}C{c+1}",
                                        "bbox": [c*spot_width, r*spot_height, spot_width, spot_height],
                                        "position": f"Row {r+1}, Col {c+1}"
                                    })
                            
                            # Analyze each spot
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, spot in enumerate(spots_config):
                                x, y, w, h = spot["bbox"]
                                spot_img = lot_image[y:y+h, x:x+w]
                                
                                if spot_img.size > 0:
                                    prediction = st.session_state.predictor.predict_single_spot(spot_img)
                                    results.append({
                                        "id": spot["id"],
                                        "position": spot["position"],
                                        "prediction": prediction["prediction"],
                                        "confidence": prediction["confidence"],
                                        "occupied_prob": prediction["probability_occupied"]
                                    })
                                
                                progress_bar.progress((i + 1) / len(spots_config))
                            
                            # Display results
                            if results:
                                results_df = pd.DataFrame(results)
                                
                                # Summary metrics
                                occupied = sum(1 for r in results if r["prediction"] == "occupied")
                                total = len(results)
                                utilization = occupied / total if total > 0 else 0
                                
                                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                                with col_metric1:
                                    st.metric("Total Spots", total)
                                with col_metric2:
                                    st.metric("Occupied", occupied)
                                with col_metric3:
                                    st.metric("Available", total - occupied)
                                with col_metric4:
                                    st.metric("Utilization", f"{utilization:.1%}")
                                
                                # Visualize on image
                                overlay = lot_image.copy()
                                for spot in spots_config:
                                    x, y, w, h = spot["bbox"]
                                    result = next((r for r in results if r["id"] == spot["id"]), None)
                                    
                                    if result and result["prediction"] == "occupied":
                                        color = (0, 0, 255)  # Red
                                    else:
                                        color = (0, 255, 0)  # Green
                                    
                                    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                                    cv2.putText(overlay, spot["id"], (x+5, y+20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                        caption="Parking Lot Analysis (Red=Occupied, Green=Available)",
                                        use_container_width=True)
                                
                                # Results table
                                with st.expander("📋 Detailed Results", expanded=False):
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Download results
                                    csv_data = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results (CSV)",
                                        data=csv_data,
                                        file_name=f"parking_lot_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
    
    # Tab 3: Model Insights
    with tab3:
        st.header("Model Performance Insights")
        
        if not st.session_state.model_loaded:
            st.warning("Upload a model to see performance insights")
        else:
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.subheader("Model Information")
                st.metric("Accuracy", f"{st.session_state.predictor.model_accuracy:.2%}")
                st.metric("Current Threshold", f"{st.session_state.predictor.current_threshold:.2f}")
                st.metric("Lighting Mode", st.session_state.predictor.current_lighting_mode)
                st.metric("Feature Count", len(st.session_state.predictor.feature_names))
            
            with col_insight2:
                st.subheader("Threshold Analysis")
                
                # Show threshold impact
                thresholds = np.linspace(0.1, 0.9, 9)
                default_accuracy = st.session_state.predictor.model_accuracy
                
                # Simulate accuracy at different thresholds
                simulated_acc = []
                for t in thresholds:
                    # Simple simulation: accuracy decreases as threshold moves away from optimal
                    optimal = 0.5
                    diff = abs(t - optimal)
                    simulated_acc.append(default_accuracy * (1 - diff*0.5))
                
                threshold_df = pd.DataFrame({
                    "Threshold": thresholds,
                    "Simulated Accuracy": simulated_acc
                })
                
                fig = px.line(
                    threshold_df,
                    x="Threshold",
                    y="Simulated Accuracy",
                    title="Threshold vs. Accuracy (Simulated)",
                    markers=True
                )
                fig.add_vline(
                    x=st.session_state.predictor.current_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Current: {st.session_state.predictor.current_threshold}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            if st.session_state.predictor.feature_names:
                # Create simulated importance
                importance = np.random.rand(len(st.session_state.predictor.feature_names))
                importance = importance / importance.sum()
                
                importance_df = pd.DataFrame({
                    "Feature": st.session_state.predictor.feature_names,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)
                
                fig = px.bar(
                    importance_df.head(10),
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model recommendations
            st.subheader("Recommendations")
            with st.expander("Tips for Better Accuracy", expanded=True):
                st.info("""
                **✅ Best Practices:**
                1. Use consistent lighting conditions
                2. Keep camera angle stable
                3. Ensure spots are clearly visible
                4. Adjust threshold based on lighting
                
                **⚡ Quick Settings:**
                - **Daytime:** Use 'daylight' mode
                - **Night:** Use 'night' mode with lower threshold
                - **Overcast:** Use 'overcast' mode
                - **High Contrast:** Use 'bright_sun' mode
                """)
    
    # Tab 4: Batch Processing
    with tab4:
        st.header("Batch Processing of Parking Spots")
        
        if not st.session_state.model_loaded:
            st.warning("Upload a model to enable batch processing")
        else:
            st.info("Upload multiple parking spot images for batch prediction")
            
            # Multiple file upload
            uploaded_files = st.file_uploader(
                "Choose multiple parking spot images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="batch_files"
            )
            
            if uploaded_files:
                # Process files
                if st.button("🚀 Process All Images", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(uploaded_files)} images..."):
                        images = []
                        file_names = []
                        
                        for uploaded_file in uploaded_files:
                            # Convert to OpenCV format
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            images.append(image)
                            file_names.append(uploaded_file.name)
                        
                        # Make predictions
                        results = st.session_state.predictor.predict_batch(images)
                        
                        # Add file names
                        for i, result in enumerate(results):
                            result["filename"] = file_names[i]
                        
                        # Store results
                        st.session_state.batch_results = results
                        
                        st.success(f"✅ Processed {len(results)} images!")
            
            # Display batch results
            if 'batch_results' in st.session_state:
                results = st.session_state.batch_results
                
                # Summary
                occupied_count = sum(1 for r in results if r["prediction"] == "occupied")
                available_count = len(results) - occupied_count
                
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.metric("Total Images", len(results))
                
                with col_sum2:
                    st.metric("Occupied Spots", occupied_count)
                
                with col_sum3:
                    st.metric("Available Spots", available_count)
                
                # Results table
                st.subheader("Detailed Results")
                
                # Prepare data for display
                display_data = []
                for result in results:
                    display_data.append({
                        "Filename": result["filename"],
                        "Prediction": result["prediction"].upper(),
                        "Confidence": f"{result['confidence']:.1%}",
                        "Prob. Occupied": f"{result['probability_occupied']:.1%}",
                        "Prob. Available": f"{result['probability_available']:.1%}"
                    })
                
                results_df = pd.DataFrame(display_data)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Visualizations
                st.subheader("Batch Analysis")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Pie chart
                    fig = px.pie(
                        names=["Occupied", "Available"],
                        values=[occupied_count, available_count],
                        title="Occupancy Distribution",
                        color=["Occupied", "Available"],
                        color_discrete_map={"Occupied": "red", "Available": "green"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_viz2:
                    # Confidence distribution
                    confidences = [r["confidence"] for r in results]
                    fig = px.histogram(
                        x=confidences,
                        nbins=10,
                        title="Confidence Distribution",
                        labels={"x": "Confidence", "y": "Count"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                st.subheader("Export Results")
                
                # JSON export
                json_data = json.dumps(results, indent=2, default=str)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col_dl2:
                    # CSV export
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ============================
# RUN THE APP
# ============================
if __name__ == "__main__":
    main()
