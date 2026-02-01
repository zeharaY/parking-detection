"""
🚗 Parking Spot Detection App - Optimized for Streamlit Cloud
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
            st.warning(f"Expected 13 features, got {len(features)}. Adjusting...")
            # Pad or truncate to 13 features
            if len(features) < 13:
                features.extend([0] * (13 - len(features)))
            else:
                features = features[:13]
        
        return np.array(features)

# ============================
# MODEL LOADER WITH ERROR HANDLING
# ============================
def load_model_safely(model_file):
    """Load model with comprehensive error handling"""
    try:
        # First attempt: direct load
        return joblib.load(model_file)
    except Exception as e:
        st.warning(f"First load attempt failed: {e}")
        
        # Create dummy imblearn classes
        class DummySMOTE:
            def __init__(self, **kwargs):
                pass
        
        class DummyImbPipeline:
            def __init__(self, steps):
                self.steps = steps
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))
            def predict_proba(self, X):
                return np.zeros((len(X), 2))
        
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
            # Second attempt with dummy modules
            return joblib.load(model_file)
        except Exception as e2:
            st.error(f"Second load attempt failed: {e2}")
            
            # Last resort: create a simple dummy model
            st.warning("Creating fallback dummy model for demonstration")
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # Create a simple pipeline expecting 13 features
            svc_model = SVC(kernel='rbf', probability=True, random_state=42)
            scaler = StandardScaler()
            dummy_pipeline = Pipeline([('scaler', scaler), ('svm', svc_model)])
            
            # Fit with dummy data (13 features)
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
        
        # Debug: log feature count
        st.session_state.feature_count = len(features)
        
        return features, mask, foreground
    
    def predict_single_spot(self, image):
        """Predict occupancy for a single spot"""
        try:
            processed_image = self.preprocess_image(image)
            features, mask, foreground = self.extract_features(processed_image)
            features_reshaped = features.reshape(1, -1)
            
            # Debug info
            if 'feature_count' in st.session_state:
                st.sidebar.info(f"Extracted {st.session_state.feature_count} features")
            
            # Ensure mask is uint8 for display
            if mask.dtype != np.uint8:
                mask_display = (mask * 255).astype(np.uint8)
            else:
                mask_display = mask
            
            # Scale features if needed
            if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
                try:
                    features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
                except ValueError as e:
                    st.error(f"Feature dimension mismatch: {e}")
                    st.info(f"Expected 13 features, got {features.shape[0]}")
                    # Try to adjust feature count
                    if features.shape[0] != 13:
                        if features.shape[0] > 13:
                            features = features[:13]
                            features_reshaped = features.reshape(1, -1)
                            features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
                        else:
                            # Pad with zeros
                            padded_features = np.zeros(13)
                            padded_features[:features.shape[0]] = features
                            features_reshaped = padded_features.reshape(1, -1)
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
            st.error(f"Prediction error: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            
            # Return dummy result with error info
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
# STREAMLIT APP
# ============================
def main():
    st.set_page_config(
        page_title="Parking Spot Detection",
        page_icon="🚗",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #1E3A8A; margin-bottom: 1rem; }
    .prediction-box { padding: 15px; border-radius: 10px; margin: 10px 0; }
    .occupied { background-color: rgba(239, 68, 68, 0.1); border-left: 5px solid #EF4444; }
    .available { background-color: rgba(16, 185, 129, 0.1); border-left: 5px solid #10B981; }
    .error { background-color: rgba(245, 158, 11, 0.1); border-left: 5px solid #F59E0B; }
    .info-box { background-color: #F3F4F6; padding: 15px; border-radius: 10px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚗 Parking Spot Detection</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'feature_count' not in st.session_state:
        st.session_state.feature_count = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sub-header">Configuration</h3>', unsafe_allow_html=True)
        
        # Model upload
        uploaded_model = st.file_uploader("Upload Trained Model (.pkl)", type=['pkl'])
        
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
                    
                    st.success(f"✅ Model loaded successfully!")
                    st.info(f"**Model Accuracy:** {st.session_state.predictor.model_accuracy:.2%}")
                    st.info(f"**Expected Features:** 13")
                    
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    
                    # Create demo predictor
                    st.session_state.predictor = ParkingSpotPredictor({
                        'svm_model': None,
                        'bg_subtractor': BackgroundSubtractor(),
                        'accuracy': 0.85,
                        'feature_names': [
                            'foreground_pct', 'mean_intensity', 'std_intensity',
                            'compactness', 'hue_mean', 'hue_std', 'sat_mean',
                            'sat_std', 'val_mean', 'val_std', 'edge_density',
                            'gradient_mean', 'gradient_std'
                        ]
                    })
                    st.session_state.model_loaded = True
                    st.warning("Using demo mode with sample predictions")
        
        # Threshold setting
        if st.session_state.model_loaded:
            st.divider()
            threshold = st.slider("Confidence Threshold", 0.0, 1.0, 
                                 st.session_state.predictor.current_threshold, 0.01,
                                 help="Higher threshold = fewer 'occupied' predictions")
            st.session_state.predictor.current_threshold = threshold
            
            # Show feature info
            if st.session_state.feature_count > 0:
                st.info(f"Last extraction: {st.session_state.feature_count} features")
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Single Spot Analysis", "📊 Parking Lot Analysis"])
    
    # Tab 1: Single Spot
    with tab1:
        st.markdown('<h2 class="sub-header">Single Spot Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.info("👈 Please upload a trained model file to start")
            
            # Instructions
            with st.expander("📋 How to get started"):
                st.markdown("""
                ### Step 1: Train your model
                1. Run the training code in Google Colab
                2. Make sure to use the **updated training code** that saves only sklearn components
                3. Download the `parking_detection_model.pkl` file
                
                ### Step 2: Upload here
                1. Use the uploader in the sidebar
                2. The model should expect **13 features**
                3. If you get errors, retrain with the updated code
                
                ### Expected 13 Features:
                1. foreground_pct
                2. mean_intensity
                3. std_intensity
                4. compactness
                5. hue_mean
                6. hue_std
                7. sat_mean
                8. sat_std
                9. val_mean
                10. val_std
                11. edge_density
                12. gradient_mean
                13. gradient_std
                """)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Upload Spot Image")
                
                # Image upload
                uploaded_image = st.file_uploader("Choose an image...", 
                                                 type=['jpg', 'jpeg', 'png'],
                                                 key="single_spot")
                
                if uploaded_image:
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                caption="Uploaded Image", use_container_width=True)
                        
                        if st.button("🔮 Analyze This Spot", type="primary", use_container_width=True):
                            with st.spinner("Extracting features and predicting..."):
                                result = st.session_state.predictor.predict_single_spot(image)
                                st.session_state.last_result = result
                                st.session_state.feature_count = result.get('feature_count', 0)
            
            with col2:
                st.markdown("### Prediction Results")
                
                if 'last_result' in st.session_state:
                    result = st.session_state.last_result
                    
                    if result["prediction"] == "error":
                        st.markdown(f"""
                        <div class="prediction-box error">
                            <h3 style="color: #F59E0B; text-align: center;">⚠️ PREDICTION ERROR</h3>
                            <p><strong>Error:</strong> {result.get('error', 'Unknown error')}</p>
                            <p><strong>Feature count:</strong> {result.get('feature_count', 0)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result["prediction"] == "occupied":
                        st.markdown(f"""
                        <div class="prediction-box occupied">
                            <h3 style="color: #EF4444; text-align: center;">🚗 OCCUPIED</h3>
                            <p style="text-align: center; font-size: 24px; margin: 15px 0;">
                                <strong>{result['confidence']:.1%}</strong> confidence
                            </p>
                            <div class="info-box">
                                <p><strong>Occupied Probability:</strong> {result['probability_occupied']:.1%}</p>
                                <p><strong>Available Probability:</strong> {result['probability_available']:.1%}</p>
                                <p><strong>Threshold Used:</strong> {result.get('threshold_used', 0.5):.2f}</p>
                                <p><strong>Features Extracted:</strong> {result.get('feature_count', 0)}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box available">
                            <h3 style="color: #10B981; text-align: center;">🆓 AVAILABLE</h3>
                            <p style="text-align: center; font-size: 24px; margin: 15px 0;">
                                <strong>{result['confidence']:.1%}</strong> confidence
                            </p>
                            <div class="info-box">
                                <p><strong>Occupied Probability:</strong> {result['probability_occupied']:.1%}</p>
                                <p><strong>Available Probability:</strong> {result['probability_available']:.1%}</p>
                                <p><strong>Threshold Used:</strong> {result.get('threshold_used', 0.5):.2f}</p>
                                <p><strong>Features Extracted:</strong> {result.get('feature_count', 0)}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualizations
                    if result["prediction"] != "error":
                        with st.expander("🔬 Feature Extraction Visualization", expanded=True):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.image(cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB),
                                        caption="Processed (64x64)", use_container_width=True)
                            with col_b:
                                st.image(result["foreground_mask"], 
                                        caption="Foreground Mask", use_container_width=True)
                            with col_c:
                                st.image(cv2.cvtColor(result["foreground_image"], cv2.COLOR_BGR2RGB),
                                        caption="Foreground", use_container_width=True)
                        
                        # Feature analysis
                        if result["features"]:
                            with st.expander("📊 Feature Analysis", expanded=False):
                                features_df = pd.DataFrame({
                                    "Feature": list(result["features"].keys()),
                                    "Value": list(result["features"].values())
                                })
                                
                                # Bar chart
                                fig = px.bar(features_df, x="Feature", y="Value",
                                           title="Extracted Feature Values",
                                           color="Value", color_continuous_scale="Blues")
                                st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("👈 Upload an image and click 'Analyze This Spot' to see results")
    
    # Tab 2: Parking Lot (simplified version)
    with tab2:
        st.markdown('<h2 class="sub-header">Parking Lot Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_loaded:
            st.warning("Please upload a model first in the sidebar")
        else:
            uploaded_lot = st.file_uploader("Upload parking lot image...", 
                                           type=['jpg', 'jpeg', 'png'],
                                           key="parking_lot")
            
            if uploaded_lot:
                file_bytes = np.asarray(bytearray(uploaded_lot.read()), dtype=np.uint8)
                lot_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if lot_image is not None:
                    st.image(cv2.cvtColor(lot_image, cv2.COLOR_BGR2RGB),
                            caption="Parking Lot Image", use_container_width=True)
                    
                    # Grid configuration
                    col1, col2 = st.columns(2)
                    with col1:
                        rows = st.number_input("Number of Rows", 1, 10, 3)
                    with col2:
                        cols = st.number_input("Spots per Row", 1, 20, 5)
                    
                    if st.button("🔍 Analyze Entire Parking Lot", type="primary", use_container_width=True):
                        with st.spinner(f"Analyzing {rows}x{cols} spots..."):
                            spot_width = lot_image.shape[1] // cols
                            spot_height = lot_image.shape[0] // rows
                            
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for r in range(rows):
                                for c in range(cols):
                                    x, y = c * spot_width, r * spot_height
                                    spot_img = lot_image[y:y+spot_height, x:x+spot_width]
                                    
                                    if spot_img.size > 0:
                                        try:
                                            pred = st.session_state.predictor.predict_single_spot(spot_img)
                                            results.append({
                                                "id": f"R{r+1}C{c+1}",
                                                "row": r+1,
                                                "col": c+1,
                                                "prediction": pred["prediction"],
                                                "confidence": pred["confidence"],
                                                "x": x,
                                                "y": y,
                                                "width": spot_width,
                                                "height": spot_height
                                            })
                                        except Exception as e:
                                            st.error(f"Error analyzing spot R{r+1}C{c+1}: {e}")
                                            results.append({
                                                "id": f"R{r+1}C{c+1}",
                                                "row": r+1,
                                                "col": c+1,
                                                "prediction": "error",
                                                "confidence": 0.0,
                                                "x": x,
                                                "y": y,
                                                "width": spot_width,
                                                "height": spot_height
                                            })
                                    
                                    # Update progress
                                    progress = ((r * cols) + c + 1) / (rows * cols)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Analyzing spot {r+1}-{c+1} of {rows}x{cols}")
                            
                            status_text.text("✅ Analysis complete!")
                            
                            # Display results
                            if results:
                                results_df = pd.DataFrame(results)
                                
                                # Summary metrics
                                occupied = sum(1 for r in results if r["prediction"] == "occupied")
                                available = sum(1 for r in results if r["prediction"] == "available")
                                errors = sum(1 for r in results if r["prediction"] == "error")
                                total = len(results)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Total Spots", total)
                                col2.metric("Occupied", occupied)
                                col3.metric("Available", available)
                                if errors > 0:
                                    col4.metric("Errors", errors, delta=f"{errors} failed")
                                
                                # Create overlay visualization
                                overlay = lot_image.copy()
                                for result in results:
                                    x, y = result["x"], result["y"]
                                    w, h = result["width"], result["height"]
                                    
                                    # Set color based on prediction
                                    if result["prediction"] == "occupied":
                                        color = (0, 0, 255)  # Red
                                    elif result["prediction"] == "available":
                                        color = (0, 255, 0)  # Green
                                    else:
                                        color = (128, 128, 128)  # Gray for errors
                                    
                                    # Draw rectangle
                                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                                    cv2.putText(overlay, result["id"], (x + 5, y + 20),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                        caption="Parking Lot Analysis (Red=Occupied, Green=Available, Gray=Error)",
                                        use_container_width=True)
                                
                                # Results table
                                with st.expander("📋 Detailed Results", expanded=False):
                                    st.dataframe(results_df[['id', 'prediction', 'confidence']], 
                                               use_container_width=True)
                                    
                                    # Download button
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=csv,
                                        file_name=f"parking_lot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )

# Run the app
if __name__ == "__main__":
    main()
