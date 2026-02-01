"""
🚗 Parking Spot Detection App - Simplified for Streamlit Cloud
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

# ============================
# BACKGROUND SUBTRACTOR
# ============================
class BackgroundSubtractor:
    def __init__(self, method='mog2', learning_rate=0.001):
        self.method = method
        self.learning_rate = learning_rate
        self._init_cv2_subtractor()

    def _init_cv2_subtractor(self):
        if self.method == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16)
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)

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
        """Extract 13 features (matching training)"""
        features = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_pixels = mask.size
        foreground_pixels = np.sum(mask > 0)
        foreground_percentage = foreground_pixels / total_pixels if total_pixels > 0 else 0
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
                features.append(0)
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
            edge_density = np.sum(edges[mask > 0]) / (foreground_pixels * 255) if foreground_pixels > 0 else 0
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

        # Ensure exactly 13 features
        if len(features) < 13:
            features.extend([0] * (13 - len(features)))
        elif len(features) > 13:
            features = features[:13]
        
        return np.array(features)

# ============================
# MODEL LOADER
# ============================
def load_model_safely(model_file):
    """Load model with error handling"""
    try:
        # Create dummy imblearn module
        import types
        
        class DummySMOTE:
            def __init__(self, **kwargs):
                pass
        
        class DummyPipeline:
            def __init__(self, steps):
                self.steps = steps
            def fit(self, X, y):
                return self
        
        # Create dummy modules
        imblearn_module = types.ModuleType('imblearn')
        over_sampling_module = types.ModuleType('imblearn.over_sampling')
        pipeline_module = types.ModuleType('imblearn.pipeline')
        
        over_sampling_module.SMOTE = DummySMOTE
        pipeline_module.Pipeline = DummyPipeline
        imblearn_module.over_sampling = over_sampling_module
        imblearn_module.pipeline = pipeline_module
        
        # Add to sys.modules
        sys.modules['imblearn'] = imblearn_module
        sys.modules['imblearn.over_sampling'] = over_sampling_module
        sys.modules['imblearn.pipeline'] = pipeline_module
        
        return joblib.load(model_file)
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================
# PREDICTOR CLASS
# ============================
class ParkingSpotPredictor:
    def __init__(self, model_data):
        if model_data is None:
            # Create demo predictor
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            svc = SVC(kernel='rbf', probability=True, random_state=42)
            scaler = StandardScaler()
            pipeline = Pipeline([('scaler', scaler), ('svm', svc)])
            
            # Fit with dummy data
            X = np.random.randn(100, 13)
            y = np.random.randint(0, 2, 100)
            pipeline.fit(X, y)
            
            self.model = pipeline
            self.bg_subtractor = BackgroundSubtractor()
            self.model_accuracy = 0.85
        else:
            self.model = model_data.get('svm_model')
            self.bg_subtractor = model_data.get('bg_subtractor', BackgroundSubtractor())
            self.model_accuracy = model_data.get('accuracy', 0.0)
        
        self.feature_names = [
            'foreground_pct', 'mean_intensity', 'std_intensity',
            'compactness', 'hue_mean', 'hue_std', 'sat_mean',
            'sat_std', 'val_mean', 'val_std', 'edge_density',
            'gradient_mean', 'gradient_std'
        ]
        self.current_threshold = 0.5
    
    def preprocess_image(self, image):
        """Resize to 64x64"""
        return cv2.resize(image, (64, 64))
    
    def extract_features(self, image):
        foreground, mask = self.bg_subtractor.apply(image)
        features = self.bg_subtractor.extract_features(image, mask)
        return features, mask, foreground
    
    def predict_single_spot(self, image):
        try:
            processed_image = self.preprocess_image(image)
            features, mask, foreground = self.extract_features(processed_image)
            features_reshaped = features.reshape(1, -1)
            
            # Convert mask for display
            if mask.dtype != np.uint8:
                mask_display = (mask * 255).astype(np.uint8)
            else:
                mask_display = mask
            
            # Scale features
            if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
                try:
                    features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
                except:
                    # If feature mismatch, adjust
                    if len(features) != 13:
                        if len(features) > 13:
                            features = features[:13]
                        else:
                            features = np.pad(features, (0, 13 - len(features)))
                        features_reshaped = features.reshape(1, -1)
                        features_reshaped = self.model.named_steps['scaler'].transform(features_reshaped)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_reshaped)[0]
                confidence = max(proba)
                prediction = 1 if proba[1] >= self.current_threshold else 0
            else:
                prediction = self.model.predict(features_reshaped)[0]
                proba = [1 - prediction, prediction]
                confidence = 0.8
            
            return {
                "prediction": "occupied" if prediction == 1 else "available",
                "confidence": float(confidence),
                "probability_occupied": float(proba[1]),
                "probability_available": float(proba[0]),
                "features": dict(zip(self.feature_names[:len(features)], features.tolist())),
                "foreground_mask": mask_display,
                "foreground_image": foreground,
                "processed_image": processed_image
            }
        except Exception as e:
            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e)
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
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; }
    .prediction-box { padding: 15px; border-radius: 10px; margin: 10px 0; }
    .occupied { background-color: rgba(239, 68, 68, 0.1); border-left: 5px solid #EF4444; }
    .available { background-color: rgba(16, 185, 129, 0.1); border-left: 5px solid #10B981; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚗 Parking Spot Detection</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.title("Configuration")
        
        # Model upload
        uploaded_model = st.file_uploader("Upload Model (.pkl)", type=['pkl'])
        
        if uploaded_model and not st.session_state.model_loaded:
            with st.spinner("Loading..."):
                try:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        tmp.write(uploaded_model.getvalue())
                        tmp_path = tmp.name
                    
                    # Load model
                    model_data = load_model_safely(tmp_path)
                    
                    if model_data:
                        st.session_state.predictor = ParkingSpotPredictor(model_data)
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded!")
                    else:
                        st.warning("Using demo mode")
                        st.session_state.predictor = ParkingSpotPredictor(None)
                        st.session_state.model_loaded = True
                    
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Using demo mode")
                    st.session_state.predictor = ParkingSpotPredictor(None)
                    st.session_state.model_loaded = True
        
        # Demo mode button
        if not st.session_state.model_loaded:
            if st.button("Use Demo Mode", type="primary"):
                st.session_state.predictor = ParkingSpotPredictor(None)
                st.session_state.model_loaded = True
                st.success("Demo mode activated!")
        
        # Settings
        if st.session_state.model_loaded:
            st.divider()
            threshold = st.slider("Threshold", 0.0, 1.0, st.session_state.predictor.current_threshold, 0.01)
            st.session_state.predictor.current_threshold = threshold
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Upload a model or use demo mode to start")
        st.markdown("""
        ### How to get started:
        1. **Upload a trained model** (.pkl file) in the sidebar
        2. **Or click "Use Demo Mode"** for a demonstration
        3. **Upload images** to analyze parking spots
        
        ### Expected model format:
        - Trained with the parking spot detection training code
        - Should contain an SVM model and BackgroundSubtractor
        - Should expect 13 features
        """)
    else:
        tab1, tab2 = st.tabs(["Single Spot", "Parking Lot"])
        
        with tab1:
            st.header("Single Spot Analysis")
            
            uploaded_image = st.file_uploader("Upload spot image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_image:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                caption="Original Image", use_container_width=True)
                        
                        if st.button("Analyze", type="primary"):
                            with st.spinner("Processing..."):
                                result = st.session_state.predictor.predict_single_spot(image)
                                st.session_state.result = result
                    
                    with col2:
                        if 'result' in st.session_state:
                            result = st.session_state.result
                            
                            if result["prediction"] == "occupied":
                                st.markdown(f"""
                                <div class="prediction-box occupied">
                                    <h3 style="color: #EF4444; text-align: center;">🚗 OCCUPIED</h3>
                                    <p style="text-align: center; font-size: 24px;">
                                        Confidence: <strong>{result['confidence']:.1%}</strong>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif result["prediction"] == "available":
                                st.markdown(f"""
                                <div class="prediction-box available">
                                    <h3 style="color: #10B981; text-align: center;">🆓 AVAILABLE</h3>
                                    <p style="text-align: center; font-size: 24px;">
                                        Confidence: <strong>{result['confidence']:.1%}</strong>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                            
                            # Visualizations
                            if "processed_image" in result:
                                with st.expander("Feature Extraction"):
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.image(cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB),
                                                caption="Processed")
                                    with cols[1]:
                                        st.image(result["foreground_mask"], caption="Foreground Mask")
                                    with cols[2]:
                                        st.image(cv2.cvtColor(result["foreground_image"], cv2.COLOR_BGR2RGB),
                                                caption="Foreground")
        
        with tab2:
            st.header("Parking Lot Analysis")
            
            uploaded_lot = st.file_uploader("Upload parking lot image", type=['jpg', 'jpeg', 'png'], key="lot")
            
            if uploaded_lot:
                file_bytes = np.asarray(bytearray(uploaded_lot.read()), dtype=np.uint8)
                lot_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if lot_image is not None:
                    st.image(cv2.cvtColor(lot_image, cv2.COLOR_BGR2RGB),
                            caption="Parking Lot", use_container_width=True)
                    
                    cols = st.columns(2)
                    with cols[0]:
                        rows = st.number_input("Rows", 1, 10, 3)
                    with cols[1]:
                        cols_num = st.number_input("Columns", 1, 20, 5)
                    
                    if st.button("Analyze Parking Lot", type="primary"):
                        with st.spinner(f"Analyzing {rows}x{cols_num} spots..."):
                            spot_width = lot_image.shape[1] // cols_num
                            spot_height = lot_image.shape[0] // rows
                            
                            results = []
                            progress_bar = st.progress(0)
                            
                            for r in range(rows):
                                for c in range(cols_num):
                                    x, y = c * spot_width, r * spot_height
                                    spot_img = lot_image[y:y+spot_height, x:x+spot_width]
                                    
                                    if spot_img.size > 0:
                                        pred = st.session_state.predictor.predict_single_spot(spot_img)
                                        results.append({
                                            "id": f"R{r+1}C{c+1}",
                                            "prediction": pred["prediction"],
                                            "confidence": pred["confidence"]
                                        })
                                    
                                    progress = ((r * cols_num) + c + 1) / (rows * cols_num)
                                    progress_bar.progress(progress)
                            
                            # Display results
                            if results:
                                occupied = sum(1 for r in results if r["prediction"] == "occupied")
                                total = len(results)
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Spots", total)
                                col2.metric("Occupied", occupied)
                                col3.metric("Available", total - occupied)
                                
                                # Create overlay
                                overlay = lot_image.copy()
                                for r in range(rows):
                                    for c in range(cols_num):
                                        x, y = c * spot_width, r * spot_height
                                        result = results[r * cols_num + c]
                                        
                                        color = (0, 0, 255) if result["prediction"] == "occupied" else (0, 255, 0)
                                        cv2.rectangle(overlay, (x, y), (x+spot_width, y+spot_height), color, 2)
                                        cv2.putText(overlay, result["id"], (x+5, y+20), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                                        caption="Analysis Results", use_container_width=True)

# Run app
if __name__ == "__main__":
    main()
