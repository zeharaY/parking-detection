"""
🚗 Parking Spot Detection App
Fixed for Streamlit Cloud
"""
import subprocess
import sys

# Try to import imblearn, if it fails, install it
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE
    
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ============================
# BACKGROUND SUBTRACTOR CLASS (MUST MATCH TRAINING)
# ============================
class BackgroundSubtractor:
    """Custom background subtraction for parking spot analysis"""
    
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
            for i in range(3):
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
# BACKGROUND SUBTRACTION SVM CLASS (MUST MATCH TRAINING)
# ============================
class BackgroundSubtractionSVM:
    """
    SVM classifier with background subtraction features
    THIS CLASS MUST EXACTLY MATCH THE ONE IN TRAINING
    """
    
    def __init__(self, use_smote=True):
        self.use_smote = use_smote
        self.scaler = StandardScaler()
        self.svm = None
        self.bg_subtractor = BackgroundSubtractor(method='mog2')
    
    def create_pipeline(self):
        """Create ML pipeline with optional SMOTE"""
        steps = [
            ('scaler', self.scaler),
        ]
        
        if self.use_smote:
            steps.append(('smote', SMOTE(random_state=42)))
        
        steps.append(('svm', SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )))
        
        return ImbPipeline(steps) if self.use_smote else Pipeline(steps)
    
    def train(self, X_features, y, param_grid=None):
        """Train SVM with grid search"""
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            }
        
        # Create pipeline
        pipeline = self.create_pipeline()
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_features, y)
        self.svm = grid_search.best_estimator_
        
        return grid_search.best_score_
    
    def predict(self, X_features):
        """Make predictions"""
        if self.svm is None:
            raise ValueError("Model not trained yet")
        return self.svm.predict(X_features)
    
    def predict_proba(self, X_features):
        """Get prediction probabilities"""
        if self.svm is None:
            raise ValueError("Model not trained yet")
        return self.svm.predict_proba(X_features)
    
    def evaluate(self, X_test_features, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, y_pred

# ============================
# PARKING SPOT PREDICTOR CLASS
# ============================
class ParkingSpotPredictor:
    """Main class for making predictions"""
    
    def __init__(self, model_data):
        """Initialize predictor with trained model"""
        # The model_data contains a BackgroundSubtractionSVM instance
        self.model_wrapper = model_data['svm_model']  # This is the BackgroundSubtractionSVM object
        self.bg_subtractor = model_data['bg_subtractor']
        self.feature_names = model_data.get('feature_names', [])
        self.model_accuracy = model_data.get('accuracy', 0.0)
        
        # Get the actual sklearn model from the wrapper
        if hasattr(self.model_wrapper, 'svm') and self.model_wrapper.svm is not None:
            self.sklearn_model = self.model_wrapper.svm
        else:
            # Fallback: use the wrapper itself
            self.sklearn_model = self.model_wrapper
        
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
        try:
            # Try to use the wrapper's predict_proba
            if hasattr(self.model_wrapper, 'predict_proba'):
                proba = self.model_wrapper.predict_proba(features_reshaped)[0]
            # Try the sklearn model
            elif hasattr(self.sklearn_model, 'predict_proba'):
                proba = self.sklearn_model.predict_proba(features_reshaped)[0]
            else:
                # Fallback
                prediction = self.model_wrapper.predict(features_reshaped)[0]
                proba = [1 - prediction, prediction]
            
            confidence = max(proba)
            prediction = 1 if proba[1] >= self.current_threshold else 0
            
        except Exception as e:
            # Ultimate fallback
            st.warning(f"Using fallback prediction: {str(e)}")
            prediction = 1 if features[0] > 0.1 else 0  # Simple heuristic
            proba = [1 - prediction, prediction]
            confidence = 0.7
        
        # Get feature values for explanation
        feature_values = {}
        if self.feature_names and len(self.feature_names) == len(features):
            for i, (name, value) in enumerate(zip(self.feature_names, features)):
                feature_values[name] = float(value)
        else:
            # Create generic feature names
            for i, value in enumerate(features):
                feature_values[f"feature_{i}"] = float(value)
        
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
                        st.write(f"**Feature Count:** {len(st.session_state.predictor.feature_names)}")
                        
                        # Show model structure
                        st.write("**Model Structure:**")
                        st.write(f"- Model Wrapper: {type(st.session_state.predictor.model_wrapper).__name__}")
                        st.write(f"- SVM Model: {type(st.session_state.predictor.sklearn_model).__name__}")
                        st.write(f"- Background Subtractor: Loaded")
                
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
                index=0
            )
            
            # Threshold adjustment
            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.predictor.current_threshold,
                step=0.01
            )
            
            if st.button("Apply Settings"):
                st.session_state.predictor.set_lighting_mode(lighting_mode)
                st.session_state.predictor.adjust_threshold(threshold)
                st.success("Settings applied!")
        
        st.divider()
        
        # Quick Demo
        st.subheader("🖼️ Quick Demo")
        
        if st.button("Generate Demo Images"):
            # Create demo images
            demo_empty = np.ones((100, 100, 3), dtype=np.uint8) * 150
            demo_occupied = np.ones((100, 100, 3), dtype=np.uint8) * 150
            cv2.rectangle(demo_occupied, (20, 20), (80, 80), (0, 0, 200), -1)
            
            st.session_state.demo_images = {
                "empty": demo_empty,
                "occupied": demo_occupied
            }
            st.success("Demo images generated!")
        
        if 'demo_images' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Use Empty Spot"):
                    st.session_state.demo_image = st.session_state.demo_images["empty"]
                    st.session_state.demo_label = "empty"
                    st.rerun()
            
            with col2:
                if st.button("Use Occupied Spot"):
                    st.session_state.demo_image = st.session_state.demo_images["occupied"]
                    st.session_state.demo_label = "occupied"
                    st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Parking Spot Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a trained model in the sidebar first.")
            
            # Show what the app expects
            with st.expander("What model file do I need?"):
                st.write("""
                The app expects a `.pkl` file containing:
                - A trained `BackgroundSubtractionSVM` model
                - A `BackgroundSubtractor` object
                - Model accuracy
                - Feature names
                
                **To create this file:**
                1. Run your training code
                2. It will save: `parking_detection_model.pkl`
                3. Upload that file here
                """)
        else:
            # Image upload
            uploaded_image = st.file_uploader(
                "Choose a parking spot image",
                type=['jpg', 'jpeg', 'png']
            )
            
            # Use demo image if available
            image_to_predict = None
            if 'demo_image' in st.session_state:
                st.info(f"Using demo image ({st.session_state.demo_label})")
                image_to_predict = st.session_state.demo_image.copy()
                
                # Display demo image
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB),
                            caption="Demo Image", use_container_width=True)
                
                with col_img2:
                    if st.button("Clear Demo", type="secondary"):
                        del st.session_state.demo_image
                        del st.session_state.demo_label
                        st.rerun()
            
            elif uploaded_image:
                # Convert uploaded image
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image_to_predict = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image_to_predict is not None:
                    st.image(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB),
                            caption="Uploaded Image", use_container_width=True)
            
            if image_to_predict is not None:
                if st.button("🔮 Predict Occupancy", type="primary", use_container_width=True):
                    with st.spinner("Analyzing parking spot..."):
                        result = st.session_state.predictor.predict_single_spot(image_to_predict)
                        st.session_state.last_prediction = result
    
    with col2:
        st.header("Prediction Results")
        
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
            
            # Feature extraction visualization
            with st.expander("🔬 Feature Extraction", expanded=False):
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.image(cv2.cvtColor(result["processed_image"], cv2.COLOR_BGR2RGB),
                            caption="Preprocessed", use_container_width=True)
                with col_v2:
                    st.image(result["foreground_mask"], 
                            caption="Foreground Mask", use_container_width=True, cmap="gray")
                with col_v3:
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
                    fig = px.bar(
                        features_df,
                        x="Feature",
                        y="Value",
                        title="Extracted Feature Values",
                        color="Value",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.model_loaded:
            st.info("👈 Upload an image and click 'Predict Occupancy' to see results here")
    
    # Additional sections
    if st.session_state.model_loaded:
        st.divider()
        
        # Feature importance
        with st.expander("📈 Model Insights", expanded=False):
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.metric("Model Accuracy", f"{st.session_state.predictor.model_accuracy:.2%}")
                st.metric("Current Threshold", f"{st.session_state.predictor.current_threshold:.2f}")
                st.metric("Lighting Mode", st.session_state.predictor.current_lighting_mode)
            
            with col_insight2:
                # Simulated feature importance
                if st.session_state.predictor.feature_names:
                    importance = np.random.rand(len(st.session_state.predictor.feature_names))
                    importance = importance / importance.sum()
                    
                    importance_df = pd.DataFrame({
                        "Feature": st.session_state.predictor.feature_names[:10],
                        "Importance": importance[:10]
                    })
                    
                    fig = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation='h',
                        title="Top 10 Features (Simulated)",
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Batch processing
        with st.expander("📁 Batch Processing", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            
            if uploaded_files and st.button("Process All"):
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    results = []
                    
                    for uploaded_file in uploaded_files:
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            result = st.session_state.predictor.predict_single_spot(image)
                            result["filename"] = uploaded_file.name
                            results.append(result)
                    
                    if results:
                        # Summary
                        occupied = sum(1 for r in results if r["prediction"] == "occupied")
                        total = len(results)
                        
                        st.metric("Total Images", total)
                        st.metric("Occupied Spots", occupied)
                        st.metric("Available Spots", total - occupied)
                        
                        # Results table
                        display_data = []
                        for result in results:
                            display_data.append({
                                "Filename": result["filename"],
                                "Prediction": result["prediction"].upper(),
                                "Confidence": f"{result['confidence']:.1%}"
                            })
                        
                        results_df = pd.DataFrame(display_data)
                        st.dataframe(results_df, use_container_width=True)

# ============================
# RUN THE APP
# ============================
if __name__ == "__main__":
    main()
