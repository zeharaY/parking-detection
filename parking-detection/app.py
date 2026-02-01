"""
🚗 Parking Spot Detection App
Simplified for Streamlit Cloud deployment
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

# ============================
# CUSTOM BACKGROUND SUBTRACTOR CLASS
# ============================
class BackgroundSubtractor:
    """
    Custom background subtraction for parking spot analysis
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
        self.model = model_data['svm_model']
        self.bg_subtractor = model_data['bg_subtractor']
        self.feature_names = model_data.get('feature_names', [])
        self.model_accuracy = model_data.get('accuracy', 0.0)
        
        self.lighting_profiles = {
            "daylight": {"threshold": 0.5},
            "dusk": {"threshold": 0.45},
            "night": {"threshold": 0.4},
            "overcast": {"threshold": 0.48},
            "bright_sun": {"threshold": 0.52}
        }
        
        self.current_threshold = 0.5
        self.current_lighting_mode = "daylight"
    
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
        return resized
    
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
        
        # Convert mask for display
        if mask.dtype != np.uint8:
            mask_display = (mask * 255).astype(np.uint8)
        else:
            mask_display = mask
        
        return {
            "prediction": "occupied" if prediction == 1 else "available",
            "confidence": float(confidence),
            "probability_occupied": float(proba[1]),
            "probability_available": float(proba[0]),
            "threshold_used": float(self.current_threshold),
            "lighting_mode": self.current_lighting_mode,
            "features": dict(zip(self.feature_names, features.tolist())) if self.feature_names else {},
            "processed_image": processed_image,
            "foreground_mask": mask_display,
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
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
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
                    
                    # Load model
                    model_data = joblib.load(tmp_path)
                    
                    # Initialize predictor
                    st.session_state.predictor = ParkingSpotPredictor(model_data)
                    st.session_state.model_loaded = True
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    st.success(f"✅ Model loaded (Accuracy: {st.session_state.predictor.model_accuracy:.2%})")
                    
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
            
            if st.button("Apply Settings", use_container_width=True):
                st.session_state.predictor.set_lighting_mode(lighting_mode)
                st.session_state.predictor.adjust_threshold(threshold)
                st.success("Settings applied!")
        
        st.divider()
        
        # Quick Demo
        st.subheader("🖼️ Quick Demo")
        
        demo_option = st.radio(
            "Try a demo image:",
            ["Upload your own", "Sample Empty Spot", "Sample Occupied Spot"],
            index=0
        )
        
        if demo_option == "Sample Empty Spot":
            demo_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
            st.session_state.demo_image = demo_image
            st.info("Demo empty spot loaded.")
        
        elif demo_option == "Sample Occupied Spot":
            demo_image = np.ones((100, 100, 3), dtype=np.uint8) * 150
            cv2.rectangle(demo_image, (20, 20), (80, 80), (0, 0, 200), -1)
            st.session_state.demo_image = demo_image
            st.info("Demo occupied spot loaded.")
        
        st.divider()
        
        # Deployment info
        st.subheader("🌐 Deployment Info")
        st.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content tabs
    tab1, tab2 = st.tabs([
        "🔍 Single Spot Prediction", 
        "📊 Parking Lot Analysis"
    ])
    
    # Tab 1: Single Spot Prediction
    with tab1:
        st.header("Single Spot Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please upload a trained model in the sidebar first.")
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
                    st.info("Using demo image")
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
                
                if st.session_state.last_prediction:
                    result = st.session_state.last_prediction
                    
                    # Display prediction
                    if result["prediction"] == "occupied":
                        st.markdown(f"""
                        <div class="prediction-box occupied">
                            <h2 style="color: #EF4444; text-align: center;">🚗 OCCUPIED</h2>
                            <div style="text-align: center; font-size: 20px; margin: 15px 0;">
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
                            <div style="text-align: center; font-size: 20px; margin: 15px 0;">
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
                                    caption="Preprocessed", use_container_width=True)
                        with col_v2:
                            st.image(result["foreground_mask"], 
                                    caption="Foreground Mask", use_container_width=True)
                        with col_v3:
                            st.image(cv2.cvtColor(result["foreground_image"], cv2.COLOR_BGR2RGB),
                                    caption="Foreground", use_container_width=True)
                else:
                    st.info("👈 Upload an image and click 'Predict Occupancy' to see results")
    
    # Tab 2: Parking Lot Analysis - SIMPLIFIED
    with tab2:
        st.header("Parking Lot Analysis")
        
        if not st.session_state.model_loaded:
            st.warning("Please upload a model first")
        else:
            st.info("Upload a full parking lot image to analyze multiple spots")
            
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
                        rows = st.number_input("Number of Rows", 1, 10, 3, key="rows_input")
                    with col_config2:
                        cols = st.number_input("Spots per Row", 1, 20, 5, key="cols_input")
                    
                    if st.button("🔍 Analyze Parking Lot", type="primary", use_container_width=True):
                        # Create spot grid
                        spot_width = lot_image.shape[1] // cols
                        spot_height = lot_image.shape[0] // rows
                        
                        # Analyze each spot with progress
                        results = []
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        for r in range(rows):
                            for c in range(cols):
                                x, y = c * spot_width, r * spot_height
                                spot_img = lot_image[y:y+spot_height, x:x+spot_width]
                                
                                if spot_img.size > 0:
                                    try:
                                        prediction = st.session_state.predictor.predict_single_spot(spot_img)
                                        results.append({
                                            "id": f"R{r+1}C{c+1}",
                                            "position": f"Row {r+1}, Col {c+1}",
                                            "prediction": prediction["prediction"],
                                            "confidence": prediction["confidence"],
                                            "occupied_prob": prediction["probability_occupied"]
                                        })
                                    except Exception as e:
                                        # Skip spots that cause errors
                                        continue
                                
                                # Update progress
                                current_progress = ((r * cols) + c + 1) / (rows * cols)
                                progress_bar.progress(current_progress)
                                progress_text.text(f"Analyzing spot {r+1}-{c+1} of {rows}x{cols}")
                        
                        progress_text.text("Analysis complete!")
                        
                        # Display results
                        if results:
                            results_df = pd.DataFrame(results)
                            
                            # Summary metrics
                            occupied = sum(1 for r in results if r["prediction"] == "occupied")
                            total = len(results)
                            utilization = occupied / total if total > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Spots", total)
                            with col2:
                                st.metric("Occupied", occupied)
                            with col3:
                                st.metric("Available", total - occupied)
                            
                            # Visualize on image
                            overlay = lot_image.copy()
                            for result in results:
                                # Parse row and column from id
                                parts = result["id"][1:].split('C')
                                r = int(parts[0]) - 1
                                c = int(parts[1]) - 1
                                
                                x, y = c * spot_width, r * spot_height
                                
                                if result["prediction"] == "occupied":
                                    color = (0, 0, 255)  # Red
                                else:
                                    color = (0, 255, 0)  # Green
                                
                                cv2.rectangle(overlay, (x, y), (x+spot_width, y+spot_height), color, 2)
                                cv2.putText(overlay, result["id"], (x+5, y+20), 
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
                        else:
                            st.error("No valid spots were analyzed. Try a different image or grid configuration.")

# ============================
# RUN THE APP
# ============================
if __name__ == "__main__":
    main()
