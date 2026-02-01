# test_app.py - Minimal version
import streamlit as st
import joblib
import tempfile
import os

st.title("Test Model Loading")

uploaded_model = st.file_uploader("Upload model", type=['pkl'])

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        f.write(uploaded_model.read())
        model_path = f.name
    
    try:
        # Try to load and inspect
        model = joblib.load(model_path)
        st.success("Model loaded!")
        
        st.write("**Model keys:**", list(model.keys()))
        
        if 'svm_model' in model:
            st.write(f"**svm_model type:** {type(model['svm_model'])}")
            
            # Check attributes
            st.write("**svm_model attributes:**")
            for attr in dir(model['svm_model']):
                if not attr.startswith('_'):
                    st.write(f"- {attr}")
    
    except Exception as e:
        st.error(f"Error: {e}")
    
    finally:
        os.unlink(model_path)
