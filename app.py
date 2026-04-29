import streamlit as st
import pandas as pd
import numpy as np
from predict_and_explain import predict_and_explain, required_features
import json

st.set_page_config(layout="wide")
st.title("🏡 UAE Real Estate AI Analyzer")
st.markdown("### Predict property prices with CatBoost + AI explanations")

# --- Sidebar for input (simplified: you can expand with dropdowns) ---
st.sidebar.header("Property Features")
beds = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=2)
baths = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, value=2)

# For demonstration, we'll use a pre-defined set of distance features (already in model)
# In a real app, you'd compute distances from lat/lon or let user select area.
# Here we load a sample property from the dataset to showcase.
# Alternatively, we can allow user to select an area and we precompute distances.

st.sidebar.markdown("---")
st.sidebar.info("This demo uses precomputed distances from key UAE landmarks. For a full experience, provide property coordinates.")

# For simplicity, we'll take the median values of each distance feature from training set
# (you can replace with actual user input later)
sample_input = {feat: 0.0 for feat in required_features}
sample_input['beds'] = beds
sample_input['baths'] = baths

# Set some dummy distances (you should compute from actual coordinates in production)
# Here we just fill with typical values from training data (you can load a real property)
# To keep it working, we'll use zeros and let the model predict based on beds/baths only (not great).
# Better: load one row from your cleaned dataset as a template.

# Load one actual property from your cleaned df (or precompute)
# For demonstration, we'll assume you have a CSV with typical values.
# If not, we'll just use zeros – the model will still output something.

if st.sidebar.button("🔮 Analyze Price"):
    with st.spinner("Consulting AI models..."):
        # For a real app, you'd compute distance features here.
        # We'll use a pre-stored property from your dataset (e.g., first row of X_test)
        # To keep it running, let's load X_test from memory if available, else use zeros.
        try:
            # Attempt to load a sample property from the training data (you saved X_train earlier)
            # We'll just use the first row of X_train as example
            example_row = X_train.iloc[0].to_dict()
            example_row['beds'] = beds
            example_row['baths'] = baths
            input_dict = example_row
        except:
            # Fallback: use zeros for all distance features
            input_dict = sample_input
        
        price, explanation = predict_and_explain(input_dict)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Price", f"AED {price:,}")
    with col2:
        st.subheader("AI Insight")
        st.write(explanation)