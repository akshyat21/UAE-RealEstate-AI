# app.py
import streamlit as st
from predict_and_explain import predict_from_input

st.set_page_config(layout="wide")
st.title("🏡 UAE Real Estate AI Analyzer")
st.markdown("### Enter property details to get an AI‑powered price estimate and explanation")

# ---------- Sidebar or main area for inputs ----------
col1, col2 = st.columns(2)

with col1:
    beds = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2, step=1)
    baths = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Townhouse", "Penthouse"])
    
with col2:
    area_name = st.selectbox("Area (Neighbourhood)", [
        "Jumeirah Village Circle (JVC)", "Dubai Marina", "Downtown Dubai", 
        "Palm Jumeirah", "Business Bay", "Other"
    ])
    furnishing = st.selectbox("Furnishing Status", ["Unfurnished", "Furnished"])
    completion = st.selectbox("Completion Status", ["Ready", "Off-Plan"])

if st.button("🔮 Predict Price", type="primary"):
    with st.spinner("Consulting CatBoost + AI engine..."):
        price, explanation = predict_from_input(beds, baths, area_name, property_type, furnishing, completion)
    
    st.success("Prediction complete!")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Predicted Price", f"AED {price:,}")
    with col2:
        st.subheader("AI Insight")
        st.write(explanation)