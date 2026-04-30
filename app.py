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

# Property summary preview
st.markdown("---")
st.subheader("📋 Property Summary")
st.info(f"""
**Location:** {area_name}  
**Type:** {property_type}  
**Bedrooms:** {beds}  |  **Bathrooms:** {baths}  
**Furnishing:** {furnishing}  |  **Completion:** {completion}
""")
st.markdown("---")

with st.expander("📘 How this works (click to expand)"):
    st.markdown("""
    - **Price prediction** uses a **CatBoost model** trained on 41,000+ real UAE property listings.
    - **40+ location features** include distance to Burj Khalifa, Dubai Mall, metro stations, beaches, and airports (computed via Haversine formula).
    - **AI explanation** powered by **Groq's Llama 3.3** – generates a plain‑English reason for the price based on location prestige and property attributes.
    - **Model performance:** R² = 0.73, RMSE = 2.46M AED.
    """)

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