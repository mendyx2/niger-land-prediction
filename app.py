import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

st.set_page_config(page_title="Niger State Land Valuation", layout="centered")

@st.cache_resource
def load_assets():
    preprocessor = joblib.load('preprocessor.joblib')
    model = XGBRegressor()
    model.load_model('model.json')
    return preprocessor, model

preprocessor, model = load_assets()

st.title("🏡 Niger State Land Price Predictor")
st.write("An AI-powered valuation system for the 25 LGAs of Niger State.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    lga = st.selectbox("Local Government Area", options=[
        'Agaie', 'Agwara', 'Bida', 'Borgu', 'Bosso', 'Chanchaga', 'Edati', 'Gbako', 
        'Gurara', 'Katcha', 'Kontagora', 'Lapai', 'Lavun', 'Magama', 'Mariga', 
        'Mashegu', 'Mokwa', 'Munya', 'Paikoro', 'Rafi', 'Rijau', 'Shiroro', 'Suleja', 'Tafa', 'Wushishi'
    ])
    geopolitical_zone = st.selectbox("Geopolitical Zone", options=['Zone A', 'Zone B', 'Zone C'])
    land_size_sqm = st.number_input("Land Size (sqm)", min_value=100.0, max_value=50000.0, value=600.0)
    title_type = st.selectbox("Title Status", options=['C of O', 'Governor Consent', 'Gazette', 'Excision', 'Customary', 'Right of Occupancy'])

with col2:
    land_use = st.selectbox("Land Use Category", options=['Residential', 'Commercial', 'Agricultural', 'Industrial'])
    dist_main_road = st.number_input("Distance to Main Road (km)", min_value=0.0, max_value=50.0, value=0.5)
    electricity = st.radio("Electricity Access", options=["Yes", "No"])
    water = st.radio("Pipe-borne Water Access", options=["Yes", "No"])

if st.button("Predict Land Valuation", use_container_width=True):
    input_df = pd.DataFrame([{
        'lga': lga,
        'geopolitical_zone': geopolitical_zone,
        'land_size_sqm': land_size_sqm,
        'title_type': title_type,
        'land_use': land_use,
        'dist_main_road_km': dist_main_road,
        'electricity_access': 1 if electricity == "Yes" else 0,
        'water_access': 1 if water == "Yes" else 0,
        'latitude': 9.6139,
        'longitude': 6.5569
    }])
    
    transformed_input = preprocessor.transform(input_df)
    prediction = model.predict(transformed_input)[0]
    st.success(f"### Estimated Value: **₦{prediction:,.2f}**")
