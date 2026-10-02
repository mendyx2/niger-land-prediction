import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

st.set_page_config(page_title="Niger State Land Valuation", layout="centered")

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model('model.json')
    return model

model = load_model()

st.title("🏡 Niger State Land Price Predictor")
st.write("An AI-powered valuation system for the 25 LGAs of Niger State.")

st.markdown("---")

col1, col2 = st.columns(2)

lga_list = [
    'Agaie', 'Agwara', 'Bida', 'Borgu', 'Bosso', 'Chanchaga', 'Edati', 'Gbako', 
    'Gurara', 'Katcha', 'Kontagora', 'Lapai', 'Lavun', 'Magama', 'Mariga', 
    'Mashegu', 'Mokwa', 'Munya', 'Paikoro', 'Rafi', 'Rijau', 'Shiroro', 'Suleja', 'Tafa', 'Wushishi'
]

zone_list = ['Zone A', 'Zone B', 'Zone C']
title_list = ['C of O', 'Governor Consent', 'Gazette', 'Excision', 'Customary', 'Right of Occupancy']
use_list = ['Residential', 'Commercial', 'Agricultural', 'Industrial']

with col1:
    lga = st.selectbox("Local Government Area", options=lga_list)
    geopolitical_zone = st.selectbox("Geopolitical Zone", options=zone_list)
    land_size_sqm = st.number_input("Land Size (sqm)", min_value=100.0, max_value=50000.0, value=600.0)
    title_type = st.selectbox("Title Status", options=title_list)

with col2:
    land_use = st.selectbox("Land Use Category", options=use_list)
    dist_main_road = st.number_input("Distance to Main Road (km)", min_value=0.0, max_value=50.0, value=0.5)
    electricity = st.radio("Electricity Access", options=["Yes", "No"])
    water = st.radio("Pipe-borne Water Access", options=["Yes", "No"])

if st.button("Predict Land Valuation", use_container_width=True):
    # 1. Ask the model exactly what features it was trained on
    expected_features = model.get_booster().feature_names
    
    if expected_features is None or expected_features[0] == 'f0':
        st.error("Model format error: Please ensure you uploaded the model.json generated using pandas get_dummies.")
    else:
        # 2. Create an empty dictionary filled with 0s matching the model's exact expectations
        feature_dict = {feat: 0.0 for feat in expected_features}
        
        # 3. Intelligently map the user inputs into the correct expected columns
        for feat in expected_features:
            # Map numerical variables
            if 'land_size' in feat: 
                feature_dict[feat] = float(land_size_sqm)
            elif 'dist_main_road' in feat: 
                feature_dict[feat] = float(dist_main_road)
            elif 'electricity' in feat: 
                feature_dict[feat] = 1.0 if electricity == "Yes" else 0.0
            elif 'water' in feat: 
                feature_dict[feat] = 1.0 if water == "Yes" else 0.0
            elif 'latitude' in feat: 
                feature_dict[feat] = 9.6139
            elif 'longitude' in feat: 
                feature_dict[feat] = 6.5569
            # Map categorical variables (if the selected option name is anywhere in the column name, turn it to 1)
            elif str(lga) in feat or str(geopolitical_zone) in feat or str(title_type) in feat or str(land_use) in feat:
                feature_dict[feat] = 1.0
                
        # 4. Convert perfectly aligned features into a DataFrame and Predict
        final_df = pd.DataFrame([feature_dict])
        prediction = model.predict(final_df)[0]
        
        st.success(f"### Estimated Value: **₦{prediction:,.2f}**")
