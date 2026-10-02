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
    # 1. Create input dictionary
    input_data = {
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
    }
    
    input_df = pd.DataFrame([input_data])
    
    # 2. One-Hot Encode categorical variables matching training categories
    categorical_cols = {
        'lga': lga_list,
        'geopolitical_zone': zone_list,
        'title_type': title_list,
        'land_use': use_list
    }
    
    encoded_features = []
    for col, categories in categorical_cols.items():
        for category in categories:
            col_name = f"cat__{col}_{category}"
            input_df[col_name] = 1.0 if input_data[col] == category else 0.0
            encoded_features.append(col_name)
            
    # Drop raw string columns
    input_df = input_df.drop(columns=['lga', 'geopolitical_zone', 'title_type', 'land_use'])
    
    # Reorder columns exactly as XGBoost expects them
    numerical_cols = ['land_size_sqm', 'dist_main_road_km', 'electricity_access', 'water_access', 'latitude', 'longitude']
    final_df = input_df[encoded_features + numerical_cols]
    
    # 3. Predict directly with XGBoost
    prediction = model.predict(final_df)[0]
    st.success(f"### Estimated Value: **₦{prediction:,.2f}**")
