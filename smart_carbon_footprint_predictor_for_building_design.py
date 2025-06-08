import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import streamlit as st
import os

# -----------------------------
# 1. Train model if not exists
# -----------------------------

MODEL_PATH = 'carbon_model.pkl'
FEATURES_PATH = 'feature_order.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    np.random.seed(42)
    data = {
        'floor_area': np.random.uniform(50, 5000, 1000),
        'num_floors': np.random.randint(1, 20, 1000),
        'concrete_volume': np.random.uniform(10, 1000, 1000),
        'steel_mass': np.random.uniform(100, 5000, 1000),
        'climate_zone': np.random.choice(['Tropical', 'Temperate', 'Continental'], 1000),
        'transport_radius': np.random.uniform(10, 500, 1000),
        'co2e': np.random.uniform(5000, 500000, 1000)
    }

    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['climate_zone'])

    X = df.drop('co2e', axis=1)
    y = df['co2e']

    feature_order = list(X.columns)
    joblib.dump(feature_order, FEATURES_PATH)

    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

# -----------------------------
# 2. Load model & feature order
# -----------------------------

model = joblib.load(MODEL_PATH)
feature_order = joblib.load(FEATURES_PATH)

# -----------------------------
# 3. Streamlit App
# -----------------------------

st.set_page_config(page_title="Carbon Footprint Predictor", page_icon="🏗️")
st.title("🏗️ Smart Carbon Footprint Predictor")

with st.form("input_form"):
    floor_area = st.number_input("Floor Area (m²)", min_value=50.0, max_value=5000.0, value=1000.0)
    num_floors = st.number_input("Number of Floors", min_value=1, max_value=50, value=5)
    concrete = st.number_input("Concrete Volume (m³)", min_value=0.0, max_value=1000.0, value=500.0)
    steel = st.number_input("Steel Mass (kg)", min_value=0.0, max_value=5000.0, value=2000.0)
    climate = st.selectbox("Climate Zone", ["Tropical", "Temperate", "Continental"])
    transport = st.slider("Transport Radius (km)", min_value=10, max_value=500, value=100)

    if st.form_submit_button("Predict"):
        input_dict = {
            'floor_area': floor_area,
            'num_floors': num_floors,
            'concrete_volume': concrete,
            'steel_mass': steel,
            'transport_radius': transport,
            'climate_zone_Continental': 1 if climate == 'Continental' else 0,
            'climate_zone_Temperate': 1 if climate == 'Temperate' else 0,
            'climate_zone_Tropical': 1 if climate == 'Tropical' else 0
        }

        input_df = pd.DataFrame([input_dict])[feature_order]
        prediction = model.predict(input_df)[0]
        st.metric("Predicted CO₂e", f"{prediction:,.0f} kg")
