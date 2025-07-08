import streamlit as st
import pandas as pd
import pickle
import cloudpickle
import joblib

# Load model and pipeline assets
model = joblib.load("xgb_model.pkl")

with open("feature_columns.pkl", "rb") as f:
    feature_names = pickle.load(f)

# FIX: Use cloudpickle to avoid sklearn internal errors
with open("preprocessor.joblib", "rb") as f:
    preprocessor = cloudpickle.load(f)

# Streamlit Page Config
st.set_page_config(page_title="⚡ Electricity Demand Predictor", layout="centered")
st.title("⚡ Predict Shifted Demand-Supply")

input_data = {}

# --- UI: Time Features ---
with st.expander("🗓️ Time Information"):
    input_data["month"] = st.number_input("Month", 1, 12, step=1)
    input_data["week"] = st.number_input("Week", 1, 52, step=1)
    input_data["day"] = st.number_input("Day", 1, 31, step=1)
    input_data["hour"] = st.number_input("Hour", 0, 23, step=1)

# --- UI: Weather Features ---
with st.expander("🌦️ Weather & Conditions"):
    input_data["temp_min"] = st.number_input("Min Temp (°C)", value=20.0)
    input_data["temp_max"] = st.number_input("Max Temp (°C)", value=30.0)
    input_data["pressure"] = st.number_input("Pressure (hPa)", value=1012.0)
    input_data["humidity"] = st.number_input("Humidity (%)", value=60.0)
    input_data["wind_speed"] = st.number_input("Wind Speed (m/s)", value=4.0)
    input_data["clouds_all"] = st.number_input("Cloud Cover (%)", value=30.0)
    input_data["weather_main"] = st.selectbox("Weather Condition", ["Clear", "Clouds", "Rain", "Fog", "Snow"])

# --- UI: Market Features ---
with st.expander("💰 Market Data"):
    input_data["price actual"] = st.number_input("Actual Price", value=55.0)
    input_data["total load actual"] = st.number_input("Total Load Actual (MW)", value=50000.0)


# --- UI: Energy Generation ---
with st.expander("⚡ Energy Generation (MW)"):
    generation_features = [
        "generation biomass", "generation fossil brown coal/lignite", "generation fossil gas",
        "generation fossil hard coal", "generation fossil oil",
        "generation hydro pumped storage consumption",
        "generation hydro run-of-river and poundage", "generation hydro water reservoir",
        "generation nuclear", "generation other", "generation other renewable",
        "generation solar", "generation waste", "generation wind"
    ]
    for feature in generation_features:
        input_data[feature] = st.number_input(feature.replace("generation ", "").title(), value=100.0, key=feature)

# --- Prediction Button ---
if st.button("🔍 Predict Shifted Demand-Supply"):
    try:
        # Build DataFrame only with required feature columns
        input_df = pd.DataFrame([input_data])

        # Ensure column alignment with trained model
        input_df = input_df[feature_names]

        # Preprocess
        X_processed = preprocessor.transform(input_df)
        if hasattr(X_processed, "toarray"):  # For sparse matrices
            X_processed = X_processed.toarray()

        # Predict
        prediction = model.predict(X_processed)[0]
        st.success(f"📈 Predicted Shifted Demand-Supply: **{prediction:,.2f} MW**")

    except Exception as e:
        st.error(f"🚫 Something went wrong: {e}")
