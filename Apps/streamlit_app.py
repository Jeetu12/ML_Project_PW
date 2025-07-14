import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Crypto Liquidity Predictor (Colab)", layout="centered")

st.title("💧 Crypto Liquidity Predictor (Colab)")
st.markdown("Predict the **liquidity ratio** of a cryptocurrency using a trained ML model.")

# Load model
model_path = "crypto_liquidity_model.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please train and save the model first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)


# Input Fields
price = st.number_input("Current Price (USD)", value=100.0)
pct_1h = st.number_input("1h Change (%)", value=0.5)
pct_24h = st.number_input("24h Change (%)", value=1.2)
pct_7d = st.number_input("7d Change (%)", value=-3.5)
volume = st.number_input("24h Volume", value=5_000_000.0)
mkt_cap = st.number_input("Market Cap", value=100_000_000.0)
volatility = abs(pct_24h)

# Prediction
if st.button("Predict Liquidity Ratio"):
    input_data = np.array([[price, pct_1h, pct_24h, pct_7d, volume, mkt_cap, volatility]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔍 Predicted Liquidity Ratio: {round(prediction, 6)}")

    st.markdown("### 📌 Inputs")
    st.json({
        "Price": price,
        "1h Change": pct_1h,
        "24h Change": pct_24h,
        "7d Change": pct_7d,
        "Volume": volume,
        "Market Cap": mkt_cap,
        "Volatility": volatility
    })
