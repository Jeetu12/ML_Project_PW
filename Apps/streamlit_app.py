!pip install streamlit
import streamlit as st
import requests
import pickle
import numpy as np
import os # Import os

import warnings
warnings.filterwarnings("ignore")

# Load model
# Ensure the model file exists in the correct path
model_path = "models/crypto_liquidity_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_path}. Please ensure the model training and saving steps were completed successfully.")
    st.stop() # Stop the app if the model is not found

# CoinGecko API URL (example, actual API integration might be different)
# For this example, we will simulate prediction using the loaded model
# API_URL = "https://api.coingecko.com/api/v3/coins/markets"

st.set_page_config(page_title="Crypto Liquidity Predictor")
st.title("💧 Crypto Liquidity Ratio Predictor")

st.markdown("Enter crypto market values:")

# User input
price = st.number_input("Price (USD)", value=40000.0)
h1 = st.number_input("1h % Change", value=0.0)
h24 = st.number_input("24h % Change", value=0.0)
d7 = st.number_input("7d % Change", value=0.0)
volume = st.number_input("24h Volume (USD)", value=1e9)
market_cap = st.number_input("Market Cap (USD)", value=1e10)

if st.button("Predict Liquidity Ratio"):
    volatility = abs(h24)
    # The model expects a 2D array: [[price, h1, h24, d7, volume, market_cap, volatility]]
    input_data = np.array([[price, h1, h24, d7, volume, market_cap, volatility]])

    # Make prediction using the loaded model
    try:
        pred = model.predict(input_data)[0]
        st.success(f"Predicted Liquidity Ratio: {pred:.6f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


st.markdown("---")
st.write("This app uses a trained machine learning model to predict the liquidity ratio based on user-provided market data.")
