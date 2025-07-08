import streamlit as st
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

st.title("💧 Crypto Liquidity Predictor")
st.markdown("Predict the **liquidity ratio** of a cryptocurrency based on live financial indicators.")

# Load model
model_path = "crypto_liquidity_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file `{model_path}` not found. Please train and save the model first.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Sidebar inputs
st.sidebar.header("📊 Input Parameters")
price = st.sidebar.number_input("Current Price (USD)", value=100.0)
change_1h = st.sidebar.number_input("1h % Change", value=0.5)
change_24h = st.sidebar.number_input("24h % Change", value=1.2)
change_7d = st.sidebar.number_input("7d % Change", value=-3.5)
volume_24h = st.sidebar.number_input("24h Trading Volume", value=5_000_000.0)
market_cap = st.sidebar.number_input("Market Cap", value=100_000_000.0)

volatility = abs(change_24h)

# Input vector
input_data = np.array([[price, change_1h, change_24h, change_7d, volume_24h, market_cap, volatility]])

# Predict and display
if st.button("Predict Liquidity Ratio"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔍 Predicted Liquidity Ratio: `{round(prediction, 6)}`")

    st.markdown("### 🔢 Input Summary")
    st.json({
        "Price (USD)": price,
        "1h Change (%)": change_1h,
        "24h Change (%)": change_24h,
        "7d Change (%)": change_7d,
        "24h Volume": volume_24h,
        "Market Cap": market_cap,
        "Volatility": volatility
    })

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
