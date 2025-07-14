import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

st.title("💧 Crypto Liquidity Ratio Predictor")
st.markdown("Upload cryptocurrency market data to predict the liquidity ratio using a trained machine learning model.")

# === Load the Model ===
model_path = "crypto_liquidity_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Trained model file not found! Please make sure 'crypto_liquidity_model.pkl' is in the same directory.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.success("✅ Trained model loaded successfully.")

# === Input Form ===
st.sidebar.header("📊 Input Parameters")
price = st.sidebar.number_input("Price (USD)", value=100.0)
pct_1h = st.sidebar.number_input("1h Change (%)", value=0.5)
pct_24h = st.sidebar.number_input("24h Change (%)", value=1.2)
pct_7d = st.sidebar.number_input("7d Change (%)", value=-3.5)
vol_24h = st.sidebar.number_input("24h Trading Volume", value=5_000_000.0)
mkt_cap = st.sidebar.number_input("Market Capitalization", value=100_000_000.0)

volatility = abs(pct_24h)

# === Prediction ===
if st.button("🚀 Predict Liquidity Ratio"):
    input_data = np.array([[price, pct_1h, pct_24h, pct_7d, vol_24h, mkt_cap, volatility]])
    prediction = model.predict(input_data)[0]
    st.success(f"🔍 Predicted Liquidity Ratio: `{round(prediction, 6)}`")

    st.markdown("### 🧾 Input Summary")
    st.json({
        "Price": price,
        "1h Change": pct_1h,
        "24h Change": pct_24h,
        "7d Change": pct_7d,
        "24h Volume": vol_24h,
        "Market Cap": mkt_cap,
        "Volatility": volatility
    })

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn")
