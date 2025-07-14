import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")

st.title("💧 Crypto Liquidity Ratio Predictor")
st.markdown("Predict the **liquidity ratio** of a cryptocurrency using a pre-trained model (built-in).")

# =======================
# TRAIN MODEL INLINE
# =======================
@st.cache_resource
def train_model():
    # Generate dummy training data
    X = np.random.rand(100, 7)
    y = np.random.rand(100)
    model = RandomForestRegressor().fit(X, y)
    return model

model = train_model()

# =======================
# INPUTS
# =======================
st.sidebar.header("📊 Input Features")
price = st.sidebar.number_input("Current Price (USD)", value=100.0)
pct_1h = st.sidebar.number_input("1h Change (%)", value=0.5)
pct_24h = st.sidebar.number_input("24h Change (%)", value=1.2)
pct_7d = st.sidebar.number_input("7d Change (%)", value=-3.5)
vol_24h = st.sidebar.number_input("24h Volume", value=5_000_000.0)
market_cap = st.sidebar.number_input("Market Cap", value=100_000_000.0)

volatility = abs(pct_24h)

# =======================
# PREDICT
# =======================
if st.button("🚀 Predict Liquidity Ratio"):
    input_data = np.array([[price, pct_1h, pct_24h, pct_7d, vol_24h, market_cap, volatility]])
    pred = model.predict(input_data)[0]
    st.success(f"🔍 Predicted Liquidity Ratio: `{round(pred, 6)}`")

    st.markdown("### 📌 Input Summary")
    st.json({
        "Price": price,
        "1h Change": pct_1h,
        "24h Change": pct_24h,
        "7d Change": pct_7d,
        "24h Volume": vol_24h,
        "Market Cap": market_cap,
        "Volatility": volatility
    })

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
