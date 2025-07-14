import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="wide")

st.title("Crypto Liquidity Prediction App")

# === Upload 2 CSVs ===
uploaded_files = st.file_uploader("Upload 2 CoinGecko CSV Files", type=["csv"], accept_multiple_files=True)

if len(uploaded_files) != 2:
    st.warning("Please upload **2 CSV files** to continue.")
    st.stop()

# === Load and Combine Data ===
try:
    df1 = pd.read_csv(uploaded_files[0])
    df2 = pd.read_csv(uploaded_files[1])
    df = pd.concat([df1, df2], ignore_index=True)
except Exception as e:
    st.error(f"Error reading uploaded files: {e}")
    st.stop()

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['coin', 'date'], inplace=True)
df.dropna(inplace=True)

for col in ['1h', '24h', '7d']:
    if col in df.columns:
        df[col] = df[col].astype(float)

df['liquidity_ratio'] = df['24h_volume'] / df['mkt_cap']
df['volatility'] = df['24h'].abs()

st.subheader("🔍 Sample Data Preview")
st.dataframe(df.head())

# === EDA: Correlation Heatmap ===
st.subheader("Feature Correlation Heatmap")
features_corr = ['price', '24h_volume', 'mkt_cap', 'liquidity_ratio', 'volatility']
if all(col in df.columns for col in features_corr):
    fig, ax = plt.subplots()
    sns.heatmap(df[features_corr].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.warning("Not all required columns available for correlation heatmap.")

# === Model Training ===
st.subheader("Train Random Forest Regressor")

features = ['price', '1h', '24h', '7d', '24h_volume', 'mkt_cap', 'volatility']
target = 'liquidity_ratio'

X = df[features].select_dtypes(include=np.number)
y = df[target]

X.dropna(inplace=True)
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Grid Search (optional but included for completeness)
params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
grid = GridSearchCV(RandomForestRegressor(), params, cv=3, scoring='r2')
grid.fit(X_train, y_train)
best_params = grid.best_params_

# Save model
model_filename = "crypto_liquidity_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

st.success(f"Model trained and saved as `{model_filename}`")
st.write("Best Parameters from GridSearch:", best_params)

# === Evaluation ===
st.subheader("Model Performance")
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.metric(label="R² Score", value=f"{r2:.4f}")
st.metric(label="RMSE", value=f"{rmse:.4f}")

# === Prediction Form ===
st.subheader("Predict Liquidity Ratio")

price = st.number_input("Current Price (USD)", value=100.0)
pct_1h = st.number_input("1h Change (%)", value=0.5)
pct_24h = st.number_input("24h Change (%)", value=1.2)
pct_7d = st.number_input("7d Change (%)", value=-3.5)
vol_24h = st.number_input("24h Volume", value=5_000_000.0)
market_cap = st.number_input("Market Cap", value=100_000_000.0)

volatility = abs(pct_24h)

if st.button("Predict Now"):
    input_data = np.array([[price, pct_1h, pct_24h, pct_7d, vol_24h, market_cap, volatility]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Liquidity Ratio: **{round(prediction, 6)}**")

    st.markdown("### Input Summary")
    st.json({
        "Price": price,
        "1h %": pct_1h,
        "24h %": pct_24h,
        "7d %": pct_7d,
        "24h Volume": vol_24h,
        "Market Cap": market_cap,
        "Volatility": volatility
    })

st.markdown("---")
st.caption("Built with using Streamlit + RandomForestRegressor")
