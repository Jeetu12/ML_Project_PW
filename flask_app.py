from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

app = Flask(__name__)

model_path = "crypto_liquidity_model.pkl"

def train_model():
    X = np.random.rand(100, 7)
    y = np.random.rand(100)
    model = RandomForestRegressor()
    model.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

@app.route("/")
def home():
    return " Crypto Liquidity Prediction API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        required = ["price", "1h", "24h", "7d", "24h_volume", "mkt_cap"]
        if not all(k in data for k in required):
            return jsonify({"error": "Missing required fields."}), 400

        features = [
            data["price"],
            data["1h"],
            data["24h"],
            data["7d"],
            data["24h_volume"],
            data["mkt_cap"],
            abs(data["24h"])  # volatility
        ]

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        return jsonify({"liquidity_ratio": round(prediction, 6)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
