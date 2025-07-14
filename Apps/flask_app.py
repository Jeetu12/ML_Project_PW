from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

app = Flask(__name__)

# === TRAIN MODEL (embedded in app) ===
# This creates a self-contained model if no .pkl file exists
model_file = "crypto_liquidity_model.pkl"

if not os.path.exists(model_file):
    X = np.random.rand(100, 7)
    y = np.random.rand(100)

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # Save trained model
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
else:
    # Load existing model
    with open(model_file, "rb") as f:
        model = pickle.load(f)


@app.route("/")
def home():
    return "Crypto Liquidity Prediction API is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = ["price", "1h", "24h", "7d", "24h_volume", "mkt_cap"]
        if not all(k in data for k in required_fields):
            return jsonify({"error": f"Missing required fields: {required_fields}"}), 400

        # Prepare input
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
    port = int(os.environ.get("PORT", 5000))  # for Render deployment
    app.run(debug=True, host="0.0.0.0", port=port)
