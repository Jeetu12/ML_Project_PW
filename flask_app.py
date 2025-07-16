from flask import Flask, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import pickle

app = Flask(__name__)

model_path = "crypto_liquidity_model.pkl"

# ====== Train model if not available ======
def train_model():
    print(" Training new RandomForestRegressor model...")
    x = np.random.rand(100, 7)
    y = np.random.rand(100)
    model = RandomForestRegressor()
    model.fit(x, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(" Model trained and saved as crypto_liquidity_model.pkl")
    return model

# ====== Load or Train Model ======
if os.path.exists(model_path):
    print(" Loading trained model from file...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

# ====== Health Check ======
@app.route("/")
def home():
    return " Crypto Liquidity Prediction API is Live!"

# ====== Prediction Endpoint ======
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        print("\n Incoming Request Data:")
        print(data)

        required = ["price", "1h", "24h", "7d", "24h_volume", "mkt_cap"]
        missing = [key for key in required if key not in data]

        if missing:
            print(f" Missing keys: {missing}")
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        features = [
            data["price"],
            data["1h"],
            data["24h"],
            data["7d"],
            data["24h_volume"],
            data["mkt_cap"],
            abs(data["24h"])  # volatility
        ]

        print("\n Parsed Features for Prediction:")
        print(features)

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        prediction_rounded = round(prediction, 6)

        print(f"\n Predicted Liquidity Ratio: {prediction_rounded}")

        return jsonify({
            "input_features": {
                "price": data["price"],
                "1h": data["1h"],
                "24h": data["24h"],
                "7d": data["7d"],
                "24h_volume": data["24h_volume"],
                "mkt_cap": data["mkt_cap"],
                "volatility": abs(data["24h"])
            },
            "predicted_liquidity_ratio": prediction_rounded
        })

    except Exception as e:
        print(f"\n Internal Server Error: {e}")
        return jsonify({"error": str(e)}), 500

# ====== Start App ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n Starting Flask API on port {port}...")
    app.run(host="0.0.0.0", port=port)
