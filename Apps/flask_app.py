from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = "crypto_liquidity_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise Exception(f"Model file not found or corrupt: {e}")

# HTML Template (Simple UI)
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Liquidity Predictor</title>
    <style>
        body { background-color: #121212; color: #f1f1f1; font-family: Arial; padding: 30px; }
        input, label { display: block; margin-top: 10px; }
        h1 { color: #00ffcc; }
        button { margin-top: 15px; }
    </style>
</head>
<body>
    <h1>💧 Crypto Liquidity Prediction</h1>
    <form action="/predict" method="post">
        <label>Price (USD): <input type="number" step="0.01" name="price" required></label>
        <label>1h Change (%): <input type="number" step="0.01" name="1h" required></label>
        <label>24h Change (%): <input type="number" step="0.01" name="24h" required></label>
        <label>7d Change (%): <input type="number" step="0.01" name="7d" required></label>
        <label>24h Volume: <input type="number" name="24h_volume" required></label>
        <label>Market Cap: <input type="number" name="mkt_cap" required></label>
        <button type="submit">Predict</button>
    </form>
    {% if prediction %}
        <h2>📊 Predicted Liquidity Ratio: {{ prediction }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            data = {
                "price": float(request.form["price"]),
                "1h": float(request.form["1h"]),
                "24h": float(request.form["24h"]),
                "7d": float(request.form["7d"]),
                "24h_volume": float(request.form["24h_volume"]),
                "mkt_cap": float(request.form["mkt_cap"]),
            }
            features = [
                data["price"], data["1h"], data["24h"], data["7d"],
                data["24h_volume"], data["mkt_cap"], abs(data["24h"])
            ]
            features_array = np.array(features).reshape(1, -1)
            result = model.predict(features_array)[0]
            prediction = round(result, 6)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(html_template, prediction=prediction)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict(flat=True)
        required = ["price", "1h", "24h", "7d", "24h_volume", "mkt_cap"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = [
            float(data["price"]), float(data["1h"]), float(data["24h"]),
            float(data["7d"]), float(data["24h_volume"]),
            float(data["mkt_cap"]), abs(float(data["24h"]))
        ]
        features_array = np.array(features).reshape(1, -1)
        pred = model.predict(features_array)[0]
        return jsonify({
            "input": data,
            "predicted_liquidity_ratio": round(pred, 6)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
