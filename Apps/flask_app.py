# Ensure Flask is installed
!pip install flask --quiet

# Verify installation (Optional, but good for debugging)
# try:
#     import flask
#     print("Flask imported successfully after installation.")
# except ModuleNotFoundError:
#     print("Flask still not found after installation attempt.")

from flask import Flask, request, jsonify
import pickle
import numpy as np
import os # Import os to check file existence

import warnings
warnings.filterwarnings("ignore")

# Load model
# Make sure the model file exists in the correct path or provide the full path
model_path = "crypto_liquidity_model.pkl" # Define the model path
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: '{model_path}' not found. Please ensure the model file is in the correct directory.")
    # Handle the error appropriately, e.g., exit or raise an exception
    # In a notebook cell, `exit()` will stop the kernel, which might not be desired.
    # Raising an error is usually better for debugging.
    raise FileNotFoundError(f"Model file not found at {model_path}")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    raise # Re-raise the exception


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Ensure all required keys are in the input data
        required_keys = ["price", "1h", "24h", "7d", "24h_volume", "mkt_cap"]
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing one or more required keys in input data."}), 400

        features = [
            data["price"],
            data["1h"],
            data["24h"],
            data["7d"],
            data["24h_volume"],
            data["mkt_cap"],
            abs(data["24h"])
        ]
        # Reshape features to be a 2D array as expected by model.predict
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        return jsonify({"liquidity_ratio": round(prediction, 4)})
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Crypto Liquidity Prediction API is running. Use POST /predict."

if __name__ == "__main__":
    # Use a different port if 5000 is already in use
    # Also, in Colab, running a Flask app directly like this
    # requires extra steps like using ngrok for external access.
    # For simple local testing within Colab, this might work,
    # but for external access, consider using a tool like ngrok.
    print("Running Flask app...")
    # Note: Running Flask directly in a Colab cell like this can block the cell execution.
    # You might need to run it in a separate thread or process if you need to execute
    # other code in subsequent cells.
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to run Flask app: {e}")

