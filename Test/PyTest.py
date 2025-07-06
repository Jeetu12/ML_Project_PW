"""
Unit tests for the liquidity prediction model and API.
Run with pytest.
"""

import pytest
import joblib
import numpy as np
from app.api import app  # importing Flask app

# Load model
model = joblib.load('models/liquidity_model.pkl')

def test_model_prediction_shape():
    # Use a dummy input to test the model output
    X_dummy = np.array([[1000.0, 0.01, 0.02]])
    pred = model.predict(X_dummy)
    assert pred.shape == (1,)
    assert isinstance(pred[0], float)

def test_model_api_prediction():
    # Use Flask test client to simulate API call
    client = app.test_client()
    data = {"price": 1000.0, "liquidity_ratio": 0.01, "volatility": 0.02}
    response = client.post('/predict', json=data)
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'predicted_liquidity_ratio' in json_data

def test_api_invalid_input():
    client = app.test_client()
    response = client.post('/predict', json={"price":1000})
    assert response.status_code == 400
