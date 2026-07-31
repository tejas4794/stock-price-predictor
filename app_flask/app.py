import sys
import os

# Allow importing from src/
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

FEATURE_COLS = [
    "close_lag_1", "close_lag_3", "close_lag_5", "close_lag_7",
    "ma_7", "ma_30", "volatility_7", "daily_return", "day_of_week"
]

app = Flask(__name__)

# Load model once at startup, not per-request
model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
logger.info("Model and scaler loaded successfully")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        logger.info(f"Received prediction request: {payload}")

        # Expecting all FEATURE_COLS values plus current "close" price
        missing = [col for col in FEATURE_COLS + ["close"] if col not in payload]
        if missing:
            logger.warning(f"Missing fields in request: {missing}")
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        X = pd.DataFrame([{col: payload[col] for col in FEATURE_COLS}])
        X_scaled = scaler.transform(X)

        pred_return = model.predict(X_scaled)[0]
        current_close = payload["close"]
        predicted_price = current_close * (1 + pred_return)

        logger.info(f"Prediction: return={pred_return:.5f}, predicted_price={predicted_price:.2f}")

        return jsonify({
            "predicted_return": float(pred_return),
            "predicted_next_close": float(predicted_price)
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)