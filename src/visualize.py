import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [
    "close_lag_1", "close_lag_3", "close_lag_5", "close_lag_7",
    "ma_7", "ma_30", "volatility_7", "daily_return", "day_of_week"
]

def plot_predictions():
    df = pd.read_csv(os.path.join(DATA_DIR, "AAPL_features.csv"), parse_dates=["Date"])

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:]

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    X_test = test_df[FEATURE_COLS]
    X_test_scaled = scaler.transform(X_test)

    pred_returns = model.predict(X_test_scaled)
    actual_today_close = test_df["Close"].values
    actual_next_close = actual_today_close * (1 + test_df["target_return"].values)
    pred_next_close = actual_today_close * (1 + pred_returns)

    logger.info("Plotting actual vs predicted prices")

    plt.figure(figsize=(12, 6))
    plt.plot(test_df["Date"], actual_next_close, label="Actual", linewidth=2)
    plt.plot(test_df["Date"], pred_next_close, label="Predicted", linewidth=2, linestyle="--")
    plt.title("Actual vs Predicted Next-Day Close Price (AAPL)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(BASE_DIR, "models", "prediction_plot.png")
    plt.savefig(output_path)
    logger.info(f"Saved plot to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_predictions()