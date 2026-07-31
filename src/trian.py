import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# "Close" removed from features — it leaked the answer directly
FEATURE_COLS = [
    "close_lag_1", "close_lag_3", "close_lag_5", "close_lag_7",
    "ma_7", "ma_30", "volatility_7", "daily_return", "day_of_week"
]
TARGET_COL = "target_return"

def load_data():
    filepath = os.path.join(DATA_DIR, "AAPL_features.csv")
    logger.info(f"Loading features from {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return df

def time_based_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    return train_df, test_df

def train_and_evaluate():
    df = load_data()
    train_df, test_df = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    }

    # Everything is converted to actual price terms before measuring error,
    # so RMSE is interpretable in dollars, not raw return units
    actual_today_close = test_df["Close"].values
    actual_next_close = actual_today_close * (1 + y_test.values)

    naive_next_close = actual_today_close * (1 + np.zeros(len(y_test)))  # naive: "no change"
    naive_rmse = np.sqrt(mean_squared_error(actual_next_close, naive_next_close))
    logger.info(f"Naive baseline (no change) -> RMSE: {naive_rmse:.4f}")

    best_model_name, best_rmse, best_model = None, float("inf"), None

    for name, model in models.items():
        logger.info(f"Training {name}")
        model.fit(X_train_scaled, y_train)
        pred_returns = model.predict(X_test_scaled)
        pred_next_close = actual_today_close * (1 + pred_returns)

        rmse = np.sqrt(mean_squared_error(actual_next_close, pred_next_close))
        mae = mean_absolute_error(actual_next_close, pred_next_close)
        r2 = r2_score(actual_next_close, pred_next_close)

        logger.info(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f} (naive baseline RMSE: {naive_rmse:.4f})")

        if rmse < best_rmse:
            best_model_name, best_rmse, best_model = name, rmse, model

    logger.info(f"Best model: {best_model_name} with RMSE {best_rmse:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    logger.info(f"Saved {best_model_name} and scaler to {MODELS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()