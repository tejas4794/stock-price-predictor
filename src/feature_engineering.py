import pandas as pd
import numpy as np
import os
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> stock-predictor/
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_and_clean(filepath):
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, header=[0, 1])  # handle multi-level header from yfinance

    df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df

def add_features(df):
    logger.info("Adding engineered features")

    for lag in [1, 3, 5, 7]:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["ma_30"] = df["Close"].rolling(window=30).mean()
    df["volatility_7"] = df["Close"].rolling(window=7).std()
    df["daily_return"] = df["Close"].pct_change()
    df["day_of_week"] = df["Date"].dt.dayofweek
    # df["target"] = df["Close"].shift(-1)
    df["target_return"] = df["Close"].shift(-1) / df["Close"] - 1  

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN, {len(df)} rows remain")

    return df

if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, "AAPL_raw.csv")
    output_path = os.path.join(DATA_DIR, "AAPL_features.csv")

    df = load_and_clean(input_path)
    df = add_features(df)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved feature-engineered data to {output_path}")
    print(df.head())
    print(df.columns.tolist())