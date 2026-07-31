import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [
    "close_lag_1", "close_lag_3", "close_lag_5", "close_lag_7",
    "ma_7", "ma_30", "volatility_7", "daily_return", "day_of_week"
]

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Stock Price Predictor")
st.caption("Educational project — not financial advice.")

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return model, scaler

model, scaler = load_model_and_scaler()

ticker = st.text_input("Enter stock ticker", value="AAPL")

if st.button("Predict Next-Day Close"):
    try:
        logger.info(f"Fetching live data for {ticker}")
        df = yf.download(ticker, period="3mo")

        if df.empty:
            st.error(f"No data found for ticker '{ticker}'. Check the symbol and try again.")
        else:
            df = df.reset_index()
            df.columns = [col[0] if isinstance(col, tuple) and col[0] != '' else col for col in df.columns]

            df["close_lag_1"] = df["Close"].shift(1)
            df["close_lag_3"] = df["Close"].shift(3)
            df["close_lag_5"] = df["Close"].shift(5)
            df["close_lag_7"] = df["Close"].shift(7)
            df["ma_7"] = df["Close"].rolling(7).mean()
            df["ma_30"] = df["Close"].rolling(30).mean()
            df["volatility_7"] = df["Close"].rolling(7).std()
            df["daily_return"] = df["Close"].pct_change()
            df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek

            latest = df.dropna().iloc[-1]
            X = pd.DataFrame([latest[FEATURE_COLS]])
            X_scaled = scaler.transform(X)

            pred_return = model.predict(X_scaled)[0]
            current_close = latest["Close"]
            predicted_price = current_close * (1 + pred_return)

            logger.info(f"{ticker}: current={current_close:.2f}, predicted={predicted_price:.2f}")

            col1, col2 = st.columns(2)
            col1.metric("Current Close", f"${current_close:.2f}")
            col2.metric("Predicted Next Close", f"${predicted_price:.2f}", f"{pred_return*100:.3f}%")

            st.line_chart(df.set_index("Date")["Close"].tail(60))

    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        st.error(f"Something went wrong: {e}")