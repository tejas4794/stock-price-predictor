import yfinance as yf
import pandas as pd
import os
from logger import get_logger

logger = get_logger(__name__)

# These two lines make paths work no matter which folder you run the script from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> stock-predictor/
DATA_DIR = os.path.join(BASE_DIR, "data")

def fetch_stock_data(ticker="AAPL", period="5y", save=True):
    logger.info(f"Fetching data for {ticker}, period={period}")
    try:
        df = yf.download(ticker, period=period)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        df.reset_index(inplace=True)
        logger.info(f"Fetched {len(df)} rows for {ticker}")

        if save:
            os.makedirs(DATA_DIR, exist_ok=True)
            filepath = os.path.join(DATA_DIR, f"{ticker}_raw.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved raw data to {filepath}")

        return df
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    data = fetch_stock_data("AAPL")
    print(data.head())