import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta

# Define your stocks and model/scaler paths
stocks = {
    'RELIANCE': 'models/reliance_lstm.h5',
    'HDFCBANK': 'models/hdfcbank_lstm.h5',
    'NIFTY': 'models/nifty_lstm.h5',
    'BANKNIFTY': 'models/banknifty_lstm.h5'
}

scalers = {
    'RELIANCE': 'models/reliance_scaler.pkl',
    'HDFCBANK': 'models/hdfcbank_scaler.pkl',
    'NIFTY': 'models/nifty_scaler.pkl',
    'BANKNIFTY': 'models/banknifty_scaler.pkl'
}

# Define ticker symbols for yfinance
tickers = {
    'RELIANCE': 'RELIANCE.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK'
}

def fetch_recent_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=3)  # ensure enough candles
    df = yf.download(ticker, start=start, end=end, interval='5m', progress=False)
    return df

def predict_next_close(stock):
    model = tf.keras.models.load_model(stocks[stock])
    scaler = joblib.load(scalers[stock])
    df = fetch_recent_data(tickers[stock])

    # Clean and extract features
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
    if len(df) < 50:
        print(f"❌ Not enough candles for {stock}")
        return

    recent = df[-50:].copy()
    close_prices = recent[['Close']]  # Only use 'Close' for scaling
    scaled = scaler.transform(close_prices)
    X_input = np.array([scaled])  # Shape (1, 50, 1)

    pred_scaled = model.predict(X_input)[0][0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    curr_price = float(df['Close'].iloc[-1].item())

    # 🖨️ Output
    print(f"\n📈 {stock}")
    print(f"🕒 Current Time: {datetime.now().strftime('%H:%M')}")
    print(f"💰 Current Price: ₹{curr_price:.2f}")
    print(f"🔮 Predicted 5-min Close: ₹{pred_price:.2f}")
    print("✅ Stable prediction for next 5-min window")

# ---- MAIN LOOP ----
for stock in stocks:
    predict_next_close(stock)
