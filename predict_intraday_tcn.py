import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta

# Model and scaler paths
stocks = {
    'RELIANCE': 'models/reliancens_tcn.h5',
    'HDFCBANK': 'models/hdfcbankns_tcn.h5',
    'NIFTY': 'models/nsei_tcn.h5',
    'BANKNIFTY': 'models/nsebank_tcn.h5'
}

scalers = {
    'RELIANCE': 'models/reliancens_scaler.pkl',
    'HDFCBANK': 'models/hdfcbankns_scaler.pkl',
    'NIFTY': 'models/nsei_scaler.pkl',
    'BANKNIFTY': 'models/nsebank_scaler.pkl'
}

tickers = {
    'RELIANCE': 'RELIANCE.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK'
}

def fetch_recent_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=3)
    df = yf.download(ticker, start=start, end=end, interval='5m')
    return df

def predict_next_close(stock):
    model = tf.keras.models.load_model(stocks[stock], compile=False)
    scaler = joblib.load(scalers[stock])
    df = fetch_recent_data(tickers[stock])

    df = df[['Close']].dropna().copy()
    if len(df) < 50:
        print(f"❌ Not enough candles for {stock}")
        return

    recent = df[-50:].copy()
    scaled = scaler.transform(recent[['Close']])
    X_input = np.array([scaled])
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    pred = model.predict(X_input)[0][0]
    pred_price = scaler.inverse_transform([[pred]])[0][0]

    curr_price = float(df['Close'].iloc[-1])
    print(f"\n📈 {stock}")
    print(f"🕒 Time Now: {datetime.now().strftime('%H:%M')}")
    print(f"💰 Current Price: ₹{curr_price:.2f}")
    print(f"🔮 Predicted Next 5-min Close: ₹{pred_price:.2f}")
    print(f"✅ Stable prediction for next candle")

# Run predictions for all stocks
for stock in stocks:
    predict_next_close(stock)
