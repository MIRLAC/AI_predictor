import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

stocks = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}

lookback = 50
end = datetime.now()
start = end - timedelta(days=1)

for stock, ticker in stocks.items():
    print(f"\n📈 {stock}")

    try:
        # Load model and scaler
        model_path = f"models/{stock.lower()}_eod.h5"
        scaler_path = f"models/{stock.lower()}_eod_scaler.pkl"
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        # Fetch data
        df = yf.download(ticker, interval="5m", start=start, end=end, progress=False)
        df = df[['Close']].dropna()

        if len(df) < lookback:
            print("❌ Not enough data to predict.")
            continue

        recent_close = df[-lookback:]['Close'].values
        X_input = scaler.transform(recent_close.reshape(-1, 1)).reshape(1, lookback, 1)

        # Predict
        pred_scaled = model.predict(X_input)[0][0]
        predicted_close = scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = df['Close'].iloc[-1].item()  # ✅ Ensures it's a float


        # Print
        print(f"🕒 Time Now: {datetime.now().strftime('%H:%M')}")
        print(f"💰 Current Price: ₹{current_price:.2f}")
        print(f"📅 Predicted EOD Close (15:30): ₹{predicted_close:.2f}")
    except Exception as e:
        print(f"❌ Error predicting {stock}: {e}")
