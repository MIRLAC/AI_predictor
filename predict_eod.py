# ✅ ANGEL ONE VERSION of predict_eod.py (No yfinance)
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from login_helper import create_session

# Setup Angel One SmartAPI session
session = create_session(
    client_id="J57353407",
    client_pwd="2222",
    totp_secret="CGPGDARIZ6EE2C555AR2RZDZ2Q",
    market_feed_key="GkzJlEzd",
    historical_feed_key="iAOWimJg"
)

smartapi = session

symbols = {
    "RELIANCE": {"symbol_token": "2885", "exchange": "NSE"},
    "HDFCBANK": {"symbol_token": "1333", "exchange": "NSE"},
    "NIFTY": {"symbol_token": "99926000", "exchange": "NSE"},
    "BANKNIFTY": {"symbol_token": "99926009", "exchange": "NSE"}
}

lookback = 50
now = datetime.now()
now = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)
todate = now.strftime("%Y-%m-%d %H:%M")
fromdate = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")

for stock, meta in symbols.items():
    print(f"\n📈 {stock}")
    try:
        # Load model and scaler
        model_path = f"models/{stock.lower()}_eod.h5"
        scaler_path = f"models/{stock.lower()}_eod_scaler.pkl"
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        params = {
            "exchange": meta["exchange"],
            "symboltoken": meta["symbol_token"],
            "interval": "FIVE_MINUTE",
            "fromdate": fromdate,
            "todate": todate
        }

        response = smartapi.getCandleData(params)
        data = response.get("data", [])

        if not data:
            print("❌ No candle data returned.")
            continue

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.set_index('datetime', inplace=True)
        df = df[['close']]
        df.rename(columns={'close': 'Close'}, inplace=True)

        if len(df) < lookback:
            print("❌ Not enough data to predict.")
            continue

        recent_close = df[-lookback:]['Close'].values
        X_input = scaler.transform(recent_close.reshape(-1, 1)).reshape(1, lookback, 1)

        # Predict
        pred_scaled = model.predict(X_input)[0][0]
        predicted_close = scaler.inverse_transform([[pred_scaled]])[0][0]
        current_price = df['Close'].iloc[-1].item()

        # Print
        print(f"🕒 Time Now: {datetime.now().strftime('%H:%M')}")
        print(f"💰 Current Price: ₹{current_price:.2f}")
        print(f"📅 Predicted EOD Close (15:30): ₹{predicted_close:.2f}")

    except Exception as e:
        print(f"❌ Error predicting {stock}: {e}")
