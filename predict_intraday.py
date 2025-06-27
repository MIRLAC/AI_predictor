# ✅ ANGEL ONE VERSION of predict_intraday.py
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import pandas as pd
import joblib
from datetime import datetime, timedelta
import time
import os
import numpy as np
from login_helper import create_session

# ==== Start session ====
session = create_session(
    client_id="J57353407",
    client_pwd="2222",
    totp_secret="CGPGDARIZ6EE2C555AR2RZDZ2Q",
    market_feed_key="GkzJlEzd",
    historical_feed_key="iAOWimJg"
)

smartapi = session

# ==== Define symbols and models ====
symbols = {
    'RELIANCE': {'symbol_token': '2885', 'exchange': 'NSE', 'model': 'reliancens_intraday_model.pkl'},
    'HDFCBANK': {'symbol_token': '1333', 'exchange': 'NSE', 'model': 'hdfcbankns_intraday_model.pkl'},
    'NIFTY': {'symbol_token': '99926000', 'exchange': 'NSE', 'model': 'nifty50_intraday_model.pkl'},
    'BANKNIFTY': {'symbol_token': '99926009', 'exchange': 'NSE', 'model': 'banknifty_intraday_model.pkl'}
}

# ==== Predict ====
for stock, meta in symbols.items():
    try:
        print(f"\n📥 Fetching data for {stock}")

        now = datetime.now()
        now = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)
        todate = now.strftime("%Y-%m-%d %H:%M")
        fromdate = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")

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
            print(f"❌ No data for {stock}.")
            continue

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.set_index('datetime', inplace=True)
        df.rename(columns={'close': 'Close'}, inplace=True)

        print(f"📊 Raw candles fetched for {stock}: {df.shape[0]}")

        # Add indicators
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df.dropna(inplace=True)

        if df.empty or len(df) < 1:
            print("❌ Not enough data after indicators.")
            continue

        latest = df.iloc[-1]
        latest_features = latest[['RSI', 'MACD', 'SMA_20']].values.reshape(1, -1)
        current_price = latest['Close']
        predict_for = df.index[-1] + pd.Timedelta(minutes=5)

        model_path = f"models/{meta['model']}"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        predicted_price = float(model.predict(latest_features)[0])

        # Output
        print(f"🕒 Prediction Time: {predict_for.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ₹{current_price:.2f}")
        print(f"🔮 Predicted Price: ₹{predicted_price:.2f}")

        diff = predicted_price - current_price
        if diff > 1.0:
            print("📈 Recommendation: BUY")
        elif diff < -0.5:
            print("📉 Recommendation: SELL")
        else:
            print("⏸ Recommendation: WAIT")

    except Exception as e:
        print(f"❌ Error for {stock}: {e}")
    time.sleep(2)
