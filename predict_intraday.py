from login_helper import create_session
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
import pandas as pd
import joblib
from datetime import datetime, timedelta

# ==== Your SmartAPI credentials ====
api_key = "IbvVzG5v"
client_id = "J57353407"
client_pwd = "2222"
totp_secret = "CGPGDARIZ6EE2C555AR2RZDZ2Q"

# ==== Start session ====
smartapi = create_session(api_key, client_id, client_pwd, totp_secret)

# ==== Define symbol tokens ====
symbols = {
    'RELIANCE-EQ': {
        'symbol_token': '2885',
        'exchange': 'NSE',
        'model': 'reliancens_intraday_model.pkl'
    },
    'HDFCBANK-EQ': {
        'symbol_token': '1333',
        'exchange': 'NSE',
        'model': 'hdfcbankns_intraday_model.pkl'
    },
    'NIFTY 50': {
        'symbol_token': '99926000',
        'exchange': 'NSE',
        'model': 'nifty50_intraday_model.pkl'
    },
    'BANK NIFTY': {
        'symbol_token': '99926009',
        'exchange': 'NSE',
        'model': 'banknifty_intraday_model.pkl'
    }
}

for scrip, meta in symbols.items():
    try:
        print(f"\n📥 Fetching data for {scrip}")

        now = datetime.now()
        earlier = now - timedelta(hours=1)

        candles = smartapi.getCandleData(
            exchange=meta['exchange'],
            symboltoken=meta['symbol_token'],
            interval="FIVE_MINUTE",
            fromdate=earlier.strftime("%Y-%m-%d %H:%M"),
            todate=now.strftime("%Y-%m-%d %H:%M")
        )

        if not candles.get('data'):
            print(f"❌ No candle data received for {scrip}")
            continue

        df = pd.DataFrame(candles['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.set_index('datetime', inplace=True)
        df.rename(columns={'close': 'Close'}, inplace=True)

        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df.dropna(inplace=True)

        if df.empty:
            print("❌ Not enough data after indicators.")
            continue

        latest = df[['RSI', 'MACD', 'SMA_20']].iloc[[-1]]
        current_price = df['Close'].iloc[-1]

        model = joblib.load(f"models/{meta['model']}")
        prediction = model.predict(latest)[0]

        print(f"💰 Current: ₹{current_price:.2f} | Prediction: ₹{prediction:.2f}")
        diff = prediction - current_price

        if diff > 0.5:
            print("📈 Recommendation: BUY")
        elif diff < -0.5:
            print("📉 Recommendation: SELL")
        else:
            print("⏸ Recommendation: WAIT")

    except Exception as e:
        print(f"❌ Error for {scrip}: {e}")

