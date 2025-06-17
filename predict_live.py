import yfinance as yf
import pandas as pd
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from plyer import notification
import os

# Helper to extract indicators
def add_indicators(df):
    close = df['Close'].squeeze()
    df['RSI'] = RSIIndicator(close=close).rsi()
    df['MACD'] = MACD(close=close).macd()
    df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()

    print(f"🔍 Rows before dropna: {len(df)}")
    df.dropna(inplace=True)
    print(f"✅ Rows after dropna: {len(df)}")
    return df


# List of tickers and corresponding model names
tickers = {
    "HDFCBANK.NS": "hdfcbankns_rf_model.pkl",
    "RELIANCE.NS": "reliancens_rf_model.pkl",
    "^NSEI": "nsei_rf_model.pkl",
    "^NSEBANK": "nsebank_rf_model.pkl"
}

# Run once for all tickers
for symbol, model_name in tickers.items():
    print(f"\n📥 Fetching data for {symbol}...")
    try:
        df = yf.download(symbol, period="3mo", interval="1d", progress=False)

        if df.empty:
            print(f"❌ No data available for {symbol}")
            continue

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = add_indicators(df)

        if df.empty:
            print(f"❌ No data available after indicators for {symbol}")
            continue

        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        latest = df.iloc[-1]
        X = latest[['RSI', 'MACD', 'SMA_50']].values.reshape(1, -1)
        prediction = model.predict(X)[0]
        action = "📈 BUY" if prediction == 1 else "📉 SELL"

        print(f"🔮 {symbol} Prediction: {action}")
        notification.notify(
            title=f"📊 {symbol} Stock Signal",
            message=f"Action: {action}",
            timeout=10
        )

    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
