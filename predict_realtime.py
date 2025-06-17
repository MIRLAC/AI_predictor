import yfinance as yf
import pandas as pd
import joblib
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from plyer import notification
import os

# ✅ List of stocks and corresponding model files
stocks = {
    "HDFCBANK.NS": "models/hdfcbankns_rf_model.pkl",
    "RELIANCE.NS": "models/reliancens_rf_model.pkl",
    # Add more stocks and models here
}

def preprocess_and_predict(ticker, model_path):
    print(f"\n📥 Fetching data for {ticker}...")
    df = yf.download(ticker, period="30d", interval="1h", progress=False)

    # Handle multi-index columns (if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Add indicators
    try:
        df['RSI'] = RSIIndicator(close=df['Close']).rsi().squeeze()
        df['MACD'] = MACD(close=df['Close']).macd().squeeze()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator().squeeze()
        df.dropna(inplace=True)
    except Exception as e:
        print(f"⚠️ Error computing indicators for {ticker}: {e}")
        return

    if df.empty:
        print(f"❌ No data for {ticker} after preprocessing.")
        return

    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path}")

    # Make prediction
    latest = df.iloc[-1]
    X_latest = pd.DataFrame([latest[['RSI', 'MACD', 'SMA_50']]])
    prediction = model.predict(X_latest)[0]
    action = "📈 BUY" if prediction == 1 else "📉 SELL"

    print(f"🔮 {ticker} Prediction: {action}")

    # Send notification
    notification.notify(
        title=f"📊 {ticker} Signal",
        message=f"Action: {action}",
        timeout=10
    )

# ✅ Infinite loop every 60 seconds
print("🔁 Starting real-time prediction loop...")
while True:
    for stock, model_file in stocks.items():
        preprocess_and_predict(stock, model_file)

    print("⏳ Waiting 60 seconds...\n")
    time.sleep(60)
