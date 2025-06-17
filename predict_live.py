import yfinance as yf
import pandas as pd
import ta
import joblib
from datetime import datetime
from plyer import notification
import os

# ---------------------------
# ✅ Choose your stock ticker
# ---------------------------
ticker = 'HDFCBANK.NS'  # e.g., RELIANCE.NS, ^NSEI

# Generate matching model file name
model_file = f"models/{ticker.replace('^', '').replace('.', '').lower()}_rf_model.pkl"

# Check model exists
if not os.path.exists(model_file):
    print(f"❌ Model file not found: {model_file}")
    exit()

# Load model
model = joblib.load(model_file)
print(f"✅ Loaded model: {model_file}")

# Fetch latest stock data
print(f"📥 Fetching data for {ticker}...")
df = yf.download(ticker, period="30d", interval="1h")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Clean up columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)

# Add indicators
close_series = df['Close'].squeeze()  # Ensures it's a 1D Series
df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
df['MACD'] = ta.trend.MACD(close=close_series).macd()
df['SMA_50'] = ta.trend.SMAIndicator(close=close_series, window=50).sma_indicator()


# Drop NaNs created by indicators
df.dropna(inplace=True)
print(df.tail())  # Check what the last few rows look like

if df.empty:
    print("❌ No data available after preprocessing. Please check the stock symbol or data format.")
    exit()


# Get latest row
latest = df.iloc[-1]
features = latest[['RSI', 'MACD', 'SMA_50']].values.reshape(1, -1)

# Predict
prediction = model.predict(features)[0]
result = "📈 BUY" if prediction == 1 else "📉 SELL/HOLD"

# Output
print(f"\n🔍 Prediction for {ticker}")
print(f"Time: {latest.name}")
print(f"Close Price: ₹{latest['Close']:.2f}")
print(f"Model Suggests: {result}")

# Notify
notification.notify(
    title=f"Stock Signal: {ticker}",
    message=f"{result} at ₹{latest['Close']:.2f} ({latest.name})",
    app_name="Stock Predictor",
    timeout=10
)
