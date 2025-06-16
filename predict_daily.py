import yfinance as yf
import pandas as pd
import ta
import joblib
from plyer import notification
import sys

# ----------------------------
# Select your stock from below:
# ----------------------------
# Options: 'HDFCBANK.NS', 'RELIANCE.NS', '^NSEI', '^NSEBANK'
ticker = 'HDFCBANK.NS'  # Change this as needed

# Automatically load the corresponding model
# Example: HDFCBANK.NS -> hdfcbankns_rf_model.pkl
model_file = ticker.replace('.', '').replace('^', '').lower() + '_rf_model.pkl'

try:
    model = joblib.load(model_file)
except FileNotFoundError:
    print(f"Model file '{model_file}' not found. Make sure the model exists.")
    sys.exit(1)

# Download stock data
try:
    data = yf.download(ticker, period='30d', interval='1h')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
except Exception as e:
    print(f"Error downloading data: {e}")
    sys.exit(1)

# Compute indicators
close_series = data['Close'].squeeze()
data['RSI'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
data['MACD'] = ta.trend.MACD(close_series).macd()
data['SMA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()

# Drop NaN rows
data.dropna(inplace=True)

# Safety check
if data.empty:
    print("Not enough data after computing indicators. Try a longer period.")
    sys.exit(1)
# Prepare features and predict
print("Latest row used for prediction:")
print(data.tail(1))

print("Latest data timestamp:", data.index[-1])

latest_row = data.iloc[-1]
print(f"Predicting for time: {latest_row.name} — Close: {latest_row['Close']}")


# Prepare features and predict
latest = data.iloc[-1][['RSI', 'MACD', 'SMA_50']].values.reshape(1, -1)
prediction = model.predict(latest)[0]

# Output result
result = f"Prediction for {ticker}: {'📈 UP' if prediction == 1 else '📉 DOWN'}"
print(result)

# Desktop notification
notification.notify(
    title="Stock Prediction Complete ✅",
    message=result,
    app_name="Stock Predictor",
    timeout=10
)
