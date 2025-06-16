import yfinance as yf
import pandas as pd
import ta
import joblib
from plyer import notification
ticker = 'HDFCBANK.NS'
model = joblib.load('hdfcbankns_rf_model.pkl')  # Update model file per stock
data = yf.download('HDFCBANK.NS', period='30d', interval='1h')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
close_series = data['Close'].squeeze()


data['RSI'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
data['MACD'] = ta.trend.MACD(close_series).macd()
data['SMA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()

data.dropna(inplace=True)

if data.empty:
    print("Not enough data.")
    exit()

latest = data.iloc[-1][['RSI', 'MACD', 'SMA_50']].values.reshape(1, -1)
prediction = model.predict(latest)[0]
print(f"Prediction: {'📈 UP' if prediction == 1 else '📉 DOWN'}")
# Show a desktop notification
notification.notify(
    title="Stock Prediction Complete ✅",
    message="Today's stock prediction has been generated.",
    app_name="Stock Predictor",
    timeout=10
)