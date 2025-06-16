import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier
import joblib

data = yf.download('HDFCBANK.NS', start='2020-01-01', end='2024-12-31')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Fix here: use squeeze() to get 1D Series
close_series = data['Close'].squeeze()

data['RSI'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
data['MACD'] = ta.trend.MACD(close_series).macd()
data['SMA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()

data.dropna(inplace=True)

data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

X = data[['RSI', 'MACD', 'SMA_50']]
y = data['Target']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'hdfcbank_rf_model.pkl')

print("✅ Model training complete and saved as hdfcbank_rf_model.pkl")
