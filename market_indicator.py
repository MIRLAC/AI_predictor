import yfinance as yf
import ta
import pandas as pd

ticker = 'HDFCBANK.NS'  # Change to RELIANCE.NS, ^NSEI, ^NSEBANK, etc.

data = yf.download(ticker, start='2020-01-01', end='2024-12-31')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
close = data['Close'].squeeze()  # Ensures 1D
data['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
data['MACD'] = ta.trend.MACD(close=close).macd()
data['SMA_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()


print(data.tail())
