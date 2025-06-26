import yfinance as yf
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def add_indicators(df):
    close = df['Close']

    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    close = pd.Series(close, dtype='float64')

    df['RSI'] = RSIIndicator(close=close).rsi()
    df['MACD'] = MACD(close=close).macd_diff()
    df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
    return df.dropna()

def train_model(symbol, filename):
    print(f"\n📊 Training model for {symbol}")
    df = yf.download(symbol, period="5d", interval="5m", auto_adjust=True)
    if df.empty:
        print("❌ No data.")
        return

    df = add_indicators(df)
    # Predicting 10 minutes into the future
    df['Target'] = df['Close'].shift(-2)
    df.dropna(inplace=True)

    features = df[['RSI', 'MACD', 'SMA_20']]
    target = df['Target']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{filename}")
    print(f"✅ Model saved to models/{filename}")

symbols = {
    'HDFCBANK.NS': 'hdfcbankns_intraday_model.pkl',
    'RELIANCE.NS': 'reliancens_intraday_model.pkl',
    '^NSEI': 'nifty50_intraday_model.pkl',
    '^NSEBANK': 'banknifty_intraday_model.pkl'
}

for sym, file in symbols.items():
    train_model(sym, file)