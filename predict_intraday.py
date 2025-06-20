import yfinance as yf
import pandas as pd
import joblib
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def add_indicators(df):
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    elif isinstance(close, pd.Series) and close.ndim > 1:
        close = close.iloc[:, 0]
    close = pd.Series(close, dtype='float64')

    df['RSI'] = RSIIndicator(close=close).rsi()
    df['MACD'] = MACD(close=close).macd_diff()
    df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()

    return df.dropna()

stocks = {
    'HDFCBANK.NS': 'hdfcbankns_intraday_model.pkl',
    'RELIANCE.NS': 'reliancens_intraday_model.pkl',
    '^NSEI': 'nsei_intraday_model.pkl',
    '^NSEBANK': 'nsebank_intraday_model.pkl'
}


for symbol, model_file in stocks.items():
    print(f"\n📥 Checking live intraday for {symbol}...")
    try:
        # Fetch last ~1 day of 5-minute data
        df = yf.download(symbol, period="1d", interval="5m", auto_adjust=True)
        if df.empty:
            print(f"❌ No data for {symbol}")
            continue

        df = add_indicators(df)
        if df.empty or len(df) < 2:
            print("❌ Not enough data after indicators.")
            continue

        latest = df[['RSI', 'MACD', 'SMA_20']].iloc[[-1]]
        current_price = float(df['Close'].iloc[-1].item())
        print(f"💰 Current price: {current_price:.2f}")


        model = joblib.load(f"models/{model_file}")
        prediction = model.predict(latest)[0]

        print(f"⏰ Time: {df.index[-1]}")
        print(f"💰 Price: ₹{current_price:.2f}")
        print(f"🔮 Prediction: ₹{prediction:.2f}")

        # Decision logic
        change_pct = ((prediction - current_price) / current_price) * 100
        if change_pct > 0.5:
            decision = "📈 BUY now"
        elif change_pct < -0.5:
            decision = "📉 SELL now"
        else:
            decision = "⏸ WAIT"

        print(f"🧠 {symbol} → {decision} | Target: ₹{prediction:.2f} | Change: {change_pct:+.2f}%")

    except Exception as e:
        print(f"❌ Error for {symbol}: {e}")
