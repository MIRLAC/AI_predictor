import yfinance as yf
import pandas as pd
import joblib
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def add_indicators(df):
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close = pd.Series(close, dtype='float64')

    df['RSI'] = RSIIndicator(close=close).rsi()
    df['MACD'] = MACD(close=close).macd_diff()
    df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
    return df.dropna()

# Define stock symbols and model paths
stocks = {
    'HDFCBANK.NS': 'hdfcbankns_rf_model.pkl',
    'RELIANCE.NS': 'reliancens_rf_model.pkl',
    '^NSEI': 'nsei_rf_model.pkl',
    '^NSEBANK': 'nsebank_rf_model.pkl'
}

summary = []

for symbol, model_file in stocks.items():
    print(f"\n📥 Fetching data for {symbol}...")

    try:
        df = yf.download(symbol, period="1d", interval="5m", auto_adjust=True)
        if df.empty:
            print(f"❌ No data available for {symbol}")
            continue

        df = add_indicators(df)
        if df.empty:
            print(f"❌ Not enough data after adding indicators for {symbol}")
            continue

        # Extract latest indicator values
        latest = df[['RSI', 'MACD', 'SMA_50']].iloc[[-1]]
        print("📐 Input DataFrame to model:")
        print("📏 Shape:", latest.shape)
        print("🧾 Content:\n", latest)

        current_price = df['Close'].iloc[-1].item()
        print(f"💰 Current price: {current_price:.2f}")

        model_path = os.path.join("models", model_file)
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        input_data = pd.DataFrame(latest.values, columns=latest.columns)

        print("🔧 Input reshaped to:", input_data.shape)

        predicted_price = float(model.predict(input_data)[0])
        print(f"🎯 Predicted price: {predicted_price:.2f}")

        change_pct = ((predicted_price - current_price) / current_price) * 100
        if change_pct > 0.5:
            decision = "📈 BUY"
        elif change_pct < -0.5:
            decision = "📉 SELL"
        else:
            decision = "⏸ HOLD"

        print(f"🔮 {symbol} | ₹{current_price:.2f} → ₹{predicted_price:.2f} ({change_pct:+.2f}%) → {decision}")

        summary.append({
            'Ticker': symbol,
            'Current Price': round(current_price, 2),
            'Predicted Price': round(predicted_price, 2),
            'Change (%)': round(change_pct, 2),
            'Action': decision
        })

    except Exception as e:
        print(f"❌ Error with {symbol}: {e}")
        import traceback
        traceback.print_exc()

# Print summary
if summary:
    print("\n📊 Summary of Predictions:\n")
    print(pd.DataFrame(summary).to_string(index=False))
