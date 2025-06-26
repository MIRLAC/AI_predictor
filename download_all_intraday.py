import yfinance as yf
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Define your stocks and tickers
stocks = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}

# Download and save intraday data
for stock, ticker in stocks.items():
    print(f"📥 Downloading {stock} (Ticker: {ticker})...")
    df = yf.download(ticker, interval="5m", period="30d", prepost=False)
    
    if df.empty:
        print(f"⚠️ Failed to download data for {stock}. Skipping.")
        continue

    file_path = f"data/{stock}_intraday.csv"
    df.to_csv(file_path)
    print(f"✅ Saved to {file_path}\n")

print("🎉 All data downloaded.")
