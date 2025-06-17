import yfinance as yf
import pandas as pd

# List of stocks you want to support
tickers = ['HDFCBANK.NS', 'RELIANCE.NS', '^NSEI', '^NSEBANK']

for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, period='60d', interval='1h')
    data.dropna(inplace=True)

    # Optional: Save as CSV for reuse
    data.to_csv(f"data/{ticker.replace('^', '').replace('.', '')}_data.csv")

print("✅ Data collection completed.")
