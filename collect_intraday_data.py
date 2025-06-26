import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

symbols = {
    "HDFCBANK": "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS",
    "NIFTY_50": "^NSEI",
    "BANK_NIFTY": "^NSEBANK"
}

end_time = datetime.now()
start_time = end_time - timedelta(days=7)  # 1 week of 5-minute data

for name, symbol in symbols.items():
    df = yf.download(tickers=symbol, start=start_time, end=end_time, interval="5m", progress=False)
    if not df.empty:
        df.to_csv(f"{name}_5min.csv")
        print(f"✅ Saved: {name}_5min.csv")
    else:
        print(f"⚠️ No data for {name}")
