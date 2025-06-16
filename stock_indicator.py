import yfinance as yf

# Replace 'AAPL' with any Indian stock, like HDFCBANK.NS or RELIANCE.NS
ticker = 'HDFCBANK.NS'
data = yf.download(ticker, start='2020-01-01', end='2024-12-31')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(data.head())
