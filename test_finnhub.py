import requests

API_KEY = "d1eltq1r01qghj41lbjgd1eltq1r01qghj41lbk0"  # Replace with your actual Finnhub API key
symbol = "NSE:RELIANCE"

url = "https://finnhub.io/api/v1/quote"
params = {
    "symbol": symbol,
    "token": API_KEY
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    print(f"✅ Current Price of {symbol}: ₹{data['c']}")
else:
    print("❌ Failed to fetch data:", response.text)
