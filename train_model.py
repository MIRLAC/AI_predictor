import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import ta

# Step 1: Download historical data
ticker = 'HDFCBANK.NS'
data = yf.download(ticker, period='90d', interval='1h')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
close_series = data['Close'].squeeze()

# Step 2: Add technical indicators
data['RSI'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
data['MACD'] = ta.trend.MACD(close_series).macd()
data['SMA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()

# Step 3: Drop NA and define target (1 if next close > current close, else 0)
data.dropna(inplace=True)
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

# Step 4: Prepare features and labels
features = data[['RSI', 'MACD', 'SMA_50']]
labels = data['Target']

# Step 5: Split and train
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained. Accuracy: {accuracy:.2f}")

# Step 7: Save the model
joblib.dump(model, 'hdfcbankns_rf_model.pkl')
print("Model saved as 'hdfcbankns_rf_model.pkl'")
