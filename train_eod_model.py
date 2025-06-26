import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import joblib
import os

stock = "RELIANCE"
file_path = f"data/{stock}_intraday.csv"
df = pd.read_csv(file_path, parse_dates=['Datetime'])

df['Date'] = df['Datetime'].dt.date
lookback = 50  # Number of 5-min candles to use

X, y = [], []
for day in df['Date'].unique():
    day_df = df[df['Date'] == day]
    if len(day_df) < lookback:
        continue
    close_prices = day_df['Close'].values
    for i in range(lookback, len(close_prices)):
        X.append(close_prices[i-lookback:i])
        y.append(close_prices[-1])  # EOD Close

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, lookback)).reshape(-1, lookback, 1)
y_scaled = scaler.fit_transform(y)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, f"models/{stock.lower()}_eod_scaler.pkl")

# Build model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(lookback, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y_scaled, epochs=30, batch_size=16)

# Save model
model.save(f"models/{stock.lower()}_eod.h5")
print("✅ EOD model saved.")
