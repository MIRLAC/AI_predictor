import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

stocks = ["RELIANCE", "HDFCBANK", "NIFTY", "BANKNIFTY"]
lookback = 50

os.makedirs("models", exist_ok=True)

for stock in stocks:
    print(f"\n📈 Training EOD model for {stock}...")

    file_path = f"data/{stock}_intraday.csv"

    # ⬇️ Read file by skipping 2 metadata rows
    df = pd.read_csv(file_path, skiprows=2)

    # ✅ Manually assign proper column names
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

    # ✅ Convert 'Datetime' to actual datetime type
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # ✅ Clean NaNs
    df.dropna(inplace=True)

    # Extract just the date part for daily grouping
    df['Date'] = df['Datetime'].dt.date

    X, y = [], []
    for date in df['Date'].unique():
        day_df = df[df['Date'] == date]
        if len(day_df) < lookback:
            continue
        closes = day_df['Close'].values
        for i in range(lookback, len(closes)):
            X.append(closes[i-lookback:i])
            y.append(closes[-1])  # End of day close

    if not X:
        print(f"⚠️ Not enough data for {stock}, skipping.")
        continue

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, lookback)).reshape(-1, lookback, 1)
    y_scaled = scaler.fit_transform(y)

    # Save scaler
    joblib.dump(scaler, f"models/{stock.lower()}_eod_scaler.pkl")

    # Build and train model
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(lookback, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_scaled, y_scaled, epochs=30, batch_size=16, verbose=0)

    # Save model
    model.save(f"models/{stock.lower()}_eod.h5")
    print(f"✅ {stock} EOD model saved.")


