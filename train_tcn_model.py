import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import joblib

input_folder = 'processed'
model_folder = 'models'
os.makedirs(model_folder, exist_ok=True)

SEQUENCE_LENGTH = 50

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        stock_name = file.replace('_data.csv', '').lower()
        print(f"\n🚀 Training TCN model for {stock_name.upper()}...")

        filepath = os.path.join(input_folder, file)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df = df[['Close']].dropna()

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

        model = models.Sequential([
            layers.Input(shape=(SEQUENCE_LENGTH, 1)),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)

        model_path = os.path.join(model_folder, f"{stock_name}_tcn.h5")
        scaler_path = os.path.join(model_folder, f"{stock_name}_scaler.pkl")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        print(f"✅ Saved model to {model_path}")
        print(f"✅ Saved scaler to {scaler_path}")
