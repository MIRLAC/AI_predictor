import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib

# Parameters
SEQ_LEN = 50  # Use last 50 candles to predict next one
EPOCHS = 10
BATCH_SIZE = 32

def load_and_preprocess(csv_file):
    df = pd.read_csv(csv_file, parse_dates=True, index_col=0)

    # Use Close only or more features if needed
    features = df[['Close']].values

    # Normalize
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # Build sequences
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i][0])  # Predict next close

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(csv_file, model_path, scaler_path):
    X, y, scaler = load_and_preprocess(csv_file)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved model to {model_path} and scaler to {scaler_path}")

# Train all 4 models
stock_files = {
    "RELIANCE": "processed/RELIANCENS_data.csv",
    "HDFCBANK": "processed/HDFCBANKNS_data.csv",
    "NIFTY": "processed/NSEI_data.csv",
    "BANKNIFTY": "processed/NSEBANK_data.csv",
}

for name, path in stock_files.items():
    print(f"\n🚀 Training model for {name}")
    train_model(path, f"models/{name}_lstm.h5", f"models/{name}_scaler.pkl")
