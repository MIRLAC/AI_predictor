import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path, sequence_length=50):
    df = pd.read_csv(file_path)

    # Ensure proper datetime and sorting
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.sort_values('Datetime', inplace=True)

    # Use only these features (can be expanded later)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[features]

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 3])  # 3 = index of 'Close'

    X, y = np.array(X), np.array(y)

    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler
