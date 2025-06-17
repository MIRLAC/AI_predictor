import pandas as pd
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def add_indicators(df):
    # Ensure 'Close' is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Drop any rows where Close is NaN after conversion
    df.dropna(subset=['Close'], inplace=True)

    # Compute indicators
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['MACD'] = MACD(df['Close']).macd()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    
    return df.dropna()

input_folder = 'data'
output_folder = 'processed'
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        full_path = os.path.join(input_folder, file)
        df = pd.read_csv(full_path, index_col=0, parse_dates=True)

        # Optional: Ensure all key columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = add_indicators(df)
        df.to_csv(os.path.join(output_folder, file))
        print(f"✅ Processed and saved: {file}")
