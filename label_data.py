import pandas as pd
import os

features_folder = 'features'
labeled_folder = 'labeled'
os.makedirs(labeled_folder, exist_ok=True)

def label_data(df):
    df['Signal'] = 0  # Default Hold

    # Apply simple rule-based labeling
    df.loc[(df['RSI'] < 30) & (df['MACD'] > 0), 'Signal'] = 1    # Buy
    df.loc[(df['RSI'] > 70) & (df['MACD'] < 0), 'Signal'] = -1   # Sell

    return df

for filename in os.listdir(features_folder):
    if filename.endswith('.csv'):
        print(f"Labeling {filename}...")
        path = os.path.join(features_folder, filename)

        df = pd.read_csv(path, parse_dates=True, index_col=0)

        # Add signals
        df = label_data(df)

        # Save to labeled folder
        save_path = os.path.join(labeled_folder, filename)
        df.to_csv(save_path)
        print(f"Saved: {save_path}")

print("✅ Labeling complete.")
