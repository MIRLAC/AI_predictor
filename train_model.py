import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

input_folder = 'processed'
model_folder = 'models'
os.makedirs(model_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        print(f"\n📈 Training for {file}...")
        filepath = os.path.join(input_folder, file)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        # Label: 1 if next close > current close, else 0
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # Features
        X = df[['RSI', 'MACD', 'SMA_50']]
        y = df['Target']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Model Accuracy: {acc:.2f}")

        # Save model with matching name
        model_name = file.replace('_data.csv', '').lower() + '_rf_model.pkl'
        joblib.dump(model, os.path.join(model_folder, model_name))
        print(f"📦 Model saved as {model_name}")
