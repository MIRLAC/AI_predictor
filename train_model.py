import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

input_folder = 'processed'
model_folder = 'models'
os.makedirs(model_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith('.csv'):
        print(f"\n📈 Training for {file}...")
        filepath = os.path.join(input_folder, file)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        # Label: Predict next day's close price
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Features
        X = df[['RSI', 'MACD', 'SMA_50']]
        y = df['Target'].squeeze()  # Ensures it's 1D



        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Regression Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 Mean Squared Error: {mse:.2f}")
        print(f"✅ R² Score: {r2:.2f}")

        # Save model with matching name
        model_name = file.replace('_data.csv', '').lower() + '_rf_model.pkl'
        joblib.dump(model, os.path.join(model_folder, model_name))
        print(f"📦 Model saved as {model_name}")
