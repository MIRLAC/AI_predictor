import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import ta
import joblib
from plyer import notification
import os

# Supported stocks and tickers
stock_options = {
    "HDFC Bank": "HDFCBANK.NS",
    "Reliance": "RELIANCE.NS",
    "NIFTY 50": "^NSEI",
    "Bank NIFTY": "^NSEBANK"
}

# Get model file name from ticker
def get_model_filename(ticker):
    return ticker.replace('.', '').replace('^', '').lower() + '_rf_model.pkl'

def fetch_and_predict(ticker):
    model_file = get_model_filename(ticker)
    if not os.path.exists(model_file):
        return None, f"Model file '{model_file}' not found."

    try:
        model = joblib.load(model_file)
        data = yf.download(ticker, period='30d', interval='1h')
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        close = data['Close']

        # Calculate indicators
        data['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
        data['MACD'] = ta.trend.MACD(close).macd()
        data['SMA_50'] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        data.dropna(inplace=True)

        if data.empty:
            return None, "Not enough data to calculate indicators."

        latest = data.iloc[-1][['RSI', 'MACD', 'SMA_50']].values.reshape(1, -1)
        prediction = model.predict(latest)[0]

        # Output format
        up_down = "📈 UP (Buy)" if prediction == 1 else "📉 DOWN (Sell/Hold)"
        info = (
            f"Stock: {ticker}\n"
            f"Prediction: {up_down}\n\n"
            f"Latest Indicators:\n"
            f"RSI: {data['RSI'].iloc[-1]:.2f}\n"
            f"MACD: {data['MACD'].iloc[-1]:.2f}\n"
            f"SMA 50: {data['SMA_50'].iloc[-1]:.2f}\n"
            f"Close Price: {data['Close'].iloc[-1]:.2f}"
        )

        # Send notification
        notification.notify(
            title=f"Prediction for {ticker}",
            message=up_down,
            timeout=8
        )
        return info, None

    except Exception as e:
        return None, f"Error: {str(e)}"

def on_predict():
    stock_name = stock_var.get()
    ticker = stock_options[stock_name]
    result_text.set("Fetching data and predicting...\nPlease wait.")
    window.update()

    result, error = fetch_and_predict(ticker)
    if error:
        messagebox.showerror("Error", error)
        result_text.set("")
    else:
        result_text.set(result)

# GUI setup
window = tk.Tk()
window.title("📊 NIFTY 50 Stock Predictor")
window.geometry("450x400")
window.resizable(False, False)

ttk.Label(window, text="Select Stock and Predict", font=("Arial", 16)).pack(pady=10)

stock_var = tk.StringVar()
ticker_dropdown = ttk.Combobox(window, textvariable=stock_var, values=list(stock_options.keys()), state='readonly', font=("Arial", 12))
ticker_dropdown.pack(pady=10)
ticker_dropdown.current(0)

ttk.Button(window, text="Predict", command=on_predict).pack(pady=10)

result_text = tk.StringVar()
ttk.Label(window, textvariable=result_text, font=("Consolas", 11), justify='left').pack(pady=10, padx=10)

window.mainloop()
