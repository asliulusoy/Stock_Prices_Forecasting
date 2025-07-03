import numpy as np
import pandas as pd
from datetime import timedelta
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf

# Load once
model = load_model("Numerical_Analysis/ndx_lstm.h5")
scaler = joblib.load("Numerical_Analysis/scaler.save")

def get_processed_data(end_date, lookback=60):
    ticker = "^NDX"
    end_date = pd.to_datetime(end_date)
    start_date = end_date - timedelta(days=lookback * 2)

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
    close_data = data['Close']

    if len(close_data) < lookback:
        raise ValueError("Not enough data to make prediction")

    close_data = close_data[-lookback:]  # Last 60 days
    scaled_data = scaler.transform(close_data.values.reshape(-1, 1))
    X_input = np.reshape(scaled_data, (1, lookback, 1))
    return X_input

def predict_price(date_str):
    X_input = get_processed_data(date_str)
    scaled_pred = model.predict(X_input)
    actual_pred = scaler.inverse_transform(scaled_pred)
    return float(actual_pred[0][0])

def generate_lstm_predictions(start_date, end_date, lstm_model):
    dates = pd.date_range(start_date, end_date, freq="B")  # business days
    predictions = []

    for date in dates:
        try:
            pred = lstm_model.predict_price(date.strftime("%Y-%m-%d"))
            predictions.append((date.date(), pred))
        except:
            predictions.append((date.date(), None))

    return pd.DataFrame(predictions, columns=["date", "lstm_pred"])
