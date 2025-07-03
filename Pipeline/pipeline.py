import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import os
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = "lstm_model.h5"

def load_data():
    data = yf.download(tickers='^NDX', start = '2024-09-30',end = '2025-06-02')
    data.columns = [f"{col[0]}" for col in data.columns]
    return data


def engineer_features(data,target_type):
    data=data.copy()
    data['RSI']=ta.rsi(data.Close, length=15)
    data['EMAF']=ta.ema(data.Close, length=20)
    data['EMAM']=ta.ema(data.Close, length=100)
    data['EMAS']=ta.ema(data.Close, length=150)
    indicator_cols = ['RSI', 'EMAF', 'EMAM', 'EMAS']
    data[indicator_cols] = data[indicator_cols].fillna(method='bfill')
    data['Adj Close'] = data[('Close')]


    if target_type == 'daily':
        data['Target'] = data['Adj Close'] - data['Open']
        data['Target'] = data['Target'].shift(-1)
        data['TargetClass'] = [1 if data.Target[i]>0 else 0 for i in range(len(data))]
        data['TargetNextClose'] = data['Adj Close'].shift(-1)
        target_column = 'TargetClass'
        horizon = 1
        data.dropna(inplace=True)
        data.reset_index(inplace = True)

    elif target_type == 'weekly':
        data['TargetWeek'] = data['Adj Close'].shift(-5) - data['Adj Close']
        data['TargetWeekClass'] = [1 if val > 0 else 0 for val in data['TargetWeek']]
        target_column = 'TargetWeekClass'
        horizon = 5
        data.dropna(inplace=True)
        data.reset_index(inplace = True)

    elif target_type == 'monthly':
        data['TargetMonth'] = data['Adj Close'].shift(-20) - data['Adj Close']
        data['TargetMonthClass'] = [1 if val > 0 else 0 for val in data['TargetMonth']]
        target_column = 'TargetMonthClass'
        horizon = 20
        data.dropna(inplace=True)
        data.reset_index(inplace = True)
        
    else:
        raise ValueError("Invalid target_type. Use 'daily', 'weekly', or 'monthly'.")
    
    target_cols_to_drop = [col for col in data.columns if 'Target' in col and target_column not in col]
    data.drop(columns=target_cols_to_drop, inplace=True)
    return data, target_column, horizon


def merge_sentiment(data, sentiment_path):
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    sentiment_df.rename(columns={'date': 'Date'}, inplace=True)
    data = pd.merge(data, sentiment_df, how='left', on='Date')
    data[['avg_sentiment', 'avg_score', 'article_count']] = data[['avg_sentiment', 'avg_score', 'article_count']].fillna(method='ffill')
    return data


def prepare_dataset(data, target_column, backcandles=15, horizon=1):
    data = data.iloc[:-horizon]  # Remove future-leaking samples
    data = data[data['Date'] > pd.to_datetime('2024-12-31')].reset_index(drop=True)
    feature_columns = ['High', 'Low', 'Open', 'Volume', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'avg_sentiment']
    X, y = [], data[target_column].values[backcandles:]
    for j in range(len(feature_columns)):
        X.append([data[feature_columns].iloc[i-backcandles:i, j].values for i in range(backcandles, len(data))])
    X = np.moveaxis(X, [0], [2])
    return np.array(X), y, feature_columns


def scale_dataset(X, scaler=None):
    orig_shape = X.shape
    X_reshaped = X.reshape(-1, orig_shape[2])
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(orig_shape)
    return X_scaled, scaler

def scale_and_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)
    X_train_scaled, scaler = scale_dataset(X_train)
    X_test_scaled, _ = scale_dataset(X_test, scaler)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(100, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(50)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict(X_test).flatten()
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold = {best_t:.2f}, F1 = {max(f1_scores):.4f}")

    # Classification
    y_pred = (y_prob >= best_t).astype(int)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, label=f'AP = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Distribution of Labels
    unique, counts = np.unique(y_test, return_counts=True)
    print("Label distribution in test set:", dict(zip(unique, counts)))
    return best_t

def prepare_full_dataset_for_prediction(data, backcandles=15):
    # Prepare the features without target, for prediction on full dataset (starting from date index after backcandles)
    feature_columns = ['High', 'Low', 'Open', 'Volume', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'avg_sentiment']
    data = data.reset_index(drop=True)
    X = []
    for j in range(len(feature_columns)):
        X.append([data[feature_columns].iloc[i-backcandles:i, j].values for i in range(backcandles, len(data))])
    X = np.moveaxis(X, [0], [2])
    return np.array(X)

def predict_direction(model, scaler, data, backcandles=15, threshold=0.5):
    X_full = prepare_full_dataset_for_prediction(data, backcandles)
    X_full_scaled, _ = scale_dataset(X_full, scaler)
    y_prob = model.predict(X_full_scaled).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    # Align prediction dates
    dates = data['Date'].iloc[backcandles:].reset_index(drop=True)
    return pd.DataFrame({'Date': dates, 'Prediction': y_pred})


def main():
    raw_data = load_data()

    sentiment_path = "data/cleaned_scores.csv"
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["date"])
    sentiment_df.rename(columns={'date': 'Date'}, inplace=True)

    target_types = ['daily', 'weekly', 'monthly']
    models = {}
    scalers = {}
    thresholds = {}
    data_dict = {}

    # Store data with engineered features for each target
    for target_type in target_types:
        print(f"\n--- Processing target: {target_type} ---")
        data, target_column, horizon = engineer_features(raw_data, target_type)
        data = pd.merge(data, sentiment_df, how='left', on='Date')
        data[['avg_sentiment', 'avg_score', 'article_count']] = data[['avg_sentiment', 'avg_score', 'article_count']].fillna(method='ffill')
        
        X, y, _ = prepare_dataset(data, target_column=target_column, backcandles=15, horizon=horizon)
        X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)
        model = build_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1, shuffle=True)

        best_t = evaluate_model(model, X_test, y_test)

        os.makedirs("models", exist_ok=True)
        model.save(f"models/lstm_model_{target_type}.h5")

        models[target_type] = model
        scalers[target_type] = scaler
        thresholds[target_type] = best_t
        data_dict[target_type] = data

    # Now produce predictions for the date range 2025-01-01 to 2025-06-01
    start_date = pd.to_datetime("2025-01-01")
    end_date = pd.to_datetime("2025-06-01")

    # Use the daily data as base since it has daily sentiment and actual directions
    base_data = data_dict['daily']
    mask = (base_data['Date'] >= start_date) & (base_data['Date'] <= end_date)
    base_data_period = base_data.loc[mask].reset_index(drop=True)

    # Prepare predictions for each model over the full dataset period (need to include dates outside mask for backcandles)
    # We'll use the full data (including before 2025-01-01) for prediction to have valid sequences.
    results_df = base_data_period[['Date']].copy()
    results_df['daily_sentiment_direction'] = base_data_period['TargetClass']

    for target_type in target_types:
        data_full = data_dict[target_type]
        pred_df = predict_direction(models[target_type], scalers[target_type], data_full, backcandles=15, threshold=thresholds[target_type])
        # Filter predictions for date range
        pred_df = pred_df[(pred_df['Date'] >= start_date) & (pred_df['Date'] <= end_date)].reset_index(drop=True)
        colname = f"{target_type}_direction_prediction"
        results_df = results_df.merge(pred_df[['Date', 'Prediction']], on='Date', how='left')
        results_df.rename(columns={'Prediction': colname}, inplace=True)

    # Add sentiment info from sentiment_df (article_count, avg_score)
    sentiment_filtered = sentiment_df[(sentiment_df['Date'] >= start_date) & (sentiment_df['Date'] <= end_date)].reset_index(drop=True)
    sentiment_filtered = sentiment_filtered[['Date', 'article_count', 'avg_score']]
    results_df = results_df.merge(sentiment_filtered, on='Date', how='left')

    # Save CSV
    output_path = "prediction_summary_2025-01-01_to_2025-06-01.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved prediction summary to {output_path}")

if __name__ == "__main__":
    main()