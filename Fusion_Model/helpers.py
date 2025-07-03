import yfinance as yf
import pandas as pd
from datetime import timedelta
from Numerical_Analysis.lstm_model import predict_price
from News_Analysis.newsapi_fetch_news import fetch_news
from News_Analysis.news_filter import filter_news
from News_Analysis.sentiment_analysis import apply_sentiment


def get_ndx_prices(start_date, end_date):
    df = yf.download("^NDX", start=start_date, end=end_date)
    df = df.reset_index()  # Ensure 'Date' is a column
    df["date"] = pd.to_datetime(df["Date"]) 
    return df[["date", "Close"]].rename(columns={"Close": "true_price"})


def generate_lstm_predictions(start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq="B")
    predictions = []
    for date in dates:
        try:
            pred = predict_price(date.strftime("%Y-%m-%d"))
            predictions.append((date, pred))
        except:
            predictions.append((date.date(), None))
    return pd.DataFrame(predictions, columns=["date", "lstm_pred"])

def collect_sentiment_series(start_date, end_date, api_key):
    sentiment_data = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        news = fetch_news(from_date=date_str, to_date=date_str, api_key=api_key)
        if not news.empty:
            news = filter_news(news)
            news = apply_sentiment(news)
            avg_sentiment = news["sentiment_score"].mean()
            sentiment_data.append((current, avg_sentiment))
        else:
            sentiment_data.append((current.date(), 0))
        current += timedelta(days=1)
    return pd.DataFrame(sentiment_data, columns=["date", "sentiment_score"])



def build_fusion_training_data(prices_df, lstm_df, sentiment_df):
    # Flatten MultiIndex if needed
    if isinstance(prices_df.columns, pd.MultiIndex):
        prices_df.columns = ['_'.join(filter(None, col)).strip() for col in prices_df.columns]

    # Ensure 'date' column is datetime64[ns] in all DataFrames
    for df in [prices_df, lstm_df, sentiment_df]:
        df["date"] = pd.to_datetime(df["date"])

    print("Prices DF:\n", prices_df.head())
    print("LSTM DF:\n", lstm_df.head())
    print("Sentiment DF:\n", sentiment_df.head())

    # Merge
    df = prices_df.merge(lstm_df, on="date", how="inner")
    df = df.merge(sentiment_df, on="date", how="left")
    df = df.dropna()

    return df["lstm_pred"].tolist(), df["sentiment_score"].tolist(), df["true_price_^NDX"].tolist()



