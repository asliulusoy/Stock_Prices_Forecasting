from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    if not text:
        return 0
    score = analyzer.polarity_scores(text)
    return score["compound"]

def apply_sentiment(df):
    if df.empty or "combined_text" not in df.columns:
        print("[WARNING] No data or missing 'combined_text' column for sentiment analysis.")
        print("News DataFrame columns:", df.columns)
        return df
    df["sentiment_score"] = df["combined_text"].apply(get_sentiment_score)
    return df


def aggregate_sentiment_by_date(df):
    return df.groupby("publishedAt")["sentiment_score"].mean().reset_index()