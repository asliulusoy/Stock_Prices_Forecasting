import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_news(query="NASDAQ", from_date=None, to_date=None, api_key="-"):
    url = "https://newsapi.org/v2/everything"
    if not from_date:
        from_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.today().strftime('%Y-%m-%d')

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": api_key,
        "pageSize": 100
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        articles = data.get("articles", [])
        
        if not articles:
            print(f"[INFO] No articles found for {from_date}")
            return pd.DataFrame()

        records = []
        for a in articles:
            if not all(k in a for k in ["publishedAt", "title", "description", "content", "source"]):
                continue
            records.append({
                "publishedAt": a["publishedAt"],
                "title": a["title"],
                "description": a["description"],
                "content": a["content"],
                "source": a["source"]["name"] if isinstance(a["source"], dict) else a["source"]
            })

        news_df = pd.DataFrame(records)
        if "publishedAt" not in news_df.columns or news_df.empty:
            print(f"[WARN] No usable articles for {from_date}")
            return pd.DataFrame()

        news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"]).dt.date
        return news_df

    except Exception as e:
        print(f"[ERROR] Failed fetching news: {e}")
        return pd.DataFrame()
