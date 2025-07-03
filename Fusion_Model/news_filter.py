import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

NDX_KEYWORDS = [
    "NASDAQ", "NDX", "tech sector", "big tech", "FAANG", "stock index",
    "technology stocks", "semiconductors", "growth stocks", "NASDAQ-100",
    "Apple", "Microsoft", "Amazon", "Meta", "Google", "NVIDIA", "Tesla",
    "AI stocks", "chipmakers", "tech rally", "market trend", "earnings report",
    "stock market", "Wall Street", "Fed", "interest rate", "inflation", "SP500"
]

def is_relevant_news(text: str, threshold: float = 0.25) -> bool:
    if not text:
        return False

    text_lower = text.lower()

    # Fast keyword match (early exit if obvious match)
    for keyword in NDX_KEYWORDS:
        if keyword.lower() in text_lower:
            return True

    # Semantic similarity for subtle relevance
    vectorizer = TfidfVectorizer().fit([text] + NDX_KEYWORDS)
    vectors = vectorizer.transform([text] + NDX_KEYWORDS)
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).mean()

    return similarity > threshold

def filter_news(df):
    df["combined_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["is_relevant"] = df["combined_text"].apply(is_relevant_news)
    return df[df["is_relevant"]].drop(columns=["combined_text", "is_relevant"])
