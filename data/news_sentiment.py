"""
News fetching and sentiment analysis.
Uses NewsAPI for news and DistilRoBERTa-Financial for sentiment analysis.
"""

import re
import requests
import streamlit as st
from transformers import pipeline

from config.settings import NEWS_API_KEY, DataConfig


# Sentiment model configuration
# DistilRoBERTa-Financial: 98.2% accuracy, 2x faster than FinBERT
SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Lazy-loaded sentiment pipeline
_sentiment_pipeline = None


def _get_sentiment_pipeline():
    """Get or initialize the DistilRoBERTa-Financial sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    return _sentiment_pipeline


def get_news(stock_symbol: str) -> list:
    """
    Fetch news articles for a stock from NewsAPI.
    
    Args:
        stock_symbol: Stock symbol (e.g., "RELIANCE")
    
    Returns:
        List of news articles
    """
    if not NEWS_API_KEY:
        st.warning("⚠️ NEWS_API_KEY not configured. Set environment variable.")
        return []
    
    query = DataConfig.STOCK_NAME_MAPPING.get(stock_symbol, stock_symbol)
    params = {
        "q": query, 
        "apiKey": NEWS_API_KEY, 
        "language": "en", 
        "sortBy": "publishedAt"
    }
    
    try:
        response = requests.get(DataConfig.NEWS_API_URL, params=params, timeout=10)
        if response.status_code != 200:
            st.warning(f"News API returned status {response.status_code}")
            return []
        return response.json().get("articles", [])
    except Exception as e:
        st.warning(f"Could not fetch news: {str(e)}")
        return []


def analyze_sentiment(text: str) -> tuple:
    """
    Analyze sentiment of text using FinBERT.
    
    Args:
        text: Text to analyze
    
    Returns:
        Tuple of (sentiment_label, confidence_score)
    """
    if not text:
        return "neutral", 0.0
    
    pipeline = _get_sentiment_pipeline()
    result = pipeline(text[:512])[0]  # FinBERT has 512 token limit
    return result['label'], result['score']


def filter_relevant_news(news_articles: list, stock_name: str) -> list:
    """
    Filter news articles to keep only those relevant to the stock.
    
    Args:
        news_articles: List of news articles from API
        stock_name: Stock name to filter by
    
    Returns:
        Filtered list of relevant articles
    """
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        if title and re.search(stock_name, title, re.IGNORECASE):
            filtered_articles.append(article)
    return filtered_articles


def analyze_news_sentiment(news_articles: list, stock_symbol: str) -> dict:
    """
    Analyze sentiment of all relevant news articles grouped by date.
    
    Args:
        news_articles: List of news articles
        stock_symbol: Stock symbol for filtering
    
    Returns:
        Dictionary with date keys and list of (sentiment, score) tuples
    """
    filtered_news = filter_relevant_news(news_articles, stock_symbol)
    daily_sentiment = {}
    
    for article in filtered_news:
        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
        sentiment, score = analyze_sentiment(text)
        date = article.get("publishedAt", "")[0:10]
        
        if date in daily_sentiment:
            daily_sentiment[date].append((sentiment, score))
        else:
            daily_sentiment[date] = [(sentiment, score)]
    
    return daily_sentiment


def get_sentiment_summary(daily_sentiment: dict) -> dict:
    """
    Generate a summary of sentiment data.
    
    Args:
        daily_sentiment: Dictionary of daily sentiments
    
    Returns:
        Dictionary with sentiment statistics
    """
    if not daily_sentiment:
        return {
            "text": "Neutral (No recent news)",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "sentiment_ratio": 0.5
        }
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for date_sentiments in daily_sentiment.values():
        for label, _ in date_sentiments:
            if label == 'positive':
                positive_count += 1
            elif label == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
    
    total = positive_count + negative_count
    
    if total > 0:
        sentiment_ratio = positive_count / total
        if sentiment_ratio > 0.6:
            text = f"Bullish ({positive_count}/{total} positive articles)"
        elif sentiment_ratio < 0.4:
            text = f"Bearish ({negative_count}/{total} negative articles)"
        else:
            text = f"Mixed ({positive_count} positive, {negative_count} negative)"
    else:
        sentiment_ratio = 0.5
        text = "Neutral (No sentiment data)"
    
    return {
        "text": text,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "sentiment_ratio": sentiment_ratio
    }
