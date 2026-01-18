"""
Multi-Source Sentiment Analysis System.
Combines RSS feeds, NewsAPI, Reddit, and Google Trends for maximum accuracy.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# RSS Feed parsing
import feedparser

# Google Trends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    TrendReq = None

# Reddit API
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None

from config.settings import (
    NEWS_API_KEY,
    REDDIT_CLIENT_ID, 
    REDDIT_CLIENT_SECRET, 
    REDDIT_USER_AGENT
)

# NewsAPI configuration
NEWS_API_URL = "https://newsapi.org/v2/everything"


# ==============================================
# RSS FEED SOURCES (Most Reliable)
# ==============================================

RSS_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_news": "https://www.moneycontrol.com/rss/latestnews.xml",
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
}

# Stock name mapping for relevance filtering
STOCK_KEYWORDS = {
    "RELIANCE": ["reliance", "ril", "jio", "mukesh ambani"],
    "TCS": ["tcs", "tata consultancy", "tata tech"],
    "INFY": ["infosys", "infy", "salil parekh"],
    "HDFCBANK": ["hdfc bank", "hdfc", "housing development"],
    "ICICIBANK": ["icici bank", "icici"],
    "SBIN": ["sbi", "state bank of india"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "ITC": ["itc limited", "itc hotels"],
    "KOTAKBANK": ["kotak mahindra", "kotak bank"],
    "LT": ["larsen", "l&t", "larsen toubro"],
    "NIFTY": ["nifty", "nifty50", "nifty 50", "index"],
    "BANKNIFTY": ["bank nifty", "banknifty", "banking index"],
    "FII": ["fii", "foreign institutional", "fiis"],
    "DII": ["dii", "domestic institutional", "diis"],
}

# Subreddits for Indian market sentiment
INDIAN_MARKET_SUBREDDITS = [
    "IndianStockMarket",
    "DalalStreetTalks", 
    "IndiaInvestments",
    "indianstreetbets",
]


class MultiSourceSentiment:
    """
    Multi-source sentiment aggregator for high-accuracy market analysis.
    
    Sources:
    1. RSS News Feeds (Moneycontrol, ET, LiveMint, Business Standard) - 30%
    2. NewsAPI (global financial news) - 25%
    3. Reddit (Indian market subreddits) - 25%
    4. Google Trends (retail interest proxy) - 20%
    """
    
    def __init__(self, sentiment_model: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
        """
        Initialize multi-source sentiment analyzer.
        
        Args:
            sentiment_model: HuggingFace model for sentiment classification
        """
        self.sentiment_model = sentiment_model
        self._sentiment_pipeline = None
        self._reddit_client = None
        self._pytrends = None
        
    def _get_sentiment_pipeline(self):
        """Lazy load sentiment pipeline."""
        if self._sentiment_pipeline is None:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline("sentiment-analysis", model=self.sentiment_model)
        return self._sentiment_pipeline
    
    def _get_reddit_client(self):
        """Initialize Reddit client if credentials available."""
        if self._reddit_client is None and PRAW_AVAILABLE:
            if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
                try:
                    self._reddit_client = praw.Reddit(
                        client_id=REDDIT_CLIENT_ID,
                        client_secret=REDDIT_CLIENT_SECRET,
                        user_agent=REDDIT_USER_AGENT
                    )
                except Exception as e:
                    st.warning(f"Reddit client init failed: {str(e)[:50]}")
        return self._reddit_client
    
    def _get_pytrends(self):
        """Initialize PyTrends client."""
        if self._pytrends is None and PYTRENDS_AVAILABLE:
            try:
                self._pytrends = TrendReq(hl='en-IN', tz=330)  # India timezone
            except Exception:
                pass
        return self._pytrends
    
    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Tuple of (label: 'positive'|'negative'|'neutral', confidence: 0-1)
        """
        if not text or len(text.strip()) < 10:
            return "neutral", 0.5
        
        try:
            pipeline = self._get_sentiment_pipeline()
            result = pipeline(text[:512])[0]
            return result['label'].lower(), result['score']
        except Exception:
            return "neutral", 0.5
    
    # ==============================================
    # RSS FEED COLLECTION (Most Reliable Source)
    # ==============================================
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def fetch_rss_news(_self, stock_symbol: str = None, max_articles: int = 50) -> List[Dict]:
        """
        Fetch news from multiple RSS feeds.
        
        Args:
            stock_symbol: Optional stock to filter for
            max_articles: Maximum articles to fetch per feed
        
        Returns:
            List of article dictionaries
        """
        all_articles = []
        keywords = []
        
        if stock_symbol:
            keywords = STOCK_KEYWORDS.get(stock_symbol.upper(), [stock_symbol.lower()])
        
        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    published = entry.get('published', entry.get('updated', ''))
                    link = entry.get('link', '')
                    
                    # Parse date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                    except Exception:
                        pub_date = datetime.now()
                    
                    # Filter by keyword if specified
                    if keywords:
                        text_lower = f"{title} {summary}".lower()
                        if not any(kw in text_lower for kw in keywords):
                            continue
                    
                    all_articles.append({
                        'source': feed_name,
                        'title': title,
                        'summary': summary[:500] if summary else '',
                        'date': pub_date,
                        'link': link,
                        'type': 'rss'
                    })
                    
            except Exception as e:
                continue  # Skip failed feeds silently
        
        # Sort by date (newest first)
        all_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return all_articles[:max_articles * 2]  # Return top articles
    
    # ==============================================
    # NEWSAPI COLLECTION (Global Financial News)
    # ==============================================
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def fetch_newsapi_articles(_self, stock_symbol: str = None, max_articles: int = 30) -> List[Dict]:
        """
        Fetch news from NewsAPI.
        
        Args:
            stock_symbol: Stock symbol to search for
            max_articles: Maximum articles to fetch
        
        Returns:
            List of article dictionaries
        """
        if not NEWS_API_KEY:
            return []
        
        all_articles = []
        
        # Build search query
        if stock_symbol:
            keywords = STOCK_KEYWORDS.get(stock_symbol.upper(), [stock_symbol])
            query = " OR ".join(keywords[:3])  # Use top 3 keywords
        else:
            query = "stock market India NSE"
        
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles
        }
        
        try:
            response = requests.get(NEWS_API_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                for article in articles:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    published = article.get("publishedAt", "")
                    source_name = article.get("source", {}).get("name", "NewsAPI")
                    
                    # Parse date
                    try:
                        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    except Exception:
                        pub_date = datetime.now()
                    
                    all_articles.append({
                        'source': f"NewsAPI: {source_name}",
                        'title': title,
                        'summary': description[:500] if description else '',
                        'date': pub_date,
                        'type': 'newsapi'
                    })
                    
        except Exception as e:
            pass  # Return empty list on error
        
        return all_articles
    
    # ==============================================
    # REDDIT COLLECTION
    # ==============================================
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_reddit_posts(_self, stock_symbol: str = None, max_posts: int = 30) -> List[Dict]:
        """
        Fetch posts from Indian market subreddits.
        
        Args:
            stock_symbol: Optional stock to filter for
            max_posts: Maximum posts per subreddit
        
        Returns:
            List of post dictionaries
        """
        reddit = _self._get_reddit_client()
        if reddit is None:
            return []
        
        keywords = []
        if stock_symbol:
            keywords = STOCK_KEYWORDS.get(stock_symbol.upper(), [stock_symbol.lower()])
        
        all_posts = []
        
        for subreddit_name in INDIAN_MARKET_SUBREDDITS:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                for post in subreddit.new(limit=max_posts):
                    title = post.title
                    text = post.selftext
                    
                    # Filter by keyword if specified
                    if keywords:
                        content_lower = f"{title} {text}".lower()
                        if not any(kw in content_lower for kw in keywords):
                            continue
                    
                    all_posts.append({
                        'source': f"r/{subreddit_name}",
                        'title': title,
                        'summary': text[:500] if text else '',
                        'date': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'comments': post.num_comments,
                        'type': 'reddit'
                    })
                    
            except Exception as e:
                continue  # Skip failed subreddits
        
        # Sort by score (engagement) and date
        all_posts.sort(key=lambda x: (x['score'], x['date']), reverse=True)
        
        return all_posts[:max_posts * 2]
    
    # ==============================================
    # GOOGLE TRENDS (Retail Interest Proxy)
    # ==============================================
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_google_trends(_self, stock_symbol: str, days: int = 7) -> Dict:
        """
        Fetch Google Trends data for a stock/keyword.
        
        Args:
            stock_symbol: Stock symbol to search
            days: Lookback period
        
        Returns:
            Dictionary with trend data and signals
        """
        pytrends = _self._get_pytrends()
        if pytrends is None:
            return {'available': False, 'signal': 0, 'trend': 'unknown'}
        
        try:
            # Build search keywords
            keywords = [stock_symbol]
            if stock_symbol in STOCK_KEYWORDS:
                keywords.extend(STOCK_KEYWORDS[stock_symbol][:2])  # Add up to 2 aliases
            
            # Limit to 5 keywords max
            keywords = keywords[:5]
            
            timeframe = f'now {days}-d'
            pytrends.build_payload(keywords, timeframe=timeframe, geo='IN')
            
            interest = pytrends.interest_over_time()
            
            if interest.empty:
                return {'available': False, 'signal': 0, 'trend': 'unknown'}
            
            # Calculate trend signal
            recent_avg = interest[keywords[0]].tail(2).mean()
            earlier_avg = interest[keywords[0]].head(2).mean()
            
            if earlier_avg > 0:
                trend_change = (recent_avg - earlier_avg) / earlier_avg * 100
            else:
                trend_change = 0
            
            # Determine trend direction
            if trend_change > 20:
                trend = 'rising_fast'
                signal = 0.3  # High interest can mean euphoria (be cautious)
            elif trend_change > 5:
                trend = 'rising'
                signal = 0.1
            elif trend_change < -20:
                trend = 'falling_fast'
                signal = -0.2  # Falling interest might mean capitulation
            elif trend_change < -5:
                trend = 'falling'
                signal = -0.1
            else:
                trend = 'stable'
                signal = 0
            
            return {
                'available': True,
                'signal': signal,
                'trend': trend,
                'change_pct': trend_change,
                'current_interest': recent_avg,
                'keywords': keywords
            }
            
        except Exception as e:
            return {'available': False, 'signal': 0, 'trend': 'error', 'error': str(e)[:50]}
    
    # ==============================================
    # COMBINED SENTIMENT ANALYSIS
    # ==============================================
    
    def analyze_all_sources(self, stock_symbol: str) -> Dict:
        """
        Fetch and analyze sentiment from all sources.
        
        Args:
            stock_symbol: Stock symbol to analyze
        
        Returns:
            Comprehensive sentiment analysis dictionary
        """
        results = {
            'stock': stock_symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'combined_sentiment': 0,
            'combined_label': 'neutral',
            'confidence': 0,
            'article_count': 0
        }
        
        # 1. RSS News (Weight: 30%)
        rss_articles = self.fetch_rss_news(stock_symbol, max_articles=30)
        rss_sentiments = []
        
        for article in rss_articles:
            text = f"{article['title']} {article['summary']}"
            label, score = self.analyze_text(text)
            
            # Convert to numerical
            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            else:
                sentiment_value = 0
            
            rss_sentiments.append({
                'text': article['title'][:100],
                'source': article['source'],
                'sentiment': label,
                'score': score,
                'value': sentiment_value,
                'date': article['date'].isoformat() if hasattr(article['date'], 'isoformat') else str(article['date'])
            })
        
        rss_avg = np.mean([s['value'] for s in rss_sentiments]) if rss_sentiments else 0
        
        results['sources']['rss'] = {
            'available': len(rss_sentiments) > 0,
            'count': len(rss_sentiments),
            'average_sentiment': float(rss_avg),
            'weight': 0.30,
            'articles': rss_sentiments[:10]
        }
        
        # 2. NewsAPI (Weight: 25%)
        newsapi_articles = self.fetch_newsapi_articles(stock_symbol, max_articles=20)
        newsapi_sentiments = []
        
        for article in newsapi_articles:
            text = f"{article['title']} {article['summary']}"
            label, score = self.analyze_text(text)
            
            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            else:
                sentiment_value = 0
            
            newsapi_sentiments.append({
                'text': article['title'][:100],
                'source': article['source'],
                'sentiment': label,
                'score': score,
                'value': sentiment_value
            })
        
        newsapi_avg = np.mean([s['value'] for s in newsapi_sentiments]) if newsapi_sentiments else 0
        
        results['sources']['newsapi'] = {
            'available': len(newsapi_sentiments) > 0,
            'count': len(newsapi_sentiments),
            'average_sentiment': float(newsapi_avg),
            'weight': 0.25,
            'articles': newsapi_sentiments[:10]
        }
        
        # 3. Reddit (Weight: 25%)
        reddit_posts = self.fetch_reddit_posts(stock_symbol, max_posts=20)
        reddit_sentiments = []
        
        for post in reddit_posts:
            text = f"{post['title']} {post['summary']}"
            label, score = self.analyze_text(text)
            
            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            else:
                sentiment_value = 0
            
            # Weight by engagement (score)
            engagement_boost = min(post.get('score', 0) / 100, 0.5)
            sentiment_value *= (1 + engagement_boost)
            
            reddit_sentiments.append({
                'text': post['title'][:100],
                'source': post['source'],
                'sentiment': label,
                'score': score,
                'value': sentiment_value,
                'engagement': post.get('score', 0)
            })
        
        reddit_avg = np.mean([s['value'] for s in reddit_sentiments]) if reddit_sentiments else 0
        
        results['sources']['reddit'] = {
            'available': len(reddit_sentiments) > 0,
            'count': len(reddit_sentiments),
            'average_sentiment': float(reddit_avg),
            'weight': 0.25,
            'posts': reddit_sentiments[:10]
        }
        
        # 4. Google Trends (Weight: 20%)
        trends_data = self.fetch_google_trends(stock_symbol)
        
        results['sources']['google_trends'] = {
            'available': trends_data.get('available', False),
            'trend': trends_data.get('trend', 'unknown'),
            'signal': trends_data.get('signal', 0),
            'change_pct': trends_data.get('change_pct', 0),
            'weight': 0.20
        }
        
        # ==============================================
        # WEIGHTED ENSEMBLE CALCULATION
        # ==============================================
        
        weights = {
            'rss': 0.30,
            'newsapi': 0.25,
            'reddit': 0.25,
            'trends': 0.20
        }
        
        # Adjust weights based on availability
        total_available_weight = 0
        if results['sources']['rss']['available']:
            total_available_weight += weights['rss']
        if results['sources']['newsapi']['available']:
            total_available_weight += weights['newsapi']
        if results['sources']['reddit']['available']:
            total_available_weight += weights['reddit']
        if results['sources']['google_trends']['available']:
            total_available_weight += weights['trends']
        
        if total_available_weight == 0:
            results['combined_sentiment'] = 0
            results['combined_label'] = 'neutral'
            results['confidence'] = 0
            return results
        
        # Normalize weights
        adjusted_weights = {}
        source_mapping = {'rss': 'rss', 'newsapi': 'newsapi', 'reddit': 'reddit', 'trends': 'google_trends'}
        for key, weight in weights.items():
            source_key = source_mapping[key]
            if results['sources'].get(source_key, {}).get('available', False):
                adjusted_weights[key] = weight / total_available_weight
            else:
                adjusted_weights[key] = 0
        
        # Calculate combined sentiment
        combined = 0
        combined += rss_avg * adjusted_weights.get('rss', 0)
        combined += newsapi_avg * adjusted_weights.get('newsapi', 0)
        combined += reddit_avg * adjusted_weights.get('reddit', 0)
        combined += trends_data.get('signal', 0) * adjusted_weights.get('trends', 0)
        
        results['combined_sentiment'] = float(combined)
        
        # Determine label
        if combined > 0.15:
            results['combined_label'] = 'bullish'
        elif combined > 0.05:
            results['combined_label'] = 'slightly_bullish'
        elif combined < -0.15:
            results['combined_label'] = 'bearish'
        elif combined < -0.05:
            results['combined_label'] = 'slightly_bearish'
        else:
            results['combined_label'] = 'neutral'
        
        # Confidence based on article count and agreement
        total_articles = len(rss_sentiments) + len(newsapi_sentiments) + len(reddit_sentiments)
        results['article_count'] = total_articles
        
        # Higher confidence with more articles and stronger sentiment
        results['confidence'] = min(total_articles / 25, 1.0) * min(abs(combined) * 5, 1.0)
        
        return results
    
    def get_sentiment_for_model(self, stock_symbol: str) -> Dict:
        """
        Get sentiment features formatted for ML model input.
        
        Args:
            stock_symbol: Stock symbol
        
        Returns:
            Dictionary with sentiment features for model
        """
        analysis = self.analyze_all_sources(stock_symbol)
        
        # Convert to model features
        features = {
            'sentiment_score': analysis['combined_sentiment'],
            'sentiment_label': analysis['combined_label'],
            'sentiment_confidence': analysis['confidence'],
            'rss_sentiment': analysis['sources'].get('rss', {}).get('average_sentiment', 0),
            'reddit_sentiment': analysis['sources'].get('reddit', {}).get('average_sentiment', 0),
            'trends_signal': analysis['sources'].get('google_trends', {}).get('signal', 0),
            'article_count': analysis['article_count'],
            'data_quality': 'high' if analysis['article_count'] > 10 else 'medium' if analysis['article_count'] > 5 else 'low'
        }
        
        return features


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

_multi_sentiment_instance = None

def get_multi_sentiment_analyzer() -> MultiSourceSentiment:
    """Get singleton instance of multi-source sentiment analyzer."""
    global _multi_sentiment_instance
    if _multi_sentiment_instance is None:
        _multi_sentiment_instance = MultiSourceSentiment()
    return _multi_sentiment_instance


def analyze_stock_sentiment(stock_symbol: str) -> Dict:
    """
    Convenience function to analyze sentiment for a stock.
    
    Args:
        stock_symbol: Stock symbol (e.g., "RELIANCE", "TCS")
    
    Returns:
        Comprehensive sentiment analysis
    """
    analyzer = get_multi_sentiment_analyzer()
    return analyzer.analyze_all_sources(stock_symbol)


def get_sentiment_features(stock_symbol: str) -> Dict:
    """
    Get sentiment features for ML model integration.
    
    Args:
        stock_symbol: Stock symbol
    
    Returns:
        Dictionary of sentiment features
    """
    analyzer = get_multi_sentiment_analyzer()
    return analyzer.get_sentiment_for_model(stock_symbol)
