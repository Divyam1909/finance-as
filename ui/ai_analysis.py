"""
AI Analysis utilities.
Gemini integration and fallback analysis generation.
"""

import streamlit as st

from config.settings import GEMINI_API_KEY, ModelConfig, TradingConfig

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


def initialize_gemini():
    """
    Initialize Gemini API with safety settings.
    
    Returns:
        Gemini model object or None if unavailable
    """
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("⚠️ google-generativeai library not installed")
        return None
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
        st.sidebar.warning("⚠️ No Gemini API key configured")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.sidebar.error(f"Gemini init error: {e}")
        return None


def generate_gemini_analysis(stock_symbol: str, current_price: float, 
                             predicted_prices, metrics: dict, fundamentals: dict,
                             sentiment_summary: dict, technical_indicators: dict,
                             volatility_data, fusion_weights: dict = None) -> str:
    """
    Generate comprehensive AI analysis using Google Gemini.
    
    Args:
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        predicted_prices: DataFrame with predicted prices
        metrics: Dictionary with model metrics (accuracy, rmse)
        fundamentals: Dictionary with fundamental data
        sentiment_summary: Dictionary with sentiment data
        technical_indicators: Dictionary with technical indicators
        volatility_data: Volatility value
        fusion_weights: Dynamic fusion model weights (optional)
    
    Returns:
        Markdown-formatted analysis string
    """
    model = initialize_gemini()
    if model is None:
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices, 
                                          metrics, sentiment_summary, technical_indicators)
    
    # Calculate key metrics for prompt
    if not predicted_prices.empty:
        price_forecast_end = predicted_prices['Predicted Price'].iloc[-1]
        forecast_days = len(predicted_prices)
    else:
        price_forecast_end = current_price
        forecast_days = 0
    
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    
    # Prepare sentiment summary text
    sentiment_text = "Neutral (No recent news)"
    if sentiment_summary:
        positive_count = sum(1 for s in sentiment_summary.values() for label, _ in s if label == 'positive')
        negative_count = sum(1 for s in sentiment_summary.values() for label, _ in s if label == 'negative')
        total = positive_count + negative_count
        if total > 0:
            sentiment_ratio = positive_count / total
            if sentiment_ratio > 0.6:
                sentiment_text = f"Bullish ({positive_count}/{total} positive articles)"
            elif sentiment_ratio < 0.4:
                sentiment_text = f"Bearish ({negative_count}/{total} negative articles)"
            else:
                sentiment_text = f"Mixed ({positive_count} positive, {negative_count} negative)"
    
    # Fusion weights summary
    fusion_text = "Not available"
    if fusion_weights:
        fusion_text = f"Technical: {fusion_weights.get('technical', 0)*100:.1f}%, " \
                      f"Sentiment: {fusion_weights.get('sentiment', 0)*100:.1f}%, " \
                      f"Volatility: {fusion_weights.get('volatility', 0)*100:.1f}%"
    
    # Expert prompt
    prompt = f"""
# EXPERT STOCK ANALYSIS REQUEST

You are a senior quantitative analyst at a top-tier investment bank with 20+ years of experience in Indian equity markets. Provide a comprehensive, actionable analysis for a sophisticated retail investor.

## STOCK DATA: {stock_symbol} (NSE)

### Current Market Data:
- **Current Price:** ₹{current_price:,.2f}
- **Today's Change:** Included in price action

### AI Model Predictions:
- **{forecast_days}-Day Price Forecast:** ₹{price_forecast_end:,.2f}
- **Predicted Return:** {forecast_return:+.2f}%
- **Model Directional Accuracy:** {metrics.get('accuracy', 0):.1f}% (on out-of-sample test data)
- **Prediction RMSE:** {metrics.get('rmse', 0):.4f}

### Technical Indicators:
- **RSI (14):** {technical_indicators.get('RSI', 'N/A')}
- **5-Day Volatility:** {technical_indicators.get('Volatility_5D', 0)*100:.2f}%
- **20-Day Volatility:** {technical_indicators.get('Volatility_20D', 0)*100:.2f}%
- **Price vs 20-MA:** {technical_indicators.get('Price_vs_MA20', 0)*100:+.2f}%
- **MACD Histogram:** {technical_indicators.get('MACD_Histogram', 'N/A')}

### Sentiment Analysis (FinBERT NLP):
- **News Sentiment:** {sentiment_text}

### Fundamental Data:
- **Forward P/E:** {fundamentals.get('Forward P/E', 'N/A')}
- **PEG Ratio:** {fundamentals.get('PEG Ratio', 'N/A')}
- **ROE:** {fundamentals.get('ROE', 'N/A')}
- **Debt/Equity:** {fundamentals.get('Debt/Equity', 'N/A')}

### Dynamic Fusion Model Weights:
{fusion_text}

---

## REQUIRED OUTPUT FORMAT (Be concise, max 300 words total):

### 🎯 VERDICT
[One of: STRONG BUY 🟢 | BUY 🟢 | HOLD 🟡 | SELL 🔴 | STRONG SELL 🔴]

### 📊 OUTLOOK
- **Short-term (1-5 days):** [Bullish/Bearish/Neutral + 1 sentence why]
- **Medium-term (1-4 weeks):** [Bullish/Bearish/Neutral + 1 sentence why]

### 💡 KEY INSIGHT
[Single most important factor driving this recommendation - 2 sentences max]

### ⚠️ RISK FACTORS
[2-3 bullet points of key risks to monitor]

### 📈 TRADE SETUP (if actionable)
- **Entry Zone:** [Price range or "Wait for..."]
- **Stop Loss:** [Price level or % from entry]
- **Target:** [Price level or % gain expected]

---

**IMPORTANT GUIDELINES:**
1. Be direct and actionable - avoid vague language
2. If model accuracy is below 55%, explicitly note low confidence
3. Weight technical signals more when sentiment is mixed
4. Consider Indian market hours and global cues
5. Never guarantee returns - use probabilistic language
6. If data is insufficient, say so clearly
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Gemini API call failed: {str(e)[:100]}. Using fallback analysis.")
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices,
                                          metrics, sentiment_summary, technical_indicators)


def generate_fallback_analysis(stock_symbol: str, current_price: float,
                               predicted_prices, metrics: dict,
                               sentiment_summary: dict, technical_indicators: dict) -> str:
    """
    Generate structured analysis without Gemini API (template-based fallback).
    
    Args:
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        predicted_prices: DataFrame with predicted prices
        metrics: Dictionary with model metrics
        sentiment_summary: Dictionary with sentiment data
        technical_indicators: Dictionary with technical indicators
    
    Returns:
        Markdown-formatted analysis string
    """
    if not predicted_prices.empty:
        price_forecast_end = predicted_prices['Predicted Price'].iloc[-1]
    else:
        price_forecast_end = current_price
    
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    accuracy = metrics.get('accuracy', 50)
    
    # Determine verdict
    if accuracy < ModelConfig.LOW_CONFIDENCE_THRESHOLD:
        confidence = "Low Confidence"
    elif accuracy < ModelConfig.MEDIUM_CONFIDENCE_THRESHOLD:
        confidence = "Moderate Confidence"
    else:
        confidence = "Good Confidence"
    
    if forecast_return > 5 and accuracy > 60:
        verdict = "BUY 🟢"
        outlook = "Bullish"
    elif forecast_return > 2 and accuracy > 55:
        verdict = "HOLD (Positive Bias) 🟡"
        outlook = "Slightly Bullish"
    elif forecast_return < -5 and accuracy > 60:
        verdict = "SELL 🔴"
        outlook = "Bearish"
    elif forecast_return < -2 and accuracy > 55:
        verdict = "HOLD (Caution) 🟡"
        outlook = "Slightly Bearish"
    else:
        verdict = "HOLD 🟡"
        outlook = "Neutral"
    
    rsi = technical_indicators.get('RSI', 50)
    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    
    volatility = technical_indicators.get('Volatility_5D', 0)
    vol_text = "High volatility detected - position size accordingly" if volatility > 0.02 else "Normal volatility levels"
    
    return f"""
### 🎯 VERDICT: {verdict}
**{confidence}** | Model Accuracy: {accuracy:.1f}%

### 📊 OUTLOOK
- **Short-term:** {outlook} | Predicted {forecast_return:+.1f}% move
- **RSI Signal:** {rsi_signal} ({rsi:.1f})

### 💡 KEY INSIGHT
The hybrid AI model (XGBoost + GRU) predicts a {'positive' if forecast_return > 0 else 'negative'} return over the forecast period. {'However, model accuracy is below 55%, suggesting low predictive confidence.' if accuracy < 55 else 'Model shows reasonable directional accuracy on test data.'}

### ⚠️ RISK FACTORS
- Model predictions are probabilistic, not guarantees
- {vol_text}
- External market factors may override technical signals

*Analysis generated using template mode.*
"""


def generate_recommendation(predicted_prices, current_price: float, 
                            accuracy: float, avg_sentiment: float) -> tuple:
    """
    Generate investment recommendation based on predictions.
    
    Args:
        predicted_prices: DataFrame with predicted prices
        current_price: Current stock price
        accuracy: Model directional accuracy
        avg_sentiment: Average sentiment score
    
    Returns:
        Tuple of (recommendation_label, reason_text)
    """
    avg_prediction = predicted_prices['Predicted Price'].mean()
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    # Enhanced sentiment factor with confidence scaling
    sentiment_factor = 1 + (avg_sentiment * (accuracy/100))
    adjusted_change = price_change * sentiment_factor
    
    # Modified thresholds with confidence weighting
    confidence_weight = accuracy / 100
    
    if adjusted_change > TradingConfig.STRONG_BUY_THRESHOLD * confidence_weight and accuracy > 72:
        return "STRONG BUY", "High confidence in significant price increase"
    elif adjusted_change > TradingConfig.BUY_THRESHOLD * confidence_weight and accuracy > 65:
        return "BUY", "Good confidence in moderate price increase"
    elif adjusted_change > 0 and accuracy > 60:
        return "HOLD (Positive)", "Potential for slight growth"
    elif adjusted_change < TradingConfig.STRONG_SELL_THRESHOLD * confidence_weight and accuracy > 72:
        return "STRONG SELL", "High confidence in significant price drop"
    elif adjusted_change < TradingConfig.SELL_THRESHOLD * confidence_weight and accuracy > 65:
        return "SELL", "Good confidence in moderate price drop"
    elif adjusted_change < 0 and accuracy > 60:
        return "HOLD (Caution)", "Potential for slight decline"
    else:
        return "HOLD", "Unclear direction - consider other factors"
