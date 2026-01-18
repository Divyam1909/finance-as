# 🏆 ProTrader AI - Professional Stock Analytics Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered stock prediction and analysis platform for **Indian markets (NSE)** featuring multi-model fusion, official FII/DII data integration, multi-source sentiment analysis, and mathematical pattern recognition.

---

## 🎯 What Makes This Project Novel

### 1. **Dynamic Fusion Framework (Bayesian Multi-Expert System)**
Unlike traditional ensemble methods that use fixed weights, our framework dynamically adjusts expert weights based on **real-time uncertainty estimation**:

```
w_i = exp(-σ²_i) / Σ exp(-σ²_j)
```

- **Technical Expert** (GRU): Price pattern analysis using 128→64→32 unit architecture
- **Sentiment Expert** (Dense NN): News sentiment with 8-feature extraction
- **Volatility Expert** (MLP): India VIX + stock volatility analysis

Weights automatically shift based on which expert has been most accurate recently.

### 2. **Multi-Source Sentiment Aggregation**
Combines 4 independent data sources with weighted fusion:

| Source | Weight | Description |
|--------|--------|-------------|
| RSS Feeds | 30% | Moneycontrol, Economic Times, LiveMint, Business Standard |
| NewsAPI | 25% | Global financial news aggregation |
| Reddit | 25% | r/IndianStockMarket, r/DalalStreetTalks, r/IndiaInvestments |
| Google Trends | 20% | Retail interest proxy via search volume |

Uses **DistilRoBERTa-Financial** (98.2% accuracy, 2x faster than FinBERT) for sentiment classification.

### 3. **14-Feature Hybrid Model**
Combines XGBoost + GRU with comprehensive feature engineering:

```
Features (14):
├── Price/Technical (5): Log Returns, Volatility, RSI, Volume Ratio, MA Divergence
├── Sentiment (3): Base Sentiment, Multi-Source Score, Confidence
├── Institutional (4): FII/DII Net (Normalized), 5-Day Rolling Averages
└── Market Fear (2): VIX Normalized, VIX Change Rate
```

### 4. **Official NSE India Data Integration**
- Real-time FII/DII (Foreign/Domestic Institutional Investor) data from NSE API
- India VIX (market fear index) integration
- Custom Indian market holiday calendar for accurate forecasting

### 5. **Mathematical Pattern Detection**
Uses `scipy.signal.argrelextrema` for scientifically validated pattern detection:
- Double Top/Bottom
- Head & Shoulders / Inverse H&S
- Support/Resistance levels
- Trend analysis with linear regression

---

## 📊 Platform Features

| Tab | Description |
|-----|-------------|
| 📊 **Dashboard** | Main analysis, AI predictions, accuracy charts, Gemini AI commentary |
| 🔬 **Dynamic Fusion** | Real-time expert weight visualization, uncertainty tracking |
| 📈 **Technicals & Risk** | Fibonacci levels, ATR, trade setup calculator with risk/reward |
| 🏛️ **Fundamentals** | P/E, ROE, debt ratios from Yahoo Finance |
| 💼 **FII/DII Analysis** | Official NSE institutional investor activity charts |
| 📰 **Multi-Source Sentiment** | 4-source sentiment analysis with source breakdown |
| 🛠️ **Backtest** | Strategy backtesting with Sharpe ratio, max drawdown, equity curves |
| 📐 **Pattern Analysis** | Mathematical chart pattern detection |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/protrader-ai.git
cd protrader-ai

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install streamlit yfinance pandas numpy xgboost tensorflow transformers plotly scikit-learn python-dotenv requests google-generativeai feedparser praw pytrends
```

### 2. Configure API Keys (Optional but Recommended)
Create a `.env` file:
```bash
GEMINI_API_KEY=your_gemini_key          # For AI analysis commentary
NEWS_API_KEY=your_newsapi_key           # For enhanced news sentiment
REDDIT_CLIENT_ID=your_reddit_id         # For Reddit sentiment
REDDIT_CLIENT_SECRET=your_reddit_secret
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
finance/
├── app.py                      # Main Streamlit application (831 lines)
├── config/
│   └── settings.py             # Configuration constants, API keys, model params
├── data/
│   ├── stock_data.py           # Yahoo Finance stock data fetching
│   ├── fii_dii.py              # NSE FII/DII official data
│   ├── vix_data.py             # India VIX + synthetic VIX fallback
│   ├── news_sentiment.py       # NewsAPI + FinBERT sentiment
│   └── multi_sentiment.py      # 4-source sentiment aggregator (695 lines)
├── models/
│   ├── hybrid_model.py         # XGBoost + GRU ensemble (420 lines)
│   ├── fusion_framework.py     # Bayesian multi-expert fusion (274 lines)
│   ├── technical_expert.py     # GRU-based technical model
│   ├── sentiment_expert.py     # Dense NN for sentiment
│   ├── volatility_expert.py    # MLP for VIX analysis
│   ├── visual_analyst.py       # Mathematical pattern detection (408 lines)
│   ├── backtester.py           # Vectorized backtesting engine
│   └── optimizer.py            # Optuna hyperparameter tuning
├── ui/
│   ├── charts.py               # Plotly chart generation
│   └── ai_analysis.py          # Gemini AI integration
├── utils/
│   ├── technical_indicators.py # TA feature calculation
│   └── risk_manager.py         # ATR, Fibonacci, trade setup
├── indian_stocks.csv           # NSE stock symbols list
├── .env                        # API keys (gitignored)
└── README.md                   # This file
```

---

## 🔧 Model Architecture

### Hybrid Model Pipeline
```
Input Data
    │
    ├─→ Feature Engineering (14 features)
    │       ├── Technical: Log Returns, Volatility, RSI, Volume Ratio, MA Div
    │       ├── Sentiment: Score, Multi-Source, Confidence  
    │       ├── Institutional: FII/DII Net Normalized, 5D Averages
    │       └── Volatility: VIX Normalized, VIX Change
    │
    ├─→ XGBoost Regressor
    │       └── 100 trees, max_depth=3, lr=0.05
    │
    ├─→ GRU Neural Network
    │       └── 32 units, dropout=0.2, 20 epochs
    │
    └─→ Simple Ensemble (50/50 average)
            │
            └─→ Predicted Return → Future Price Projection
```

### Dynamic Fusion Framework
```
                    ┌─────────────────┐
                    │  Stock Data     │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Technical   │   │   Sentiment   │   │   Volatility  │
│    Expert     │   │    Expert     │   │    Expert     │
│   (GRU NN)    │   │  (Dense NN)   │   │    (MLP)      │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        ├───── Uncertainty ─┼───── Uncertainty ─┤
        │         σ²        │         σ²        │         σ²
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────┐
│              Bayesian Weight Calculator                 │
│           w_i = exp(-σ²_i) / Σ exp(-σ²_j)              │
└─────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Combined Prediction
```

---

## 📈 Performance Metrics

The platform uses strict **walk-forward validation** to prevent look-ahead bias:

- **Direction Accuracy**: Percentage of correct up/down predictions (target: >60%)
- **RMSE**: Root Mean Square Error of return predictions
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

---

## 🔑 API Keys Setup

### Gemini API (Free - Powers AI Commentary)
1. Visit: https://makersuite.google.com/app/apikey
2. Create API key
3. Add to `.env`: `GEMINI_API_KEY=your_key`

### NewsAPI (Free tier - 100 requests/day)
1. Visit: https://newsapi.org/register
2. Sign up and get key
3. Add to `.env`: `NEWS_API_KEY=your_key`

### Reddit API (Free - For social sentiment)
1. Visit: https://www.reddit.com/prefs/apps
2. Create "script" type application
3. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=your_id
   REDDIT_CLIENT_SECRET=your_secret
   ```

---

## 🚨 Known Limitations & Future Improvements

### Current Limitations
1. **NSE API Reliability**: FII/DII data may be unavailable if NSE website is down
2. **VIX Data**: Falls back to synthetic VIX (NIFTY volatility) when India VIX unavailable
3. **Real-time Data**: Uses end-of-day data; not suitable for intraday trading
4. **Model Training**: GRU training can be slow on CPU (GPU recommended)

### Planned Improvements
- [ ] **Attention Mechanisms**: Add transformer-based attention to GRU
- [ ] **Options Data Integration**: IV, Put-Call ratio from NSE
- [ ] **Intraday Support**: 5-minute candle data for day trading
- [ ] **Portfolio Optimization**: Multi-stock portfolio with correlation analysis
- [ ] **MLflow Integration**: Model versioning and experiment tracking
- [ ] **Real-time Streaming**: WebSocket-based live data updates
- [ ] **Mobile App**: React Native companion app

---

## 📦 Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
tensorflow>=2.13.0
transformers>=4.30.0
plotly>=5.15.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
requests>=2.31.0
google-generativeai>=0.3.0
feedparser>=6.0.0          # RSS feed parsing
praw>=7.7.0                # Reddit API
pytrends>=4.9.0            # Google Trends
scipy>=1.11.0              # Pattern detection
```

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- ❌ Not financial advice
- ❌ Past performance ≠ future results
- ❌ Do not use for real trading without extensive backtesting
- ✅ Always do your own research
- ✅ Consult a SEBI-registered financial advisor before investing

---

## 📄 License

MIT License - Free for personal and commercial use.

---

## 🙏 Credits

- **Data Sources**: Yahoo Finance, NSE India
- **Sentiment Model**: [DistilRoBERTa-Financial](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
- **AI Commentary**: Google Gemini
- **Pattern Detection**: SciPy signal processing

---

**Version**: 3.0 | **Last Updated**: January 2026 | **Author**: ProTrader AI Team
