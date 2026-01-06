# ProTrader AI: Upgrade Documentation

This document details the upgrades made to `rps3_f.py`, transforming it from a prediction dashboard into a professional trading workstation.

## Comparison: Previous vs. Current

| Feature | Previous Version | Current Version (Upgrade) |
| :--- | :--- | :--- |
| **Model** | Static Hybrid (XGBoost + GRU) | **AutoML Optimized** Hybrid (Optuna Tuning) |
| **Data Source** | Technical + News Sentiment | Technical + Sentiment + **Fundamental Data** |
| **Risk Mgmt** | None | **ATR Stop-Loss, Fibonacci, Kelly Criterion** |
| **Backtest** | Simple accuracy metric | **Vectorized Strategy Simulation** (Sharpe, Drawdown) |
| **UI** | Single Page Scroll | **Multi-Tab Dashboard** (Analytics, Fundamentals, Risk) |
| **Forecast** | Simple Price Curve | **Trade Setup Card** (Entry, Stop, Target) |

## New Features Detail

### 1. Fundamental Analysis Module
**Location:** Tab 4 ("Fundamentals")
*   Fetches real-time financial ratios from Yahoo Finance.
*   Metrics: `P/E Ratio`, `PEG`, `Debt/Equity`, `ROE`, `Profit Margins`.
*   **Purpose:** Ensures trading decisions are backed by financial health, not just chart patterns.

### 2. Professional Risk Management
**Location:** Tab 3 ("Technicals & Risk")
*   **ATR (Average True Range):** Calculates market volatility to set dynamic stop-losses.
*   **Fibonacci Retracements:** Automatically calculates support/resistance levels.
*   **Trade Setup Card:** Generates a complete trade plan ("Buy at X, Stop at Y, Target Z") based on the AI's confidence.

### 3. Vectorized Backtesting Engine
**Location:** Tab 5 ("Backtest")
*   Simulates the strategy logic over historical data.
*   **Metrics:**
    *   **Sharpe Ratio:** Risk-adjusted return.
    *   **Max Drawdown:** The worst peak-to-valley loss.
    *   **Equity Curve:** Visualizes portfolio growth over time.
    *   **Win Rate:** Percentage of profitable trades.

### 4. AutoML (Hyperparameter Optimization)
**Location:** Sidebar ("Advanced Settings")
*   Uses `optuna` (Bayesian optimization) to find the perfect settings for XGBoost and GRU *for the specific stock selected*.
*   **Note:** This is computationally intensive. Enable the checkbox only when you want to retrain for maximum accuracy.

### 5. Dynamic Fusion Framework (Enhanced)
**Location:** Tab 2 ("Dynamic Fusion")
*   Existing logic preserved but visualized better.
*   Shows how the AI shifts weight between "Technical", "Sentiment", and "Volatility" experts day-by-day.

## Usage Instructions

1.  **Dependencies:** Ensure `optuna`, `yfinance`, `xgboost`, `streamlit`, `plotly`, `tensorflow` are installed.
2.  **Run:** `streamlit run rps3_f.py`
3.  **Workflow:**
    *   Select Stock & Date Range.
    *   Check "Enable AutoML" if you want to optimize (takes ~1-2 mins).
    *   Click **Launch Analysis**.
    *   Review **Dashboard** for price/sentiment.
    *   Check **Fundamentals** to ensure company health.
    *   Use **Technicals & Risk** to find your Entry/Stop-Loss.
    *   (Optional) Run **Backtest** to validate the strategy.

## File Changes
*   **`rps3_f.py`**:
    *   Added `ModelOptimizer` class (Line ~997).
    *   Added `RiskManager` class (Line ~850).
    *   Added `VectorizedBacktester` class (Line ~950).
    *   Added `get_fundamental_data` function.
    *   Replaced Main UI loop with Tab-based structure (Line ~1430+).
