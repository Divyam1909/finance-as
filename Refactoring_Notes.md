# Professional Grade Refactoring Update

## Key Changes for Authenticity

We have stripped away the "marketing fluff" and implemented rigid financial machine learning standards:

### 1. Target Standardization (Stationarity)
*   **Old:** Predicted absolute prices (e.g., ₹2500). This is statistically flawed because prices drift (non-stationary).
*   **New:** Predicts **Log Returns** (e.g., +0.015). This allows the model to learn market *physics* rather than memorizing price levels.

### 2. Elimination of Data Leakage (Strict Walk-Forward)
*   **Old:** Calculated technical indicators on the whole dataset, then split. This let the model "see" the future volatility.
*   **New:** Strict separation. The scaler and model ONLY see data from the training window. Test Performance is now a **true Out-of-Sample** metric.

### 3. Removed "Prophet" (Trend Leakage)
*   **Reason:** Prophet fits a curve for the *entire* history. Using it to predict "tomorrow" cheats because it knows the end of the curve.
*   **Action:** Removed completely. We rely solely on the correlation between Signals and Next-Day Returns.

### 4. Recursive Forecasting
*   **Logic:** To predict 10 days out, the model predicts Day 1 Return -> Calculates Day 1 Price -> Predicts Day 2 Return. This accumulates uncertainty naturally, giving you a realistic (and often wider) fan of outcomes.

### 5. Reality-Check Metrics
*   **RMSE:** Root Mean Squared Error on *Returns* (not prices).
*   **Directional Accuracy:** How often did we get the Sign (+/-) right? This is the only metric that matters for trading.

## How to Interpret the New UI
*   **Directional Accuracy**: If this is > 53-55%, you have an edge. 50% is random variance.
*   **Backtest**: The Equity Curve is now based on simply buying when predicted return > 0 and selling when < 0. It reflects the *raw predictive power* of the model.
