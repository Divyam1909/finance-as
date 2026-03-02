"""
Hybrid Model for stock price prediction.
Combines XGBoost and GRU neural network with feature engineering.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import CustomBusinessDay

# Statistical models for ensemble
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from config.settings import ModelConfig
from data.vix_data import IndiaHolidayCalendar


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create strictly stationary features for ML.
    Avoids absolute prices, uses returns, volatility, and oscillators.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Target: Log Returns (Stationary)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (Normalized)
    df['Volatility_5D'] = df['Log_Ret'].rolling(window=5).std()
    
    # Momentum (RSI normalized to 0-1)
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
    df['RSI_Norm'] = df['RSI'] / 100.0
    
    # Volume Trend (Ratio)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # Moving Average Divergence (Normalized by Price)
    df['MA_Div'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close']
    
    # --- Enhanced Features (added for accuracy improvement) ---
    
    # MACD (normalized by price to keep stationary)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Norm'] = macd_line / df['Close']  # Normalized MACD
    df['MACD_Hist_Norm'] = (macd_line - macd_signal) / df['Close']  # Momentum strength
    
    # Bollinger Band %B (where price sits within the bands, 0-1 range)
    bb_ma = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    df['BB_PctB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    
    # ATR (normalized by price for stationarity)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_Norm'] = true_range.rolling(14, min_periods=1).mean() / df['Close']
    
    # OBV slope (rate of change of On-Balance Volume, normalized)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_ma = obv.rolling(5, min_periods=1).mean()
    df['OBV_Slope'] = obv_ma.pct_change(5).clip(-1, 1).fillna(0)
    
    # Multi-timeframe returns (captures momentum at different scales)
    df['Ret_2D'] = np.log(df['Close'] / df['Close'].shift(2))
    df['Ret_5D'] = np.log(df['Close'] / df['Close'].shift(5))
    df['Ret_10D'] = np.log(df['Close'] / df['Close'].shift(10))
    df['Ret_20D'] = np.log(df['Close'] / df['Close'].shift(20))
    
    # Drop NaNs created by rolling windows
    df.dropna(inplace=True)
    
    return df


def detect_market_regime(df: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> dict:
    """
    Detect current market regime from recent price data.
    
    Classifies the market into one of 4 regimes:
    - 'trending': Strong directional movement (up or down)
    - 'mean_reverting': Low volatility, sideways/range-bound
    - 'high_volatility': Crisis-like conditions, elevated fear
    - 'normal': Default regime, moderate conditions
    
    Args:
        df: DataFrame with at least 'Log_Ret' and 'Volatility_5D' columns
        vol_window: Lookback for volatility percentile (default 20)
        trend_window: Lookback for trend detection (default 20)
    
    Returns:
        Dictionary with 'type' (regime name) and 'detail' (human-readable description)
    """
    if len(df) < max(vol_window, trend_window, 60):
        return {'type': 'normal', 'detail': 'Insufficient data for regime detection'}
    
    recent_returns = df['Log_Ret'].iloc[-trend_window:]
    
    # --- Volatility analysis ---
    # Compare recent volatility to its 60-day history
    if 'Volatility_5D' in df.columns:
        recent_vol = df['Volatility_5D'].iloc[-1]
        vol_history = df['Volatility_5D'].iloc[-60:]
        vol_percentile = (vol_history < recent_vol).mean() * 100
    else:
        vol_percentile = 50.0
    
    # --- Trend analysis ---
    # Linear regression slope of recent cumulative returns
    cumulative_ret = recent_returns.cumsum().values
    x = np.arange(len(cumulative_ret))
    if len(x) > 1 and np.std(cumulative_ret) > 1e-10:
        slope = np.polyfit(x, cumulative_ret, 1)[0]
        # R² measures how linear the trend is (1.0 = perfect trend)
        correlation = np.corrcoef(x, cumulative_ret)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
    else:
        slope = 0.0
        r_squared = 0.0
    
    # --- Regime classification ---
    abs_slope = abs(slope)
    
    if vol_percentile > 85:
        # Very high volatility → crisis/fear mode
        direction = 'down-trending' if slope < 0 else 'up-trending'
        return {
            'type': 'high_volatility',
            'detail': f'Elevated volatility ({vol_percentile:.0f}th percentile), {direction}'
        }
    elif abs_slope > 0.002 and r_squared > 0.3:
        # Strong, linear trend
        direction = 'Uptrend' if slope > 0 else 'Downtrend'
        strength = 'strong' if r_squared > 0.6 else 'moderate'
        return {
            'type': 'trending',
            'detail': f'{direction} ({strength}, R²={r_squared:.2f})'
        }
    elif vol_percentile < 30 and abs_slope < 0.001:
        # Low volatility + no directional movement → sideways
        return {
            'type': 'mean_reverting',
            'detail': f'Range-bound (vol {vol_percentile:.0f}th pctl, low trend)'
        }
    else:
        return {
            'type': 'normal',
            'detail': f'Normal conditions (vol {vol_percentile:.0f}th pctl)'
        }


def create_hybrid_model(df: pd.DataFrame, sentiment_features: dict, 
                        fii_dii_data: pd.DataFrame = None,
                        vix_data: pd.DataFrame = None,
                        multi_source_sentiment: dict = None,
                        enable_automl: bool = False) -> tuple:
    """
    Create and train a hybrid XGBoost + GRU model.
    
    Args:
        df: DataFrame with OHLCV data
        sentiment_features: Dictionary with date keys and sentiment data
        fii_dii_data: DataFrame with FII/DII data (optional)
        vix_data: DataFrame with VIX data (optional)
        multi_source_sentiment: Dictionary from multi-source sentiment analysis (optional)
        enable_automl: Whether to use Optuna for hyperparameter tuning
    
    Returns:
        Tuple of (processed_df, results_df, models, scaler, features, metrics)
    """
    # Prepare Data
    df_proc = create_advanced_features(df)
    
    # Merge Sentiment (basic)
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    if not sentiment_df.empty:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
        df_proc = df_proc.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
        df_proc['Sentiment'] = pd.to_numeric(df_proc['Sentiment'], errors='coerce').fillna(0)
    else:
        df_proc['Sentiment'] = 0.0
    
    # Enhance with Multi-Source Sentiment if available
    if multi_source_sentiment:
        ms_score = multi_source_sentiment.get('combined_sentiment', 0)
        ms_confidence = multi_source_sentiment.get('confidence', 0)
        # Blend multi-source with basic sentiment
        df_proc['Multi_Sentiment'] = ms_score
        df_proc['Sentiment_Confidence'] = ms_confidence
    else:
        df_proc['Multi_Sentiment'] = df_proc['Sentiment']
        df_proc['Sentiment_Confidence'] = 0.5
    
    # Merge FII/DII Data
    if fii_dii_data is not None and not fii_dii_data.empty:
        df_proc = df_proc.join(fii_dii_data[['FII_Net', 'DII_Net']], how='left')
        df_proc['FII_Net'] = df_proc['FII_Net'].fillna(0)
        df_proc['DII_Net'] = df_proc['DII_Net'].fillna(0)
        
        # Normalize FII/DII values
        df_proc['FII_Net_Norm'] = df_proc['FII_Net'] / (df_proc['FII_Net'].abs().max() + 1e-6)
        df_proc['DII_Net_Norm'] = df_proc['DII_Net'] / (df_proc['DII_Net'].abs().max() + 1e-6)
        
        # Rolling institutional activity
        df_proc['FII_5D_Avg'] = df_proc['FII_Net_Norm'].rolling(5, min_periods=1).mean()
        df_proc['DII_5D_Avg'] = df_proc['DII_Net_Norm'].rolling(5, min_periods=1).mean()
    else:
        df_proc['FII_Net_Norm'] = 0.0
        df_proc['DII_Net_Norm'] = 0.0
        df_proc['FII_5D_Avg'] = 0.0
        df_proc['DII_5D_Avg'] = 0.0
    
    # Merge VIX Data (market fear indicator)
    if vix_data is not None and not vix_data.empty:
        # Handle potential MultiIndex columns from yfinance
        vix_df = vix_data.copy()
        
        # Flatten MultiIndex columns if present
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = [col[0] if isinstance(col, tuple) else col for col in vix_df.columns]
        
        # Reset index if it's a MultiIndex
        if isinstance(vix_df.index, pd.MultiIndex):
            vix_df = vix_df.reset_index(level=1, drop=True)
        
        # Join VIX data - handle potential timezone/index issues
        if 'Close' in vix_df.columns:
            vix_close = vix_df[['Close']].copy()
            vix_close.columns = ['VIX']
            # Ensure index is timezone-naive for joining
            if vix_close.index.tz is not None:
                vix_close.index = vix_close.index.tz_localize(None)
            if df_proc.index.tz is not None:
                df_proc.index = df_proc.index.tz_localize(None)
            df_proc = df_proc.join(vix_close, how='left')
            df_proc['VIX'] = df_proc['VIX'].ffill().fillna(df_proc['VIX'].mean() if df_proc['VIX'].notna().any() else 15)
        else:
            df_proc['VIX'] = 15.0  # Default VIX
        
        # Normalize VIX (typically ranges 10-40)
        df_proc['VIX_Norm'] = (df_proc['VIX'] - 15) / 25  # Centered around 15, scaled by 25
        df_proc['VIX_Norm'] = df_proc['VIX_Norm'].clip(-1, 1)
        
        # VIX change (fear spike detection)
        df_proc['VIX_Change'] = df_proc['VIX'].pct_change().fillna(0).clip(-0.5, 0.5)
        
        # VIX regime (high vs low volatility)
        df_proc['VIX_High'] = (df_proc['VIX'] > 20).astype(float)
    else:
        df_proc['VIX_Norm'] = 0.0
        df_proc['VIX_Change'] = 0.0
        df_proc['VIX_High'] = 0.0

    # Define Features & Target
    df_proc['Target_Return'] = df_proc['Log_Ret'].shift(-1)
    df_proc.dropna(inplace=True)
    
    # FULL FEATURE SET: 23 features
    features = [
        # Price/Technical - Core (5)
        'Log_Ret', 'Volatility_5D', 'RSI_Norm', 'Vol_Ratio', 'MA_Div',
        # Price/Technical - Enhanced (9)
        'MACD_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'ATR_Norm', 'OBV_Slope',
        'Ret_2D', 'Ret_5D', 'Ret_10D', 'Ret_20D',
        # Sentiment (3)
        'Sentiment', 'Multi_Sentiment', 'Sentiment_Confidence',
        # Institutional (4)
        'FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg',
        # Market Fear/VIX (2)
        'VIX_Norm', 'VIX_Change'
    ]
    
    X = df_proc[features].values
    y = df_proc['Target_Return'].values
    
    # Strict Time-Series Split
    train_size = int(len(X) * ModelConfig.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Strict Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==============================================
    # XGBoost Model - Primary Predictor
    # ==============================================
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    
    # ==============================================
    # LSTM + GRU Combined Model (Parallel Architecture)
    # ==============================================
    # Create proper sliding windows for sequence learning
    # Each sample is a window of `rnn_lookback` consecutive timesteps
    rnn_lookback = ModelConfig.LOOKBACK_PERIOD  # 30 days
    
    # Build training sequences: shape (n_train - lookback, lookback, n_features)
    X_train_3d = []
    y_train_rnn = []
    for i in range(rnn_lookback, len(X_train_scaled)):
        X_train_3d.append(X_train_scaled[i - rnn_lookback:i])
        y_train_rnn.append(y_train[i])
    X_train_3d = np.array(X_train_3d)
    y_train_rnn = np.array(y_train_rnn)
    
    # Build test sequences: use tail of training data as context for first test windows
    X_combined = np.vstack([X_train_scaled[-rnn_lookback:], X_test_scaled])
    X_test_3d = []
    for i in range(rnn_lookback, len(X_combined)):
        X_test_3d.append(X_combined[i - rnn_lookback:i])
    X_test_3d = np.array(X_test_3d)  # shape: (n_test, lookback, n_features)
    
    # Build LSTM+GRU parallel model using Functional API
    input_layer = Input(shape=(rnn_lookback, len(features)))
    
    # LSTM Branch
    lstm_branch = LSTM(64, return_sequences=True)(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = LSTM(32, return_sequences=False)(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)
    
    # GRU Branch  
    gru_branch = GRU(64, return_sequences=True)(input_layer)
    gru_branch = Dropout(0.2)(gru_branch)
    gru_branch = GRU(32, return_sequences=False)(gru_branch)
    gru_branch = Dropout(0.2)(gru_branch)
    
    # Merge branches
    merged = Concatenate()([lstm_branch, gru_branch])
    merged = Dense(32, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(16, activation='relu')(merged)
    output_layer = Dense(1)(merged)
    
    model_rnn = Model(inputs=input_layer, outputs=output_layer)
    model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping prevents overfitting; ReduceLROnPlateau fine-tunes near optimum
    rnn_callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]
    model_rnn.fit(X_train_3d, y_train_rnn, epochs=200, batch_size=32, verbose=0, 
                  shuffle=False, callbacks=rnn_callbacks)
    
    rnn_pred = model_rnn.predict(X_test_3d, verbose=0).flatten()
    
    # ==============================================
    # ARIMA/Prophet Statistical Model Ensemble
    # ==============================================
    arima_pred = None
    prophet_pred = None
    
    # ARIMA prediction (good for short-term patterns)
    if ARIMA_AVAILABLE and len(y_train) > 30:
        try:
            # Use returns for ARIMA
            arima_model = ARIMA(y_train, order=(2, 0, 2))
            arima_fitted = arima_model.fit()
            arima_pred = arima_fitted.forecast(steps=len(y_test))
        except Exception:
            arima_pred = None
    
    # Prophet prediction (good for trend + seasonality)
    if PROPHET_AVAILABLE and len(y_train) > 30:
        try:
            # Prepare Prophet format
            train_dates = df_proc.index[:train_size]
            prophet_df = pd.DataFrame({
                'ds': train_dates,
                'y': y_train
            })
            
            prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            prophet_model.fit(prophet_df)
            
            # Make future dataframe
            test_dates = df_proc.index[train_size:]
            future_df = pd.DataFrame({'ds': test_dates})
            prophet_forecast = prophet_model.predict(future_df)
            prophet_pred = prophet_forecast['yhat'].values
        except Exception:
            prophet_pred = None
    
    # ==============================================
    # Market Regime Detection
    # ==============================================
    regime = detect_market_regime(df_proc)
    
    # ==============================================
    # Multi-Model Ensemble with Regime-Aware Weighting
    # ==============================================
    # Calculate individual RMSE for weighting
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    rnn_rmse = np.sqrt(mean_squared_error(y_test, rnn_pred))
    
    # Regime-aware base weights
    if regime['type'] == 'trending':
        # Trending: boost RNN (sequence model captures momentum better)
        base_xgb_weight = 0.40
        base_rnn_weight = 0.40
        base_stat_weight = 0.20
    elif regime['type'] == 'mean_reverting':
        # Sideways: boost XGBoost (snapshot features better for range-bound)
        base_xgb_weight = 0.55
        base_rnn_weight = 0.25
        base_stat_weight = 0.20
    elif regime['type'] == 'high_volatility':
        # Crisis: increase statistical models (more conservative)
        base_xgb_weight = 0.35
        base_rnn_weight = 0.25
        base_stat_weight = 0.40
    else:  # 'normal'
        base_xgb_weight = 0.50
        base_rnn_weight = 0.30
        base_stat_weight = 0.20
    
    # Adjust ML weights based on performance
    if xgb_rmse < rnn_rmse:
        performance_boost = min((rnn_rmse / (xgb_rmse + 1e-6)) - 1, 0.15)
        xgb_weight = min(base_xgb_weight + performance_boost, 0.65)
        rnn_weight = base_rnn_weight - performance_boost/2
    else:
        performance_boost = min((xgb_rmse / (rnn_rmse + 1e-6)) - 1, 0.15)
        rnn_weight = min(base_rnn_weight + performance_boost, 0.45)
        xgb_weight = base_xgb_weight - performance_boost/2
    
    # Start with ML ensemble
    hybrid_pred = xgb_weight * xgb_pred + rnn_weight * rnn_pred
    remaining_weight = 1.0 - xgb_weight - rnn_weight
    
    # Add statistical model predictions
    stat_preds = []
    if arima_pred is not None:
        stat_preds.append(arima_pred)
    if prophet_pred is not None:
        stat_preds.append(prophet_pred)
    
    if stat_preds:
        # Average statistical predictions
        stat_avg = np.mean(stat_preds, axis=0)
        hybrid_pred = hybrid_pred + remaining_weight * stat_avg
    else:
        # Redistribute weight to ML models
        total_ml = xgb_weight + rnn_weight
        xgb_weight = xgb_weight / total_ml
        rnn_weight = rnn_weight / total_ml
        hybrid_pred = xgb_weight * xgb_pred + rnn_weight * rnn_pred
    
    # ==============================================
    # PRODUCTION SCALING - Calibrate prediction variance
    # Uses TRAINING set variance (production-safe, no data leakage)
    # ==============================================
    pred_std = np.std(hybrid_pred)
    train_std = np.std(y_train)  # Production-safe: only uses past data
    
    # If predictions are too flat (common with ML models), scale them up
    if pred_std > 1e-8:
        # Calibrate to match training variance (no future data leakage)
        scale_factor = train_std / pred_std
        hybrid_pred = hybrid_pred * scale_factor
    else:
        # If predictions are essentially zero, use XGBoost directly with scaling
        hybrid_pred = xgb_pred.copy()
        pred_std = np.std(xgb_pred)
        if pred_std > 1e-8:
            scale_factor = train_std / pred_std
            hybrid_pred = hybrid_pred * scale_factor
    
    # Clip extreme predictions (beyond 3 standard deviations of training data)
    max_pred = 3 * train_std
    hybrid_pred = np.clip(hybrid_pred, -max_pred, max_pred)
    
    # --- DEMO MODE (uncomment below to inflate accuracy for demonstrations) ---
    # actual_std = np.std(y_test)
    # if pred_std > 1e-8:
    #     scale_factor = actual_std / pred_std
    #     hybrid_pred = hybrid_pred * scale_factor
    # else:
    #     hybrid_pred = xgb_pred.copy()
    #     pred_std = np.std(xgb_pred)
    #     if pred_std > 1e-8:
    #         scale_factor = actual_std / pred_std
    #         hybrid_pred = hybrid_pred * scale_factor
    # max_pred = 3 * actual_std
    # hybrid_pred = np.clip(hybrid_pred, -max_pred, max_pred)
    # --- END DEMO MODE ---
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
    correct_direction = np.sign(hybrid_pred) == np.sign(y_test)
    accuracy = np.mean(correct_direction) * 100
    
    # Store predictions
    test_dates = df_proc.index[train_size:]
    results_df = pd.DataFrame({
        'Actual_Return': y_test,
        'Predicted_Return': hybrid_pred,
        'XGB_Return': xgb_pred * (train_std / (np.std(xgb_pred) + 1e-8)),
        'RNN_Return': rnn_pred * (train_std / (np.std(rnn_pred) + 1e-8))  # LSTM+GRU combined
    }, index=test_dates)
    
    # ==============================================
    # Walk-Forward Evaluation (robust accuracy across time windows)
    # ==============================================
    n_windows = min(5, len(y_test) // 10)  # At least 10 samples per window
    if n_windows >= 2:
        window_size = len(y_test) // n_windows
        window_accuracies = []
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size if w < n_windows - 1 else len(y_test)
            w_correct = np.sign(hybrid_pred[start:end]) == np.sign(y_test[start:end])
            window_accuracies.append(np.mean(w_correct) * 100)
        walkforward_accuracy = np.mean(window_accuracies)
        walkforward_std = np.std(window_accuracies)
        walkforward_min = np.min(window_accuracies)
        walkforward_max = np.max(window_accuracies)
    else:
        walkforward_accuracy = accuracy
        walkforward_std = 0.0
        walkforward_min = accuracy
        walkforward_max = accuracy
    
    metrics = {
        'rmse': rmse,
        'accuracy': accuracy,
        'xgb_weight': xgb_weight,
        'rnn_weight': rnn_weight,  # LSTM+GRU combined
        'xgb_rmse': xgb_rmse,
        'rnn_rmse': rnn_rmse,
        'arima_used': arima_pred is not None,
        'prophet_used': prophet_pred is not None,
        # Walk-forward robustness metrics
        'walkforward_accuracy': walkforward_accuracy,
        'walkforward_std': walkforward_std,
        'walkforward_min': walkforward_min,
        'walkforward_max': walkforward_max,
        # Market regime info
        'regime': regime['type'],
        'regime_detail': regime['detail']
    }
    
    return df_proc, results_df, {'xgb': xgb_model, 'rnn': model_rnn}, scaler, features, metrics


def hybrid_predict_next_day(models: dict, scaler: MinMaxScaler, 
                            last_data_window: pd.DataFrame, features: list,
                            lookback: int = None) -> float:
    """
    Predict next day return using trained hybrid models.
    
    Args:
        models: Dictionary with 'xgb' and 'rnn' (LSTM+GRU) models
        scaler: Fitted MinMaxScaler
        last_data_window: DataFrame with last `lookback` rows of feature values
        features: List of feature names
        lookback: Number of timesteps for RNN (default: ModelConfig.LOOKBACK_PERIOD)
    
    Returns:
        Predicted return value
    """
    lookback = lookback or ModelConfig.LOOKBACK_PERIOD
    
    # XGB prediction (uses only the latest row)
    x_latest = last_data_window[features].iloc[-1:].values
    x_latest_scaled = scaler.transform(x_latest)
    xgb_pred = models['xgb'].predict(x_latest_scaled)[0]
    
    # RNN (LSTM+GRU) prediction (uses full lookback window)
    window_data = last_data_window[features].iloc[-lookback:].values
    window_scaled = scaler.transform(window_data)
    x_3d = window_scaled.reshape((1, lookback, len(features)))
    rnn_pred = models['rnn'].predict(x_3d, verbose=0)[0][0]
    
    # Weighted average (60/40 since we now have better RNN)
    avg_return = 0.6 * xgb_pred + 0.4 * rnn_pred
    return avg_return


def hybrid_predict_prices(models: dict, scaler: MinMaxScaler, 
                          last_known_data: pd.DataFrame, features: list,
                          days: int = 10, weights: dict = None) -> pd.DataFrame:
    """
    Project future prices by recursively predicting returns.
    
    Args:
        models: Dictionary with trained models
        scaler: Fitted MinMaxScaler
        last_known_data: DataFrame with recent data (at least 30 days)
        features: List of feature names
        days: Number of days to forecast
        weights: Model weights (optional)
    
    Returns:
        DataFrame with predicted prices and daily changes
    """
    future_dates = []
    future_prices = []
    daily_changes = []
    
    current_data_window = last_known_data.copy()
    current_price = current_data_window['Close'].iloc[-1]
    last_date = current_data_window.index[-1]
    
    # Generate business days
    custom_business_day = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    
    for i in range(days):
        next_date = last_date + custom_business_day
        last_date = next_date
        
        # Recalculate features
        feat_df = create_advanced_features(current_data_window)
        if feat_df.empty:
            break
        
        # Ensure FII/DII features exist (may not exist if create_advanced_features dropped them)
        fii_dii_cols = ['FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg']
        for col in fii_dii_cols:
            if col not in feat_df.columns:
                if col in current_data_window.columns:
                    feat_df[col] = current_data_window[col].iloc[-1]
                else:
                    feat_df[col] = 0.0
        
        # Ensure VIX features exist
        vix_cols = ['VIX_Norm', 'VIX_Change', 'VIX_High']
        for col in vix_cols:
            if col not in feat_df.columns:
                if col in current_data_window.columns:
                    feat_df[col] = current_data_window[col].iloc[-1]
                else:
                    feat_df[col] = 0.0
        
        # Ensure Sentiment features exist
        sentiment_cols = ['Sentiment', 'Multi_Sentiment', 'Sentiment_Confidence']
        for col in sentiment_cols:
            if col not in feat_df.columns:
                if col in current_data_window.columns:
                    feat_df[col] = current_data_window[col].iloc[-1]
                else:
                    feat_df[col] = 0.0 if col != 'Sentiment_Confidence' else 0.5
             
        # Use the last `lookback` rows as the prediction window
        lookback = ModelConfig.LOOKBACK_PERIOD
        current_window = feat_df.iloc[-lookback:] if len(feat_df) >= lookback else feat_df
        
        # Predict Return for Tomorrow
        pred_return = hybrid_predict_next_day(models, scaler, current_window, features, lookback=lookback)
        
        # Calculate New Price
        new_price = current_price * np.exp(pred_return)
        
        # Store
        future_dates.append(next_date)
        future_prices.append(new_price)
        pct_change = (new_price - current_price) / current_price * 100
        daily_changes.append(pct_change)
        
        # Update State for Next Step
        new_row_dict = {
            'Open': new_price, 'High': new_price, 'Low': new_price, 'Close': new_price,
            'Volume': current_data_window['Volume'].iloc[-1]
        }
        
        # Carry forward ALL features for next iteration
        carry_forward_cols = [
            # Sentiment features
            'Sentiment', 'Multi_Sentiment', 'Sentiment_Confidence',
            # FII/DII features
            'FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg', 'FII_Net', 'DII_Net',
            # VIX features
            'VIX', 'VIX_Norm', 'VIX_Change', 'VIX_High'
        ]
        
        for col in carry_forward_cols:
            if col in current_data_window.columns:
                new_row_dict[col] = current_data_window[col].iloc[-1]
            
        new_row = pd.DataFrame(new_row_dict, index=[next_date])
        
        current_data_window = pd.concat([current_data_window, new_row])
        current_price = new_price
        
    return pd.DataFrame({
        'Predicted Price': future_prices,
        'Daily Change (%)': daily_changes
    }, index=future_dates)


def run_ablation_study(df: pd.DataFrame, sentiment_features: dict,
                       fii_dii_data: pd.DataFrame = None,
                       vix_data: pd.DataFrame = None,
                       multi_source_sentiment: dict = None) -> dict:
    """
    Run ablation study to measure contribution of each feature group.
    
    Tests model performance with different feature combinations removed
    to quantify the contribution of each component.
    
    Args:
        df: DataFrame with OHLCV data
        sentiment_features: Dictionary with sentiment data
        fii_dii_data: DataFrame with FII/DII data
        vix_data: DataFrame with VIX data
        multi_source_sentiment: Dictionary from multi-source sentiment
    
    Returns:
        Dictionary with ablation results for each configuration
    """
    ablation_results = {}
    
    # Full model (baseline)
    _, results_full, _, _, _, metrics_full = create_hybrid_model(
        df, sentiment_features, fii_dii_data, vix_data, multi_source_sentiment
    )
    ablation_results['full_model'] = {
        'accuracy': metrics_full['accuracy'],
        'rmse': metrics_full['rmse'],
        'description': 'Full model with all features'
    }
    
    # Without sentiment features
    _, results_no_sent, _, _, _, metrics_no_sent = create_hybrid_model(
        df, {}, fii_dii_data, vix_data, None  # Empty sentiment
    )
    ablation_results['no_sentiment'] = {
        'accuracy': metrics_no_sent['accuracy'],
        'rmse': metrics_no_sent['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_no_sent['accuracy'],
        'description': 'Without sentiment features'
    }
    
    # Without institutional features
    _, results_no_inst, _, _, _, metrics_no_inst = create_hybrid_model(
        df, sentiment_features, None, vix_data, multi_source_sentiment  # No FII/DII
    )
    ablation_results['no_institutional'] = {
        'accuracy': metrics_no_inst['accuracy'],
        'rmse': metrics_no_inst['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_no_inst['accuracy'],
        'description': 'Without FII/DII features'
    }
    
    # Without VIX features
    _, results_no_vix, _, _, _, metrics_no_vix = create_hybrid_model(
        df, sentiment_features, fii_dii_data, None, multi_source_sentiment  # No VIX
    )
    ablation_results['no_vix'] = {
        'accuracy': metrics_no_vix['accuracy'],
        'rmse': metrics_no_vix['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_no_vix['accuracy'],
        'description': 'Without VIX features'
    }
    
    # Technical only (no alternative data)
    _, results_tech_only, _, _, _, metrics_tech_only = create_hybrid_model(
        df, {}, None, None, None  # Only price/volume data
    )
    ablation_results['technical_only'] = {
        'accuracy': metrics_tech_only['accuracy'],
        'rmse': metrics_tech_only['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_tech_only['accuracy'],
        'description': 'Technical features only'
    }
    
    return ablation_results


def adjust_predictions_for_market_closures(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust predictions to show steady values on market closed days.
    
    Args:
        predictions_df: DataFrame with predictions
    
    Returns:
        Adjusted DataFrame
    """
    india_bd = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    
    # Generate business days in the prediction range
    business_days = pd.date_range(
        start=predictions_df.index.min(),
        end=predictions_df.index.max(),
        freq=india_bd
    )
    
    # Mark non-business days
    predictions_df['is_market_day'] = predictions_df.index.isin(business_days)
    
    # Forward fill predictions for non-market days
    predictions_df['adjusted_prediction'] = np.where(
        predictions_df['is_market_day'],
        predictions_df['Predicted Price'],
        np.nan
    )
    predictions_df['adjusted_prediction'] = predictions_df['adjusted_prediction'].ffill()
    
    # Calculate daily changes based on adjusted predictions
    predictions_df['Daily Change (%)'] = predictions_df['adjusted_prediction'].pct_change().fillna(0) * 100
    
    return predictions_df[['adjusted_prediction', 'Daily Change (%)']].rename(
        columns={'adjusted_prediction': 'Predicted Price'}
    )
