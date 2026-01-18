"""
Vectorized Backtester for strategy evaluation.
Fast backtesting engine for evaluating trading strategies.
"""

import numpy as np
import pandas as pd

from config.settings import TradingConfig


class VectorizedBacktester:
    """
    Fast Vectorized Backtesting Engine.
    
    Evaluates trading strategies based on return predictions
    and calculates key performance metrics.
    """
    
    def __init__(self, data: pd.DataFrame, signals: pd.Series):
        """
        Initialize the backtester.
        
        Args:
            data: DataFrame with 'Actual_Return' column
            signals: Series of trading signals (-1, 0, 1)
        """
        self.data = data
        self.signals = signals
        
    def run_backtest(self, initial_capital: float = None) -> dict:
        """
        Run vectorized backtest based on returns.
        
        Strategy: If Signal[T] is BUY (1), get Return[T]
                  If Signal[T] is SELL (-1), get -Return[T]
                  If Signal[T] is HOLD (0), get 0
        
        Args:
            initial_capital: Starting capital (default from config)
        
        Returns:
            Dictionary with performance metrics and equity curve
        """
        initial_capital = initial_capital or TradingConfig.DEFAULT_INITIAL_CAPITAL
        df = self.data.copy()
        
        # Strategy Return = Signal * Actual_Return
        df['Strategy_Return'] = self.signals * df['Actual_Return']
        
        # Equity Curve
        df['Equity_Curve'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
        
        # Calculate Metrics
        total_return = (df['Equity_Curve'].iloc[-1] / initial_capital) - 1
        
        # Sharpe Ratio (annualized)
        if df['Strategy_Return'].std() != 0:
            sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Drawdown
        cumulative_returns = (1 + df['Strategy_Return']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win Rate
        winning_trades = df[df['Strategy_Return'] > 0]
        total_trades = df[df['Strategy_Return'] != 0]
        win_rate = len(winning_trades) / len(total_trades) if len(total_trades) > 0 else 0
        
        # Additional Metrics
        avg_win = winning_trades['Strategy_Return'].mean() if len(winning_trades) > 0 else 0
        losing_trades = df[df['Strategy_Return'] < 0]
        avg_loss = losing_trades['Strategy_Return'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['Strategy_Return'].sum() / losing_trades['Strategy_Return'].sum()) \
            if len(losing_trades) > 0 and losing_trades['Strategy_Return'].sum() != 0 else float('inf')
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Equity Curve": df['Equity_Curve'],
            "Strategy Returns": df['Strategy_Return']
        }

    def generate_signals_from_predictions(self, predictions: pd.Series, 
                                          threshold: float = None) -> pd.Series:
        """
        Generate trading signals from return predictions.
        
        Args:
            predictions: Series of predicted returns
            threshold: Minimum predicted return to trigger signal
        
        Returns:
            Series of trading signals (-1, 0, 1)
        """
        threshold = threshold or TradingConfig.SIGNAL_THRESHOLD
        
        signals = pd.Series(0, index=predictions.index)
        signals[predictions > threshold] = 1  # Long
        signals[predictions < -threshold] = -1  # Short
        
        return signals
    
    def calculate_metrics_summary(self, backtest_results: dict) -> pd.DataFrame:
        """
        Create a summary DataFrame of backtest metrics.
        
        Args:
            backtest_results: Dictionary from run_backtest()
        
        Returns:
            DataFrame with metrics summary
        """
        metrics = {
            "Total Return": f"{backtest_results['Total Return']*100:.2f}%",
            "Sharpe Ratio": f"{backtest_results['Sharpe Ratio']:.2f}",
            "Max Drawdown": f"{backtest_results['Max Drawdown']*100:.2f}%",
            "Win Rate": f"{backtest_results['Win Rate']*100:.1f}%",
            "Profit Factor": f"{backtest_results['Profit Factor']:.2f}"
        }
        
        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
