"""
Vectorized Backtester for strategy evaluation.
Fast backtesting engine for evaluating trading strategies.
Includes statistical significance testing for research-grade analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import TradingConfig


def calculate_statistical_significance(predictions: np.ndarray, actuals: np.ndarray, 
                                       n_bootstrap: int = 1000) -> dict:
    """
    Calculate statistical significance of prediction performance.
    
    Uses paired t-test against random walk baseline and bootstrap
    confidence intervals for direction accuracy.
    
    Args:
        predictions: Array of predicted returns
        actuals: Array of actual returns
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary with p-value, confidence intervals, and test statistics
    """
    # Direction accuracy
    correct = np.sign(predictions) == np.sign(actuals)
    accuracy = np.mean(correct)
    
    # Paired t-test: Is accuracy significantly different from 50%?
    # H0: accuracy = 0.5 (random guessing)
    # Using binomial test for proportion
    n_correct = np.sum(correct)
    n_total = len(correct)
    
    # Binomial test (two-sided)
    p_value_binomial = stats.binom_test(n_correct, n_total, p=0.5, alternative='greater')
    
    # Bootstrap confidence interval for accuracy
    bootstrap_accuracies = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(correct), size=len(correct), replace=True)
        bootstrap_acc = np.mean(correct[indices])
        bootstrap_accuracies.append(bootstrap_acc)
    
    ci_lower = np.percentile(bootstrap_accuracies, 2.5)
    ci_upper = np.percentile(bootstrap_accuracies, 97.5)
    
    # RMSE comparison with random walk (predict 0)
    rmse_model = np.sqrt(np.mean((predictions - actuals) ** 2))
    rmse_random_walk = np.sqrt(np.mean(actuals ** 2))  # Predicting 0
    
    # Paired t-test on squared errors
    squared_errors_model = (predictions - actuals) ** 2
    squared_errors_rw = actuals ** 2
    t_stat, p_value_rmse = stats.ttest_rel(squared_errors_model, squared_errors_rw)
    
    # Effect size (Cohen's d)
    diff = squared_errors_rw - squared_errors_model
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)
    
    return {
        'accuracy': accuracy,
        'p_value_accuracy': p_value_binomial,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rmse_model': rmse_model,
        'rmse_baseline': rmse_random_walk,
        'p_value_rmse': p_value_rmse if t_stat < 0 else 1 - p_value_rmse/2,  # One-sided
        't_statistic': t_stat,
        'cohens_d': cohens_d,
        'n_samples': n_total
    }


def calculate_sharpe_significance(strategy_returns: np.ndarray, 
                                  benchmark_returns: np.ndarray,
                                  n_bootstrap: int = 1000) -> dict:
    """
    Calculate statistical significance of Sharpe ratio difference.
    
    Args:
        strategy_returns: Array of strategy daily returns
        benchmark_returns: Array of benchmark daily returns
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dictionary with Sharpe ratios, p-value, and confidence intervals
    """
    def calc_sharpe(returns):
        if np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    sharpe_strategy = calc_sharpe(strategy_returns)
    sharpe_benchmark = calc_sharpe(benchmark_returns)
    sharpe_diff = sharpe_strategy - sharpe_benchmark
    
    # Bootstrap test for Sharpe difference
    bootstrap_diffs = []
    n = len(strategy_returns)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_strategy = calc_sharpe(strategy_returns[indices])
        boot_benchmark = calc_sharpe(benchmark_returns[indices])
        bootstrap_diffs.append(boot_strategy - boot_benchmark)
    
    # P-value: proportion of bootstrap samples where diff <= 0
    p_value = np.mean(np.array(bootstrap_diffs) <= 0)
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return {
        'sharpe_strategy': sharpe_strategy,
        'sharpe_benchmark': sharpe_benchmark,
        'sharpe_difference': sharpe_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


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
        
        # Calculate statistical significance if we have predictions
        stat_sig = None
        if 'Predicted_Return' in self.data.columns:
            predictions = self.data['Predicted_Return'].values
            actuals = self.data['Actual_Return'].values
            stat_sig = calculate_statistical_significance(predictions, actuals)
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Profit Factor": profit_factor,
            "Equity Curve": df['Equity_Curve'],
            "Strategy Returns": df['Strategy_Return'],
            "Statistical Significance": stat_sig
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
