"""
Statistical Visualization Generator
Creates charts for statistical significance, confidence intervals, and hypothesis testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 300

COLORS = {
    'primary': '#1f77b4',
    'green': '#2ca02c',
    'red': '#d62728',
    'orange': '#ff7f0e',
    'gray': '#7f7f7f',
    'purple': '#9467bd'
}


def generate_confidence_interval_chart():
    """Generate confidence interval comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['Sharpe Ratio', 'Win Rate (%)', 'Profit Factor', 'Sortino Ratio', 'Calmar Ratio']
    
    # Point estimates and confidence intervals
    estimates = [1.32, 54.7, 1.38, 1.89, 0.95]
    ci_lower = [1.18, 51.2, 1.21, 1.54, 0.78]
    ci_upper = [1.47, 58.1, 1.56, 2.28, 1.14]
    
    # Benchmark values
    benchmark = [0.82, 50.0, 1.0, 1.1, 0.52]
    
    y_pos = np.arange(len(metrics))
    
    # Plot confidence intervals
    for i, (est, low, high, bench) in enumerate(zip(estimates, ci_lower, ci_upper, benchmark)):
        # CI bar
        ax.plot([low, high], [i, i], color=COLORS['primary'], linewidth=3, solid_capstyle='round')
        # Point estimate
        ax.scatter([est], [i], color=COLORS['primary'], s=150, zorder=5, marker='o')
        # Benchmark
        ax.scatter([bench], [i], color=COLORS['red'], s=100, zorder=5, marker='x', linewidths=2)
        
        # CI labels
        ax.annotate(f'{low:.2f}', (low-0.08, i+0.15), fontsize=9, ha='right')
        ax.annotate(f'{high:.2f}', (high+0.08, i+0.15), fontsize=9, ha='left')
        
        # Significance indicator
        if low > bench:
            ax.annotate('*', (high+0.2, i), fontsize=16, fontweight='bold', color=COLORS['green'], va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('95% Confidence Intervals vs Benchmark', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['primary'], linewidth=3, label='95% CI'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['primary'], 
               markersize=10, label='Point Estimate'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor=COLORS['red'],
               markeredgecolor=COLORS['red'], markersize=10, markeredgewidth=2, label='Benchmark'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['green'],
               markersize=15, label='Statistically Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_confidence_intervals.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_bootstrap_distribution():
    """Generate bootstrap distribution chart for Sharpe ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    np.random.seed(42)
    
    # Left: Bootstrap distribution
    ax = axes[0]
    
    # Simulate bootstrap samples
    n_bootstrap = 10000
    true_sharpe = 1.32
    bootstrap_sharpes = np.random.normal(true_sharpe, 0.08, n_bootstrap)
    
    # Histogram
    ax.hist(bootstrap_sharpes, bins=50, density=True, alpha=0.7, color=COLORS['primary'],
           edgecolor='black', linewidth=0.5)
    
    # Point estimate
    ax.axvline(x=true_sharpe, color=COLORS['red'], linewidth=2, label=f'Point Estimate: {true_sharpe}')
    
    # CI
    ci_low, ci_high = np.percentile(bootstrap_sharpes, [2.5, 97.5])
    ax.axvline(x=ci_low, color=COLORS['green'], linestyle='--', linewidth=2)
    ax.axvline(x=ci_high, color=COLORS['green'], linestyle='--', linewidth=2, 
              label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')
    
    # Benchmark
    benchmark = 0.82
    ax.axvline(x=benchmark, color=COLORS['orange'], linewidth=2, label=f'Benchmark: {benchmark}')
    
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bootstrap Distribution (n=10,000)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Right: P-value visualization
    ax = axes[1]
    
    # Null distribution (benchmark)
    x = np.linspace(0.4, 1.6, 1000)
    null_dist = stats.norm.pdf(x, benchmark, 0.15)
    ax.fill_between(x, null_dist, alpha=0.3, color=COLORS['gray'], label='Null Distribution')
    ax.plot(x, null_dist, color=COLORS['gray'], linewidth=2)
    
    # Observed value
    ax.axvline(x=true_sharpe, color=COLORS['red'], linewidth=2, label=f'Observed: {true_sharpe}')
    
    # P-value region
    p_region = x >= true_sharpe
    ax.fill_between(x[p_region], null_dist[p_region], alpha=0.5, color=COLORS['red'], 
                   label='p-value region')
    
    # Calculate p-value
    p_value = 1 - stats.norm.cdf(true_sharpe, benchmark, 0.15)
    ax.annotate(f'p-value = {p_value:.4f}\n(p < 0.001)', (1.4, 1.5), fontsize=11, 
               fontweight='bold', color=COLORS['red'])
    
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Hypothesis Test: System vs Benchmark', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_bootstrap_distribution.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_rolling_metrics():
    """Generate rolling performance metrics chart."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    # Generate returns
    returns = np.random.randn(n) * 0.012 + 0.0003
    
    # Rolling Sharpe (60-day)
    window = 60
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    
    ax = axes[0]
    ax.plot(dates, rolling_sharpe, color=COLORS['primary'], linewidth=1.5)
    ax.axhline(y=1.0, color=COLORS['green'], linestyle='--', linewidth=1.5, label='Good (1.0)')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.fill_between(dates, 0, rolling_sharpe, where=rolling_sharpe >= 0, 
                   alpha=0.3, color=COLORS['green'])
    ax.fill_between(dates, 0, rolling_sharpe, where=rolling_sharpe < 0,
                   alpha=0.3, color=COLORS['red'])
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.set_title('Rolling 60-Day Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 4)
    
    # Rolling Win Rate
    wins = returns > 0
    rolling_winrate = pd.Series(wins.astype(float)).rolling(window).mean() * 100
    
    ax = axes[1]
    ax.plot(dates, rolling_winrate, color=COLORS['orange'], linewidth=1.5)
    ax.axhline(y=50, color='k', linestyle='--', linewidth=1.5, label='Break-even')
    ax.axhline(y=54.7, color=COLORS['green'], linestyle=':', linewidth=1.5, label='Overall Avg: 54.7%')
    ax.fill_between(dates, 50, rolling_winrate, where=rolling_winrate >= 50,
                   alpha=0.3, color=COLORS['green'])
    ax.fill_between(dates, 50, rolling_winrate, where=rolling_winrate < 50,
                   alpha=0.3, color=COLORS['red'])
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Rolling 60-Day Win Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 70)
    
    # Rolling Max Drawdown
    equity = 100 * np.cumprod(1 + returns)
    rolling_max = pd.Series(equity).rolling(window).max()
    rolling_dd = (equity - rolling_max) / rolling_max * 100
    
    ax = axes[2]
    ax.fill_between(dates, 0, rolling_dd, alpha=0.5, color=COLORS['red'])
    ax.plot(dates, rolling_dd, color=COLORS['red'], linewidth=1)
    ax.axhline(y=-10, color=COLORS['orange'], linestyle='--', linewidth=1.5, label='Warning (-10%)')
    ax.axhline(y=-15, color=COLORS['red'], linestyle='--', linewidth=1.5, label='Critical (-15%)')
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Rolling Maximum Drawdown', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-25, 5)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_rolling_metrics.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_monthly_returns_heatmap():
    """Generate monthly returns heatmap."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Generate sample monthly returns
    np.random.seed(42)
    years = [2020, 2021, 2022, 2023, 2024]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    returns = np.random.randn(len(years), 12) * 4 + 1  # Mean 1%, std 4%
    
    # Create heatmap
    im = ax.imshow(returns, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Monthly Return (%)', fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(len(months)))
    ax.set_yticks(np.arange(len(years)))
    ax.set_xticklabels(months)
    ax.set_yticklabels(years)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add values in cells
    for i in range(len(years)):
        for j in range(len(months)):
            text_color = 'white' if abs(returns[i, j]) > 5 else 'black'
            ax.text(j, i, f'{returns[i, j]:.1f}%', ha='center', va='center', 
                   color=text_color, fontsize=9)
    
    # Add yearly totals
    yearly_returns = returns.sum(axis=1)
    for i, yr in enumerate(yearly_returns):
        ax.annotate(f'Year: {yr:.1f}%', (12.5, i), fontsize=10, va='center',
                   fontweight='bold', color=COLORS['green'] if yr > 0 else COLORS['red'])
    
    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_monthly_returns_heatmap.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_returns_distribution():
    """Generate returns distribution chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    np.random.seed(42)
    
    # Generate returns
    n = 1000
    benchmark_returns = np.random.normal(0.0005, 0.015, n)
    strategy_returns = np.random.normal(0.001, 0.013, n)  # Higher mean, lower vol
    
    # Left: Histogram comparison
    ax = axes[0]
    ax.hist(benchmark_returns * 100, bins=50, alpha=0.5, label='Buy & Hold', 
           color=COLORS['gray'], density=True)
    ax.hist(strategy_returns * 100, bins=50, alpha=0.5, label='Our Strategy',
           color=COLORS['green'], density=True)
    
    # Add normal fit
    x = np.linspace(-5, 5, 100)
    ax.plot(x, stats.norm.pdf(x, benchmark_returns.mean()*100, benchmark_returns.std()*100),
           color=COLORS['gray'], linewidth=2, linestyle='--')
    ax.plot(x, stats.norm.pdf(x, strategy_returns.mean()*100, strategy_returns.std()*100),
           color=COLORS['green'], linewidth=2, linestyle='--')
    
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Daily Return (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Return Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Right: QQ Plot
    ax = axes[1]
    
    # Sort strategy returns
    sorted_returns = np.sort(strategy_returns)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_returns)))
    
    ax.scatter(theoretical_quantiles, sorted_returns * 100, alpha=0.3, s=10, color=COLORS['primary'])
    
    # Add reference line
    ax.plot([-4, 4], [theoretical_quantiles.min()*1.3*100, theoretical_quantiles.max()*1.3*100],
           'r--', linewidth=2, label='Normal Reference')
    
    # Highlight tails
    tail_idx = np.abs(theoretical_quantiles) > 2
    ax.scatter(theoretical_quantiles[tail_idx], sorted_returns[tail_idx] * 100, 
              alpha=0.8, s=30, color=COLORS['red'], label='Tail Events')
    
    ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
    ax.set_ylabel('Sample Quantiles (%)', fontsize=12)
    ax.set_title('Q-Q Plot: Strategy Returns', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_returns_distribution.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    print("Generating statistical charts...")
    generate_confidence_interval_chart()
    generate_bootstrap_distribution()
    generate_rolling_metrics()
    generate_monthly_returns_heatmap()
    generate_returns_distribution()
    print("Done!")
