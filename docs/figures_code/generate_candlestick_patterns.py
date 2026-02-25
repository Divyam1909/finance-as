"""
Candlestick Chart Pattern Generator
Creates professional candlestick charts showing pattern examples with real OHLC data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DPI = 300

COLORS = {
    'green': '#2ca02c',
    'red': '#d62728',
    'gray': '#7f7f7f'
}


def plot_candlesticks(ax, df, start_idx=0, end_idx=None):
    """Plot candlestick chart on given axes."""
    if end_idx is None:
        end_idx = len(df)
    
    df_plot = df.iloc[start_idx:end_idx].copy()
    df_plot = df_plot.reset_index(drop=True)
    
    width = 0.6
    
    for idx in range(len(df_plot)):
        row = df_plot.iloc[idx]
        is_green = row['Close'] >= row['Open']
        color = COLORS['green'] if is_green else COLORS['red']
        
        # Body
        body_bottom = min(row['Open'], row['Close'])
        body_height = abs(row['Close'] - row['Open'])
        rect = Rectangle((idx - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Wicks
        ax.plot([idx, idx], [row['Low'], body_bottom], color='black', linewidth=0.8)
        ax.plot([idx, idx], [body_bottom + body_height, row['High']], color='black', linewidth=0.8)
    
    ax.set_xlim(-1, len(df_plot))
    ax.autoscale_view()


def generate_double_top_candlestick():
    """Generate realistic double top with candlesticks."""
    np.random.seed(42)
    
    n = 60
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    # Create price action
    close = np.zeros(n)
    
    # Initial uptrend
    close[:15] = np.linspace(100, 115, 15) + np.random.randn(15) * 0.5
    # First peak
    close[15:22] = 118 + 3*np.sin(np.linspace(0, np.pi, 7)) + np.random.randn(7) * 0.3
    # Pullback
    close[22:30] = np.linspace(117, 112, 8) + np.random.randn(8) * 0.4
    # Second peak
    close[30:38] = 112 + (118-112)*np.sin(np.linspace(0, np.pi, 8)/2)**2 + np.random.randn(8) * 0.3
    close[35:38] = 120 + np.random.randn(3) * 0.3
    # Breakdown
    close[38:45] = np.linspace(118, 111, 7) + np.random.randn(7) * 0.5
    close[45:] = np.linspace(110, 102, n-45) + np.random.randn(n-45) * 0.6
    
    # Generate OHLC
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    open_prices = close + np.random.randn(n) * 0.8
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_candlesticks(ax, df)
    
    # Mark pattern
    # First peak
    ax.scatter([18], [df['High'].iloc[18]], c=COLORS['red'], s=150, marker='^', zorder=10)
    ax.annotate('Peak 1', (18, df['High'].iloc[18]+1), ha='center', fontsize=11, fontweight='bold')
    
    # Second peak
    ax.scatter([36], [df['High'].iloc[36]], c=COLORS['red'], s=150, marker='^', zorder=10)
    ax.annotate('Peak 2', (36, df['High'].iloc[36]+1), ha='center', fontsize=11, fontweight='bold')
    
    # Neckline
    neckline = 111
    ax.axhline(y=neckline, color='orange', linestyle='--', linewidth=2, label='Neckline')
    ax.annotate('Neckline Support', (55, neckline+0.5), fontsize=10, color='orange')
    
    # Breakout arrow
    ax.annotate('', xy=(42, 109), xytext=(42, 114),
               arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2))
    ax.annotate('Breakdown!', (43, 111), fontsize=11, fontweight='bold', color=COLORS['red'])
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Double Top Pattern - Candlestick Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(OUTPUT_DIR, 'fig_candlestick_double_top.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_head_shoulders_candlestick():
    """Generate realistic head and shoulders with candlesticks."""
    np.random.seed(123)
    
    n = 80
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    close = np.zeros(n)
    
    # Initial rise
    close[:10] = np.linspace(100, 108, 10) + np.random.randn(10) * 0.3
    # Left shoulder
    close[10:20] = 108 + 5*np.sin(np.linspace(0, np.pi, 10)) + np.random.randn(10) * 0.3
    # Down to neckline
    close[20:25] = np.linspace(113, 107, 5) + np.random.randn(5) * 0.3
    # Head (higher)
    close[25:40] = 107 + 13*np.sin(np.linspace(0, np.pi, 15)) + np.random.randn(15) * 0.4
    # Down to neckline
    close[40:45] = np.linspace(120, 107, 5) + np.random.randn(5) * 0.3
    # Right shoulder
    close[45:55] = 107 + 5*np.sin(np.linspace(0, np.pi, 10)) + np.random.randn(10) * 0.3
    # Breakdown
    close[55:65] = np.linspace(112, 105, 10) + np.random.randn(10) * 0.4
    close[65:] = np.linspace(104, 95, n-65) + np.random.randn(n-65) * 0.5
    
    high = close + np.abs(np.random.randn(n)) * 1.2
    low = close - np.abs(np.random.randn(n)) * 1.2
    open_prices = close + np.random.randn(n) * 0.6
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close
    })
    
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_candlesticks(ax, df)
    
    # Mark pattern components
    # Left shoulder
    ls_idx = 15
    ax.scatter([ls_idx], [df['High'].iloc[ls_idx]], c='orange', s=150, marker='^', zorder=10)
    ax.annotate('Left\nShoulder', (ls_idx, df['High'].iloc[ls_idx]+1.5), ha='center', fontsize=10, fontweight='bold')
    
    # Head
    head_idx = 32
    ax.scatter([head_idx], [df['High'].iloc[head_idx]], c=COLORS['red'], s=200, marker='^', zorder=10)
    ax.annotate('Head', (head_idx, df['High'].iloc[head_idx]+1), ha='center', fontsize=12, fontweight='bold')
    
    # Right shoulder
    rs_idx = 50
    ax.scatter([rs_idx], [df['High'].iloc[rs_idx]], c='orange', s=150, marker='^', zorder=10)
    ax.annotate('Right\nShoulder', (rs_idx, df['High'].iloc[rs_idx]+1.5), ha='center', fontsize=10, fontweight='bold')
    
    # Neckline
    neckline = 107
    ax.plot([20, 75], [neckline, neckline], color=COLORS['green'], linestyle='--', linewidth=2, label='Neckline')
    
    # Breakdown point
    ax.scatter([58], [neckline], c=COLORS['red'], s=200, marker='x', zorder=10)
    ax.annotate('Neckline\nBreak', (58, neckline-3), ha='center', fontsize=10, fontweight='bold', color=COLORS['red'])
    
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Head and Shoulders Pattern - Candlestick Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(OUTPUT_DIR, 'fig_candlestick_head_shoulders.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


def generate_volume_confirmation():
    """Generate chart showing volume confirmation of patterns."""
    np.random.seed(42)
    
    n = 50
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    # Price data - breakout scenario
    close = np.zeros(n)
    close[:20] = 100 + np.random.randn(20) * 0.8  # Consolidation
    close[20:30] = np.linspace(100, 98, 10) + np.random.randn(10) * 0.5  # Squeeze
    close[30:35] = np.linspace(98, 105, 5) + np.random.randn(5) * 0.3  # Breakout
    close[35:] = np.linspace(106, 115, n-35) + np.random.randn(n-35) * 0.6  # Continuation
    
    high = close + np.abs(np.random.randn(n)) * 1
    low = close - np.abs(np.random.randn(n)) * 1
    open_prices = close + np.random.randn(n) * 0.5
    
    # Volume data - spike on breakout
    volume = np.abs(np.random.randn(n)) * 50000 + 100000
    volume[30:35] = volume[30:35] * 2.5  # Volume spike
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1], sharex=True)
    
    # Price chart
    plot_candlesticks(ax1, df)
    
    # Resistance line
    ax1.axhline(y=102, color='red', linestyle='--', linewidth=1.5, label='Resistance')
    
    # Breakout annotation
    ax1.annotate('Breakout!', (32, 107), fontsize=12, fontweight='bold', color=COLORS['green'])
    ax1.annotate('', xy=(32, 105), xytext=(32, 102),
                arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2))
    
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Volume Confirmation of Breakout', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    colors = [COLORS['green'] if df['Close'].iloc[i] >= df['Open'].iloc[i] else COLORS['red'] 
              for i in range(len(df))]
    ax2.bar(range(len(df)), df['Volume'], color=colors, alpha=0.7, width=0.8)
    
    # Volume average
    vol_avg = df['Volume'].rolling(20).mean()
    ax2.plot(range(len(df)), vol_avg, color='blue', linewidth=2, label='20-day Avg Volume')
    
    # Highlight volume spike
    ax2.axvspan(30, 35, alpha=0.3, color='yellow', label='Volume Surge')
    ax2.annotate('2.5x Average\nVolume', (32, df['Volume'].iloc[32]), fontsize=10, 
                fontweight='bold', ha='center')
    
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_volume_confirmation.png')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    print("Generating candlestick pattern charts...")
    generate_double_top_candlestick()
    generate_head_shoulders_candlestick()
    generate_volume_confirmation()
    print("Done!")
