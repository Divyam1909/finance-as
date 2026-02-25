# Complete Figure Placement Guide for Part 3 Paper

This document tells you EXACTLY where each figure goes in the LaTeX paper.

---

## Section 1: Introduction

### Figure 18: System Architecture
- **File**: `fig_system_architecture.png`
- **Source**: `mermaid_fig18_system_architecture.mmd`
- **Placement**: After paragraph introducing the framework (around line 85)
- **Caption**: "Complete system architecture showing the integration of pattern recognition, risk management, and backtesting modules."
- **LaTeX Code**:
```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\textwidth]{fig_system_architecture.png}
    \caption{Complete system architecture integrating pattern recognition, risk management, and statistical validation modules.}
    \label{fig:system_architecture}
\end{figure}
```

---

## Section 3: Pattern Recognition Module

### Figure 1: Pattern Recognition Pipeline
- **File**: `fig_pattern_pipeline.png`
- **Source**: `mermaid_fig01_pattern_pipeline.mmd`
- **Placement**: Start of Section 3.1
- **Caption**: "Pattern recognition pipeline from raw OHLCV data to validated pattern outputs."

### Figure 2: Peak and Trough Detection
- **File**: `fig_peak_detection.png`
- **Source**: Python `generate_peak_detection()`
- **Placement**: Section 3.2, after discussing prominence filtering
- **Caption**: "Peak and trough detection using scipy.signal.find\_peaks with prominence filtering."

### Figure 3: Double Top Pattern
- **File**: `fig_double_top_example.png`
- **Source**: Python `generate_double_top()`
- **Placement**: Section 3.3.1
- **Caption**: "Anatomy of a Double Top pattern showing key components: two peaks at similar levels, neckline support, and price target calculation."

### Figure 4: Double Bottom Pattern
- **File**: `fig_double_bottom_example.png`
- **Source**: Python `generate_double_bottom()`
- **Placement**: Section 3.3.2
- **Caption**: "Anatomy of a Double Bottom pattern showing bullish reversal structure."

### Figure 5: Head and Shoulders
- **File**: `fig_head_shoulders.png`
- **Source**: Python `generate_head_shoulders()`
- **Placement**: Section 3.3.3
- **Caption**: "Head and Shoulders pattern with shoulder height requirements and neckline identification."

### Figure 6: Triangle Patterns
- **File**: `fig_triangle_patterns.png`
- **Source**: Python `generate_triangle_patterns()`
- **Placement**: Section 3.3.5
- **Caption**: "Three types of triangle patterns: ascending (bullish), descending (bearish), and symmetrical (neutral)."

### Figure 7: Wedge Patterns
- **File**: `fig_wedge_patterns.png`
- **Source**: Python `generate_wedge_patterns()`
- **Placement**: Section 3.3.6
- **Caption**: "Rising wedge (bearish reversal) and falling wedge (bullish reversal) patterns."

### Supplementary: Pattern Decision Tree
- **File**: `fig_pattern_decision_tree.png`
- **Source**: `mermaid_fig_pattern_decision_tree.mmd`
- **Placement**: Appendix B or Section 3.4
- **Caption**: "Decision tree for pattern classification based on peak/trough configurations."

---

## Section 4: Risk Management Module

### Figure 8: ATR Calculation
- **File**: `fig_atr_calculation.png`
- **Source**: Python `generate_atr_calculation()`
- **Placement**: Section 4.1, after ATR formula
- **Caption**: "True Range components and ATR calculation showing high-low range, gap components, and 14-day moving average."

### Figure 9: Stop-Loss Placement
- **File**: `fig_stop_loss_placement.png`
- **Source**: Python `generate_stop_loss_placement()`
- **Placement**: Section 4.2
- **Caption**: "Dynamic ATR-based stop-loss placement with confidence-adjusted multipliers and risk-reward visualization."

### Figure 10: Fibonacci Levels
- **File**: `fig_fibonacci_levels.png`
- **Source**: Python `generate_fibonacci_levels()`
- **Placement**: Section 4.4
- **Caption**: "Fibonacci retracement levels from 90-day swing high/low with golden zone (50\%-61.8\%) highlighted."

### Figure 11: Kelly Criterion
- **File**: `fig_kelly_criterion.png`
- **Source**: Python `generate_kelly_criterion()`
- **Placement**: Section 4.5
- **Caption**: "Left: Expected log growth rate as function of bet fraction for different scenarios. Right: Equity curves comparing full Kelly vs fractional Kelly sizing."

### Figure 12: Position Sizing by Volatility
- **File**: `fig_position_sizing.png`
- **Source**: Python `generate_position_sizing()`
- **Placement**: Section 4.6
- **Caption**: "VIX-based volatility regime detection and corresponding position size multipliers."

### Supplementary: Risk Workflow
- **File**: `fig_risk_workflow.png`
- **Source**: `mermaid_fig_risk_workflow.mmd`
- **Placement**: Start of Section 4 or Appendix
- **Caption**: "Risk management workflow from signal generation to trade setup output."

---

## Section 6: Experimental Results

### Figure 13: Equity Curve Comparison
- **File**: `fig_backtest_equity.png`
- **Source**: Python `generate_backtest_equity()`
- **Placement**: Section 6.3
- **Caption**: "Equity curve comparison: Buy \& Hold vs Technical Only vs Risk Management vs Full System over the test period (2023-2024)."

### Figure 14: Pattern Accuracy
- **File**: `fig_pattern_accuracy.png`
- **Source**: Python `generate_pattern_accuracy()`
- **Placement**: Section 6.2
- **Caption**: "Pattern detection precision by pattern type with sample sizes (n) indicated."

### Figure 15: Drawdown Comparison
- **File**: `fig_drawdown_comparison.png`
- **Source**: Python `generate_drawdown_comparison()`
- **Placement**: Section 6.4
- **Caption**: "Drawdown analysis comparing Buy \& Hold maximum drawdown vs system-managed drawdown with equity curves."

### Figure 16: Crisis Performance
- **File**: `fig_crisis_performance.png`
- **Source**: Python `generate_crisis_performance()`
- **Placement**: Section 6.5
- **Caption**: "System performance during crisis periods: Indian General Election (Apr-Jun 2024) and FII Selling (Oct-Nov 2024)."

### Figure 17: Ablation Study
- **File**: `fig_ablation_study.png`
- **Source**: Python `generate_ablation_study()`
- **Placement**: Section 6.6
- **Caption**: "Ablation study showing contribution of each component to overall system performance measured by Sharpe ratio and maximum drawdown."

### Supplementary: Backtest Flow
- **File**: `fig_backtest_flow.png`
- **Source**: `mermaid_fig_backtest_flow.mmd`
- **Placement**: Section 5 or Appendix
- **Caption**: "Vectorized backtesting workflow with statistical validation."

---

## How to Generate All Figures

### Python Figures (matplotlib)
```bash
cd C:\Users\divya\Desktop\finance\docs\figures_code
pip install matplotlib numpy pandas scipy  # if not installed
python generate_all_figures.py
```
Output goes to: `C:\Users\divya\Desktop\finance\docs\figures\`

### Mermaid Diagrams
1. Go to https://mermaid.live/
2. Copy content from `.mmd` files
3. Export as PNG (2x resolution for print)
4. Save to `docs/figures/` with correct filename

Or use VS Code:
1. Install "Markdown Preview Mermaid Support" extension
2. Open `.mmd` file
3. Right-click > Export as PNG

---

## Summary Table

| Section | Figure # | Filename | Type | Dimensions |
|---------|----------|----------|------|------------|
| 1.3 | 18 | fig_system_architecture | Mermaid | Full width |
| 3.1 | 1 | fig_pattern_pipeline | Mermaid | Full width |
| 3.2 | 2 | fig_peak_detection | Python | 12×6 |
| 3.3.1 | 3 | fig_double_top_example | Python | 12×7 |
| 3.3.2 | 4 | fig_double_bottom_example | Python | 12×7 |
| 3.3.3 | 5 | fig_head_shoulders | Python | 14×7 |
| 3.3.5 | 6 | fig_triangle_patterns | Python | 15×5 |
| 3.3.6 | 7 | fig_wedge_patterns | Python | 12×5 |
| 4.1 | 8 | fig_atr_calculation | Python | 12×8 |
| 4.2 | 9 | fig_stop_loss_placement | Python | 12×7 |
| 4.4 | 10 | fig_fibonacci_levels | Python | 12×8 |
| 4.5 | 11 | fig_kelly_criterion | Python | 14×6 |
| 4.6 | 12 | fig_position_sizing | Python | 12×8 |
| 6.2 | 14 | fig_pattern_accuracy | Python | 12×7 |
| 6.3 | 13 | fig_backtest_equity | Python | 14×7 |
| 6.4 | 15 | fig_drawdown_comparison | Python | 14×8 |
| 6.5 | 16 | fig_crisis_performance | Python | 14×6 |
| 6.6 | 17 | fig_ablation_study | Python | 12×7 |
| App | - | fig_pattern_decision_tree | Mermaid | Full width |
| App | - | fig_risk_workflow | Mermaid | Full width |
| App | - | fig_backtest_flow | Mermaid | Full width |

---

## LaTeX Preamble Requirements

Make sure your paper has these in the preamble:
```latex
\usepackage{graphicx}
\usepackage{float}
\graphicspath{{figures/}}
```

---

## Color Legend (for consistency)

| Element | Hex Color | RGB |
|---------|-----------|-----|
| Primary Blue | #1f77b4 | (31,119,180) |
| Bullish Green | #2ca02c | (44,160,44) |
| Bearish Red | #d62728 | (214,39,40) |
| Warning Orange | #ff7f0e | (255,127,14) |
| Neutral Gray | #7f7f7f | (127,127,127) |
| Highlight Purple | #9467bd | (148,103,189) |
