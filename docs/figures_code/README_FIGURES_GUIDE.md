# Figures Guide for Research Paper: Part 3 (Risk Management & Pattern Recognition)

This folder contains all the code needed to generate figures for the research paper.

## Quick Summary

| Figure # | Filename | Description | Location in Paper |
|----------|----------|-------------|-------------------|
| Fig 1 | `fig_pattern_pipeline.png` | Pattern Recognition Pipeline (Mermaid) | Section 3.1 |
| Fig 2 | `fig_peak_detection.png` | Peak/Trough Detection Example | Section 3.2 |
| Fig 3 | `fig_double_top_example.png` | Double Top Pattern Illustration | Section 3.3.1 |
| Fig 4 | `fig_double_bottom_example.png` | Double Bottom Pattern Illustration | Section 3.3.2 |
| Fig 5 | `fig_head_shoulders.png` | Head & Shoulders Pattern | Section 3.3.3 |
| Fig 6 | `fig_triangle_patterns.png` | Triangle Patterns (3 types) | Section 3.3.5 |
| Fig 7 | `fig_wedge_patterns.png` | Wedge Patterns (2 types) | Section 3.3.6 |
| Fig 8 | `fig_atr_calculation.png` | ATR Components Visualization | Section 4.1 |
| Fig 9 | `fig_stop_loss_placement.png` | Dynamic Stop-Loss Placement | Section 4.2 |
| Fig 10 | `fig_fibonacci_levels.png` | Fibonacci Retracement Levels | Section 4.4 |
| Fig 11 | `fig_kelly_criterion.png` | Kelly Criterion Curve | Section 4.5 |
| Fig 12 | `fig_position_sizing.png` | Position Sizing by Volatility | Section 4.6 |
| Fig 13 | `fig_backtest_equity.png` | Equity Curve Comparison | Section 6.3 |
| Fig 14 | `fig_pattern_accuracy.png` | Pattern Detection Accuracy | Section 6.2 |
| Fig 15 | `fig_drawdown_comparison.png` | Drawdown Analysis | Section 6.4 |
| Fig 16 | `fig_crisis_performance.png` | Crisis Period Performance | Section 6.5 |
| Fig 17 | `fig_ablation_study.png` | Ablation Study Results | Section 6.6 |
| Fig 18 | `fig_system_architecture.png` | Full System Architecture (Mermaid) | Section 1.3 |

## How to Generate Figures

### Step 1: Run the Python script
```bash
cd C:\Users\divya\Desktop\finance\docs\figures_code
python generate_all_figures.py
```

### Step 2: For Mermaid diagrams
- Use https://mermaid.live/ to render the .mmd files
- Or use VS Code with Mermaid extension
- Export as PNG at 2x resolution for print quality

### Step 3: Place generated PNGs in `docs/figures/` folder

### Step 4: Update LaTeX to reference figures
The paper already has `\graphicspath{{figures/}}` set up.

## File Naming Convention

All generated figures should be saved as:
- Format: PNG (300 DPI for print quality)
- Naming: `fig_descriptive_name.png`
- Location: `C:\Users\divya\Desktop\finance\docs\figures\`

## Color Scheme

Use consistent colors throughout:
- Primary Blue: #1f77b4
- Green (Bullish): #2ca02c
- Red (Bearish): #d62728
- Orange (Warning): #ff7f0e
- Gray (Neutral): #7f7f7f
- Purple (Highlight): #9467bd
