"""
Generate Dynamic Fusion Framework architecture diagram for Part 3 paper.
Run: python generate_fusion_architecture.py
Output: docs/figures/fig_fusion_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
data_color = '#3498DB'
tech_color = '#E74C3C'
sent_color = '#27AE60'
vol_color = '#F39C12'
weight_color = '#9B59B6'
fusion_color = '#1ABC9C'
output_color = '#2C3E50'
uncertainty_color = '#E67E22'
text_color = 'white'

def draw_box(ax, x, y, w, h, color, label, fontsize=9, alpha=0.9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#2C3E50', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    lines = label.split('\n')
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h/2 + (len(lines)/2 - i - 0.5) * 0.22,
                line, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)

def draw_arrow(ax, x1, y1, x2, y2, color='#7F8C8D', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# Title
ax.text(8, 8.5, 'Dynamic Fusion Framework with Bayesian Uncertainty Weighting',
        ha='center', fontsize=15, fontweight='bold', color='#2C3E50')

# ============ DATA SOURCES (Left) ============
ax.text(1.2, 7.6, 'Data Sources', ha='center', fontsize=11, fontweight='bold', color='#2C3E50')

draw_box(ax, 0.2, 6.2, 2.0, 1.0, data_color, 'OHLCV Data\n+ Indicators')
draw_box(ax, 0.2, 4.2, 2.0, 1.0, data_color, 'Sentiment\nRSS, News,\nReddit, Trends')
draw_box(ax, 0.2, 2.2, 2.0, 1.0, data_color, 'India VIX\n+ Stock Vol')

# ============ EXPERT MODELS (Middle-Left) ============
ax.text(4.5, 7.6, 'Expert Models', ha='center', fontsize=11, fontweight='bold', color='#2C3E50')

# Technical Expert
draw_box(ax, 3.2, 6.0, 2.6, 1.4, tech_color, 'Technical Expert\nGRU (128→64→32)\nBatchNorm + Dropout')

# Sentiment Expert
draw_box(ax, 3.2, 4.0, 2.6, 1.4, sent_color, 'Sentiment Expert\nDense (64→32→16)\n8 Features Input')

# Volatility Expert
draw_box(ax, 3.2, 2.0, 2.6, 1.4, vol_color, 'Volatility Expert\nMLP (32→16→8)\n3 VIX Features')

# Arrows: Data -> Experts
draw_arrow(ax, 2.2, 6.7, 3.2, 6.7, tech_color)
draw_arrow(ax, 2.2, 4.7, 3.2, 4.7, sent_color)
draw_arrow(ax, 2.2, 2.7, 3.2, 2.7, vol_color)

# ============ UNCERTAINTY ESTIMATION (Middle) ============
ax.text(7.8, 7.6, 'Uncertainty\nEstimation', ha='center', fontsize=10, fontweight='bold', color='#2C3E50')

draw_box(ax, 6.8, 6.2, 2.0, 1.0, uncertainty_color, 'σ²_tech\nRecent MSE')
draw_box(ax, 6.8, 4.2, 2.0, 1.0, uncertainty_color, 'σ²_sent\nRecent MSE')
draw_box(ax, 6.8, 2.2, 2.0, 1.0, uncertainty_color, 'σ²_vol\nRecent MSE')

# Arrows: Experts -> Uncertainty
draw_arrow(ax, 5.8, 6.7, 6.8, 6.7, '#E67E22')
draw_arrow(ax, 5.8, 4.7, 6.8, 4.7, '#E67E22')
draw_arrow(ax, 5.8, 2.7, 6.8, 2.7, '#E67E22')

# ============ BAYESIAN WEIGHTING (Middle-Right) ============
draw_box(ax, 9.8, 3.5, 2.4, 2.5, weight_color,
         'Bayesian\nWeight\nCalculation\n\nwᵢ = exp(-σᵢ²)\n/ Σ exp(-σⱼ²)')

# Arrows: Uncertainty -> Weighting
draw_arrow(ax, 8.8, 6.7, 9.8, 5.5, weight_color)
draw_arrow(ax, 8.8, 4.7, 9.8, 4.75, weight_color)
draw_arrow(ax, 8.8, 2.7, 9.8, 4.0, weight_color)

# ============ FUSION OUTPUT (Right) ============
draw_box(ax, 12.8, 3.7, 2.5, 2.0, fusion_color,
         'Fused\nPrediction\n\nΣ wᵢ × ŷᵢ')

draw_arrow(ax, 12.2, 4.75, 12.8, 4.7, fusion_color)

# Final Output
draw_box(ax, 13.0, 1.2, 2.0, 1.2, output_color, 'Return\nPrediction\nŷ_fused')
draw_arrow(ax, 14.05, 3.7, 14.0, 2.4, output_color)

# ============ EXAMPLE WEIGHTS (Bottom) ============
ax.text(8, 0.6, 'Example: Low Vol → w_tech=0.52, w_sent=0.31, w_vol=0.17  |  '
        'High Vol → w_tech=0.38, w_sent=0.22, w_vol=0.40',
        ha='center', fontsize=9, style='italic', color='#7F8C8D',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#BDC3C7'))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=tech_color, label='Technical Expert (GRU)'),
    mpatches.Patch(facecolor=sent_color, label='Sentiment Expert (Dense)'),
    mpatches.Patch(facecolor=vol_color, label='Volatility Expert (MLP)'),
    mpatches.Patch(facecolor=uncertainty_color, label='Uncertainty (σ²)'),
    mpatches.Patch(facecolor=weight_color, label='Bayesian Weighting'),
    mpatches.Patch(facecolor=fusion_color, label='Fused Output'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

plt.tight_layout()
plt.savefig('docs/figures/fig_fusion_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: docs/figures/fig_fusion_architecture.png")
