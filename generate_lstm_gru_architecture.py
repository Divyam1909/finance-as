"""
Generate LSTM-GRU parallel architecture diagram for Part 3 paper.
Run: python generate_lstm_gru_architecture.py
Output: docs/figures/fig_lstm_gru_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
input_color = '#3498DB'
lstm_color = '#E74C3C'
gru_color = '#2ECC71'
merge_color = '#9B59B6'
dense_color = '#F39C12'
output_color = '#1ABC9C'
dropout_color = '#BDC3C7'
text_color = 'white'

def draw_box(ax, x, y, w, h, color, label, fontsize=9, alpha=0.9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='#2C3E50', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#2C3E50'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Title
ax.text(7, 7.5, 'Parallel LSTM-GRU Architecture', ha='center', va='center',
        fontsize=16, fontweight='bold', color='#2C3E50')

# Input Layer
draw_box(ax, 0.5, 3.0, 2.0, 1.8, input_color, 'Input\n14 Features\n(1 timestep)')

# LSTM Branch (top)
ax.text(5.5, 6.5, 'LSTM Branch', ha='center', fontsize=11, fontweight='bold', color=lstm_color)
draw_box(ax, 3.5, 5.5, 1.8, 1.0, lstm_color, 'LSTM\n64 units')
draw_box(ax, 5.8, 5.5, 1.5, 0.7, dropout_color, 'Drop\n0.2', fontsize=8)
draw_box(ax, 7.8, 5.5, 1.8, 1.0, lstm_color, 'LSTM\n32 units')
draw_box(ax, 10.1, 5.5, 1.5, 0.7, dropout_color, 'Drop\n0.2', fontsize=8)

# GRU Branch (bottom)
ax.text(5.5, 1.3, 'GRU Branch', ha='center', fontsize=11, fontweight='bold', color=gru_color)
draw_box(ax, 3.5, 1.5, 1.8, 1.0, gru_color, 'GRU\n64 units')
draw_box(ax, 5.8, 1.5, 1.5, 0.7, dropout_color, 'Drop\n0.2', fontsize=8)
draw_box(ax, 7.8, 1.5, 1.8, 1.0, gru_color, 'GRU\n32 units')
draw_box(ax, 10.1, 1.5, 1.5, 0.7, dropout_color, 'Drop\n0.2', fontsize=8)

# Arrows for LSTM branch
draw_arrow(ax, 2.5, 4.5, 3.5, 6.0, lstm_color)   # Input -> LSTM1
draw_arrow(ax, 5.3, 6.0, 5.8, 5.85)               # LSTM1 -> Drop
draw_arrow(ax, 7.3, 5.85, 7.8, 6.0)               # Drop -> LSTM2
draw_arrow(ax, 9.6, 6.0, 10.1, 5.85)              # LSTM2 -> Drop

# Arrows for GRU branch
draw_arrow(ax, 2.5, 3.3, 3.5, 2.0, gru_color)    # Input -> GRU1
draw_arrow(ax, 5.3, 2.0, 5.8, 1.85)               # GRU1 -> Drop
draw_arrow(ax, 7.3, 1.85, 7.8, 2.0)               # Drop -> GRU2
draw_arrow(ax, 9.6, 2.0, 10.1, 1.85)              # GRU2 -> Drop

# Concatenate
draw_box(ax, 11.2, 3.0, 1.0, 1.8, merge_color, 'Cat', fontsize=10)
draw_arrow(ax, 11.6, 5.5, 11.7, 4.8, merge_color)  # LSTM -> Cat
draw_arrow(ax, 11.6, 2.5, 11.7, 3.0, merge_color)  # GRU -> Cat

# Dense layers (right side, stacked vertically)
draw_box(ax, 12.5, 4.2, 1.2, 0.8, dense_color, 'Dense\n32', fontsize=9)
draw_box(ax, 12.5, 3.0, 1.2, 0.8, dense_color, 'Dense\n16', fontsize=9)
draw_box(ax, 12.5, 1.8, 1.2, 0.8, output_color, 'Dense\n1', fontsize=9)

# Arrows for dense
draw_arrow(ax, 12.2, 3.9, 12.5, 4.6)
draw_arrow(ax, 13.1, 4.2, 13.1, 3.8)
draw_arrow(ax, 13.1, 3.0, 13.1, 2.6)

# Output label
ax.text(13.1, 1.2, 'Predicted\nReturn', ha='center', fontsize=10, fontweight='bold', color='#2C3E50')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=lstm_color, label='LSTM Layers'),
    mpatches.Patch(facecolor=gru_color, label='GRU Layers'),
    mpatches.Patch(facecolor=merge_color, label='Concatenate'),
    mpatches.Patch(facecolor=dense_color, label='Dense Layers'),
    mpatches.Patch(facecolor=dropout_color, label='Dropout (0.2)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('docs/figures/fig_lstm_gru_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: docs/figures/fig_lstm_gru_architecture.png")
