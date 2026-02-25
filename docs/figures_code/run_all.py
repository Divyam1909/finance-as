"""
Master script to generate ALL figures for the research paper.
Run this single file to generate everything.
"""

import subprocess
import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("GENERATING ALL FIGURES FOR RESEARCH PAPER PART 3")
print("=" * 60)
print()

# List of scripts to run
scripts = [
    'generate_all_figures.py',
    'generate_candlestick_patterns.py',
    'generate_statistical_charts.py'
]

for script in scripts:
    print(f"\n{'='*40}")
    print(f"Running: {script}")
    print('='*40)
    
    try:
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: {script} exited with code {result.returncode}")
    except Exception as e:
        print(f"ERROR running {script}: {e}")

print("\n" + "=" * 60)
print("FIGURE GENERATION COMPLETE!")
print("=" * 60)
print(f"\nOutput folder: {os.path.abspath('../figures/')}")
print("\nNext steps:")
print("1. Render Mermaid diagrams (.mmd files) using https://mermaid.live/")
print("2. Copy all PNGs to the docs/figures/ folder")
print("3. Update LaTeX to include figures using \\includegraphics")
print("\nSee FIGURE_PLACEMENT_GUIDE.md for detailed instructions.")
