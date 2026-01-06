# 👁️ AI Visual Analysis Module Guide

## Overview
We have integrated a State-of-the-Art **Computer Vision** system that analyzes stock charts exactly like a human professional trader would—by looking at the patterns, shapes, and geometry.

## Technology Stack
*   **Engine**: YOLOv8 (You Only Look Once) Neural Network by Ultralytics.
*   **Models**:
    1.  `stockmarket-pattern-detection-yolov8`: Detects classic chart patterns.
    2.  `stockmarket-future-prediction`: Classifies price action into "Bullish (Up)" or "Bearish (Down)" trends.

## How to Use
1.  Launch the app (`streamlit run rps3_f.py`).
2.  Select your stock and date range.
3.  Go to the new tab: **TAB 6: 👁️ Visual Analysis**.
4.  Click the **"📸 Run Visual Inference"** button.

## What Happens?
1.  **Image Generation**: The system dynamically generates a high-contrast, clean candlestick chart of the last 50 days (optimized for machine vision).
2.  **Neural Inference**: The image is passed through two deep learning models.
3.  **Visualization**: You will see your chart returned with **Bounding Boxes** showing exactly what the AI saw.
    *   *Example*: A box around a "W-Bottom" or a "Head and Shoulders".
4.  **Bias**: The system calculates a "Visual Bias" (Bullish/Bearish) based on the confidence sum of the detected trend signals.

## "Best in the World" Accuracy
This integration provides a **third layer of confirmation**:
1.  **Layer 1**: Numerical/Statistical (XGBoost/GRU) -> Finds math correlations.
2.  **Layer 2**: Fundamental -> Finds value.
3.  **Layer 3**: Visual (YOLO) -> Finds **structure**.

When all three layers align, the probability of a successful trade is maximized.
