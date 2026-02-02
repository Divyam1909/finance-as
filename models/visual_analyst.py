"""
Professional-Grade Pattern Detection System.
Uses scipy peak detection with strict quality validation.
Only detects high-confidence, actionable patterns.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import linregress
from typing import List, Dict, Optional, Tuple
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config.settings import ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID
from utils.roboflow_client import RoboflowClient


class PatternAnalyst:
    """
    Professional pattern detection with strict quality criteria.
    
    Key Improvements:
    - Stricter tolerance (1% vs 2%)
    - Minimum pattern height requirements (5% minimum)
    - Volume confirmation
    - Time-based validation (patterns must form over reasonable timeframe)
    - Maximum 1-2 patterns per type to avoid noise
    """
    
    def __init__(self, order: int = 7):
        """
        Initialize the Pattern Analyst.
        
        Args:
            order: Number of points on each side for extrema detection (higher = fewer, cleaner peaks)
        """
        self.order = order
        self.min_pattern_height = 0.05  # 5% minimum pattern height
        self.max_patterns_per_type = 1  # Only show best pattern per type
        
        # Initialize Roboflow Client
        self.vision_client = None
        if ROBOFLOW_API_KEY:
            self.vision_client = RoboflowClient(api_key=ROBOFLOW_API_KEY)
            
    def _generate_chart_image(self, df: pd.DataFrame, window: int = 60) -> Optional[bytes]:
        """
        Generate a candlestick chart image for vision analysis.
        Returns bytes of the PNG image.
        """
        try:
            df_slice = df.tail(window).copy()
            if df_slice.empty:
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Simple line chart for patterns (vision models often trained on lines or candles)
            # Using line for cleaner pattern visibility for the model
            ax.plot(df_slice.index, df_slice['Close'], color='black', linewidth=2)
            
            # Remove axes for pure pattern detection (optional, depending on model training)
            # Keeping minimal axes for context
            ax.grid(True, alpha=0.3)
            plt.title(f"Price Action ({window}D)", fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            print(f"Error generating chart image: {e}")
            return None

    def analyze_patterns_with_vision(self, df: pd.DataFrame) -> List[Dict]:
        """
        Use Roboflow Vision API to detect patterns.
        """
        if not self.vision_client:
            return []
            
        try:
            # Generate image
            image_bytes = self._generate_chart_image(df)
            if not image_bytes:
                return []
                
            # Call API
            result = self.vision_client.run_workflow(
                workspace=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": image_bytes},
                use_cache=True
            )
            
            vision_patterns = []
            
            # Robust JSON Parsing
            predictions = []
            if isinstance(result, list) and len(result) > 0:
                # Format: [{'outputs': [{'predictions': {'predictions': [...]}}]}]
                if 'outputs' in result[0]:
                    outputs = result[0]['outputs']
                    if len(outputs) > 0 and 'predictions' in outputs[0]:
                        preds_node = outputs[0]['predictions']
                        # Check where the actual list is
                        if 'predictions' in preds_node:
                            predictions = preds_node['predictions']
                        elif isinstance(preds_node, list):
                            predictions = preds_node
            elif isinstance(result, dict) and 'predictions' in result:
                 predictions = result['predictions']
            
            # Mapping Logic
            class_map = {
                "W_Bottom": "Double Bottom",
                "M_Head": "Double Top", 
                "H_S": "Head & Shoulders",
                "Inv_H_S": "Inverse H&S"
            }
            
            for pred in predictions:
                label_raw = pred.get('class', 'Unknown')
                label_human = class_map.get(label_raw, label_raw.replace("_", " "))
                
                conf = pred.get('confidence', 0)
                if conf < 0.4: continue # Skip low confidence
                
                # Bounding Box
                x, y = pred.get('x', 0), pred.get('y', 0)
                w, h = pred.get('width', 0), pred.get('height', 0)
                
                # Determine type based on label
                p_type = 'Bullish' if 'Bottom' in label_human or 'Inv' in label_human or 'Bull' in label_human else 'Bearish'
                
                vision_patterns.append({
                    'Pattern': label_human,
                    'Type': f"{p_type} (AI Vision)",
                    'Confidence': round(conf * 100, 1),
                    'Target': 'N/A', # Vision doesn't calculate target easily
                    'Meta': {
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'raw_label': label_raw
                    },
                    'Status': 'Detected'
                })
                
            return vision_patterns
            
        except Exception as e:
            print(f"Vision analysis failed: {e}")
            return []
    
    def find_peaks_and_troughs(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find significant peaks and troughs using scipy with prominence filtering.
        """
        prices_arr = prices.values
        
        # Use prominence-based peak detection for cleaner results
        price_range = prices_arr.max() - prices_arr.min()
        min_prominence = price_range * 0.02  # 2% of price range minimum prominence
        
        # Find peaks
        peaks, peak_props = find_peaks(prices_arr, prominence=min_prominence, distance=self.order)
        
        # Find troughs (invert the data)
        troughs, trough_props = find_peaks(-prices_arr, prominence=min_prominence, distance=self.order)
        
        return peaks, troughs
    
    def _validate_pattern_quality(self, df: pd.DataFrame, start_idx: int, end_idx: int, 
                                   pattern_height: float, current_price: float) -> bool:
        """
        Validate pattern quality with multiple criteria.
        
        Args:
            df: Price dataframe
            start_idx: Pattern start index
            end_idx: Pattern end index
            pattern_height: Height of the pattern
            current_price: Current price
            
        Returns:
            True if pattern meets quality criteria
        """
        # 1. Minimum pattern height (5% of current price)
        if pattern_height < current_price * self.min_pattern_height:
            return False
        
        # 2. Pattern should form over reasonable timeframe (5-40 trading days)
        pattern_duration = end_idx - start_idx
        if pattern_duration < 5 or pattern_duration > 40:
            return False
        
        # 3. Pattern should be recent (within last 45 days)
        if len(df) - end_idx > 45:
            return False
        
        return True
    
    def _check_volume_confirmation(self, df: pd.DataFrame, breakout_idx: int, pattern_type: str) -> bool:
        """
        Check if there's volume confirmation at potential breakout point.
        Higher volume at neckline suggests stronger pattern.
        """
        if 'Volume' not in df.columns:
            return True  # Skip if no volume data
        
        try:
            avg_volume = df['Volume'].iloc[max(0, breakout_idx-10):breakout_idx].mean()
            recent_volume = df['Volume'].iloc[breakout_idx:min(len(df), breakout_idx+3)].mean()
            
            # Volume should be at least 80% of average
            return recent_volume >= avg_volume * 0.8
        except Exception:
            return True
    
    def detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Double Top pattern with strict validation.
        Only returns highest quality pattern.
        """
        patterns = []
        df_analysis = df.tail(50)  # Only analyze last 50 days
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(peak_idx) < 2 or len(trough_idx) < 1:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        # Look at recent peaks only
        for i in range(len(peak_idx) - 1):
            p1_idx, p2_idx = peak_idx[i], peak_idx[i + 1]
            p1_price, p2_price = prices.iloc[p1_idx], prices.iloc[p2_idx]
            
            # Strict: peaks must be within 1% of each other
            price_diff_pct = abs(p1_price - p2_price) / p1_price
            if price_diff_pct > 0.01:
                continue
            
            # Find trough between peaks
            troughs_between = trough_idx[(trough_idx > p1_idx) & (trough_idx < p2_idx)]
            if len(troughs_between) == 0:
                continue
            
            trough_price = prices.iloc[troughs_between[0]]
            avg_peak = (p1_price + p2_price) / 2
            pattern_height = avg_peak - trough_price
            
            # Validate quality
            if not self._validate_pattern_quality(df_analysis, p1_idx, p2_idx, pattern_height, current_price):
                continue
            
            # Calculate confidence
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.1), 1) * 10  # Bonus for larger patterns
            confidence = min(confidence + height_score, 99)
            
            # Check if price has broken below neckline (confirmation)
            confirmed = current_price < trough_price
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = trough_price - pattern_height
                
                best_pattern = {
                    'Pattern': 'Double Top',
                    'Type': 'Bearish Reversal',
                    'Neckline': round(trough_price, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Peak_Price': round(avg_peak, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Double Bottom pattern with strict validation.
        """
        patterns = []
        df_analysis = df.tail(50)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(trough_idx) < 2 or len(peak_idx) < 1:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        for i in range(len(trough_idx) - 1):
            t1_idx, t2_idx = trough_idx[i], trough_idx[i + 1]
            t1_price, t2_price = prices.iloc[t1_idx], prices.iloc[t2_idx]
            
            # Strict: troughs must be within 1% of each other
            price_diff_pct = abs(t1_price - t2_price) / t1_price
            if price_diff_pct > 0.01:
                continue
            
            peaks_between = peak_idx[(peak_idx > t1_idx) & (peak_idx < t2_idx)]
            if len(peaks_between) == 0:
                continue
            
            peak_price = prices.iloc[peaks_between[0]]
            avg_trough = (t1_price + t2_price) / 2
            pattern_height = peak_price - avg_trough
            
            if not self._validate_pattern_quality(df_analysis, t1_idx, t2_idx, pattern_height, current_price):
                continue
            
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.1), 1) * 10
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > peak_price
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = peak_price + pattern_height
                
                best_pattern = {
                    'Pattern': 'Double Bottom',
                    'Type': 'Bullish Reversal',
                    'Neckline': round(peak_price, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Trough_Price': round(avg_trough, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Head & Shoulders with strict validation.
        Requires head to be at least 3% higher than shoulders.
        """
        patterns = []
        df_analysis = df.tail(60)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(peak_idx) < 3 or len(trough_idx) < 2:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        for i in range(len(peak_idx) - 2):
            ls_idx, h_idx, rs_idx = peak_idx[i], peak_idx[i+1], peak_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            # Head must be at least 3% higher than both shoulders
            avg_shoulder = (ls_price + rs_price) / 2
            head_height_pct = (h_price - avg_shoulder) / avg_shoulder
            
            if head_height_pct < 0.03:
                continue
            
            # Shoulders must be within 2% of each other
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > 0.02:
                continue
            
            # Find neckline troughs
            troughs_1 = trough_idx[(trough_idx > ls_idx) & (trough_idx < h_idx)]
            troughs_2 = trough_idx[(trough_idx > h_idx) & (trough_idx < rs_idx)]
            
            if len(troughs_1) == 0 or len(troughs_2) == 0:
                continue
            
            neckline = (prices.iloc[troughs_1[0]] + prices.iloc[troughs_2[0]]) / 2
            pattern_height = h_price - neckline
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_height_pct * 100, 10)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price < neckline
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = neckline - pattern_height
                
                best_pattern = {
                    'Pattern': 'Head & Shoulders',
                    'Type': 'Bearish Reversal',
                    'Neckline': round(neckline, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Head_Price': round(h_price, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Inverse H&S with strict validation.
        """
        patterns = []
        df_analysis = df.tail(60)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(trough_idx) < 3 or len(peak_idx) < 2:
            return patterns
        
        best_pattern = None
        best_confidence = 0
        
        for i in range(len(trough_idx) - 2):
            ls_idx, h_idx, rs_idx = trough_idx[i], trough_idx[i+1], trough_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            # Head must be at least 3% lower than shoulders
            avg_shoulder = (ls_price + rs_price) / 2
            head_depth_pct = (avg_shoulder - h_price) / avg_shoulder
            
            if head_depth_pct < 0.03:
                continue
            
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > 0.02:
                continue
            
            peaks_1 = peak_idx[(peak_idx > ls_idx) & (peak_idx < h_idx)]
            peaks_2 = peak_idx[(peak_idx > h_idx) & (peak_idx < rs_idx)]
            
            if len(peaks_1) == 0 or len(peaks_2) == 0:
                continue
            
            neckline = (prices.iloc[peaks_1[0]] + prices.iloc[peaks_2[0]]) / 2
            pattern_height = neckline - h_price
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_depth_pct * 100, 10)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > neckline
            
            if confidence > best_confidence:
                best_confidence = confidence
                target = neckline + pattern_height
                
                best_pattern = {
                    'Pattern': 'Inverse H&S',
                    'Type': 'Bullish Reversal',
                    'Neckline': round(neckline, 2),
                    'Target': round(target, 2),
                    'Confidence': round(confidence, 1),
                    'Status': 'CONFIRMED' if confirmed else 'Forming',
                    'Head_Price': round(h_price, 2)
                }
        
        return [best_pattern] if best_pattern else []
    
    def detect_trend(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Detect current trend using multiple confirmation methods.
        """
        prices = df['Close'].tail(window)
        
        # Linear regression
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = linregress(x, prices.values)
        r_squared = r_value ** 2  # How well the trend fits
        
        # MA crossover
        ma_short = df['Close'].rolling(5).mean().iloc[-1]
        ma_long = df['Close'].rolling(20).mean().iloc[-1]
        
        # Price structure
        recent_highs = df['High'].tail(10)
        recent_lows = df['Low'].tail(10)
        
        higher_highs = (recent_highs.diff().dropna() > 0).sum() / len(recent_highs.diff().dropna())
        higher_lows = (recent_lows.diff().dropna() > 0).sum() / len(recent_lows.diff().dropna())
        
        # Score trend
        bullish_score = 0
        bearish_score = 0
        
        if slope > 0:
            bullish_score += 1 + (r_squared * 0.5)  # Extra credit for strong fit
        else:
            bearish_score += 1 + (r_squared * 0.5)
            
        if ma_short > ma_long:
            bullish_score += 1
        else:
            bearish_score += 1
            
        if higher_highs > 0.6 and higher_lows > 0.6:
            bullish_score += 1
        elif higher_highs < 0.4 and higher_lows < 0.4:
            bearish_score += 1
        
        total_score = bullish_score + bearish_score
        if bullish_score > bearish_score:
            trend = "Bullish"
            strength = (bullish_score / total_score) * 100 if total_score > 0 else 50
        elif bearish_score > bullish_score:
            trend = "Bearish"
            strength = (bearish_score / total_score) * 100 if total_score > 0 else 50
        else:
            trend = "Neutral"
            strength = 50
        
        return {
            'Trend': trend,
            'Strength': round(strength, 1),
            'Slope': round(slope, 4),
            'R_Squared': round(r_squared, 3),
            'MA_Signal': 'Bullish' if ma_short > ma_long else 'Bearish',
            'Structure': 'Higher Highs/Lows' if higher_highs > 0.6 else 'Lower Highs/Lows' if higher_highs < 0.4 else 'Mixed'
        }
    
    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        Detect significant support and resistance levels.
        Uses clustering to find key price levels.
        """
        df_slice = df.tail(lookback)
        prices = df_slice['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        # Get price levels
        resistance_levels = [prices.iloc[idx] for idx in peak_idx[-5:]] if len(peak_idx) > 0 else []
        support_levels = [prices.iloc[idx] for idx in trough_idx[-5:]] if len(trough_idx) > 0 else []
        
        # Find nearest levels above and below current price
        resistance_above = [r for r in resistance_levels if r > current_price * 1.005]
        support_below = [s for s in support_levels if s < current_price * 0.995]
        
        nearest_resistance = min(resistance_above) if resistance_above else None
        nearest_support = max(support_below) if support_below else None
        
        return {
            'Current_Price': round(current_price, 2),
            'Nearest_Resistance': round(nearest_resistance, 2) if nearest_resistance else 'N/A',
            'Nearest_Support': round(nearest_support, 2) if nearest_support else 'N/A',
            'All_Resistance': [round(r, 2) for r in sorted(set(resistance_levels), reverse=True)[:3]],
            'All_Support': [round(s, 2) for s in sorted(set(support_levels), reverse=True)[:3]]
        }
    
    def detect_triangle_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Triangle Patterns (Ascending, Descending, Symmetrical).
        Uses slope convergence checking.
        """
        patterns = []
        df_analysis = df.tail(40)
        prices = df_analysis['Close']
        highs = df_analysis['High']
        lows = df_analysis['Low']
        current_price = prices.iloc[-1]
        
        peak_idx, _ = self.find_peaks_and_troughs(highs)
        _, trough_idx = self.find_peaks_and_troughs(lows)
        
        if len(peak_idx) < 3 or len(trough_idx) < 3:
            return patterns
            
        # Get recent peaks/troughs for trendlines
        recent_peaks = highs.iloc[peak_idx[-3:]]
        recent_troughs = lows.iloc[trough_idx[-3:]]
        
        # Calculate slopes
        x_peaks = np.arange(len(recent_peaks))
        slope_res, _, r_res, _, _ = linregress(x_peaks, recent_peaks.values)
        
        x_troughs = np.arange(len(recent_troughs))
        slope_sup, _, r_sup, _, _ = linregress(x_troughs, recent_troughs.values)
        
        # Check linearity (must be decent straight lines)
        if r_res**2 < 0.6 or r_sup**2 < 0.6:
            return patterns
            
        # 1. Ascending Triangle: Flat resistance, rising support
        if abs(slope_res) < 0.002 and slope_sup > 0.002:
            confidence = (r_res**2 + r_sup**2) / 2 * 100
            patterns.append({
                'Pattern': 'Ascending Triangle',
                'Type': 'Bullish Continuation',
                'Confidence': round(confidence, 1),
                'Target': round(current_price * 1.05, 2), # Approx target
                'Status': 'Forming'
            })
            
        # 2. Descending Triangle: Falling resistance, flat support
        elif slope_res < -0.002 and abs(slope_sup) < 0.002:
             confidence = (r_res**2 + r_sup**2) / 2 * 100
             patterns.append({
                'Pattern': 'Descending Triangle',
                'Type': 'Bearish Continuation',
                'Confidence': round(confidence, 1),
                'Target': round(current_price * 0.95, 2), # Approx target
                'Status': 'Forming'
            })
            
        # 3. Symmetrical Triangle: Converging slopes
        elif slope_res < -0.001 and slope_sup > 0.001:
             confidence = (r_res**2 + r_sup**2) / 2 * 100
             patterns.append({
                'Pattern': 'Symmetrical Triangle',
                'Type': 'Neutral/Breakout',
                'Confidence': round(confidence, 1),
                'Target': round(current_price * (1.05 if slope_sup > abs(slope_res) else 0.95), 2),
                'Status': 'Forming'
            })
            
        return patterns

    def detect_wedge_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Rising/Falling Wedges.
        Both slopes point in same direction but converge.
        """
        patterns = []
        df_analysis = df.tail(40)
        highs = df_analysis['High']
        lows = df_analysis['Low']
        current_price = df_analysis['Close'].iloc[-1]
        
        peak_idx, _ = self.find_peaks_and_troughs(highs)
        _, trough_idx = self.find_peaks_and_troughs(lows)
        
        if len(peak_idx) < 3 or len(trough_idx) < 3:
            return patterns
            
        # Slopes
        x_peaks = np.arange(3)
        slope_res, _, r_res, _, _ = linregress(x_peaks, highs.iloc[peak_idx[-3:]].values)
        slope_sup, _, r_sup, _, _ = linregress(x_peaks, lows.iloc[trough_idx[-3:]].values)
        
        # 1. Rising Wedge: Both slopes positive, support steeper than resistance (converging)
        # Actually resistance usually flatter in rising wedge, or both rising with convergence
        if slope_res > 0 and slope_sup > 0:
            if slope_sup > slope_res: # Converging up
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                patterns.append({
                    'Pattern': 'Rising Wedge',
                    'Type': 'Bearish Reversal',
                    'Confidence': round(confidence, 1),
                    'Target': round(lows.iloc[trough_idx[-3]], 2), # Base of wedge
                    'Status': 'Forming'
                })
                
        # 2. Falling Wedge: Both slopes negative, resistance steeper than support (converging)
        elif slope_res < 0 and slope_sup < 0:
            if slope_res < slope_sup: # Converging down (slope_res is more negative)
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                patterns.append({
                    'Pattern': 'Falling Wedge',
                    'Type': 'Bullish Reversal',
                    'Confidence': round(confidence, 1),
                    'Target': round(highs.iloc[peak_idx[-3]], 2), # Top of wedge
                    'Status': 'Forming'
                })
        
        return patterns

    def analyze_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive pattern analysis.
        Returns only high-quality, actionable patterns.
        """
        all_patterns = []
        
        # Detect patterns (each returns at most 1 best pattern)
        all_patterns.extend(self.detect_double_top(df))
        all_patterns.extend(self.detect_double_bottom(df))
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(df))
        
        # New Geometric Patterns
        all_patterns.extend(self.detect_triangle_pattern(df))
        all_patterns.extend(self.detect_wedge_pattern(df))
        
        # Add Vision Patterns
        vision_patterns = self.analyze_patterns_with_vision(df)
        all_patterns.extend(vision_patterns)
        
        # Filter to only show patterns with confidence >= 80% (lowered slightly for triangles)
        high_quality_patterns = [p for p in all_patterns if p and p.get('Confidence', 0) >= 80]
        
        # Sort by confidence
        high_quality_patterns.sort(key=lambda x: x.get('Confidence', 0), reverse=True)
        
        # Get trend and S/R
        trend = self.detect_trend(df)
        sr_levels = self.detect_support_resistance(df)
        
        # Calculate bias
        bullish_patterns = sum(1 for p in high_quality_patterns if 'Bullish' in p.get('Type', ''))
        bearish_patterns = sum(1 for p in high_quality_patterns if 'Bearish' in p.get('Type', ''))
        
        if bullish_patterns > bearish_patterns:
            pattern_bias = "Bullish"
        elif bearish_patterns > bullish_patterns:
            pattern_bias = "Bearish"
        else:
            pattern_bias = trend['Trend']
        
        return {
            'patterns': high_quality_patterns[:3],  # Max 3 patterns
            'trend': trend,
            'support_resistance': sr_levels,
            'overall_bias': pattern_bias,
            'pattern_count': len(high_quality_patterns)
        }


# Backward compatibility
VISUAL_AI_AVAILABLE = True
