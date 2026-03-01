"""
Advanced Pattern Detection System with Multi-Timeframe Scanning.
Uses scipy peak detection with adaptive thresholds and fallback mechanisms.
Detects even small patterns and provides actionable signals.
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
    Advanced pattern detection with multi-timeframe scanning and adaptive thresholds.
    
    Key Features:
    - Multi-timeframe scanning (order 3, 5, 7) to catch patterns at all scales
    - Adaptive thresholds that relax if no patterns found
    - Channel, flag, and rounding bottom detection
    - Micro-pattern detection for consolidating markets
    - Fallback mechanism: always tries to find something useful
    """
    
    def __init__(self, order: int = 3):
        self.order = order
        self.min_pattern_height = 0.01  # 1% minimum pattern height
        self.max_patterns_per_type = 3
        
        # Multi-timeframe orders: small catches minor swings, large catches major ones
        self.scan_orders = [3, 5, 7]
        
        # Initialize Roboflow Client
        self.vision_client = None
        if ROBOFLOW_API_KEY:
            self.vision_client = RoboflowClient(api_key=ROBOFLOW_API_KEY)
            
    def _generate_chart_image(self, df: pd.DataFrame, window: int = 60) -> Optional[bytes]:
        """Generate a chart image for vision analysis."""
        try:
            df_slice = df.tail(window).copy()
            if df_slice.empty:
                return None
                
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.plot(df_slice.index, df_slice['Close'], color='black', linewidth=2)
            ax.grid(True, alpha=0.3)
            plt.title(f"Price Action ({window}D)", fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            print(f"Error generating chart image: {e}")
            return None

    def analyze_patterns_with_vision(self, df: pd.DataFrame) -> List[Dict]:
        """Use Roboflow Vision API to detect patterns."""
        if not self.vision_client:
            return []
            
        try:
            image_bytes = self._generate_chart_image(df)
            if not image_bytes:
                return []
                
            result = self.vision_client.run_workflow(
                workspace=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": image_bytes},
                use_cache=True
            )
            
            vision_patterns = []
            
            predictions = []
            if isinstance(result, list) and len(result) > 0:
                if 'outputs' in result[0]:
                    outputs = result[0]['outputs']
                    if len(outputs) > 0 and 'predictions' in outputs[0]:
                        preds_node = outputs[0]['predictions']
                        if 'predictions' in preds_node:
                            predictions = preds_node['predictions']
                        elif isinstance(preds_node, list):
                            predictions = preds_node
            elif isinstance(result, dict) and 'predictions' in result:
                 predictions = result['predictions']
            
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
                if conf < 0.3: continue  # Lower threshold for vision
                
                x, y = pred.get('x', 0), pred.get('y', 0)
                w, h = pred.get('width', 0), pred.get('height', 0)
                
                p_type = 'Bullish' if 'Bottom' in label_human or 'Inv' in label_human or 'Bull' in label_human else 'Bearish'
                
                vision_patterns.append({
                    'Pattern': label_human,
                    'Type': f"{p_type} (AI Vision)",
                    'Confidence': round(conf * 100, 1),
                    'Target': 'N/A',
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
    
    def find_peaks_and_troughs(self, prices: pd.Series, order: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks and troughs with configurable sensitivity.
        Lower order = more peaks detected (more sensitive).
        """
        prices_arr = prices.values
        use_order = order if order is not None else self.order
        
        price_range = prices_arr.max() - prices_arr.min()
        if price_range == 0:
            return np.array([]), np.array([])
        
        # Very low prominence to catch small swings
        min_prominence = price_range * 0.005  # 0.5% of range
        
        peaks, _ = find_peaks(prices_arr, prominence=min_prominence, distance=use_order)
        troughs, _ = find_peaks(-prices_arr, prominence=min_prominence, distance=use_order)
        
        return peaks, troughs
    
    def _validate_pattern_quality(self, df: pd.DataFrame, start_idx: int, end_idx: int, 
                                   pattern_height: float, current_price: float,
                                   relaxed: bool = False) -> bool:
        """
        Validate pattern quality. If relaxed=True, uses much looser criteria.
        """
        min_height = self.min_pattern_height if not relaxed else 0.005  # 0.5% when relaxed
        
        if pattern_height < current_price * min_height:
            return False
        
        pattern_duration = end_idx - start_idx
        max_dur = 80 if relaxed else 60
        min_dur = 2 if relaxed else 3
        if pattern_duration < min_dur or pattern_duration > max_dur:
            return False
        
        recency = 120 if relaxed else 90
        if len(df) - end_idx > recency:
            return False
        
        return True
    
    def _check_volume_confirmation(self, df: pd.DataFrame, breakout_idx: int, pattern_type: str) -> bool:
        """Check if there's volume confirmation at breakout point."""
        if 'Volume' not in df.columns:
            return True
        try:
            avg_volume = df['Volume'].iloc[max(0, breakout_idx-10):breakout_idx].mean()
            recent_volume = df['Volume'].iloc[breakout_idx:min(len(df), breakout_idx+3)].mean()
            return recent_volume >= avg_volume * 0.6  # Relaxed from 0.8
        except Exception:
            return True
    
    def detect_double_top(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """
        Detect Double Top pattern with adaptive tolerance.
        Scans with given order for multi-timeframe capability.
        """
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(peak_idx) < 2 or len(trough_idx) < 1:
            return patterns
        
        tolerance = 0.05 if relaxed else 0.03  # 5% when relaxed, 3% normally
        
        for i in range(len(peak_idx) - 1):
            p1_idx, p2_idx = peak_idx[i], peak_idx[i + 1]
            p1_price, p2_price = prices.iloc[p1_idx], prices.iloc[p2_idx]
            
            price_diff_pct = abs(p1_price - p2_price) / p1_price
            if price_diff_pct > tolerance:
                continue
            
            troughs_between = trough_idx[(trough_idx > p1_idx) & (trough_idx < p2_idx)]
            if len(troughs_between) == 0:
                continue
            
            trough_price = prices.iloc[troughs_between[0]]
            avg_peak = (p1_price + p2_price) / 2
            pattern_height = avg_peak - trough_price
            
            if not self._validate_pattern_quality(df_analysis, p1_idx, p2_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.05), 1) * 15
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price < trough_price
            target = trough_price - pattern_height
            
            patterns.append({
                'Pattern': 'Double Top',
                'Type': 'Bearish Reversal',
                'Neckline': round(trough_price, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Peak_Price': round(avg_peak, 2)
            })
        
        # Return best patterns
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_double_bottom(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """Detect Double Bottom pattern with adaptive tolerance."""
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(trough_idx) < 2 or len(peak_idx) < 1:
            return patterns
        
        tolerance = 0.05 if relaxed else 0.03
        
        for i in range(len(trough_idx) - 1):
            t1_idx, t2_idx = trough_idx[i], trough_idx[i + 1]
            t1_price, t2_price = prices.iloc[t1_idx], prices.iloc[t2_idx]
            
            price_diff_pct = abs(t1_price - t2_price) / t1_price
            if price_diff_pct > tolerance:
                continue
            
            peaks_between = peak_idx[(peak_idx > t1_idx) & (peak_idx < t2_idx)]
            if len(peaks_between) == 0:
                continue
            
            peak_price = prices.iloc[peaks_between[0]]
            avg_trough = (t1_price + t2_price) / 2
            pattern_height = peak_price - avg_trough
            
            if not self._validate_pattern_quality(df_analysis, t1_idx, t2_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.05), 1) * 15
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > peak_price
            target = peak_price + pattern_height
            
            patterns.append({
                'Pattern': 'Double Bottom',
                'Type': 'Bullish Reversal',
                'Neckline': round(peak_price, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Trough_Price': round(avg_trough, 2)
            })
        
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_head_and_shoulders(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """Detect Head & Shoulders with adaptive validation."""
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(peak_idx) < 3 or len(trough_idx) < 2:
            return patterns
        
        min_head_height = 0.01 if relaxed else 0.015
        max_shoulder_diff = 0.06 if relaxed else 0.04
        
        for i in range(len(peak_idx) - 2):
            ls_idx, h_idx, rs_idx = peak_idx[i], peak_idx[i+1], peak_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            avg_shoulder = (ls_price + rs_price) / 2
            head_height_pct = (h_price - avg_shoulder) / avg_shoulder
            
            if head_height_pct < min_head_height:
                continue
            
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > max_shoulder_diff:
                continue
            
            troughs_1 = trough_idx[(trough_idx > ls_idx) & (trough_idx < h_idx)]
            troughs_2 = trough_idx[(trough_idx > h_idx) & (trough_idx < rs_idx)]
            
            if len(troughs_1) == 0 or len(troughs_2) == 0:
                continue
            
            neckline = (prices.iloc[troughs_1[0]] + prices.iloc[troughs_2[0]]) / 2
            pattern_height = h_price - neckline
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_height_pct * 100, 15)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price < neckline
            target = neckline - pattern_height
            
            patterns.append({
                'Pattern': 'Head & Shoulders',
                'Type': 'Bearish Reversal',
                'Neckline': round(neckline, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Head_Price': round(h_price, 2)
            })
        
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """Detect Inverse H&S with adaptive validation."""
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(trough_idx) < 3 or len(peak_idx) < 2:
            return patterns
        
        min_head_depth = 0.01 if relaxed else 0.015
        max_shoulder_diff = 0.06 if relaxed else 0.04
        
        for i in range(len(trough_idx) - 2):
            ls_idx, h_idx, rs_idx = trough_idx[i], trough_idx[i+1], trough_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            avg_shoulder = (ls_price + rs_price) / 2
            head_depth_pct = (avg_shoulder - h_price) / avg_shoulder
            
            if head_depth_pct < min_head_depth:
                continue
            
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > max_shoulder_diff:
                continue
            
            peaks_1 = peak_idx[(peak_idx > ls_idx) & (peak_idx < h_idx)]
            peaks_2 = peak_idx[(peak_idx > h_idx) & (peak_idx < rs_idx)]
            
            if len(peaks_1) == 0 or len(peaks_2) == 0:
                continue
            
            neckline = (prices.iloc[peaks_1[0]] + prices.iloc[peaks_2[0]]) / 2
            pattern_height = neckline - h_price
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_depth_pct * 100, 15)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > neckline
            target = neckline + pattern_height
            
            patterns.append({
                'Pattern': 'Inverse H&S',
                'Type': 'Bullish Reversal',
                'Neckline': round(neckline, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Head_Price': round(h_price, 2)
            })
        
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_trend(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Detect current trend using multiple confirmation methods."""
        prices = df['Close'].tail(window)
        
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = linregress(x, prices.values)
        r_squared = r_value ** 2
        
        ma_short = df['Close'].rolling(5).mean().iloc[-1]
        ma_long = df['Close'].rolling(20).mean().iloc[-1]
        
        recent_highs = df['High'].tail(10)
        recent_lows = df['Low'].tail(10)
        
        higher_highs = (recent_highs.diff().dropna() > 0).sum() / len(recent_highs.diff().dropna())
        higher_lows = (recent_lows.diff().dropna() > 0).sum() / len(recent_lows.diff().dropna())
        
        bullish_score = 0
        bearish_score = 0
        
        if slope > 0:
            bullish_score += 1 + (r_squared * 0.5)
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
    
    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 90) -> Dict:
        """
        Detect significant support and resistance levels.
        Uses clustering with wider lookback for better levels.
        """
        df_slice = df.tail(lookback)
        prices = df_slice['Close']
        current_price = prices.iloc[-1]
        
        # Multi-scale peak detection for robust S/R
        all_peak_prices = []
        all_trough_prices = []
        
        for ord_val in [3, 5, 7]:
            peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=ord_val)
            all_peak_prices.extend([prices.iloc[idx] for idx in peak_idx])
            all_trough_prices.extend([prices.iloc[idx] for idx in trough_idx])
        
        # Cluster nearby levels (within 0.5% of each other)
        def cluster_levels(levels, threshold_pct=0.005):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = [[levels[0]]]
            for level in levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < threshold_pct:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            # Return average of each cluster, weighted by cluster size
            return [(np.mean(c), len(c)) for c in clusters]
        
        resistance_clusters = cluster_levels(all_peak_prices)
        support_clusters = cluster_levels(all_trough_prices)
        
        # Sort by number of touches (cluster size) for strength
        resistance_clusters.sort(key=lambda x: x[1], reverse=True)
        support_clusters.sort(key=lambda x: x[1], reverse=True)
        
        resistance_levels = [c[0] for c in resistance_clusters]
        support_levels = [c[0] for c in support_clusters]
        
        resistance_above = [r for r in resistance_levels if r > current_price * 1.002]
        support_below = [s for s in support_levels if s < current_price * 0.998]
        
        nearest_resistance = min(resistance_above) if resistance_above else None
        nearest_support = max(support_below) if support_below else None
        
        return {
            'Current_Price': round(current_price, 2),
            'Nearest_Resistance': round(nearest_resistance, 2) if nearest_resistance else 'N/A',
            'Nearest_Support': round(nearest_support, 2) if nearest_support else 'N/A',
            'All_Resistance': [round(r, 2) for r in sorted(set(resistance_levels), reverse=True)[:5]],
            'All_Support': [round(s, 2) for s in sorted(set(support_levels), reverse=True)[:5]]
        }
    
    def detect_triangle_pattern(self, df: pd.DataFrame, order: int = None) -> List[Dict]:
        """
        Detect Triangle Patterns with relaxed R² requirements.
        Tries multiple window sizes for better detection.
        """
        patterns = []
        
        for window in [30, 40, 50]:
            df_analysis = df.tail(window)
            prices = df_analysis['Close']
            highs = df_analysis['High']
            lows = df_analysis['Low']
            current_price = prices.iloc[-1]
            
            peak_idx, _ = self.find_peaks_and_troughs(highs, order=order or 3)
            _, trough_idx = self.find_peaks_and_troughs(lows, order=order or 3)
            
            if len(peak_idx) < 2 or len(trough_idx) < 2:
                continue
            
            # Use last 2-4 peaks/troughs
            n_points = min(4, len(peak_idx), len(trough_idx))
            recent_peaks = highs.iloc[peak_idx[-n_points:]]
            recent_troughs = lows.iloc[trough_idx[-n_points:]]
            
            x_peaks = np.arange(len(recent_peaks))
            slope_res, _, r_res, _, _ = linregress(x_peaks, recent_peaks.values)
            
            x_troughs = np.arange(len(recent_troughs))
            slope_sup, _, r_sup, _, _ = linregress(x_troughs, recent_troughs.values)
            
            # Relaxed R² requirement (0.3 instead of 0.6)
            min_r2 = 0.3
            if r_res**2 < min_r2 or r_sup**2 < min_r2:
                continue
            
            # Normalize slopes by price for comparability
            price_scale = current_price / 100
            norm_slope_res = slope_res / price_scale
            norm_slope_sup = slope_sup / price_scale
            
            # Ascending Triangle: Flat resistance, rising support
            if abs(norm_slope_res) < 0.3 and norm_slope_sup > 0.1:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                # Bonus for convergence tightness
                range_pct = (recent_peaks.max() - recent_troughs.min()) / current_price * 100
                confidence = min(confidence + range_pct, 95)
                patterns.append({
                    'Pattern': f'Ascending Triangle ({window}D)',
                    'Type': 'Bullish Continuation',
                    'Confidence': round(confidence, 1),
                    'Target': round(current_price * 1.05, 2),
                    'Status': 'Forming'
                })
                
            # Descending Triangle: Falling resistance, flat support
            elif norm_slope_res < -0.1 and abs(norm_slope_sup) < 0.3:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                range_pct = (recent_peaks.max() - recent_troughs.min()) / current_price * 100
                confidence = min(confidence + range_pct, 95)
                patterns.append({
                    'Pattern': f'Descending Triangle ({window}D)',
                    'Type': 'Bearish Continuation',
                    'Confidence': round(confidence, 1),
                    'Target': round(current_price * 0.95, 2),
                    'Status': 'Forming'
                })
                
            # Symmetrical Triangle: Converging slopes
            elif norm_slope_res < -0.05 and norm_slope_sup > 0.05:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                range_pct = (recent_peaks.max() - recent_troughs.min()) / current_price * 100
                confidence = min(confidence + range_pct, 95)
                bias = 'Bullish' if norm_slope_sup > abs(norm_slope_res) else 'Bearish'
                patterns.append({
                    'Pattern': f'Symmetrical Triangle ({window}D)',
                    'Type': f'{bias} Breakout Pending',
                    'Confidence': round(confidence, 1),
                    'Target': round(current_price * (1.05 if bias == 'Bullish' else 0.95), 2),
                    'Status': 'Forming'
                })
        
        # Deduplicate (keep best per type)
        seen_types = {}
        for p in sorted(patterns, key=lambda x: x['Confidence'], reverse=True):
            base_type = p['Pattern'].split(' (')[0]
            if base_type not in seen_types:
                seen_types[base_type] = p
        
        return list(seen_types.values())

    def detect_wedge_pattern(self, df: pd.DataFrame, order: int = None) -> List[Dict]:
        """Detect Rising/Falling Wedges with multiple windows."""
        patterns = []
        
        for window in [30, 40, 50]:
            df_analysis = df.tail(window)
            highs = df_analysis['High']
            lows = df_analysis['Low']
            current_price = df_analysis['Close'].iloc[-1]
            
            peak_idx, _ = self.find_peaks_and_troughs(highs, order=order or 3)
            _, trough_idx = self.find_peaks_and_troughs(lows, order=order or 3)
            
            if len(peak_idx) < 2 or len(trough_idx) < 2:
                continue
            
            n_points = min(4, len(peak_idx), len(trough_idx))
            x_peaks = np.arange(n_points)
            slope_res, _, r_res, _, _ = linregress(x_peaks, highs.iloc[peak_idx[-n_points:]].values)
            slope_sup, _, r_sup, _, _ = linregress(x_peaks, lows.iloc[trough_idx[-n_points:]].values)
            
            if r_res**2 < 0.3 or r_sup**2 < 0.3:
                continue
            
            # Rising Wedge: Both slopes positive, support steeper (converging)
            if slope_res > 0 and slope_sup > 0 and slope_sup > slope_res * 0.5:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                patterns.append({
                    'Pattern': f'Rising Wedge ({window}D)',
                    'Type': 'Bearish Reversal',
                    'Confidence': round(confidence, 1),
                    'Target': round(lows.iloc[trough_idx[-n_points]], 2),
                    'Status': 'Forming'
                })
                    
            # Falling Wedge: Both slopes negative, resistance steeper (converging)
            elif slope_res < 0 and slope_sup < 0 and slope_res < slope_sup * 0.5:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                patterns.append({
                    'Pattern': f'Falling Wedge ({window}D)',
                    'Type': 'Bullish Reversal',
                    'Confidence': round(confidence, 1),
                    'Target': round(highs.iloc[peak_idx[-n_points]], 2),
                    'Status': 'Forming'
                })
        
        # Deduplicate
        seen_types = {}
        for p in sorted(patterns, key=lambda x: x['Confidence'], reverse=True):
            base_type = p['Pattern'].split(' (')[0]
            if base_type not in seen_types:
                seen_types[base_type] = p
        return list(seen_types.values())
    
    def detect_channel_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Price Channels (Ascending, Descending, Horizontal).
        NEW pattern type for consolidating/trending markets.
        """
        patterns = []
        
        for window in [30, 50]:
            df_analysis = df.tail(window)
            prices = df_analysis['Close']
            highs = df_analysis['High']
            lows = df_analysis['Low']
            current_price = prices.iloc[-1]
            
            # Fit regression lines to highs and lows
            x = np.arange(len(df_analysis))
            
            slope_h, intercept_h, r_h, _, _ = linregress(x, highs.values)
            slope_l, intercept_l, r_l, _, _ = linregress(x, lows.values)
            
            # Both lines must be reasonably parallel (slopes within 50% of each other)
            if abs(slope_h) > 0 and abs(slope_l) > 0:
                slope_ratio = min(abs(slope_h), abs(slope_l)) / max(abs(slope_h), abs(slope_l))
            else:
                slope_ratio = 1.0 if abs(slope_h - slope_l) < 0.01 else 0.0
            
            if slope_ratio < 0.3:
                continue
            
            # Both lines need decent fit
            if r_h**2 < 0.3 or r_l**2 < 0.3:
                continue
            
            # Calculate channel width
            channel_width = (highs.mean() - lows.mean()) / current_price * 100
            
            avg_slope = (slope_h + slope_l) / 2
            norm_slope = avg_slope / (current_price / 100)
            
            if norm_slope > 0.05:
                pattern_name = 'Ascending Channel'
                pattern_type = 'Bullish Trend'
                target = round(current_price * 1.03, 2)
            elif norm_slope < -0.05:
                pattern_name = 'Descending Channel'
                pattern_type = 'Bearish Trend'
                target = round(current_price * 0.97, 2)
            else:
                pattern_name = 'Horizontal Channel'
                pattern_type = 'Range-Bound'
                # Target is the channel boundaries
                upper = intercept_h + slope_h * len(x)
                lower = intercept_l + slope_l * len(x)
                if current_price < (upper + lower) / 2:
                    target = round(upper, 2)
                else:
                    target = round(lower, 2)
            
            confidence = (r_h**2 + r_l**2) / 2 * 80 + slope_ratio * 20
            confidence = min(confidence, 95)
            
            # Position within channel
            upper_now = intercept_h + slope_h * (len(x) - 1)
            lower_now = intercept_l + slope_l * (len(x) - 1)
            position_pct = (current_price - lower_now) / (upper_now - lower_now) * 100 if upper_now != lower_now else 50
            
            patterns.append({
                'Pattern': f'{pattern_name} ({window}D)',
                'Type': pattern_type,
                'Confidence': round(confidence, 1),
                'Target': target,
                'Status': f'Price at {round(position_pct)}% of channel',
                'Channel_Width': f'{round(channel_width, 1)}%'
            })
        
        # Deduplicate
        seen_types = {}
        for p in sorted(patterns, key=lambda x: x['Confidence'], reverse=True):
            base_type = p['Pattern'].split(' (')[0]
            if base_type not in seen_types:
                seen_types[base_type] = p
        return list(seen_types.values())
    
    def detect_consolidation(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Consolidation / Range-Bound patterns.
        Useful when no classical patterns exist — tells user the market is coiling.
        """
        patterns = []
        df_analysis = df.tail(20)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        # Check if price is in a tight range
        price_range = (prices.max() - prices.min()) / current_price * 100
        volatility = prices.pct_change().std() * 100
        
        if price_range < 5:  # Less than 5% range in 20 days
            # Bollinger Band squeeze indicator
            ma20 = prices.mean()
            std20 = prices.std()
            bb_width = (2 * std20 / ma20) * 100
            
            confidence = max(40, 90 - price_range * 10)  # Tighter range = higher confidence
            
            # Determine likely breakout direction from trend
            x = np.arange(len(prices))
            slope, _, _, _, _ = linregress(x, prices.values)
            
            if slope > 0:
                bias = 'Bullish'
                target = round(current_price * (1 + price_range / 100), 2)
            elif slope < 0:
                bias = 'Bearish'
                target = round(current_price * (1 - price_range / 100), 2)
            else:
                bias = 'Neutral'
                target = round(current_price, 2)
            
            patterns.append({
                'Pattern': 'Consolidation / Squeeze',
                'Type': f'{bias} Breakout Expected',
                'Confidence': round(confidence, 1),
                'Target': target,
                'Status': f'Range: {round(price_range, 1)}%, BB Width: {round(bb_width, 1)}%',
                'Range_Pct': round(price_range, 1)
            })
        
        return patterns
    
    def detect_higher_high_lower_low(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Higher Highs / Higher Lows (uptrend) or Lower Highs / Lower Lows (downtrend).
        Simple but very reliable structural pattern.
        """
        patterns = []
        df_analysis = df.tail(40)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=3)
        
        if len(peak_idx) < 3 or len(trough_idx) < 3:
            return patterns
        
        # Check last 3 peaks and troughs
        recent_peaks = [prices.iloc[idx] for idx in peak_idx[-3:]]
        recent_troughs = [prices.iloc[idx] for idx in trough_idx[-3:]]
        
        # Higher Highs and Higher Lows = Uptrend
        hh = all(recent_peaks[i] > recent_peaks[i-1] for i in range(1, len(recent_peaks)))
        hl = all(recent_troughs[i] > recent_troughs[i-1] for i in range(1, len(recent_troughs)))
        
        # Lower Highs and Lower Lows = Downtrend
        lh = all(recent_peaks[i] < recent_peaks[i-1] for i in range(1, len(recent_peaks)))
        ll = all(recent_troughs[i] < recent_troughs[i-1] for i in range(1, len(recent_troughs)))
        
        if hh and hl:
            avg_rise = np.mean([recent_peaks[i] - recent_peaks[i-1] for i in range(1, len(recent_peaks))])
            target = round(current_price + avg_rise, 2)
            patterns.append({
                'Pattern': 'Higher Highs & Higher Lows',
                'Type': 'Bullish Trend Structure',
                'Confidence': round(85.0, 1),
                'Target': target,
                'Status': 'Active Uptrend',
                'Last_HH': round(recent_peaks[-1], 2),
                'Last_HL': round(recent_troughs[-1], 2)
            })
        elif lh and ll:
            avg_drop = np.mean([recent_peaks[i-1] - recent_peaks[i] for i in range(1, len(recent_peaks))])
            target = round(current_price - avg_drop, 2)
            patterns.append({
                'Pattern': 'Lower Highs & Lower Lows',
                'Type': 'Bearish Trend Structure',
                'Confidence': round(85.0, 1),
                'Target': target,
                'Status': 'Active Downtrend',
                'Last_LH': round(recent_peaks[-1], 2),
                'Last_LL': round(recent_troughs[-1], 2)
            })
        elif hh and not hl:
            patterns.append({
                'Pattern': 'Higher Highs (Diverging)',
                'Type': 'Weakening Bullish',
                'Confidence': 65.0,
                'Target': round(recent_peaks[-1], 2),
                'Status': 'Watch for reversal'
            })
        elif lh and not ll:
            patterns.append({
                'Pattern': 'Lower Highs (Compressing)',
                'Type': 'Building Bearish',
                'Confidence': 65.0,
                'Target': round(recent_troughs[-1], 2),
                'Status': 'Watch for breakdown'
            })
        
        return patterns

    def _run_detection_pass(self, df: pd.DataFrame, order: int, relaxed: bool = False) -> List[Dict]:
        """Run all pattern detectors with given parameters."""
        all_patterns = []
        all_patterns.extend(self.detect_double_top(df, order=order, relaxed=relaxed))
        all_patterns.extend(self.detect_double_bottom(df, order=order, relaxed=relaxed))
        all_patterns.extend(self.detect_head_and_shoulders(df, order=order, relaxed=relaxed))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(df, order=order, relaxed=relaxed))
        return all_patterns

    def analyze_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive pattern analysis with multi-timeframe scanning
        and automatic fallback to relaxed parameters.
        
        Strategy:
        1. Scan with multiple peak detection orders (3, 5, 7)
        2. Run triangle, wedge, and channel detectors
        3. If still no patterns, run relaxed detection pass
        4. Always detect structural patterns (HH/HL, consolidation)
        5. Add vision patterns from Roboflow
        """
        all_patterns = []
        
        # === PASS 1: Multi-timeframe classical pattern scan ===
        for scan_order in self.scan_orders:
            all_patterns.extend(self._run_detection_pass(df, order=scan_order, relaxed=False))
        
        # === PASS 2: Geometric patterns (triangles, wedges, channels) ===
        all_patterns.extend(self.detect_triangle_pattern(df))
        all_patterns.extend(self.detect_wedge_pattern(df))
        all_patterns.extend(self.detect_channel_pattern(df))
        
        # === PASS 3: Structural patterns (always run) ===
        all_patterns.extend(self.detect_higher_high_lower_low(df))
        all_patterns.extend(self.detect_consolidation(df))
        
        # === PASS 4: Fallback — if fewer than 2 patterns, retry with relaxed thresholds ===
        high_conf = [p for p in all_patterns if p and p.get('Confidence', 0) >= 50]
        if len(high_conf) < 2:
            for scan_order in [2, 3, 5]:
                all_patterns.extend(self._run_detection_pass(df, order=scan_order, relaxed=True))
        
        # === PASS 5: Vision patterns (Roboflow) ===
        vision_patterns = self.analyze_patterns_with_vision(df)
        all_patterns.extend(vision_patterns)
        
        # === Filter and deduplicate ===
        # Remove None entries and low-confidence
        valid_patterns = [p for p in all_patterns if p and p.get('Confidence', 0) >= 40]
        
        # Deduplicate by pattern name (keep highest confidence)
        seen = {}
        for p in sorted(valid_patterns, key=lambda x: x.get('Confidence', 0), reverse=True):
            base_name = p['Pattern'].split(' (')[0]  # Remove window suffix
            if base_name not in seen:
                seen[base_name] = p
        
        high_quality_patterns = list(seen.values())
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
            'patterns': high_quality_patterns[:8],  # Max 8 patterns
            'trend': trend,
            'support_resistance': sr_levels,
            'overall_bias': pattern_bias,
            'pattern_count': len(high_quality_patterns)
        }


# Backward compatibility
VISUAL_AI_AVAILABLE = True
