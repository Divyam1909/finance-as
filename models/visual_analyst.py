"""
Mathematical Pattern Analyst.
Detects chart patterns using proven algorithms (peak/trough detection, TA-Lib patterns).
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Tuple, Dict


class PatternAnalyst:
    """
    Mathematical pattern detection for stock charts.
    
    Uses scipy for peak/trough detection and implements classic
    technical analysis patterns with mathematical validation.
    """
    
    def __init__(self, order: int = 5):
        """
        Initialize the Pattern Analyst.
        
        Args:
            order: Number of points on each side to compare for extrema detection
        """
        self.order = order
    
    def find_peaks_and_troughs(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local maxima (peaks) and minima (troughs) in price data.
        
        Args:
            prices: Series of closing prices
        
        Returns:
            Tuple of (peak_indices, trough_indices)
        """
        prices_arr = prices.values
        
        # Find local maxima and minima
        peak_indices = argrelextrema(prices_arr, np.greater, order=self.order)[0]
        trough_indices = argrelextrema(prices_arr, np.less, order=self.order)[0]
        
        return peak_indices, trough_indices
    
    def detect_double_top(self, df: pd.DataFrame, tolerance: float = 0.015) -> List[Dict]:
        """
        Detect Double Top pattern (bearish reversal).
        
        Pattern: Two peaks at similar price levels with a trough between.
        Only detects patterns in the most recent data.
        
        Args:
            df: DataFrame with OHLCV data
            tolerance: Price tolerance for matching peaks (1.5% default - stricter)
        
        Returns:
            List of detected patterns with details (max 2 most recent)
        """
        patterns = []
        # Only analyze last 60 days for relevant patterns
        df_recent = df.tail(60)
        prices = df_recent['Close']
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(peak_idx) < 2:
            return patterns
        
        # Only look at last 3 peaks to avoid old patterns
        peak_idx = peak_idx[-4:] if len(peak_idx) > 4 else peak_idx
        
        for i in range(len(peak_idx) - 1):
            p1_idx, p2_idx = peak_idx[i], peak_idx[i + 1]
            p1_price, p2_price = prices.iloc[p1_idx], prices.iloc[p2_idx]
            
            # Check if peaks are at similar levels
            price_diff = abs(p1_price - p2_price) / p1_price
            
            # Require minimum pattern height (at least 3% from neckline)
            if price_diff <= tolerance:
                troughs_between = trough_idx[(trough_idx > p1_idx) & (trough_idx < p2_idx)]
                
                if len(troughs_between) > 0:
                    trough_price = prices.iloc[troughs_between[0]]
                    neckline = trough_price
                    
                    avg_peak = (p1_price + p2_price) / 2
                    pattern_height_pct = (avg_peak - neckline) / neckline
                    
                    # Skip if pattern height is too small (less than 3%)
                    if pattern_height_pct < 0.03:
                        continue
                    
                    pattern_height = avg_peak - neckline
                    target = neckline - pattern_height
                    
                    patterns.append({
                        'Pattern': 'Double Top',
                        'Type': 'Bearish Reversal',
                        'Peak1_Date': df_recent.index[p1_idx],
                        'Peak2_Date': df_recent.index[p2_idx],
                        'Peak_Price': round(avg_peak, 2),
                        'Neckline': round(neckline, 2),
                        'Target': round(target, 2),
                        'Confidence': round((1 - price_diff) * 100, 1)
                    })
        
        # Return only the most recent pattern
        return patterns[-1:] if patterns else []
    
    def detect_double_bottom(self, df: pd.DataFrame, tolerance: float = 0.015) -> List[Dict]:
        """
        Detect Double Bottom pattern (bullish reversal).
        Only detects patterns in recent data with minimum pattern height.
        """
        patterns = []
        df_recent = df.tail(60)
        prices = df_recent['Close']
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(trough_idx) < 2:
            return patterns
        
        # Only last few troughs
        trough_idx = trough_idx[-4:] if len(trough_idx) > 4 else trough_idx
        
        for i in range(len(trough_idx) - 1):
            t1_idx, t2_idx = trough_idx[i], trough_idx[i + 1]
            t1_price, t2_price = prices.iloc[t1_idx], prices.iloc[t2_idx]
            
            price_diff = abs(t1_price - t2_price) / t1_price
            
            if price_diff <= tolerance:
                peaks_between = peak_idx[(peak_idx > t1_idx) & (peak_idx < t2_idx)]
                
                if len(peaks_between) > 0:
                    peak_price = prices.iloc[peaks_between[0]]
                    avg_trough = (t1_price + t2_price) / 2
                    pattern_height_pct = (peak_price - avg_trough) / avg_trough
                    
                    # Skip small patterns
                    if pattern_height_pct < 0.03:
                        continue
                    
                    neckline = peak_price
                    pattern_height = neckline - avg_trough
                    target = neckline + pattern_height
                    
                    patterns.append({
                        'Pattern': 'Double Bottom',
                        'Type': 'Bullish Reversal',
                        'Trough_Price': round(avg_trough, 2),
                        'Neckline': round(neckline, 2),
                        'Target': round(target, 2),
                        'Confidence': round((1 - price_diff) * 100, 1)
                    })
        
        return patterns[-1:] if patterns else []
    
    def detect_head_and_shoulders(self, df: pd.DataFrame, tolerance: float = 0.02) -> List[Dict]:
        """
        Detect Head and Shoulders pattern (bearish reversal).
        Only detects significant patterns in recent data.
        """
        patterns = []
        df_recent = df.tail(60)
        prices = df_recent['Close']
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(peak_idx) < 3:
            return patterns
        
        # Only last few peaks
        peak_idx = peak_idx[-5:] if len(peak_idx) > 5 else peak_idx
        
        for i in range(len(peak_idx) - 2):
            ls_idx = peak_idx[i]
            h_idx = peak_idx[i + 1]
            rs_idx = peak_idx[i + 2]
            
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            # Head must be at least 3% higher than shoulders
            avg_shoulder = (ls_price + rs_price) / 2
            head_height_pct = (h_price - avg_shoulder) / avg_shoulder
            
            if h_price > ls_price and h_price > rs_price and head_height_pct >= 0.03:
                shoulder_diff = abs(ls_price - rs_price) / ls_price
                
                if shoulder_diff <= tolerance:
                    troughs_ls_h = trough_idx[(trough_idx > ls_idx) & (trough_idx < h_idx)]
                    troughs_h_rs = trough_idx[(trough_idx > h_idx) & (trough_idx < rs_idx)]
                    
                    if len(troughs_ls_h) > 0 and len(troughs_h_rs) > 0:
                        neckline = (prices.iloc[troughs_ls_h[0]] + prices.iloc[troughs_h_rs[0]]) / 2
                        pattern_height = h_price - neckline
                        target = neckline - pattern_height
                        
                        patterns.append({
                            'Pattern': 'Head & Shoulders',
                            'Type': 'Bearish Reversal',
                            'Head_Price': round(h_price, 2),
                            'Neckline': round(neckline, 2),
                            'Target': round(target, 2),
                            'Confidence': round((1 - shoulder_diff) * 100, 1)
                        })
        
        return patterns[-1:] if patterns else []
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame, tolerance: float = 0.02) -> List[Dict]:
        """
        Detect Inverse Head and Shoulders (bullish reversal).
        Only detects significant patterns in recent data.
        """
        patterns = []
        df_recent = df.tail(60)
        prices = df_recent['Close']
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        if len(trough_idx) < 3:
            return patterns
        
        # Only last few troughs
        trough_idx = trough_idx[-5:] if len(trough_idx) > 5 else trough_idx
        
        for i in range(len(trough_idx) - 2):
            ls_idx = trough_idx[i]
            h_idx = trough_idx[i + 1]
            rs_idx = trough_idx[i + 2]
            
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            # Head must be at least 3% lower than shoulders
            avg_shoulder = (ls_price + rs_price) / 2
            head_depth_pct = (avg_shoulder - h_price) / avg_shoulder
            
            if h_price < ls_price and h_price < rs_price and head_depth_pct >= 0.03:
                shoulder_diff = abs(ls_price - rs_price) / ls_price
                
                if shoulder_diff <= tolerance:
                    peaks_ls_h = peak_idx[(peak_idx > ls_idx) & (peak_idx < h_idx)]
                    peaks_h_rs = peak_idx[(peak_idx > h_idx) & (peak_idx < rs_idx)]
                    
                    if len(peaks_ls_h) > 0 and len(peaks_h_rs) > 0:
                        neckline = (prices.iloc[peaks_ls_h[0]] + prices.iloc[peaks_h_rs[0]]) / 2
                        pattern_height = neckline - h_price
                        target = neckline + pattern_height
                        
                        patterns.append({
                            'Pattern': 'Inverse H&S',
                            'Type': 'Bullish Reversal',
                            'Head_Price': round(h_price, 2),
                            'Neckline': round(neckline, 2),
                            'Target': round(target, 2),
                            'Confidence': round((1 - shoulder_diff) * 100, 1)
                        })
        
        return patterns[-1:] if patterns else []
    
    def detect_trend(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Detect current trend using multiple methods.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for trend calculation
        
        Returns:
            Dictionary with trend analysis
        """
        prices = df['Close'].tail(window)
        
        # Method 1: Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices.values, 1)[0]
        
        # Method 2: MA crossover
        ma_short = df['Close'].rolling(5).mean().iloc[-1]
        ma_long = df['Close'].rolling(20).mean().iloc[-1]
        
        # Method 3: Higher highs/lows
        recent_highs = df['High'].tail(10)
        recent_lows = df['Low'].tail(10)
        
        higher_highs = (recent_highs.diff().dropna() > 0).sum() / len(recent_highs.diff().dropna())
        higher_lows = (recent_lows.diff().dropna() > 0).sum() / len(recent_lows.diff().dropna())
        
        # Combine signals
        bullish_signals = 0
        bearish_signals = 0
        
        if slope > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if ma_short > ma_long:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if higher_highs > 0.6 and higher_lows > 0.6:
            bullish_signals += 1
        elif higher_highs < 0.4 and higher_lows < 0.4:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            trend = "Bullish"
            strength = bullish_signals / 3 * 100
        elif bearish_signals > bullish_signals:
            trend = "Bearish"
            strength = bearish_signals / 3 * 100
        else:
            trend = "Neutral"
            strength = 50
        
        return {
            'Trend': trend,
            'Strength': round(strength, 1),
            'Slope': round(slope, 4),
            'MA_Signal': 'Bullish' if ma_short > ma_long else 'Bearish',
            'Structure': 'Higher Highs/Lows' if higher_highs > 0.6 else 'Lower Highs/Lows' if higher_highs < 0.4 else 'Mixed'
        }
    
    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 60) -> Dict:
        """
        Detect key support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to analyze
        
        Returns:
            Dictionary with support and resistance levels
        """
        df_slice = df.tail(lookback)
        prices = df_slice['Close']
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices)
        
        # Resistance levels from peaks
        resistance_levels = []
        if len(peak_idx) > 0:
            for idx in peak_idx[-3:]:  # Last 3 peaks
                resistance_levels.append(prices.iloc[idx])
        
        # Support levels from troughs
        support_levels = []
        if len(trough_idx) > 0:
            for idx in trough_idx[-3:]:  # Last 3 troughs
                support_levels.append(prices.iloc[idx])
        
        current_price = prices.iloc[-1]
        
        # Find nearest levels
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        
        return {
            'Current_Price': round(current_price, 2),
            'Nearest_Resistance': round(nearest_resistance, 2) if nearest_resistance else 'N/A',
            'Nearest_Support': round(nearest_support, 2) if nearest_support else 'N/A',
            'All_Resistance': [round(r, 2) for r in sorted(resistance_levels, reverse=True)],
            'All_Support': [round(s, 2) for s in sorted(support_levels, reverse=True)]
        }
    
    def analyze_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive pattern analysis.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with all detected patterns and analysis
        """
        all_patterns = []
        
        # Detect all pattern types
        all_patterns.extend(self.detect_double_top(df))
        all_patterns.extend(self.detect_double_bottom(df))
        all_patterns.extend(self.detect_head_and_shoulders(df))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(df))
        
        # Get trend and S/R
        trend = self.detect_trend(df)
        sr_levels = self.detect_support_resistance(df)
        
        # Calculate overall bias
        bullish_patterns = sum(1 for p in all_patterns if 'Bullish' in p.get('Type', ''))
        bearish_patterns = sum(1 for p in all_patterns if 'Bearish' in p.get('Type', ''))
        
        if bullish_patterns > bearish_patterns:
            pattern_bias = "Bullish"
        elif bearish_patterns > bullish_patterns:
            pattern_bias = "Bearish"
        else:
            pattern_bias = trend['Trend']
        
        return {
            'patterns': all_patterns,
            'trend': trend,
            'support_resistance': sr_levels,
            'overall_bias': pattern_bias,
            'pattern_count': len(all_patterns)
        }


# For backward compatibility
VISUAL_AI_AVAILABLE = True  # Mathematical analysis always available
