"""
Technical Analysis Module
Handles divergence detection, confirmations, and volume analysis
"""

import sys
import os
# Add parent directory to path to access config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema
from typing import Optional, Dict, List, Any, Tuple
import logging
from config import *

class TechnicalAnalyzer:
    """Handles technical analysis operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_1h_confirmation(self, exchange: str, symbol: str, setup_type: str, data_fetcher) -> bool:
        """Enhanced 1H confirmation with BALANCED bounce/rejection requirements"""
        try:
            df_1h = data_fetcher.fetch_ohlcv(exchange, symbol, '1h')
            if df_1h is None or len(df_1h) < 5:
                return True  # Don't reject due to missing data
            
            # Calculate 1H indicators
            bb_1h = ta.bbands(df_1h['close'], length=20, std=2)
            df_1h['bb_upper'] = bb_1h['BBU_20_2.0']
            df_1h['bb_lower'] = bb_1h['BBL_20_2.0']
            df_1h['bb_position'] = ((df_1h['close'] - df_1h['bb_lower']) / 
                                   (df_1h['bb_upper'] - df_1h['bb_lower']) * 100)
            df_1h['rsi'] = ta.rsi(df_1h['close'], length=14)
            
            current_1h = df_1h.iloc[-1]
            
            if setup_type == 'LONG':
                # Must be in lower 30% OR showing upward momentum (less restrictive)
                in_lower_zone = current_1h['bb_position'] <= 30
                green_candle = current_1h['close'] > current_1h['open']
                confirmed = in_lower_zone or green_candle  # OR instead of AND
                        
            elif setup_type == 'SHORT':
                # Must be in upper 30% OR showing downward momentum (less restrictive)
                in_upper_zone = current_1h['bb_position'] >= 70
                red_candle = current_1h['close'] < current_1h['open']
                confirmed = in_upper_zone or red_candle  # OR instead of AND
            
            return confirmed
            
        except Exception as e:
            self.logger.debug(f"Error getting 1H confirmation for {symbol}: {e}")
            return True  # Don't reject due to errors

    def find_adaptive_swings(self, df: pd.DataFrame, column: str, swing_type: str = 'low', 
                            min_periods: int = 3, max_periods: int = 40) -> List[Dict]:
        """
        Find swings with adaptive lookback periods for multiple timeframe analysis
        """
        try:
            swings = []
            data = df[column].values if swing_type == 'low' else df[column].values
            
            # Multiple swing detection windows
            timeframes = [
                {'name': 'short', 'window': 3, 'lookback': 8},     # 3-8 periods (12-32h)
                {'name': 'medium', 'window': 5, 'lookback': 20},   # 5-20 periods (20h-3.3d)  
                {'name': 'long', 'window': 8, 'lookback': 40}      # 8-40 periods (1.3-6.7d)
            ]
            
            for tf in timeframes:
                if swing_type == 'low':
                    swing_indices = argrelextrema(data, np.less, order=tf['window'])[0]
                else:
                    swing_indices = argrelextrema(data, np.greater, order=tf['window'])[0]
                
                # Focus on recent swings within lookback period
                recent_indices = [idx for idx in swing_indices if idx >= len(df) - tf['lookback']]
                
                for idx in recent_indices:
                    swings.append({
                        'index': idx,
                        'value': data[idx],
                        'timeframe': tf['name'],
                        'timestamp': df.index[idx] if hasattr(df.index, '__getitem__') else idx,
                        'significance': self._calculate_swing_significance(df, idx, column)
                    })
            
            # Sort by recency and significance
            swings.sort(key=lambda x: (x['index'], x['significance']), reverse=True)
            return swings
            
        except Exception as e:
            self.logger.debug(f"Error finding adaptive swings for {column}: {e}")
            return []

    def _calculate_swing_significance(self, df: pd.DataFrame, idx: int, column: str) -> float:
        """Calculate how significant a swing is based on magnitude and context"""
        try:
            if idx < 5 or idx >= len(df) - 1:
                return 0.0
                
            # Look at surrounding values to determine significance
            window = 10
            start_idx = max(0, idx - window)
            end_idx = min(len(df), idx + window)
            
            local_data = df[column].iloc[start_idx:end_idx]
            swing_value = df[column].iloc[idx]
            
            # Calculate how extreme this swing is compared to local range
            local_range = local_data.max() - local_data.min()
            if local_range == 0:
                return 0.0
                
            # Distance from local mean as significance measure
            local_mean = local_data.mean()
            significance = abs(swing_value - local_mean) / local_range
            
            return min(significance, 1.0)  # Cap at 1.0
            
        except Exception:
            return 0.0

    def detect_enhanced_bullish_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced bullish divergence detection with multi-timeframe analysis
        Price making lower lows while indicators making higher lows
        """
        try:
            # Find price swings (lows)
            price_swings = self.find_adaptive_swings(df, 'low', 'low')
            
            if len(price_swings) < 2:
                return self._empty_divergence_result()
            
            # Indicators to check for divergence
            indicators = [
                ('rsi', 'RSI'),
                ('macd', 'MACD'),
                ('stoch_k', 'STOCH'),
                ('obv', 'OBV'),
                ('cci', 'CCI')
            ]
            
            divergent_signals = []
            total_strength = 0
            
            # Check each indicator across multiple timeframes
            for indicator_col, indicator_name in indicators:
                if indicator_col not in df.columns:
                    continue
                    
                indicator_swings = self.find_adaptive_swings(df, indicator_col, 'low')
                
                if len(indicator_swings) < 2:
                    continue
                
                # Find divergence patterns across different timeframes
                divergence_found = self._analyze_divergence_patterns(
                    price_swings, indicator_swings, 'bullish'
                )
                
                if divergence_found['detected']:
                    divergent_signals.append({
                        'indicator': indicator_name,
                        'strength': divergence_found['strength'],
                        'timeframe': divergence_found['timeframe'],
                        'confidence': divergence_found['confidence']
                    })
                    total_strength += divergence_found['strength']
            
            # Determine overall divergence result
            detected = len(divergent_signals) >= 1
            confidence = self._calculate_overall_confidence(divergent_signals)
            
            return {
                'detected': detected,
                'strength': len(divergent_signals),
                'total_strength': total_strength,
                'indicators': [signal['indicator'] for signal in divergent_signals],
                'confidence': confidence,
                'signals': divergent_signals,
                'timeframes_detected': list(set([s['timeframe'] for s in divergent_signals]))
            }
            
        except Exception as e:
            self.logger.debug(f"Error in enhanced bullish divergence detection: {e}")
            return self._empty_divergence_result()

    def detect_enhanced_bearish_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced bearish divergence detection with multi-timeframe analysis
        Price making higher highs while indicators making lower highs
        """
        try:
            # Find price swings (highs)
            price_swings = self.find_adaptive_swings(df, 'high', 'high')
            
            if len(price_swings) < 2:
                return self._empty_divergence_result()
            
            # Indicators to check for divergence
            indicators = [
                ('rsi', 'RSI'),
                ('macd', 'MACD'),
                ('stoch_k', 'STOCH'),
                ('obv', 'OBV'),
                ('cci', 'CCI')
            ]
            
            divergent_signals = []
            total_strength = 0
            
            # Check each indicator across multiple timeframes
            for indicator_col, indicator_name in indicators:
                if indicator_col not in df.columns:
                    continue
                    
                indicator_swings = self.find_adaptive_swings(df, indicator_col, 'high')
                
                if len(indicator_swings) < 2:
                    continue
                
                # Find divergence patterns across different timeframes
                divergence_found = self._analyze_divergence_patterns(
                    price_swings, indicator_swings, 'bearish'
                )
                
                if divergence_found['detected']:
                    divergent_signals.append({
                        'indicator': indicator_name,
                        'strength': divergence_found['strength'],
                        'timeframe': divergence_found['timeframe'],
                        'confidence': divergence_found['confidence']
                    })
                    total_strength += divergence_found['strength']
            
            # Determine overall divergence result
            detected = len(divergent_signals) >= 1
            confidence = self._calculate_overall_confidence(divergent_signals)
            
            return {
                'detected': detected,
                'strength': len(divergent_signals),
                'total_strength': total_strength,
                'indicators': [signal['indicator'] for signal in divergent_signals],
                'confidence': confidence,
                'signals': divergent_signals,
                'timeframes_detected': list(set([s['timeframe'] for s in divergent_signals]))
            }
            
        except Exception as e:
            self.logger.debug(f"Error in enhanced bearish divergence detection: {e}")
            return self._empty_divergence_result()

    def _analyze_divergence_patterns(self, price_swings: List[Dict], 
                                    indicator_swings: List[Dict], 
                                    divergence_type: str) -> Dict[str, Any]:
        """
        Analyze divergence patterns between price and indicator swings - ENHANCED SENSITIVITY
        """
        try:
            if len(price_swings) < 2 or len(indicator_swings) < 2:
                return {'detected': False, 'strength': 0, 'timeframe': 'none', 'confidence': 'None'}
            
            best_divergence = {'detected': False, 'strength': 0, 'timeframe': 'none', 'confidence': 'None'}
            
            # ENHANCED: Check more swing combinations
            for price_swing1 in price_swings[:5]:  # Increased from 3 to 5
                for price_swing2 in price_swings[:8]:  # Increased from 5 to 8
                    if price_swing1['index'] >= price_swing2['index']:
                        continue
                        
                    # Find corresponding indicator swings in similar time periods
                    for ind_swing1 in indicator_swings[:5]:  # Limit to prevent too many combinations
                        for ind_swing2 in indicator_swings[:8]:
                            if ind_swing1['index'] >= ind_swing2['index']:
                                continue
                                
                            # ENHANCED: More flexible proximity matching (±8 periods instead of ±5)
                            if (abs(price_swing1['index'] - ind_swing1['index']) <= 8 and
                                abs(price_swing2['index'] - ind_swing2['index']) <= 8):
                                
                                divergence_result = self._check_divergence_condition(
                                    price_swing1, price_swing2, ind_swing1, ind_swing2, divergence_type
                                )
                                
                                if (divergence_result['detected'] and 
                                    divergence_result['strength'] > best_divergence['strength']):
                                    best_divergence = divergence_result
            
            return best_divergence
            
        except Exception as e:
            return {'detected': False, 'strength': 0, 'timeframe': 'none', 'confidence': 'None'}

    def _check_divergence_condition(self, price_swing1: Dict, price_swing2: Dict,
                                   ind_swing1: Dict, ind_swing2: Dict, 
                                   divergence_type: str) -> Dict[str, Any]:
        """
        Check if specific swing pairs show divergence - ENHANCED SENSITIVITY
        """
        try:
            if divergence_type == 'bullish':
                # Bullish: Price lower low, Indicator higher low
                price_divergence = price_swing2['value'] < price_swing1['value']  # Later price is lower
                indicator_divergence = ind_swing2['value'] > ind_swing1['value']  # Later indicator is higher
            else:
                # Bearish: Price higher high, Indicator lower high  
                price_divergence = price_swing2['value'] > price_swing1['value']  # Later price is higher
                indicator_divergence = ind_swing2['value'] < ind_swing1['value']  # Later indicator is lower
            
            detected = price_divergence and indicator_divergence
            
            if not detected:
                return {'detected': False, 'strength': 0, 'timeframe': 'none', 'confidence': 'None'}
            
            # ENHANCED: More lenient strength calculation
            price_change_pct = abs(price_swing2['value'] - price_swing1['value']) / price_swing1['value']
            
            # ENHANCED: More sensitive timeframe detection
            period_distance = abs(price_swing2['index'] - price_swing1['index'])
            if period_distance <= 10:  # Increased from 8
                timeframe = 'short'
                base_strength = 1.5  # Increased from 1.0
            elif period_distance <= 25:  # Increased from 20
                timeframe = 'medium'
                base_strength = 2.0  # Increased from 1.5
            else:
                timeframe = 'long'
                base_strength = 2.5  # Increased from 2.0
            
            # ENHANCED: More generous strength calculation
            significance_weight = (price_swing1['significance'] + price_swing2['significance'] + 
                                 ind_swing1['significance'] + ind_swing2['significance']) / 4
            
            # ENHANCED: Reduced multiplier to make detection easier
            strength = base_strength * (1 + significance_weight * 0.5) * (1 + price_change_pct * 5)  # Reduced multipliers
            
            # ENHANCED: Lower confidence thresholds
            if strength >= 2.0:  # Reduced from 3.0
                confidence = 'Strong'
            elif strength >= 1.5:  # Reduced from 2.0
                confidence = 'Moderate'
            elif strength >= 0.8:  # Reduced from 1.0
                confidence = 'Weak'
            else:
                confidence = 'Very Weak'
            
            return {
                'detected': True,
                'strength': strength,
                'timeframe': timeframe,
                'confidence': confidence
            }
            
        except Exception:
            return {'detected': False, 'strength': 0, 'timeframe': 'none', 'confidence': 'None'}

    def _calculate_overall_confidence(self, signals: List[Dict]) -> str:
        """Calculate overall confidence from multiple divergence signals"""
        if not signals:
            return 'None'
        
        total_strength = sum(signal['strength'] for signal in signals)
        num_signals = len(signals)
        
        # Weight by number of confirming indicators
        if num_signals >= 3 and total_strength >= 6.0:
            return 'Very Strong'
        elif num_signals >= 2 and total_strength >= 4.0:
            return 'Strong'
        elif num_signals >= 2 and total_strength >= 2.0:
            return 'Moderate'
        elif num_signals >= 1:
            return 'Weak'
        else:
            return 'None'

    def _empty_divergence_result(self) -> Dict[str, Any]:
        """Return empty divergence result structure"""
        return {
            'detected': False,
            'strength': 0,
            'total_strength': 0,
            'indicators': [],
            'confidence': 'None',
            'signals': [],
            'timeframes_detected': []
        }