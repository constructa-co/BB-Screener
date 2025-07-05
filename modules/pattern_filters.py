"""
Pattern Filters - ChatGPT Pattern Filtering System
Implements ChatGPT's recommendations to reduce false positives and noise
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

class PatternFilters:
    """
    Pattern filtering system implementing ChatGPT's recommendations:
    1. ATR Significance Filter - Minimum 1.5x ATR for pattern validity
    2. Volume Confirmation Filter - 1.5x+ average volume required
    3. Pattern Clustering Detection - Multiple patterns within 3 candles
    4. Range Filter - Minimum candle range requirements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Filter thresholds (ChatGPT recommendations)
        self.filter_thresholds = {
            'atr_significance': 1.5,  # Minimum 1.5x ATR for significance
            'volume_confirmation': 1.5,  # Minimum 1.5x average volume
            'minimum_range_atr': 0.8,  # Minimum 0.8x ATR for candle range
            'clustering_window': 3  # Pattern clustering within 3 candles
        }
        
        # ADX thresholds for trend filtering (ChatGPT recommendation)
        self.adx_thresholds = {
            'strong_trend': 25,  # ADX > 25 = trending market
            'weak_trend': 20  # ADX < 20 = choppy market
        }
        
        self.logger.info("Pattern Filters initialized with ChatGPT recommendations")

    def is_atr_significant(self, pattern_data: Dict, atr_value: float) -> bool:
        """Check if pattern meets ATR significance threshold"""
        try:
            pattern_range = pattern_data.get('range', 0)
            if atr_value == 0:
                return True  # Default to pass if no ATR data
            
            atr_multiple = pattern_range / atr_value
            return atr_multiple >= self.filter_thresholds['atr_significance']
        except Exception as e:
            self.logger.warning(f"ATR significance check failed: {e}")
            return True  # Default to pass on error

    def has_volume_confirmation(self, pattern_data: Dict, df: pd.DataFrame) -> bool:
        """Check if pattern has volume confirmation"""
        try:
            if len(df) < 20:
                return True  # Default to pass if insufficient data
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()
            
            if avg_volume == 0:
                return True  # Default to pass if no volume data
            
            volume_multiple = current_volume / avg_volume
            return volume_multiple >= self.filter_thresholds['volume_confirmation']
        except Exception as e:
            self.logger.warning(f"Volume confirmation check failed: {e}")
            return True  # Default to pass on error

    def has_minimum_range(self, pattern_data: Dict, atr_value: float) -> bool:
        """Check if pattern meets minimum range requirements"""
        try:
            pattern_range = pattern_data.get('range', 0)
            if atr_value == 0:
                return True  # Default to pass if no ATR data
            
            range_multiple = pattern_range / atr_value
            return range_multiple >= self.filter_thresholds['minimum_range_atr']
        except Exception as e:
            self.logger.warning(f"Minimum range check failed: {e}")
            return True  # Default to pass on error

    def passes_all_filters(self, pattern_data: Dict, df: pd.DataFrame, atr_value: float) -> bool:
        """Check if pattern passes all filters"""
        return (self.is_atr_significant(pattern_data, atr_value) and
                self.has_volume_confirmation(pattern_data, df) and
                self.has_minimum_range(pattern_data, atr_value))