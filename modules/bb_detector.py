# bb_detector.py - Bollinger Band Detection Module
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from config import *

logger = logging.getLogger(__name__)

class BBDetector:
    """Bollinger Band bounce detection with adaptive stops"""
    
    def __init__(self):
        self.bb_period = BB_CONFIG["period"]
        self.bb_std_dev = BB_CONFIG["std_dev"]
        self.atr_stop_multiplier = BB_CONFIG["atr_stop_multiplier"]
        self.min_candles = BB_CONFIG["min_candles_required"]
        
    def analyze_bb_setup(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze BB setup with BALANCED criteria for true BB bounces"""
        if df is None or len(df) < self.min_candles:
            return self._get_empty_setup()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Volume pre-filter - RELAXED (was 1.0)
        if last['volume_ratio'] < 0.8:  # Allow 80% of average volume
            return self._get_empty_setup()
        
        # Check for actual BB band touches in recent candles
        recent_3_candles = df.tail(3)
        
        # Calculate scores for both LONG and SHORT
        long_score = self._calculate_long_score(recent_3_candles, last)
        short_score = self._calculate_short_score(recent_3_candles, last)
        
        # Determine setup - BACK TO ORIGINAL threshold
        if long_score >= short_score and long_score >= 5:  # Back to original threshold
            setup_type = 'LONG'
            bb_score = long_score
            entry = last['close']
            
            # NEW: Use adaptive stop loss calculation
            stop = self._calculate_adaptive_stop_loss(
                entry_price=entry,
                atr=last['atr'],
                bb_upper=last['bb_upper'],
                bb_lower=last['bb_lower'],
                setup_type='LONG',
                df=df
            )
            target1 = last['bb_middle']
            
        elif short_score > long_score and short_score >= 5:  # Back to original threshold
            setup_type = 'SHORT'
            bb_score = short_score
            entry = last['close']
            
            # NEW: Use adaptive stop loss calculation
            stop = self._calculate_adaptive_stop_loss(
                entry_price=entry,
                atr=last['atr'],
                bb_upper=last['bb_upper'],
                bb_lower=last['bb_lower'],
                setup_type='SHORT',
                df=df
            )
            target1 = last['bb_middle']
            
        else:
            return self._get_empty_setup_with_score(max(long_score, short_score))
        
        # Quality assessment
        if bb_score >= 9:
            setup_quality = 'Excellent'
        elif bb_score >= 7:
            setup_quality = 'Good'
        elif bb_score >= 5:
            setup_quality = 'Fair'
        else:
            setup_quality = 'Poor'
        
        # Risk/Reward calculation
        if setup_type != 'NONE' and entry != 0 and stop != 0:
            if setup_type == 'LONG':
                risk = entry - stop
                reward = target1 - entry
            else:
                risk = stop - entry
                reward = entry - target1
            
            risk_reward = round(reward / risk, 2) if risk > 0 else 0
        else:
            risk_reward = 0
        
        return {
            'setup_type': setup_type,
            'bb_score': bb_score,
            'setup_quality': setup_quality,
            'entry': entry,
            'stop': stop,
            'target1': target1,
            'risk_reward': risk_reward
        }

    def _calculate_adaptive_stop_loss(self, entry_price: float, atr: float, bb_upper: float, bb_lower: float, setup_type: str, df: pd.DataFrame) -> float:
        """Calculate adaptive stop loss using scientific approach"""
        try:
            # Option 1: ATR-based with minimum floor
            atr_stop_distance = max(
                self.atr_stop_multiplier * atr,  # 2x ATR
                entry_price * 0.025  # 2.5% minimum stop distance
            )
            
            # Option 3: BB-width based stop
            bb_width = bb_upper - bb_lower
            bb_stop_distance = bb_width * 0.12  # 12% of BB channel width
            
            # Use the LARGER of the two (more conservative)
            final_stop_distance = max(atr_stop_distance, bb_stop_distance)
            
            # Apply based on setup type
            if setup_type == 'LONG':
                stop_loss = entry_price - final_stop_distance
                # Ensure stop is never above entry for LONG
                stop_loss = min(stop_loss, entry_price * 0.975)  # At least 2.5% below entry
                
                # Additional check: don't go below recent significant low
                recent_low = df['low'].tail(10).min()
                stop_loss = max(stop_loss, recent_low * 0.99)  # 1% below recent low
                
            else:  # SHORT
                stop_loss = entry_price + final_stop_distance
                # Ensure stop is never below entry for SHORT
                stop_loss = max(stop_loss, entry_price * 1.025)  # At least 2.5% above entry
                
                # Additional check: don't go above recent significant high
                recent_high = df['high'].tail(10).max()
                stop_loss = min(stop_loss, recent_high * 1.01)  # 1% above recent high
            
            logger.debug(f"{setup_type} stop calculation: ATR_dist={atr_stop_distance:.6f}, BB_dist={bb_stop_distance:.6f}, Final_dist={final_stop_distance:.6f}, Stop={stop_loss:.6f}")
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stop loss: {e}")
            # Fallback: simple 3% stop
            if setup_type == 'LONG':
                return entry_price * 0.97
            else:
                return entry_price * 1.03

    def _calculate_long_score(self, recent_3_candles: pd.DataFrame, last: pd.Series) -> int:
        """Calculate LONG setup score - BALANCED thresholds"""
        long_score = 0
        
        # 1. BB Touch (3 points) - Must actually touch lower band
        if any(recent_3_candles['low'] <= recent_3_candles['bb_lower']):
            long_score += 3
        
        # 2. BB Position (2 points) - BALANCED thresholds
        if last['bb_pct'] <= 0.05:      # Extremely extreme (back to original)
            long_score += 2
        elif last['bb_pct'] <= 0.08:    # Very extreme (compromise)
            long_score += 1
        
        # 3. RSI (2 points) - BALANCED oversold levels
        if last['rsi'] <= 28:        # Extremely oversold (compromise)
            long_score += 2
        elif last['rsi'] <= 38:      # Oversold (compromise)
            long_score += 1
        
        # 4. Volume (2 points) - RELAXED requirements
        if last['volume_ratio'] >= 1.8:     # High conviction (reduced from 2.0)
            long_score += 2
        elif last['volume_ratio'] >= 1.3:   # Good volume (reduced from 1.5)
            long_score += 1
        
        # 5. Bounce Confirmation (1 point) - Basic bounce
        if last['close'] > last['low'] and last['close'] > last['bb_lower']:
            long_score += 1
        
        return long_score

    def _calculate_short_score(self, recent_3_candles: pd.DataFrame, last: pd.Series) -> int:
        """Calculate SHORT setup score - ENHANCED for better detection"""
        short_score = 0
        
        # 1. BB Touch (3 points) - Must actually touch upper band
        if any(recent_3_candles['high'] >= recent_3_candles['bb_upper']):
            short_score += 3
        # NEW: Give 2 points for being very close to upper band
        elif any(recent_3_candles['high'] >= recent_3_candles['bb_upper'] * 0.998):
            short_score += 2
        
        # 2. BB Position (2 points) - ENHANCED thresholds
        if last['bb_pct'] >= 0.92:     # Very extreme (enhanced)
            short_score += 2
        elif last['bb_pct'] >= 0.88:   # Extreme (enhanced)
            short_score += 1
        
        # 3. RSI (2 points) - ENHANCED overbought levels
        if last['rsi'] >= 68:        # Overbought (enhanced)
            short_score += 2
        elif last['rsi'] >= 58:      # Moderately overbought (enhanced)
            short_score += 1
        
        # 4. Volume (2 points) - RELAXED requirements
        if last['volume_ratio'] >= 1.8:     # High conviction (reduced from 2.0)
            short_score += 2
        elif last['volume_ratio'] >= 1.3:   # Good volume (reduced from 1.5)
            short_score += 1
        
        # 5. Rejection Confirmation (1 point) - Enhanced rejection
        if last['close'] < last['high'] and last['close'] < last['bb_upper']:
            short_score += 1
        # NEW: Alternative check for rejection when BB% >= 85%
        elif last['bb_pct'] >= 0.85 and last['close'] < last['open']:
            short_score += 1
        
        return short_score

    def _get_empty_setup(self) -> Dict[str, Any]:
        """Return empty setup structure"""
        return {
            'setup_type': 'NONE',
            'bb_score': 0,
            'setup_quality': 'None',
            'entry': 0,
            'stop': 0,
            'target1': 0,
            'risk_reward': 0
        }

    def _get_empty_setup_with_score(self, score: int) -> Dict[str, Any]:
        """Return empty setup structure with score"""
        return {
            'setup_type': 'NONE',
            'bb_score': score,
            'setup_quality': 'Poor' if score > 0 else 'None',
            'entry': 0,
            'stop': 0,
            'target1': 0,
            'risk_reward': 0
        }