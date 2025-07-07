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
        """Analyze BB setup with COMPLETE detailed scoring breakdown"""
        if df is None or len(df) < self.min_candles:
            return self._get_empty_setup()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        recent_3_candles = df.tail(3)
        
        # Volume pre-filter - RELAXED
        if last['volume_ratio'] < 0.8:
            return self._get_empty_setup()
        
        # Initialize detailed scoring tracker
        scoring_details = {
            'breakdown': [],
            'indicator_values': {},
            'total_possible': 34,
            'tier_scores': {
                'base_bb': 0,
                'money_flow': 0, 
                'bb_specific': 0,
                'volume_momentum': 0,
                'divergence': 0
            }
        }
        
        # Calculate scores with detailed tracking
        long_score = self._calculate_long_score_detailed(recent_3_candles, last, df, scoring_details)
        short_score = self._calculate_short_score_detailed(recent_3_candles, last, df, scoring_details)
        
        # Determine setup
        if long_score > short_score and long_score >= 12:
            setup_type = 'LONG'
            bb_score = long_score
            entry = last['close']
            stop = self._calculate_adaptive_stop_loss(entry, last['atr'], last['bb_upper'], last['bb_lower'], 'LONG', df)
            target1 = last['bb_middle']
            
        elif short_score > long_score and short_score >= 12:
            setup_type = 'SHORT'
            bb_score = short_score
            entry = last['close']
            stop = self._calculate_adaptive_stop_loss(entry, last['atr'], last['bb_upper'], last['bb_lower'], 'SHORT', df)
            target1 = last['bb_middle']
            
        else:
            return self._get_empty_setup_with_score(max(long_score, short_score))
        
        # Enhanced quality assessment
        if bb_score >= 25:
            setup_quality = 'Exceptional'
        elif bb_score >= 22:
            setup_quality = 'Excellent'
        elif bb_score >= 18:
            setup_quality = 'Very Good'
        elif bb_score >= 15:
            setup_quality = 'Good'
        elif bb_score >= 12:
            setup_quality = 'Fair'
        else:
            setup_quality = 'Poor'
        
        # Risk/Reward calculation
        if setup_type == 'LONG':
            risk_reward = (target1 - entry) / (entry - stop) if (entry - stop) > 0 else 0
        else:
            risk_reward = (entry - target1) / (stop - entry) if (stop - entry) > 0 else 0
        
        return {
            'setup_type': setup_type,
            'bb_score': bb_score,
            'setup_quality': setup_quality,
            'entry': entry,
            'stop': stop,
            'target1': target1,
            'risk_reward': risk_reward,
            'scoring_details': scoring_details  # NEW: Detailed breakdown
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

    def _calculate_long_score(self, recent_3_candles: pd.DataFrame, last: pd.Series, df: pd.DataFrame) -> int:
        """Calculate LONG setup score - COMPLETE with ALL validated signals"""
        long_score = 0
        
        # === TIER 1: ORIGINAL BB SCORING (Base Foundation) ===
        
        # 1. BB Touch (3 points) - Must actually touch lower band
        if any(recent_3_candles['low'] <= recent_3_candles['bb_lower']):
            long_score += 3
        
        # 2. BB Position (2 points) - BALANCED thresholds
        if last['bb_pct'] <= 0.05:      # Extremely extreme
            long_score += 2
        elif last['bb_pct'] <= 0.08:    # Very extreme
            long_score += 1
        
        # 3. RSI (2 points) - BALANCED oversold levels
        if last['rsi'] <= 28:        # Extremely oversold
            long_score += 2
        elif last['rsi'] <= 38:      # Oversold
            long_score += 1
        
        # 4. Volume (2 points) - RELAXED requirements
        if last['volume_ratio'] >= 1.8:     # High conviction
            long_score += 2
        elif last['volume_ratio'] >= 1.3:   # Good volume
            long_score += 1
        
        # 5. Bounce Confirmation (1 point) - Basic bounce
        if last['close'] > last['low'] and last['close'] > last['bb_lower']:
            long_score += 1
        
        # === TIER 2: MONEY FLOW INDICATORS (Highest Success Rates) ===
        
        # 6. Money Flow Index (4 points) - 88.0% SUCCESS RATE (HIGHEST PRIORITY)
        mfi = self._calculate_money_flow_index(df)
        if mfi <= 20:  # MFI Oversold
            long_score += 4  # Maximum weight for best signal
        elif mfi <= 30:  # MFI Very Oversold
            long_score += 2
        
        # 7. Chaikin Money Flow (3 points) - 74-75% SUCCESS RATE
        cmf = self._calculate_chaikin_money_flow(df)
        if cmf < -0.1:  # Strong selling pressure (contrarian signal)
            long_score += 3
        elif cmf < -0.05:  # Moderate selling pressure
            long_score += 2
        elif abs(cmf) < 0.05:  # Neutral money flow
            long_score += 1
        
        # === TIER 3: BB-SPECIFIC INDICATORS (High Success Rates) ===
        
        # 8. BB Expansion (2 points) - 79.7% SUCCESS RATE
        bb_expansion = self._calculate_bb_expansion(df)
        if bb_expansion > 1.2:  # High expansion (volatility increasing)
            long_score += 2
        elif bb_expansion > 1.1:  # Moderate expansion
            long_score += 1
        
        # 9. BB Reversal Setup (2 points) - 76.8% SUCCESS RATE
        if self._calculate_bb_reversal_setup(df):
            long_score += 2
        
        # 10. BB Trend Analysis (2 points) - 68-80% SUCCESS RATE
        bb_trend = self._calculate_bb_trend(df)
        if bb_trend == "Downtrend":      # 80.8% success - best for LONG bounces
            long_score += 2
        elif bb_trend == "Uptrend":      # 77.9% success - good for LONG bounces
            long_score += 1
        # Sideways = 68.6% success - no bonus
        
        # 11. BB Squeeze (2 points) - 67.5% SUCCESS RATE
        if self._calculate_bb_squeeze(df):
            long_score += 2  # Squeeze often precedes strong moves
        
        # === TIER 4: VOLUME & MOMENTUM INDICATORS (Medium-High Success) ===
        
        # 12. Volume Surge (2 points) - 76.9% SUCCESS RATE
        if self._calculate_volume_surge(df):
            long_score += 2
        
        # 13. Stochastic Oversold (2 points) - 75.7% SUCCESS RATE  
        if self._calculate_stoch_oversold(df):
            long_score += 2
        
        # 14. CCI Extreme (2 points) - 73.4% SUCCESS RATE
        cci_status = self._calculate_cci_extreme(df)
        if cci_status == "Oversold":
            long_score += 2
        
        # === TIER 5: DIVERGENCE INDICATORS (Medium Success) ===
        
        # 15. MACD Divergence (2 points) - 73.2% SUCCESS RATE
        if self._calculate_macd_divergence(df):
            long_score += 2
        
        # 16. RSI Divergence (1 point) - 72.1% SUCCESS RATE
        if self._calculate_rsi_divergence(df):
            long_score += 1
        
        return long_score

    def _calculate_short_score_detailed(self, recent_3_candles: pd.DataFrame, last: pd.Series, 
                                      df: pd.DataFrame, scoring_details: Dict) -> int:
        """Calculate SHORT score with detailed breakdown tracking"""
        short_score = 0
        
        # === TIER 1: BASE BB SCORING ===
        base_score = 0
        
        # 1. BB Touch (3 points)
        if any(recent_3_candles['high'] >= recent_3_candles['bb_upper']):
            points = 3
            base_score += points
            scoring_details['breakdown'].append(f"BB Touch Upper Band: +{points} pts")
        elif any(recent_3_candles['high'] >= recent_3_candles['bb_upper'] * 0.998):
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"BB Near Upper Band: +{points} pts")
        
        # 2. BB Position (2 points)
        if last['bb_pct'] >= 0.92:
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"BB Position Extreme ({last['bb_pct']:.3f}): +{points} pts")
        elif last['bb_pct'] >= 0.88:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"BB Position Very High ({last['bb_pct']:.3f}): +{points} pt")
        
        # 3. RSI (2 points)
        if last['rsi'] >= 68:
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"RSI Extreme Overbought ({last['rsi']:.1f}): +{points} pts")
        elif last['rsi'] >= 58:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"RSI Overbought ({last['rsi']:.1f}): +{points} pt")
        
        # 4. Volume (2 points)
        if last['volume_ratio'] >= 1.8:
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"High Volume ({last['volume_ratio']:.1f}x): +{points} pts")
        elif last['volume_ratio'] >= 1.3:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"Good Volume ({last['volume_ratio']:.1f}x): +{points} pt")
        
        # 5. Rejection Confirmation (1 point)
        if last['close'] < last['high'] and last['close'] < last['bb_upper']:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"Rejection Confirmation: +{points} pt")
        elif last['bb_pct'] >= 0.85 and last['close'] < last['open']:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"Rejection Alternative: +{points} pt")
        
        scoring_details['tier_scores']['base_bb'] = base_score
        short_score += base_score
        
        # === TIER 2: MONEY FLOW INDICATORS ===
        money_flow_score = 0
        
        # 6. Money Flow Index (4 points) - 88.1% SUCCESS
        mfi = self._calculate_money_flow_index(df)
        scoring_details['indicator_values']['mfi'] = mfi
        if mfi >= 80:
            points = 4
            money_flow_score += points
            scoring_details['breakdown'].append(f"MFI Overbought ({mfi:.1f}): +{points} pts ⭐")
        elif mfi >= 70:
            points = 2
            money_flow_score += points
            scoring_details['breakdown'].append(f"MFI Very Overbought ({mfi:.1f}): +{points} pts")
        
        # 7. Chaikin Money Flow (3 points) - 75% SUCCESS
        cmf = self._calculate_chaikin_money_flow(df)
        scoring_details['indicator_values']['cmf'] = cmf
        if cmf > 0.1:
            points = 3
            money_flow_score += points
            scoring_details['breakdown'].append(f"CMF Strong Buying ({cmf:.3f}): +{points} pts")
        elif cmf > 0.05:
            points = 2
            money_flow_score += points
            scoring_details['breakdown'].append(f"CMF Moderate Buying ({cmf:.3f}): +{points} pts")
        elif abs(cmf) < 0.05:
            points = 1
            money_flow_score += points
            scoring_details['breakdown'].append(f"CMF Neutral ({cmf:.3f}): +{points} pt")
        
        scoring_details['tier_scores']['money_flow'] = money_flow_score
        short_score += money_flow_score
        
        # === TIER 3: BB-SPECIFIC INDICATORS ===
        bb_specific_score = 0
        
        # 8. BB Expansion (2 points) - 79.7% SUCCESS
        bb_expansion = self._calculate_bb_expansion(df)
        scoring_details['indicator_values']['bb_expansion'] = bb_expansion
        if bb_expansion > 1.2:
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB High Expansion ({bb_expansion:.2f}x): +{points} pts")
        elif bb_expansion > 1.1:
            points = 1
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Moderate Expansion ({bb_expansion:.2f}x): +{points} pt")
        
        # 9. BB Reversal Setup (2 points) - 76.8% SUCCESS
        bb_reversal = self._calculate_bb_reversal_setup(df)
        if bb_reversal:
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Reversal Setup Detected: +{points} pts")
        
        # 10. BB Trend (2 points) - 68-80% SUCCESS
        bb_trend = self._calculate_bb_trend(df)
        scoring_details['indicator_values']['bb_trend'] = bb_trend
        if bb_trend == "Uptrend":
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Trend Uptrend: +{points} pts")
        elif bb_trend == "Downtrend":
            points = 1
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Trend Downtrend: +{points} pt")
        
        # 11. BB Squeeze (2 points) - 67.5% SUCCESS
        bb_squeeze = self._calculate_bb_squeeze(df)
        if bb_squeeze:
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Squeeze Detected: +{points} pts")
        
        scoring_details['tier_scores']['bb_specific'] = bb_specific_score
        short_score += bb_specific_score
        
        # === TIER 4: VOLUME & MOMENTUM ===
        volume_momentum_score = 0
        
        # 12. Volume Surge (2 points) - 76.9% SUCCESS
        volume_surge = self._calculate_volume_surge(df)
        scoring_details['indicator_values']['volume_surge'] = volume_surge
        if volume_surge:
            points = 2
            volume_momentum_score += points
            scoring_details['breakdown'].append(f"Volume Surge Detected: +{points} pts")
        
        # 13. Stochastic Overbought (2 points) - 73.6% SUCCESS
        stoch_overbought = self._calculate_stoch_overbought(df)
        if stoch_overbought:
            points = 2
            volume_momentum_score += points
            scoring_details['breakdown'].append(f"Stochastic Overbought: +{points} pts")
        
        # 14. CCI Extreme (2 points) - 73.4% SUCCESS
        cci_status = self._calculate_cci_extreme(df)
        scoring_details['indicator_values']['cci'] = cci_status
        if cci_status == "Overbought":
            points = 2
            volume_momentum_score += points
            scoring_details['breakdown'].append(f"CCI Overbought: +{points} pts")
        
        scoring_details['tier_scores']['volume_momentum'] = volume_momentum_score
        short_score += volume_momentum_score
        
        # === TIER 5: DIVERGENCE INDICATORS ===
        divergence_score = 0
        
        # 15. MACD Divergence (2 points) - 73.2% SUCCESS
        macd_div = self._calculate_macd_divergence(df)
        if macd_div:
            points = 2
            divergence_score += points
            scoring_details['breakdown'].append(f"MACD Divergence: +{points} pts")
        
        # 16. RSI Divergence (1 point) - 72.1% SUCCESS
        rsi_div = self._calculate_rsi_divergence(df)
        if rsi_div:
            points = 1
            divergence_score += points
            scoring_details['breakdown'].append(f"RSI Divergence: +{points} pt")
        
        scoring_details['tier_scores']['divergence'] = divergence_score
        short_score += divergence_score
        
        return short_score

    def _calculate_short_score(self, recent_3_candles: pd.DataFrame, last: pd.Series, df: pd.DataFrame) -> int:
        """Calculate SHORT setup score - COMPLETE with ALL validated signals"""
        short_score = 0
        
        # === TIER 1: ORIGINAL BB SCORING (Base Foundation) ===
        
        # 1. BB Touch (3 points) - Must actually touch upper band
        if any(recent_3_candles['high'] >= recent_3_candles['bb_upper']):
            short_score += 3
        elif any(recent_3_candles['high'] >= recent_3_candles['bb_upper'] * 0.998):
            short_score += 2
        
        # 2. BB Position (2 points) - ENHANCED thresholds
        if last['bb_pct'] >= 0.92:     # Very extreme
            short_score += 2
        elif last['bb_pct'] >= 0.88:   # Extreme
            short_score += 1
        
        # 3. RSI (2 points) - ENHANCED overbought levels
        if last['rsi'] >= 68:        # Overbought
            short_score += 2
        elif last['rsi'] >= 58:      # Moderately overbought
            short_score += 1
        
        # 4. Volume (2 points) - RELAXED requirements
        if last['volume_ratio'] >= 1.8:     # High conviction
            short_score += 2
        elif last['volume_ratio'] >= 1.3:   # Good volume
            short_score += 1
        
        # 5. Rejection Confirmation (1 point) - Enhanced rejection
        if last['close'] < last['high'] and last['close'] < last['bb_upper']:
            short_score += 1
        elif last['bb_pct'] >= 0.85 and last['close'] < last['open']:
            short_score += 1
        
        # === TIER 2: MONEY FLOW INDICATORS (Highest Success Rates) ===
        
        # 6. Money Flow Index (4 points) - 88.0% SUCCESS RATE (HIGHEST PRIORITY)
        mfi = self._calculate_money_flow_index(df)
        if mfi >= 80:  # MFI Overbought
            short_score += 4  # Maximum weight for best signal
        elif mfi >= 70:  # MFI Very Overbought
            short_score += 2
        
        # 7. Chaikin Money Flow (3 points) - 74-75% SUCCESS RATE
        cmf = self._calculate_chaikin_money_flow(df)
        if cmf > 0.1:  # Strong buying pressure (contrarian signal)
            short_score += 3
        elif cmf > 0.05:  # Moderate buying pressure
            short_score += 2
        elif abs(cmf) < 0.05:  # Neutral money flow
            short_score += 1
        
        # === TIER 3: BB-SPECIFIC INDICATORS (High Success Rates) ===
        
        # 8. BB Expansion (2 points) - 79.7% SUCCESS RATE
        bb_expansion = self._calculate_bb_expansion(df)
        if bb_expansion > 1.2:  # High expansion (volatility increasing)
            short_score += 2
        elif bb_expansion > 1.1:  # Moderate expansion
            short_score += 1
        
        # 9. BB Reversal Setup (2 points) - 76.8% SUCCESS RATE
        if self._calculate_bb_reversal_setup(df):
            short_score += 2
        
        # 10. BB Trend Analysis (2 points) - 68-80% SUCCESS RATE
        bb_trend = self._calculate_bb_trend(df)
        if bb_trend == "Uptrend":        # 77.9% success - best for SHORT bounces
            short_score += 2
        elif bb_trend == "Downtrend":    # 80.8% success - good for SHORT bounces  
            short_score += 1
        # Sideways = 68.6% success - no bonus
        
        # 11. BB Squeeze (2 points) - 67.5% SUCCESS RATE
        if self._calculate_bb_squeeze(df):
            short_score += 2  # Squeeze often precedes strong moves
        
        # === TIER 4: VOLUME & MOMENTUM INDICATORS (Medium-High Success) ===
        
        # 12. Volume Surge (2 points) - 76.9% SUCCESS RATE
        if self._calculate_volume_surge(df):
            short_score += 2
        
        # 13. Stochastic Overbought (2 points) - 73.6% SUCCESS RATE
        if self._calculate_stoch_overbought(df):
            short_score += 2
        
        # 14. CCI Extreme (2 points) - 73.4% SUCCESS RATE
        cci_status = self._calculate_cci_extreme(df)
        if cci_status == "Overbought":
            short_score += 2
        
        # === TIER 5: DIVERGENCE INDICATORS (Medium Success) ===
        
        # 15. MACD Divergence (2 points) - 73.2% SUCCESS RATE
        if self._calculate_macd_divergence(df):
            short_score += 2
        
        # 16. RSI Divergence (1 point) - 72.1% SUCCESS RATE
        if self._calculate_rsi_divergence(df):
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

    def _calculate_money_flow_index(self, df: pd.DataFrame) -> float:
        """Calculate Money Flow Index (MFI) - your #1 validated signal (88.1% success)"""
        try:
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate money flow
            money_flow = typical_price * df['volume']
            
            # Calculate positive and negative money flow
            positive_flow = money_flow.where(typical_price.diff() > 0, 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price.diff() < 0, 0).rolling(14).sum()
            
            # Avoid division by zero
            negative_flow = negative_flow.replace(0, 0.01)
            
            # Calculate MFI
            mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
            
            # Return the most recent MFI value
            return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
            
        except Exception as e:
            logger.debug(f"MFI calculation error: {e}")
            return 50.0  # Return neutral MFI on error

    def _calculate_chaikin_money_flow(self, df: pd.DataFrame) -> float:
        """Calculate Chaikin Money Flow (CMF) - institutional money tracking (75% success)"""
        try:
            # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            
            # Handle division by zero (when high == low)
            mf_multiplier = mf_multiplier.fillna(0)
            
            # Money Flow Volume = MF Multiplier × Volume
            mf_volume = mf_multiplier * df['volume']
            
            # 20-period CMF = Sum(MF Volume) / Sum(Volume)
            cmf = mf_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
            
            return float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.debug(f"CMF calculation error: {e}")
            return 0.0  # Neutral CMF

    def _calculate_bb_expansion(self, df: pd.DataFrame) -> float:
        """Calculate BB Expansion indicator (79.9% success rate)"""
        try:
            # BB Width = (Upper Band - Lower Band) / Middle Band
            bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Compare current width to 20-period average
            avg_width = bb_width.rolling(20).mean()
            current_width = bb_width.iloc[-1]
            avg_width_current = avg_width.iloc[-1]
            
            # Expansion ratio (current width vs average)
            if avg_width_current > 0:
                expansion_ratio = current_width / avg_width_current
            else:
                expansion_ratio = 1.0
                
            return float(expansion_ratio)
            
        except Exception as e:
            logger.debug(f"BB Expansion calculation error: {e}")
            return 1.0  # Neutral expansion

    def _calculate_bb_squeeze(self, df: pd.DataFrame) -> bool:
        """Detect BB Squeeze condition (67.6% success rate)"""
        try:
            # BB Width calculation
            bb_width = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Squeeze condition: current width < 80% of 20-period average
            avg_width = bb_width.rolling(20).mean().iloc[-1]
            current_width = bb_width.iloc[-1]
            
            # Return True if in squeeze (low volatility)
            return current_width < (avg_width * 0.8)
            
        except Exception as e:
            logger.debug(f"BB Squeeze calculation error: {e}")
            return False

    def _calculate_bb_reversal_setup(self, df: pd.DataFrame) -> bool:
        """Calculate BB Reversal Setup indicator (76.8% success rate)"""
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Reversal setup conditions
            # For LONG: Price near lower band + opposite momentum building
            long_reversal = (
                last['bb_pct'] <= 0.15 and  # Near lower band
                last['close'] > prev['close'] and  # Starting to reverse up
                last['volume'] > df['volume'].rolling(10).mean().iloc[-1]  # Volume confirmation
            )
            
            # For SHORT: Price near upper band + opposite momentum building  
            short_reversal = (
                last['bb_pct'] >= 0.85 and  # Near upper band
                last['close'] < prev['close'] and  # Starting to reverse down
                last['volume'] > df['volume'].rolling(10).mean().iloc[-1]  # Volume confirmation
            )
            
            return long_reversal or short_reversal
            
        except Exception as e:
            logger.debug(f"BB Reversal Setup calculation error: {e}")
            return False

    def _calculate_bb_trend(self, df: pd.DataFrame) -> str:
        """Calculate BB Trend direction (68-80% success rates by trend)"""
        try:
            # Use middle band (20-period SMA) for trend direction
            middle_band = df['bb_middle']
            
            # Compare current vs 5 periods ago for trend
            current = middle_band.iloc[-1]
            past = middle_band.iloc[-6] if len(middle_band) >= 6 else middle_band.iloc[0]
            
            change_pct = (current - past) / past if past > 0 else 0
            
            if change_pct > 0.02:  # 2% increase
                return "Uptrend"    # 77.9% success rate
            elif change_pct < -0.02:  # 2% decrease  
                return "Downtrend"  # 80.8% success rate
            else:
                return "Sideways"   # 68.6% success rate
                
        except Exception as e:
            logger.debug(f"BB Trend calculation error: {e}")
            return "Sideways"

    def _calculate_volume_surge(self, df: pd.DataFrame) -> bool:
        """Calculate Volume Surge indicator (76.9% success rate)"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Volume surge: current volume > 1.5x average
            return current_volume > (avg_volume * 1.5)
            
        except Exception as e:
            logger.debug(f"Volume Surge calculation error: {e}")
            return False

    def _calculate_stoch_oversold(self, df: pd.DataFrame) -> bool:
        """Calculate Stochastic Oversold (75.7% success rate)"""
        try:
            # Calculate Stochastic %K
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            
            # %D is 3-period SMA of %K
            d_percent = k_percent.rolling(3).mean()
            
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            
            # Oversold condition: both %K and %D < 20
            return current_k < 20 and current_d < 20
            
        except Exception as e:
            logger.debug(f"Stoch Oversold calculation error: {e}")
            return False

    def _calculate_stoch_overbought(self, df: pd.DataFrame) -> bool:
        """Calculate Stochastic Overbought (73.6% success rate)"""
        try:
            # Calculate Stochastic %K
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            
            # %D is 3-period SMA of %K
            d_percent = k_percent.rolling(3).mean()
            
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            
            # Overbought condition: both %K and %D > 80
            return current_k > 80 and current_d > 80
            
        except Exception as e:
            logger.debug(f"Stoch Overbought calculation error: {e}")
            return False

    def _calculate_cci_extreme(self, df: pd.DataFrame) -> str:
        """Calculate CCI Extreme conditions (73.4% success rate)"""
        try:
            # Calculate Commodity Channel Index
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma_tp) / (0.015 * mad)
            current_cci = cci.iloc[-1]
            
            if current_cci > 100:
                return "Overbought"
            elif current_cci < -100:
                return "Oversold"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.debug(f"CCI calculation error: {e}")
            return "Neutral"

    def _calculate_macd_divergence(self, df: pd.DataFrame) -> bool:
        """Calculate MACD Divergence (73.2% success rate)"""
        try:
            # Calculate MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            
            # Look for divergence in last 5 periods
            price_recent = df['close'].tail(5)
            macd_recent = macd_line.tail(5)
            
            # Bullish divergence: price making lower lows, MACD making higher lows
            price_trend = price_recent.iloc[-1] < price_recent.iloc[0]
            macd_trend = macd_recent.iloc[-1] > macd_recent.iloc[0]
            
            bullish_divergence = price_trend and macd_trend
            
            # Bearish divergence: price making higher highs, MACD making lower highs
            price_trend_up = price_recent.iloc[-1] > price_recent.iloc[0]
            macd_trend_down = macd_recent.iloc[-1] < macd_recent.iloc[0]
            
            bearish_divergence = price_trend_up and macd_trend_down
            
            return bullish_divergence or bearish_divergence
            
        except Exception as e:
            logger.debug(f"MACD Divergence calculation error: {e}")
            return False

    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> bool:
        """Calculate RSI Divergence (72.1% success rate)"""
        try:
            # RSI should already be in df, but calculate if needed
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = df['rsi']
                
            # Look for divergence in last 5 periods
            price_recent = df['close'].tail(5)
            rsi_recent = rsi.tail(5)
            
            # Bullish divergence: price making lower lows, RSI making higher lows
            price_trend = price_recent.iloc[-1] < price_recent.iloc[0]
            rsi_trend = rsi_recent.iloc[-1] > rsi_recent.iloc[0]
            
            bullish_divergence = price_trend and rsi_trend
            
            # Bearish divergence: price making higher highs, RSI making lower highs
            price_trend_up = price_recent.iloc[-1] > price_recent.iloc[0]
            rsi_trend_down = rsi_recent.iloc[-1] < rsi_recent.iloc[0]
            
            bearish_divergence = price_trend_up and rsi_trend_down
            
            return bullish_divergence or bearish_divergence
            
        except Exception as e:
            logger.debug(f"RSI Divergence calculation error: {e}")
            return False

    def _calculate_long_score_detailed(self, recent_3_candles: pd.DataFrame, last: pd.Series, 
                                     df: pd.DataFrame, scoring_details: Dict) -> int:
        """Calculate LONG score with detailed breakdown tracking"""
        long_score = 0
        
        # === TIER 1: BASE BB SCORING ===
        base_score = 0
        
        # 1. BB Touch (3 points)
        if any(recent_3_candles['low'] <= recent_3_candles['bb_lower']):
            points = 3
            base_score += points
            scoring_details['breakdown'].append(f"BB Touch Lower Band: +{points} pts")
        
        # 2. BB Position (2 points)
        if last['bb_pct'] <= 0.05:
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"BB Position Extreme ({last['bb_pct']:.3f}): +{points} pts")
        elif last['bb_pct'] <= 0.08:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"BB Position Very Low ({last['bb_pct']:.3f}): +{points} pt")
        
        # 3. RSI (2 points)
        if last['rsi'] <= 28:
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"RSI Extreme Oversold ({last['rsi']:.1f}): +{points} pts")
        elif last['rsi'] <= 38:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"RSI Oversold ({last['rsi']:.1f}): +{points} pt")
        
        # 4. Volume (2 points)
        if last['volume_ratio'] >= 1.8:
            points = 2
            base_score += points
            scoring_details['breakdown'].append(f"High Volume ({last['volume_ratio']:.1f}x): +{points} pts")
        elif last['volume_ratio'] >= 1.3:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"Good Volume ({last['volume_ratio']:.1f}x): +{points} pt")
        
        # 5. Bounce Confirmation (1 point)
        if last['close'] > last['low'] and last['close'] > last['bb_lower']:
            points = 1
            base_score += points
            scoring_details['breakdown'].append(f"Bounce Confirmation: +{points} pt")
        
        scoring_details['tier_scores']['base_bb'] = base_score
        long_score += base_score
        
        # === TIER 2: MONEY FLOW INDICATORS ===
        money_flow_score = 0
        
        # 6. Money Flow Index (4 points) - 88.1% SUCCESS
        mfi = self._calculate_money_flow_index(df)
        scoring_details['indicator_values']['mfi'] = mfi
        if mfi <= 20:
            points = 4
            money_flow_score += points
            scoring_details['breakdown'].append(f"MFI Oversold ({mfi:.1f}): +{points} pts ⭐")
        elif mfi <= 30:
            points = 2
            money_flow_score += points
            scoring_details['breakdown'].append(f"MFI Very Oversold ({mfi:.1f}): +{points} pts")
        
        # 7. Chaikin Money Flow (3 points) - 75% SUCCESS
        cmf = self._calculate_chaikin_money_flow(df)
        scoring_details['indicator_values']['cmf'] = cmf
        if cmf < -0.1:
            points = 3
            money_flow_score += points
            scoring_details['breakdown'].append(f"CMF Strong Selling ({cmf:.3f}): +{points} pts")
        elif cmf < -0.05:
            points = 2
            money_flow_score += points
            scoring_details['breakdown'].append(f"CMF Moderate Selling ({cmf:.3f}): +{points} pts")
        elif abs(cmf) < 0.05:
            points = 1
            money_flow_score += points
            scoring_details['breakdown'].append(f"CMF Neutral ({cmf:.3f}): +{points} pt")
        
        scoring_details['tier_scores']['money_flow'] = money_flow_score
        long_score += money_flow_score
        
        # === TIER 3: BB-SPECIFIC INDICATORS ===
        bb_specific_score = 0
        
        # 8. BB Expansion (2 points) - 79.7% SUCCESS
        bb_expansion = self._calculate_bb_expansion(df)
        scoring_details['indicator_values']['bb_expansion'] = bb_expansion
        if bb_expansion > 1.2:
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB High Expansion ({bb_expansion:.2f}x): +{points} pts")
        elif bb_expansion > 1.1:
            points = 1
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Moderate Expansion ({bb_expansion:.2f}x): +{points} pt")
        
        # 9. BB Reversal Setup (2 points) - 76.8% SUCCESS
        bb_reversal = self._calculate_bb_reversal_setup(df)
        if bb_reversal:
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Reversal Setup Detected: +{points} pts")
        
        # 10. BB Trend (2 points) - 68-80% SUCCESS
        bb_trend = self._calculate_bb_trend(df)
        scoring_details['indicator_values']['bb_trend'] = bb_trend
        if bb_trend == "Downtrend":
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Trend Downtrend: +{points} pts")
        elif bb_trend == "Uptrend":
            points = 1
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Trend Uptrend: +{points} pt")
        
        # 11. BB Squeeze (2 points) - 67.5% SUCCESS
        bb_squeeze = self._calculate_bb_squeeze(df)
        if bb_squeeze:
            points = 2
            bb_specific_score += points
            scoring_details['breakdown'].append(f"BB Squeeze Detected: +{points} pts")
        
        scoring_details['tier_scores']['bb_specific'] = bb_specific_score
        long_score += bb_specific_score
        
        # === TIER 4: VOLUME & MOMENTUM ===
        volume_momentum_score = 0
        
        # 12. Volume Surge (2 points) - 76.9% SUCCESS
        volume_surge = self._calculate_volume_surge(df)
        scoring_details['indicator_values']['volume_surge'] = volume_surge
        if volume_surge:
            points = 2
            volume_momentum_score += points
            scoring_details['breakdown'].append(f"Volume Surge Detected: +{points} pts")
        
        # 13. Stochastic Oversold (2 points) - 75.7% SUCCESS
        stoch_oversold = self._calculate_stoch_oversold(df)
        if stoch_oversold:
            points = 2
            volume_momentum_score += points
            scoring_details['breakdown'].append(f"Stochastic Oversold: +{points} pts")
        
        # 14. CCI Extreme (2 points) - 73.4% SUCCESS
        cci_status = self._calculate_cci_extreme(df)
        scoring_details['indicator_values']['cci'] = cci_status
        if cci_status == "Oversold":
            points = 2
            volume_momentum_score += points
            scoring_details['breakdown'].append(f"CCI Oversold: +{points} pts")
        
        scoring_details['tier_scores']['volume_momentum'] = volume_momentum_score
        long_score += volume_momentum_score
        
        # === TIER 5: DIVERGENCE INDICATORS ===
        divergence_score = 0
        
        # 15. MACD Divergence (2 points) - 73.2% SUCCESS
        macd_div = self._calculate_macd_divergence(df)
        if macd_div:
            points = 2
            divergence_score += points
            scoring_details['breakdown'].append(f"MACD Divergence: +{points} pts")
        
        # 16. RSI Divergence (1 point) - 72.1% SUCCESS
        rsi_div = self._calculate_rsi_divergence(df)
        if rsi_div:
            points = 1
            divergence_score += points
            scoring_details['breakdown'].append(f"RSI Divergence: +{points} pt")
        
        scoring_details['tier_scores']['divergence'] = divergence_score
        long_score += divergence_score
        
        return long_score