# risk_manager.py - Risk Management Module
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from config import *

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management and probability calculation"""
    
    def __init__(self):
        self.atr_stop_multiplier = BB_CONFIG["atr_stop_multiplier"]
        self.min_risk_reward = RISK_CONFIG.get("min_risk_reward", 1.0)
        self.max_risk_percent = RISK_CONFIG.get("max_risk_percent", 15.0)
        
    def calculate_adaptive_stop_loss(self, entry_price: float, atr: float, bb_upper: float, bb_lower: float, setup_type: str) -> float:
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
            else:  # SHORT
                stop_loss = entry_price + final_stop_distance
                # Ensure stop is never below entry for SHORT
                stop_loss = max(stop_loss, entry_price * 1.025)  # At least 2.5% above entry
            
            logger.debug(f"Adaptive stop calculation: ATR_dist={atr_stop_distance:.6f}, BB_dist={bb_stop_distance:.6f}, Final={final_stop_distance:.6f}")
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stop loss: {e}")
            # Fallback: simple 3% stop
            if setup_type == 'LONG':
                return entry_price * 0.97
            else:
                return entry_price * 1.03

    def calculate_comprehensive_probability(self, df: pd.DataFrame, bb_analysis: Dict[str, Any], 
                                          bull_div: Dict[str, Any], bear_div: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Calculate probability with CLEAR, SIMPLE scoring"""
        
        if df is None:
            return 0, {}
        
        last = df.iloc[-1]
        
        # Start with base probability based on BB score
        if bb_analysis['bb_score'] >= 8:
            probability = 75  # Excellent setup
        elif bb_analysis['bb_score'] >= 6:
            probability = 65  # Good setup
        elif bb_analysis['bb_score'] >= 5:
            probability = 55  # Fair setup
        else:
            probability = 45  # Weak setup
        
        # Divergence bonus - SIMPLE scoring
        if bb_analysis['setup_type'] == 'LONG' and bull_div['detected']:
            if bull_div['strength'] >= 2:  # 2+ indicators
                probability += 10  # Strong divergence bonus
            else:
                probability += 5   # Weak divergence bonus
                
        elif bb_analysis['setup_type'] == 'SHORT' and bear_div['detected']:
            if bear_div['strength'] >= 2:  # 2+ indicators
                probability += 10  # Strong divergence bonus
            else:
                probability += 5   # Weak divergence bonus
        
        # Volume bonus - SIMPLE scoring
        if last['volume_ratio'] >= 2.0:
            probability += 5  # Strong volume
        elif last['volume_ratio'] >= 1.5:
            probability += 3  # Good volume
        elif last['volume_ratio'] < 1.0:
            probability -= 5  # Weak volume penalty
        
        # Risk/Reward bonus
        if bb_analysis['risk_reward'] >= 2.0:
            probability += 3
        elif bb_analysis['risk_reward'] < 1.0:
            probability -= 5
        
        # Cap probability
        probability = min(max(probability, 30), 90)
        
        # Simple confirmations
        confirmations = {
            'bb_setup_quality': bb_analysis['setup_quality'],
            'divergence_present': bull_div['detected'] if bb_analysis['setup_type'] == 'LONG' else bear_div['detected'],
            'divergence_strength': bull_div['confidence'] if bb_analysis['setup_type'] == 'LONG' else bear_div['confidence'],
            'volume_confirmation': last['volume_ratio'] >= 1.5,
            'momentum_alignment': True,  # Simplified
            'risk_reward_acceptable': bb_analysis['risk_reward'] >= 1.5,
            'risk_acceptable': True  # Let user decide
        }
        
        return probability, confirmations

    def apply_quality_filters(self, bb_analysis: Dict[str, Any], risk_metrics: Dict[str, Any]) -> bool:
        """Apply RELAXED quality filters - only reject truly terrible setups"""
        try:
            # Only reject if extremely poor quality
            risk_pct = risk_metrics.get('risk_pct', 0)
            risk_reward = bb_analysis.get('risk_reward', 0)
            
            # RELAXED FILTERS: Only reject truly terrible setups
            if risk_pct > self.max_risk_percent:  # >15% risk
                logger.debug(f"Rejected: Risk too high {risk_pct}%")
                return False
                
            if risk_reward < 0.3:  # R:R worse than 1:3
                logger.debug(f"Rejected: R:R too poor {risk_reward}")
                return False
            
            # Let most setups through for user evaluation
            return True
            
        except Exception as e:
            logger.error(f"Error in quality filters: {e}")
            return True  # Default to accepting if error

    def calculate_risk_metrics(self, entry: float, stop: float, target: float, setup_type: str) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            if setup_type == 'LONG':
                risk_amount = entry - stop
                reward_amount = target - entry
            else:  # SHORT
                risk_amount = stop - entry
                reward_amount = entry - target
            
            # Calculate percentages
            risk_pct = (risk_amount / entry * 100) if entry > 0 else 0
            reward_pct = (reward_amount / entry * 100) if entry > 0 else 0
            
            # Risk/Reward ratio
            risk_reward_ratio = (reward_amount / risk_amount) if risk_amount > 0 else 0
            
            # Risk assessment
            if risk_pct <= 1.0:
                risk_category = "Very Low"
            elif risk_pct <= 2.5:
                risk_category = "Low"
            elif risk_pct <= 5.0:
                risk_category = "Medium"
            elif risk_pct <= 10.0:
                risk_category = "High"
            else:
                risk_category = "Very High"
            
            return {
                'risk_amount': round(risk_amount, 6),
                'reward_amount': round(reward_amount, 6),
                'risk_pct': round(risk_pct, 2),
                'reward_pct': round(reward_pct, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'risk_category': risk_category,
                'acceptable_risk': risk_pct <= 8.0  # More lenient
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'risk_amount': 0,
                'reward_amount': 0,
                'risk_pct': 0,
                'reward_pct': 0,
                'risk_reward_ratio': 0,
                'risk_category': 'Unknown',
                'acceptable_risk': False
            }

    def categorize_setup_quality(self, probability: int, risk_pct: float, risk_reward: float) -> str:
        """Categorize setup quality with risk consideration"""
        try:
            # Downgrade if excessive risk or poor R:R
            adjusted_prob = probability
            if risk_pct > 4.0:
                adjusted_prob -= 5
            if risk_reward < 1.0:
                adjusted_prob -= 10
            
            if adjusted_prob >= 75:
                return 'PREMIUM'
            elif adjusted_prob >= 70:
                return 'HIGH'
            elif adjusted_prob >= 65:
                return 'GOOD'
            elif adjusted_prob >= 60:
                return 'FAIR'
            elif adjusted_prob >= 55:
                return 'MARGINAL'
            else:
                return 'WEAK'
                
        except Exception as e:
            logger.error(f"Error categorizing setup quality: {e}")
            return 'UNKNOWN'

    def recommend_action(self, tier: str, setup_type: str, risk_pct: float) -> str:
        """Recommend trading action based on setup quality"""
        try:
            if tier in ['PREMIUM', 'HIGH'] and setup_type != 'NONE':
                return 'TAKE TRADE'
            elif tier == 'GOOD' and setup_type != 'NONE':
                return 'CONSIDER'
            elif tier == 'FAIR' and setup_type != 'NONE':
                return 'MONITOR'
            elif setup_type != 'NONE':
                return 'WATCH ONLY'
            else:
                return 'NO SETUP'
                
        except Exception as e:
            logger.error(f"Error recommending action: {e}")
            return 'UNKNOWN'