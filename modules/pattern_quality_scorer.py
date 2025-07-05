"""
Pattern Quality Scorer - Objective Pattern Quality Assessment
Mathematical scoring system with ChatGPT improvements
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

class PatternQualityScorer:
    """
    Objective pattern quality assessment using mathematical criteria
    Implements ChatGPT's recommendations for reducing subjectivity
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality scoring weights (total = 100%)
        self.scoring_weights = {
            'geometric_precision': 0.40,  # 40% - Pattern geometry accuracy
            'volume_confirmation': 0.25,  # 25% - Volume validation
            'bb_proximity': 0.20,         # 20% - BB band proximity
            'market_regime': 0.15         # 15% - Market regime alignment
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 85,  # 85-100%
            'good': 70,       # 70-84%
            'fair': 60,       # 60-69%
            'poor': 0         # Below 60%
        }
        
        self.logger.info("Pattern Quality Scorer initialized with objective mathematical criteria")
    
    def calculate_quality(self, pattern_data: Dict, df: pd.DataFrame, atr_value: float) -> float:
        """
        Calculate comprehensive pattern quality score (0-100)
        
        Args:
            pattern_data: Pattern detection results
            df: OHLCV data for context
            atr_value: Average True Range for significance
            
        Returns:
            Quality score (0-100)
        """
        try:
            # Component scores
            geometric_score = self._calculate_geometric_precision(pattern_data)
            volume_score = self._calculate_volume_confirmation(pattern_data, df)
            proximity_score = self._calculate_bb_proximity(pattern_data)
            regime_score = self._calculate_market_regime_alignment(pattern_data)
            
            # Weighted final score
            final_score = (
                geometric_score * self.scoring_weights['geometric_precision'] +
                volume_score * self.scoring_weights['volume_confirmation'] +
                proximity_score * self.scoring_weights['bb_proximity'] +
                regime_score * self.scoring_weights['market_regime']
            )
            
            # Log detailed scoring breakdown
            self.logger.debug(f"Quality scoring for {pattern_data.get('pattern_type', 'unknown')}: "
                            f"Geometric: {geometric_score:.1f}, Volume: {volume_score:.1f}, "
                            f"Proximity: {proximity_score:.1f}, Regime: {regime_score:.1f}, "
                            f"Final: {final_score:.1f}")
            
            return min(final_score, 100.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern quality: {str(e)}")
            return 0.0
    
    def _calculate_geometric_precision(self, pattern_data: Dict) -> float:
        """Calculate geometric precision score (0-100)"""
        pattern_type = pattern_data.get('pattern_type', '')
        
        # Different geometric criteria for different pattern types
        if pattern_type in ['hammer', 'inverted_hammer', 'shooting_star', 'hanging_man']:
            return self._score_hammer_geometry(pattern_data)
        
        elif pattern_type in ['bullish_engulfing', 'bearish_engulfing']:
            return self._score_engulfing_geometry(pattern_data)
        
        elif pattern_type == 'doji':
            return self._score_doji_geometry(pattern_data)
        
        elif pattern_type in ['piercing_line', 'dark_cloud_cover']:
            return self._score_penetration_geometry(pattern_data)
        
        elif pattern_type in ['morning_star', 'evening_star']:
            return self._score_star_geometry(pattern_data)
        
        elif pattern_type in ['tweezer_top', 'tweezer_bottom']:
            return self._score_tweezer_geometry(pattern_data)
        
        elif pattern_type in ['bullish_harami', 'bearish_harami']:
            return self._score_harami_geometry(pattern_data)
        
        else:
            # Generic scoring for other patterns
            return self._score_generic_geometry(pattern_data)
    
    def _score_hammer_geometry(self, pattern_data: Dict) -> float:
        """Score hammer-type pattern geometry"""
        body_ratio = pattern_data.get('body_ratio', 0)
        lower_wick_ratio = pattern_data.get('lower_wick_ratio', 0)
        upper_wick_ratio = pattern_data.get('upper_wick_ratio', 0)
        
        score = 50  # Base score
        
        # Ideal body ratio: 10-25% (small but not too small)
        if 0.10 <= body_ratio <= 0.25:
            score += 25
        elif 0.05 <= body_ratio <= 0.30:
            score += 15
        
        # Ideal lower wick: 60-80% (long but not excessive)
        if 0.60 <= lower_wick_ratio <= 0.80:
            score += 20
        elif 0.50 <= lower_wick_ratio <= 0.85:
            score += 10
        
        # Upper wick should be minimal: <10%
        if upper_wick_ratio <= 0.05:
            score += 5
        elif upper_wick_ratio <= 0.10:
            score += 2
        
        return min(score, 100)
    
    def _score_engulfing_geometry(self, pattern_data: Dict) -> float:
        """Score engulfing pattern geometry"""
        engulfing_ratio = pattern_data.get('engulfing_ratio', 0)
        
        score = 50  # Base score
        
        # Ideal engulfing: 120-200% (significant but not excessive)
        if 1.2 <= engulfing_ratio <= 2.0:
            score += 40
        elif 1.1 <= engulfing_ratio <= 2.5:
            score += 25
        elif engulfing_ratio >= 1.0:
            score += 10
        
        return min(score, 100)
    
    def _score_doji_geometry(self, pattern_data: Dict) -> float:
        """Score doji pattern geometry"""
        body_ratio = pattern_data.get('body_ratio', 0)
        lower_wick_ratio = pattern_data.get('lower_wick_ratio', 0)
        upper_wick_ratio = pattern_data.get('upper_wick_ratio', 0)
        
        score = 50  # Base score
        
        # Body should be very small: <3% ideal, <5% acceptable
        if body_ratio <= 0.01:
            score += 30
        elif body_ratio <= 0.03:
            score += 20
        elif body_ratio <= 0.05:
            score += 10
        
        # Wicks should be significant (indicates indecision)
        total_wick_ratio = lower_wick_ratio + upper_wick_ratio
        if total_wick_ratio >= 0.80:
            score += 20
        elif total_wick_ratio >= 0.60:
            score += 10
        
        return min(score, 100)
    
    def _score_penetration_geometry(self, pattern_data: Dict) -> float:
        """Score piercing line / dark cloud cover geometry"""
        penetration_ratio = pattern_data.get('penetration_ratio', 0)
        
        score = 50  # Base score
        
        # Ideal penetration: 50-70% (significant reversal signal)
        if 0.50 <= penetration_ratio <= 0.70:
            score += 40
        elif 0.40 <= penetration_ratio <= 0.80:
            score += 25
        elif penetration_ratio >= 0.30:
            score += 10
        
        return min(score, 100)
    
    def _score_star_geometry(self, pattern_data: Dict) -> float:
        """Score morning/evening star geometry"""
        middle_body_ratio = pattern_data.get('middle_body_ratio', 0)
        gap_down = pattern_data.get('gap_down', False)
        gap_up = pattern_data.get('gap_up', False)
        
        score = 50  # Base score
        
        # Star candle should have small body
        if middle_body_ratio <= 0.15:
            score += 20
        elif middle_body_ratio <= 0.25:
            score += 10
        
        # Gaps add to pattern strength
        if gap_down and gap_up:
            score += 20
        elif gap_down or gap_up:
            score += 10
        
        return min(score, 100)
    
    def _score_tweezer_geometry(self, pattern_data: Dict) -> float:
        """Score tweezer pattern geometry"""
        precision = pattern_data.get('high_precision', pattern_data.get('low_precision', 0))
        
        score = 50  # Base score
        
        # High precision in level testing
        if precision >= 0.995:  # Within 0.5%
            score += 40
        elif precision >= 0.99:  # Within 1%
            score += 25
        elif precision >= 0.985:  # Within 1.5%
            score += 15
        
        return min(score, 100)
    
    def _score_harami_geometry(self, pattern_data: Dict) -> float:
        """Score harami pattern geometry"""
        size_ratio = pattern_data.get('size_ratio', 0)
        
        score = 50  # Base score
        
        # Smaller inner candle indicates stronger reversal potential
        if size_ratio <= 0.30:
            score += 30
        elif size_ratio <= 0.50:
            score += 20
        elif size_ratio <= 0.70:
            score += 10
        
        return min(score, 100)
    
    def _score_generic_geometry(self, pattern_data: Dict) -> float:
        """Generic geometric scoring for other patterns"""
        confidence = pattern_data.get('confidence', 0)
        
        # Use pattern's internal confidence as geometric score
        return min(confidence, 100)
    
    def _calculate_volume_confirmation(self, pattern_data: Dict, df: pd.DataFrame) -> float:
        """Calculate volume confirmation score (0-100)"""
        try:
            if len(df) < 10:
                return 50  # Default if insufficient data
            
            # Get current volume and average volume
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()  # 20-period average
            
            if avg_volume == 0:
                return 50  # Default if no volume data
            
            volume_multiplier = current_volume / avg_volume
            
            # Score based on volume multiplier
            if volume_multiplier >= 2.0:  # 2x+ volume
                return 100
            elif volume_multiplier >= 1.5:  # 1.5x+ volume (ChatGPT threshold)
                return 85
            elif volume_multiplier >= 1.2:  # 1.2x+ volume
                return 70
            elif volume_multiplier >= 1.0:  # Average volume
                return 55
            else:  # Below average volume
                return max(20, volume_multiplier * 50)
                
        except Exception as e:
            self.logger.warning(f"Error calculating volume confirmation: {str(e)}")
            return 50
    
    def _calculate_bb_proximity(self, pattern_data: Dict) -> float:
        """Calculate Bollinger Band proximity score (0-100)"""
        # This would need BB data passed in - for now return default
        # In full implementation, would check distance from BB bands
        
        # Placeholder scoring based on ATR multiple
        atr_multiple = pattern_data.get('atr_multiple', 0)
        
        if atr_multiple >= 1.5:  # Significant candle
            return 85
        elif atr_multiple >= 1.0:  # Average significance
            return 70
        elif atr_multiple >= 0.5:  # Some significance
            return 50
        else:
            return 30
    
    def _calculate_market_regime_alignment(self, pattern_data: Dict) -> float:
        """Calculate market regime alignment score (0-100)"""
        # This would integrate with market regime analyzer
        # For now, return neutral score
        
        pattern_direction = pattern_data.get('direction', 'neutral')
        
        # Placeholder: neutral patterns always get medium score
        # Directional patterns would be scored based on regime alignment
        if pattern_direction == 'neutral':
            return 65
        else:
            return 70  # Would be calculated based on actual regime data
    
    def get_quality_category(self, score: float) -> str:
        """Get quality category from score"""
        if score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif score >= self.quality_thresholds['good']:
            return 'good'
        elif score >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def get_quality_description(self, score: float) -> str:
        """Get human-readable quality description"""
        category = self.get_quality_category(score)
        
        descriptions = {
            'excellent': f'Excellent pattern quality ({score:.1f}%) - High confidence signal',
            'good': f'Good pattern quality ({score:.1f}%) - Reliable signal',
            'fair': f'Fair pattern quality ({score:.1f}%) - Moderate confidence',
            'poor': f'Poor pattern quality ({score:.1f}%) - Low confidence, avoid'
        }
        
        return descriptions.get(category, f'Unknown quality ({score:.1f}%)')
    
    def get_scoring_breakdown(self, pattern_data: Dict, df: pd.DataFrame, atr_value: float) -> Dict:
        """Get detailed scoring breakdown for analysis"""
        geometric_score = self._calculate_geometric_precision(pattern_data)
        volume_score = self._calculate_volume_confirmation(pattern_data, df)
        proximity_score = self._calculate_bb_proximity(pattern_data)
        regime_score = self._calculate_market_regime_alignment(pattern_data)
        
        total_score = (
            geometric_score * self.scoring_weights['geometric_precision'] +
            volume_score * self.scoring_weights['volume_confirmation'] +
            proximity_score * self.scoring_weights['bb_proximity'] +
            regime_score * self.scoring_weights['market_regime']
        )
        
        return {
            'geometric_precision': {
                'score': geometric_score,
                'weight': self.scoring_weights['geometric_precision'],
                'contribution': geometric_score * self.scoring_weights['geometric_precision']
            },
            'volume_confirmation': {
                'score': volume_score,
                'weight': self.scoring_weights['volume_confirmation'],
                'contribution': volume_score * self.scoring_weights['volume_confirmation']
            },
            'bb_proximity': {
                'score': proximity_score,
                'weight': self.scoring_weights['bb_proximity'],
                'contribution': proximity_score * self.scoring_weights['bb_proximity']
            },
            'market_regime': {
                'score': regime_score,
                'weight': self.scoring_weights['market_regime'],
                'contribution': regime_score * self.scoring_weights['market_regime']
            },
            'total_score': total_score,
            'quality_category': self.get_quality_category(total_score),
            'quality_description': self.get_quality_description(total_score)
        }