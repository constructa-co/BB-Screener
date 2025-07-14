"""
Minimal Confidence Module for Main Scanner Integration
====================================================

Simple module that adds composite scoring, tier classification, 
and rationale generation to existing trade analysis.

Key Functions:
- calculate_composite_confidence(): Unified scoring
- get_confidence_tier(): Quality classification  
- generate_rationale(): Natural language explanation
- enhance_trade_with_confidence(): Main integration method
"""

import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

class MinimalConfidenceModule:
    """
    Lightweight confidence module that enhances existing trade analysis
    with composite scoring and tier classification.
    """
    
    def __init__(self):
        # Indicator performance tracking (based on your recent data)
        self.indicator_performance = {
            'mfi_oversold': {'success_rate': 89.0, 'weight': 1.0},
            'mfi_overbought': {'success_rate': 76.1, 'weight': 0.89},  # Degraded from 85%
            'stoch_oversold': {'success_rate': 77.4, 'weight': 1.0},
            'stoch_overbought': {'success_rate': 73.8, 'weight': 0.95},
            'volume_surge': {'success_rate': 75.2, 'weight': 1.0},
            'bb_squeeze': {'success_rate': 67.4, 'weight': 0.92},
            'bb_expansion': {'success_rate': 78.0, 'weight': 1.0},
            'rsi_divergence': {'success_rate': 72.8, 'weight': 0.96},
            'macd_divergence': {'success_rate': 74.1, 'weight': 0.98}
        }
        
        logger.info("Minimal Confidence Module initialized")
    
    def normalize_bb_score(self, bb_score: float, max_score: float = 34.0) -> float:
        """Normalize BB score to 0-100 scale."""
        return min(100.0, (bb_score / max_score) * 100)
    
    def calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """Calculate sentiment score from existing sentiment data."""
        
        # Extract sentiment components with defaults
        lunar_score = sentiment_data.get('lunar_galaxy_score', 50)
        tm_grade = sentiment_data.get('tm_trader_grade', 50)
        
        # Simple weighted average (can be enhanced later)
        sentiment_score = (lunar_score * 0.6 + tm_grade * 0.4)
        
        return min(100.0, sentiment_score)
    
    def get_regime_weights(self, market_regime: Dict) -> Dict[str, float]:
        """Get confidence weights based on market regime."""
        
        regime_type = market_regime.get('regime_type', 'MIXED')
        
        # Default weights
        weights = {
            'technical': 0.50,
            'historical': 0.35,
            'sentiment': 0.15
        }
        
        # Adjust based on regime
        if regime_type == 'BEAR_MARKET':
            weights['technical'] = 0.40
            weights['historical'] = 0.45  # Trust historical more in bear
            weights['sentiment'] = 0.15
        elif regime_type == 'BULL_MARKET':
            weights['technical'] = 0.55  # Technical setups work better in bull
            weights['historical'] = 0.25
            weights['sentiment'] = 0.20
        
        return weights
    
    def calculate_composite_confidence(self, trade_data: Dict, market_regime: Dict) -> Dict[str, float]:
        """Calculate composite confidence from existing trade data."""
        
        # Extract technical confidence (normalized BB score)
        bb_score = trade_data.get('bb_score', 0)
        technical_confidence = self.normalize_bb_score(bb_score)
        
        # Add confluence bonuses from your existing analysis
        confluence_bonus = 0
        if trade_data.get('volume_surge', False):
            confluence_bonus += 5
        if trade_data.get('mfi_oversold', False):
            confluence_bonus += 8  # Strong signal
        if trade_data.get('stoch_oversold', False):
            confluence_bonus += 6
        if trade_data.get('bb_expansion', False):
            confluence_bonus += 4
        
        technical_confidence = min(100.0, technical_confidence + confluence_bonus)
        
        # Extract historical confidence
        historical_win_rate = trade_data.get('historical_win_rate', trade_data.get('historical_probability', 0))
        similar_count = trade_data.get('similar_trade_count', trade_data.get('total_bounces_analyzed', 0))
        
        # Apply sample size penalty
        if similar_count < 20:
            sample_penalty = 0.8  # 20% penalty for small samples
        elif similar_count < 50:
            sample_penalty = 0.9  # 10% penalty for medium samples  
        else:
            sample_penalty = 1.0  # No penalty for large samples
        
        historical_confidence = historical_win_rate * sample_penalty
        
        # Calculate sentiment confidence
        sentiment_data = trade_data.get('sentiment_data', {})
        sentiment_confidence = self.calculate_sentiment_score(sentiment_data)
        
        # Get regime-specific weights
        weights = self.get_regime_weights(market_regime)
        
        # Calculate composite score
        composite_confidence = (
            technical_confidence * weights['technical'] +
            historical_confidence * weights['historical'] +
            sentiment_confidence * weights['sentiment']
        )
        
        return {
            'technical_confidence': round(technical_confidence, 1),
            'historical_confidence': round(historical_confidence, 1),
            'sentiment_confidence': round(sentiment_confidence, 1),
            'composite_confidence': round(composite_confidence, 1)
        }
    
    def get_confidence_tier(self, composite_confidence: float) -> Tuple[str, str]:
        """Get confidence tier label and emoji."""
        
        if composite_confidence >= 85:
            return "🔥 INSTITUTIONAL", "🔥"
        elif composite_confidence >= 70:
            return "⭐ HIGH CONFIDENCE", "⭐"
        elif composite_confidence >= 60:
            return "⚠️ WATCH-ONLY", "⚠️"
        else:
            return "❌ FILTERED OUT", "❌"
    
    def generate_rationale(self, trade_data: Dict) -> str:
        """Generate natural language rationale for confidence score."""
        
        rationale_parts = []
        
        # Check key indicators
        if trade_data.get('mfi_oversold', False):
            mfi_rate = self.indicator_performance['mfi_oversold']['success_rate']
            rationale_parts.append(f"MFI Oversold ({mfi_rate}% success)")
        
        if trade_data.get('volume_surge', False):
            volume_rate = self.indicator_performance['volume_surge']['success_rate']
            rationale_parts.append(f"Volume Surge ({volume_rate}% success)")
        
        if trade_data.get('stoch_oversold', False):
            stoch_rate = self.indicator_performance['stoch_oversold']['success_rate']
            rationale_parts.append(f"Stoch Oversold ({stoch_rate}% success)")
        
        if trade_data.get('bb_expansion', False):
            bb_rate = self.indicator_performance['bb_expansion']['success_rate']
            rationale_parts.append(f"BB Expansion ({bb_rate}% success)")
        
        # Add market context if available
        market_regime = trade_data.get('market_regime', {})
        if market_regime.get('alt_season_indicator') == 'ALT_SEASON':
            rationale_parts.append("Alt Season conditions")
        
        # Create final rationale
        if rationale_parts:
            if len(rationale_parts) >= 3:
                return f"High confidence: {' + '.join(rationale_parts[:3])}"
            else:
                return f"Moderate confidence: {' + '.join(rationale_parts)}"
        else:
            return "Standard confidence: Basic BB setup"
    
    def get_confidence_threshold(self, market_regime: Dict) -> float:
        """Get dynamic confidence threshold based on market regime."""
        
        regime_type = market_regime.get('regime_type', 'MIXED')
        
        if regime_type == 'BEAR_MARKET':
            return 75.0  # Higher bar in bear markets
        elif regime_type == 'BULL_MARKET':
            return 65.0  # Can accept lower in bull markets
        else:
            return 70.0  # Default for mixed markets
    
    def enhance_trade_with_confidence(self, trade_data: Dict, market_regime: Dict) -> Dict:
        """
        Main integration method: enhance existing trade data with confidence scores.
        
        Args:
            trade_data: Dict containing existing trade analysis
            market_regime: Dict containing market regime data
            
        Returns:
            Dict: Enhanced trade data with confidence scores added
        """
        
        try:
            # Check if trade_data is None or empty
            if trade_data is None:
                logger.warning("Received None trade_data in enhance_trade_with_confidence")
                return {}
            
            # Check if market_regime is None
            if market_regime is None:
                market_regime = {}
            # Calculate confidence scores
            confidence_scores = self.calculate_composite_confidence(trade_data, market_regime)
            
            # Get tier classification
            composite_confidence = confidence_scores['composite_confidence']
            tier_label, tier_emoji = self.get_confidence_tier(composite_confidence)
            
            # Generate rationale
            rationale = self.generate_rationale(trade_data)
            
            # Create enhanced trade data (preserve all original data)
            enhanced_trade = {
                **trade_data,  # All original fields preserved
                
                # New confidence fields
                'technical_confidence': confidence_scores['technical_confidence'],
                'historical_confidence': confidence_scores['historical_confidence'],
                'sentiment_confidence': confidence_scores['sentiment_confidence'],
                'composite_confidence': composite_confidence,
                'confidence_tier': tier_label,
                'confidence_emoji': tier_emoji,
                'confidence_rationale': rationale,
                
                # Metadata
                'confidence_timestamp': trade_data.get('timestamp', ''),
                'confidence_threshold': self.get_confidence_threshold(market_regime)
            }
            
            logger.debug(f"Enhanced {trade_data.get('symbol', 'UNKNOWN')} with {composite_confidence}% confidence")
            
            return enhanced_trade
            
        except Exception as e:
            logger.error(f"Error enhancing trade with confidence: {e}")
            
            # Return original data with error indicators
            return {
                **trade_data,
                'composite_confidence': 0.0,
                'confidence_tier': 'ERROR',
                'confidence_rationale': f'Confidence calculation error: {str(e)}'
            }
    
    def filter_trades_by_confidence(self, enhanced_trades: List[Dict], market_regime: Dict) -> Dict:
        """Filter and rank trades by confidence."""
        
        threshold = self.get_confidence_threshold(market_regime)
        
        # Filter trades meeting minimum confidence
        qualified_trades = [
            trade for trade in enhanced_trades 
            if trade.get('composite_confidence', 0) >= threshold
        ]
        
        # Sort by composite confidence (descending)
        qualified_trades.sort(key=lambda x: x.get('composite_confidence', 0), reverse=True)
        
        # Categorize by tier
        institutional = [t for t in qualified_trades if t.get('composite_confidence', 0) >= 85]
        high_confidence = [t for t in qualified_trades if 70 <= t.get('composite_confidence', 0) < 85]
        watch_only = [t for t in qualified_trades if 60 <= t.get('composite_confidence', 0) < 70]
        
        return {
            'all_qualified': qualified_trades,
            'institutional': institutional,
            'high_confidence': high_confidence,
            'watch_only': watch_only,
            'total_qualified': len(qualified_trades),
            'confidence_threshold': threshold
        }

# Integration helper functions for main scanner
def enhance_all_trades_with_confidence(trades: List[Dict], market_regime: Dict, 
                                     confidence_module: MinimalConfidenceModule = None) -> List[Dict]:
    """
    Helper function to enhance all trades with confidence scores.
    Can be called directly from main_scanner.py
    """
    
    if confidence_module is None:
        confidence_module = MinimalConfidenceModule()
    
    enhanced_trades = []
    for trade in trades:
        enhanced_trade = confidence_module.enhance_trade_with_confidence(trade, market_regime)
        enhanced_trades.append(enhanced_trade)
    
    return enhanced_trades

def get_top_confidence_trades(enhanced_trades: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Helper function to get top N trades by confidence.
    Can be called directly from main_scanner.py
    """
    
    # Sort by composite confidence
    sorted_trades = sorted(
        enhanced_trades, 
        key=lambda x: x.get('composite_confidence', 0), 
        reverse=True
    )
    
    return sorted_trades[:limit]

# Example usage
if __name__ == "__main__":
    
    # Example integration test
    confidence_module = MinimalConfidenceModule()
    
    # Example trade data (as it would come from your main scanner)
    example_trade = {
        'symbol': 'BTC/USDT',
        'bb_score': 28,  # Out of 34
        'historical_win_rate': 82.3,
        'similar_trade_count': 47,
        'volume_surge': True,
        'mfi_oversold': True,
        'stoch_oversold': False,
        'bb_expansion': True,
        'sentiment_data': {
            'lunar_galaxy_score': 72,
            'tm_trader_grade': 68
        },
        'entry_price': 67234.50,
        'stop_loss': 69456.20,
        'take_profit': 64123.80,
        'risk_reward_ratio': 2.1
    }
    
    # Example market regime
    example_regime = {
        'regime_type': 'MIXED',
        'regime_confidence': 68
    }
    
    # Test confidence enhancement
    enhanced_trade = confidence_module.enhance_trade_with_confidence(example_trade, example_regime)
    
    print("🎯 CONFIDENCE ENHANCEMENT TEST:")
    print(f"Symbol: {enhanced_trade['symbol']}")
    print(f"Technical Confidence: {enhanced_trade['technical_confidence']}%")
    print(f"Historical Confidence: {enhanced_trade['historical_confidence']}%")
    print(f"Sentiment Confidence: {enhanced_trade['sentiment_confidence']}%")
    print(f"Composite Confidence: {enhanced_trade['composite_confidence']}%")
    print(f"Tier: {enhanced_trade['confidence_tier']}")
    print(f"Rationale: {enhanced_trade['confidence_rationale']}")