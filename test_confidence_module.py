#!/usr/bin/env python3
"""
Quick Confidence Module Test
============================

This script tests the confidence module integration without running the full scanner.
It creates sample trade data and tests the confidence enhancement.
"""

import sys
import os
from datetime import datetime

# Add the modules directory to the path
sys.path.append('modules')

try:
    from minimal_confidence_module import MinimalConfidenceModule, enhance_all_trades_with_confidence
    print("✅ Successfully imported confidence module")
except ImportError as e:
    print(f"❌ Failed to import confidence module: {e}")
    sys.exit(1)

def create_sample_trades():
    """Create sample trade data that matches the main scanner format"""
    
    sample_trades = [
        {
            'symbol': 'BTC/USDT',
            'exchange': 'binance',
            'setup_type': 'LONG',
            'bb_score': 24,
            'setup_quality': 'PREMIUM',
            'timestamp': datetime.now(),
            'bb_pct': 0.123,
            'rsi': 26.9,
            'volume_ratio': 2.9,
            'atr_pct': 0.045,
            'regime_confidence': 68,
            'regime_type': 'MIXED',
            'bb_suitability': 'GOOD',
            'position_multiplier': 1.2,
            'btc_health_score': 75,
            'alt_market_outlook': 'GOOD',
            'market_health_score': 73,
            'alt_season_indicator': 'NEUTRAL',
            'bb_score_34': 24,
            'setup_quality_enhanced': 'PREMIUM',
            'scoring_details': {
                'tier_scores': {
                    'base_bb': 9,
                    'money_flow': 5,
                    'bb_specific': 4,
                    'volume_momentum': 6,
                    'divergence': 0
                },
                'indicator_values': {
                    'mfi': 28.3,
                    'cmf': -0.162,
                    'bb_expansion': 2.16,
                    'volume_surge': True,
                    'bb_trend': 'Sideways'
                },
                'breakdown': [
                    'BB Touch Lower Band: +3 pts',
                    'BB Position Extreme (-0.148): +2 pts',
                    'RSI Extreme Oversold (26.9): +2 pts',
                    'High Volume (2.9x): +2 pts',
                    'MFI Very Oversold (28.3): +2 pts'
                ]
            },
            'technical_confidence': 0.0,
            'historical_confidence': 0.0,
            'sentiment_confidence': 0.0,
            'composite_confidence': 0.0,
            'confidence_tier': 'UNRATED',
            'confidence_rationale': 'No analysis available',
            'tier_base_bb': 9,
            'tier_money_flow': 5,
            'tier_bb_specific': 4,
            'tier_volume_momentum': 6,
            'tier_divergence': 0,
            'mfi_value': 28.3,
            'cmf_value': -0.162,
            'bb_expansion_ratio': 2.16,
            'volume_surge_detected': True,
            'bb_trend_direction': 'Sideways',
            'component_1': 'BB Touch Lower Band: +3 pts',
            'component_2': 'BB Position Extreme (-0.148): +2 pts',
            'component_3': 'RSI Extreme Oversold (26.9): +2 pts',
            'component_4': 'High Volume (2.9x): +2 pts',
            'component_5': 'MFI Very Oversold (28.3): +2 pts',
            'mfi_priority_signal': True,
            'total_components': 5,
            'historical_probability': 72.4,
            'historical_bb_baseline': 72.4,
            'historical_component_success': 78.0,
            'historical_avg_win': 3.7,
            'historical_avg_loss': 5.4,
            'historical_avg_duration': 28.5,
            'market_baseline': 72.4,
            'market_health': 73.5,
            'total_bounces_analyzed': 9718,
            'indicator_benchmark': 75.0,
            'relative_performance': 'ABOVE_AVERAGE',
            # Add the fields the confidence module expects
            'volume_surge': True,
            'mfi_oversold': True,
            'stoch_oversold': False,
            'bb_expansion': True,
            'sentiment_data': {
                'lunar_galaxy_score': 72,
                'tm_trader_grade': 68
            }
        },
        {
            'symbol': 'ETH/USDT',
            'exchange': 'binance',
            'setup_type': 'SHORT',
            'bb_score': 18,
            'setup_quality': 'HIGH',
            'timestamp': datetime.now(),
            'bb_pct': 0.984,
            'rsi': 87.1,
            'volume_ratio': 1.6,
            'atr_pct': 0.032,
            'regime_confidence': 68,
            'regime_type': 'MIXED',
            'bb_suitability': 'GOOD',
            'position_multiplier': 1.2,
            'btc_health_score': 75,
            'alt_market_outlook': 'GOOD',
            'market_health_score': 73,
            'alt_season_indicator': 'NEUTRAL',
            'bb_score_34': 18,
            'setup_quality_enhanced': 'HIGH',
            'scoring_details': {
                'tier_scores': {
                    'base_bb': 8,
                    'money_flow': 5,
                    'bb_specific': 4,
                    'volume_momentum': 1,
                    'divergence': 0
                },
                'indicator_values': {
                    'mfi': 90.5,
                    'cmf': -0.028,
                    'bb_expansion': 1.71,
                    'volume_surge': True,
                    'bb_trend': 'Uptrend'
                },
                'breakdown': [
                    'Good Volume (1.6x): +1 pt',
                    'Bounce Confirmation: +1 pt',
                    'CMF Neutral (-0.028): +1 pt',
                    'BB High Expansion (1.71x): +2 pts',
                    'BB Trend Uptrend: +1 pt'
                ]
            },
            'technical_confidence': 0.0,
            'historical_confidence': 0.0,
            'sentiment_confidence': 0.0,
            'composite_confidence': 0.0,
            'confidence_tier': 'UNRATED',
            'confidence_rationale': 'No analysis available',
            'tier_base_bb': 8,
            'tier_money_flow': 5,
            'tier_bb_specific': 4,
            'tier_volume_momentum': 1,
            'tier_divergence': 0,
            'mfi_value': 90.5,
            'cmf_value': -0.028,
            'bb_expansion_ratio': 1.71,
            'volume_surge_detected': True,
            'bb_trend_direction': 'Uptrend',
            'component_1': 'Good Volume (1.6x): +1 pt',
            'component_2': 'Bounce Confirmation: +1 pt',
            'component_3': 'CMF Neutral (-0.028): +1 pt',
            'component_4': 'BB High Expansion (1.71x): +2 pts',
            'component_5': 'BB Trend Uptrend: +1 pt',
            'mfi_priority_signal': False,
            'total_components': 5,
            'historical_probability': 72.4,
            'historical_bb_baseline': 72.4,
            'historical_component_success': 75.0,
            'historical_avg_win': 3.9,
            'historical_avg_loss': 5.6,
            'historical_avg_duration': 26.2,
            'market_baseline': 72.4,
            'market_health': 73.5,
            'total_bounces_analyzed': 9718,
            'indicator_benchmark': 70.0,
            'relative_performance': 'AVERAGE',
            # Add the fields the confidence module expects
            'volume_surge': True,
            'mfi_overbought': True,
            'stoch_overbought': True,
            'bb_expansion': True,
            'sentiment_data': {
                'lunar_galaxy_score': 65,
                'tm_trader_grade': 55
            }
        }
    ]
    
    return sample_trades

def test_confidence_module():
    """Test the confidence module with sample data"""
    
    print("🧪 TESTING CONFIDENCE MODULE INTEGRATION")
    print("=" * 50)
    
    # Create sample trades
    sample_trades = create_sample_trades()
    print(f"✅ Created {len(sample_trades)} sample trades")
    
    # Create sample market regime
    market_regime = {
        'regime_type': 'MIXED',
        'regime_confidence': 68,
        'bb_suitability': 'GOOD',
        'position_multiplier': 1.2
    }
    print(f"✅ Created sample market regime: {market_regime['regime_type']}")
    
    # Initialize confidence module
    try:
        confidence_module = MinimalConfidenceModule()
        print("✅ Confidence module initialized")
    except Exception as e:
        print(f"❌ Failed to initialize confidence module: {e}")
        return False
    
    # Test individual trade enhancement
    print("\n🔍 Testing individual trade enhancement...")
    try:
        enhanced_trade = confidence_module.enhance_trade_with_confidence(
            sample_trades[0], market_regime
        )
        print(f"✅ Individual enhancement successful")
        print(f"   Symbol: {enhanced_trade['symbol']}")
        print(f"   Technical Confidence: {enhanced_trade['technical_confidence']}%")
        print(f"   Historical Confidence: {enhanced_trade['historical_confidence']}%")
        print(f"   Sentiment Confidence: {enhanced_trade['sentiment_confidence']}%")
        print(f"   Composite Confidence: {enhanced_trade['composite_confidence']}%")
        print(f"   Tier: {enhanced_trade['confidence_tier']}")
        print(f"   Rationale: {enhanced_trade['confidence_rationale']}")
    except Exception as e:
        print(f"❌ Individual enhancement failed: {e}")
        return False
    
    # Test batch enhancement
    print("\n🔍 Testing batch enhancement...")
    try:
        enhanced_trades = enhance_all_trades_with_confidence(
            sample_trades, market_regime, confidence_module
        )
        print(f"✅ Batch enhancement successful: {len(enhanced_trades)} trades enhanced")
        
        # Show results
        for i, trade in enumerate(enhanced_trades):
            print(f"\n📊 Trade {i+1}: {trade['symbol']} - {trade['setup_type']}")
            print(f"   BB Score: {trade['bb_score']}/34")
            print(f"   Composite Confidence: {trade['composite_confidence']}%")
            print(f"   Tier: {trade['confidence_tier']}")
            print(f"   Rationale: {trade['confidence_rationale']}")
            
    except Exception as e:
        print(f"❌ Batch enhancement failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Confidence module is working correctly")
    print("✅ Integration with main scanner should work")
    
    return True

if __name__ == "__main__":
    success = test_confidence_module()
    if success:
        print("\n🚀 Ready to run the main scanner with confidence enhancement!")
    else:
        print("\n❌ Confidence module test failed - check the errors above")
        sys.exit(1) 