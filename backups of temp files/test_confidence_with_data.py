#!/usr/bin/env python3
"""
Test Confidence Enhancement with Sample Data
============================================

This script creates sample trade data and tests the confidence enhancement.
"""

import sys
import os
from datetime import datetime

# Add the current directory to the path
sys.path.append('.')

try:
    from main_scanner import ModularBBScanner
    from modules.minimal_confidence_module import enhance_all_trades_with_confidence
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def create_sample_trade_data():
    """Create sample trade data that matches the main scanner format"""
    
    sample_trades = [
        {
            'symbol': 'DYM/USDT',
            'exchange': 'binance',
            'setup_type': 'SHORT',
            'bb_score': 22,
            'setup_quality': 'PREMIUM',
            'timestamp': datetime.now(),
            'bb_pct': 0.85,
            'rsi': 72.5,
            'volume_ratio': 2.1,
            'atr_pct': 0.038,
            'regime_confidence': 68,
            'regime_type': 'MIXED',
            'bb_suitability': 'GOOD',
            'position_multiplier': 1.2,
            'btc_health_score': 75,
            'alt_market_outlook': 'GOOD',
            'market_health_score': 73,
            'alt_season_indicator': 'ALT_SEASON',
            'bb_score_34': 22,
            'setup_quality_enhanced': 'PREMIUM',
            'scoring_details': {
                'breakdown': ['BB Touch Upper Band: +3 pts', 'BB Position Extreme (0.850): +2 pts'],
                'tier_scores': {'base_bb': 5, 'money_flow': 7, 'bb_specific': 6, 'volume_momentum': 4, 'divergence': 0},
                'indicator_values': {'mfi': 82.5, 'cmf': 0.12, 'bb_expansion': 1.35, 'volume_surge': True, 'bb_trend': 'Uptrend'}
            },
            'technical_confidence': 0.0,
            'historical_confidence': 0.0,
            'sentiment_confidence': 0.0,
            'composite_confidence': 0.0,
            'confidence_tier': 'UNRATED',
            'confidence_rationale': 'No analysis available',
            'tier_base_bb': 5,
            'tier_money_flow': 7,
            'tier_bb_specific': 6,
            'tier_volume_momentum': 4,
            'tier_divergence': 0,
            'mfi_value': 82.5,
            'cmf_value': 0.12,
            'bb_expansion_ratio': 1.35,
            'volume_surge_detected': True,
            'bb_trend_direction': 'Uptrend',
            'component_1': 'BB Touch Upper Band: +3 pts',
            'component_2': 'BB Position Extreme (0.850): +2 pts',
            'component_3': '',
            'component_4': '',
            'component_5': '',
            'mfi_priority_signal': True,
            'total_components': 2,
            'historical_probability': 78.5,
            'historical_bb_baseline': 72.4,
            'historical_component_success': 75.2,
            'historical_avg_win': 4.2,
            'historical_avg_loss': 2.1,
            'historical_avg_duration': 18.5,
            'market_baseline': 72.4,
            'market_health': 73.5,
            'total_bounces_analyzed': 8500,
            'indicator_benchmark': 76.8,
            'relative_performance': 'ABOVE_AVERAGE',
            'probability': 78,
            'entry': 0.2456,
            'stop': 0.2523,
            'target1': 0.2389,
            'risk_reward': 4.4,
            'risk_pct': 1.5,
            'gain_pct': 6.6,
            'divergence_detected': True,
            'divergence_strength': 'STRONG',
            'divergence_confidence': 85,
            'divergence_indicators': 'RSI, MACD',
            'volume_confirmation': True,
            'momentum_alignment': True,
            'rr_acceptable': True,
            'risk_acceptable': True,
            'patterns_detected': 'Bearish Engulfing, Shooting Star',
            'significant_patterns': 'Bearish Engulfing',
            'pattern_confidence': 82,
            'pattern_boost': 5
        }
    ]
    
    return sample_trades

def test_confidence_enhancement():
    """Test confidence enhancement with sample data"""
    
    print("🧪 TESTING CONFIDENCE ENHANCEMENT WITH SAMPLE DATA")
    print("=" * 60)
    
    # Initialize scanner
    scanner = ModularBBScanner()
    print("✅ Scanner initialized")
    
    # Create sample trade data
    sample_trades = create_sample_trade_data()
    print(f"✅ Created {len(sample_trades)} sample trades")
    
    # Show original data
    first_trade = sample_trades[0]
    print(f"\n📊 ORIGINAL TRADE DATA:")
    print(f"   Symbol: {first_trade.get('symbol', 'UNKNOWN')}")
    print(f"   BB Score: {first_trade.get('bb_score', 'NOT_FOUND')}")
    print(f"   Historical Probability: {first_trade.get('historical_probability', 'NOT_FOUND')}")
    print(f"   Total Bounces Analyzed: {first_trade.get('total_bounces_analyzed', 'NOT_FOUND')}")
    print(f"   Composite Confidence: {first_trade.get('composite_confidence', 'NOT_FOUND')}")
    
    # Mock market regime
    market_regime = {
        'regime_type': 'MIXED',
        'regime_confidence': 65,
        'btc_health_score': 75,
        'alt_market_outlook': 'GOOD'
    }
    
    # Test confidence enhancement
    print(f"\n🎯 TESTING CONFIDENCE ENHANCEMENT...")
    try:
        enhanced_trades = enhance_all_trades_with_confidence(
            sample_trades, market_regime, scanner.confidence_module
        )
        
        print(f"✅ Confidence enhancement successful: {len(enhanced_trades)} trades enhanced")
        
        if enhanced_trades:
            first_enhanced = enhanced_trades[0]
            print(f"\n📊 ENHANCED TRADE DATA:")
            print(f"   Symbol: {first_enhanced.get('symbol', 'UNKNOWN')}")
            print(f"   BB Score: {first_enhanced.get('bb_score', 'NOT_FOUND')}")
            print(f"   Technical Confidence: {first_enhanced.get('technical_confidence', 'NOT_FOUND')}")
            print(f"   Historical Confidence: {first_enhanced.get('historical_confidence', 'NOT_FOUND')}")
            print(f"   Sentiment Confidence: {first_enhanced.get('sentiment_confidence', 'NOT_FOUND')}")
            print(f"   Composite Confidence: {first_enhanced.get('composite_confidence', 'NOT_FOUND')}")
            print(f"   Confidence Tier: {first_enhanced.get('confidence_tier', 'NOT_FOUND')}")
            print(f"   Confidence Rationale: {first_enhanced.get('confidence_rationale', 'NOT_FOUND')[:100]}...")
            
            # Check if confidence fields were properly updated
            if first_enhanced.get('composite_confidence', 0) > 0:
                print(f"\n🎉 SUCCESS! Confidence enhancement is working!")
                return True
            else:
                print(f"\n❌ FAILED! Composite confidence is still 0")
                return False
        
    except Exception as e:
        print(f"❌ Confidence enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False

if __name__ == "__main__":
    success = test_confidence_enhancement()
    
    if success:
        print("\n🏆 CONFIDENCE ENHANCEMENT TEST PASSED!")
        print("🚀 The confidence module is working correctly!")
    else:
        print("\n❌ CONFIDENCE ENHANCEMENT TEST FAILED!")
        sys.exit(1) 