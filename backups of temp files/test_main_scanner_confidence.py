#!/usr/bin/env python3
"""
Quick Main Scanner Confidence Test
==================================

This script tests the main scanner with confidence module integration
without running the full scan.
"""

import sys
import os
import asyncio

# Add the current directory to the path
sys.path.append('.')

try:
    from main_scanner import ModularBBScanner
    print("✅ Successfully imported main scanner")
except ImportError as e:
    print(f"❌ Failed to import main scanner: {e}")
    sys.exit(1)

async def test_confidence_integration():
    """Test confidence module integration in main scanner"""
    
    print("🧪 TESTING MAIN SCANNER CONFIDENCE INTEGRATION")
    print("=" * 50)
    
    # Initialize scanner
    scanner = ModularBBScanner()
    print("✅ Scanner initialized")
    
    # Test confidence module initialization
    if hasattr(scanner, 'confidence_module') and scanner.confidence_module:
        print("✅ Confidence module initialized")
    else:
        print("❌ Confidence module not initialized")
        return False
    
    # Create sample trade data to test confidence enhancement
    print("\n🎯 Testing confidence enhancement with sample data...")
    try:
        from modules.minimal_confidence_module import enhance_all_trades_with_confidence
        
        # Create sample trade data
        sample_trades = [
            {
                'symbol': 'BTC/USDT',
                'exchange': 'binance',
                'setup_type': 'LONG',
                'bb_score': 24,
                'setup_quality': 'PREMIUM',
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
                'historical_probability': 78.5,
                'total_bounces_analyzed': 8500,
                'mfi_value': 18.5,
                'cmf_value': -0.08,
                'bb_expansion_ratio': 1.35,
                'volume_surge_detected': True,
                'bb_trend_direction': 'Downtrend',
                'stoch_oversold': True,
                'bb_expansion': True,
                'mfi_oversold': True,
                'volume_surge': True
            }
        ]
        
        # Mock market regime data
        market_regime = {
            'regime_type': 'MIXED',
            'regime_confidence': 65,
            'btc_health_score': 75,
            'alt_market_outlook': 'GOOD'
        }
        
        enhanced_trades = enhance_all_trades_with_confidence(
            sample_trades, market_regime, scanner.confidence_module
        )
        
        print(f"✅ Confidence enhancement successful: {len(enhanced_trades)} trades enhanced")
        
        if enhanced_trades:
            first_enhanced = enhanced_trades[0]
            print(f"   Enhanced Symbol: {first_enhanced.get('symbol', 'UNKNOWN')}")
            print(f"   Enhanced BB Score: {first_enhanced.get('bb_score', 'NOT_FOUND')}")
            print(f"   Composite Confidence: {first_enhanced.get('composite_confidence', 'NOT_FOUND')}")
            print(f"   Confidence Tier: {first_enhanced.get('confidence_tier', 'NOT_FOUND')}")
            print(f"   Confidence Rationale: {first_enhanced.get('confidence_rationale', 'NOT_FOUND')[:100]}...")
            
            # Check if confidence fields are properly set
            confidence_fields = ['technical_confidence', 'historical_confidence', 'sentiment_confidence', 'composite_confidence', 'confidence_tier', 'confidence_rationale']
            for field in confidence_fields:
                value = first_enhanced.get(field, 'NOT_FOUND')
                print(f"   {field}: {value}")
        
    except Exception as e:
        print(f"❌ Confidence enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Main test function"""
    success = await test_confidence_integration()
    
    if success:
        print("\n🎉 Main scanner confidence integration test passed!")
    else:
        print("\n❌ Main scanner confidence integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 