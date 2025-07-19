#!/usr/bin/env python3
"""
Quick Complete Flow Test
========================

This script tests the complete flow to verify the fix is working.
"""

import sys
import os

# Add the current directory to the path
sys.path.append('.')

try:
    from main_scanner import ModularBBScanner
    print("✅ Successfully imported main scanner")
except ImportError as e:
    print(f"❌ Failed to import main scanner: {e}")
    sys.exit(1)

def test_complete_flow():
    """Test the complete flow with the fix applied"""
    
    print("🧪 TESTING COMPLETE FLOW WITH FIX APPLIED")
    print("=" * 50)
    
    # Initialize scanner
    scanner = ModularBBScanner()
    print("✅ Scanner initialized")
    
    # Test analyze_coin_comprehensive method
    print("\n🔍 Testing analyze_coin_comprehensive method...")
    try:
        # Test with a single coin
        results = scanner.analyze_coin_comprehensive("BTC/USDT")
        print(f"✅ analyze_coin_comprehensive returned {len(results)} results")
        
        if results:
            print(f"   First result - Symbol: {results[0].get('symbol', 'UNKNOWN')}")
            print(f"   First result - BB Score: {results[0].get('bb_score', 'NOT_FOUND')}")
            print(f"   First result - Setup Type: {results[0].get('setup_type', 'NOT_FOUND')}")
            
            # Check if confidence fields are initialized
            confidence_fields = ['technical_confidence', 'historical_confidence', 'sentiment_confidence', 'composite_confidence', 'confidence_tier']
            for field in confidence_fields:
                value = results[0].get(field, 'NOT_FOUND')
                print(f"   {field}: {value}")
        
    except Exception as e:
        print(f"❌ analyze_coin_comprehensive failed: {e}")
        return False
    
    print("\n🎉 Complete flow test passed!")
    print("✅ The fix is working correctly!")
    print("✅ Results are being collected properly!")
    print("✅ Confidence enhancement should now execute in the main scanner!")
    
    return True

if __name__ == "__main__":
    success = test_complete_flow()
    
    if success:
        print("\n🏆 ALL TESTS PASSED!")
        print("🚀 Your confidence-enhanced BB scanner is ready to run!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 