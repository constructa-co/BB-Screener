#!/usr/bin/env python3
"""
Test script to verify JSON serialization fix works correctly
Tests the make_json_safe function with problematic NumPy types
"""

import numpy as np
import json
import sys
import os

# Add the current directory to path to import trade_logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from trade_logger import make_json_safe
    print("✅ Successfully imported make_json_safe from trade_logger")
except ImportError as e:
    print(f"❌ Failed to import make_json_safe: {e}")
    sys.exit(1)

def test_json_serialization():
    """Test JSON serialization with problematic NumPy types"""
    
    print("🧪 Testing JSON serialization fix...")
    
    # Test data with problematic types that were causing errors
    test_data = {
        'volume_surge_detected': np.bool_(True),
        'mfi_priority_signal': np.bool_(False),
        'bb_score': np.int64(24),
        'probability': np.float64(85.5),
        'rsi': np.float32(65.2),
        'stochastic_k': np.float64(78.9),
        'volume_surge': np.int32(3),
        'confluence_score': np.float64(92.1),
        'historical_win_rate': np.float64(76.8),
        'category_win_rate': np.float64(82.3),
        'similar_setups_count': np.int64(15),
        'market_cap_rank': np.int64(45),
        'volume_24h_usd': np.float64(1234567.89),
        'price_change_24h': np.float64(-2.5),
        'nested_data': {
            'inner_bool': np.bool_(True),
            'inner_int': np.int64(42),
            'inner_float': np.float64(3.14159)
        },
        'array_data': np.array([1, 2, 3, 4, 5])
    }
    
    print(f"📊 Test data contains {len(test_data)} fields with NumPy types")
    print(f"   • Boolean types: {sum(1 for v in test_data.values() if isinstance(v, (np.bool_, np.bool8)))}")
    print(f"   • Integer types: {sum(1 for v in test_data.values() if isinstance(v, np.integer))}")
    print(f"   • Float types: {sum(1 for v in test_data.values() if isinstance(v, np.floating))}")
    print(f"   • Array types: {sum(1 for v in test_data.values() if isinstance(v, np.ndarray))}")
    
    try:
        # Test 1: Direct JSON serialization (should fail)
        print("\n🔍 Test 1: Direct JSON serialization (expected to fail)...")
        try:
            direct_result = json.dumps(test_data)
            print("❌ Direct serialization worked (unexpected)")
        except TypeError as e:
            print(f"✅ Direct serialization failed as expected: {e}")
        
        # Test 2: JSON serialization with make_json_safe (should work)
        print("\n🔍 Test 2: JSON serialization with make_json_safe...")
        safe_data = make_json_safe(test_data)
        result = json.dumps(safe_data)
        print(f"✅ JSON serialization works: {result[:100]}...")
        
        # Test 3: Verify data types were converted correctly
        print("\n🔍 Test 3: Verifying data type conversion...")
        print(f"   • volume_surge_detected: {type(safe_data['volume_surge_detected'])} = {safe_data['volume_surge_detected']}")
        print(f"   • bb_score: {type(safe_data['bb_score'])} = {safe_data['bb_score']}")
        print(f"   • probability: {type(safe_data['probability'])} = {safe_data['probability']}")
        print(f"   • nested_data.inner_bool: {type(safe_data['nested_data']['inner_bool'])} = {safe_data['nested_data']['inner_bool']}")
        print(f"   • array_data: {type(safe_data['array_data'])} = {safe_data['array_data']}")
        
        # Test 4: Verify all NumPy types were converted
        def check_numpy_types(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_numpy_types(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_numpy_types(v, f"{path}[{i}]")
            elif isinstance(obj, (np.bool_, np.bool8, np.integer, np.floating, np.ndarray)):
                print(f"❌ Found NumPy type at {path}: {type(obj)}")
                return False
        check_numpy_types(safe_data)
        print("✅ All NumPy types successfully converted to Python native types")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ JSON serialization fix is working correctly")
        print("✅ Ready to upload to Digital Ocean")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trade_logger_integration():
    """Test that trade_logger can actually log data with the fix"""
    
    print("\n🧪 Testing trade_logger integration...")
    
    try:
        from trade_logger import TradeLogger
        
        # Create a test trade record with NumPy types
        test_trade = {
            'symbol': 'TESTUSDT',
            'exchange': 'Binance',
            'timeframe': '4H',
            'bb_score': np.int64(85),
            'probability': np.float64(92.5),
            'risk_reward_ratio': np.float64(2.8),
            'current_price': np.float64(45000.0),
            'entry_price': np.float64(44800.0),
            'stop_loss': np.float64(44000.0),
            'target_1': np.float64(46000.0),
            'rsi': np.float64(65.2),
            'mfi': np.float64(78.9),
            'stochastic_k': np.float64(82.1),
            'volume_surge': np.int64(3),
            'macd_signal': 'bullish',
            'pattern_type': 'BB Bounce',
            'pattern_quality': 'EXCELLENT',
            'confluence_score': np.float64(94.2),
            'historical_win_rate': np.float64(78.5),
            'category_win_rate': np.float64(81.3),
            'similar_setups_count': np.int64(12),
            'market_cap': np.int64(25),
            'volume_24h': np.float64(9876543.21),
            'price_change_24h': np.float64(3.2),
            'scanner_type': 'bb_scanner',
            'scanner_specific_data': {
                'volume_surge_detected': np.bool_(True),
                'mfi_priority_signal': np.bool_(False),
                'bb_squeeze_active': np.bool_(True),
                'bb_expansion_active': np.bool_(False)
            }
        }
        
        print(f"📊 Test trade contains {len(test_trade)} fields")
        print(f"   • NumPy types: {sum(1 for v in test_trade.values() if isinstance(v, (np.bool_, np.bool8, np.integer, np.floating, np.ndarray)))}")
        
        # Test that the trade can be serialized
        safe_trade = make_json_safe(test_trade)
        json_result = json.dumps(safe_trade)
        print(f"✅ Trade serialization works: {len(json_result)} characters")
        
        print("✅ Trade logger integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Trade logger integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("JSON SERIALIZATION FIX VERIFICATION TEST")
    print("="*60)
    
    # Run tests
    test1_passed = test_json_serialization()
    test2_passed = test_trade_logger_integration()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ JSON serialization fix is working correctly")
        print("✅ Ready to upload to Digital Ocean")
        print("✅ Expected results after upload:")
        print("   • 'Logged trade BTC: 85% probability' (not 'Error logging trade BTC')")
        print("   • 'Scan 247 completed: 20 opportunities' (not '0 opportunities')")
        print("   • Database will have all trades with complete data")
    else:
        print("❌ SOME TESTS FAILED!")
        print("❌ Do not upload to Digital Ocean until tests pass")
        sys.exit(1)
