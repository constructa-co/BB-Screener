#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from datetime import datetime

def json_serial(obj):
    """JSON serializer for objects not serializable by default"""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

def test_json_serialization():
    print("🧪 TESTING JSON SERIALIZATION FIX")
    print("="*50)
    
    # Test data with boolean values
    test_data = {
        'symbol': 'BTC',
        'entry_price': 45000.0,
        'volume_surge_detected': True,  # This was causing the error
        'mfi_priority_signal': False,   # This was causing the error
        'alt_season_indicator': True,   # This was causing the error
        'numpy_bool': np.bool_(True),
        'numpy_float': np.float64(123.45),
        'pandas_timestamp': pd.Timestamp.now(),
        'normal_string': 'test'
    }
    
    print("📊 Test data:")
    for key, value in test_data.items():
        print(f"   {key}: {value} (type: {type(value)})")
    
    try:
        # Try with the old method (should fail)
        print("\n❌ Testing old method (should fail):")
        old_result = json.dumps(test_data, default=str)
        print("   Old method worked (unexpected)")
    except Exception as e:
        print(f"   ✅ Old method failed as expected: {e}")
    
    try:
        # Try with the new method (should work)
        print("\n✅ Testing new method (should work):")
        new_result = json.dumps(test_data, default=json_serial)
        print("   ✅ New method worked!")
        print(f"   Result: {new_result[:100]}...")
        
        # Test parsing back
        parsed = json.loads(new_result)
        print("   ✅ Successfully parsed back!")
        
    except Exception as e:
        print(f"   ❌ New method failed: {e}")
    
    print("\n🎯 JSON Serialization Test Complete!")

if __name__ == "__main__":
    test_json_serialization()
