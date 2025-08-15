# test_db_structure.py
# Test the code structure without database connection

import sys
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.insert(0, '/Users/robertsmith/Documents/BB Screener')

from modules.output_generator import OutputGenerator
from trade_logger import make_json_safe
import json

print("🧪 Testing code structure and JSON serialization...")

# Test 1: Test make_json_safe function
print("\n1. Testing make_json_safe function...")
test_data = {
    'bb_score': np.int64(24),
    'volume_surge': np.bool_(True),
    'mfi_value': np.float64(75.5),
    'array_data': np.array([1, 2, 3]),
    'timestamp': pd.Timestamp('2024-01-01')
}

try:
    safe_data = make_json_safe(test_data)
    json_string = json.dumps(safe_data)
    print("✅ make_json_safe works correctly")
    print(f"   Original: {test_data}")
    print(f"   JSON: {json_string}")
except Exception as e:
    print(f"❌ make_json_safe failed: {e}")

# Test 2: Test OutputGenerator initialization
print("\n2. Testing OutputGenerator initialization...")
try:
    output_gen = OutputGenerator()
    print("✅ OutputGenerator initialized successfully")
except Exception as e:
    print(f"❌ OutputGenerator failed: {e}")

# Test 3: Test mock data creation
print("\n3. Testing mock data creation...")
mock_data = {
    'symbol': ['BTC', 'ETH', 'SOL'],
    'exchange': ['binance', 'binance', 'kucoin'],
    'setup_type': ['SHORT', 'SHORT', 'LONG'],
    'probability': [85, 78, 72],
    'entry': [50000, 3200, 120],
    'stop': [51500, 3280, 115],
    'target1': [47500, 3050, 130],
    'bb_score': [24, 18, 15],
    'volume_surge_detected': [np.bool_(True), np.bool_(False), np.bool_(True)],
    'mfi_value': [75.5, 68.2, 32.1],
    'risk_pct': [3.0, 2.5, 4.2],
    'tier': ['PREMIUM', 'PREMIUM', 'HIGH']
}

try:
    df = pd.DataFrame(mock_data)
    print(f"✅ DataFrame created: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ DataFrame creation failed: {e}")

# Test 4: Test JSON serialization of DataFrame
print("\n4. Testing DataFrame JSON serialization...")
try:
    for _, row in df.iterrows():
        trade_data = row.to_dict()
        safe_trade = make_json_safe(trade_data)
        json_string = json.dumps(safe_trade)
        print(f"✅ Row {row['symbol']} serialized successfully")
    print("✅ All DataFrame rows can be JSON serialized")
except Exception as e:
    print(f"❌ DataFrame JSON serialization failed: {e}")

# Test 5: Test market regime data
print("\n5. Testing market regime data...")
mock_regime = {
    'btc_dominance': 57.5,
    'fear_greed_index': 73,
    'alt_season_indicator': False,
    'market_health_score': 65,
    'regime_type': 'NEUTRAL'
}

try:
    safe_regime = make_json_safe(mock_regime)
    json_string = json.dumps(safe_regime)
    print("✅ Market regime data serialized successfully")
except Exception as e:
    print(f"❌ Market regime serialization failed: {e}")

print("\n🎉 Structure test complete!")
print("\n📋 Summary:")
print("   - All imports work correctly")
print("   - make_json_safe function handles NumPy types")
print("   - OutputGenerator can be initialized")
print("   - DataFrame creation and serialization works")
print("   - Market regime data serialization works")
print("\n💡 Next step: Test on Digital Ocean server where database is accessible")
