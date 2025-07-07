# test_mfi_integration.py
# Test that MFI is working in your enhanced BB detector

import sys
import os
import pandas as pd
import numpy as np

sys.path.append('modules')

def test_mfi_integration():
    """Test that MFI calculation is working in BB detector"""
    
    print("🧪 TESTING MFI INTEGRATION")
    print("=" * 50)
    
    try:
        # Import your enhanced BB detector
        from modules.bb_detector import BBDetector
        
        detector = BBDetector()
        print("✅ BBDetector imported successfully")
        
        # Check if MFI method exists
        if hasattr(detector, '_calculate_money_flow_index'):
            print("✅ MFI calculation method found")
            
            # Create test data
            test_data = pd.DataFrame({
                'high': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 115],
                'low': [98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 113],
                'close': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 114],
                'volume': [1000, 1200, 1100, 900, 1300, 1500, 1400, 1600, 1800, 1700, 1900, 2100, 2000, 2200, 2300]
            })
            
            # Test MFI calculation
            mfi_value = detector._calculate_money_flow_index(test_data)
            print(f"✅ MFI calculation successful: {mfi_value:.2f}")
            
            # Validate MFI range
            if 0 <= mfi_value <= 100:
                print("✅ MFI value in valid range (0-100)")
                
                # Test MFI signal classification
                if mfi_value <= 20:
                    signal = "OVERSOLD (4 points in LONG scoring)"
                elif mfi_value <= 30:
                    signal = "VERY OVERSOLD (2 points in LONG scoring)"
                elif mfi_value >= 80:
                    signal = "OVERBOUGHT (4 points in SHORT scoring)"
                elif mfi_value >= 70:
                    signal = "VERY OVERBOUGHT (2 points in SHORT scoring)"
                else:
                    signal = "NEUTRAL (0 points)"
                
                print(f"📊 MFI Signal: {signal}")
                print("✅ MFI integration working correctly!")
                
            else:
                print(f"⚠️  MFI value out of range: {mfi_value}")
                
        else:
            print("❌ MFI calculation method not found")
            print("Make sure you added the _calculate_money_flow_index method")
            
    except Exception as e:
        print(f"❌ Error testing MFI integration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎯 MFI INTEGRATION TEST COMPLETE")
    return True

if __name__ == "__main__":
    test_mfi_integration()