# test_enhanced_scoring.py
# Check if your BB detector actually has the enhanced scoring

import sys
import os
import pandas as pd

sys.path.append('modules')

def test_enhanced_scoring():
    """Test if enhanced scoring with all indicators is actually working"""
    
    print("🔍 CHECKING ENHANCED SCORING INTEGRATION")
    print("=" * 60)
    
    try:
        from bb_detector import BBDetector
        
        detector = BBDetector()
        
        # Check which methods exist
        methods_to_check = [
            '_calculate_money_flow_index',
            '_calculate_chaikin_money_flow', 
            '_calculate_bb_expansion',
            '_calculate_bb_squeeze'
        ]
        
        print("📊 CHECKING CALCULATION METHODS:")
        for method in methods_to_check:
            if hasattr(detector, method):
                print(f"✅ {method} - Found")
            else:
                print(f"❌ {method} - Missing")
        
        # Create test data with BB columns
        test_data = pd.DataFrame({
            'high': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 115],
            'low': [98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 113],
            'close': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 114],
            'volume': [1000, 1200, 1100, 900, 1300, 1500, 1400, 1600, 1800, 1700, 1900, 2100, 2000, 2200, 2300],
            'bb_upper': [102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 117],
            'bb_lower': [96, 98, 100, 99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 111],
            'bb_middle': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 114],
            'bb_pct': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            'rsi': [30, 35, 40, 35, 45, 50, 45, 55, 60, 55, 65, 70, 65, 75, 80],
            'volume_ratio': [1.0, 1.2, 1.1, 0.9, 1.3, 1.5, 1.4, 1.6, 1.8, 1.7, 1.9, 2.1, 2.0, 2.2, 2.3],
            'atr': [2.0] * 15
        })
        
        recent_3 = test_data.tail(3)
        last_row = test_data.iloc[-1]
        
        print("\n📈 TESTING SCORING METHODS:")
        
        # Test if methods accept df parameter (enhanced version)
        try:
            long_score = detector._calculate_long_score(recent_3, last_row, test_data)
            print(f"✅ Long Score (Enhanced): {long_score}")
            
            short_score = detector._calculate_short_score(recent_3, last_row, test_data)
            print(f"✅ Short Score (Enhanced): {short_score}")
            
            # Check if scores are higher (indicating enhanced scoring)
            if long_score > 10 or short_score > 10:
                print("🎉 ENHANCED SCORING IS ACTIVE! (High scores detected)")
                print("   This indicates MFI+CMF+BB indicators are working")
            else:
                print("⚠️  BASIC SCORING ONLY (Low scores)")
                print("   Enhanced indicators may not be integrated yet")
                
        except TypeError as e:
            if "unexpected keyword argument" in str(e) or "takes" in str(e):
                print("❌ SCORING METHODS NOT UPDATED")
                print("   Still using old method signatures (missing df parameter)")
            else:
                print(f"❌ Error in scoring: {e}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_enhanced_scoring()