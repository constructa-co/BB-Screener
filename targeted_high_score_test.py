# targeted_high_score_test.py
# Create test data specifically designed to trigger HIGH scores

import sys
import os
import pandas as pd
import numpy as np

sys.path.append('modules')

def create_high_score_test_data():
    """Create test data designed to trigger maximum scoring"""
    
    # Create data that should trigger OVERSOLD signals for LONG setup
    test_data = pd.DataFrame({
        # Price data trending down (for oversold conditions)
        'high': [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
        'low': [118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89],
        'close': [119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90],
        
        # Volume spike at the end (for volume surge)
        'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 3000, 3500],
        
        # BB bands - make current price touch lower band
        'bb_upper': [125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96],
        'bb_lower': [115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86],  # Last close (90) touches lower band (86)
        'bb_middle': [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91],
        
        # BB percentage - very oversold
        'bb_pct': [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        
        # RSI - very oversold  
        'rsi': [40, 35, 30, 25, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4],
        
        # Volume ratio - spike at end
        'volume_ratio': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.5],
        
        # ATR
        'atr': [2.0] * 20
    })
    
    return test_data

def test_targeted_scoring():
    """Test with data designed to maximize scoring"""
    
    print("🎯 TARGETED HIGH SCORE TEST")
    print("=" * 50)
    
    try:
        from modules.bb_detector import BBDetector
        
        detector = BBDetector()
        
        # Create targeted test data
        test_data = create_high_score_test_data()
        recent_3 = test_data.tail(3)
        last_row = test_data.iloc[-1]
        
        print("📊 TEST DATA DESIGNED FOR HIGH SCORES:")
        print(f"   Last Close: {last_row['close']}")
        print(f"   BB Lower: {last_row['bb_lower']} (Close should touch)")
        print(f"   BB %: {last_row['bb_pct']:.3f} (Very oversold)")
        print(f"   RSI: {last_row['rsi']} (Very oversold)")
        print(f"   Volume Ratio: {last_row['volume_ratio']} (High volume)")
        
        print("\n🧪 INDIVIDUAL INDICATOR TESTS:")
        
        # Test each indicator
        mfi = detector._calculate_money_flow_index(test_data)
        print(f"MFI: {mfi:.2f} ({'OVERSOLD +4pts' if mfi <= 20 else 'NEUTRAL +0pts'})")
        
        cmf = detector._calculate_chaikin_money_flow(test_data)
        print(f"CMF: {cmf:.4f} ({'SELLING +3pts' if cmf < -0.1 else 'MODERATE +2pts' if cmf < -0.05 else 'NEUTRAL +1pt' if abs(cmf) < 0.05 else 'NO BONUS'})")
        
        volume_surge = detector._calculate_volume_surge(test_data)
        print(f"Volume Surge: {volume_surge} ({'SURGE +2pts' if volume_surge else 'NO SURGE +0pts'})")
        
        bb_expansion = detector._calculate_bb_expansion(test_data)
        print(f"BB Expansion: {bb_expansion:.2f} ({'HIGH +2pts' if bb_expansion > 1.2 else 'MODERATE +1pt' if bb_expansion > 1.1 else 'NO BONUS +0pts'})")
        
        print("\n🎯 ACTUAL SCORING:")
        long_score = detector._calculate_long_score(recent_3, last_row, test_data)
        print(f"🔥 LONG SCORE: {long_score}")
        
        # Manual calculation verification
        manual_score = 0
        print(f"\n🔍 MANUAL VERIFICATION:")
        
        # Check BB Touch
        bb_touch = any(recent_3['low'] <= recent_3['bb_lower'])
        if bb_touch:
            manual_score += 3
            print(f"✅ BB Touch: +3 points")
        else:
            print(f"❌ BB Touch: +0 points")
            
        # Check BB Position
        if last_row['bb_pct'] <= 0.05:
            manual_score += 2
            print(f"✅ BB Position (extreme): +2 points")
        elif last_row['bb_pct'] <= 0.08:
            manual_score += 1
            print(f"✅ BB Position (very): +1 point")
        else:
            print(f"❌ BB Position: +0 points")
            
        # Check RSI
        if last_row['rsi'] <= 28:
            manual_score += 2
            print(f"✅ RSI (extreme oversold): +2 points")
        elif last_row['rsi'] <= 38:
            manual_score += 1
            print(f"✅ RSI (oversold): +1 point")
        else:
            print(f"❌ RSI: +0 points")
            
        # Check Volume
        if last_row['volume_ratio'] >= 1.8:
            manual_score += 2
            print(f"✅ Volume (high): +2 points")
        elif last_row['volume_ratio'] >= 1.3:
            manual_score += 1
            print(f"✅ Volume (good): +1 point")
        else:
            print(f"❌ Volume: +0 points")
            
        # Bounce confirmation
        if last_row['close'] > last_row['low'] and last_row['close'] > last_row['bb_lower']:
            manual_score += 1
            print(f"✅ Bounce confirmation: +1 point")
        else:
            print(f"❌ Bounce confirmation: +0 points")
        
        print(f"\n📊 RESULTS:")
        print(f"   Manual calculation: {manual_score} base points")
        print(f"   Actual score: {long_score} total points")
        print(f"   Enhanced indicators: {long_score - manual_score} additional points")
        
        if long_score >= 12:
            print(f"✅ PASSES institutional threshold (>=12)")
        else:
            print(f"❌ FAILS institutional threshold (>=12)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_targeted_scoring()
    