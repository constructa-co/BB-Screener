# debug_comprehensive_scoring.py
# Test the complete 16-tier enhanced scoring system

import sys
import os
import pandas as pd
import numpy as np

sys.path.append('modules')

def debug_comprehensive_scoring():
    """Debug the complete 16-tier enhanced scoring system"""
    
    print("🔍 TESTING COMPREHENSIVE 16-TIER SCORING SYSTEM")
    print("=" * 70)
    
    try:
        from bb_detector import BBDetector
        
        detector = BBDetector()
        print("✅ BBDetector imported successfully")
        
        # Check all calculation methods exist
        methods_to_check = [
            '_calculate_money_flow_index',
            '_calculate_chaikin_money_flow', 
            '_calculate_bb_expansion',
            '_calculate_bb_squeeze',
            '_calculate_bb_reversal_setup',
            '_calculate_bb_trend',
            '_calculate_volume_surge',
            '_calculate_stoch_oversold',
            '_calculate_stoch_overbought',
            '_calculate_cci_extreme',
            '_calculate_macd_divergence',
            '_calculate_rsi_divergence'
        ]
        
        print("\n📊 CHECKING ALL 12 CALCULATION METHODS:")
        missing_methods = []
        for method in methods_to_check:
            if hasattr(detector, method):
                print(f"✅ {method}")
            else:
                print(f"❌ {method} - MISSING")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n🚨 MISSING METHODS: {len(missing_methods)}")
            return False
        
        # Create comprehensive test data designed to trigger HIGH scores
        test_data = pd.DataFrame({
            'high': [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 115, 117, 116, 118, 120, 119],
            'low': [98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 113, 115, 114, 116, 118, 117],
            'close': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 114, 116, 115, 117, 119, 118],
            'volume': [1000, 1200, 1100, 900, 1300, 1500, 1400, 1600, 1800, 1700, 1900, 2100, 2000, 2200, 2300, 2500, 2400, 2600, 2800, 2700],
            'bb_upper': [102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 117, 119, 118, 120, 122, 121],
            'bb_lower': [96, 98, 100, 99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 111, 113, 112, 114, 116, 115],
            'bb_middle': [99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 114, 116, 115, 117, 119, 118],
            'bb_pct': [0.05] * 20,  # Always oversold for LONG signals
            'rsi': [15] * 20,       # Very oversold for maximum RSI points
            'volume_ratio': [2.5] * 20,  # High volume for maximum volume points
            'atr': [2.0] * 20
        })
        
        recent_3 = test_data.tail(3)
        last_row = test_data.iloc[-1]
        
        print("\n🧪 TESTING INDIVIDUAL INDICATORS:")
        total_expected_points = 0
        
        # Test each calculation method individually
        try:
            mfi = detector._calculate_money_flow_index(test_data)
            print(f"✅ MFI Value: {mfi:.2f}")
            if mfi <= 20:
                print("   → +4 points (MFI Oversold)")
                total_expected_points += 4
            elif mfi <= 30:
                print("   → +2 points (MFI Very Oversold)")
                total_expected_points += 2
            else:
                print("   → +0 points (MFI Neutral)")
        except Exception as e:
            print(f"❌ MFI Error: {e}")
            
        try:
            cmf = detector._calculate_chaikin_money_flow(test_data)
            print(f"✅ CMF Value: {cmf:.4f}")
            if cmf < -0.1:
                print("   → +3 points (Strong selling pressure)")
                total_expected_points += 3
            elif cmf < -0.05:
                print("   → +2 points (Moderate selling pressure)")
                total_expected_points += 2
            elif abs(cmf) < 0.05:
                print("   → +1 point (Neutral money flow)")
                total_expected_points += 1
            else:
                print("   → +0 points (No CMF bonus)")
        except Exception as e:
            print(f"❌ CMF Error: {e}")
            
        try:
            bb_exp = detector._calculate_bb_expansion(test_data)
            print(f"✅ BB Expansion: {bb_exp:.2f}")
            if bb_exp > 1.2:
                print("   → +2 points (High expansion)")
                total_expected_points += 2
            elif bb_exp > 1.1:
                print("   → +1 point (Moderate expansion)")
                total_expected_points += 1
            else:
                print("   → +0 points (No expansion bonus)")
        except Exception as e:
            print(f"❌ BB Expansion Error: {e}")
            
        try:
            bb_squeeze = detector._calculate_bb_squeeze(test_data)
            print(f"✅ BB Squeeze: {bb_squeeze}")
            if bb_squeeze:
                print("   → +2 points (Squeeze detected)")
                total_expected_points += 2
            else:
                print("   → +0 points (No squeeze)")
        except Exception as e:
            print(f"❌ BB Squeeze Error: {e}")
            
        try:
            volume_surge = detector._calculate_volume_surge(test_data)
            print(f"✅ Volume Surge: {volume_surge}")
            if volume_surge:
                print("   → +2 points (Volume surge detected)")
                total_expected_points += 2
            else:
                print("   → +0 points (No volume surge)")
        except Exception as e:
            print(f"❌ Volume Surge Error: {e}")
        
        print(f"\n📈 EXPECTED POINTS FROM NEW INDICATORS: {total_expected_points}")
        
        # Test base scoring (should be ~10 points with our test data)
        expected_base = 3 + 2 + 2 + 2 + 1  # BB Touch + Position + RSI + Volume + Bounce
        print(f"📈 EXPECTED BASE SCORING: {expected_base} points")
        print(f"📈 TOTAL EXPECTED SCORE: {expected_base + total_expected_points} points")
        
        print("\n🎯 ACTUAL SCORING TEST:")
        
        # Test actual scoring
        long_score = detector._calculate_long_score(recent_3, last_row, test_data)
        print(f"🔥 ACTUAL LONG SCORE: {long_score}")
        
        # Analyze results
        if long_score >= 20:
            print("🎉 EXCEPTIONAL! Comprehensive scoring is working perfectly!")
            quality = "Exceptional (25+)" if long_score >= 25 else "Excellent (22+)" if long_score >= 22 else "Very Good (18+)"
            print(f"   Quality Level: {quality}")
        elif long_score >= 12:
            print("✅ GOOD! Enhanced scoring is active and working!")
            print(f"   Quality Level: Fair-Good (12-17 points)")
        elif long_score >= 8:
            print("⚠️  PARTIAL: Some enhanced scoring working, but not complete")
        else:
            print("❌ PROBLEM: Enhanced scoring is not working properly")
            
        print(f"\n📊 SCORING ANALYSIS:")
        print(f"   Expected: {expected_base + total_expected_points} points")
        print(f"   Actual: {long_score} points")
        print(f"   Difference: {long_score - (expected_base + total_expected_points)} points")
        
        # Test threshold logic
        print(f"\n🎯 THRESHOLD TEST:")
        if long_score >= 12:
            print("✅ PASSES new institutional threshold (>=12)")
        else:
            print("❌ FAILS new institutional threshold (>=12)")
            
        return long_score >= 12
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    success = debug_comprehensive_scoring()
    if success:
        print("🚀 COMPREHENSIVE SCORING SYSTEM: OPERATIONAL!")
    else:
        print("🔧 COMPREHENSIVE SCORING SYSTEM: NEEDS DEBUGGING")