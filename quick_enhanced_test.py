# quick_enhanced_test.py
# Quick test to verify enhanced scoring is working

import sys
import os
sys.path.append('modules')

def test_xlm_scoring():
    """Test enhanced scoring on XLM (which showed score 20)"""
    
    print("🔍 TESTING ENHANCED SCORING ON XLM")
    print("=" * 50)
    
    try:
        from main_scanner import ModularBBScanner
        
        scanner = ModularBBScanner()
        
        # Get XLM data (same as the scan)
        df = scanner.data_fetcher.fetch_ohlcv('bybit', 'XLM')
        
        if df is not None:
            print(f"✅ XLM data fetched: {len(df)} candles")
            
            # Add technical indicators manually
            import pandas_ta as ta
            
            # Add all required indicators
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0'] 
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_pct'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']))
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            print(f"✅ Technical indicators added")
            
            # Test enhanced BB analysis
            bb_analysis = scanner.bb_detector.analyze_bb_setup(df)
            
            print(f"\n📊 XLM ENHANCED SCORING RESULTS:")
            print(f"   Setup Type: {bb_analysis.get('setup_type', 'NONE')}")
            print(f"   BB Score: {bb_analysis.get('bb_score', 0)}")
            print(f"   Setup Quality: {bb_analysis.get('setup_quality', 'None')}")
            
            # Manual indicator checks
            print(f"\n🔍 MANUAL INDICATOR VERIFICATION:")
            
            last = df.iloc[-1]
            print(f"   Current Price: ${last['close']:.6f}")
            print(f"   BB Upper: ${last['bb_upper']:.6f}")
            print(f"   BB Lower: ${last['bb_lower']:.6f}")
            print(f"   BB %: {last['bb_pct']:.3f}")
            print(f"   RSI: {last['rsi']:.1f}")
            print(f"   Volume Ratio: {last['volume_ratio']:.1f}x")
            
            # Test individual enhanced methods
            if hasattr(scanner.bb_detector, '_calculate_money_flow_index'):
                mfi = scanner.bb_detector._calculate_money_flow_index(df)
                print(f"   MFI: {mfi:.1f}")
                
            if hasattr(scanner.bb_detector, '_calculate_chaikin_money_flow'):
                cmf = scanner.bb_detector._calculate_chaikin_money_flow(df)
                print(f"   CMF: {cmf:.4f}")
                
            if hasattr(scanner.bb_detector, '_calculate_volume_surge'):
                vol_surge = scanner.bb_detector._calculate_volume_surge(df)
                print(f"   Volume Surge: {vol_surge}")
                
            # Validate the score
            expected_range = "12-30 points"
            actual_score = bb_analysis.get('bb_score', 0)
            
            if actual_score >= 12:
                print(f"\n✅ ENHANCED SCORING WORKING!")
                print(f"   Score {actual_score} is in expected range {expected_range}")
            else:
                print(f"\n⚠️  SCORING ISSUE:")
                print(f"   Score {actual_score} below expected range {expected_range}")
                
        else:
            print("❌ Failed to fetch XLM data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xlm_scoring()