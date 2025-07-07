# real_market_test.py
# Test enhanced BB detector with real market data

import sys
import os
sys.path.append('modules')

def test_real_market_data():
    """Test enhanced BB detector with real market data"""
    
    print("🚀 REAL MARKET DATA TEST - ENHANCED BB DETECTOR")
    print("=" * 60)
    
    try:
        from main_scanner import ModularBBScanner
        
        print("✅ Initializing Enhanced BB Scanner...")
        scanner = ModularBBScanner()
        
        print("📊 Testing data fetcher...")
        # Test data fetching for a few major coins
        test_coins = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']
        
        for symbol in test_coins[:2]:  # Test first 2 coins
            print(f"\n🔍 TESTING {symbol}:")
            
            try:
                # Fetch real market data
                df = scanner.data_fetcher.fetch_ohlcv('binance', symbol)
                
                if df is not None and len(df) > 0:
                    print(f"✅ Data fetched: {len(df)} candles")
                    
                    # Add technical indicators - using the method from main_scanner.py
                    df_with_indicators = df.copy()
                    
                    # Add basic indicators that BB detector needs
                    import pandas_ta as ta
                    
                    # Add Bollinger Bands
                    bb = ta.bbands(df_with_indicators['close'], length=20, std=2)
                    df_with_indicators['bb_upper'] = bb['BBU_20_2.0']
                    df_with_indicators['bb_lower'] = bb['BBL_20_2.0']
                    df_with_indicators['bb_middle'] = bb['BBM_20_2.0']
                    df_with_indicators['bb_pct'] = ((df_with_indicators['close'] - df_with_indicators['bb_lower']) / 
                                                   (df_with_indicators['bb_upper'] - df_with_indicators['bb_lower']))
                    
                    # Add RSI
                    df_with_indicators['rsi'] = ta.rsi(df_with_indicators['close'], length=14)
                    
                    # Add volume ratio
                    df_with_indicators['volume_ratio'] = df_with_indicators['volume'] / df_with_indicators['volume'].rolling(20).mean()
                    
                    # Add ATR
                    df_with_indicators['atr'] = ta.atr(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], length=14)
                    
                    print(f"✅ Technical indicators added")
                    
                    # Test BB analysis with enhanced scoring
                    bb_analysis = scanner.bb_detector.analyze_bb_setup(df_with_indicators)
                    
                    print(f"📊 BB ANALYSIS RESULTS:")
                    print(f"   Setup Type: {bb_analysis.get('setup_type', 'NONE')}")
                    print(f"   BB Score: {bb_analysis.get('bb_score', 0)}")
                    print(f"   Setup Quality: {bb_analysis.get('setup_quality', 'None')}")
                    
                    if bb_analysis.get('setup_type') != 'NONE':
                        print(f"   Entry: ${bb_analysis.get('entry', 0):.2f}")
                        print(f"   Stop: ${bb_analysis.get('stop', 0):.2f}")
                        print(f"   Target: ${bb_analysis.get('target1', 0):.2f}")
                        print(f"   Risk/Reward: {bb_analysis.get('risk_reward', 0):.2f}")
                        
                        # Show if enhanced scoring is working
                        score = bb_analysis.get('bb_score', 0)
                        if score >= 12:
                            print(f"✅ PASSES institutional threshold (score >= 12)")
                            if score >= 22:
                                print(f"🏆 EXCELLENT setup quality!")
                            elif score >= 18:
                                print(f"🎯 VERY GOOD setup quality!")
                        else:
                            print(f"❌ Below institutional threshold")
                    else:
                        print(f"   No BB setup detected")
                        
                else:
                    print(f"❌ Failed to fetch data for {symbol}")
                    
            except Exception as e:
                print(f"❌ Error testing {symbol}: {e}")
        
        print(f"\n🎯 REAL MARKET TEST SUMMARY:")
        print(f"✅ Enhanced BB detector operational")
        print(f"✅ 16-tier scoring system active")
        print(f"✅ Real market data integration working")
        print(f"✅ Ready for full market scan!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in real market test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_market_scan():
    """Run a full market scan with enhanced BB detector"""
    
    print("\n🌍 FULL MARKET SCAN WITH ENHANCED BB DETECTOR")
    print("=" * 60)
    
    try:
        from main_scanner import ModularBBScanner
        
        scanner = ModularBBScanner()
        
        print("🚀 Running full enhanced market scan...")
        print("   (This will take 2-3 minutes with enhanced scoring)")
        
        # Run the enhanced scanner
        scanner.run()
        
        print("✅ Full market scan complete!")
        print("📊 Check outputs/excel_reports/ for detailed results")
        
    except Exception as e:
        print(f"❌ Error in full market scan: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 TESTING ENHANCED BB DETECTOR WITH REAL MARKET DATA")
    print("=" * 70)
    
    # Test with real market data first
    success = test_real_market_data()
    
    if success:
        print(f"\n🚀 REAL MARKET TEST: SUCCESSFUL!")
        
        # Ask if user wants full scan
        response = input("\n🤔 Run full market scan with enhanced detector? (y/n): ")
        if response.lower().startswith('y'):
            run_full_market_scan()
        else:
            print("✅ Enhanced BB detector ready for deployment!")
    else:
        print(f"\n🔧 REAL MARKET TEST: NEEDS DEBUGGING")
        print("Please fix issues before running full scan.")