#!/usr/bin/env python3
"""
Quick test to debug BB detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import MarketDataFetcher
from modules.bb_detector import BBDetector

def test_bb_detection():
    print("🔍 Testing BB Detection...")
    
    # Initialize components
    data_fetcher = MarketDataFetcher()
    bb_detector = BBDetector()
    
    # Test with a few popular coins
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        try:
            # Get data
            df = data_fetcher.fetch_ohlcv('binance', symbol, '4h')
            if df is None or df.empty:
                print(f"   ❌ No data for {symbol}")
                continue
                
            print(f"   ✅ Got {len(df)} candles")
            
            # Test BB detection
            result = bb_detector.analyze_bb_setup(df)
            
            setup_type = result.get('setup_type', 'NONE')
            bb_score = result.get('bb_score', 0)
            
            print(f"   🎯 Setup: {setup_type} | Score: {bb_score}")
            
            if setup_type != 'NONE':
                print(f"   💰 Entry: {result.get('entry', 0):.4f}")
                print(f"   🛑 Stop: {result.get('stop', 0):.4f}")
                print(f"   🎯 Target: {result.get('target1', 0):.4f}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ BB Detection test complete")

if __name__ == "__main__":
    test_bb_detection()
