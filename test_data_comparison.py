#!/usr/bin/env python3
"""
Test script to compare data fetching between local and Digital Ocean
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import MarketDataFetcher
from modules.bb_detector import BBDetector
import pandas as pd

def test_data_fetching():
    print("🔍 Testing data fetching and BB detection...")
    
    # Initialize components
    fetcher = MarketDataFetcher()
    detector = BBDetector()
    
    # Test with a specific coin that should have data
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'ELF/USDT']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        try:
            # Fetch data
            data = fetcher.fetch_ohlcv('binance', symbol, '4h')
            print(f"   Data shape: {data.shape if data is not None else 'None'}")
            
            if data is not None and not data.empty:
                print(f"   Latest price: {data['close'].iloc[-1]:.6f}")
                print(f"   Data points: {len(data)}")
                
                # Test BB detection
                result = detector.detect_setup(data, symbol)
                print(f"   BB Setup: {result['setup_type'] if result else 'None'}")
                print(f"   BB Score: {result.get('bb_score', 0) if result else 0}")
            else:
                print("   ❌ No data returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_data_fetching()
