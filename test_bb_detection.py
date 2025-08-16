#!/usr/bin/env python3
"""
Test BB detection to see why it's not finding setups
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import MarketDataFetcher
from modules.bb_detector import BBDetector
import pandas as pd

def test_bb_detection():
    print("🔍 Testing BB detection...")
    
    # Initialize components
    fetcher = MarketDataFetcher()
    detector = BBDetector()
    
    # Test multiple exchanges and symbols
    test_cases = [
        ('BTC', 'binance'),
        ('BTC', 'bybit'),
        ('BTC', 'kucoin'),
        ('BTC', 'okx'),
        ('ETH', 'binance'),
        ('ETH', 'bybit'),
        ('ETH', 'kucoin'),
        ('ETH', 'okx'),
    ]
    
    for symbol, exchange in test_cases:
        print(f"\n📊 Testing {symbol} on {exchange}...")
        
        try:
            # Fetch data
            data = fetcher.fetch_ohlcv(exchange, symbol, '4h')
            print(f"   Data shape: {data.shape if data is not None else 'None'}")
            
            if data is not None and not data.empty:
                print(f"   ✅ Success! Latest price: {data['close'].iloc[-1]:.6f}")
                print(f"   Data points: {len(data)}")
                
                # Test BB detection
                result = detector.detect_setup(data, symbol)
                print(f"   BB Setup: {result['setup_type'] if result else 'None'}")
                print(f"   BB Score: {result.get('bb_score', 0) if result else 0}")
                
                if result and result['setup_type'] != 'NONE':
                    print(f"   🎯 SETUP FOUND! Probability: {result.get('probability', 0)}%")
                    return True
                    
            else:
                print("   ❌ No data returned")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n❌ No setups found in any test case")
    return False

if __name__ == "__main__":
    test_bb_detection()
