#!/usr/bin/env python3
"""
Test the full fetch_ohlcv method including technical indicators
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import MarketDataFetcher

def test_full_fetch():
    print("🔍 Testing full fetch_ohlcv method...")
    
    # Initialize data fetcher
    fetcher = MarketDataFetcher()
    
    # Test with BTC on bybit
    symbol = 'BTC'
    exchange = 'bybit'
    
    print(f"\n📊 Testing {symbol} on {exchange}...")
    
    try:
        # Call the full fetch_ohlcv method
        df = fetcher.fetch_ohlcv(exchange, symbol, '4h')
        
        if df is not None and not df.empty:
            print(f"   ✅ Full fetch successful!")
            print(f"   DataFrame shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Latest price: ${df['close'].iloc[-1]:,.2f}")
            print(f"   BB Upper: ${df['bb_upper'].iloc[-1]:,.2f}")
            print(f"   BB Lower: ${df['bb_lower'].iloc[-1]:,.2f}")
            print(f"   RSI: {df['rsi'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ Full fetch returned None or empty DataFrame")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_fetch()
