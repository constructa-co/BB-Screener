#!/usr/bin/env python3
"""
Debug test for our data fetcher to see exactly what's failing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.data_fetcher import MarketDataFetcher

def test_data_fetcher_debug():
    print("🔍 Debugging data fetcher...")
    
    # Initialize data fetcher
    fetcher = MarketDataFetcher()
    
    print(f"Available exchanges: {fetcher.get_available_exchanges()}")
    
    # Test with BTC on bybit
    symbol = 'BTC'
    exchange = 'bybit'
    
    print(f"\n📊 Testing {symbol} on {exchange}...")
    
    try:
        # Check if exchange exists
        if exchange not in fetcher.exchanges:
            print(f"   ❌ Exchange {exchange} not found in fetcher.exchanges")
            return
            
        ex = fetcher.exchanges[exchange]
        print(f"   ✅ Exchange {exchange} found")
        
        # Check markets
        market = f"{symbol}/USDT"
        print(f"   Looking for market: {market}")
        print(f"   Total markets loaded: {len(ex.markets)}")
        
        if market in ex.markets:
            print(f"   ✅ Market {market} found in markets")
        else:
            print(f"   ❌ Market {market} NOT found in markets")
            print(f"   Sample markets: {list(ex.markets.keys())[:5]}")
            
            # Try alternative markets
            alt_markets = [f"{symbol}/USD", f"{symbol}/BUSD"]
            for alt_market in alt_markets:
                if alt_market in ex.markets:
                    print(f"   ✅ Alternative market {alt_market} found")
                    market = alt_market
                    break
            else:
                print(f"   ❌ No alternative markets found either")
                return
        
        # Try to fetch OHLCV
        print(f"   Attempting to fetch OHLCV for {market}...")
        candles = ex.fetch_ohlcv(market, timeframe='4h', limit=200)
        
        if candles and len(candles) >= 50:
            print(f"   ✅ OHLCV fetch successful: {len(candles)} candles")
            latest_price = candles[-1][4]  # Close price
            print(f"   Latest price: ${latest_price:,.2f}")
        else:
            print(f"   ❌ OHLCV fetch failed or insufficient data")
            print(f"   Candles: {candles}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_fetcher_debug()
