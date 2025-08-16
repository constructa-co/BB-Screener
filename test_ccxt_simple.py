#!/usr/bin/env python3
"""
Simple ccxt test to see if the issue is with our data fetcher or ccxt itself
"""

import ccxt
import pandas as pd

def test_ccxt_direct():
    print("🔍 Testing ccxt directly...")
    
    # Test different exchanges
    exchanges = ['bybit', 'kucoin', 'okx']
    
    for exchange_name in exchanges:
        print(f"\n📊 Testing {exchange_name}...")
        
        try:
            # Initialize exchange
            exchange = getattr(ccxt, exchange_name)()
            print(f"   Exchange initialized: {exchange.id}")
            
            # Load markets
            markets = exchange.load_markets()
            print(f"   Markets loaded: {len(markets)} markets")
            
            # Check if BTC/USDT exists
            if 'BTC/USDT' in markets:
                print(f"   ✅ BTC/USDT found in markets")
                
                # Try to fetch OHLCV
                try:
                    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=10)
                    print(f"   ✅ OHLCV fetch successful: {len(ohlcv)} candles")
                    if ohlcv:
                        latest_price = ohlcv[-1][4]  # Close price
                        print(f"   Latest BTC price: ${latest_price:,.2f}")
                except Exception as e:
                    print(f"   ❌ OHLCV fetch failed: {e}")
            else:
                print(f"   ❌ BTC/USDT not found in markets")
                
        except Exception as e:
            print(f"   ❌ Exchange initialization failed: {e}")

if __name__ == "__main__":
    test_ccxt_direct()
