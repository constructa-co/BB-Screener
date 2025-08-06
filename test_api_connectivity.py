#!/usr/bin/env python3
"""
Test API Connectivity
Diagnose API key and connectivity issues
"""

import os
import requests
import ccxt
from dotenv import load_dotenv

def test_binance_api():
    """Test Binance API connectivity"""
    print("🔍 Testing Binance API...")
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Binance API keys not found in .env")
        return False
    
    try:
        # Test public API first
        print("📡 Testing public API...")
        response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=10)
        if response.status_code == 200:
            print("✅ Public API working")
            data = response.json()
            print(f"   BTC Price: ${float(data['price']):,.2f}")
        else:
            print(f"❌ Public API failed: {response.status_code}")
            return False
        
        # Test authenticated API
        print("🔐 Testing authenticated API...")
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'sandbox': False,
            'enableRateLimit': True
        })
        
        # Test account info
        try:
            account = exchange.fetch_balance()
            print("✅ Authenticated API working")
            print(f"   USDT Balance: {account.get('USDT', {}).get('free', 0):.2f}")
            return True
        except Exception as e:
            print(f"❌ Authenticated API failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_kucoin_api():
    """Test KuCoin API connectivity"""
    print("\n🔍 Testing KuCoin API...")
    
    load_dotenv()
    api_key = os.getenv('KUCOIN_API_KEY')
    
    if not api_key:
        print("❌ KuCoin API key not found in .env")
        return False
    
    try:
        # Test public API
        response = requests.get("https://api.kucoin.com/api/v1/market/orderbook?symbol=BTC-USDT", timeout=10)
        if response.status_code == 200:
            print("✅ KuCoin public API working")
            return True
        else:
            print(f"❌ KuCoin API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ KuCoin API test failed: {e}")
        return False

def test_bybit_api():
    """Test Bybit API connectivity"""
    print("\n🔍 Testing Bybit API...")
    
    try:
        # Test public API
        response = requests.get("https://api.bybit.com/v2/public/tickers?symbol=BTCUSDT", timeout=10)
        if response.status_code == 200:
            print("✅ Bybit public API working")
            return True
        else:
            print(f"❌ Bybit API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Bybit API test failed: {e}")
        return False

def test_data_fetcher():
    """Test the data fetcher module"""
    print("\n🔍 Testing Data Fetcher...")
    
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from modules.data_fetcher import MarketDataFetcher
        
        fetcher = MarketDataFetcher()
        print(f"✅ Data fetcher initialized")
        print(f"   Available exchanges: {list(fetcher.exchanges.keys())}")
        
        # Test data fetching
        for exchange_name in ['binance', 'kucoin', 'bybit']:
            if exchange_name in fetcher.exchanges:
                try:
                    df = fetcher.fetch_ohlcv(exchange_name, 'BTCUSDT', '4h')
                    if df is not None and not df.empty:
                        print(f"✅ {exchange_name}: Data fetched successfully ({len(df)} candles)")
                    else:
                        print(f"❌ {exchange_name}: No data returned")
                except Exception as e:
                    print(f"❌ {exchange_name}: Error - {e}")
            else:
                print(f"⚠️ {exchange_name}: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Data fetcher test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 API Connectivity Test Suite")
    print("=" * 50)
    
    results = {
        'binance': test_binance_api(),
        'kucoin': test_kucoin_api(),
        'bybit': test_bybit_api(),
        'data_fetcher': test_data_fetcher()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test}: {status}")
    
    working_apis = sum(results.values())
    total_apis = len(results)
    
    print(f"\n🎯 Summary: {working_apis}/{total_apis} APIs working")
    
    if working_apis >= 2:
        print("✅ Sufficient APIs working for backtest")
    else:
        print("❌ Need more working APIs for reliable backtest")

if __name__ == "__main__":
    main() 