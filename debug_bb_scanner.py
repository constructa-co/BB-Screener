#!/usr/bin/env python3

import ccxt
import pandas as pd
import numpy as np
from modules.bb_detector import BBDetector

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators like ICT scanner"""
    # Volume metrics
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def test_bb_detection():
    print("🔍 Testing BB Detection Logic with CCXT...")
    
    # Initialize exchange like ICT scanner
    exchange = ccxt.kucoin({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    bb = BBDetector()
    
    print("✅ Components initialized")
    
    # Test on LRC/USDT (found by ICT scanner)
    symbol = 'LRC/USDT'
    
    print(f"\n📊 Testing {symbol}...")
    
    try:
        # Fetch data like ICT scanner
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=200)
        
        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Add indicators
        df = add_indicators(df)
        
        print(f"Data shape: {df.shape}")
        print(f"Latest price: {df['close'].iloc[-1]}")
        print(f"Data range: {df.index[0]} to {df.index[-1]}")
        print(f"Volume ratio: {df['volume_ratio'].iloc[-1]:.2f}")
        
        # Test BB analysis (fixed method call)
        result = bb.analyze_bb_setup(df)
        print(f"\n🔍 BB Analysis Result:")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        if result:
            print(f"\n📈 Setup Details:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No setup found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bb_detection()
