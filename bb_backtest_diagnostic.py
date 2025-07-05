#!/usr/bin/env python3
"""
BB Backtest Diagnostic - Validate Results and Fix Issues
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_fetcher import MarketDataFetcher

class BBBacktestDiagnostic:
    """
    Diagnostic tool to validate BB backtest results
    """
    
    def __init__(self):
        print("🔍 BB BACKTEST DIAGNOSTIC TOOL")
        print("=" * 50)
        
        self.data_fetcher = MarketDataFetcher()
        
    def run_diagnostic(self):
        """Run comprehensive diagnostic"""
        
        print("\n🧪 DIAGNOSTIC TEST 1: BB TOUCH DETECTION")
        print("-" * 40)
        
        # Test with BTC only for detailed analysis
        symbol = 'BTC'
        exchange = 'binance'
        
        # Get recent data
        df = self.data_fetcher.fetch_ohlcv(exchange, symbol, '4h')
        if df is None:
            print("❌ Could not fetch BTC data")
            return
        
        print(f"✅ Fetched {len(df)} candles for {symbol}")
        
        # Test different BB touch tolerances
        self._test_bb_touch_tolerances(df, symbol)
        
        print("\n🧪 DIAGNOSTIC TEST 2: SUCCESS RATE THRESHOLDS")
        print("-" * 40)
        
        # Test different success definitions
        self._test_success_thresholds(df, symbol)
        
        print("\n🧪 DIAGNOSTIC TEST 3: CONFLUENCE FACTOR VALIDATION")
        print("-" * 40)
        
        # Test confluence factor calculations
        self._test_confluence_factors(df, symbol)
        
    def _test_bb_touch_tolerances(self, df: pd.DataFrame, symbol: str):
        """Test different BB touch detection tolerances"""
        
        tolerances = [0.1, 0.3, 0.5, 1.0, 2.0]  # Percentage tolerances
        
        for tolerance in tolerances:
            lower_touches = 0
            upper_touches = 0
            
            for i in range(10, len(df) - 5):
                current = df.iloc[i]
                
                # Lower band touches
                distance_to_lower = abs(current['low'] - current['bb_lower']) / current['bb_lower'] * 100
                if distance_to_lower <= tolerance:
                    lower_touches += 1
                
                # Upper band touches
                distance_to_upper = abs(current['high'] - current['bb_upper']) / current['bb_upper'] * 100
                if distance_to_upper <= tolerance:
                    upper_touches += 1
            
            total_touches = lower_touches + upper_touches
            print(f"   {tolerance:>4.1f}% tolerance: {total_touches:>4d} touches ({lower_touches:>3d} lower, {upper_touches:>3d} upper)")
    
    def _test_success_thresholds(self, df: pd.DataFrame, symbol: str):
        """Test different success rate definitions"""
        
        # Find some BB touches with 0.5% tolerance (reasonable)
        touches = []
        
        for i in range(10, len(df) - 20):
            current = df.iloc[i]
            
            # Check for lower band touch (potential LONG)
            distance_to_lower = abs(current['low'] - current['bb_lower']) / current['bb_lower'] * 100
            if distance_to_lower <= 0.5:
                touches.append({
                    'index': i,
                    'direction': 'LONG',
                    'entry_price': current['close'],
                    'timestamp': current.name
                })
            
            # Check for upper band touch (potential SHORT)
            distance_to_upper = abs(current['high'] - current['bb_upper']) / current['bb_upper'] * 100
            if distance_to_upper <= 0.5:
                touches.append({
                    'index': i,
                    'direction': 'SHORT',
                    'entry_price': current['close'],
                    'timestamp': current.name
                })
        
        print(f"   Found {len(touches)} BB touches with 0.5% tolerance")
        
        if len(touches) == 0:
            print("   ❌ No touches found - tolerance might be too strict")
            return
        
        # Test different success thresholds
        thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        for threshold in thresholds:
            successes = 0
            
            for touch in touches[:50]:  # Test first 50 for speed
                success = self._calculate_touch_success(df, touch, threshold)
                if success:
                    successes += 1
            
            success_rate = successes / min(50, len(touches)) * 100
            print(f"   {threshold:>4.1f}% threshold: {success_rate:>5.1f}% success rate")
    
    def _calculate_touch_success(self, df: pd.DataFrame, touch: dict, threshold: float) -> bool:
        """Calculate if a touch was successful with given threshold"""
        
        i = touch['index']
        direction = touch['direction']
        entry_price = touch['entry_price']
        
        # Look ahead 10 periods
        end_index = min(i + 10, len(df) - 1)
        future_data = df.iloc[i+1:end_index+1]
        
        if len(future_data) == 0:
            return False
        
        if direction == 'LONG':
            # For LONG: success = price goes up by threshold%
            max_gain = ((future_data['high'].max() - entry_price) / entry_price * 100)
            return max_gain >= threshold
        else:  # SHORT
            # For SHORT: success = price goes down by threshold%
            max_gain = ((entry_price - future_data['low'].min()) / entry_price * 100)
            return max_gain >= threshold
    
    def _test_confluence_factors(self, df: pd.DataFrame, symbol: str):
        """Test confluence factor calculations"""
        
        print("   Testing confluence factor calculations...")
        
        # Test on recent data
        recent_data = df.tail(20)
        
        for i, (timestamp, row) in enumerate(recent_data.iterrows()):
            if i < 10:  # Need history for divergence
                continue
                
            # Test RSI
            rsi = row.get('rsi', 50)
            rsi_oversold = rsi < 30
            rsi_overbought = rsi > 70
            
            # Test Volume
            volume_ratio = row.get('volume_ratio', 1.0)
            volume_surge = volume_ratio > 1.5
            
            # Test Stochastic
            stoch_k = row.get('stoch_k', 50)
            stoch_oversold = stoch_k < 20
            stoch_overbought = stoch_k > 80
            
            print(f"   {timestamp.strftime('%m-%d %H:%M')}: RSI={rsi:5.1f} Vol={volume_ratio:4.1f}x Stoch={stoch_k:5.1f}")
            print(f"                     Oversold: RSI={rsi_oversold} Stoch={stoch_oversold} VolSurge={volume_surge}")
        
        # Test divergence on last 10 periods
        print("\n   Testing divergence calculations...")
        test_data = df.tail(10)
        
        if 'rsi' in test_data.columns and 'macd' in test_data.columns:
            # Simple divergence test
            price_trend = test_data['close'].iloc[-1] > test_data['close'].iloc[0]
            rsi_trend = test_data['rsi'].iloc[-1] > test_data['rsi'].iloc[0]
            macd_trend = test_data['macd'].iloc[-1] > test_data['macd'].iloc[0]
            
            rsi_divergence = price_trend != rsi_trend
            macd_divergence = price_trend != macd_trend
            
            print(f"   Price trend: {'UP' if price_trend else 'DOWN'}")
            print(f"   RSI trend: {'UP' if rsi_trend else 'DOWN'} (Divergence: {rsi_divergence})")
            print(f"   MACD trend: {'UP' if macd_trend else 'DOWN'} (Divergence: {macd_divergence})")
        else:
            print("   ❌ RSI/MACD data not available")
    
    def suggest_improvements(self):
        """Suggest improvements based on diagnostic"""
        
        print("\n🎯 RECOMMENDED IMPROVEMENTS")
        print("=" * 40)
        
        print("1. 🔧 TIGHTEN BB TOUCH DETECTION:")
        print("   - Use 0.3% tolerance instead of 1.0%")
        print("   - Add bounce confirmation (price moves away from band)")
        
        print("\n2. 📊 ADJUST SUCCESS THRESHOLD:")
        print("   - Use 1.0% or 1.5% instead of 2.0%")
        print("   - Consider direction-specific thresholds")
        
        print("\n3. 🧠 FIX CONFLUENCE FACTORS:")
        print("   - Improve divergence calculation (use proper peaks/troughs)")
        print("   - Fix pattern detection (use actual candlestick patterns)")
        print("   - Validate volume surge threshold")
        
        print("\n4. ✅ VALIDATE AGAINST KNOWN RESULTS:")
        print("   - Test with manual BB bounce examples")
        print("   - Compare with your scanner's 65-78% probability trades")
        print("   - Ensure confluence improves success rates")

if __name__ == "__main__":
    diagnostic = BBBacktestDiagnostic()
    diagnostic.run_diagnostic()
    diagnostic.suggest_improvements()