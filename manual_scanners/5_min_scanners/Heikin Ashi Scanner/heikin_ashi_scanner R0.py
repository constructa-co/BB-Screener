#!/usr/bin/env python3
"""
HEIKIN ASHI RANGE SCANNER - SIMPLIFIED FOR DEBUGGING
Simple standalone scanner to find coins perfect for manual scalping
Strategy: Wait for HA color change, enter trade, exit on next color change
Target: 2-3% moves in ranging markets, perfect for 5x leverage scalping
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class HeikinAshiRangeScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
    def get_top_coins(self, limit=200):
        """Get top coins by volume for scanning"""
        try:
            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"
            params = {
                'start': 1,
                'limit': limit,
                'sortBy': 'volume_24h',
                'sortType': 'desc',
                'convert': 'USD',
                'cryptoType': 'all',
                'tagType': 'all'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            coins = []
            for coin in data['data']['cryptoCurrencyList']:
                symbol = coin['symbol']
                volume_24h = coin['quotes'][0]['volume24h']
                market_cap = coin['quotes'][0]['marketCap']
                
                # Filter criteria for good scalping candidates
                if (volume_24h > 10_000_000 and  # $10M+ volume
                    market_cap > 50_000_000 and  # $50M+ market cap
                    market_cap < 20_000_000_000): # Under $20B (avoid BTC/ETH)
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 by volume
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            # Fallback list of popular trading pairs
            return ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC']
    
    def get_binance_klines(self, symbol, interval='5m', limit=100):
        """Get 5-minute klines from Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
                
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            return None
    
    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candles"""
        ha_df = df.copy()
        
        # Initialize first HA candle
        ha_df.loc[0, 'ha_close'] = (df.loc[0, 'open'] + df.loc[0, 'high'] + 
                                   df.loc[0, 'low'] + df.loc[0, 'close']) / 4
        ha_df.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2
        
        # Calculate subsequent HA candles
        for i in range(1, len(df)):
            ha_df.loc[i, 'ha_close'] = (df.loc[i, 'open'] + df.loc[i, 'high'] + 
                                       df.loc[i, 'low'] + df.loc[i, 'close']) / 4
            ha_df.loc[i, 'ha_open'] = (ha_df.loc[i-1, 'ha_open'] + ha_df.loc[i-1, 'ha_close']) / 2
            ha_df.loc[i, 'ha_high'] = max(df.loc[i, 'high'], ha_df.loc[i, 'ha_open'], ha_df.loc[i, 'ha_close'])
            ha_df.loc[i, 'ha_low'] = min(df.loc[i, 'low'], ha_df.loc[i, 'ha_open'], ha_df.loc[i, 'ha_close'])
        
        # Determine HA candle color (True = Green/Bullish, False = Red/Bearish)
        ha_df['ha_bullish'] = ha_df['ha_close'] > ha_df['ha_open']
        
        return ha_df
    
    def analyze_ranging_quality(self, df, symbol=""):
        """Analyze if the coin is in a steady ranging pattern - MUCH SIMPLER"""
        # Get last 40 candles
        recent_df = df.tail(40).copy()
        
        # Calculate price range
        high_40 = recent_df['high'].max()
        low_40 = recent_df['low'].min()
        range_pct = ((high_40 - low_40) / low_40) * 100
        current_price = df['close'].iloc[-1]
        
        # Tightened filters based on successful results
        if range_pct < 2.0 or range_pct > 6.0:  # Was 1.5-8%, now 2-6%
            return None
        
        # Calculate trend strength (was missing!)
        x = np.arange(len(recent_df))
        y = recent_df['close'].values
        slope, _ = np.polyfit(x, y, 1)
        trend_strength = abs(slope * len(recent_df) / y.mean()) * 100
        
        # Stricter trend check to get more horizontal movement
        if trend_strength > 2.0:  # Was 3%, now 2%
            return None
        
        # Better support/resistance requirements
        upper_zone = high_40 * 0.985  # Top 1.5%
        lower_zone = low_40 * 1.015   # Bottom 1.5%
        
        upper_touches = len(recent_df[recent_df['high'] >= upper_zone])
        lower_touches = len(recent_df[recent_df['low'] <= lower_zone])
        
        if upper_touches < 2 or lower_touches < 2:  # Need 2 touches each level
            return None
        
        # Stricter volume check
        if recent_df['volume'].std() / recent_df['volume'].mean() > 1.5:  # Was 2.0
            return None
        
        print(f"✅ {symbol}: PASSED - Range:{range_pct:.1f}% Trend:{trend_strength:.1f}%")
        
        return {
            'range_pct': range_pct,
            'trend_strength': trend_strength,
            'current_price': current_price,
            'high_40': high_40,
            'low_40': low_40,
            'upper_touches': upper_touches,
            'lower_touches': lower_touches
        }
    
    def analyze_heikin_ashi_setup(self, ha_df):
        """Analyze HA setup for trading opportunities"""
        if len(ha_df) < 10:
            return None
        
        # Get recent HA candles
        recent_ha = ha_df.tail(10)
        
        # Current and previous candle colors
        current_bullish = recent_ha['ha_bullish'].iloc[-1]
        prev_bullish = recent_ha['ha_bullish'].iloc[-2]
        
        # Check for color changes
        color_changes = 0
        for i in range(len(recent_ha) - 1):
            if recent_ha['ha_bullish'].iloc[i] != recent_ha['ha_bullish'].iloc[i+1]:
                color_changes += 1
        
        # Look for potential setups
        setup_type = None
        confidence = 50
        
        # Just changed color
        if current_bullish != prev_bullish:
            setup_type = "LONG" if current_bullish else "SHORT"
            confidence = 80
        else:
            setup_type = "LONG" if current_bullish else "SHORT"
            confidence = 40
        
        return {
            'setup_type': setup_type,
            'confidence': confidence,
            'current_bullish': current_bullish,
            'color_changes': color_changes
        }
    
    def scan_for_opportunities(self):
        """Main scanning function - simplified"""
        print("🎯 HEIKIN ASHI RANGE SCANNER - DEBUG MODE")
        print("=" * 60)
        print("🔍 Scanning with BALANCED filters for quality ranging...")
        print("🎯 Target: PENGU-style opportunities (2-4% ranges, minimal trend)")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'filtered': 0, 'passed': 0}
        
        for i, symbol in enumerate(coins):
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get price data
                df = self.get_binance_klines(symbol)
                if df is None or len(df) < 50:
                    print("No data")
                    stats['no_data'] += 1
                    continue
                
                # Calculate Heikin Ashi
                ha_df = self.calculate_heikin_ashi(df)
                
                # Analyze ranging quality
                range_analysis = self.analyze_ranging_quality(df, symbol)
                if range_analysis is None:
                    stats['filtered'] += 1
                    continue
                
                # Analyze HA setup
                ha_analysis = self.analyze_heikin_ashi_setup(ha_df)
                if ha_analysis is None:
                    print(f"❌ {symbol}: HA failed")
                    stats['filtered'] += 1
                    continue
                
                # Better scoring system
                score = 40  # Lower base score
                
                # Perfect range (2.0-4.0% like PENGU)
                if 2.0 <= range_analysis['range_pct'] <= 4.0:
                    score += 30  # High reward for perfect range
                elif range_analysis['range_pct'] <= 6.0:
                    score += 15  # Some reward for good range
                
                # Horizontal movement (critical)
                if range_analysis['trend_strength'] < 0.5:
                    score += 25  # Perfect horizontal
                elif range_analysis['trend_strength'] < 1.0:
                    score += 15  # Good horizontal
                elif range_analysis['trend_strength'] < 2.0:
                    score += 5   # Acceptable
                
                # Touch quality (need good S/R levels)
                total_touches = range_analysis['upper_touches'] + range_analysis['lower_touches']
                if total_touches >= 6:
                    score += 15  # Excellent level definition
                elif total_touches >= 4:
                    score += 10  # Good levels
                elif total_touches >= 3:
                    score += 5   # Acceptable
                
                # HA activity (want some color changes for scalping)
                if 2 <= ha_analysis['color_changes'] <= 4:
                    score += 10  # Perfect for scalping
                elif ha_analysis['color_changes'] >= 1:
                    score += 5   # Some activity
                
                # Only keep quality opportunities (higher threshold)
                if score < 75:  # Was no threshold, now 75/100
                    continue
                
                stats['passed'] += 1
                opportunities.append({
                    'symbol': symbol,
                    'score': score,
                    'setup_type': ha_analysis['setup_type'],
                    'confidence': ha_analysis['confidence'],
                    'range_pct': range_analysis['range_pct'],
                    'trend_strength': range_analysis['trend_strength'],
                    'current_price': range_analysis['current_price'],
                    'support': range_analysis['low_40'],
                    'resistance': range_analysis['high_40'],
                    'current_bullish': ha_analysis['current_bullish'],
                    'color_changes': ha_analysis['color_changes']
                })
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                continue
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {stats['scanned']}")
        print(f"No data: {stats['no_data']}")
        print(f"Filtered out: {stats['filtered']}")
        print(f"Passed filters: {stats['passed']}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display scanning results"""
        print("\n" + "=" * 80)
        print(f"🎯 FOUND {len(opportunities)} OPPORTUNITIES")
        print("=" * 80)
        
        if not opportunities:
            print("❌ No opportunities found")
            print("💡 Filters might be too strict, or market not ranging")
            return
        
        print("📊 Results (sorted by quality score):")
        print()
        
        for i, opp in enumerate(opportunities[:15], 1):
            status_emoji = "🟢" if opp['current_bullish'] else "🔴"
            setup_emoji = "📈" if opp['setup_type'] == 'LONG' else "📉"
            
            print(f"{i:2}. {setup_emoji} {opp['symbol']:8} | Score: {opp['score']:2}/100 | "
                  f"Range: {opp['range_pct']:.1f}% | Trend: {opp['trend_strength']:.1f}% | {status_emoji}")
            print(f"     Current: ${opp['current_price']:.6f} | "
                  f"Support: {opp['support']:.6f} | Resistance: {opp['resistance']:.6f}")
            print(f"     Setup: {opp['setup_type']} ({opp['confidence']}%) | "
                  f"HA Changes: {opp['color_changes']}")
            print()

def main():
    scanner = HeikinAshiRangeScanner()
    opportunities = scanner.scan_for_opportunities()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()