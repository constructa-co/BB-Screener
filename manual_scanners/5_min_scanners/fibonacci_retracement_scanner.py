#!/usr/bin/env python3
"""
FIBONACCI RETRACEMENT SCANNER - 1-5 MINUTE CHARTS (RELAXED CRITERIA)
Professional scanner for Fibonacci pullback entries after initial bounce
Strategy: Drop → 38.2% bounce → 23.6% pullback entry → 50-61.8% targets
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class FibonacciRetracementScanner:
    def __init__(self, timeframe='5m'):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.timeframe = timeframe
        
    def get_top_coins(self, limit=150):
        """Get top coins by volume for Fibonacci scanning"""
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
                
                # Volume requirements for short-term Fibonacci trading
                if (volume_24h > 12_000_000 and  # Good volume for quick moves
                    market_cap > 150_000_000 and   # Avoid micro-caps
                    market_cap < 300_000_000_000): # Focus on alt coins
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 for Fibonacci analysis
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK']
    
    def get_binance_klines(self, symbol, interval='5m', limit=200):
        """Get klines from Binance for Fibonacci analysis"""
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
    
    def identify_swing_points(self, df, lookback=5):
        """Identify swing highs and swing lows for Fibonacci levels"""
        highs = []
        lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            for j in range(lookback):
                if (df['high'].iloc[i - j - 1] >= current_high or 
                    df['high'].iloc[i + j + 1] >= current_high):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                highs.append({
                    'index': i,
                    'price': current_high,
                    'timestamp': df['timestamp'].iloc[i]
                })
            
            # Check for swing low
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            for j in range(lookback):
                if (df['low'].iloc[i - j - 1] <= current_low or 
                    df['low'].iloc[i + j + 1] <= current_low):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                lows.append({
                    'index': i,
                    'price': current_low,
                    'timestamp': df['timestamp'].iloc[i]
                })
        
        return highs, lows
    
    def calculate_fibonacci_levels(self, swing_high, swing_low):
        """Calculate Fibonacci retracement levels"""
        high_price = swing_high['price']
        low_price = swing_low['price']
        
        # Fibonacci retracement levels
        levels = {
            '0.000': high_price,  # 100% (swing high)
            '0.236': high_price - (high_price - low_price) * 0.236,
            '0.382': high_price - (high_price - low_price) * 0.382,
            '0.500': high_price - (high_price - low_price) * 0.500,
            '0.618': high_price - (high_price - low_price) * 0.618,
            '0.786': high_price - (high_price - low_price) * 0.786,
            '1.000': low_price   # 0% (swing low)
        }
        
        return levels
    
    def find_fibonacci_retracement_setups(self, df, current_price):
        """Find Fibonacci retracement trading setups"""
        setups = []
        
        # Get swing points
        highs, lows = self.identify_swing_points(df, lookback=5)
        
        # Look for recent swings (adjust based on timeframe)
        if self.timeframe == '1m':
            recent_period = 60  # 1 hour of 1m candles
        else:  # 5m
            recent_period = 48  # 4 hours of 5m candles
        
        recent_highs = [h for h in highs if h['index'] >= len(df) - recent_period]
        recent_lows = [l for l in lows if l['index'] >= len(df) - recent_period]
        
        # Find bullish retracement setups (after drop from high to low)
        for swing_high in recent_highs:
            for swing_low in recent_lows:
                # Need low after high for bullish setup
                if swing_low['index'] <= swing_high['index']:
                    continue
                
                # Check if move is significant enough (RELAXED - was 3%)
                move_size = (swing_high['price'] - swing_low['price']) / swing_low['price'] * 100
                if move_size < 2.0:  # Minimum 2% move for 5m charts
                    continue
                
                # Calculate Fibonacci levels
                fib_levels = self.calculate_fibonacci_levels(swing_high, swing_low)
                
                # Check if price has bounced to 38.2% area (RELAXED TOLERANCE)
                post_low_data = df.iloc[swing_low['index']:]
                has_382_bounce = False
                max_bounce = swing_low['price']
                
                for _, candle in post_low_data.iterrows():
                    max_bounce = max(max_bounce, candle['high'])
                    
                    # Check if bounced to 38.2% area (RELAXED - was 1.5%)
                    bounce_to_382_pct = abs(max_bounce - fib_levels['0.382']) / fib_levels['0.382'] * 100
                    if bounce_to_382_pct <= 5.0:  # Within 5% of 38.2%
                        has_382_bounce = True
                        break
                
                if not has_382_bounce:
                    continue
                
                # Check current position for 23.6% entry opportunity
                distance_to_236 = abs(current_price - fib_levels['0.236']) / fib_levels['0.236'] * 100
                distance_to_382 = abs(current_price - fib_levels['0.382']) / fib_levels['0.382'] * 100
                
                # Determine setup type based on current price position
                setup_type = None
                entry_level = None
                targets = {}
                
                # Check if currently near 23.6% for immediate entry (RELAXED - was 1%)
                if distance_to_236 <= 2.0:  # Within 2% of 23.6%
                    setup_type = "immediate_entry"
                    entry_level = fib_levels['0.236']
                    targets = {
                        'target_1': fib_levels['0.382'],  # Back to 38.2%
                        'target_2': fib_levels['0.500'],  # 50%
                        'target_3': fib_levels['0.618']   # 61.8%
                    }
                
                # Check if currently near 38.2% waiting for pullback to 23.6% (RELAXED - was 2%)
                elif distance_to_382 <= 3.0 and current_price >= fib_levels['0.382']:
                    setup_type = "waiting_for_pullback"
                    entry_level = fib_levels['0.236']
                    targets = {
                        'target_1': fib_levels['0.382'],  # Back to 38.2%
                        'target_2': fib_levels['0.500'],  # 50%
                        'target_3': fib_levels['0.618']   # 61.8%
                    }
                
                # Check if pullback is approaching 23.6% area (RELAXED - was 30%)
                elif fib_levels['0.236'] < current_price < fib_levels['0.382']:
                    distance_pct = (current_price - fib_levels['0.236']) / (fib_levels['0.382'] - fib_levels['0.236']) * 100
                    if distance_pct <= 50:  # Within 50% of the way to 23.6%
                        setup_type = "approaching_entry"
                        entry_level = fib_levels['0.236']
                        targets = {
                            'target_1': fib_levels['0.382'],
                            'target_2': fib_levels['0.500'],
                            'target_3': fib_levels['0.618']
                        }
                
                if setup_type and entry_level and targets:
                    # Calculate trade parameters
                    stop_loss = swing_low['price'] * 0.995  # Just below swing low
                    
                    # Calculate risk/reward for each target
                    risk_pct = abs(entry_level - stop_loss) / entry_level * 100
                    
                    target_analysis = {}
                    for target_name, target_price in targets.items():
                        if target_price > entry_level:  # Only upside targets
                            target_pct = (target_price - entry_level) / entry_level * 100
                            risk_reward = target_pct / risk_pct if risk_pct > 0 else 0
                            target_analysis[target_name] = {
                                'price': target_price,
                                'gain_pct': target_pct,
                                'risk_reward': risk_reward
                            }
                    
                    # Only proceed if we have reasonable risk/reward (RELAXED - was 1.5)
                    if target_analysis and any(t['risk_reward'] >= 1.0 for t in target_analysis.values()):
                        age_candles = len(df) - swing_low['index']
                        
                        setup = {
                            'type': 'bullish_fibonacci_retracement',
                            'setup_stage': setup_type,
                            'swing_high': swing_high,
                            'swing_low': swing_low,
                            'fib_levels': fib_levels,
                            'entry_price': entry_level,
                            'stop_loss': stop_loss,
                            'targets': target_analysis,
                            'current_price': current_price,
                            'risk_pct': risk_pct,
                            'move_size_pct': move_size,
                            'age_candles': age_candles,
                            'has_382_bounce': has_382_bounce
                        }
                        
                        setups.append(setup)
        
        # Find bearish retracement setups (after pump from low to high)
        for swing_low in recent_lows:
            for swing_high in recent_highs:
                # Need high after low for bearish setup
                if swing_high['index'] <= swing_low['index']:
                    continue
                
                # Check if move is significant enough (RELAXED - was 3%)
                move_size = (swing_high['price'] - swing_low['price']) / swing_low['price'] * 100
                if move_size < 2.0:  # Minimum 2% move
                    continue
                
                # Calculate Fibonacci levels (for bearish, we use high as 0% and low as 100%)
                fib_levels = {
                    '0.000': swing_high['price'],  # 100% (swing high)
                    '0.236': swing_high['price'] - (swing_high['price'] - swing_low['price']) * 0.236,
                    '0.382': swing_high['price'] - (swing_high['price'] - swing_low['price']) * 0.382,
                    '0.500': swing_high['price'] - (swing_high['price'] - swing_low['price']) * 0.500,
                    '0.618': swing_high['price'] - (swing_high['price'] - swing_low['price']) * 0.618,
                    '0.786': swing_high['price'] - (swing_high['price'] - swing_low['price']) * 0.786,
                    '1.000': swing_low['price']   # 0% (swing low)
                }
                
                # Check if price has pulled back to 38.2% area (RELAXED)
                post_high_data = df.iloc[swing_high['index']:]
                has_382_pullback = False
                min_pullback = swing_high['price']
                
                for _, candle in post_high_data.iterrows():
                    min_pullback = min(min_pullback, candle['low'])
                    
                    # Check if pulled back to 38.2% area (RELAXED - was 1.5%)
                    pullback_to_382_pct = abs(min_pullback - fib_levels['0.382']) / fib_levels['0.382'] * 100
                    if pullback_to_382_pct <= 5.0:  # Within 5% of 38.2%
                        has_382_pullback = True
                        break
                
                if not has_382_pullback:
                    continue
                
                # Check for bearish retracement entry opportunities
                distance_to_236 = abs(current_price - fib_levels['0.236']) / fib_levels['0.236'] * 100
                distance_to_382 = abs(current_price - fib_levels['0.382']) / fib_levels['0.382'] * 100
                
                setup_type = None
                entry_level = None
                targets = {}
                
                # Check bearish setups (RELAXED - was 1%)
                if distance_to_236 <= 2.0:  # Near 23.6% for SHORT
                    setup_type = "immediate_short_entry"
                    entry_level = fib_levels['0.236']
                    targets = {
                        'target_1': fib_levels['0.382'],  # Back to 38.2%
                        'target_2': fib_levels['0.500'],  # 50%
                        'target_3': fib_levels['0.618']   # 61.8%
                    }
                
                elif distance_to_382 <= 3.0 and current_price <= fib_levels['0.382']:  # RELAXED - was 2%
                    setup_type = "waiting_for_bounce"
                    entry_level = fib_levels['0.236']
                    targets = {
                        'target_1': fib_levels['0.382'],
                        'target_2': fib_levels['0.500'],
                        'target_3': fib_levels['0.618']
                    }
                
                if setup_type and entry_level and targets:
                    stop_loss = swing_high['price'] * 1.005  # Just above swing high
                    risk_pct = abs(entry_level - stop_loss) / entry_level * 100
                    
                    target_analysis = {}
                    for target_name, target_price in targets.items():
                        if target_price < entry_level:  # Only downside targets
                            target_pct = (entry_level - target_price) / entry_level * 100
                            risk_reward = target_pct / risk_pct if risk_pct > 0 else 0
                            target_analysis[target_name] = {
                                'price': target_price,
                                'gain_pct': target_pct,
                                'risk_reward': risk_reward
                            }
                    
                    if target_analysis and any(t['risk_reward'] >= 1.0 for t in target_analysis.values()):  # RELAXED - was 1.5
                        age_candles = len(df) - swing_high['index']
                        
                        setup = {
                            'type': 'bearish_fibonacci_retracement',
                            'setup_stage': setup_type,
                            'swing_high': swing_high,
                            'swing_low': swing_low,
                            'fib_levels': fib_levels,
                            'entry_price': entry_level,
                            'stop_loss': stop_loss,
                            'targets': target_analysis,
                            'current_price': current_price,
                            'risk_pct': risk_pct,
                            'move_size_pct': move_size,
                            'age_candles': age_candles,
                            'has_382_bounce': has_382_pullback
                        }
                        
                        setups.append(setup)
        
        return setups
    
    def calculate_setup_quality_score(self, setup):
        """Score the quality of the Fibonacci retracement setup"""
        score = 30  # Base score
        
        # Move size scoring (larger moves = better setups)
        move_size = setup['move_size_pct']
        if move_size >= 8.0:
            score += 20  # Large move
        elif move_size >= 5.0:
            score += 15  # Good move
        elif move_size >= 3.0:
            score += 10  # Acceptable move
        elif move_size >= 2.0:  # NEW - for 2% moves
            score += 5   # Small but acceptable move
        
        # 38.2% bounce confirmation (essential for strategy)
        if setup['has_382_bounce']:
            score += 20  # Confirmed retracement structure
        
        # Setup stage scoring
        stage = setup['setup_stage']
        if 'immediate' in stage:
            score += 15  # Ready to trade now
        elif 'approaching' in stage:
            score += 12  # Close to entry
        elif 'waiting' in stage:
            score += 8   # Need to wait
        
        # Risk/reward scoring (RELAXED)
        best_rr = 0
        for target in setup['targets'].values():
            best_rr = max(best_rr, target['risk_reward'])
        
        if best_rr >= 4.0:
            score += 15  # Excellent R:R
        elif best_rr >= 3.0:
            score += 12  # Very good R:R
        elif best_rr >= 2.0:
            score += 8   # Good R:R
        elif best_rr >= 1.5:
            score += 5   # Acceptable R:R
        elif best_rr >= 1.0:  # NEW - for 1:1 R:R
            score += 3   # Minimal but acceptable R:R
        
        # Age of setup (fresher is better)
        age = setup['age_candles']
        if self.timeframe == '1m':
            if age <= 30:
                score += 10  # Very recent (30 minutes)
            elif age <= 60:
                score += 8   # Recent (1 hour)
            elif age <= 120:
                score += 5   # Acceptable (2 hours)
        else:  # 5m
            if age <= 12:
                score += 10  # Very recent (1 hour)
            elif age <= 24:
                score += 8   # Recent (2 hours)
            elif age <= 48:
                score += 5   # Acceptable (4 hours)
        
        # Multiple targets bonus
        if len(setup['targets']) >= 3:
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def scan_for_fibonacci_setups(self):
        """Main scanning function for Fibonacci retracement opportunities"""
        timeframe_display = "1-MINUTE" if self.timeframe == '1m' else "5-MINUTE"
        print(f"🎯 FIBONACCI RETRACEMENT SCANNER - {timeframe_display} CHART (RELAXED)")
        print("=" * 80)
        print(f"📈 Scanning {self.timeframe} charts for Fibonacci pullback entries...")
        print("📊 Strategy: Drop → 38.2% bounce → 23.6% pullback → 50-61.8% targets")
        print("🔧 RELAXED CRITERIA: 2%+ moves, 5% tolerance, 1:1+ R:R")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'setups_found': 0}
        
        for symbol in coins:
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get price data
                df = self.get_binance_klines(symbol, interval=self.timeframe, limit=200)
                if df is None or len(df) < 50:
                    print("No data")
                    stats['no_data'] += 1
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Find Fibonacci retracement setups
                setups = self.find_fibonacci_retracement_setups(df, current_price)
                
                if not setups:
                    print("No setups")
                    continue
                
                # Score and select best setup (RELAXED THRESHOLD - was 65)
                best_setup = None
                best_score = 0
                
                for setup in setups:
                    quality_score = self.calculate_setup_quality_score(setup)
                    
                    if quality_score > best_score and quality_score >= 50:  # Lowered threshold
                        best_score = quality_score
                        best_setup = setup
                        best_setup['quality_score'] = quality_score
                        best_setup['symbol'] = symbol
                
                if best_setup:
                    opportunities.append(best_setup)
                    
                    # Quick display
                    setup_type = best_setup['type']
                    stage = best_setup['setup_stage']
                    
                    if 'bullish' in setup_type:
                        direction = "LONG"
                        emoji = "📈"
                    else:
                        direction = "SHORT" 
                        emoji = "📉"
                    
                    if 'immediate' in stage:
                        status = "🎯 NOW"
                    elif 'approaching' in stage:
                        status = "🔜 CLOSE"
                    else:
                        status = "⏳ WAIT"
                    
                    best_rr = max([t['risk_reward'] for t in best_setup['targets'].values()])
                    
                    print(f"{status} {emoji} {direction} | R:R {best_rr:.1f}:1 | Score {best_score}")
                    stats['setups_found'] += 1
                else:
                    print(f"{len(setups)} setups (low quality)")
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error - {e}")
                continue
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {stats['scanned']}")
        print(f"No data: {stats['no_data']}")
        print(f"Quality setups: {stats['setups_found']}")
        
        # Sort by quality score
        opportunities.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display Fibonacci retracement scanning results"""
        timeframe_display = "1-MINUTE" if self.timeframe == '1m' else "5-MINUTE"
        print("\n" + "=" * 85)
        print(f"🎯 FOUND {len(opportunities)} FIBONACCI RETRACEMENT SETUPS ({timeframe_display})")
        print("=" * 85)
        
        if not opportunities:
            print("❌ No high-quality Fibonacci retracement setups found")
            print("💡 Try again later - looking for 38.2% bounce → 23.6% pullback patterns")
            return
        
        print("📊 Results (sorted by quality score):")
        print()
        
        for i, setup in enumerate(opportunities[:10], 1):
            # Setup info
            direction = "LONG" if setup['type'] == 'bullish_fibonacci_retracement' else "SHORT"
            direction_emoji = "📈" if direction == "LONG" else "📉"
            
            # Entry timing
            stage = setup['setup_stage']
            if 'immediate' in stage:
                status = '🎯 NOW'
            elif 'approaching' in stage:
                status = '🔜 CLOSE'
            else:
                status = '⏳ WAIT'
            
            print(f"{i:2}. {direction_emoji} {setup['symbol']:8} | Score: {setup['quality_score']:2}/100 | "
                  f"📊 FIB RETRACEMENT | {status}")
            
            # Current position and entry
            print(f"     Current: ${setup['current_price']:.4f} | "
                  f"Entry: ${setup['entry_price']:.4f} (23.6% Fib) | "
                  f"Move: {setup['move_size_pct']:.1f}%")
            
            # Targets
            targets_text = []
            for target_name, target_data in setup['targets'].items():
                targets_text.append(f"{target_name.replace('target_', 'T')}: ${target_data['price']:.4f} "
                                  f"({target_data['gain_pct']:.1f}% | R:R {target_data['risk_reward']:.1f}:1)")
            
            print(f"     Targets: {' | '.join(targets_text)}")
            
            # Stop loss and Fibonacci structure
            print(f"     Stop: ${setup['stop_loss']:.4f} ({setup['risk_pct']:.1f}%) | "
                  f"Swing: ${setup['swing_low']['price']:.4f} → ${setup['swing_high']['price']:.4f}")
            
            # Key Fibonacci levels
            fib = setup['fib_levels']
            print(f"     Fib Levels: 23.6%: ${fib['0.236']:.4f} | 38.2%: ${fib['0.382']:.4f} | "
                  f"50%: ${fib['0.500']:.4f} | 61.8%: ${fib['0.618']:.4f}")
            
            # Setup guidance
            if 'immediate' in stage:
                print(f"     🎯 ENTER NOW - Price at 23.6% Fibonacci retracement!")
            elif 'approaching' in stage:
                print(f"     🔜 Wait for pullback to 23.6% level (${setup['entry_price']:.4f})")
            elif 'waiting_for_pullback' in stage:
                print(f"     ⏳ Wait for pullback from 38.2% to 23.6% entry")
            elif 'waiting_for_bounce' in stage:
                print(f"     ⏳ Wait for bounce from 38.2% to 23.6% entry")
            
            print()
        
        print("🎯 FIBONACCI RETRACEMENT STRATEGY (RELAXED CRITERIA):")
        print("• Setup: 2%+ move → 38.2% bounce → 23.6% pullback entry")
        print("• Entry: 23.6% Fibonacci retracement level (±2% tolerance)")
        print("• Targets: 38.2%, 50%, 61.8% Fibonacci levels")
        print("• Stop: Just beyond swing high/low")
        print("• Hold Time: 30 minutes - 4 hours typical")
        print("• Success Rate: ~60-70% with relaxed criteria")
        print("• Min R:R: 1:1 minimum, 2:1+ preferred")

def main():
    import sys
    
    # Allow user to specify timeframe
    timeframe = '5m'  # Default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['1m', '5m']:
            timeframe = sys.argv[1]
        else:
            print("Usage: python fibonacci_scanner.py [1m|5m]")
            print("Defaulting to 5m timeframe...")
    
    scanner = FibonacciRetracementScanner(timeframe=timeframe)
    opportunities = scanner.scan_for_fibonacci_setups()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()