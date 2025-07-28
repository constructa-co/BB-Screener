#!/usr/bin/env python3
"""
WYCKOFF ACCUMULATION/DISTRIBUTION SCANNER - 4H CHARTS
Professional scanner for Wyckoff methodology patterns
Strategy: Accumulation (Spring) → Markup | Distribution (Upthrust) → Markdown
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from scipy.signal import argrelextrema

class WyckoffScanner:
    def __init__(self, timeframe='4h'):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.timeframe = timeframe
        
    def get_top_coins(self, limit=150):
        """Get top coins by volume for Wyckoff scanning"""
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
                
                # Volume requirements for Wyckoff analysis (need institutional activity)
                if (volume_24h > 20_000_000 and      # Higher volume for institutional detection
                    market_cap > 300_000_000 and     # Larger caps for cleaner patterns
                    market_cap < 500_000_000_000):   # Focus on alt coins with institutional interest
                    coins.append(symbol)
            
            return coins[:80]  # Top 80 for Wyckoff analysis
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK']
    
    def get_binance_klines(self, symbol, interval='4h', limit=500):
        """Get klines from Binance for Wyckoff analysis (need more data for patterns)"""
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
            
            # Add technical indicators for Wyckoff analysis
            df = self.add_wyckoff_indicators(df)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'volume_sma']]
            
        except Exception as e:
            return None
    
    def add_wyckoff_indicators(self, df):
        """Add indicators needed for Wyckoff analysis"""
        # Simple moving averages for trend context
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Volume moving average for volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def identify_swing_points(self, df, lookback=8):
        """Identify swing highs and lows for Wyckoff structure (wider lookback for 4H)"""
        highs = []
        lows = []
        
        try:
            # Use scipy for more accurate swing detection
            high_indices = argrelextrema(df['high'].values, np.greater, order=lookback)[0]
            low_indices = argrelextrema(df['low'].values, np.less, order=lookback)[0]
            
            for idx in high_indices:
                if idx >= lookback and idx < len(df) - lookback:
                    highs.append({
                        'index': idx,
                        'price': df['high'].iloc[idx],
                        'timestamp': df['timestamp'].iloc[idx],
                        'volume': df['volume'].iloc[idx]
                    })
            
            for idx in low_indices:
                if idx >= lookback and idx < len(df) - lookback:
                    lows.append({
                        'index': idx,
                        'price': df['low'].iloc[idx],
                        'timestamp': df['timestamp'].iloc[idx],
                        'volume': df['volume'].iloc[idx]
                    })
                    
        except Exception:
            # Fallback to manual detection if scipy fails
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
                        'timestamp': df['timestamp'].iloc[i],
                        'volume': df['volume'].iloc[i]
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
                        'timestamp': df['timestamp'].iloc[i],
                        'volume': df['volume'].iloc[i]
                    })
        
        return highs, lows
    
    def detect_accumulation_phase(self, df, current_price, highs, lows):
        """Detect Wyckoff Accumulation Phase patterns"""
        accumulation_setups = []
        
        # Look for recent significant lows (potential support)
        recent_lows = [l for l in lows if l['index'] >= len(df) - 100]  # Last 100 candles (~16 days on 4H)
        
        if len(recent_lows) < 2:
            return accumulation_setups
        
        # Find potential trading ranges (accumulation zones)
        for i, primary_low in enumerate(recent_lows[:-1]):
            for secondary_low in recent_lows[i+1:]:
                
                # Check if we have a potential accumulation range
                low_1 = primary_low['price']
                low_2 = secondary_low['price']
                
                # Both lows should be within 5% of each other (accumulation range)
                price_diff_pct = abs(low_1 - low_2) / min(low_1, low_2) * 100
                if price_diff_pct > 5.0:
                    continue
                
                # Define accumulation range
                support_level = min(low_1, low_2)
                range_high_idx = max(primary_low['index'], secondary_low['index'])
                
                # Find the high of the trading range
                range_data = df.iloc[min(primary_low['index'], secondary_low['index']):range_high_idx + 20]
                if len(range_data) < 10:
                    continue
                
                resistance_level = range_data['high'].max()
                range_size_pct = (resistance_level - support_level) / support_level * 100
                
                # Range should be meaningful (3-20% for crypto)
                if range_size_pct < 3.0 or range_size_pct > 20.0:
                    continue
                
                # Look for Spring pattern (fake breakdown below support)
                spring_detected = False
                spring_strength = 0
                post_range_data = df.iloc[range_high_idx:]
                
                for idx, candle in post_range_data.iterrows():
                    # Check for spring (low below support but close above)
                    if (candle['low'] < support_level * 0.995 and  # Breaks support by 0.5%
                        candle['close'] > support_level):            # But closes back above
                        
                        spring_detected = True
                        
                        # Spring strength based on volume and rejection
                        volume_ratio = candle['volume'] / df['volume_sma'].iloc[idx] if pd.notna(df['volume_sma'].iloc[idx]) else 1
                        rejection_strength = (candle['close'] - candle['low']) / (candle['high'] - candle['low'])
                        
                        spring_strength = min(100, (volume_ratio * 30) + (rejection_strength * 40) + 30)
                        break
                
                # Check current position relative to accumulation
                current_position = "unknown"
                entry_signal = False
                
                if current_price <= support_level * 1.02:  # Within 2% of support
                    current_position = "at_support"
                    if spring_detected:
                        entry_signal = True
                elif support_level < current_price < resistance_level:
                    current_position = "in_range"  
                elif current_price >= resistance_level * 0.98:  # Within 2% of resistance
                    current_position = "at_resistance"
                elif current_price > resistance_level * 1.02:  # Above range
                    current_position = "markup_phase"
                    if spring_detected:  # Sign of Strength breakout
                        entry_signal = True
                
                # Calculate time in accumulation
                accumulation_duration = range_high_idx - min(primary_low['index'], secondary_low['index'])
                
                # Volume analysis during accumulation
                range_volume_data = df.iloc[min(primary_low['index'], secondary_low['index']):range_high_idx]
                avg_volume_in_range = range_volume_data['volume'].mean() if len(range_volume_data) > 0 else 0
                overall_avg_volume = df['volume'].mean()
                volume_ratio_range = avg_volume_in_range / overall_avg_volume if overall_avg_volume > 0 else 1
                
                # Setup quality scoring
                setup_score = 50  # Base score
                
                # Spring detection bonus
                if spring_detected:
                    setup_score += spring_strength * 0.3
                
                # Range quality bonus
                if 5 <= range_size_pct <= 15:  # Ideal range size
                    setup_score += 15
                elif 3 <= range_size_pct <= 20:  # Acceptable range
                    setup_score += 10
                
                # Volume confirmation bonus
                if volume_ratio_range > 1.2:  # Above average volume in range
                    setup_score += 10
                
                # Duration bonus (longer accumulation = stronger)
                if accumulation_duration >= 30:  # 30+ candles = 5+ days
                    setup_score += 15
                elif accumulation_duration >= 15:  # 15+ candles = 2.5+ days
                    setup_score += 10
                
                # Current position bonus
                if entry_signal:
                    setup_score += 20
                elif current_position in ["at_support", "markup_phase"]:
                    setup_score += 10
                
                setup_score = min(setup_score, 100)
                
                if setup_score >= 65:  # Quality threshold
                    
                    # Calculate targets
                    target_1 = resistance_level  # Break of range high
                    target_2 = resistance_level + (range_size_pct * 0.01 * resistance_level)  # Range projection
                    stop_loss = support_level * 0.98  # Just below support
                    
                    # Risk/reward calculation
                    if current_position == "markup_phase":
                        entry_price = current_price
                    else:
                        entry_price = support_level * 1.01  # Just above support
                    
                    risk_pct = abs(entry_price - stop_loss) / entry_price * 100
                    reward_1_pct = (target_1 - entry_price) / entry_price * 100
                    reward_2_pct = (target_2 - entry_price) / entry_price * 100
                    
                    risk_reward_1 = reward_1_pct / risk_pct if risk_pct > 0 else 0
                    risk_reward_2 = reward_2_pct / risk_pct if risk_pct > 0 else 0
                    
                    accumulation_setup = {
                        'type': 'wyckoff_accumulation',
                        'phase': 'accumulation',
                        'pattern': 'spring' if spring_detected else 'accumulation_range',
                        'support_level': support_level,
                        'resistance_level': resistance_level,
                        'current_position': current_position,
                        'entry_signal': entry_signal,
                        'entry_price': entry_price,
                        'target_1': target_1,
                        'target_2': target_2,
                        'stop_loss': stop_loss,
                        'risk_pct': risk_pct,
                        'reward_1_pct': reward_1_pct,
                        'reward_2_pct': reward_2_pct,
                        'risk_reward_1': risk_reward_1,
                        'risk_reward_2': risk_reward_2,
                        'range_size_pct': range_size_pct,
                        'spring_detected': spring_detected,
                        'spring_strength': spring_strength,
                        'accumulation_duration': accumulation_duration,
                        'volume_ratio': volume_ratio_range,
                        'setup_score': setup_score
                    }
                    
                    accumulation_setups.append(accumulation_setup)
        
        return accumulation_setups
    
    def detect_distribution_phase(self, df, current_price, highs, lows):
        """Detect Wyckoff Distribution Phase patterns"""
        distribution_setups = []
        
        # Look for recent significant highs (potential resistance)
        recent_highs = [h for h in highs if h['index'] >= len(df) - 100]  # Last 100 candles
        
        if len(recent_highs) < 2:
            return distribution_setups
        
        # Find potential distribution ranges
        for i, primary_high in enumerate(recent_highs[:-1]):
            for secondary_high in recent_highs[i+1:]:
                
                # Check if we have a potential distribution range
                high_1 = primary_high['price']
                high_2 = secondary_high['price']
                
                # Both highs should be within 5% of each other (distribution range)
                price_diff_pct = abs(high_1 - high_2) / max(high_1, high_2) * 100
                if price_diff_pct > 5.0:
                    continue
                
                # Define distribution range
                resistance_level = max(high_1, high_2)
                range_low_idx = max(primary_high['index'], secondary_high['index'])
                
                # Find the low of the trading range
                range_data = df.iloc[min(primary_high['index'], secondary_high['index']):range_low_idx + 20]
                if len(range_data) < 10:
                    continue
                
                support_level = range_data['low'].min()
                range_size_pct = (resistance_level - support_level) / support_level * 100
                
                # Range should be meaningful (3-20% for crypto)
                if range_size_pct < 3.0 or range_size_pct > 20.0:
                    continue
                
                # Look for Upthrust pattern (fake breakout above resistance)
                upthrust_detected = False
                upthrust_strength = 0
                post_range_data = df.iloc[range_low_idx:]
                
                for idx, candle in post_range_data.iterrows():
                    # Check for upthrust (high above resistance but close below)
                    if (candle['high'] > resistance_level * 1.005 and  # Breaks resistance by 0.5%
                        candle['close'] < resistance_level):            # But closes back below
                        
                        upthrust_detected = True
                        
                        # Upthrust strength based on volume and rejection
                        volume_ratio = candle['volume'] / df['volume_sma'].iloc[idx] if pd.notna(df['volume_sma'].iloc[idx]) else 1
                        rejection_strength = (candle['high'] - candle['close']) / (candle['high'] - candle['low'])
                        
                        upthrust_strength = min(100, (volume_ratio * 30) + (rejection_strength * 40) + 30)
                        break
                
                # Check current position relative to distribution
                current_position = "unknown"
                entry_signal = False
                
                if current_price >= resistance_level * 0.98:  # Within 2% of resistance
                    current_position = "at_resistance"
                    if upthrust_detected:
                        entry_signal = True
                elif support_level < current_price < resistance_level:
                    current_position = "in_range"
                elif current_price <= support_level * 1.02:  # Within 2% of support
                    current_position = "at_support"
                elif current_price < support_level * 0.98:  # Below range
                    current_position = "markdown_phase"
                    if upthrust_detected:  # Sign of Weakness breakdown
                        entry_signal = True
                
                # Calculate time in distribution
                distribution_duration = range_low_idx - min(primary_high['index'], secondary_high['index'])
                
                # Volume analysis during distribution
                range_volume_data = df.iloc[min(primary_high['index'], secondary_high['index']):range_low_idx]
                avg_volume_in_range = range_volume_data['volume'].mean() if len(range_volume_data) > 0 else 0
                overall_avg_volume = df['volume'].mean()
                volume_ratio_range = avg_volume_in_range / overall_avg_volume if overall_avg_volume > 0 else 1
                
                # Setup quality scoring
                setup_score = 50  # Base score
                
                # Upthrust detection bonus
                if upthrust_detected:
                    setup_score += upthrust_strength * 0.3
                
                # Range quality bonus
                if 5 <= range_size_pct <= 15:  # Ideal range size
                    setup_score += 15
                elif 3 <= range_size_pct <= 20:  # Acceptable range
                    setup_score += 10
                
                # Volume confirmation bonus (higher volume in distribution)
                if volume_ratio_range > 1.3:  # High volume distribution
                    setup_score += 15
                elif volume_ratio_range > 1.1:  # Above average volume
                    setup_score += 10
                
                # Duration bonus
                if distribution_duration >= 30:  # 30+ candles
                    setup_score += 15
                elif distribution_duration >= 15:  # 15+ candles
                    setup_score += 10
                
                # Current position bonus
                if entry_signal:
                    setup_score += 20
                elif current_position in ["at_resistance", "markdown_phase"]:
                    setup_score += 10
                
                setup_score = min(setup_score, 100)
                
                if setup_score >= 65:  # Quality threshold
                    
                    # Calculate targets
                    target_1 = support_level  # Break of range low
                    target_2 = support_level - (range_size_pct * 0.01 * support_level)  # Range projection
                    stop_loss = resistance_level * 1.02  # Just above resistance
                    
                    # Risk/reward calculation
                    if current_position == "markdown_phase":
                        entry_price = current_price
                    else:
                        entry_price = resistance_level * 0.99  # Just below resistance
                    
                    risk_pct = abs(stop_loss - entry_price) / entry_price * 100
                    reward_1_pct = (entry_price - target_1) / entry_price * 100
                    reward_2_pct = (entry_price - target_2) / entry_price * 100
                    
                    risk_reward_1 = reward_1_pct / risk_pct if risk_pct > 0 else 0
                    risk_reward_2 = reward_2_pct / risk_pct if risk_pct > 0 else 0
                    
                    distribution_setup = {
                        'type': 'wyckoff_distribution',
                        'phase': 'distribution',
                        'pattern': 'upthrust' if upthrust_detected else 'distribution_range',
                        'support_level': support_level,
                        'resistance_level': resistance_level,
                        'current_position': current_position,
                        'entry_signal': entry_signal,
                        'entry_price': entry_price,
                        'target_1': target_1,
                        'target_2': target_2,
                        'stop_loss': stop_loss,
                        'risk_pct': risk_pct,
                        'reward_1_pct': reward_1_pct,
                        'reward_2_pct': reward_2_pct,
                        'risk_reward_1': risk_reward_1,
                        'risk_reward_2': risk_reward_2,
                        'range_size_pct': range_size_pct,
                        'upthrust_detected': upthrust_detected,
                        'upthrust_strength': upthrust_strength,
                        'distribution_duration': distribution_duration,
                        'volume_ratio': volume_ratio_range,
                        'setup_score': setup_score
                    }
                    
                    distribution_setups.append(distribution_setup)
        
        return distribution_setups
    
    def find_wyckoff_setups(self, df, current_price):
        """Find all Wyckoff accumulation and distribution setups"""
        # Get swing points
        highs, lows = self.identify_swing_points(df)
        
        setups = []
        
        # Detect accumulation patterns
        accumulation_setups = self.detect_accumulation_phase(df, current_price, highs, lows)
        setups.extend(accumulation_setups)
        
        # Detect distribution patterns  
        distribution_setups = self.detect_distribution_phase(df, current_price, highs, lows)
        setups.extend(distribution_setups)
        
        return setups
    
    def scan_for_wyckoff_setups(self):
        """Main scanning function for Wyckoff patterns"""
        print(f"🏛️ WYCKOFF METHODOLOGY SCANNER - 4H CHART")
        print("=" * 75)
        print(f"📈 Scanning 4H charts for Wyckoff Accumulation/Distribution patterns...")
        print("📊 Strategy: Spring → Markup | Upthrust → Markdown")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'setups_found': 0}
        
        for symbol in coins:
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get price data (more data needed for Wyckoff)
                df = self.get_binance_klines(symbol, interval=self.timeframe, limit=500)
                if df is None or len(df) < 100:
                    print("No data")
                    stats['no_data'] += 1
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Find Wyckoff patterns
                setups = self.find_wyckoff_setups(df, current_price)
                
                if not setups:
                    print("No patterns")
                    continue
                
                # Select best setup per symbol
                best_setup = max(setups, key=lambda x: x['setup_score'])
                best_setup['symbol'] = symbol
                best_setup['current_price'] = current_price
                
                opportunities.append(best_setup)
                
                # Quick display
                phase = best_setup['phase'].title()
                pattern = best_setup['pattern'].replace('_', ' ').title()
                entry_signal = "🎯 READY" if best_setup['entry_signal'] else "⏳ WAIT"
                direction = "📈 LONG" if phase == "Accumulation" else "📉 SHORT"
                
                print(f"{entry_signal} | {direction} | {pattern} | Score: {best_setup['setup_score']:.0f}")
                stats['setups_found'] += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error - {e}")
                continue
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {stats['scanned']}")
        print(f"No data: {stats['no_data']}")
        print(f"Wyckoff patterns: {stats['setups_found']}")
        
        # Sort by setup score
        opportunities.sort(key=lambda x: x['setup_score'], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display Wyckoff scanning results"""
        print("\n" + "=" * 90)
        print(f"🏛️ FOUND {len(opportunities)} WYCKOFF METHODOLOGY SETUPS (4H CHART)")
        print("=" * 90)
        
        if not opportunities:
            print("❌ No high-quality Wyckoff patterns found")
            print("💡 Wyckoff patterns need time to develop - try again later")
            return
        
        print("📊 Results (sorted by setup score):")
        print()
        
        for i, setup in enumerate(opportunities[:10], 1):
            # Setup info
            phase = setup['phase'].title()
            pattern = setup['pattern'].replace('_', ' ').title()
            direction = "LONG" if phase == "Accumulation" else "SHORT"
            direction_emoji = "📈" if direction == "LONG" else "📉"
            
            # Entry status
            status = '🎯 READY' if setup['entry_signal'] else '⏳ WAIT'
            
            print(f"{i:2}. {direction_emoji} {setup['symbol']:8} | Score: {setup['setup_score']:3.0f}/100 | "
                  f"🏛️ {pattern.upper()} | {status}")
            
            # Current position and levels
            print(f"     Current: ${setup['current_price']:.6f} | Position: {setup['current_position'].replace('_', ' ').title()}")
            print(f"     Range: ${setup['support_level']:.6f} - ${setup['resistance_level']:.6f} "
                  f"({setup['range_size_pct']:.1f}% range)")
            
            # Entry and targets
            print(f"     Entry: ${setup['entry_price']:.6f} | Stop: ${setup['stop_loss']:.6f} "
                  f"({setup['risk_pct']:.1f}% risk)")
            
            # Targets
            print(f"     Targets: T1: ${setup['target_1']:.6f} ({setup['reward_1_pct']:.1f}% | R:R {setup['risk_reward_1']:.1f}:1) | "
                  f"T2: ${setup['target_2']:.6f} ({setup['reward_2_pct']:.1f}% | R:R {setup['risk_reward_2']:.1f}:1)")
            
            # Pattern details
            if setup['phase'] == 'accumulation':
                spring_status = "Spring Detected ✅" if setup['spring_detected'] else "No Spring Yet"
                print(f"     Pattern: {spring_status} | Volume Ratio: {setup['volume_ratio']:.1f}x | "
                      f"Duration: {setup['accumulation_duration']} candles")
            else:
                upthrust_status = "Upthrust Detected ✅" if setup['upthrust_detected'] else "No Upthrust Yet"  
                print(f"     Pattern: {upthrust_status} | Volume Ratio: {setup['volume_ratio']:.1f}x | "
                      f"Duration: {setup['distribution_duration']} candles")
            
            # Trading guidance
            if setup['entry_signal']:
                if phase == "Accumulation":
                    print(f"     🎯 LONG SETUP: Spring/SOS confirmed - Enter on pullback to support")
                else:
                    print(f"     🎯 SHORT SETUP: Upthrust/SOW confirmed - Enter on bounce to resistance")
            else:
                if phase == "Accumulation":
                    print(f"     ⏳ ACCUMULATION: Wait for Spring or Sign of Strength above ${setup['resistance_level']:.6f}")
                else:
                    print(f"     ⏳ DISTRIBUTION: Wait for Upthrust or Sign of Weakness below ${setup['support_level']:.6f}")
            
            print()
        
        print("🏛️ WYCKOFF METHODOLOGY STRATEGY:")
        print("• ACCUMULATION: Smart money accumulating → Spring (fake breakdown) → Markup")
        print("• DISTRIBUTION: Smart money distributing → Upthrust (fake breakout) → Markdown")
        print("• Entry: After Spring/Upthrust confirmation or on SOS/SOW breakouts")
        print("• Targets: Range projection (1x range size beyond breakout)")
        print("• Hold Time: 1-8 weeks typical (institutional money moves)")
        print("• Success Rate: 75-85% when patterns complete with volume")
        print("• Best R:R: 3:1+ typical due to large institutional moves")

def main():
    scanner = WyckoffScanner(timeframe='4h')
    opportunities = scanner.scan_for_wyckoff_setups()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()