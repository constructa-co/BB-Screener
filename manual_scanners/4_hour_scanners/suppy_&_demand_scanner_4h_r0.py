#!/usr/bin/env python3
"""
SUPPLY & DEMAND ZONE SCANNER - 4H CHARTS
Professional scanner for institutional supply and demand levels
Strategy: Fresh zones = 70-80% success rate | Aged zones = 50-60% success rate
Timeframe: 4H optimized for swing trading (1-4 week holds, 15-50% targets)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from scipy.signal import argrelextrema

class SupplyDemandScanner4H:
    def __init__(self, timeframe='4h'):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.timeframe = timeframe
        
    def get_top_coins(self, limit=120):
        """Get top coins by volume for Supply & Demand scanning"""
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
                
                # Volume requirements for institutional S&D detection
                if (volume_24h > 18_000_000 and      # Need institutional activity
                    market_cap > 250_000_000 and     # Sufficient size for clear levels
                    market_cap < 400_000_000_000):   # Focus on alt coins
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 for S&D analysis
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK']
    
    def get_binance_klines(self, symbol, interval='4h', limit=500):
        """Get klines from Binance for Supply & Demand analysis"""
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
            
            # Add technical indicators for S&D analysis
            df = self.add_supply_demand_indicators(df)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_sma', 'atr', 'body_size']]
            
        except Exception as e:
            return None
    
    def add_supply_demand_indicators(self, df):
        """Add indicators for Supply & Demand analysis"""
        # Volume moving average for institutional activity detection
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # ATR for measuring aggressive moves
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Body size for measuring institutional activity
        df['body_size'] = np.abs(df['close'] - df['open'])
        
        return df
    
    def detect_aggressive_moves(self, df, min_move_pct=3.0):
        """Detect aggressive institutional moves that create S&D zones"""
        aggressive_moves = []
        
        for i in range(20, len(df) - 1):  # Need history for volume comparison
            current_candle = df.iloc[i]
            
            # Calculate move characteristics
            body_size_pct = (current_candle['body_size'] / current_candle['open']) * 100
            volume_ratio = current_candle['volume'] / df['volume_sma'].iloc[i] if pd.notna(df['volume_sma'].iloc[i]) else 1
            
            # Check for bullish aggressive move (DEMAND creation)
            if (current_candle['close'] > current_candle['open'] and  # Bullish candle
                body_size_pct >= min_move_pct and                      # Significant move
                volume_ratio >= 1.5):                                 # High volume
                
                # Look for the base (demand zone) before the move
                base_start = max(0, i - 10)  # Look back 10 candles for base
                base_data = df.iloc[base_start:i]
                
                if len(base_data) > 2:
                    demand_zone_low = base_data['low'].min()
                    demand_zone_high = base_data['high'].max()
                    zone_range_pct = (demand_zone_high - demand_zone_low) / demand_zone_low * 100
                    
                    # Zone should be reasonable size (not too tight, not too wide)
                    if 1.0 <= zone_range_pct <= 8.0:
                        aggressive_moves.append({
                            'type': 'demand',
                            'zone_low': demand_zone_low,
                            'zone_high': demand_zone_high,
                            'creation_index': i,
                            'creation_timestamp': current_candle['timestamp'],
                            'move_strength': body_size_pct,
                            'volume_ratio': volume_ratio,
                            'zone_range_pct': zone_range_pct,
                            'tests': 0,  # Will be calculated later
                            'last_test_index': None
                        })
            
            # Check for bearish aggressive move (SUPPLY creation)
            elif (current_candle['close'] < current_candle['open'] and  # Bearish candle
                  body_size_pct >= min_move_pct and                      # Significant move
                  volume_ratio >= 1.5):                                 # High volume
                
                # Look for the base (supply zone) before the move
                base_start = max(0, i - 10)  # Look back 10 candles for base
                base_data = df.iloc[base_start:i]
                
                if len(base_data) > 2:
                    supply_zone_low = base_data['low'].min()
                    supply_zone_high = base_data['high'].max()
                    zone_range_pct = (supply_zone_high - supply_zone_low) / supply_zone_low * 100
                    
                    # Zone should be reasonable size
                    if 1.0 <= zone_range_pct <= 8.0:
                        aggressive_moves.append({
                            'type': 'supply',
                            'zone_low': supply_zone_low,
                            'zone_high': supply_zone_high,
                            'creation_index': i,
                            'creation_timestamp': current_candle['timestamp'],
                            'move_strength': body_size_pct,
                            'volume_ratio': volume_ratio,
                            'zone_range_pct': zone_range_pct,
                            'tests': 0,  # Will be calculated later
                            'last_test_index': None
                        })
        
        return aggressive_moves
    
    def analyze_zone_tests(self, df, zones):
        """Analyze how many times each zone has been tested"""
        for zone in zones:
            creation_index = zone['creation_index']
            zone_low = zone['zone_low']
            zone_high = zone['zone_high']
            
            # Look at price action after zone creation
            post_creation_data = df.iloc[creation_index + 1:]
            
            test_count = 0
            last_test_index = None
            
            for idx, candle in post_creation_data.iterrows():
                # Check if price returned to the zone
                if zone['type'] == 'demand':
                    # For demand zones, look for price returning to zone from above
                    if (candle['low'] <= zone_high and 
                        candle['high'] >= zone_low):
                        test_count += 1
                        last_test_index = idx
                        
                elif zone['type'] == 'supply':
                    # For supply zones, look for price returning to zone from below
                    if (candle['high'] >= zone_low and 
                        candle['low'] <= zone_high):
                        test_count += 1
                        last_test_index = idx
            
            zone['tests'] = test_count
            zone['last_test_index'] = last_test_index
        
        return zones
    
    def classify_zone_freshness(self, zone):
        """Classify zone as Fresh, Aged, or Broken based on tests"""
        if zone['tests'] == 0:
            return 'FRESH'  # Never tested = highest probability
        elif zone['tests'] == 1:
            return 'TESTED_ONCE'  # Tested once = good probability
        elif zone['tests'] <= 3:
            return 'AGED'  # Multiple tests = lower probability
        else:
            return 'BROKEN'  # Too many tests = likely broken
    
    def calculate_zone_strength(self, zone):
        """Calculate zone strength score (0-100)"""
        score = 50  # Base score
        
        # Move strength bonus (stronger creation = stronger zone)
        if zone['move_strength'] >= 8.0:
            score += 20
        elif zone['move_strength'] >= 5.0:
            score += 15
        elif zone['move_strength'] >= 3.0:
            score += 10
        
        # Volume confirmation bonus
        if zone['volume_ratio'] >= 3.0:
            score += 15
        elif zone['volume_ratio'] >= 2.0:
            score += 10
        elif zone['volume_ratio'] >= 1.5:
            score += 5
        
        # Zone freshness bonus
        freshness = self.classify_zone_freshness(zone)
        if freshness == 'FRESH':
            score += 20
        elif freshness == 'TESTED_ONCE':
            score += 10
        elif freshness == 'AGED':
            score -= 10
        else:  # BROKEN
            score -= 30
        
        # Zone size bonus (tighter zones = more precise)
        if zone['zone_range_pct'] <= 2.0:
            score += 10
        elif zone['zone_range_pct'] <= 4.0:
            score += 5
        
        return min(max(score, 0), 100)
    
    def find_trading_opportunities(self, df, zones, current_price):
        """Find current trading opportunities at S&D zones"""
        opportunities = []
        current_index = len(df) - 1
        
        for zone in zones:
            # Skip broken zones
            if self.classify_zone_freshness(zone) == 'BROKEN':
                continue
            
            zone_strength = self.calculate_zone_strength(zone)
            
            # Only consider high-quality zones
            if zone_strength < 60:
                continue
            
            # Calculate distance from current price to zone
            if zone['type'] == 'demand':
                zone_center = (zone['zone_low'] + zone['zone_high']) / 2
                distance_pct = (current_price - zone_center) / zone_center * 100
                
                # Check if price is approaching or at demand zone
                if -2.0 <= distance_pct <= 15.0:  # Within zone or slightly above
                    # Determine entry signal strength
                    if -1.0 <= distance_pct <= 1.0:  # In the zone
                        entry_signal = "IMMEDIATE"
                    elif distance_pct <= 5.0:  # Close to zone
                        entry_signal = "APPROACHING"
                    else:  # Further away
                        entry_signal = "WATCH"
                    
                    # Calculate targets (next supply zone or resistance)
                    target_zones = [z for z in zones if z['type'] == 'supply' and 
                                   z['zone_low'] > current_price and
                                   self.classify_zone_freshness(z) != 'BROKEN']
                    
                    if target_zones:
                        nearest_supply = min(target_zones, key=lambda x: x['zone_low'])
                        target_1 = nearest_supply['zone_low']
                        target_2 = nearest_supply['zone_high']
                    else:
                        # Use percentage targets if no supply zone found
                        target_1 = current_price * 1.15  # 15% target
                        target_2 = current_price * 1.25  # 25% target
                    
                    # Calculate stop loss (below demand zone)
                    stop_loss = zone['zone_low'] * 0.98
                    
                    # Risk/reward calculation
                    entry_price = zone_center
                    risk_pct = abs(entry_price - stop_loss) / entry_price * 100
                    reward_1_pct = (target_1 - entry_price) / entry_price * 100
                    reward_2_pct = (target_2 - entry_price) / entry_price * 100
                    
                    risk_reward_1 = reward_1_pct / risk_pct if risk_pct > 0 else 0
                    risk_reward_2 = reward_2_pct / risk_pct if risk_pct > 0 else 0
                    
                    opportunities.append({
                        'direction': 'LONG',
                        'zone_type': 'demand',
                        'zone': zone,
                        'entry_signal': entry_signal,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target_1': target_1,
                        'target_2': target_2,
                        'risk_pct': risk_pct,
                        'reward_1_pct': reward_1_pct,
                        'reward_2_pct': reward_2_pct,
                        'risk_reward_1': risk_reward_1,
                        'risk_reward_2': risk_reward_2,
                        'zone_strength': zone_strength,
                        'freshness': self.classify_zone_freshness(zone),
                        'distance_pct': distance_pct
                    })
            
            elif zone['type'] == 'supply':
                zone_center = (zone['zone_low'] + zone['zone_high']) / 2
                distance_pct = (zone_center - current_price) / current_price * 100
                
                # Check if price is approaching or at supply zone
                if -2.0 <= distance_pct <= 15.0:  # Within zone or slightly below
                    # Determine entry signal strength
                    if -1.0 <= distance_pct <= 1.0:  # In the zone
                        entry_signal = "IMMEDIATE"
                    elif distance_pct <= 5.0:  # Close to zone
                        entry_signal = "APPROACHING"
                    else:  # Further away
                        entry_signal = "WATCH"
                    
                    # Calculate targets (next demand zone or support)
                    target_zones = [z for z in zones if z['type'] == 'demand' and 
                                   z['zone_high'] < current_price and
                                   self.classify_zone_freshness(z) != 'BROKEN']
                    
                    if target_zones:
                        nearest_demand = max(target_zones, key=lambda x: x['zone_high'])
                        target_1 = nearest_demand['zone_high']
                        target_2 = nearest_demand['zone_low']
                    else:
                        # Use percentage targets if no demand zone found
                        target_1 = current_price * 0.85  # 15% target
                        target_2 = current_price * 0.75  # 25% target
                    
                    # Calculate stop loss (above supply zone)
                    stop_loss = zone['zone_high'] * 1.02
                    
                    # Risk/reward calculation
                    entry_price = zone_center
                    risk_pct = abs(stop_loss - entry_price) / entry_price * 100
                    reward_1_pct = (entry_price - target_1) / entry_price * 100
                    reward_2_pct = (entry_price - target_2) / entry_price * 100
                    
                    risk_reward_1 = reward_1_pct / risk_pct if risk_pct > 0 else 0
                    risk_reward_2 = reward_2_pct / risk_pct if risk_pct > 0 else 0
                    
                    opportunities.append({
                        'direction': 'SHORT',
                        'zone_type': 'supply',
                        'zone': zone,
                        'entry_signal': entry_signal,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target_1': target_1,
                        'target_2': target_2,
                        'risk_pct': risk_pct,
                        'reward_1_pct': reward_1_pct,
                        'reward_2_pct': reward_2_pct,
                        'risk_reward_1': risk_reward_1,
                        'risk_reward_2': risk_reward_2,
                        'zone_strength': zone_strength,
                        'freshness': self.classify_zone_freshness(zone),
                        'distance_pct': distance_pct
                    })
        
        return opportunities
    
    def scan_for_supply_demand_setups(self):
        """Main scanning function for Supply & Demand opportunities"""
        print(f"📦 SUPPLY & DEMAND ZONE SCANNER - 4H CHART")
        print("=" * 75)
        print(f"📈 Scanning 4H charts for institutional Supply & Demand zones...")
        print("📊 Strategy: Fresh zones = 70-80% success | Aged zones = 50-60% success")
        print("🎯 Optimized: 1-4 week holds, 15-50% targets, institutional levels")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'zones_found': 0, 'opportunities': 0}
        
        for symbol in coins:
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get price data
                df = self.get_binance_klines(symbol, interval=self.timeframe, limit=500)
                if df is None or len(df) < 100:
                    print("No data")
                    stats['no_data'] += 1
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Detect aggressive moves that create S&D zones
                zones = self.detect_aggressive_moves(df)
                
                if not zones:
                    print("No zones")
                    continue
                
                # Analyze zone testing history
                zones = self.analyze_zone_tests(df, zones)
                
                # Find trading opportunities
                symbol_opportunities = self.find_trading_opportunities(df, zones, current_price)
                
                if not symbol_opportunities:
                    print(f"{len(zones)} zones (no trades)")
                    stats['zones_found'] += len(zones)
                    continue
                
                # Select best opportunity per symbol
                best_opportunity = max(symbol_opportunities, key=lambda x: x['zone_strength'])
                best_opportunity['symbol'] = symbol
                best_opportunity['current_price'] = current_price
                
                opportunities.append(best_opportunity)
                
                # Quick display
                direction = best_opportunity['direction']
                zone_type = best_opportunity['zone_type'].title()
                freshness = best_opportunity['freshness']
                entry_signal = best_opportunity['entry_signal']
                
                if entry_signal == "IMMEDIATE":
                    status = "🎯 NOW"
                elif entry_signal == "APPROACHING":
                    status = "🔜 CLOSE"
                else:
                    status = "⏳ WATCH"
                
                direction_emoji = "📈" if direction == "LONG" else "📉"
                
                print(f"{status} | {direction_emoji} {direction} | {zone_type} | {freshness} | "
                      f"Score: {best_opportunity['zone_strength']:.0f}")
                
                stats['zones_found'] += len(zones)
                stats['opportunities'] += 1
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error - {e}")
                continue
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {stats['scanned']}")
        print(f"No data: {stats['no_data']}")
        print(f"S&D zones found: {stats['zones_found']}")
        print(f"Trading opportunities: {stats['opportunities']}")
        
        # Sort by zone strength
        opportunities.sort(key=lambda x: x['zone_strength'], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display Supply & Demand scanning results"""
        print("\n" + "=" * 90)
        print(f"📦 FOUND {len(opportunities)} SUPPLY & DEMAND TRADING OPPORTUNITIES (4H)")
        print("=" * 90)
        
        if not opportunities:
            print("❌ No high-quality Supply & Demand opportunities found")
            print("💡 S&D zones form after institutional activity - try again later")
            return
        
        print("📊 Results (sorted by zone strength):")
        print()
        
        for i, opp in enumerate(opportunities[:10], 1):
            # Setup info
            direction = opp['direction']
            zone_type = opp['zone_type'].title()
            freshness = opp['freshness']
            direction_emoji = "📈" if direction == "LONG" else "📉"
            
            # Entry status
            if opp['entry_signal'] == "IMMEDIATE":
                status = '🎯 NOW'
            elif opp['entry_signal'] == "APPROACHING":
                status = '🔜 CLOSE'
            else:
                status = '⏳ WATCH'
            
            print(f"{i:2}. {direction_emoji} {opp['symbol']:8} | Strength: {opp['zone_strength']:3.0f}/100 | "
                  f"📦 {zone_type.upper()} | {freshness} | {status}")
            
            # Current position and zone details
            zone = opp['zone']
            print(f"     Current: ${opp['current_price']:.6f} | Distance: {opp['distance_pct']:+.1f}%")
            print(f"     Zone: ${zone['zone_low']:.6f} - ${zone['zone_high']:.6f} "
                  f"({zone['zone_range_pct']:.1f}% range)")
            
            # Zone creation details
            creation_date = zone['creation_timestamp'].strftime('%Y-%m-%d')
            print(f"     Created: {creation_date} | Move: {zone['move_strength']:.1f}% | "
                  f"Volume: {zone['volume_ratio']:.1f}x | Tests: {zone['tests']}")
            
            # Entry and targets
            print(f"     Entry: ${opp['entry_price']:.6f} | Stop: ${opp['stop_loss']:.6f} "
                  f"({opp['risk_pct']:.1f}% risk)")
            
            # Targets
            print(f"     Targets: T1: ${opp['target_1']:.6f} ({opp['reward_1_pct']:.1f}% | R:R {opp['risk_reward_1']:.1f}:1) | "
                  f"T2: ${opp['target_2']:.6f} ({opp['reward_2_pct']:.1f}% | R:R {opp['risk_reward_2']:.1f}:1)")
            
            # Trading guidance
            if opp['entry_signal'] == "IMMEDIATE":
                print(f"     🎯 ENTER NOW: Price in {zone_type} zone - institutional level active")
            elif opp['entry_signal'] == "APPROACHING":
                print(f"     🔜 PREPARE: Price approaching {zone_type} zone - set alerts")
            else:
                print(f"     ⏳ MONITOR: Watch for price to reach {zone_type} zone")
            
            print()
        
        print("📦 SUPPLY & DEMAND STRATEGY:")
        print("• SUPPLY ZONES: Where institutions sold heavily (resistance)")
        print("• DEMAND ZONES: Where institutions bought heavily (support)")
        print("• FRESH ZONES: Never tested = 70-80% success rate")
        print("• AGED ZONES: Multiple tests = 50-60% success rate")
        print("• ENTRY: When price returns to zone with confirmation")
        print("• TARGETS: Next opposing zone or percentage targets")
        print("• HOLD TIME: 1-4 weeks typical (institutional timeframe)")
        print("• SUCCESS RATE: 70-80% fresh zones, 50-60% aged zones")

def main():
    scanner = SupplyDemandScanner4H(timeframe='4h')
    opportunities = scanner.scan_for_supply_demand_setups()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()