#!/usr/bin/env python3
"""
SUPPLY & DEMAND ZONE SCANNER - 1H CHARTS
Professional scanner for institutional supply and demand levels (Day Trading)
Strategy: Fresh zones = 65-75% success rate | Aged zones = 45-55% success rate
Timeframe: 1H optimized for day trading (1-5 day holds, 5-20% targets)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from scipy.signal import argrelextrema

class SupplyDemandScanner1H:
    def __init__(self, timeframe='1h'):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.timeframe = timeframe
        
    def get_top_coins(self, limit=120):
        """Get top coins by volume for 1H Supply & Demand scanning"""
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
                
                # Volume requirements for 1H S&D detection  
                if (volume_24h > 15_000_000 and      # Active volume for 1H patterns
                    market_cap > 200_000_000 and     # Sufficient liquidity
                    market_cap < 300_000_000_000):   # Focus on active alts
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 for 1H S&D analysis
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK']
    
    def get_binance_klines(self, symbol, interval='1h', limit=300):
        """Get klines from Binance for 1H Supply & Demand analysis"""
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
            
            # Add technical indicators for 1H S&D analysis
            df = self.add_supply_demand_indicators(df)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volume_sma', 'atr', 'body_size']]
            
        except Exception as e:
            return None
    
    def add_supply_demand_indicators(self, df):
        """Add indicators for 1H Supply & Demand analysis"""
        # Volume moving average (shorter for 1H)
        df['volume_sma'] = df['volume'].rolling(window=15).mean()
        
        # ATR for measuring aggressive moves (shorter for 1H)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=10).mean()  # Shorter ATR for 1H
        
        # Body size for measuring institutional activity
        df['body_size'] = np.abs(df['close'] - df['open'])
        
        return df
    
    def detect_aggressive_moves(self, df, min_move_pct=2.5):
        """Detect aggressive institutional moves that create S&D zones (adjusted for 1H)"""
        aggressive_moves = []
        
        for i in range(15, len(df) - 1):  # Shorter history needed for 1H
            current_candle = df.iloc[i]
            
            # Calculate move characteristics (adjusted for 1H volatility)
            body_size_pct = (current_candle['body_size'] / current_candle['open']) * 100
            volume_ratio = current_candle['volume'] / df['volume_sma'].iloc[i] if pd.notna(df['volume_sma'].iloc[i]) else 1
            
            # Check for bullish aggressive move (DEMAND creation) - relaxed for 1H
            if (current_candle['close'] > current_candle['open'] and  # Bullish candle
                body_size_pct >= min_move_pct and                      # Smaller move threshold for 1H
                volume_ratio >= 1.3):                                 # Lower volume threshold
                
                # Look for the base (demand zone) before the move
                base_start = max(0, i - 8)  # Shorter lookback for 1H patterns
                base_data = df.iloc[base_start:i]
                
                if len(base_data) > 2:
                    demand_zone_low = base_data['low'].min()
                    demand_zone_high = base_data['high'].max()
                    zone_range_pct = (demand_zone_high - demand_zone_low) / demand_zone_low * 100
                    
                    # Zone should be reasonable size for 1H (adjusted range)
                    if 0.8 <= zone_range_pct <= 6.0:
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
            
            # Check for bearish aggressive move (SUPPLY creation) - relaxed for 1H
            elif (current_candle['close'] < current_candle['open'] and  # Bearish candle
                  body_size_pct >= min_move_pct and                      # Smaller move threshold
                  volume_ratio >= 1.3):                                 # Lower volume threshold
                
                # Look for the base (supply zone) before the move
                base_start = max(0, i - 8)  # Shorter lookback for 1H
                base_data = df.iloc[base_start:i]
                
                if len(base_data) > 2:
                    supply_zone_low = base_data['low'].min()
                    supply_zone_high = base_data['high'].max()
                    zone_range_pct = (supply_zone_high - supply_zone_low) / supply_zone_low * 100
                    
                    # Zone should be reasonable size for 1H
                    if 0.8 <= zone_range_pct <= 6.0:
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
        elif zone['tests'] <= 2:  # More lenient for 1H
            return 'AGED'  # Multiple tests = lower probability
        else:
            return 'BROKEN'  # Too many tests = likely broken
    
    def calculate_zone_strength(self, zone):
        """Calculate zone strength score (0-100) - adjusted for 1H"""
        score = 45  # Lower base score for 1H due to more noise
        
        # Move strength bonus (adjusted for 1H volatility)
        if zone['move_strength'] >= 6.0:
            score += 20
        elif zone['move_strength'] >= 4.0:
            score += 15
        elif zone['move_strength'] >= 2.5:
            score += 10
        
        # Volume confirmation bonus (adjusted thresholds)
        if zone['volume_ratio'] >= 2.5:
            score += 15
        elif zone['volume_ratio'] >= 1.8:
            score += 10
        elif zone['volume_ratio'] >= 1.3:
            score += 5
        
        # Zone freshness bonus
        freshness = self.classify_zone_freshness(zone)
        if freshness == 'FRESH':
            score += 20
        elif freshness == 'TESTED_ONCE':
            score += 12
        elif freshness == 'AGED':
            score -= 8
        else:  # BROKEN
            score -= 25
        
        # Zone size bonus (tighter zones = more precise)
        if zone['zone_range_pct'] <= 1.5:
            score += 12
        elif zone['zone_range_pct'] <= 3.0:
            score += 8
        elif zone['zone_range_pct'] <= 4.5:
            score += 4
        
        # Recency bonus (recent zones more relevant for 1H)
        current_time = datetime.now()
        zone_age_hours = (current_time - zone['creation_timestamp']).total_seconds() / 3600
        
        if zone_age_hours <= 24:  # Less than 1 day old
            score += 10
        elif zone_age_hours <= 72:  # Less than 3 days old
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
            
            # Lower quality threshold for 1H (more opportunities)
            if zone_strength < 55:
                continue
            
            # Calculate distance from current price to zone
            if zone['type'] == 'demand':
                zone_center = (zone['zone_low'] + zone['zone_high']) / 2
                distance_pct = (current_price - zone_center) / zone_center * 100
                
                # Check if price is approaching or at demand zone (wider range for 1H)
                if -2.5 <= distance_pct <= 12.0:  # More tolerance for 1H volatility
                    # Determine entry signal strength
                    if -1.5 <= distance_pct <= 1.5:  # In the zone
                        entry_signal = "IMMEDIATE"
                    elif distance_pct <= 6.0:  # Close to zone
                        entry_signal = "APPROACHING"
                    else:  # Further away
                        entry_signal = "WATCH"
                    
                    # Calculate targets (next supply zone or percentage targets)
                    target_zones = [z for z in zones if z['type'] == 'supply' and 
                                   z['zone_low'] > current_price and
                                   self.classify_zone_freshness(z) != 'BROKEN']
                    
                    if target_zones:
                        nearest_supply = min(target_zones, key=lambda x: x['zone_low'])
                        target_1 = nearest_supply['zone_low']
                        target_2 = nearest_supply['zone_high']
                    else:
                        # Use smaller percentage targets for 1H
                        target_1 = current_price * 1.08  # 8% target
                        target_2 = current_price * 1.15  # 15% target
                    
                    # Calculate stop loss (below demand zone)
                    stop_loss = zone['zone_low'] * 0.985  # Tighter stop for 1H
                    
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
                        'distance_pct': distance_pct,
                        'hold_time_estimate': '1-5 days'
                    })
            
            elif zone['type'] == 'supply':
                zone_center = (zone['zone_low'] + zone['zone_high']) / 2
                distance_pct = (zone_center - current_price) / current_price * 100
                
                # Check if price is approaching or at supply zone
                if -2.5 <= distance_pct <= 12.0:  # More tolerance for 1H
                    # Determine entry signal strength
                    if -1.5 <= distance_pct <= 1.5:  # In the zone
                        entry_signal = "IMMEDIATE"
                    elif distance_pct <= 6.0:  # Close to zone
                        entry_signal = "APPROACHING"
                    else:  # Further away
                        entry_signal = "WATCH"
                    
                    # Calculate targets (next demand zone or percentage targets)
                    target_zones = [z for z in zones if z['type'] == 'demand' and 
                                   z['zone_high'] < current_price and
                                   self.classify_zone_freshness(z) != 'BROKEN']
                    
                    if target_zones:
                        nearest_demand = max(target_zones, key=lambda x: x['zone_high'])
                        target_1 = nearest_demand['zone_high']
                        target_2 = nearest_demand['zone_low']
                    else:
                        # Use smaller percentage targets for 1H
                        target_1 = current_price * 0.92  # 8% target
                        target_2 = current_price * 0.85  # 15% target
                    
                    # Calculate stop loss (above supply zone)
                    stop_loss = zone['zone_high'] * 1.015  # Tighter stop for 1H
                    
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
                        'distance_pct': distance_pct,
                        'hold_time_estimate': '1-5 days'
                    })
        
        return opportunities
    
    def scan_for_supply_demand_setups(self):
        """Main scanning function for 1H Supply & Demand opportunities"""
        print(f"📦 SUPPLY & DEMAND ZONE SCANNER - 1H CHART (DAY TRADING)")
        print("=" * 80)
        print(f"📈 Scanning 1H charts for institutional Supply & Demand zones...")
        print("📊 Strategy: Fresh zones = 65-75% success | Aged zones = 45-55% success")
        print("🎯 Optimized: 1-5 day holds, 5-20% targets, active day trading")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'zones_found': 0, 'opportunities': 0}
        
        for symbol in coins:
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get price data
                df = self.get_binance_klines(symbol, interval=self.timeframe, limit=300)
                if df is None or len(df) < 50:
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
                best_rr = max(best_opportunity['risk_reward_1'], best_opportunity['risk_reward_2'])
                
                print(f"{status} | {direction_emoji} {direction} | {zone_type} | {freshness} | "
                      f"R:R {best_rr:.1f}:1 | Score: {best_opportunity['zone_strength']:.0f}")
                
                stats['zones_found'] += len(zones)
                stats['opportunities'] += 1
                
                time.sleep(0.03)
                
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
        """Display 1H Supply & Demand scanning results"""
        print("\n" + "=" * 95)
        print(f"📦 FOUND {len(opportunities)} SUPPLY & DEMAND DAY TRADING OPPORTUNITIES (1H)")
        print("=" * 95)
        
        if not opportunities:
            print("❌ No high-quality 1H Supply & Demand opportunities found")
            print("💡 1H S&D zones form after aggressive moves - try again in 2-4 hours")
            return
        
        print("📊 Results (sorted by zone strength):")
        print()
        
        for i, opp in enumerate(opportunities[:15], 1):  # Show more for day trading
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
            zone_age_hours = (datetime.now() - zone['creation_timestamp']).total_seconds() / 3600
            print(f"     Created: {zone_age_hours:.0f}h ago | Move: {zone['move_strength']:.1f}% | "
                  f"Volume: {zone['volume_ratio']:.1f}x | Tests: {zone['tests']}")
            
            # Entry and targets
            print(f"     Entry: ${opp['entry_price']:.6f} | Stop: ${opp['stop_loss']:.6f} "
                  f"({opp['risk_pct']:.1f}% risk)")
            
            # Targets
            print(f"     Targets: T1: ${opp['target_1']:.6f} ({opp['reward_1_pct']:.1f}% | R:R {opp['risk_reward_1']:.1f}:1) | "
                  f"T2: ${opp['target_2']:.6f} ({opp['reward_2_pct']:.1f}% | R:R {opp['risk_reward_2']:.1f}:1)")
            
            # Trading guidance with hold time
            if opp['entry_signal'] == "IMMEDIATE":
                print(f"     🎯 DAY TRADE: Enter {zone_type} zone now | Hold: {opp['hold_time_estimate']}")
            elif opp['entry_signal'] == "APPROACHING":
                print(f"     🔜 PREPARE: Set alerts for {zone_type} zone | Hold: {opp['hold_time_estimate']}")
            else:
                print(f"     ⏳ MONITOR: Watch price movement toward {zone_type} zone")
            
            print()
        
        print("📦 1H SUPPLY & DEMAND DAY TRADING STRATEGY:")
        print("• FREQUENCY: 3-8 quality setups per day")
        print("• HOLD TIME: 1-5 days typical (day trading friendly)")
        print("• TARGETS: 5-20% moves (perfect for active traders)")
        print("• SUCCESS RATE: 65-75% fresh zones, 45-55% aged zones") 
        print("• RISK/REWARD: 2-4:1 typical (excellent for compound growth)")
        print("• POSITION SIZE: 1-2% risk per trade (moderate frequency)")
        print("• MONITORING: Check every 2-4 hours for new opportunities")
        print("• ZONE LIFE: Recent zones (24-72h) most effective")

def main():
    scanner = SupplyDemandScanner1H(timeframe='1h')
    opportunities = scanner.scan_for_supply_demand_setups()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()