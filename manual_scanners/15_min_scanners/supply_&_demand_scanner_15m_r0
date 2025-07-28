#!/usr/bin/env python3
"""
Supply & Demand Zone Scanner - 15-Minute Chart
Optimized for active scalping and high-frequency trading

Detects institutional supply/demand zones where big money entered/exited positions.
Focuses on fresh zones with high success rates for scalping opportunities.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class SupplyDemandScanner15M:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'rateLimit': 50,
            'enableRateLimit': True,
        })
        
        # 15M Scalping Parameters (optimized for high frequency)
        self.lookback_periods = 96  # 24 hours of 15M candles
        self.min_move_percent = 1.5  # Minimum 1.5% move to create zone
        self.max_move_percent = 12   # Maximum 12% move for scalping
        self.min_volume_ratio = 1.2  # Lower volume requirement for 15M
        self.zone_range_percent = 0.5 # Tighter zones for scalping (0.5-4%)
        self.max_zone_range = 4.0
        self.recency_hours = 12      # Recent zones get bonus (12 hours)
        self.min_quality_score = 50  # Lower threshold for more opportunities
        
    def get_symbols(self):
        """Get top liquid trading pairs for scalping"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets.keys() if s.endswith('USDT') and 
                      markets[s]['active'] and markets[s]['spot']]
            
            # Focus on most liquid pairs for scalping
            priority_symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
                'LTC/USDT', 'BCH/USDT', 'BNB/USDT', 'DOGE/USDT', 'SHIB/USDT',
                'PEPE/USDT', 'WIF/USDT', 'BONK/USDT', 'FLOKI/USDT', 'ARB/USDT'
            ]
            
            # Add priority symbols first, then others
            final_symbols = []
            for symbol in priority_symbols:
                if symbol in symbols:
                    final_symbols.append(symbol)
            
            # Add other liquid symbols
            other_symbols = [s for s in symbols if s not in final_symbols][:30]
            final_symbols.extend(other_symbols)
            
            return final_symbols[:50]  # Limit for faster scanning
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return []

    def get_15m_data(self, symbol):
        """Get 15-minute OHLCV data"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 15 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '15m', since=since, limit=self.lookback_periods)
            
            if len(ohlcv) < 20:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators for 15M scalping
            df['sma_volume'] = df['volume'].rolling(window=10).mean()  # Shorter SMA for 15M
            df['volume_ratio'] = df['volume'] / df['sma_volume']
            df['range_percent'] = ((df['high'] - df['low']) / df['low']) * 100
            df['body_percent'] = ((abs(df['close'] - df['open'])) / df['open']) * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def detect_supply_zones(self, df):
        """Detect supply zones where institutions sold heavily"""
        supply_zones = []
        
        for i in range(10, len(df) - 5):  # Leave buffer for zone validation
            current_candle = df.iloc[i]
            
            # Look for strong bearish moves (supply creation)
            if (current_candle['body_percent'] >= self.min_move_percent and
                current_candle['close'] < current_candle['open'] and
                current_candle['volume_ratio'] >= self.min_volume_ratio):
                
                # Find the supply zone (candle before the drop)
                supply_start_idx = max(0, i - 3)
                supply_candles = df.iloc[supply_start_idx:i]
                
                if len(supply_candles) == 0:
                    continue
                
                zone_high = supply_candles['high'].max()
                zone_low = supply_candles['low'].min()
                zone_range_pct = ((zone_high - zone_low) / zone_low) * 100
                
                # Zone validation for 15M scalping
                if (self.zone_range_percent <= zone_range_pct <= self.max_zone_range):
                    
                    # Calculate move strength from supply zone
                    move_start = current_candle['open']
                    move_end = df.iloc[min(i + 5, len(df) - 1)]['low']
                    move_percent = ((move_start - move_end) / move_start) * 100
                    
                    if move_percent >= self.min_move_percent:
                        
                        # Check if zone has been tested (aged vs fresh)
                        tests = self.count_zone_tests(df, i + 1, zone_low, zone_high, 'supply')
                        zone_age = self.get_zone_age_hours(df, i)
                        
                        supply_zones.append({
                            'type': 'SUPPLY',
                            'high': zone_high,
                            'low': zone_low,
                            'center': (zone_high + zone_low) / 2,
                            'strength': move_percent,
                            'volume_ratio': current_candle['volume_ratio'],
                            'tests': tests,
                            'age_hours': zone_age,
                            'creation_index': i,
                            'zone_range_pct': zone_range_pct
                        })
        
        return supply_zones

    def detect_demand_zones(self, df):
        """Detect demand zones where institutions bought heavily"""
        demand_zones = []
        
        for i in range(10, len(df) - 5):
            current_candle = df.iloc[i]
            
            # Look for strong bullish moves (demand creation)
            if (current_candle['body_percent'] >= self.min_move_percent and
                current_candle['close'] > current_candle['open'] and
                current_candle['volume_ratio'] >= self.min_volume_ratio):
                
                # Find the demand zone (candle before the rally)
                demand_start_idx = max(0, i - 3)
                demand_candles = df.iloc[demand_start_idx:i]
                
                if len(demand_candles) == 0:
                    continue
                
                zone_high = demand_candles['high'].max()
                zone_low = demand_candles['low'].min()
                zone_range_pct = ((zone_high - zone_low) / zone_low) * 100
                
                # Zone validation
                if (self.zone_range_percent <= zone_range_pct <= self.max_zone_range):
                    
                    # Calculate move strength from demand zone
                    move_start = current_candle['open']
                    move_end = df.iloc[min(i + 5, len(df) - 1)]['high']
                    move_percent = ((move_end - move_start) / move_start) * 100
                    
                    if move_percent >= self.min_move_percent:
                        
                        # Check zone testing
                        tests = self.count_zone_tests(df, i + 1, zone_low, zone_high, 'demand')
                        zone_age = self.get_zone_age_hours(df, i)
                        
                        demand_zones.append({
                            'type': 'DEMAND',
                            'high': zone_high,
                            'low': zone_low,
                            'center': (zone_high + zone_low) / 2,
                            'strength': move_percent,
                            'volume_ratio': current_candle['volume_ratio'],
                            'tests': tests,
                            'age_hours': zone_age,
                            'creation_index': i,
                            'zone_range_pct': zone_range_pct
                        })
        
        return demand_zones

    def count_zone_tests(self, df, start_idx, zone_low, zone_high, zone_type):
        """Count how many times a zone has been tested"""
        tests = 0
        
        for i in range(start_idx, len(df)):
            candle_low = df.iloc[i]['low']
            candle_high = df.iloc[i]['high']
            
            # Check if price touched the zone
            if zone_type == 'supply':
                if candle_high >= zone_low and candle_low <= zone_high:
                    tests += 1
            else:  # demand
                if candle_low <= zone_high and candle_high >= zone_low:
                    tests += 1
                    
            # For 15M scalping, limit test counting to recent periods
            if i - start_idx > 40:  # Only count tests in last 10 hours
                break
                
        return tests

    def get_zone_age_hours(self, df, creation_index):
        """Calculate zone age in hours"""
        if creation_index >= len(df) - 1:
            return 0
        
        creation_time = df.iloc[creation_index]['timestamp']
        current_time = df.iloc[-1]['timestamp']
        age_hours = (current_time - creation_time).total_seconds() / 3600
        
        return round(age_hours, 1)

    def calculate_zone_quality(self, zone, current_price):
        """Calculate zone quality score (0-100) optimized for 15M scalping"""
        score = 0
        
        # Base score from move strength (max 25 points)
        if zone['strength'] >= 4:
            score += 25
        elif zone['strength'] >= 2.5:
            score += 20
        elif zone['strength'] >= 1.5:
            score += 15
        else:
            score += 10
        
        # Volume confirmation (max 20 points)
        if zone['volume_ratio'] >= 2.0:
            score += 20
        elif zone['volume_ratio'] >= 1.5:
            score += 15
        elif zone['volume_ratio'] >= 1.2:
            score += 10
        else:
            score += 5
        
        # Zone freshness - critical for 15M (max 25 points)
        if zone['tests'] == 0:
            score += 25  # Fresh zone - highest probability
        elif zone['tests'] == 1:
            score += 18  # Tested once - still good
        elif zone['tests'] == 2:
            score += 10  # Aged zone - lower probability
        else:
            score += 2   # Over-tested - very low probability
        
        # Zone age bonus for recent zones (max 15 points)
        if zone['age_hours'] <= 4:
            score += 15  # Very recent
        elif zone['age_hours'] <= 8:
            score += 12  # Recent
        elif zone['age_hours'] <= 12:
            score += 8   # Moderately recent
        else:
            score += 3   # Old zone
        
        # Zone range quality (max 10 points)
        if 1.0 <= zone['zone_range_pct'] <= 2.5:
            score += 10  # Optimal range for 15M
        elif zone['zone_range_pct'] <= 4.0:
            score += 7   # Acceptable range
        else:
            score += 3   # Too wide for scalping
        
        # Current position relative to zone (max 5 points)
        distance_to_zone = abs(current_price - zone['center']) / current_price * 100
        if distance_to_zone <= 1:
            score += 5   # Very close
        elif distance_to_zone <= 3:
            score += 3   # Close
        else:
            score += 1   # Far
        
        return min(100, score)

    def get_zone_status(self, zone, current_price):
        """Determine zone trading status"""
        zone_center = zone['center']
        zone_range = abs(zone['high'] - zone['low']) / 2
        
        # 15M scalping requires tighter tolerances
        if abs(current_price - zone_center) <= zone_range * 0.8:
            return "🎯 NOW"     # Currently in zone - immediate entry
        elif abs(current_price - zone_center) <= zone_range * 1.5:
            return "🔜 CLOSE"   # Approaching zone - prepare for entry
        else:
            return "⏳ WATCH"   # Far from zone - monitor

    def calculate_targets(self, zone, current_price, other_zones):
        """Calculate profit targets based on next opposing zones"""
        if zone['type'] == 'DEMAND':
            # For LONG trades, find next supply zones above current price
            supply_zones = [z for z in other_zones if z['type'] == 'SUPPLY' and z['center'] > current_price]
            supply_zones.sort(key=lambda x: x['center'])
            
            if len(supply_zones) >= 2:
                target1 = supply_zones[0]['low']
                target2 = supply_zones[1]['low']
            elif len(supply_zones) == 1:
                target1 = supply_zones[0]['low']
                target2 = current_price * 1.04  # 4% target for 15M
            else:
                target1 = current_price * 1.025  # 2.5% target
                target2 = current_price * 1.04   # 4% target
                
        else:  # SUPPLY zone
            # For SHORT trades, find next demand zones below current price
            demand_zones = [z for z in other_zones if z['type'] == 'DEMAND' and z['center'] < current_price]
            demand_zones.sort(key=lambda x: x['center'], reverse=True)
            
            if len(demand_zones) >= 2:
                target1 = demand_zones[0]['high']
                target2 = demand_zones[1]['high']
            elif len(demand_zones) == 1:
                target1 = demand_zones[0]['high']
                target2 = current_price * 0.96  # 4% target for 15M
            else:
                target1 = current_price * 0.975  # 2.5% target
                target2 = current_price * 0.96   # 4% target
        
        return target1, target2

    def calculate_risk_reward(self, zone, current_price, target1, target2):
        """Calculate risk/reward ratios"""
        entry_price = zone['center']
        
        if zone['type'] == 'DEMAND':
            # LONG trade
            stop_loss = zone['low'] * 0.996  # Just below demand zone
            risk = (entry_price - stop_loss) / entry_price * 100
            reward1 = (target1 - entry_price) / entry_price * 100
            reward2 = (target2 - entry_price) / entry_price * 100
        else:
            # SHORT trade
            stop_loss = zone['high'] * 1.004  # Just above supply zone
            risk = (stop_loss - entry_price) / entry_price * 100
            reward1 = (entry_price - target1) / entry_price * 100
            reward2 = (entry_price - target2) / entry_price * 100
        
        rr1 = reward1 / risk if risk > 0 else 0
        rr2 = reward2 / risk if risk > 0 else 0
        
        return risk, reward1, reward2, rr1, rr2

    def analyze_symbol(self, symbol):
        """Analyze a single symbol for supply/demand zones"""
        try:
            print(f"📡 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_15m_data(symbol)
            if df is None or len(df) < 20:
                print("No data")
                return []
            
            current_price = df.iloc[-1]['close']
            
            # Detect zones
            supply_zones = self.detect_supply_zones(df)
            demand_zones = self.detect_demand_zones(df)
            all_zones = supply_zones + demand_zones
            
            if not all_zones:
                print("No zones")
                return []
            
            # Find high-quality setups
            quality_setups = []
            
            for zone in all_zones:
                quality_score = self.calculate_zone_quality(zone, current_price)
                
                if quality_score >= self.min_quality_score:
                    status = self.get_zone_status(zone, current_price)
                    target1, target2 = self.calculate_targets(zone, current_price, all_zones)
                    risk, reward1, reward2, rr1, rr2 = self.calculate_risk_reward(zone, current_price, target1, target2)
                    
                    # 15M scalping filters
                    if (risk <= 2.0 and  # Max 2% risk for scalping
                        rr1 >= 1.5 and   # Min 1.5:1 R:R for 15M
                        zone['age_hours'] <= 24):  # Recent zones only
                        
                        setup = {
                            'symbol': symbol,
                            'type': zone['type'],
                            'score': quality_score,
                            'status': status,
                            'current_price': current_price,
                            'entry': zone['center'],
                            'stop': zone['low'] * 0.996 if zone['type'] == 'DEMAND' else zone['high'] * 1.004,
                            'target1': target1,
                            'target2': target2,
                            'risk_pct': risk,
                            'reward1_pct': reward1,
                            'reward2_pct': reward2,
                            'rr1': rr1,
                            'rr2': rr2,
                            'zone_range': f"${zone['low']:.4f} - ${zone['high']:.4f}",
                            'volume_ratio': zone['volume_ratio'],
                            'tests': zone['tests'],
                            'age_hours': zone['age_hours'],
                            'strength': zone['strength']
                        }
                        quality_setups.append(setup)
            
            if quality_setups:
                print(f"Found {len(quality_setups)} setups")
            else:
                print("No quality zones")
                
            return quality_setups
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all symbols for supply/demand opportunities"""
        print("🎯 SUPPLY & DEMAND SCANNER - 15-MINUTE CHART")
        print("=" * 75)
        print("📈 Scanning 15M charts for institutional supply/demand zones...")
        print("📊 Strategy: Fresh zones → 2-8% scalping targets")
        print()
        
        symbols = self.get_symbols()
        all_setups = []
        
        for symbol in symbols:
            try:
                setups = self.analyze_symbol(symbol)
                all_setups.extend(setups)
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"📡 {symbol.replace('/USDT', '')}... Error: {e}")
                continue
        
        return all_setups

    def display_results(self, setups):
        """Display scan results in formatted output"""
        if not setups:
            print("\n📊 SCAN RESULTS:")
            print("Total scanned: 50")
            print("Quality setups: 0")
            print("=" * 89)
            print("🎯 FOUND 0 SUPPLY & DEMAND SETUPS (15-MINUTE)")
            print("=" * 89)
            print("❌ No high-quality supply/demand zones found")
            print("💡 Try again in 30-60 minutes - looking for fresh institutional zones")
            return
        
        # Sort by score descending
        setups.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: 50")
        print(f"Quality setups: {len(setups)}")
        print("=" * 89)
        print(f"🎯 FOUND {len(setups)} SUPPLY & DEMAND SETUPS (15-MINUTE)")
        print("=" * 89)
        
        for i, setup in enumerate(setups, 1):
            direction = "📈" if setup['type'] == 'DEMAND' else "📉"
            zone_type = setup['type']
            
            # Zone freshness indicator
            if setup['tests'] == 0:
                freshness = "FRESH ✅"
            elif setup['tests'] == 1:
                freshness = "TESTED_ONCE"
            else:
                freshness = "AGED"
            
            print(f"\n{i}. {direction} {setup['symbol'].replace('/USDT', '')} | Score: {setup['score']}/100 | 📦 {zone_type} | {setup['status']}")
            print(f"   Current: ${setup['current_price']:.4f} | Zone: {setup['zone_range']}")
            print(f"   Entry: ${setup['entry']:.4f} | Stop: ${setup['stop']:.4f} ({setup['risk_pct']:.1f}% risk)")
            print(f"   Targets: T1: ${setup['target1']:.4f} ({setup['reward1_pct']:.1f}% | R:R {setup['rr1']:.1f}:1) | T2: ${setup['target2']:.4f} ({setup['reward2_pct']:.1f}% | R:R {setup['rr2']:.1f}:1)")
            print(f"   Zone: {freshness} | Volume: {setup['volume_ratio']:.1f}x | Age: {setup['age_hours']:.1f}h | Strength: {setup['strength']:.1f}%")
            
            # Trading instruction
            if setup['type'] == 'DEMAND':
                print(f"   🎯 LONG SETUP: Zone bounce expected - Enter on {freshness.lower()} demand zone")
            else:
                print(f"   🎯 SHORT SETUP: Zone rejection expected - Enter on {freshness.lower()} supply zone")

def main():
    """Main execution function"""
    try:
        scanner = SupplyDemandScanner15M()
        setups = scanner.scan_all_symbols()
        scanner.display_results(setups)
        
    except KeyboardInterrupt:
        print("\n❌ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan error: {e}")

if __name__ == "__main__":
    main()