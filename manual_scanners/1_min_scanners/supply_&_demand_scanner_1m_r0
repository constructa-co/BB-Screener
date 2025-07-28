#!/usr/bin/env python3
"""
Supply & Demand Zone Scanner - 1-Minute Chart
Optimized for ultra micro-scalping and tick-level trading

Detects nano supply/demand zones for lightning-fast scalping opportunities.
Focus on ultra-fresh zones with rapid profit targets.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class SupplyDemandScanner1M:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'rateLimit': 50,
            'enableRateLimit': True,
        })
        
        # 1M Micro-Scalping Parameters (more realistic)
        self.lookback_periods = 480  # 8 hours of 1M candles
        self.min_move_percent = 0.3  # Minimum 0.3% move to create zone
        self.max_move_percent = 6    # Maximum 6% move for ultra-scalping
        self.min_volume_ratio = 0.9  # Very lenient volume requirement
        self.zone_range_percent = 0.1 # Nano-tight zones (0.1-3%)
        self.max_zone_range = 3.0
        self.recency_hours = 12      # More realistic (12 hours)
        self.min_quality_score = 25  # Much lower threshold
        
    def get_symbols(self):
        """Get comprehensive trading pairs for 1M scalping"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets.keys() if s.endswith('USDT') and 
                      markets[s]['active'] and markets[s]['spot']]
            
            # More comprehensive list for better opportunities
            priority_symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'BNB/USDT', 'DOGE/USDT',
                'MATIC/USDT', 'SHIB/USDT', 'PEPE/USDT', 'ARB/USDT', 'OP/USDT',
                'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'NEAR/USDT', 'FTM/USDT',
                'ATOM/USDT', 'ICP/USDT', 'FIL/USDT', 'RENDER/USDT', 'INJ/USDT',
                'TIA/USDT', 'SEI/USDT', 'JUP/USDT', 'PENDLE/USDT', 'WLD/USDT',
                'TAO/USDT', 'FET/USDT', 'AAVE/USDT', 'MKR/USDT', 'LDO/USDT',
                'COMP/USDT', 'SUSHI/USDT', 'CRV/USDT', 'YFI/USDT', 'SNX/USDT'
            ]
            
            # Filter for available symbols
            final_symbols = [s for s in priority_symbols if s in symbols]
            
            # Add other symbols to reach good coverage
            other_symbols = [s for s in symbols if s not in final_symbols]
            final_symbols.extend(other_symbols[:20])
            
            return final_symbols[:60]  # Increased from 15 to 60
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return []

    def get_1m_data(self, symbol):
        """Get 1-minute OHLCV data"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', since=since, limit=self.lookback_periods)
            
            if len(ohlcv) < 10:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add indicators optimized for 1M ultra-scalping
            df['sma_volume'] = df['volume'].rolling(window=5).mean()  # Ultra-short SMA
            df['volume_ratio'] = df['volume'] / df['sma_volume']
            df['range_percent'] = ((df['high'] - df['low']) / df['low']) * 100
            df['body_percent'] = ((abs(df['close'] - df['open'])) / df['open']) * 100
            df['tick_pressure'] = ((df['close'] - df['open']) / (df['high'] - df['low'])) * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def detect_supply_zones(self, df):
        """Detect nano supply zones for 1M scalping"""
        supply_zones = []
        
        for i in range(5, len(df) - 2):  # Ultra-short buffer for 1M
            current_candle = df.iloc[i]
            
            # Look for nano bearish moves (supply creation)
            if (current_candle['body_percent'] >= self.min_move_percent and
                current_candle['close'] < current_candle['open'] and
                current_candle['volume_ratio'] >= self.min_volume_ratio):
                
                # Find nano supply zone (1 candle before drop)
                supply_start_idx = max(0, i - 1)
                supply_candles = df.iloc[supply_start_idx:i]
                
                if len(supply_candles) == 0:
                    continue
                
                zone_high = supply_candles['high'].max()
                zone_low = supply_candles['low'].min()
                zone_range_pct = ((zone_high - zone_low) / zone_low) * 100
                
                # Nano-tight zone validation for 1M
                if (self.zone_range_percent <= zone_range_pct <= self.max_zone_range):
                    
                    # Calculate nano move strength
                    move_start = current_candle['open']
                    move_end = df.iloc[min(i + 2, len(df) - 1)]['low']
                    move_percent = ((move_start - move_end) / move_start) * 100
                    
                    if move_percent >= self.min_move_percent:
                        
                        # Check zone testing (ultra-recent for 1M)
                        tests = self.count_zone_tests(df, i + 1, zone_low, zone_high, 'supply')
                        zone_age = self.get_zone_age_hours(df, i)
                        
                        # Accept recent zones for 1M (more lenient)
                        if zone_age <= self.recency_hours:
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
                                'zone_range_pct': zone_range_pct,
                                'tick_pressure': current_candle['tick_pressure']
                            })
        
        return supply_zones

    def detect_demand_zones(self, df):
        """Detect nano demand zones for 1M scalping"""
        demand_zones = []
        
        for i in range(5, len(df) - 2):
            current_candle = df.iloc[i]
            
            # Look for nano bullish moves (demand creation)
            if (current_candle['body_percent'] >= self.min_move_percent and
                current_candle['close'] > current_candle['open'] and
                current_candle['volume_ratio'] >= self.min_volume_ratio):
                
                # Find nano demand zone
                demand_start_idx = max(0, i - 1)
                demand_candles = df.iloc[demand_start_idx:i]
                
                if len(demand_candles) == 0:
                    continue
                
                zone_high = demand_candles['high'].max()
                zone_low = demand_candles['low'].min()
                zone_range_pct = ((zone_high - zone_low) / zone_low) * 100
                
                # Nano-tight validation
                if (self.zone_range_percent <= zone_range_pct <= self.max_zone_range):
                    
                    # Calculate nano move strength
                    move_start = current_candle['open']
                    move_end = df.iloc[min(i + 2, len(df) - 1)]['high']
                    move_percent = ((move_end - move_start) / move_start) * 100
                    
                    if move_percent >= self.min_move_percent:
                        
                        # Check zone testing
                        tests = self.count_zone_tests(df, i + 1, zone_low, zone_high, 'demand')
                        zone_age = self.get_zone_age_hours(df, i)
                        
                        # Accept recent zones (more lenient)
                        if zone_age <= self.recency_hours:
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
                                'zone_range_pct': zone_range_pct,
                                'tick_pressure': current_candle['tick_pressure']
                            })
        
        return demand_zones

    def count_zone_tests(self, df, start_idx, zone_low, zone_high, zone_type):
        """Count zone tests for 1M (ultra-recent only)"""
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
                    
            # For 1M, only count ultra-recent tests
            if i - start_idx > 15:  # Only count tests in last 15 minutes
                break
                
        return tests

    def get_zone_age_hours(self, df, creation_index):
        """Calculate zone age in hours"""
        if creation_index >= len(df) - 1:
            return 0
        
        creation_time = df.iloc[creation_index]['timestamp']
        current_time = df.iloc[-1]['timestamp']
        age_hours = (current_time - creation_time).total_seconds() / 3600
        
        return round(age_hours, 2)  # More precision for 1M

    def calculate_zone_quality(self, zone, current_price):
        """Calculate zone quality score optimized for 1M ultra-scalping"""
        score = 0
        
        # Base score from move strength (max 25 points)
        if zone['strength'] >= 2:
            score += 25
        elif zone['strength'] >= 1.5:
            score += 20
        elif zone['strength'] >= 1:
            score += 15
        elif zone['strength'] >= 0.5:
            score += 10
        else:
            score += 5
        
        # Volume confirmation (max 20 points)
        if zone['volume_ratio'] >= 1.5:
            score += 20
        elif zone['volume_ratio'] >= 1.2:
            score += 15
        elif zone['volume_ratio'] >= 1.05:
            score += 10
        else:
            score += 5
        
        # Zone freshness - hyper-critical for 1M (max 35 points)
        if zone['tests'] == 0:
            score += 35  # Fresh zone - highest probability
        elif zone['tests'] == 1:
            score += 25  # Tested once - still good for 1M
        else:
            score += 5   # Over-tested - very low probability
        
        # Zone age - must be ultra-recent for 1M (max 15 points)
        if zone['age_hours'] <= 0.5:
            score += 15  # Ultra-recent (last 30 minutes)
        elif zone['age_hours'] <= 1:
            score += 12  # Very recent (last hour)
        elif zone['age_hours'] <= 2:
            score += 8   # Recent (last 2 hours)
        else:
            score += 2   # Too old for 1M
        
        # Zone range quality - nano-tight for 1M (max 5 points)
        if 0.3 <= zone['zone_range_pct'] <= 1:
            score += 5   # Optimal for 1M nano-scalping
        elif zone['zone_range_pct'] <= 1.5:
            score += 3   # Acceptable
        else:
            score += 1   # Too wide for nano-scalping
        
        return min(100, score)

    def get_zone_status(self, zone, current_price):
        """Determine zone trading status for 1M"""
        zone_center = zone['center']
        zone_range = abs(zone['high'] - zone['low']) / 2
        
        # Ultra-tight tolerances for 1M nano-scalping
        if abs(current_price - zone_center) <= zone_range * 0.5:
            return "🎯 NOW"     # Currently in zone
        elif abs(current_price - zone_center) <= zone_range * 1:
            return "🔜 CLOSE"   # Very close
        else:
            return "⏳ WATCH"   # Monitor

    def calculate_targets(self, zone, current_price, other_zones):
        """Calculate nano profit targets for 1M scalping"""
        if zone['type'] == 'DEMAND':
            # For nano LONG trades
            supply_zones = [z for z in other_zones if z['type'] == 'SUPPLY' and z['center'] > current_price]
            supply_zones.sort(key=lambda x: x['center'])
            
            if len(supply_zones) >= 1:
                target1 = supply_zones[0]['low']
                target2 = current_price * 1.015 if len(supply_zones) < 2 else supply_zones[1]['low']
            else:
                target1 = current_price * 1.008  # 0.8% nano target
                target2 = current_price * 1.015  # 1.5% nano target
                
        else:  # SUPPLY zone
            # For nano SHORT trades
            demand_zones = [z for z in other_zones if z['type'] == 'DEMAND' and z['center'] < current_price]
            demand_zones.sort(key=lambda x: x['center'], reverse=True)
            
            if len(demand_zones) >= 1:
                target1 = demand_zones[0]['high']
                target2 = current_price * 0.985 if len(demand_zones) < 2 else demand_zones[1]['high']
            else:
                target1 = current_price * 0.992  # 0.8% nano target
                target2 = current_price * 0.985  # 1.5% nano target
        
        return target1, target2

    def calculate_risk_reward(self, zone, current_price, target1, target2):
        """Calculate risk/reward for 1M nano-scalping"""
        entry_price = zone['center']
        
        if zone['type'] == 'DEMAND':
            # LONG trade
            stop_loss = zone['low'] * 0.999  # Ultra-tight stop for 1M
            risk = (entry_price - stop_loss) / entry_price * 100
            reward1 = (target1 - entry_price) / entry_price * 100
            reward2 = (target2 - entry_price) / entry_price * 100
        else:
            # SHORT trade
            stop_loss = zone['high'] * 1.001  # Ultra-tight stop for 1M
            risk = (stop_loss - entry_price) / entry_price * 100
            reward1 = (entry_price - target1) / entry_price * 100
            reward2 = (entry_price - target2) / entry_price * 100
        
        rr1 = reward1 / risk if risk > 0 else 0
        rr2 = reward2 / risk if risk > 0 else 0
        
        return risk, reward1, reward2, rr1, rr2

    def analyze_symbol(self, symbol):
        """Analyze symbol for 1M supply/demand zones"""
        try:
            print(f"📡 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_1m_data(symbol)
            if df is None or len(df) < 10:
                print("No data")
                return []
            
            current_price = df.iloc[-1]['close']
            
            # Detect nano zones
            supply_zones = self.detect_supply_zones(df)
            demand_zones = self.detect_demand_zones(df)
            all_zones = supply_zones + demand_zones
            
            if not all_zones:
                print("No zones")
                return []
            
            # Find nano-scalping setups
            quality_setups = []
            
            for zone in all_zones:
                quality_score = self.calculate_zone_quality(zone, current_price)
                
                if quality_score >= self.min_quality_score:
                    status = self.get_zone_status(zone, current_price)
                    target1, target2 = self.calculate_targets(zone, current_price, all_zones)
                    risk, reward1, reward2, rr1, rr2 = self.calculate_risk_reward(zone, current_price, target1, target2)
                    
                    # More lenient 1M filters
                    if (risk <= 2.0 and  # Max 2% risk for micro-scalping (increased)
                        rr1 >= 0.8 and   # Min 0.8:1 R:R for 1M (reduced)
                        zone['age_hours'] <= 12):  # Recent zones (increased)
                        
                        setup = {
                            'symbol': symbol,
                            'type': zone['type'],
                            'score': quality_score,
                            'status': status,
                            'current_price': current_price,
                            'entry': zone['center'],
                            'stop': zone['low'] * 0.999 if zone['type'] == 'DEMAND' else zone['high'] * 1.001,
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
        """Scan all symbols for 1M nano-scalping opportunities"""
        print("🎯 SUPPLY & DEMAND SCANNER - 1-MINUTE CHART")
        print("=" * 75)
        print("📈 Scanning 1M charts for nano supply/demand zones...")
        print("📊 Strategy: Ultra-fresh zones → 0.5-2% nano-scalping targets")
        print()
        
        symbols = self.get_symbols()
        all_setups = []
        
        for symbol in symbols:
            try:
                setups = self.analyze_symbol(symbol)
                all_setups.extend(setups)
                time.sleep(0.05)  # Minimal rate limiting for 1M
                
            except Exception as e:
                print(f"📡 {symbol.replace('/USDT', '')}... Error: {e}")
                continue
        
        return all_setups

    def display_results(self, setups):
        """Display 1M nano-scalping results"""
        if not setups:
            print("\n📊 SCAN RESULTS:")
            print(f"Total scanned: 60")
            print("Quality setups: 0")
            print("=" * 89)
            print("🎯 FOUND 0 SUPPLY & DEMAND SETUPS (1-MINUTE)")
            print("=" * 89)
            print("❌ No high-quality nano zones found")
            print("💡 Try again in 5-15 minutes - looking for ultra-fresh nano zones")
            return
        
        # Sort by score descending
        setups.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: 15")
        print(f"Quality setups: {len(setups)}")
        print("=" * 89)
        print(f"🎯 FOUND {len(setups)} SUPPLY & DEMAND SETUPS (1-MINUTE)")
        print("=" * 89)
        
        for i, setup in enumerate(setups, 1):
            direction = "📈" if setup['type'] == 'DEMAND' else "📉"
            zone_type = setup['type']
            
            # Zone freshness for 1M
            if setup['tests'] == 0:
                freshness = "NANO-FRESH ⚡"
            elif setup['tests'] == 1:
                freshness = "ULTRA-FRESH ✅"
            else:
                freshness = "TESTED"
            
            # Age display in minutes for 1M
            age_minutes = setup['age_hours'] * 60
            age_display = f"{age_minutes:.0f}m" if age_minutes < 60 else f"{setup['age_hours']:.1f}h"
            
            print(f"\n{i}. {direction} {setup['symbol'].replace('/USDT', '')} | Score: {setup['score']}/100 | 📦 {zone_type} | {setup['status']}")
            print(f"   Current: ${setup['current_price']:.4f} | Zone: {setup['zone_range']}")
            print(f"   Entry: ${setup['entry']:.4f} | Stop: ${setup['stop']:.4f} ({setup['risk_pct']:.2f}% risk)")
            print(f"   Targets: T1: ${setup['target1']:.4f} ({setup['reward1_pct']:.2f}% | R:R {setup['rr1']:.1f}:1) | T2: ${setup['target2']:.4f} ({setup['reward2_pct']:.2f}% | R:R {setup['rr2']:.1f}:1)")
            print(f"   Zone: {freshness} | Volume: {setup['volume_ratio']:.2f}x | Age: {age_display} | Strength: {setup['strength']:.2f}%")
            
            # Nano-scalping instruction
            if setup['type'] == 'DEMAND':
                print(f"   🎯 NANO LONG: Lightning bounce expected - {age_display} old zone")
            else:
                print(f"   🎯 NANO SHORT: Lightning rejection expected - {age_display} old zone")

def main():
    """Main execution function"""
    try:
        scanner = SupplyDemandScanner1M()
        setups = scanner.scan_all_symbols()
        scanner.display_results(setups)
        
    except KeyboardInterrupt:
        print("\n❌ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan error: {e}")

if __name__ == "__main__":
    main()