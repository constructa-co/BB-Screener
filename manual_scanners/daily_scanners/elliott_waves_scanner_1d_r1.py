#!/usr/bin/env python3
"""
Elliott Wave Scanner - Daily Chart (V8 Balanced Version)
Professional Elliott Wave pattern detection with balanced parameters

Finds quality Elliott Wave patterns with reasonable filtering to avoid false signals.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class ElliottWaveScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'rateLimit': 50,
            'enableRateLimit': True,
        })
        
        # Balanced Daily Elliott Wave Parameters
        self.lookback_periods = 150  # ~5 months of daily data
        self.min_wave_size = 4       # Minimum 4% move for wave validation
        self.min_pivot_distance = 5  # Minimum 5 days between pivots
        self.min_quality_score = 50  # Balanced threshold
        self.wave_ratios = {
            'wave2_max': 0.95,       # Wave 2 retraces max 95% of Wave 1
            'wave2_min': 0.25,       # Wave 2 retraces min 25% of Wave 1
            'wave3_min': 1.0,        # Wave 3 minimum 100% of Wave 1
            'wave4_max': 0.65,       # Wave 4 retraces max 65% of Wave 3
            'wave4_min': 0.20,       # Wave 4 retraces min 20% of Wave 3
        }
        
    def get_symbols(self):
        """Get major trading pairs for Elliott Wave analysis"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets.keys() if s.endswith('USDT') and 
                      markets[s]['active'] and markets[s]['spot']]
            
            # Focus on major cryptocurrencies only
            major_symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
                'LTC/USDT', 'BCH/USDT', 'BNB/USDT', 'DOGE/USDT', 'ATOM/USDT',
                'NEAR/USDT', 'FTM/USDT', 'ICP/USDT', 'ALGO/USDT', 'AAVE/USDT',
                'MKR/USDT', 'COMP/USDT', 'SNX/USDT', 'CRV/USDT', 'SUSHI/USDT',
                'SHIB/USDT', 'PEPE/USDT', 'WIF/USDT', 'BONK/USDT', 'ARB/USDT'
            ]
            
            # Filter available symbols
            final_symbols = [s for s in major_symbols if s in symbols]
            
            return final_symbols[:30]  # Limit to 30 major coins
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return []

    def get_daily_data(self, symbol):
        """Get daily OHLCV data"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', since=since, limit=self.lookback_periods)
            
            if len(ohlcv) < 50:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def find_significant_pivots(self, df):
        """Find significant peaks and valleys with proper filtering"""
        pivots = []
        
        # Look for significant highs and lows with 5-day window
        for i in range(5, len(df) - 5):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            date = df.iloc[i]['timestamp']
            
            # Check for significant peak (highest in 5-day window)
            window_highs = df.iloc[i-5:i+6]['high']
            if high == window_highs.max():
                # Validate with minimum move requirement
                recent_low = df.iloc[max(0, i-20):i]['low'].min()
                move_size = (high - recent_low) / recent_low * 100
                if move_size >= self.min_wave_size:
                    pivots.append({
                        'index': i,
                        'price': high,
                        'date': date,
                        'type': 'peak',
                        'move_size': move_size
                    })
            
            # Check for significant valley
            window_lows = df.iloc[i-5:i+6]['low']
            if low == window_lows.min():
                recent_high = df.iloc[max(0, i-20):i]['high'].max()
                move_size = (recent_high - low) / recent_high * 100
                if move_size >= self.min_wave_size:
                    pivots.append({
                        'index': i,
                        'price': low,
                        'date': date,
                        'type': 'valley',
                        'move_size': move_size
                    })
        
        # Filter to remove close pivots and ensure alternating pattern
        filtered_pivots = []
        for pivot in pivots:
            if not filtered_pivots:
                filtered_pivots.append(pivot)
            else:
                last_pivot = filtered_pivots[-1]
                # Ensure different type and sufficient distance
                if (pivot['type'] != last_pivot['type'] and 
                    pivot['index'] - last_pivot['index'] >= self.min_pivot_distance):
                    filtered_pivots.append(pivot)
        
        return filtered_pivots

    def validate_elliott_wave_pattern(self, pivots, pattern_type):
        """Validate Elliott Wave pattern with proper rules"""
        try:
            if len(pivots) < 5:
                return None
            
            if pattern_type == 'BULLISH':
                # Check alternating pattern: valley-peak-valley-peak-valley
                expected_types = ['valley', 'peak', 'valley', 'peak', 'valley']
            else:
                # Check alternating pattern: peak-valley-peak-valley-peak
                expected_types = ['peak', 'valley', 'peak', 'valley', 'peak']
            
            actual_types = [p['type'] for p in pivots]
            if actual_types != expected_types:
                return None
            
            # Extract wave prices
            if pattern_type == 'BULLISH':
                w0 = pivots[0]['price']  # Wave start
                w1 = pivots[1]['price']  # Wave 1 end
                w2 = pivots[2]['price']  # Wave 2 end
                w3 = pivots[3]['price']  # Wave 3 end
                w4 = pivots[4]['price']  # Wave 4 end
                
                # Elliott Wave Rules
                wave1_size = w1 - w0
                wave2_retrace = (w1 - w2) / wave1_size
                wave3_size = w3 - w2
                wave4_retrace = (w3 - w4) / wave3_size
                
                # Rule 1: Wave 2 doesn't retrace more than 100% of Wave 1
                if wave2_retrace > 1.0:
                    return None
                
                # Rule 2: Wave 4 doesn't overlap Wave 1 territory
                if w4 <= w1:
                    return None
                
                # Rule 3: Wave 3 is not the shortest
                if wave3_size < wave1_size * 0.9:  # Allow some flexibility
                    return None
                
            else:  # BEARISH
                w0 = pivots[0]['price']  # Wave start
                w1 = pivots[1]['price']  # Wave 1 end
                w2 = pivots[2]['price']  # Wave 2 end
                w3 = pivots[3]['price']  # Wave 3 end
                w4 = pivots[4]['price']  # Wave 4 end
                
                wave1_size = w0 - w1
                wave2_retrace = (w2 - w1) / wave1_size
                wave3_size = w2 - w3
                wave4_retrace = (w4 - w3) / wave3_size
                
                if wave2_retrace > 1.0:
                    return None
                if w4 >= w1:
                    return None
                if wave3_size < wave1_size * 0.9:
                    return None
            
            # Calculate quality score
            score = self.calculate_pattern_quality(pivots, pattern_type)
            
            if score >= self.min_quality_score:
                return {
                    'pivots': pivots,
                    'pattern_type': pattern_type,
                    'quality_score': score,
                    'wave1_size': abs(wave1_size),
                    'wave2_retrace': wave2_retrace * 100,
                    'wave3_size': abs(wave3_size),
                    'wave4_retrace': wave4_retrace * 100 if len(pivots) >= 5 else 0
                }
            
            return None
            
        except Exception as e:
            return None

    def calculate_pattern_quality(self, pivots, pattern_type):
        """Calculate quality score for Elliott Wave pattern"""
        score = 30  # Base score
        
        try:
            if pattern_type == 'BULLISH':
                w0, w1, w2, w3, w4 = [p['price'] for p in pivots]
                wave1_size = w1 - w0
                wave2_retrace = (w1 - w2) / wave1_size
                wave3_size = w3 - w2
                wave4_retrace = (w3 - w4) / wave3_size
                
                # Good Wave 2 retracement (25-95%)
                if self.wave_ratios['wave2_min'] <= wave2_retrace <= self.wave_ratios['wave2_max']:
                    score += 15
                
                # Strong Wave 3 (100%+ of Wave 1)
                if wave3_size >= wave1_size * self.wave_ratios['wave3_min']:
                    score += 20
                
                # Good Wave 4 retracement (20-65%)
                if self.wave_ratios['wave4_min'] <= wave4_retrace <= self.wave_ratios['wave4_max']:
                    score += 15
                
                # Wave 3 makes new high
                if w3 > w1:
                    score += 10
                
                # Strong moves (6%+ each)
                if wave1_size / w0 * 100 >= 6 and wave3_size / w2 * 100 >= 6:
                    score += 10
                    
            else:  # BEARISH (similar logic inverted)
                w0, w1, w2, w3, w4 = [p['price'] for p in pivots]
                wave1_size = w0 - w1
                wave2_retrace = (w2 - w1) / wave1_size
                wave3_size = w2 - w3
                wave4_retrace = (w4 - w3) / wave3_size
                
                if self.wave_ratios['wave2_min'] <= wave2_retrace <= self.wave_ratios['wave2_max']:
                    score += 15
                if wave3_size >= wave1_size * self.wave_ratios['wave3_min']:
                    score += 20
                if self.wave_ratios['wave4_min'] <= wave4_retrace <= self.wave_ratios['wave4_max']:
                    score += 15
                if w3 < w1:
                    score += 10
                if wave1_size / w0 * 100 >= 6 and wave3_size / w2 * 100 >= 6:
                    score += 10
            
        except Exception as e:
            pass
        
        return min(score, 100)

    def determine_current_wave(self, pattern, current_price):
        """Determine current wave position"""
        try:
            pivots = pattern['pivots']
            pattern_type = pattern['pattern_type']
            
            if pattern_type == 'BULLISH':
                if current_price > pivots[4]['price']:
                    return 'WAVE_5'
                elif current_price > pivots[3]['price']:
                    return 'WAVE_4'
                elif current_price > pivots[2]['price']:
                    return 'WAVE_3'
                else:
                    return 'COMPLETED'
            else:  # BEARISH
                if current_price < pivots[4]['price']:
                    return 'WAVE_5'
                elif current_price < pivots[3]['price']:
                    return 'WAVE_4'
                elif current_price < pivots[2]['price']:
                    return 'WAVE_3'
                else:
                    return 'COMPLETED'
        except:
            return 'UNKNOWN'

    def calculate_targets(self, pattern, current_price):
        """Calculate Elliott Wave targets"""
        try:
            pivots = pattern['pivots']
            pattern_type = pattern['pattern_type']
            
            if pattern_type == 'BULLISH':
                wave1_size = pivots[1]['price'] - pivots[0]['price']
                wave4_price = pivots[4]['price']
                
                target1 = wave4_price + (wave1_size * 0.618)  # 61.8% projection
                target2 = wave4_price + wave1_size            # 100% projection
                target3 = wave4_price + (wave1_size * 1.618)  # 161.8% projection
                
            else:  # BEARISH
                wave1_size = pivots[0]['price'] - pivots[1]['price']
                wave4_price = pivots[4]['price']
                
                target1 = wave4_price - (wave1_size * 0.618)
                target2 = wave4_price - wave1_size
                target3 = wave4_price - (wave1_size * 1.618)
            
            return target1, target2, target3
            
        except:
            return current_price, current_price, current_price

    def analyze_symbol(self, symbol):
        """Analyze single symbol for Elliott Wave patterns"""
        try:
            print(f"📡 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_daily_data(symbol)
            if df is None or len(df) < 50:
                print("No data")
                return []
            
            current_price = df.iloc[-1]['close']
            
            # Find significant pivots
            pivots = self.find_significant_pivots(df)
            if len(pivots) < 5:
                print("Insufficient pivots")
                return []
            
            patterns = []
            
            # Look for patterns in different pivot sequences
            for i in range(len(pivots) - 4):
                pivot_sequence = pivots[i:i+5]
                
                # Try bullish pattern
                bullish_pattern = self.validate_elliott_wave_pattern(pivot_sequence, 'BULLISH')
                if bullish_pattern:
                    current_wave = self.determine_current_wave(bullish_pattern, current_price)
                    target1, target2, target3 = self.calculate_targets(bullish_pattern, current_price)
                    
                    pattern_data = {
                        'symbol': symbol,
                        'pattern_type': 'BULLISH_IMPULSE',
                        'direction': 'BULLISH',
                        'current_wave': current_wave,
                        'quality_score': bullish_pattern['quality_score'],
                        'current_price': current_price,
                        'targets': [target1, target2, target3],
                        'wave_start': pivot_sequence[0]['price'],
                        'wave1_end': pivot_sequence[1]['price'],
                        'wave4_end': pivot_sequence[4]['price'],
                        'duration': (pivot_sequence[4]['date'] - pivot_sequence[0]['date']).days,
                        'start_date': pivot_sequence[0]['date'].strftime('%Y-%m-%d')
                    }
                    patterns.append(pattern_data)
                
                # Try bearish pattern
                bearish_pattern = self.validate_elliott_wave_pattern(pivot_sequence, 'BEARISH')
                if bearish_pattern:
                    current_wave = self.determine_current_wave(bearish_pattern, current_price)
                    target1, target2, target3 = self.calculate_targets(bearish_pattern, current_price)
                    
                    pattern_data = {
                        'symbol': symbol,
                        'pattern_type': 'BEARISH_IMPULSE',
                        'direction': 'BEARISH',
                        'current_wave': current_wave,
                        'quality_score': bearish_pattern['quality_score'],
                        'current_price': current_price,
                        'targets': [target1, target2, target3],
                        'wave_start': pivot_sequence[0]['price'],
                        'wave1_end': pivot_sequence[1]['price'],
                        'wave4_end': pivot_sequence[4]['price'],
                        'duration': (pivot_sequence[4]['date'] - pivot_sequence[0]['date']).days,
                        'start_date': pivot_sequence[0]['date'].strftime('%Y-%m-%d')
                    }
                    patterns.append(pattern_data)
            
            # Remove duplicates and keep highest quality
            if patterns:
                patterns.sort(key=lambda x: x['quality_score'], reverse=True)
                patterns = patterns[:2]  # Max 2 patterns per symbol
                print(f"Found {len(patterns)} patterns")
            else:
                print("No patterns")
                
            return patterns
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all symbols for Elliott Wave patterns"""
        print("🌊 ELLIOTT WAVE SCANNER - DAILY CHART (V8 BALANCED)")
        print("=" * 75)
        print("📈 Scanning daily charts for quality Elliott Wave patterns...")
        print("📊 Strategy: 5-wave impulse patterns → 50-150% major moves")
        print()
        
        symbols = self.get_symbols()
        all_patterns = []
        
        for symbol in symbols:
            try:
                patterns = self.analyze_symbol(symbol)
                all_patterns.extend(patterns)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"📡 {symbol.replace('/USDT', '')}... Error: {e}")
                continue
        
        return all_patterns

    def display_results(self, patterns):
        """Display Elliott Wave scan results"""
        if not patterns:
            print("\n📊 SCAN RESULTS:")
            print("Total scanned: 30")
            print("Quality patterns: 0")
            print("=" * 89)
            print("🌊 FOUND 0 ELLIOTT WAVE PATTERNS (DAILY)")
            print("=" * 89)
            print("❌ No quality Elliott Wave patterns found")
            print("💡 Elliott Wave patterns are rare - try different market conditions")
            return
        
        # Sort by quality score
        patterns.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: 30")
        print(f"Quality patterns: {len(patterns)}")
        print("=" * 89)
        print(f"🌊 FOUND {len(patterns)} ELLIOTT WAVE PATTERNS (DAILY)")
        print("=" * 89)
        
        for i, pattern in enumerate(patterns, 1):
            direction_icon = "📈" if pattern['direction'] == 'BULLISH' else "📉"
            
            print(f"\n{i}. {direction_icon} {pattern['symbol'].replace('/USDT', '')} | Score: {pattern['quality_score']:.0f}/100 | 🌊 {pattern['current_wave']} | {pattern['direction']}")
            print(f"   Current: ${pattern['current_price']:.4f} | Duration: {pattern['duration']} days | Start: {pattern['start_date']}")
            
            targets = pattern['targets']
            print(f"   Targets: T1: ${targets[0]:.2f} | T2: ${targets[1]:.2f} | T3: ${targets[2]:.2f}")
            
            # Wave levels
            print(f"   Wave Start: ${pattern['wave_start']:.2f} | Wave 1: ${pattern['wave1_end']:.2f} | Wave 4: ${pattern['wave4_end']:.2f}")
            
            # Trading guidance
            if pattern['current_wave'] == 'WAVE_4':
                action = "LONG" if pattern['direction'] == 'BULLISH' else "SHORT"
                print(f"   🎯 OPPORTUNITY: Wave 4 correction - {action} setup for major Wave 5!")
            elif pattern['current_wave'] == 'WAVE_5':
                print(f"   ⚠️  WARNING: Wave 5 active - major reversal expected soon!")
            elif pattern['current_wave'] == 'WAVE_3':
                print(f"   🚀 MOMENTUM: Wave 3 in progress - strongest wave continuing!")

def main():
    """Main execution function"""
    try:
        scanner = ElliottWaveScanner()
        patterns = scanner.scan_all_symbols()
        scanner.display_results(patterns)
        
    except KeyboardInterrupt:
        print("\n❌ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan error: {e}")

if __name__ == "__main__":
    main()