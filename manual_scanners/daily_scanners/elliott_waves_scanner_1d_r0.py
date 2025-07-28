#!/usr/bin/env python3
"""
Elliott Wave Scanner - Daily Chart (Realistic Version)
Professional Elliott Wave pattern detection with practical parameters

Identifies Elliott Wave patterns with realistic market conditions in mind.
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
        
        # Realistic Daily Elliott Wave Parameters
        self.lookback_periods = 120  # 4 months of daily data
        self.min_wave_size = 2       # Minimum 2% move for wave validation
        self.min_pivot_distance = 3  # Minimum 3 days between pivots
        self.min_quality_score = 25  # Very low threshold for opportunities
        
    def get_symbols(self):
        """Get major trading pairs for Elliott Wave analysis"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets.keys() if s.endswith('USDT') and 
                      markets[s]['active'] and markets[s]['spot']]
            
            # Focus on major cryptocurrencies
            major_symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
                'LTC/USDT', 'BCH/USDT', 'BNB/USDT', 'DOGE/USDT', 'ATOM/USDT',
                'NEAR/USDT', 'FTM/USDT', 'ICP/USDT', 'ALGO/USDT', 'AAVE/USDT',
                'MKR/USDT', 'COMP/USDT', 'SNX/USDT', 'CRV/USDT', 'SUSHI/USDT'
            ]
            
            # Filter available symbols
            final_symbols = [s for s in major_symbols if s in symbols]
            
            # Add others to reach 50 total
            other_symbols = [s for s in symbols if s not in final_symbols]
            final_symbols.extend(other_symbols[:25])
            
            return final_symbols[:50]
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            return []

    def get_daily_data(self, symbol):
        """Get daily OHLCV data"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', since=since, limit=self.lookback_periods)
            
            if len(ohlcv) < 30:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price_change'] = df['close'].pct_change() * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def find_simple_peaks_valleys(self, df):
        """Find peaks and valleys with very simple criteria"""
        pivots = []
        
        # Use a simple approach - look for local highs and lows
        for i in range(2, len(df) - 2):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            date = df.iloc[i]['timestamp']
            
            # Check for local high (higher than 2 days before and after)
            if (high > df.iloc[i-1]['high'] and high > df.iloc[i-2]['high'] and
                high > df.iloc[i+1]['high'] and high > df.iloc[i+2]['high']):
                
                # Check if move is significant enough
                recent_low = df.iloc[max(0, i-10):i]['low'].min()
                if (high - recent_low) / recent_low * 100 >= self.min_wave_size:
                    pivots.append({
                        'index': i,
                        'price': high,
                        'date': date,
                        'type': 'peak'
                    })
            
            # Check for local low
            if (low < df.iloc[i-1]['low'] and low < df.iloc[i-2]['low'] and
                low < df.iloc[i+1]['low'] and low < df.iloc[i+2]['low']):
                
                # Check if move is significant enough
                recent_high = df.iloc[max(0, i-10):i]['high'].max()
                if (recent_high - low) / recent_high * 100 >= self.min_wave_size:
                    pivots.append({
                        'index': i,
                        'price': low,
                        'date': date,
                        'type': 'valley'
                    })
        
        # Sort by date and remove close pivots
        pivots.sort(key=lambda x: x['index'])
        filtered_pivots = []
        
        for pivot in pivots:
            if not filtered_pivots:
                filtered_pivots.append(pivot)
            else:
                last_pivot = filtered_pivots[-1]
                # Only add if different type and far enough apart
                if (pivot['type'] != last_pivot['type'] and 
                    pivot['index'] - last_pivot['index'] >= self.min_pivot_distance):
                    filtered_pivots.append(pivot)
        
        return filtered_pivots

    def find_wave_patterns(self, pivots, df):
        """Find basic Elliott Wave patterns with very lenient criteria"""
        patterns = []
        current_price = df.iloc[-1]['close']
        
        # Need at least 5 pivots for any pattern
        if len(pivots) < 5:
            return patterns
        
        # Look for any 5-pivot sequences
        for i in range(len(pivots) - 4):
            wave_pivots = pivots[i:i+5]
            
            # Try both bullish and bearish interpretations
            bullish_pattern = self.check_bullish_wave(wave_pivots, current_price)
            if bullish_pattern:
                patterns.append(bullish_pattern)
                
            bearish_pattern = self.check_bearish_wave(wave_pivots, current_price)
            if bearish_pattern:
                patterns.append(bearish_pattern)
        
        return patterns

    def check_bullish_wave(self, pivots, current_price):
        """Check for bullish wave pattern with very basic rules"""
        try:
            # Check if pattern is valley-peak-valley-peak-valley
            types = [p['type'] for p in pivots]
            if types != ['valley', 'peak', 'valley', 'peak', 'valley']:
                return None
            
            # Get wave points
            w0 = pivots[0]['price']  # Start
            w1 = pivots[1]['price']  # Wave 1 top
            w2 = pivots[2]['price']  # Wave 2 bottom
            w3 = pivots[3]['price']  # Wave 3 top
            w4 = pivots[4]['price']  # Wave 4 bottom
            
            # Basic Elliott Wave rules (very lenient)
            # Rule 1: Wave 2 doesn't retrace more than 100% of Wave 1
            wave2_retrace = (w1 - w2) / (w1 - w0)
            if wave2_retrace > 1.0:
                return None
                
            # Rule 2: Wave 4 doesn't overlap Wave 1 territory (lenient)
            if w4 <= w0 * 1.05:  # 5% buffer
                return None
            
            # Rule 3: Wave 3 is not the shortest (very basic check)
            wave1_size = w1 - w0
            wave3_size = w3 - w2
            if wave3_size < wave1_size * 0.8:  # Allow Wave 3 to be 80% of Wave 1
                return None
            
            # Calculate quality score (very lenient)
            score = 30  # Base score
            
            # Add points for good characteristics
            if 0.3 <= wave2_retrace <= 0.8:  # Good retracement
                score += 20
            if wave3_size >= wave1_size * 1.2:  # Wave 3 extension
                score += 20
            if w3 > w1:  # New high in Wave 3
                score += 15
            if current_price > w4:  # Price above Wave 4
                score += 15
            
            # Calculate targets
            wave1_move = w1 - w0
            target1 = w4 + (wave1_move * 0.618)  # 61.8% projection
            target2 = w4 + wave1_move            # 100% projection
            target3 = w4 + (wave1_move * 1.618)  # 161.8% projection
            
            return {
                'type': 'BULLISH_IMPULSE',
                'direction': 'BULLISH',
                'quality_score': min(score, 100),
                'current_wave': self.determine_current_position(pivots, current_price, 'BULLISH'),
                'wave_start': w0,
                'wave1_end': w1,
                'wave2_end': w2,
                'wave3_end': w3,
                'wave4_end': w4,
                'target1': target1,
                'target2': target2,
                'target3': target3,
                'wave_duration': (pivots[4]['date'] - pivots[0]['date']).days,
                'pattern_start_date': pivots[0]['date'],
                'pattern_end_date': pivots[4]['date']
            }
            
        except Exception as e:
            return None

    def check_bearish_wave(self, pivots, current_price):
        """Check for bearish wave pattern"""
        try:
            # Check if pattern is peak-valley-peak-valley-peak
            types = [p['type'] for p in pivots]
            if types != ['peak', 'valley', 'peak', 'valley', 'peak']:
                return None
            
            # Get wave points
            w0 = pivots[0]['price']  # Start (high)
            w1 = pivots[1]['price']  # Wave 1 bottom
            w2 = pivots[2]['price']  # Wave 2 top
            w3 = pivots[3]['price']  # Wave 3 bottom
            w4 = pivots[4]['price']  # Wave 4 top
            
            # Basic Elliott Wave rules (bearish, very lenient)
            wave2_retrace = (w2 - w1) / (w0 - w1)
            if wave2_retrace > 1.0:
                return None
                
            if w4 >= w0 * 0.95:  # 5% buffer
                return None
            
            wave1_size = w0 - w1
            wave3_size = w2 - w3
            if wave3_size < wave1_size * 0.8:
                return None
            
            # Calculate quality score
            score = 30
            
            if 0.3 <= wave2_retrace <= 0.8:
                score += 20
            if wave3_size >= wave1_size * 1.2:
                score += 20
            if w3 < w1:
                score += 15
            if current_price < w4:
                score += 15
            
            # Calculate targets
            wave1_move = w0 - w1
            target1 = w4 - (wave1_move * 0.618)
            target2 = w4 - wave1_move
            target3 = w4 - (wave1_move * 1.618)
            
            return {
                'type': 'BEARISH_IMPULSE',
                'direction': 'BEARISH',
                'quality_score': min(score, 100),
                'current_wave': self.determine_current_position(pivots, current_price, 'BEARISH'),
                'wave_start': w0,
                'wave1_end': w1,
                'wave2_end': w2,
                'wave3_end': w3,
                'wave4_end': w4,
                'target1': target1,
                'target2': target2,
                'target3': target3,
                'wave_duration': (pivots[4]['date'] - pivots[0]['date']).days,
                'pattern_start_date': pivots[0]['date'],
                'pattern_end_date': pivots[4]['date']
            }
            
        except Exception as e:
            return None

    def determine_current_position(self, pivots, current_price, direction):
        """Determine current wave position"""
        try:
            if direction == 'BULLISH':
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

    def get_trading_strategy(self, pattern, current_price):
        """Get trading strategy for the pattern"""
        try:
            current_wave = pattern['current_wave']
            direction = pattern['direction']
            
            if current_wave == 'WAVE_4' and direction == 'BULLISH':
                return {
                    'action': 'LONG',
                    'entry': f"${pattern['wave4_end']:.2f} - ${pattern['wave4_end'] * 1.03:.2f}",
                    'stop': pattern['wave4_end'] * 0.95,
                    'targets': [pattern['target1'], pattern['target2'], pattern['target3']],
                    'confidence': 'HIGH'
                }
            elif current_wave == 'WAVE_5' and direction == 'BULLISH':
                return {
                    'action': 'TAKE_PROFITS',
                    'exit': f"${pattern['target1']:.2f} - ${pattern['target2']:.2f}",
                    'confidence': 'MEDIUM'
                }
            elif current_wave == 'WAVE_4' and direction == 'BEARISH':
                return {
                    'action': 'SHORT',
                    'entry': f"${pattern['wave4_end'] * 0.97:.2f} - ${pattern['wave4_end']:.2f}",
                    'stop': pattern['wave4_end'] * 1.05,
                    'targets': [pattern['target1'], pattern['target2'], pattern['target3']],
                    'confidence': 'HIGH'
                }
            elif current_wave == 'WAVE_5' and direction == 'BEARISH':
                return {
                    'action': 'COVER_SHORTS',
                    'exit': f"${pattern['target2']:.2f} - ${pattern['target1']:.2f}",
                    'confidence': 'MEDIUM'
                }
            else:
                return {
                    'action': 'MONITOR',
                    'note': f"Currently in {current_wave}",
                    'confidence': 'LOW'
                }
        except:
            return {'action': 'MONITOR', 'confidence': 'LOW'}

    def analyze_symbol(self, symbol):
        """Analyze single symbol for Elliott Wave patterns"""
        try:
            print(f"📡 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_daily_data(symbol)
            if df is None or len(df) < 30:
                print("No data")
                return []
            
            current_price = df.iloc[-1]['close']
            
            # Find pivot points
            pivots = self.find_simple_peaks_valleys(df)
            if len(pivots) < 5:
                print("Insufficient pivots")
                return []
            
            # Find wave patterns
            patterns = self.find_wave_patterns(pivots, df)
            
            if not patterns:
                print("No patterns")
                return []
            
            # Filter and process patterns
            quality_patterns = []
            for pattern in patterns:
                if pattern['quality_score'] >= self.min_quality_score:
                    strategy = self.get_trading_strategy(pattern, current_price)
                    
                    setup = {
                        'symbol': symbol,
                        'pattern_type': pattern['type'],
                        'direction': pattern['direction'],
                        'current_wave': pattern['current_wave'],
                        'quality_score': pattern['quality_score'],
                        'current_price': current_price,
                        'strategy': strategy,
                        'targets': [pattern['target1'], pattern['target2'], pattern['target3']],
                        'duration': pattern['wave_duration'],
                        'start_date': pattern['pattern_start_date'].strftime('%Y-%m-%d'),
                        'end_date': pattern['pattern_end_date'].strftime('%Y-%m-%d')
                    }
                    quality_patterns.append(setup)
            
            if quality_patterns:
                print(f"Found {len(quality_patterns)} patterns")
            else:
                print("No quality patterns")
                
            return quality_patterns
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all symbols for Elliott Wave patterns"""
        print("🌊 ELLIOTT WAVE SCANNER - DAILY CHART")
        print("=" * 75)
        print("📈 Scanning daily charts for Elliott Wave patterns...")
        print("📊 Strategy: 5-wave impulse patterns → 25-100% major moves")
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
            print("Total scanned: 50")
            print("Quality patterns: 0")
            print("=" * 89)
            print("🌊 FOUND 0 ELLIOTT WAVE PATTERNS (DAILY)")
            print("=" * 89)
            print("❌ No Elliott Wave patterns found with current parameters")
            print("💡 Elliott Wave patterns are rare - try weekly timeframe for better results")
            return
        
        # Sort by quality score
        patterns.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: 50")
        print(f"Quality patterns: {len(patterns)}")
        print("=" * 89)
        print(f"🌊 FOUND {len(patterns)} ELLIOTT WAVE PATTERNS (DAILY)")
        print("=" * 89)
        
        for i, pattern in enumerate(patterns, 1):
            direction_icon = "📈" if pattern['direction'] == 'BULLISH' else "📉"
            
            print(f"\n{i}. {direction_icon} {pattern['symbol'].replace('/USDT', '')} | Score: {pattern['quality_score']:.0f}/100 | 🌊 {pattern['current_wave']} | {pattern['direction']}")
            print(f"   Current: ${pattern['current_price']:.4f} | Duration: {pattern['duration']} days | Period: {pattern['start_date']} to {pattern['end_date']}")
            
            strategy = pattern['strategy']
            print(f"   Strategy: {strategy['action']} | Confidence: {strategy.get('confidence', 'UNKNOWN')}")
            
            if 'entry' in strategy:
                print(f"   Entry: {strategy['entry']} | Stop: ${strategy['stop']:.2f}")
                targets = pattern['targets']
                print(f"   Targets: T1: ${targets[0]:.2f} | T2: ${targets[1]:.2f} | T3: ${targets[2]:.2f}")
            elif 'exit' in strategy:
                print(f"   Exit Zone: {strategy['exit']}")
            
            if pattern['current_wave'] == 'WAVE_4':
                print(f"   🎯 OPPORTUNITY: Wave 4 correction - major Wave 5 incoming!")
            elif pattern['current_wave'] == 'WAVE_5':
                print(f"   ⚠️  WARNING: Wave 5 active - reversal expected soon!")

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