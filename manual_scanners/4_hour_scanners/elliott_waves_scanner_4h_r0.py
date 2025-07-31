#!/usr/bin/env python3
"""
Elliott Wave Scanner - 4H Chart (Complete Version)
Adapted from working daily scanner elliott_waves_scanner_1d_r4.py

SURGICAL CHANGES for 4H timeframe:
- Timeframe: '1d' → '4h'
- Lookback: 120 days → 30 days  
- Min wave size: 8% → 4%
- Pivot distance: 5 → 3 periods
- Quality threshold: 40 → 35
- Duration calculations: days → hours

ALL OTHER LOGIC PRESERVED from working daily scanner
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ElliottWaveScanner4H:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'rateLimit': 100,
            'enableRateLimit': True,
        })
        
        # 4H Parameters (adapted from daily)
        self.min_wave_size = 4.0          # 4% vs 8% for daily
        self.lookback_days = 30           # 30 vs 120 for daily  
        self.pivot_distance = 3           # 3 vs 5 for daily
        self.min_quality_score = 35       # 35 vs 40 for daily
        self.context_window = 5           # 5 vs 7 for daily
        
        # Elliott Wave rules (same as daily)
        self.wave3_min_ratio = 0.8
        self.wave2_max_retrace = 0.95
        self.wave4_max_retrace = 0.75
        self.wave4_overlap_tolerance = 0.05

    def get_trading_symbols(self):
        """Get high-liquidity symbols (same as daily scanner)"""
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
            'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT',
            'BCH/USDT', 'BNB/USDT', 'DOGE/USDT', 'SHIB/USDT', 'ATOM/USDT',
            'NEAR/USDT', 'ICP/USDT', 'VET/USDT', 'ALGO/USDT', 'XTZ/USDT',
            'AAVE/USDT', 'MKR/USDT', 'COMP/USDT', 'YFI/USDT', 'SNX/USDT',
            'CRV/USDT', 'UMA/USDT', 'SUSHI/USDT', 'KNC/USDT', 'ZRX/USDT',
            'ENJ/USDT', 'MANA/USDT', 'SAND/USDT', 'CHZ/USDT', 'BAT/USDT',
            'NEO/USDT', 'QTUM/USDT', 'IOTA/USDT', 'XLM/USDT', 'ONT/USDT',
            'TRX/USDT', 'ETC/USDT', 'ICX/USDT', 'ZIL/USDT', 'FET/USDT',
            'ZEC/USDT', 'DASH/USDT', 'THETA/USDT', 'ONE/USDT', 'ANKR/USDT',
            'ARB/USDT', 'OP/USDT', 'MATIC/USDT', 'FTM/USDT', 'HBAR/USDT'
        ]
        return symbols

    def get_4h_data(self, symbol, days=30):
        """Get 4H OHLCV data (adapted from daily scanner)"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe='4h',  # Changed from '1d'
                since=int(start_time.timestamp() * 1000),
                limit=1000
            )
            
            if not ohlcv:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            return None

    def find_major_pivots(self, df):
        """Find major swing pivots (adapted from daily scanner)"""
        if len(df) < self.context_window * 2 + 1:
            return []
        
        pivots = []
        
        # Find initial pivot candidates
        for i in range(self.context_window, len(df) - self.context_window):
            window_highs = df['high'].iloc[i-self.context_window:i+self.context_window+1]
            window_lows = df['low'].iloc[i-self.context_window:i+self.context_window+1]
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for major highs
            if current_high == window_highs.max():
                pivots.append({
                    'type': 'high',
                    'price': current_high,
                    'index': i,
                    'date': df.iloc[i]['timestamp']
                })
            
            # Check for major lows
            if current_low == window_lows.min():
                pivots.append({
                    'type': 'low',
                    'price': current_low,
                    'index': i,
                    'date': df.iloc[i]['timestamp']
                })
        
        # Sort by date and filter
        pivots = sorted(pivots, key=lambda x: x['date'])
        
        # Remove consecutive same-type pivots and apply minimum distance
        filtered_pivots = []
        for pivot in pivots:
            if not filtered_pivots:
                filtered_pivots.append(pivot)
                continue
            
            last_pivot = filtered_pivots[-1]
            
            # Check minimum distance (4H: 3 periods vs daily: 5 periods)
            index_diff = abs(pivot['index'] - last_pivot['index'])
            if index_diff < self.pivot_distance:
                continue
            
            # Avoid consecutive same types
            if pivot['type'] != last_pivot['type']:
                filtered_pivots.append(pivot)
            else:
                # Keep the more significant pivot
                if pivot['type'] == 'high' and pivot['price'] > last_pivot['price']:
                    filtered_pivots[-1] = pivot
                elif pivot['type'] == 'low' and pivot['price'] < last_pivot['price']:
                    filtered_pivots[-1] = pivot
        
        return filtered_pivots

    def identify_current_elliott_structure(self, pivots, df):
        """Identify current Elliott Wave structures (same logic as daily)"""
        if len(pivots) < 5:
            return []
        
        patterns = []
        current_price = df.iloc[-1]['close']
        
        # Look for 5-wave Elliott patterns
        for i in range(len(pivots) - 4):
            wave_pivots = pivots[i:i+5]
            
            # Check for valid wave sequence
            types = [p['type'] for p in wave_pivots]
            if not self.is_valid_elliott_sequence(types):
                continue
            
            # Validate Elliott Wave rules
            if not self.validate_elliott_wave_rules(wave_pivots):
                continue
                
            # Calculate pattern metrics
            pattern = self.analyze_elliott_pattern(wave_pivots, current_price, df)
            if pattern and pattern['quality_score'] >= self.min_quality_score:
                patterns.append(pattern)
        
        return patterns

    def is_pattern_fresh(self, pattern, df):
        """Validate pattern is current and actionable (not historical)"""
        try:
            current_price = pattern['current_price']
            targets = pattern['targets']
            wave_start_date = pattern['wave_start']['date']
            
            # Check pattern age (shouldn't be too old)
            hours_since_start = (df.iloc[-1]['timestamp'] - wave_start_date).total_seconds() / 3600
            if hours_since_start > 720:  # More than 30 days old
                return False
            
            # Check if targets still make sense
            if pattern['direction'] == 'BULLISH':
                # If current price is significantly above all targets, pattern is done
                if current_price > targets['T3'] * 1.05:  # 5% above max target
                    return False
                # If current price is below wave start, pattern may be invalidated  
                if current_price < pattern['wave_start']['price'] * 0.9:
                    return False
            else:  # Bearish
                # Similar logic for bearish patterns
                if current_price < targets['T3'] * 0.95:  # 5% below min target
                    return False
                if current_price > pattern['wave_start']['price'] * 1.1:
                    return False
            
            return True
            
        except:
            return False

    def is_valid_elliott_sequence(self, types):
        """Check if sequence forms valid Elliott Wave (same as daily)"""
        bullish = ['low', 'high', 'low', 'high', 'low']
        bearish = ['high', 'low', 'high', 'low', 'high']
        return types == bullish or types == bearish

    def validate_elliott_wave_rules(self, wave_pivots):
        """Validate core Elliott Wave rules (same as daily)"""
        try:
            prices = [p['price'] for p in wave_pivots]
            
            if wave_pivots[0]['type'] == 'low':  # Bullish pattern
                wave1_start, wave1_end = prices[0], prices[1]
                wave2_start, wave2_end = prices[1], prices[2]
                wave3_start, wave3_end = prices[2], prices[3]
                wave4_start, wave4_end = prices[3], prices[4]
                
                # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
                wave1_size = wave1_end - wave1_start
                wave2_retrace = (wave2_start - wave2_end) / wave1_size
                if wave2_retrace >= self.wave2_max_retrace:
                    return False
                
                # Rule 2: Wave 3 cannot be shortest
                wave3_size = wave3_end - wave3_start
                if wave3_size < wave1_size * self.wave3_min_ratio:
                    return False
                
                # Rule 3: Wave 4 cannot overlap Wave 1 significantly
                if wave4_end <= wave1_end * (1 + self.wave4_overlap_tolerance):
                    return False
                
                # Check minimum wave sizes (4H: 4% vs daily: 8%)
                wave1_pct = wave1_size / wave1_start * 100
                wave3_pct = wave3_size / wave3_start * 100
                if wave1_pct < self.min_wave_size or wave3_pct < self.min_wave_size:
                    return False
                    
            else:  # Bearish pattern
                wave1_start, wave1_end = prices[0], prices[1]
                wave2_start, wave2_end = prices[1], prices[2]
                wave3_start, wave3_end = prices[2], prices[3]
                wave4_start, wave4_end = prices[3], prices[4]
                
                # Same rules for bearish (inverted)
                wave1_size = wave1_start - wave1_end
                wave2_retrace = (wave2_end - wave2_start) / wave1_size
                if wave2_retrace >= self.wave2_max_retrace:
                    return False
                
                wave3_size = wave3_start - wave3_end
                if wave3_size < wave1_size * self.wave3_min_ratio:
                    return False
                
                if wave4_end >= wave1_end * (1 - self.wave4_overlap_tolerance):
                    return False
                
                # Check minimum wave sizes
                wave1_pct = wave1_size / wave1_start * 100
                wave3_pct = wave3_size / wave3_start * 100
                if wave1_pct < self.min_wave_size or wave3_pct < self.min_wave_size:
                    return False
            
            return True
            
        except:
            return False

    def analyze_elliott_pattern(self, wave_pivots, current_price, df):
        """Analyze Elliott Wave pattern (adapted from daily)"""
        try:
            prices = [p['price'] for p in wave_pivots]
            dates = [p['date'] for p in wave_pivots]
            
            # Determine direction
            direction = 'BULLISH' if wave_pivots[0]['type'] == 'low' else 'BEARISH'
            
            # Calculate wave characteristics with proper duration
            waves = []
            for i in range(1, len(prices)):
                start_price = prices[i-1]
                end_price = prices[i]
                start_date = dates[i-1]
                end_date = dates[i]
                move_size = abs(end_price - start_price) / start_price * 100
                duration_hours = (end_date - start_date).total_seconds() / 3600
                
                waves.append({
                    'start': {'price': start_price, 'date': start_date},
                    'end': {'price': end_price, 'date': end_date},
                    'size': move_size,
                    'duration_hours': duration_hours
                })
            
            # Determine current wave position (fixed logic)
            current_wave = self.identify_current_wave_position(wave_pivots, current_price, df)
            
            # Filter out invalidated or completed patterns
            if current_wave in ['PATTERN_TOO_OLD', 'PATTERN_INVALIDATED', 'PATTERN_COMPLETE_NEW_CYCLE']:
                return None
            
            # Calculate targets
            targets = self.calculate_fibonacci_targets(wave_pivots, direction, current_price)
            
            # Calculate quality score
            quality_score = self.calculate_pattern_quality(waves, direction)
            
            # Calculate duration (4H: hours vs daily: days)
            duration_hours = (dates[-1] - dates[0]).total_seconds() / 3600
            
            return {
                'type': 'ELLIOTT_WAVE',
                'direction': direction,
                'current_wave': current_wave,
                'quality_score': quality_score,
                'current_price': current_price,
                'targets': targets,
                'duration': duration_hours,
                'wave_start': {'price': prices[0], 'date': dates[0]},
                'waves': waves
            }
            
        except Exception as e:
            return None

    def identify_current_wave_position(self, wave_pivots, current_price, df):
        """Identify current wave position (fixed logic)"""
        try:
            last_pivot = wave_pivots[-1]
            prices = [p['price'] for p in wave_pivots]
            
            # Check if pattern is invalidated by price action
            if wave_pivots[0]['type'] == 'low':  # Bullish pattern
                wave3_high = prices[3]  # Wave 3 peak
                wave4_low = prices[4]   # Wave 4 low
                
                # If current price is significantly above Wave 3 high, pattern may be complete/invalidated
                if current_price > wave3_high * 1.02:  # 2% above Wave 3 high
                    return 'PATTERN_COMPLETE_NEW_CYCLE'
                    
                # If current price is below Wave 4 low, pattern invalidated
                if current_price < wave4_low * 0.95:  # 5% below Wave 4 low
                    return 'PATTERN_INVALIDATED'
                    
            else:  # Bearish pattern
                wave3_low = prices[3]   # Wave 3 bottom
                wave4_high = prices[4]  # Wave 4 high
                
                # If current price is significantly below Wave 3 low, pattern may be complete
                if current_price < wave3_low * 0.98:  # 2% below Wave 3 low
                    return 'PATTERN_COMPLETE_NEW_CYCLE'
                    
                # If current price is above Wave 4 high, pattern invalidated
                if current_price > wave4_high * 1.05:  # 5% above Wave 4 high
                    return 'PATTERN_INVALIDATED'
            
            # Get recent bars since last pivot for timing analysis
            recent_data = df[df['timestamp'] > last_pivot['date']]
            bars_since_pivot = len(recent_data)
            price_change = (current_price - last_pivot['price']) / last_pivot['price'] * 100
            
            # Pattern age check
            if bars_since_pivot > 168:  # More than 4 weeks old
                return 'PATTERN_TOO_OLD'
            
            # Current position analysis
            if abs(price_change) < 1.0:  # Very close to last pivot
                return 'WAVE_5_COMPLETION'
            elif bars_since_pivot > 48:  # Been developing for a while
                return 'WAVE_5_EXTENSION'
            else:
                return 'WAVE_4_OR_5'
                
        except:
            return 'UNKNOWN'

    def calculate_fibonacci_targets(self, wave_pivots, direction, current_price):
        """Calculate proper Elliott Wave Fibonacci targets"""
        try:
            prices = [p['price'] for p in wave_pivots]
            
            if direction == 'BULLISH':
                wave1_start, wave1_end = prices[0], prices[1]  # Wave 1: $0.62 → $0.67
                wave3_start, wave3_end = prices[2], prices[3]  # Wave 3: $0.64 → $1.23
                wave4_start, wave4_end = prices[3], prices[4]  # Wave 4: $1.23 → $0.84
                
                wave1_size = wave1_end - wave1_start  # $0.05
                wave3_size = wave3_end - wave3_start  # $0.59
                
                # Wave 5 projections from Wave 4 low
                wave5_start = wave4_end  # $0.84
                
                # Multiple projection methods
                # Method 1: Wave 5 = 0.618 to 1.618 x Wave 1 from Wave 4 low
                fib_618 = wave5_start + (wave1_size * 0.618)
                fib_100 = wave5_start + (wave1_size * 1.0)
                fib_162 = wave5_start + (wave1_size * 1.618)
                
                # Method 2: Wave 5 = percentage of Wave 3 size from Wave 4 low
                wave3_618 = wave5_start + (wave3_size * 0.618)
                wave3_100 = wave5_start + (wave3_size * 1.0)
                
                # Method 3: Ensure Wave 5 at least reaches Wave 3 high
                wave3_high = wave3_end
                
                # Take the most conservative realistic targets
                target1 = max(fib_100, wave3_618, current_price * 1.02)
                target2 = max(fib_162, wave3_100, wave3_high)
                target3 = max(wave3_100 * 1.2, target2 * 1.1)
                
            else:  # Bearish
                wave1_start, wave1_end = prices[0], prices[1]
                wave3_start, wave3_end = prices[2], prices[3]
                wave4_start, wave4_end = prices[3], prices[4]
                
                wave1_size = wave1_start - wave1_end
                wave3_size = wave3_start - wave3_end
                wave5_start = wave4_end
                
                # Bearish projections
                fib_618 = wave5_start - (wave1_size * 0.618)
                fib_100 = wave5_start - (wave1_size * 1.0)
                fib_162 = wave5_start - (wave1_size * 1.618)
                
                wave3_618 = wave5_start - (wave3_size * 0.618)
                wave3_100 = wave5_start - (wave3_size * 1.0)
                wave3_low = wave3_end
                
                target1 = min(fib_100, wave3_618, current_price * 0.98)
                target2 = min(fib_162, wave3_100, wave3_low)
                target3 = min(wave3_100 * 0.8, target2 * 0.9)
                
                # Realistic bearish floors
                target1 = max(target1, current_price * 0.7)
                target2 = max(target2, current_price * 0.6)
                target3 = max(target3, current_price * 0.5)
            
            return {
                'T1': round(target1, 4),
                'T2': round(target2, 4),
                'T3': round(target3, 4)
            }
            
        except:
            return {'T1': 0, 'T2': 0, 'T3': 0}

    def calculate_pattern_quality(self, waves, direction):
        """Calculate quality score (adapted from daily)"""
        score = 50  # Base score (vs 60 for daily)
        
        try:
            # Wave ratios and Fibonacci compliance (same as daily)
            if len(waves) >= 2:
                wave1_size = waves[0]['size']
                wave2_size = waves[1]['size']
                wave2_retrace = wave2_size / wave1_size * 100
                
                if 38 <= wave2_retrace <= 62:
                    score += 15
                elif 25 <= wave2_retrace <= 75:
                    score += 8
            
            if len(waves) >= 3:
                wave1_size = waves[0]['size']
                wave3_size = waves[2]['size']
                
                # Strong Wave 3 (4H adapted)
                if wave3_size >= wave1_size * 1.2:
                    score += 12
                elif wave3_size >= wave1_size:
                    score += 8
                
                # Wave 3 momentum (4H: 6% vs daily: 15%)
                if wave3_size >= 6:
                    score += 8
                elif wave3_size >= 4:
                    score += 5
            
            # Pattern completion (same as daily)
            if len(waves) >= 4:
                score += 8
                
        except:
            pass
        
        return min(score, 100)

    def analyze_symbol(self, symbol):
        """Analyze symbol for Elliott patterns (same structure as daily)"""
        try:
            print(f"📡 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_4h_data(symbol, self.lookback_days)
            if df is None or len(df) < 20:
                print("No data")
                return []
            
            current_price = df.iloc[-1]['close']
            
            # Find major pivots
            pivots = self.find_major_pivots(df)
            if len(pivots) < 5:
                print("Insufficient pivots")
                return []
            
            # Identify Elliott structures
            patterns = self.identify_current_elliott_structure(pivots, df)
            
            if not patterns:
                print("No patterns")
                return []
            
            # Format results (fix the symbol error)
            results = []
            for pattern in patterns:
                pattern['symbol'] = symbol
                pattern['current_price'] = current_price
                results.append(pattern)
            
            print(f"Found {len(results)} patterns")
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all symbols (same structure as daily)"""
        print("🌊 ELLIOTT WAVE SCANNER - 4H CHART")
        print("=" * 75)
        print("📈 Scanning 4H charts for Elliott Wave patterns...")
        print("📊 Strategy: Swing wave detection → 15-60% moves")
        
        symbols = self.get_trading_symbols()
        all_patterns = []
        
        for symbol in symbols:
            try:
                patterns = self.analyze_symbol(symbol)
                all_patterns.extend(patterns)
            except Exception as e:
                print(f"📡 {symbol.replace('/USDT', '')}... Error: {e}")
                continue
        
        return all_patterns

    def display_complete_wave_analysis(self, pattern):
        """Display complete wave analysis (full methodology from daily scanner)"""
        waves = pattern['waves']
        current_price = pattern['current_price']
        targets = pattern['targets']
        direction = pattern['direction']
        current_wave = pattern['current_wave']
        
        # Enhanced wave structure analysis
        print(f"\n   🌊 COMPLETE WAVE ANALYSIS & TRADING STRATEGY:")
        
        # Current position with detailed guidance
        if current_wave == 'WAVE_3':
            print(f"   🚀 CURRENT POSITION: Wave 3 in progress - strongest wave active!")
            if direction == 'BULLISH':
                print(f"   🎯 STRATEGY: HOLD and ADD on any dips")
                print(f"   📈 Add Zone: ${current_price * 0.98:.2f} - ${current_price * 0.95:.2f}")
                print(f"   💰 Wave 3 Target: ${targets['T1']:.2f}")
                print(f"   🎯 Wave 4 Correction: Expected to ${targets['T1'] * 0.9:.2f} - ${targets['T1'] * 0.85:.2f}")
                print(f"   🚀 Ultimate Wave 5: ${targets['T2']:.2f} - ${targets['T3']:.2f}")
        
        elif current_wave == 'WAVE_4_OR_5':
            print(f"   🎯 CURRENT POSITION: Wave 4 correction or early Wave 5")
            if direction == 'BULLISH':
                print(f"   💡 IF OWNED: HOLD - Wave 4 corrections are buying opportunities")
                print(f"   💰 IF NOT OWNED: WAIT for Wave 4 dip or enter small position")
                print(f"   📈 Ideal Entry: ${current_price * 0.96:.2f} - ${current_price * 0.92:.2f}")
                print(f"   🎯 Wave 5 Targets: T1: ${targets['T1']:.2f} | T2: ${targets['T2']:.2f} | T3: ${targets['T3']:.2f}")
                print(f"   🛡️ Stop Loss: ${current_price * 0.88:.2f}")
        
        elif current_wave == 'WAVE_5_COMPLETION':
            print(f"   ⚠️ CURRENT POSITION: Wave 5 completion - reversal expected")
            if direction == 'BULLISH':
                print(f"   💰 IF OWNED: TAKE PROFITS NOW - Wave 5 complete")
                print(f"   📉 IF NOT OWNED: AVOID or prepare for reversal")
                print(f"   🎯 Exit Levels: 75% at ${targets['T1']:.2f}, 25% at ${targets['T2']:.2f}")
                print(f"   🔄 Reversal Target: ${current_price * 0.7:.2f} - ${current_price * 0.6:.2f}")
        
        elif current_wave == 'WAVE_5_EXTENSION':
            print(f"   🚨 CURRENT POSITION: Wave 5 extending - extreme caution!")
            if direction == 'BULLISH':
                print(f"   ⚠️ IF OWNED: REDUCE position - extension signals top")
                print(f"   ❌ IF NOT OWNED: DO NOT ENTER - too risky")
                print(f"   💰 Take 50% profits NOW, 50% at ${targets['T2']:.2f}")
                print(f"   🔄 Major reversal expected to ${current_price * 0.6:.2f} - ${current_price * 0.5:.2f}")
        
        # Risk management matrix (4H specific)
        print(f"\n   📊 4H SWING RISK MANAGEMENT:")
        confidence = "HIGH" if pattern['quality_score'] >= 70 else "MEDIUM" if pattern['quality_score'] >= 50 else "LOW"
        
        if confidence == "HIGH":
            position_size = "10-15%"
            stop_distance = "4-6%"
        elif confidence == "MEDIUM":
            position_size = "5-10%" 
            stop_distance = "3-5%"
        else:
            position_size = "2-5%"
            stop_distance = "2-4%"
            
        print(f"   💎 Position Size: {position_size} of portfolio")
        print(f"   🛡️ Stop Distance: {stop_distance} from entry")
        print(f"   ⏰ Hold Duration: 1-14 days (4H swing)")
        print(f"   📊 Confidence Level: {confidence}")
        
        # Wave momentum analysis
        if len(waves) >= 3:
            wave1_size = waves[0]['size']
            wave3_size = waves[2]['size']
            wave3_strength = wave3_size / wave1_size if wave1_size > 0 else 0
            
            print(f"\n   ⚡ WAVE MOMENTUM ANALYSIS:")
            print(f"   📊 Wave 3 Strength: {wave3_strength:.1f}x Wave 1 ({'STRONG' if wave3_strength > 1.5 else 'MODERATE' if wave3_strength > 1.0 else 'WEAK'})")
            
            if wave3_strength > 2.0:
                print(f"   🚀 EXCEPTIONAL: Wave 3 > 2x Wave 1 - very bullish")
            elif wave3_strength > 1.6:
                print(f"   💪 STRONG: Wave 3 > 1.6x Wave 1 - bullish momentum")
            elif wave3_strength > 1.0:
                print(f"   ✅ VALID: Wave 3 > Wave 1 - meets Elliott rules")
            else:
                print(f"   ⚠️ WEAK: Wave 3 < Wave 1 - questionable pattern")

    def display_results(self, patterns):
        """Display results (same format as daily scanner)"""
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {len(self.get_trading_symbols())}")
        print(f"4H patterns: {len(patterns)}")
        
        if not patterns:
            print("\n" + "=" * 75)
            print("🌊 FOUND 0 ELLIOTT WAVE PATTERNS (4H)")
            print("=" * 75)
            print("❌ No high-quality 4H Elliott Wave patterns found")
            print("💡 Try daily timeframe for longer-term patterns")
            return
        
        # Sort by quality score
        patterns = sorted(patterns, key=lambda x: x['quality_score'], reverse=True)
        
        print("\n" + "=" * 75)
        print(f"🌊 FOUND {len(patterns)} ELLIOTT WAVE PATTERNS (4H)")
        print("=" * 75)
        
        for i, pattern in enumerate(patterns, 1):
            symbol = pattern['symbol'].replace('/USDT', '')
            direction_emoji = "📈" if pattern['direction'] == 'BULLISH' else "📉"
            
            print(f"{i}. {direction_emoji} {symbol} | Score: {pattern['quality_score']}/100 | "
                  f"🌊 {pattern['current_wave']} | {pattern['direction']}")
            print(f"   Current: ${pattern['current_price']:.2f} | "
                  f"Duration: {pattern['duration']:.1f}h | "
                  f"Start: {pattern['wave_start']['date'].strftime('%Y-%m-%d')}")
            print(f"   Wave Start: ${pattern['wave_start']['price']:.2f}")
            print(f"   Targets: T1: ${pattern['targets']['T1']:.2f} | "
                  f"T2: ${pattern['targets']['T2']:.2f} | "
                  f"T3: ${pattern['targets']['T3']:.2f}")
            
            # Detailed wave analysis breakdown (RESTORED)
            waves = pattern['waves']
            if len(waves) >= 3:
                print(f"   📊 DETAILED WAVE BREAKDOWN:")
                for j, wave in enumerate(waves, 1):
                    wave_type = "📈" if j % 2 == 1 else "📉"
                    duration_days = wave['duration_hours'] / 24
                    print(f"      Wave {j}: {wave_type} {wave['size']:.1f}% move | "
                          f"Duration: {duration_days:.1f} days | "
                          f"${wave['start']['price']:.2f} → ${wave['end']['price']:.2f}")
                
                # Wave relationship analysis
                w1_size = waves[0]['size']
                w2_retrace = waves[1]['size'] / waves[0]['size'] * 100 if waves[0]['size'] > 0 else 0
                w3_size = waves[2]['size']
                w3_ratio = w3_size / w1_size if w1_size > 0 else 0
                
                print(f"   📈 WAVE RELATIONSHIPS:")
                print(f"      Wave 2 Retrace: {w2_retrace:.1f}% of Wave 1 ({'✅ HEALTHY' if 30 <= w2_retrace <= 70 else '⚠️ EXTREME'})")
                print(f"      Wave 3 vs Wave 1: {w3_ratio:.1f}x ({'🚀 STRONG' if w3_ratio > 1.6 else '✅ VALID' if w3_ratio > 1.0 else '⚠️ WEAK'})")
                
                if len(waves) >= 4:
                    w4_retrace = waves[3]['size'] / waves[2]['size'] * 100 if waves[2]['size'] > 0 else 0
                    print(f"      Wave 4 Retrace: {w4_retrace:.1f}% of Wave 3 ({'✅ HEALTHY' if 20 <= w4_retrace <= 60 else '⚠️ EXTREME'})")
                
                if len(waves) >= 5:
                    w5_size = waves[4]['size']
                    w5_ratio = w5_size / w1_size if w1_size > 0 else 0
                    print(f"      Wave 5 vs Wave 1: {w5_ratio:.1f}x ({'✅ TYPICAL' if 0.6 <= w5_ratio <= 1.6 else '⚠️ ATYPICAL'})")
            
            # Current wave status (same as daily)
            if pattern['current_wave'] == 'WAVE_3':
                print(f"   🚀 MOMENTUM: Wave 3 in progress - strongest wave active!")
            elif pattern['current_wave'] == 'WAVE_4_OR_5':
                print(f"   🎯 TRANSITION: Wave 4 correction or early Wave 5")
            elif pattern['current_wave'] == 'WAVE_5_COMPLETION':
                print(f"   ⚠️ COMPLETION: Wave 5 finished - expect reversal")
            
            # Complete wave analysis and trading strategy
            self.display_complete_wave_analysis(pattern)
            
            print("-" * 75)

def main():
    """Main execution function"""
    try:
        scanner = ElliottWaveScanner4H()
        patterns = scanner.scan_all_symbols()
        scanner.display_results(patterns)
        
    except KeyboardInterrupt:
        print("\n❌ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan error: {e}")

if __name__ == "__main__":
    main()