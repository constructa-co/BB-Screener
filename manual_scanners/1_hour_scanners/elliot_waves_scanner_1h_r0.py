#!/usr/bin/env python3
"""
Elliott Wave Scanner - 1H Chart (Surgical Changes Only)
Professional Elliott Wave pattern detection focused on current major cycles

SURGICAL CHANGES from working 4H scanner:
- Timeframe: '4h' → '1h'
- Lookback: 30 → 10 days
- Min wave size: 4% → 0.4% (ultra-sensitive for 1H)
- Volume threshold: $100K → $10K (ultra-inclusive)
- Pivot detection: Ultra-aggressive for 1H patterns

ALL ELLIOTT WAVE LOGIC IDENTICAL to working daily/4H scanners
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
        
        # 1H parameters - SURGICAL CHANGES ONLY
        self.lookback_periods = 10   # 10 days (vs 30 for 4H)
        self.min_wave_size = 0.4     # 0.4% (vs 4% for 4H) - ultra-sensitive
        self.min_pivot_distance = 1  # 1 (vs 3 for 4H) - allow closer pivots
        self.min_quality_score = 20  # 20 (vs 35 for 4H) - lower threshold
        self.major_move_threshold = 1  # 1% (vs 8% for 4H) - much lower
        
    def get_symbols(self):
        """Get all major trading pairs based on market cap and liquidity"""
        try:
            markets = self.exchange.load_markets()
            
            # Get all active USDT pairs
            usdt_pairs = []
            for symbol, market in markets.items():
                if (symbol.endswith('USDT') and 
                    market['active'] and 
                    market['spot'] and
                    market.get('type') == 'spot'):
                    usdt_pairs.append(symbol)
            
            print(f"📊 Found {len(usdt_pairs)} total USDT pairs on Binance")
            
            # Filter by volume and market cap proxies
            viable_pairs = []
            for symbol in usdt_pairs:
                try:
                    # Get 24h ticker data for volume filtering
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    # Volume filtering (24h volume in USDT)
                    volume_24h = ticker.get('quoteVolume', 0)
                    price = ticker.get('last', 0)
                    
                    # Minimum criteria for 1H analysis - SURGICAL CHANGE
                    min_volume_24h = 10000    # $10K (vs $100K for 4H) - ultra-inclusive
                    min_price = 0.00001      # Avoid micro-cap tokens
                    max_price = 100000       # Reasonable price range
                    
                    if (volume_24h >= min_volume_24h and 
                        min_price <= price <= max_price):
                        viable_pairs.append({
                            'symbol': symbol,
                            'volume_24h': volume_24h,
                            'price': price,
                            'market_cap_proxy': volume_24h * price  # Rough proxy
                        })
                        
                except Exception as e:
                    # Skip pairs that error out (delisted, etc.)
                    continue
            
            # Sort by volume (liquidity proxy) descending
            viable_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            
            # Take top liquid pairs for 1H analysis
            top_pairs = viable_pairs[:200]  # 200 (vs 100 for 4H)
            
            print(f"📊 Filtered to {len(top_pairs)} high-liquidity pairs for 1H analysis")
            print(f"📊 Volume range: ${top_pairs[-1]['volume_24h']:,.0f} - ${top_pairs[0]['volume_24h']:,.0f}")
            
            # Return just the symbols
            return [pair['symbol'] for pair in top_pairs]
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            print("📊 Falling back to major cryptocurrencies...")
            
            # Fallback to major known pairs
            major_symbols = [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
                'LTC/USDT', 'BCH/USDT', 'BNB/USDT', 'DOGE/USDT', 'ATOM/USDT',
                'NEAR/USDT', 'FTM/USDT', 'ICP/USDT', 'ALGO/USDT', 'AAVE/USDT'
            ]
            return major_symbols

    def get_daily_data(self, symbol):
        """Get 1H OHLCV data - SURGICAL CHANGE: timeframe only"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=since, limit=self.lookback_periods * 24 * 2)  # SURGICAL CHANGE: '1h'
            
            if len(ohlcv) < 30:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators (IDENTICAL to 4H)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Calculate percentage moves for major structure identification
            df['price_change_pct'] = df['close'].pct_change() * 100
            df['cumulative_change'] = ((df['close'] / df['close'].iloc[0]) - 1) * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def find_major_pivots(self, df):
        """Find major pivots - SURGICAL CHANGES for 1H sensitivity"""
        pivots = []
        
        # First, identify the major trend direction and key levels
        start_price = df.iloc[0]['close']
        current_price = df.iloc[-1]['close']
        total_move = (current_price - start_price) / start_price * 100
        
        # SURGICAL CHANGE: Ultra-aggressive pivot detection for 1H
        for i in range(1, len(df) - 1):  # 1-period window (vs 2 for 4H)
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            date = df.iloc[i]['timestamp']
            
            # Check for peaks - ULTRA AGGRESSIVE: ANY local high
            window_highs = df.iloc[i-1:i+2]['high']  # 1+1 window
            if high == window_highs.max():  # Must be THE highest in window (no tolerance needed)
                # NO MOVE SIZE REQUIREMENT - any local high counts
                pivots.append({
                    'index': i,
                    'price': high,
                    'date': date,
                    'type': 'peak',
                    'move_size': 0,  # Don't calculate move size
                    'significance': self.calculate_pivot_significance(df, i, 'peak')
                })
            
            # Check for valleys - ULTRA AGGRESSIVE: ANY local low
            window_lows = df.iloc[i-1:i+2]['low']  # 1+1 window  
            if low == window_lows.min():  # Must be THE lowest in window (no tolerance needed)
                # NO MOVE SIZE REQUIREMENT - any local low counts
                pivots.append({
                    'index': i,
                    'price': low,
                    'date': date,
                    'type': 'valley',
                    'move_size': 0,  # Don't calculate move size
                    'significance': self.calculate_pivot_significance(df, i, 'valley')
                })
        
        # ULTRA AGGRESSIVE: Remove ALL filtering - keep most pivots
        pivots.sort(key=lambda x: x['index'])  # Just sort by time
        
        # Only remove exact duplicates at same index
        filtered_pivots = []
        for pivot in pivots:
            is_duplicate = False
            for existing in filtered_pivots:
                if pivot['index'] == existing['index'] and pivot['type'] == existing['type']:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_pivots.append(pivot)
        
        return filtered_pivots

    def calculate_pivot_significance(self, df, index, pivot_type):
        """Calculate how significant a pivot is - IDENTICAL to 4H"""
        try:
            price = df.iloc[index]['high'] if pivot_type == 'peak' else df.iloc[index]['low']
            volume = df.iloc[index]['volume']
            volume_ratio = volume / df['volume'].mean()
            
            # Base significance
            significance = 50
            
            # Volume significance
            if volume_ratio > 2.0:
                significance += 20
            elif volume_ratio > 1.5:
                significance += 10
            
            # Price context significance
            if pivot_type == 'peak':
                # How high is this relative to recent action
                recent_range = df.iloc[max(0, index-30):index+1]
                if price >= recent_range['high'].quantile(0.95):
                    significance += 20
                elif price >= recent_range['high'].quantile(0.85):
                    significance += 10
            else:
                recent_range = df.iloc[max(0, index-30):index+1]
                if price <= recent_range['low'].quantile(0.05):
                    significance += 20
                elif price <= recent_range['low'].quantile(0.15):
                    significance += 10
            
            # Time context - pivots in the middle of our timeframe are more relevant
            time_factor = 1.0 - abs((index / len(df)) - 0.5)
            significance += time_factor * 10
            
            return significance
            
        except:
            return 50

    def identify_current_elliott_structure(self, pivots, df):
        """Identify current Elliott Wave structure - FOCUS ON ACTIVE STRUCTURES"""
        if len(pivots) < 3:
            return None
        
        current_price = df.iloc[-1]['close']
        patterns = []
        
        # Look for the most recent major structure
        # Start from the most significant low or high in our timeframe
        major_lows = [p for p in pivots if p['type'] == 'valley']
        major_highs = [p for p in pivots if p['type'] == 'peak']
        
        if not major_lows or not major_highs:
            return patterns
        
        # Sort by significance and recency
        major_lows.sort(key=lambda x: (x['significance'], -x['index']), reverse=True)
        major_highs.sort(key=lambda x: (x['significance'], -x['index']), reverse=True)
        
        # Try to identify the current major trend structure
        recent_low = major_lows[0] if major_lows else None
        recent_high = major_highs[0] if major_highs else None
        
        if recent_low and recent_high:
            if recent_low['index'] < recent_high['index']:
                # Potential bullish structure
                pattern = self.analyze_bullish_structure(pivots, df, recent_low, recent_high)
                if pattern:
                    patterns.append(pattern)
            else:
                # Potential bearish structure
                pattern = self.analyze_bearish_structure(pivots, df, recent_high, recent_low)
                if pattern:
                    patterns.append(pattern)
        
        return patterns

    def analyze_bullish_structure(self, pivots, df, major_low, major_high):
        """Analyze potential bullish Elliott Wave structure - SURGICAL CHANGES for 1H"""
        try:
            current_price = df.iloc[-1]['close']
            
            # Find sequence of pivots from major low
            sequence = []
            for pivot in pivots:
                if pivot['index'] >= major_low['index']:
                    sequence.append(pivot)
            
            sequence.sort(key=lambda x: x['index'])
            
            if len(sequence) < 3:
                return None
            
            # Identify wave structure
            wave_start = major_low
            potential_waves = []
            
            for i, pivot in enumerate(sequence[1:], 1):
                if i == 1 and pivot['type'] == 'peak':  # Potential Wave 1
                    wave1_end = pivot
                    wave1_size = (wave1_end['price'] - wave_start['price']) / wave_start['price'] * 100
                    if wave1_size >= self.min_wave_size:
                        potential_waves.append({
                            'wave': 1,
                            'start': wave_start,
                            'end': wave1_end,
                            'size': wave1_size
                        })
                
                elif len(potential_waves) == 1 and pivot['type'] == 'valley':  # Potential Wave 2
                    wave2_end = pivot
                    wave1_size = potential_waves[0]['size']
                    wave2_retrace = (wave1_end['price'] - wave2_end['price']) / (wave1_end['price'] - wave_start['price']) * 100
                    if 20 <= wave2_retrace <= 80:  # TIGHTENED for bearish patterns
                        potential_waves.append({
                            'wave': 2,
                            'start': wave1_end,
                            'end': wave2_end,
                            'retrace': wave2_retrace
                        })
                
                elif len(potential_waves) == 2 and pivot['type'] == 'peak':  # Potential Wave 3
                    wave3_end = pivot
                    wave3_size = (wave3_end['price'] - wave2_end['price']) / wave2_end['price'] * 100
                    wave1_price_size = wave1_end['price'] - wave_start['price']
                    wave3_vs_wave1 = (wave3_end['price'] - wave2_end['price']) / wave1_price_size
                    
                    if wave3_size >= self.min_wave_size and wave3_vs_wave1 >= 0.8:  # TIGHTENED: 0.8 vs 0.2
                        potential_waves.append({
                            'wave': 3,
                            'start': wave2_end,
                            'end': wave3_end,
                            'size': wave3_size,
                            'vs_wave1': wave3_vs_wave1
                        })
                
                elif len(potential_waves) == 3 and pivot['type'] == 'valley':  # Potential Wave 4
                    wave4_end = pivot
                    wave3_size_price = wave3_end['price'] - wave2_end['price']
                    wave4_retrace = (wave3_end['price'] - wave4_end['price']) / wave3_size_price * 100
                    
                    if 25 <= wave4_retrace <= 70 and wave4_end['price'] > wave1_end['price'] * 0.90:  # TIGHTENED ranges
                        potential_waves.append({
                            'wave': 4,
                            'start': wave3_end,
                            'end': wave4_end,
                            'retrace': wave4_retrace
                        })
                
                elif len(potential_waves) == 4 and pivot['type'] == 'peak':  # Potential Wave 5
                    wave5_end = pivot
                    wave4_size_price = wave4_end['price'] - wave3_end['price']
                    wave5_size = (wave5_end['price'] - wave4_end['price']) / wave4_end['price'] * 100
                    wave1_price_size = wave1_end['price'] - wave_start['price']
                    wave5_vs_wave1 = (wave5_end['price'] - wave4_end['price']) / wave1_price_size
                    
                    if wave5_size >= self.min_wave_size and wave5_vs_wave1 >= 0.5:  # Wave 5 should be significant
                        potential_waves.append({
                            'wave': 5,
                            'start': wave4_end,
                            'end': wave5_end,
                            'size': wave5_size,
                            'vs_wave1': wave5_vs_wave1
                        })
                        break  # Complete 5-wave cycle
            
            # Determine current position and calculate targets
            if len(potential_waves) >= 3:
                current_wave = self.determine_current_position_bullish(potential_waves, current_price)
                targets = self.calculate_bullish_targets(potential_waves, current_price)
                
                quality_score = self.calculate_quality_score_bullish(potential_waves)
                
                # CRITICAL: Filter out broken structures and completed cycles
                if current_wave == 'BROKEN_STRUCTURE':
                    return None
                
                # CRITICAL: Filter out post-Wave 5 corrections (cycle completed)
                if current_wave == 'POST_WAVE_5_CORRECTION':
                    return None
                
                if quality_score >= self.min_quality_score:
                    return {
                        'type': 'BULLISH_IMPULSE',
                        'direction': 'BULLISH',
                        'waves': potential_waves,
                        'current_wave': current_wave,
                        'quality_score': quality_score,
                        'targets': targets,
                        'wave_start': wave_start,
                        'current_price': current_price,
                        'duration': (df.iloc[-1]['timestamp'] - wave_start['date']).total_seconds() / 3600  # SURGICAL CHANGE: Hours for 1H
                    }
            
            return None
            
        except Exception as e:
            return None

    def analyze_bearish_structure(self, pivots, df, major_high, major_low):
        """Analyze potential bearish Elliott Wave structure - SURGICAL CHANGES for 1H"""
        try:
            current_price = df.iloc[-1]['close']
            
            # Similar logic to bullish but inverted
            sequence = []
            for pivot in pivots:
                if pivot['index'] >= major_high['index']:
                    sequence.append(pivot)
            
            sequence.sort(key=lambda x: x['index'])
            
            if len(sequence) < 3:
                return None
            
            wave_start = major_high
            potential_waves = []
            
            for i, pivot in enumerate(sequence[1:], 1):
                if i == 1 and pivot['type'] == 'valley':  # Wave 1 down
                    wave1_end = pivot
                    wave1_size = (wave_start['price'] - wave1_end['price']) / wave_start['price'] * 100
                    if wave1_size >= self.min_wave_size:
                        potential_waves.append({
                            'wave': 1,
                            'start': wave_start,
                            'end': wave1_end,
                            'size': wave1_size
                        })
                
                elif len(potential_waves) == 1 and pivot['type'] == 'peak':  # Wave 2 up
                    wave2_end = pivot
                    wave2_retrace = (wave2_end['price'] - wave1_end['price']) / (wave_start['price'] - wave1_end['price']) * 100
                    if 5 <= wave2_retrace <= 99:  # SURGICAL CHANGE: Ultra-wide range for 1H
                        potential_waves.append({
                            'wave': 2,
                            'start': wave1_end,
                            'end': wave2_end,
                            'retrace': wave2_retrace
                        })
                
                elif len(potential_waves) == 2 and pivot['type'] == 'valley':  # Wave 3 down
                    wave3_end = pivot
                    wave3_size = (wave2_end['price'] - wave3_end['price']) / wave2_end['price'] * 100
                    if wave3_size >= self.min_wave_size:
                        potential_waves.append({
                            'wave': 3,
                            'start': wave2_end,
                            'end': wave3_end,
                            'size': wave3_size
                        })
                        break
            
            if len(potential_waves) >= 3:
                current_wave = self.determine_current_position_bearish(potential_waves, current_price)
                targets = self.calculate_bearish_targets(potential_waves, current_price)
                
                quality_score = self.calculate_quality_score_bearish(potential_waves)
                
                # CRITICAL: Filter out broken structures and completed cycles
                if current_wave == 'BROKEN_STRUCTURE':
                    return None
                
                # CRITICAL: Filter out post-Wave 5 corrections (cycle completed)
                if current_wave == 'POST_WAVE_5_CORRECTION':
                    return None
                
                if quality_score >= self.min_quality_score:
                    return {
                        'type': 'BEARISH_IMPULSE',
                        'direction': 'BEARISH',
                        'waves': potential_waves,
                        'current_wave': current_wave,
                        'quality_score': quality_score,
                        'targets': targets,
                        'wave_start': wave_start,
                        'current_price': current_price,
                        'duration': (df.iloc[-1]['timestamp'] - wave_start['date']).total_seconds() / 3600  # SURGICAL CHANGE: Hours for 1H
                    }
            
            return None
            
        except Exception as e:
            return None

    def determine_current_position_bullish(self, waves, current_price):
        """Determine current wave position in bullish structure - CORRECTED WAVE POSITION LOGIC"""
        try:
            # CORRECTED: Track wave completion sequence + current price context
            
            if len(waves) >= 4:
                # We have completed Waves 1,2,3,4 - check for Wave 5
                wave4_completion_price = waves[3]['end']['price']
                wave3_peak_price = waves[2]['end']['price']
                
                # If current price > Wave 4 completion, we're in Wave 5
                if current_price > wave4_completion_price:
                    return 'WAVE_5'
                # If current price is between Wave 4 and Wave 3, still correcting in Wave 4
                elif wave4_completion_price < current_price <= wave3_peak_price:
                    return 'WAVE_4'
                # If below Wave 4 completion, this is a BROKEN structure (not Wave 4 extension)
                else:
                    return 'BROKEN_STRUCTURE'
            
            elif len(waves) >= 5:
                # We have completed Waves 1,2,3,4,5 - check if we're in post-Wave 5 correction
                wave5_completion_price = waves[4]['end']['price']
                wave4_completion_price = waves[3]['end']['price']
                
                # If current price is below Wave 5 completion, we're in post-Wave 5 correction
                if current_price < wave5_completion_price:
                    return 'POST_WAVE_5_CORRECTION'
                # If current price is at/near Wave 5 completion, still in Wave 5
                elif current_price >= wave5_completion_price * 0.98:
                    return 'WAVE_5_ENDING'
                # If above Wave 5 completion, new cycle starting
                else:
                    return 'NEW_CYCLE_STARTING'
            
            elif len(waves) >= 3:
                # We have completed Waves 1,2,3 - looking for Wave 4 or 5
                wave3_completion_price = waves[2]['end']['price'] 
                wave2_completion_price = waves[1]['end']['price']
                
                # If above Wave 3 completion, either Wave 5 or new cycle
                if current_price > wave3_completion_price:
                    return 'WAVE_5_OR_NEW_CYCLE'
                # If between Wave 2 and Wave 3 completions, in Wave 4 correction
                elif wave2_completion_price < current_price <= wave3_completion_price:
                    return 'WAVE_4'
                # If at/below Wave 2 level, deep Wave 4 or broken structure
                else:
                    return 'DEEP_WAVE_4_OR_BROKEN'
            
            elif len(waves) >= 2:
                # We have completed Waves 1,2 - looking for Wave 3
                wave2_completion_price = waves[1]['end']['price']
                wave1_completion_price = waves[0]['end']['price']
                
                # If above Wave 2 completion, likely in Wave 3
                if current_price > wave2_completion_price:
                    return 'WAVE_3'
                # If between Wave 1 and Wave 2, still in Wave 2 correction
                elif wave1_completion_price < current_price <= wave2_completion_price:
                    return 'WAVE_2'
                # If below Wave 1 completion, broken structure
                else:
                    return 'BROKEN_STRUCTURE'
            else:
                return 'EARLY_STRUCTURE'
        except:
            return 'UNKNOWN'

    def determine_current_position_bearish(self, waves, current_price):
        """Determine current wave position in bearish structure - CORRECTED WAVE POSITION LOGIC"""
        try:
            # CORRECTED: Track wave completion sequence for bearish patterns
            
            if len(waves) >= 4:
                # We have completed Waves 1,2,3,4 - check for Wave 5
                wave4_completion_price = waves[3]['end']['price']
                wave3_low_price = waves[2]['end']['price']
                
                # For bearish: if current price < Wave 4 completion, we're in Wave 5 down
                if current_price < wave4_completion_price:
                    return 'WAVE_5'
                # If current price is between Wave 4 and Wave 3, still bouncing in Wave 4
                elif wave4_completion_price < current_price <= wave3_low_price:
                    return 'WAVE_4'
                # If above Wave 3 low, this is a BROKEN structure (not strong Wave 4)
                else:
                    return 'BROKEN_STRUCTURE'
            
            elif len(waves) >= 3:
                # We have completed Waves 1,2,3 - looking for Wave 4 or 5
                wave3_completion_price = waves[2]['end']['price']
                wave2_completion_price = waves[1]['end']['price']
                
                # If below Wave 3 completion, likely Wave 5 or extension
                if current_price < wave3_completion_price:
                    return 'WAVE_5_OR_EXTENSION'
                # If between Wave 2 and Wave 3 completions, in Wave 4 bounce
                elif wave2_completion_price < current_price <= wave3_completion_price:
                    return 'WAVE_4'
                # If above Wave 2 level, strong Wave 4 bounce
                else:
                    return 'STRONG_WAVE_4'
            
            elif len(waves) >= 2:
                # We have completed Waves 1,2 - looking for Wave 3
                wave2_completion_price = waves[1]['end']['price']
                wave1_completion_price = waves[0]['end']['price']
                
                # If below Wave 2 completion, likely in Wave 3 down
                if current_price < wave2_completion_price:
                    return 'WAVE_3'
                # If between Wave 1 and Wave 2, still in Wave 2 bounce
                elif wave1_completion_price < current_price <= wave2_completion_price:
                    return 'WAVE_2'
                # If above Wave 1 completion, broken structure
                else:
                    return 'BROKEN_STRUCTURE'
            else:
                return 'EARLY_STRUCTURE'
        except:
            return 'UNKNOWN'

    def calculate_bullish_targets(self, waves, current_price):
        """Calculate realistic bullish Elliott Wave targets - CORRECTED TARGET LOGIC"""
        try:
            if len(waves) >= 4:
                # We have Wave 4, calculate Wave 5 targets
                wave1_size = waves[0]['end']['price'] - waves[0]['start']['price']
                wave4_end = waves[3]['end']['price']
                
                # Calculate theoretical Elliott targets
                target1 = wave4_end + (wave1_size * 0.618)  # 61.8% projection
                target2 = wave4_end + wave1_size            # 100% projection
                target3 = wave4_end + (wave1_size * 1.618)  # 161.8% projection
                
                # CRITICAL: Ensure all targets are above current price
                adjusted_targets = []
                for target in [target1, target2, target3]:
                    if target <= current_price:
                        # If theoretical target is below current price, adjust upward
                        adjusted_target = current_price * (1.02 + (len(adjusted_targets) * 0.03))
                        adjusted_targets.append(adjusted_target)
                    else:
                        adjusted_targets.append(target)
                
                return adjusted_targets
            
            elif len(waves) >= 3:
                # Estimate based on Wave 3 completion
                wave3_end = waves[2]['end']['price']
                wave1_size = waves[0]['end']['price'] - waves[0]['start']['price']
                
                # Estimate Wave 4 correction (typically 38% of Wave 3)
                wave3_size = waves[2]['end']['price'] - waves[2]['start']['price']
                estimated_wave4_end = wave3_end - (wave3_size * 0.38)
                
                # Calculate theoretical targets
                target1 = estimated_wave4_end + (wave1_size * 0.618)
                target2 = estimated_wave4_end + wave1_size
                target3 = estimated_wave4_end + (wave1_size * 1.618)
                
                # CRITICAL: Ensure all targets are above current price
                adjusted_targets = []
                for target in [target1, target2, target3]:
                    if target <= current_price:
                        # If theoretical target is below current price, adjust upward
                        adjusted_target = current_price * (1.02 + (len(adjusted_targets) * 0.03))
                        adjusted_targets.append(adjusted_target)
                    else:
                        adjusted_targets.append(target)
                
                return adjusted_targets
            
            # Default targets for early structures - ensure they're above current price
            return [current_price * 1.02, current_price * 1.05, current_price * 1.08]
            
        except:
            # Fallback: ensure targets are always above current price
            return [current_price * 1.02, current_price * 1.05, current_price * 1.08]

    def calculate_bearish_targets(self, waves, current_price):
        """Calculate realistic bearish Elliott Wave targets - CORRECTED TARGET LOGIC"""
        try:
            if len(waves) >= 4:
                # We have Wave 4, calculate Wave 5 downside targets
                wave1_size = waves[0]['start']['price'] - waves[0]['end']['price']
                wave4_end = waves[3]['end']['price']
                
                # Calculate theoretical Elliott downside targets
                target1 = wave4_end - (wave1_size * 0.618)  # 61.8% projection
                target2 = wave4_end - wave1_size            # 100% projection
                target3 = wave4_end - (wave1_size * 1.618)  # 161.8% projection
                
                # CRITICAL: Ensure all targets are below current price
                adjusted_targets = []
                for target in [target1, target2, target3]:
                    if target >= current_price:
                        # If theoretical target is above current price, adjust downward
                        adjusted_target = current_price * (0.95 - (len(adjusted_targets) * 0.05))
                        adjusted_targets.append(adjusted_target)
                    else:
                        adjusted_targets.append(target)
                
                return adjusted_targets
            
            elif len(waves) >= 3:
                # Estimate based on Wave 3 completion
                wave3_end = waves[2]['end']['price']
                wave1_size = waves[0]['start']['price'] - waves[0]['end']['price']
                
                # Estimate Wave 4 bounce (typically 38% of Wave 3)
                wave3_size = waves[2]['start']['price'] - waves[2]['end']['price']
                estimated_wave4_end = wave3_end + (wave3_size * 0.38)
                
                # Calculate theoretical targets
                target1 = estimated_wave4_end - (wave1_size * 0.618)
                target2 = estimated_wave4_end - wave1_size
                target3 = estimated_wave4_end - (wave1_size * 1.618)
                
                # CRITICAL: Ensure all targets are below current price
                adjusted_targets = []
                for target in [target1, target2, target3]:
                    if target >= current_price:
                        # If theoretical target is above current price, adjust downward
                        adjusted_target = current_price * (0.95 - (len(adjusted_targets) * 0.05))
                        adjusted_targets.append(adjusted_target)
                    else:
                        adjusted_targets.append(target)
                
                return adjusted_targets
            
            # Default targets for early structures - ensure they're below current price
            return [current_price * 0.95, current_price * 0.90, current_price * 0.85]
            
        except:
            # Fallback: ensure targets are always below current price
            return [current_price * 0.95, current_price * 0.90, current_price * 0.85]

    def calculate_quality_score_bullish(self, waves):
        """Calculate quality score - MUCH MORE STRICT for quality patterns"""
        score = 20  # Lower base score
        
        try:
            if len(waves) >= 2:
                # Wave 2 retracement (very strict)
                wave2_retrace = waves[1]['retrace']
                if 30 <= wave2_retrace <= 65:  # IDEAL range
                    score += 30
                elif 20 <= wave2_retrace <= 80:  # Acceptable
                    score += 20
                else:
                    score -= 15  # Heavy penalty for bad retracements
            
            if len(waves) >= 3:
                # Wave 3 strength (much stricter)
                if waves[2]['size'] >= 3:   # 3% minimum for 1H quality
                    score += 35
                elif waves[2]['size'] >= 2:  # 2% backup
                    score += 25
                elif waves[2]['size'] >= 1:  # 1% minimal
                    score += 10
                else:
                    score -= 20  # Heavy penalty for weak Wave 3
                
                # Wave 3 vs Wave 1 ratio (strict requirements)
                if 'vs_wave1' in waves[2]:
                    ratio = waves[2]['vs_wave1']
                    if ratio >= 1.618:  # Golden ratio ideal
                        score += 40
                    elif ratio >= 1.2:  # Strong Wave 3
                        score += 30
                    elif ratio >= 0.8:  # Acceptable
                        score += 15
                    else:
                        score -= 20  # Penalty for weak Wave 3
            
            if len(waves) >= 4:
                # Wave 4 retracement (strict)
                wave4_retrace = waves[3]['retrace']
                if 25 <= wave4_retrace <= 50:  # IDEAL range
                    score += 25
                elif 20 <= wave4_retrace <= 70:  # Acceptable
                    score += 15
                else:
                    score -= 10  # Penalty for extreme Wave 4
            
        except:
            score -= 10  # Penalty for calculation errors
        
        return max(score, 0)

    def calculate_quality_score_bearish(self, waves):
        """Calculate quality score - MUCH MORE STRICT for bearish patterns"""
        score = 20  # Lower base score
        
        try:
            if len(waves) >= 2:
                wave2_retrace = waves[1]['retrace']
                if 30 <= wave2_retrace <= 65:  # IDEAL range
                    score += 30
                elif 20 <= wave2_retrace <= 80:  # Acceptable
                    score += 20
                else:
                    score -= 15  # Heavy penalty
            
            if len(waves) >= 3:
                if waves[2]['size'] >= 3:   # 3% minimum
                    score += 35
                elif waves[2]['size'] >= 2:  # 2% backup
                    score += 25
                elif waves[2]['size'] >= 1:  # 1% minimal
                    score += 10
                else:
                    score -= 20  # Heavy penalty for weak Wave 3
            
        except:
            score -= 10
        
        return max(score, 0)

    def analyze_symbol(self, symbol):
        """Analyze single symbol for current Elliott Wave structure - IDENTICAL to 4H"""
        try:
            print(f"📡 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_daily_data(symbol)
            if df is None or len(df) < 30:
                print("No data")
                return []
            
            current_price = df.iloc[-1]['close']
            
            # Find major pivots
            pivots = self.find_major_pivots(df)
            if len(pivots) < 3:
                print(f"Insufficient major pivots ({len(pivots)} found)")
                return []
            
            # Identify current Elliott structure
            patterns = self.identify_current_elliott_structure(pivots, df)
            
            if not patterns:
                print("No current structure")
                return []
            
            # Format results
            results = []
            for pattern in patterns:
                setup = {
                    'symbol': symbol,
                    'pattern_type': pattern['type'],
                    'direction': pattern['direction'],
                    'current_wave': pattern['current_wave'],
                    'quality_score': pattern['quality_score'],
                    'current_price': current_price,
                    'targets': pattern['targets'],
                    'duration': pattern['duration'],
                    'wave_start': pattern['wave_start']['price'],
                    'wave_start_date': pattern['wave_start']['date'].strftime('%Y-%m-%d %H:%M'),
                    'waves': pattern['waves']
                }
                results.append(setup)
            
            print(f"Found {len(results)} current structures")
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all high-liquidity symbols - SURGICAL CHANGES for 1H display"""
        print("🌊 ELLIOTT WAVE SCANNER - 1H CHART (CURRENT STRUCTURE FOCUS)")
        print("=" * 85)
        print("📈 Scanning for current major Elliott Wave structures...")
        print("📊 Strategy: Current cycle analysis → Intraday trend targets")
        print("💰 Filtering: $10K+ daily volume, top 200 most liquid pairs")  # SURGICAL CHANGE
        print()
        
        symbols = self.get_symbols()
        all_patterns = []
        
        print(f"\n🔍 Analyzing {len(symbols)} high-liquidity pairs...")
        print()
        
        for i, symbol in enumerate(symbols, 1):
            try:
                patterns = self.analyze_symbol(symbol)
                all_patterns.extend(patterns)
                
                # Progress indicator for large scans
                if i % 20 == 0:  # SURGICAL CHANGE: More frequent updates for 1H
                    print(f"📊 Progress: {i}/{len(symbols)} pairs analyzed...")
                
                time.sleep(0.05)  # SURGICAL CHANGE: Faster for 1H
                
            except Exception as e:
                print(f"📡 {symbol.replace('/USDT', '')}... Error: {e}")
                continue
        
        return all_patterns

    def display_complete_wave_analysis(self, pattern):
        """Display complete Elliott Wave analysis with trading signals - IDENTICAL to 4H"""
        waves = pattern['waves']
        current_price = pattern['current_price']
        targets = pattern['targets']
        direction = pattern['direction']
        current_wave = pattern['current_wave']
        
        # Current Fibonacci targets (whatever wave is active)
        print(f"\n   🎯 CURRENT WAVE FIBONACCI TARGETS:")
        if current_wave == 'WAVE_3':
            print(f"   📈 WAVE 3 TARGETS: T1: ${targets[0]:.4f} (61.8%) | T2: ${targets[1]:.4f} (100%) | T3: ${targets[2]:.4f} (161.8%)")
        elif current_wave in ['WAVE_4', 'WAVE_4_OR_5']:
            print(f"   📈 WAVE 5 TARGETS: T1: ${targets[0]:.4f} (61.8%) | T2: ${targets[1]:.4f} (100%) | T3: ${targets[2]:.4f} (161.8%)")
        elif current_wave == 'WAVE_5':
            print(f"   📈 WAVE 5 TARGETS: T1: ${targets[0]:.4f} (61.8%) | T2: ${targets[1]:.4f} (100%) | T3: ${targets[2]:.4f} (161.8%)")
        else:
            print(f"   📈 TARGETS: T1: ${targets[0]:.4f} (61.8%) | T2: ${targets[1]:.4f} (100%) | T3: ${targets[2]:.4f} (161.8%)")
        
        # Calculate complete wave cycle projections
        if direction == 'BULLISH':
            self.display_bullish_wave_cycle(pattern, waves, current_price)
        else:
            self.display_bearish_wave_cycle(pattern, waves, current_price)
    
    def display_bullish_wave_cycle(self, pattern, waves, current_price):
        """Display complete bullish Elliott Wave cycle - SURGICAL CHANGES for 1H timeframes"""
        current_wave = pattern['current_wave']
        
        if len(waves) >= 3:
            # Calculate Wave 4 and Wave 5 projections
            wave1_size = waves[0]['end']['price'] - waves[0]['start']['price']
            wave3_end = waves[2]['end']['price']
            
            # Wave 4 correction levels (23.6%, 38.2%, 50%, 61.8% of Wave 3)
            if len(waves) >= 3:
                wave3_size = waves[2]['end']['price'] - waves[2]['start']['price']
                wave4_shallow = wave3_end - (wave3_size * 0.236)
                wave4_normal = wave3_end - (wave3_size * 0.382)
                wave4_deep = wave3_end - (wave3_size * 0.50)
                wave4_max = wave3_end - (wave3_size * 0.618)
                
                print(f"\n   📉 EXPECTED WAVE 4 CORRECTION LEVELS:")
                print(f"   Shallow: ${wave4_shallow:.4f} (23.6%) | Normal: ${wave4_normal:.4f} (38.2%)")
                print(f"   Deep: ${wave4_deep:.4f} (50%) | Maximum: ${wave4_max:.4f} (61.8%)")
            
            # Wave 5 ultimate targets (from Wave 4 completion)
            estimated_wave4_end = wave4_normal if 'wave4_normal' in locals() else wave3_end * 0.85
            wave5_t1 = estimated_wave4_end + (wave1_size * 0.618)
            wave5_t2 = estimated_wave4_end + wave1_size
            wave5_t3 = estimated_wave4_end + (wave1_size * 1.618)
            wave5_t4 = estimated_wave4_end + (wave1_size * 2.618)
            
            print(f"\n   🚀 ULTIMATE WAVE 5 TARGETS (After Wave 4):")
            print(f"   W5-T1: ${wave5_t1:.4f} (61.8%) | W5-T2: ${wave5_t2:.4f} (100%)")
            print(f"   W5-T3: ${wave5_t3:.4f} (161.8%) | W5-T4: ${wave5_t4:.4f} (261.8%)")
        
        # Trading signals based on current wave - SURGICAL CHANGE: 1H adapted timeframes
        print(f"\n   💡 COMPLETE TRADING STRATEGY (1H INTRADAY):")
        
        if current_wave == 'WAVE_3':
            print(f"   📊 CURRENT STATUS: Wave 3 in progress (strongest wave)")
            print(f"   🎯 IF OWNED:")
            print(f"      • HOLD through Wave 3 completion")
            print(f"      • Take 25% profits at each Wave 3 target")
            print(f"      • Keep 25% for Wave 4 dip re-entry")
            print(f"   🎯 IF NOT OWNED:")
            print(f"      • WAIT for Wave 4 correction")
            print(f"      • BUY ZONE: ${wave4_normal:.4f} - ${wave4_deep:.4f} (Wave 4 dip)")
            print(f"      • STOP LOSS: ${wave4_max * 0.95:.4f} (below Wave 4 max)")
            print(f"      • TARGET: Wave 5 completion ${wave5_t2:.4f} - ${wave5_t3:.4f}")
            print(f"      • STRATEGY: Be patient - wait for the pullback opportunity!")
            print(f"      • TIMEFRAME: 1-6 hours (1H intraday)")  # SURGICAL CHANGE
            
        elif current_wave in ['WAVE_4', 'WAVE_4_OR_5']:
            # Smart logic: Check if we need to wait for Wave 4 or if it's completed
            wave4_entry_zone_high = wave4_shallow if 'wave4_shallow' in locals() else current_price * 0.90
            wave4_entry_zone_low = wave4_deep if 'wave4_deep' in locals() else current_price * 0.85
            
            if current_price > wave4_entry_zone_high:
                # Price too high - wait for Wave 4 correction
                print(f"   📊 CURRENT STATUS: Wave 3 completion / Wave 4 correction pending")
                print(f"   ⚠️  ACTION: WAIT for Wave 4 pullback")
                print(f"   📉 MONITOR: Price decline to Wave 4 correction zone")
                print(f"   🛒 BUY ZONE: ${wave4_normal:.4f} - ${wave4_deep:.4f} (Wave 4 dip)")
                print(f"   🛑 STOP LOSS: ${wave4_max * 0.95:.4f} (below Wave 4 max)")
                print(f"   🎯 TARGETS: Wave 5 completion ${wave5_t1:.4f} - ${wave5_t3:.4f}")
                print(f"   💡 STRATEGY: Don't chase current levels - wait for the pullback!")
                print(f"   ⏰ TIMEFRAME: 1-4 hours (1H intraday)")  # SURGICAL CHANGE
                
            elif wave4_entry_zone_low <= current_price <= wave4_entry_zone_high:
                # Price in Wave 4 correction zone - good to buy
                print(f"   📊 CURRENT STATUS: Wave 4 correction in progress")
                print(f"   🛒 ACTION: BUY NOW (Wave 4 dip opportunity)")
                print(f"   🛒 ENTRY: ${current_price * 0.998:.4f} - ${current_price * 1.002:.4f} (current levels)")
                print(f"   🛑 STOP LOSS: ${wave4_max * 0.95:.4f} (below Wave 4 max)")
                print(f"   🎯 TARGETS: Wave 5 completion ${wave5_t1:.4f} - ${wave5_t3:.4f}")
                print(f"   💡 STRATEGY: Excellent Wave 4 entry opportunity!")
                print(f"   ⏰ TIMEFRAME: 1-4 hours (1H intraday)")  # SURGICAL CHANGE
                
            else:
                # Price below Wave 4 zone - potentially in Wave 5 or oversold
                print(f"   📊 CURRENT STATUS: Wave 4 completed / Wave 5 beginning")
                print(f"   🛒 ACTION: BUY NOW (Wave 5 setup)")
                print(f"   🛒 ENTRY: ${current_price * 0.998:.4f} - ${current_price * 1.002:.4f} (current levels)")
                if len(waves) >= 4:
                    wave4_low = waves[3]['end']['price']
                    print(f"   🛑 STOP LOSS: ${wave4_low * 0.95:.4f} (below Wave 4 low)")
                else:
                    print(f"   🛑 STOP LOSS: ${current_price * 0.95:.4f} (below support)")
                print(f"   🎯 TARGETS: Wave 5 completion ${wave5_t1:.4f} - ${wave5_t3:.4f}")
                print(f"   ⏰ TIMEFRAME: 1-4 hours (1H intraday)")  # SURGICAL CHANGE
            
        elif current_wave == 'WAVE_5':
            print(f"   📊 CURRENT STATUS: Wave 5 active (final wave)")
            print(f"   ⚠️  CAUTION: Prepare for major cycle completion")
            print(f"   🎯 IF OWNED:")
            print(f"      • Take 50% profits at Wave 5 targets")
            print(f"      • Trail stop losses aggressively")
            print(f"      • Prepare for major reversal")
            print(f"   🎯 IF NOT OWNED:")
            print(f"      • AVOID new entries (high risk)")
            print(f"      • Wait for cycle completion")
            print(f"   ⏰ TIMEFRAME: 30 minutes - 2 hours (1H completion)")  # SURGICAL CHANGE
    
    def display_bearish_wave_cycle(self, pattern, waves, current_price):
        """Display complete bearish Elliott Wave cycle - SURGICAL CHANGES for 1H timeframes"""
        current_wave = pattern['current_wave']
        
        if len(waves) >= 3:
            # Similar logic for bearish patterns but inverted
            wave1_size = waves[0]['start']['price'] - waves[0]['end']['price']
            wave3_end = waves[2]['end']['price']
            
            # Wave 4 correction levels (bounces up)
            if len(waves) >= 3:
                wave3_size = waves[2]['start']['price'] - waves[2]['end']['price']
                wave4_shallow = wave3_end + (wave3_size * 0.236)
                wave4_normal = wave3_end + (wave3_size * 0.382)
                wave4_deep = wave3_end + (wave3_size * 0.50)
                wave4_max = wave3_end + (wave3_size * 0.618)
                
                print(f"\n   📈 EXPECTED WAVE 4 BOUNCE LEVELS:")
                print(f"   Shallow: ${wave4_shallow:.4f} (23.6%) | Normal: ${wave4_normal:.4f} (38.2%)")
                print(f"   Deep: ${wave4_deep:.4f} (50%) | Maximum: ${wave4_max:.4f} (61.8%)")
            
            # Wave 5 downside targets
            print(f"\n   📉 ULTIMATE WAVE 5 DOWNSIDE TARGETS:")
            target1 = current_price * 0.95  # 5% decline (1H adapted)
            target2 = current_price * 0.90  # 10% decline 
            target3 = current_price * 0.85  # 15% decline
            print(f"   W5-T1: ${target1:.4f} (5% decline) | W5-T2: ${target2:.4f} (10% decline) | W5-T3: ${target3:.4f} (15% decline)")
        
        # Bearish trading signals - SURGICAL CHANGE: 1H adapted
        print(f"\n   💡 COMPLETE TRADING STRATEGY (1H INTRADAY):")
        
        if current_wave == 'WAVE_2':
            print(f"   📊 CURRENT STATUS: Bearish Wave 2 bounce (correction)")
            print(f"   🎯 IDEAL SHORT SETUP:")
            print(f"      • Wave 2 bounce ending (best short opportunity)")
            if 'wave4_normal' in locals() and 'wave4_deep' in locals():
                print(f"      • SHORT ZONE: ${wave4_normal:.4f} - ${wave4_deep:.4f} (Wave 2 resistance)")
            else:
                print(f"      • SHORT ZONE: Current resistance levels")
            print(f"      • STOP LOSS: ${current_price * 1.02:.4f} (above Wave 2 high)")
            print(f"      • TARGET: Wave 3 decline targets")
            print(f"      • R:R: Excellent (Wave 3 strongest decline)")
            print(f"      • TIMEFRAME: 1-4 hours (1H swing)")  # SURGICAL CHANGE
            
        elif current_wave == 'WAVE_3':
            print(f"   📊 CURRENT STATUS: Bearish Wave 3 in progress (strongest decline)")
            print(f"   🎯 IF SHORT:")
            print(f"      • HOLD through Wave 3 completion")
            print(f"      • Take 25% profits at each Wave 3 target")
            print(f"      • Trail stop loss aggressively")
            print(f"   🎯 IF NOT SHORT:")
            print(f"      • AVOID new shorts (Wave 3 already started)")
            print(f"      • WAIT for Wave 4 bounce to short")
            print(f"   ⏰ TIMEFRAME: 1-4 hours (1H swing)")  # SURGICAL CHANGE
                
        elif current_wave in ['WAVE_4', 'WAVE_4_OR_5']:
            print(f"   📊 CURRENT STATUS: Wave 4 bounce ending / Wave 5 beginning")
            print(f"   🛒 ACTION: SHORT NOW (Wave 5 setup)")
            print(f"   🛒 ENTRY: ${current_price * 0.998:.4f} - ${current_price * 1.002:.4f}")
            print(f"   🛑 STOP LOSS: ${current_price * 1.02:.4f} (above Wave 4 high)")
            print(f"   🎯 TARGETS: Wave 5 decline targets")
            print(f"   ⏰ TIMEFRAME: 1-3 hours (1H swing)")  # SURGICAL CHANGE
            
        elif current_wave == 'WAVE_5':
            print(f"   📊 CURRENT STATUS: Final bearish wave (cycle completion)")
            print(f"   ⚠️  CAUTION: Prepare for major cycle reversal")
            print(f"   🎯 IF SHORT:")
            print(f"      • Take 50% profits at Wave 5 targets")
            print(f"      • Trail stops very tight")
            print(f"      • Prepare for major reversal")
            print(f"   🎯 REVERSAL WATCH:")
            print(f"      • Wave 5 completion = major cycle bottom")
            print(f"      • Prepare for new bullish cycle")
            print(f"   ⏰ TIMEFRAME: 30 minutes - 2 hours (1H completion)")  # SURGICAL CHANGE

    def display_results(self, patterns):
        """Display Elliott Wave scan results - SURGICAL CHANGES for 1H formatting"""
        if not patterns:
            symbols_count = len(self.get_symbols()) if hasattr(self, 'get_symbols') else 200
            print(f"\n📊 SCAN RESULTS:")
            print(f"Total scanned: {symbols_count}")
            print("Current structures: 0")
            print("=" * 89)
            print("🌊 FOUND 0 CURRENT ELLIOTT WAVE STRUCTURES (1H)")
            print("=" * 89)
            print("❌ No major current Elliott Wave structures found")
            print("💡 Market may be in consolidation or early structure formation")
            return
        
        # Sort by quality score and take top 15 only
        patterns.sort(key=lambda x: x['quality_score'], reverse=True)
        top_patterns = patterns[:15]  # LIMIT TO TOP 15 PATTERNS
        
        symbols_count = len(self.get_symbols()) if hasattr(self, 'get_symbols') else 200
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {symbols_count}")
        print(f"Patterns found: {len(patterns)}")
        print(f"Top patterns shown: {len(top_patterns)}")
        print("=" * 89)
        print(f"🌊 TOP {len(top_patterns)} ELLIOTT WAVE STRUCTURES (1H)")
        print("=" * 89)
        
        for i, pattern in enumerate(top_patterns, 1):
            direction_icon = "📈" if pattern['direction'] == 'BULLISH' else "📉"
            
            print(f"\n{i}. {direction_icon} {pattern['symbol'].replace('/USDT', '')} | Score: {pattern['quality_score']:.0f}/100 | 🌊 {pattern['current_wave']} | {pattern['direction']}")
            print(f"   Current: ${pattern['current_price']:.4f} | Duration: {pattern['duration']:.1f}h | Start: {pattern['wave_start_date']}")  # SURGICAL CHANGE: Hours
            print(f"   Wave Start: ${pattern['wave_start']:.4f}")
            print(f"   Targets: T1: ${pattern['targets'][0]:.4f} | T2: ${pattern['targets'][1]:.4f} | T3: ${pattern['targets'][2]:.4f}")
            
            # DETAILED WAVE BREAKDOWN - IDENTICAL to 4H
            print(f"   📊 DETAILED WAVE BREAKDOWN:")
            waves = pattern['waves']
            
            # Individual wave analysis
            for j, wave in enumerate(waves, 1):
                wave_icon = "📈" if j % 2 == 1 else "📉"  # Odd waves up, even waves down for bullish
                if pattern['direction'] == 'BEARISH':
                    wave_icon = "📉" if j % 2 == 1 else "📈"  # Inverted for bearish
                
                duration_hours = (wave['end']['date'] - wave['start']['date']).total_seconds() / 3600
                wave_size = wave.get('size', 0)  # Handle missing size gracefully
                print(f"   Wave {j}: {wave_icon} {wave_size:.1f}% move | Duration: {duration_hours:.1f}h | ${wave['start']['price']:.4f} → ${wave['end']['price']:.4f}")
            
            # WAVE RELATIONSHIPS - IDENTICAL to 4H
            if len(waves) >= 2:
                print(f"   📈 WAVE RELATIONSHIPS:")
                wave2_retrace = waves[1]['retrace']
                health_2 = "✅ HEALTHY" if 20 <= wave2_retrace <= 80 else "⚠️ WEAK" if wave2_retrace > 90 or wave2_retrace < 10 else "🚀 STRONG"
                print(f"   Wave 2 Retrace: {wave2_retrace:.1f}% of Wave 1 ({health_2})")
            
            if len(waves) >= 3:
                if 'vs_wave1' in waves[2]:
                    wave3_ratio = waves[2]['vs_wave1']
                    health_3 = "🚀 STRONG" if wave3_ratio >= 1.618 else "✅ HEALTHY" if wave3_ratio >= 1.0 else "⚠️ WEAK"
                    print(f"   Wave 3 vs Wave 1: {wave3_ratio:.1f}x ({health_3})")
            
            if len(waves) >= 4:
                wave4_retrace = waves[3]['retrace']
                health_4 = "✅ HEALTHY" if 20 <= wave4_retrace <= 60 else "⚠️ EXTREME" if wave4_retrace > 80 or wave4_retrace < 10 else "🚀 STRONG"
                print(f"   Wave 4 Retrace: {wave4_retrace:.1f}% of Wave 3 ({health_4})")
            
            # Complete Wave Cycle Analysis & Trading Signals - IDENTICAL to 4H
            self.display_complete_wave_analysis(pattern)
            
            print("-" * 85)

def main():
    """Main execution function - IDENTICAL to 4H"""
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