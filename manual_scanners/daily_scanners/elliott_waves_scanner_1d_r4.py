#!/usr/bin/env python3
"""
Elliott Wave Scanner - Daily Chart (Accurate Major Structure Version)
Professional Elliott Wave pattern detection focused on current major cycles

Identifies current major Elliott Wave structures with proper scale and timing.
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
        
        # Balanced parameters for finding actual patterns
        self.lookback_periods = 120  # 4 months - good balance
        self.min_wave_size = 8       # Minimum 8% move (more realistic)
        self.min_pivot_distance = 5  # Minimum 5 days between pivots
        self.min_quality_score = 40  # Lower threshold for opportunities
        self.major_move_threshold = 15  # 15%+ moves for major wave identification
        
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
                    
                    # Minimum criteria for daily analysis
                    min_volume_24h = 500000   # $500K+ daily volume (lower than weekly)
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
            
            # Take top liquid pairs for daily analysis
            top_pairs = viable_pairs[:150]  # More pairs for daily analysis
            
            print(f"📊 Filtered to {len(top_pairs)} high-liquidity pairs for daily analysis")
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
        """Get daily OHLCV data focused on recent major structure"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', since=since, limit=self.lookback_periods)
            
            if len(ohlcv) < 30:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators
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
        """Find major pivots representing significant market structure"""
        pivots = []
        
        # First, identify the major trend direction and key levels
        start_price = df.iloc[0]['close']
        current_price = df.iloc[-1]['close']
        total_move = (current_price - start_price) / start_price * 100
        
        # Look for major pivots with balanced sensitivity
        for i in range(7, len(df) - 3):  # Use 7-day window for major pivots
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            date = df.iloc[i]['timestamp']
            
            # Check for major peaks
            window_highs = df.iloc[i-7:i+4]['high']
            if high == window_highs.max():
                # Must be a significant move from recent lows
                recent_low = df.iloc[max(0, i-20):i]['low'].min()
                move_size = (high - recent_low) / recent_low * 100
                if move_size >= self.min_wave_size:
                    # More lenient context check
                    context_high = df.iloc[max(0, i-15):min(len(df), i+15)]['high'].max()
                    if high >= context_high * 0.90:  # Within 10% of major high
                        pivots.append({
                            'index': i,
                            'price': high,
                            'date': date,
                            'type': 'peak',
                            'move_size': move_size,
                            'significance': self.calculate_pivot_significance(df, i, 'peak')
                        })
            
            # Check for major valleys
            window_lows = df.iloc[i-7:i+4]['low']
            if low == window_lows.min():
                recent_high = df.iloc[max(0, i-20):i]['high'].max()
                move_size = (recent_high - low) / recent_high * 100
                if move_size >= self.min_wave_size:
                    context_low = df.iloc[max(0, i-15):min(len(df), i+15)]['low'].min()
                    if low <= context_low * 1.10:  # Within 10% of major low
                        pivots.append({
                            'index': i,
                            'price': low,
                            'date': date,
                            'type': 'valley',
                            'move_size': move_size,
                            'significance': self.calculate_pivot_significance(df, i, 'valley')
                        })
        
        # Filter and sort pivots by significance
        pivots.sort(key=lambda x: x['significance'], reverse=True)
        
        # Remove close pivots and ensure alternating pattern
        filtered_pivots = []
        for pivot in pivots:
            if not filtered_pivots:
                filtered_pivots.append(pivot)
            else:
                # Check if this pivot is significant and far enough from others
                is_valid = True
                for existing in filtered_pivots:
                    if abs(pivot['index'] - existing['index']) < self.min_pivot_distance:
                        # If close in time, keep the more significant one
                        if pivot['significance'] > existing['significance']:
                            filtered_pivots.remove(existing)
                        else:
                            is_valid = False
                        break
                    if pivot['type'] == existing['type'] and abs(pivot['price'] - existing['price']) / existing['price'] < 0.15:
                        # If close in price and same type, keep more significant
                        if pivot['significance'] > existing['significance']:
                            filtered_pivots.remove(existing)
                        else:
                            is_valid = False
                        break
                
                if is_valid:
                    filtered_pivots.append(pivot)
        
        # Sort by time and ensure we have meaningful structure
        filtered_pivots.sort(key=lambda x: x['index'])
        
        return filtered_pivots

    def calculate_pivot_significance(self, df, index, pivot_type):
        """Calculate how significant a pivot is in the overall structure"""
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
        """Identify the current Elliott Wave structure focusing on major moves"""
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
        """Analyze potential bullish Elliott Wave structure"""
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
                    if 20 <= wave2_retrace <= 95:  # More lenient Wave 2 retracement
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
                    
                    if wave3_size >= self.min_wave_size and wave3_vs_wave1 >= 0.8:  # More lenient Wave 3
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
                    
                    if 10 <= wave4_retrace <= 75 and wave4_end['price'] > wave1_end['price'] * 0.95:  # More lenient overlap
                        potential_waves.append({
                            'wave': 4,
                            'start': wave3_end,
                            'end': wave4_end,
                            'retrace': wave4_retrace
                        })
                        break  # We have enough for analysis
            
            # Determine current position and calculate targets
            if len(potential_waves) >= 3:
                current_wave = self.determine_current_position_bullish(potential_waves, current_price)
                targets = self.calculate_bullish_targets(potential_waves, current_price)
                
                quality_score = self.calculate_quality_score_bullish(potential_waves)
                
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
                        'duration': (df.iloc[-1]['timestamp'] - wave_start['date']).days
                    }
            
            return None
            
        except Exception as e:
            return None

    def analyze_bearish_structure(self, pivots, df, major_high, major_low):
        """Analyze potential bearish Elliott Wave structure"""
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
                    if 20 <= wave2_retrace <= 95:  # More lenient
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
                        'duration': (df.iloc[-1]['timestamp'] - wave_start['date']).days
                    }
            
            return None
            
        except Exception as e:
            return None

    def determine_current_position_bullish(self, waves, current_price):
        """Determine current wave position in bullish structure"""
        try:
            if len(waves) >= 4:
                wave4_end = waves[3]['end']['price']
                wave3_end = waves[2]['end']['price']
                if current_price > wave4_end:
                    return 'WAVE_5'
                elif current_price > wave3_end:
                    return 'WAVE_4'
                else:
                    return 'WAVE_3'
            elif len(waves) >= 3:
                wave3_end = waves[2]['end']['price']
                wave2_end = waves[1]['end']['price']
                if current_price > wave3_end:
                    return 'WAVE_4_OR_5'
                elif current_price > wave2_end:
                    return 'WAVE_3'
                else:
                    return 'WAVE_2'
            else:
                return 'EARLY_STRUCTURE'
        except:
            return 'UNKNOWN'

    def determine_current_position_bearish(self, waves, current_price):
        """Determine current wave position in bearish structure"""
        try:
            if len(waves) >= 4:
                wave4_end = waves[3]['end']['price']
                wave3_end = waves[2]['end']['price']
                wave2_end = waves[1]['end']['price']
                wave1_end = waves[0]['end']['price']
                
                # For bearish: lower prices = further in pattern
                if current_price < wave4_end:
                    return 'WAVE_5'
                elif current_price < wave3_end:
                    return 'WAVE_4_OR_5'
                elif current_price < wave2_end:
                    return 'WAVE_3'
                elif current_price < wave1_end:
                    return 'WAVE_2'
                else:
                    return 'EARLY_STRUCTURE'
                    
            elif len(waves) >= 3:
                wave3_end = waves[2]['end']['price']
                wave2_end = waves[1]['end']['price']
                wave1_end = waves[0]['end']['price']
                
                if current_price < wave3_end:
                    return 'WAVE_4_OR_5'
                elif current_price < wave2_end:
                    return 'WAVE_3'
                elif current_price < wave1_end:
                    return 'WAVE_2'
                else:
                    return 'CORRECTIVE_BOUNCE'
                    
            elif len(waves) >= 2:
                wave2_end = waves[1]['end']['price']
                wave1_end = waves[0]['end']['price']
                
                if current_price < wave2_end:
                    return 'WAVE_3'
                elif current_price < wave1_end:
                    return 'WAVE_2'
                else:
                    return 'CORRECTIVE_BOUNCE'
            else:
                return 'WAVE_1_OR_2'
        except:
            return 'UNKNOWN'

    def calculate_bullish_targets(self, waves, current_price):
        """Calculate realistic bullish Elliott Wave targets"""
        try:
            if len(waves) >= 4:
                # We have Wave 4, calculate Wave 5 targets
                wave1_size = waves[0]['end']['price'] - waves[0]['start']['price']
                wave4_end = waves[3]['end']['price']
                
                target1 = wave4_end + (wave1_size * 0.618)  # 61.8% projection
                target2 = wave4_end + wave1_size            # 100% projection
                target3 = wave4_end + (wave1_size * 1.618)  # 161.8% projection
                
                return [target1, target2, target3]
            
            elif len(waves) >= 3:
                # Estimate based on Wave 3 completion
                wave3_end = waves[2]['end']['price']
                wave1_size = waves[0]['end']['price'] - waves[0]['start']['price']
                
                # Estimate Wave 4 correction (typically 38% of Wave 3)
                wave3_size = waves[2]['end']['price'] - waves[2]['start']['price']
                estimated_wave4_end = wave3_end - (wave3_size * 0.38)
                
                target1 = estimated_wave4_end + (wave1_size * 0.618)
                target2 = estimated_wave4_end + wave1_size
                target3 = estimated_wave4_end + (wave1_size * 1.618)
                
                return [target1, target2, target3]
            
            return [current_price * 1.1, current_price * 1.2, current_price * 1.3]
            
        except:
            return [current_price * 1.1, current_price * 1.2, current_price * 1.3]

    def calculate_bearish_targets(self, waves, current_price):
        """Calculate realistic bearish Elliott Wave targets with proper math"""
        try:
            # Use simple, realistic percentage-based targets instead of complex calculations
            # that can go negative
            
            if len(waves) >= 4:
                # For complete patterns, use conservative targets
                target1 = current_price * 0.75  # 25% decline
                target2 = current_price * 0.60  # 40% decline  
                target3 = current_price * 0.45  # 55% decline
                
                return [target1, target2, target3]
            
            elif len(waves) >= 3:
                # For Wave 3 patterns, slightly more aggressive but still realistic
                target1 = current_price * 0.80  # 20% decline
                target2 = current_price * 0.65  # 35% decline
                target3 = current_price * 0.50  # 50% decline
                
                return [target1, target2, target3]
            
            # Default conservative bearish targets
            return [current_price * 0.85, current_price * 0.70, current_price * 0.55]
            
        except:
            return [current_price * 0.85, current_price * 0.70, current_price * 0.55]

    def calculate_quality_score_bullish(self, waves):
        """Calculate quality score for bullish pattern"""
        score = 40  # Base score
        
        try:
            if len(waves) >= 2:
                # Wave 2 retracement quality
                wave2_retrace = waves[1]['retrace']
                if 38 <= wave2_retrace <= 62:
                    score += 20
                elif 25 <= wave2_retrace <= 75:
                    score += 10
            
            if len(waves) >= 3:
                # Wave 3 strength
                if waves[2]['size'] >= 20:  # Strong Wave 3
                    score += 15
                elif waves[2]['size'] >= 15:
                    score += 10
                
                # Wave 3 vs Wave 1 ratio
                if hasattr(waves[2], 'vs_wave1') and waves[2]['vs_wave1'] >= 1.618:
                    score += 15
                elif hasattr(waves[2], 'vs_wave1') and waves[2]['vs_wave1'] >= 1.0:
                    score += 10
            
            if len(waves) >= 4:
                # Wave 4 retracement quality
                wave4_retrace = waves[3]['retrace']
                if 23 <= wave4_retrace <= 50:
                    score += 10
                elif 15 <= wave4_retrace <= 65:
                    score += 5
            
        except:
            pass
        
        return min(score, 100)

    def calculate_quality_score_bearish(self, waves):
        """Calculate quality score for bearish pattern"""
        score = 40  # Base score
        
        try:
            if len(waves) >= 2:
                wave2_retrace = waves[1]['retrace']
                if 38 <= wave2_retrace <= 62:
                    score += 20
                elif 25 <= wave2_retrace <= 75:
                    score += 10
            
            if len(waves) >= 3:
                if waves[2]['size'] >= 20:
                    score += 15
                elif waves[2]['size'] >= 15:
                    score += 10
            
        except:
            pass
        
        return min(score, 100)

    def analyze_symbol(self, symbol):
        """Analyze single symbol for current Elliott Wave structure"""
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
                print("Insufficient major pivots")
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
                    'wave_start_date': pattern['wave_start']['date'].strftime('%Y-%m-%d'),
                    'waves': pattern['waves']
                }
                results.append(setup)
            
            print(f"Found {len(results)} current structures")
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all high-liquidity symbols for current Elliott Wave structures"""
        print("🌊 ELLIOTT WAVE SCANNER - DAILY CHART (CURRENT STRUCTURE FOCUS)")
        print("=" * 85)
        print("📈 Scanning for current major Elliott Wave structures...")
        print("📊 Strategy: Current cycle analysis → Major trend targets")
        print("💰 Filtering: $500K+ daily volume, top 150 most liquid pairs")
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
                if i % 25 == 0:
                    print(f"📊 Progress: {i}/{len(symbols)} pairs analyzed...")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"📡 {symbol.replace('/USDT', '')}... Error: {e}")
                continue
        
        return all_patterns

    def display_complete_wave_analysis(self, pattern):
        """Display complete Elliott Wave analysis with trading signals"""
        waves = pattern['waves']
        current_price = pattern['current_price']
        targets = pattern['targets']
        direction = pattern['direction']
        current_wave = pattern['current_wave']
        
        # Current Fibonacci targets (whatever wave is active)
        print(f"\n   🎯 CURRENT WAVE FIBONACCI TARGETS:")
        if current_wave == 'WAVE_3':
            print(f"   📈 WAVE 3 TARGETS: T1: ${targets[0]:.2f} (61.8%) | T2: ${targets[1]:.2f} (100%) | T3: ${targets[2]:.2f} (161.8%)")
        elif current_wave in ['WAVE_4', 'WAVE_4_OR_5']:
            print(f"   📈 WAVE 5 TARGETS: T1: ${targets[0]:.2f} (61.8%) | T2: ${targets[1]:.2f} (100%) | T3: ${targets[2]:.2f} (161.8%)")
        elif current_wave == 'WAVE_5':
            print(f"   📈 WAVE 5 TARGETS: T1: ${targets[0]:.2f} (61.8%) | T2: ${targets[1]:.2f} (100%) | T3: ${targets[2]:.2f} (161.8%)")
        else:
            print(f"   📈 TARGETS: T1: ${targets[0]:.2f} (61.8%) | T2: ${targets[1]:.2f} (100%) | T3: ${targets[2]:.2f} (161.8%)")
        
        # Calculate complete wave cycle projections
        if direction == 'BULLISH':
            self.display_bullish_wave_cycle(pattern, waves, current_price)
        else:
            self.display_bearish_wave_cycle(pattern, waves, current_price)
    
    def display_bullish_wave_cycle(self, pattern, waves, current_price):
        """Display complete bullish Elliott Wave cycle with trading signals"""
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
                print(f"   Shallow: ${wave4_shallow:.2f} (23.6%) | Normal: ${wave4_normal:.2f} (38.2%)")
                print(f"   Deep: ${wave4_deep:.2f} (50%) | Maximum: ${wave4_max:.2f} (61.8%)")
            
            # Wave 5 ultimate targets (from Wave 4 completion)
            estimated_wave4_end = wave4_normal if 'wave4_normal' in locals() else wave3_end * 0.85
            wave5_t1 = estimated_wave4_end + (wave1_size * 0.618)
            wave5_t2 = estimated_wave4_end + wave1_size
            wave5_t3 = estimated_wave4_end + (wave1_size * 1.618)
            wave5_t4 = estimated_wave4_end + (wave1_size * 2.618)
            
            print(f"\n   🚀 ULTIMATE WAVE 5 TARGETS (After Wave 4):")
            print(f"   W5-T1: ${wave5_t1:.2f} (61.8%) | W5-T2: ${wave5_t2:.2f} (100%)")
            print(f"   W5-T3: ${wave5_t3:.2f} (161.8%) | W5-T4: ${wave5_t4:.2f} (261.8%)")
        
        # Trading signals based on current wave
        print(f"\n   💡 COMPLETE TRADING STRATEGY:")
        
        if current_wave == 'WAVE_3':
            print(f"   📊 CURRENT STATUS: Wave 3 in progress (strongest wave)")
            print(f"   🎯 IF OWNED:")
            print(f"      • HOLD through Wave 3 completion")
            print(f"      • Take 25% profits at each Wave 3 target")
            print(f"      • Keep 25% for Wave 4 dip re-entry")
            print(f"   🎯 IF NOT OWNED:")
            print(f"      • WAIT for Wave 4 correction")
            print(f"      • BUY ZONE: ${wave4_normal:.2f} - ${wave4_deep:.2f} (Wave 4 dip)")
            print(f"      • STOP LOSS: ${wave4_max * 0.95:.2f} (below Wave 4 max)")
            print(f"      • TARGET: Wave 5 completion ${wave5_t2:.2f} - ${wave5_t3:.2f}")
            print(f"      • STRATEGY: Be patient - wait for the pullback opportunity!")
            
        elif current_wave in ['WAVE_4', 'WAVE_4_OR_5']:
            # Smart logic: Check if we need to wait for Wave 4 or if it's completed
            wave4_entry_zone_high = wave4_shallow if 'wave4_shallow' in locals() else current_price * 0.90
            wave4_entry_zone_low = wave4_deep if 'wave4_deep' in locals() else current_price * 0.85
            
            if current_price > wave4_entry_zone_high:
                # Price too high - wait for Wave 4 correction
                print(f"   📊 CURRENT STATUS: Wave 3 completion / Wave 4 correction pending")
                print(f"   ⚠️  ACTION: WAIT for Wave 4 pullback")
                print(f"   📉 MONITOR: Price decline to Wave 4 correction zone")
                print(f"   🛒 BUY ZONE: ${wave4_normal:.2f} - ${wave4_deep:.2f} (Wave 4 dip)")
                print(f"   🛑 STOP LOSS: ${wave4_max * 0.95:.2f} (below Wave 4 max)")
                print(f"   🎯 TARGETS: Wave 5 completion ${wave5_t1:.2f} - ${wave5_t3:.2f}")
                print(f"   💡 STRATEGY: Don't chase current levels - wait for the pullback!")
                
            elif wave4_entry_zone_low <= current_price <= wave4_entry_zone_high:
                # Price in Wave 4 correction zone - good to buy
                print(f"   📊 CURRENT STATUS: Wave 4 correction in progress")
                print(f"   🛒 ACTION: BUY NOW (Wave 4 dip opportunity)")
                print(f"   🛒 ENTRY: ${current_price * 0.98:.2f} - ${current_price * 1.02:.2f} (current levels)")
                print(f"   🛑 STOP LOSS: ${wave4_max * 0.95:.2f} (below Wave 4 max)")
                print(f"   🎯 TARGETS: Wave 5 completion ${wave5_t1:.2f} - ${wave5_t3:.2f}")
                print(f"   💡 STRATEGY: Excellent Wave 4 entry opportunity!")
                
            else:
                # Price below Wave 4 zone - potentially in Wave 5 or oversold
                print(f"   📊 CURRENT STATUS: Wave 4 completed / Wave 5 beginning")
                print(f"   🛒 ACTION: BUY NOW (Wave 5 setup)")
                print(f"   🛒 ENTRY: ${current_price * 0.98:.2f} - ${current_price * 1.02:.2f} (current levels)")
                if len(waves) >= 4:
                    wave4_low = waves[3]['end']['price']
                    print(f"   🛑 STOP LOSS: ${wave4_low * 0.95:.2f} (below Wave 4 low)")
                else:
                    print(f"   🛑 STOP LOSS: ${current_price * 0.90:.2f} (below support)")
                print(f"   🎯 TARGETS: Wave 5 completion ${wave5_t1:.2f} - ${wave5_t3:.2f}")
            
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
    
    def display_bearish_wave_cycle(self, pattern, waves, current_price):
        """Display complete bearish Elliott Wave cycle with trading signals"""
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
                print(f"   Shallow: ${wave4_shallow:.2f} (23.6%) | Normal: ${wave4_normal:.2f} (38.2%)")
                print(f"   Deep: ${wave4_deep:.2f} (50%) | Maximum: ${wave4_max:.2f} (61.8%)")
            
            # Wave 5 downside targets (using conservative percentage-based approach)
            print(f"\n   📉 ULTIMATE WAVE 5 DOWNSIDE TARGETS:")
            target1 = current_price * 0.75  # 25% decline
            target2 = current_price * 0.60  # 40% decline
            target3 = current_price * 0.45  # 55% decline
            print(f"   W5-T1: ${target1:.2f} (25% decline) | W5-T2: ${target2:.2f} (40% decline) | W5-T3: ${target3:.2f} (55% decline)")
        
        # Bearish trading signals
        print(f"\n   💡 COMPLETE TRADING STRATEGY:")
        
        if current_wave == 'WAVE_2':
            print(f"   📊 CURRENT STATUS: Bearish Wave 2 bounce (correction)")
            print(f"   🎯 IDEAL SHORT SETUP:")
            print(f"      • Wave 2 bounce ending (best short opportunity)")
            if 'wave4_normal' in locals() and 'wave4_deep' in locals():
                print(f"      • SHORT ZONE: ${wave4_normal:.2f} - ${wave4_deep:.2f} (Wave 2 resistance)")
            else:
                print(f"      • SHORT ZONE: Current resistance levels")
            print(f"      • STOP LOSS: ${current_price * 1.15:.2f} (above Wave 2 high)")
            print(f"      • TARGET: Wave 3 decline targets")
            print(f"      • R:R: Excellent (Wave 3 strongest decline)")
            
        elif current_wave == 'WAVE_3':
            if len(waves) >= 3:
                wave3_start = waves[2]['start']['price']
                if current_price <= wave3_start * 1.05:
                    print(f"   📊 CURRENT STATUS: Bearish Wave 3 in progress (strongest decline)")
                    print(f"   🎯 IF SHORT:")
                    print(f"      • HOLD through Wave 3 completion")
                    print(f"      • Take 25% profits at each Wave 3 target")
                    print(f"      • Trail stop loss aggressively")
                    print(f"   🎯 IF NOT SHORT:")
                    print(f"      • AVOID new shorts (Wave 3 already started)")
                    print(f"      • WAIT for Wave 4 bounce to short")
                    if 'wave4_normal' in locals() and 'wave4_deep' in locals():
                        print(f"      • Next SHORT ZONE: ${wave4_normal:.2f} - ${wave4_deep:.2f}")
                else:
                    print(f"   📊 CURRENT STATUS: Above Wave 3 start - likely Wave 4 bounce")
                    print(f"   🎯 CORRECTIVE BOUNCE:")
                    if 'wave4_normal' in locals() and 'wave4_deep' in locals():
                        print(f"      • SHORT ZONE: ${wave4_normal:.2f} - ${wave4_deep:.2f} (current area)")
                    else:
                        print(f"      • SHORT ZONE: Current bounce area")
                    print(f"      • STOP LOSS: ${current_price * 1.10:.2f} (above bounce high)")
                    print(f"      • TARGET: Wave 5 decline targets")
                
        elif current_wave in ['WAVE_4', 'WAVE_4_OR_5']:
            print(f"   📊 CURRENT STATUS: Wave 4 bounce ending / Wave 5 beginning")
            print(f"   🛒 ACTION: SHORT NOW (Wave 5 setup)")
            print(f"   🛒 ENTRY: ${current_price * 0.98:.2f} - ${current_price * 1.02:.2f}")
            print(f"   🛑 STOP LOSS: ${current_price * 1.15:.2f} (above Wave 4 high)")
            print(f"   🎯 TARGETS: Wave 5 decline targets")
            
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
            
        elif current_wave == 'CORRECTIVE_BOUNCE':
            print(f"   📊 CURRENT STATUS: Corrective bounce in bearish trend")
            print(f"   🎯 POTENTIAL SHORT:")
            print(f"      • Monitor for bounce failure")
            print(f"      • SHORT on weakness below ${current_price * 0.95:.2f}")
            print(f"      • STOP above recent bounce high")
            
        else:
            print(f"   📊 CURRENT STATUS: Early bearish structure")
            print(f"   🎯 MONITOR: Wait for clearer pattern development")

    def display_results(self, patterns):
        """Display Elliott Wave scan results with detailed wave breakdown"""
        if not patterns:
            symbols_count = len(self.get_symbols()) if hasattr(self, 'get_symbols') else 150
            print(f"\n📊 SCAN RESULTS:")
            print(f"Total scanned: {symbols_count}")
            print("Current structures: 0")
            print("=" * 89)
            print("🌊 FOUND 0 CURRENT ELLIOTT WAVE STRUCTURES")
            print("=" * 89)
            print("❌ No major current Elliott Wave structures found")
            print("💡 Market may be in consolidation or early structure formation")
            return
        
        # Sort by quality score
        patterns.sort(key=lambda x: x['quality_score'], reverse=True)
        
        symbols_count = len(self.get_symbols()) if hasattr(self, 'get_symbols') else 150
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {symbols_count}")
        print(f"Current structures: {len(patterns)}")
        print("=" * 89)
        print(f"🌊 FOUND {len(patterns)} CURRENT ELLIOTT WAVE STRUCTURES")
        print("=" * 89)
        
        for i, pattern in enumerate(patterns, 1):
            direction_icon = "📈" if pattern['direction'] == 'BULLISH' else "📉"
            
            print(f"\n{i}. {direction_icon} {pattern['symbol'].replace('/USDT', '')} | Score: {pattern['quality_score']:.0f}/100 | 🌊 {pattern['current_wave']} | {pattern['direction']}")
            print(f"   Current: ${pattern['current_price']:.2f} | Duration: {pattern['duration']} days | Start: {pattern['wave_start_date']}")
            
            # DETAILED WAVE STRUCTURE BREAKDOWN
            print(f"\n   📊 DETAILED WAVE STRUCTURE:")
            waves = pattern['waves']
            
            # Wave Start
            print(f"   🏁 Wave Start: ${pattern['wave_start']:.2f} ({pattern['wave_start_date']})")
            
            # Wave 1
            if len(waves) >= 1:
                wave1 = waves[0]
                wave1_date = wave1['end']['date'].strftime('%Y-%m-%d') if hasattr(wave1['end'], 'get') and 'date' in wave1['end'] else "N/A"
                print(f"   1️⃣ Wave 1 End: ${wave1['end']['price']:.2f} ({wave1_date}) | Move: +{wave1['size']:.1f}%")
            
            # Wave 2
            if len(waves) >= 2:
                wave2 = waves[1]
                wave2_date = wave2['end']['date'].strftime('%Y-%m-%d') if hasattr(wave2['end'], 'get') and 'date' in wave2['end'] else "N/A"
                print(f"   2️⃣ Wave 2 End: ${wave2['end']['price']:.2f} ({wave2_date}) | Retrace: -{wave2['retrace']:.1f}%")
            
            # Wave 3
            if len(waves) >= 3:
                wave3 = waves[2]
                wave3_date = wave3['end']['date'].strftime('%Y-%m-%d') if hasattr(wave3['end'], 'get') and 'date' in wave3['end'] else "N/A"
                print(f"   3️⃣ Wave 3 End: ${wave3['end']['price']:.2f} ({wave3_date}) | Move: +{wave3['size']:.1f}%")
            
            # Wave 4 (if exists)
            if len(waves) >= 4:
                wave4 = waves[3]
                wave4_date = wave4['end']['date'].strftime('%Y-%m-%d') if hasattr(wave4['end'], 'get') and 'date' in wave4['end'] else "N/A"
                print(f"   4️⃣ Wave 4 End: ${wave4['end']['price']:.2f} ({wave4_date}) | Retrace: -{wave4['retrace']:.1f}%")
            
            # Current position analysis
            print(f"\n   📍 CURRENT POSITION:")
            if pattern['current_wave'] == 'WAVE_3':
                if len(waves) >= 3:
                    wave3_start = waves[2]['start']['price']
                    progress = (pattern['current_price'] - wave3_start) / wave3_start * 100
                    print(f"   🚀 Wave 3 Progress: ${wave3_start:.2f} → ${pattern['current_price']:.2f} (+{progress:.1f}%)")
                else:
                    print(f"   🚀 Wave 3 in progress from ${pattern['current_price']:.2f}")
            elif pattern['current_wave'] == 'WAVE_4_OR_5':
                print(f"   🎯 Post-Wave 3: Current ${pattern['current_price']:.2f} - Wave 4 correction or Wave 5 beginning")
            elif pattern['current_wave'] == 'WAVE_5':
                if len(waves) >= 4:
                    wave5_start = waves[3]['end']['price']
                    progress = (pattern['current_price'] - wave5_start) / wave5_start * 100
                    print(f"   ⚠️  Wave 5 Progress: ${wave5_start:.2f} → ${pattern['current_price']:.2f} (+{progress:.1f}%)")
                else:
                    print(f"   ⚠️  Wave 5 active from ${pattern['current_price']:.2f}")
            
            # Complete Wave Cycle Analysis & Trading Signals
            self.display_complete_wave_analysis(pattern)
            
            print("-" * 85)

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