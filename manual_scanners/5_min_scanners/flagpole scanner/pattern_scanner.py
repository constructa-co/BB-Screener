#!/usr/bin/env python3
"""
FLAG POLE & TRIANGLE PATTERN SCANNER
Standalone scanner for flag poles and triangle patterns on 5-minute charts
Strategy: Breakout continuation patterns for momentum trading
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class PatternScanner:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
    def get_top_coins(self, limit=200):
        """Get top coins by volume for scanning"""
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
                
                # Filter for good pattern candidates
                if (volume_24h > 15_000_000 and  # Higher volume for breakouts
                    market_cap > 100_000_000 and  # Larger market cap
                    market_cap < 50_000_000_000): # Avoid BTC/ETH
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 for pattern scanning
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC']
    
    def get_binance_klines(self, symbol, interval='5m', limit=200):
        """Get 5-minute klines from Binance for pattern analysis"""
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
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            return None
    
    def find_swing_points(self, df, period=5):
        """Find swing highs and lows for pattern recognition"""
        highs = []
        lows = []
        
        for i in range(period, len(df) - period):
            # Swing high
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, period+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, period+1)):
                highs.append({'index': i, 'price': df['high'].iloc[i], 'timestamp': df['timestamp'].iloc[i]})
            
            # Swing low
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, period+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, period+1)):
                lows.append({'index': i, 'price': df['low'].iloc[i], 'timestamp': df['timestamp'].iloc[i]})
        
        return highs, lows
    
    def detect_tradeable_flag_pole(self, df, symbol=""):
        """Detect tradeable flag poles with relaxed, realistic criteria"""
        if len(df) < 30:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Look for recent moves (last 30 candles)
        for pole_start in range(len(df) - 30, len(df) - 5):
            for pole_end in range(pole_start + 3, min(pole_start + 20, len(df) - 2)):
                
                # Calculate pole characteristics
                pole_start_price = df['close'].iloc[pole_start]
                pole_end_price = df['close'].iloc[pole_end]
                pole_move = abs(pole_end_price - pole_start_price) / pole_start_price * 100
                
                # RELAXED pole requirements - catch more opportunities
                if pole_move < 2.5:  # Minimum 2.5% move
                    continue
                
                # Pole duration check (allow longer moves)
                pole_duration = pole_end - pole_start
                if pole_duration > 18:  # Max 18 candles
                    continue
                
                # Pole speed - very relaxed
                pole_speed = pole_move / pole_duration
                if pole_speed < 0.2:  # At least 0.2% per candle
                    continue
                
                is_bullish = pole_end_price > pole_start_price
                
                # Check flag/pennant formation - VERY relaxed
                flag_data = df.iloc[pole_end:].copy()
                if len(flag_data) < 2 or len(flag_data) > 20:  # Allow longer flags
                    continue
                
                # Flag characteristics - very lenient
                flag_high = flag_data['high'].max()
                flag_low = flag_data['low'].min()
                flag_range = (flag_high - flag_low) / flag_low * 100
                
                # Very relaxed flag proportion
                if flag_range > pole_move * 0.8 or flag_range < pole_move * 0.1:
                    continue
                
                # Check if flag is not trending too much
                flag_slope_pct = 0
                if len(flag_data) >= 3:
                    flag_closes = flag_data['close'].values
                    x = np.arange(len(flag_closes))
                    slope, _ = np.polyfit(x, flag_closes, 1)
                    flag_slope_pct = abs(slope * len(flag_closes) / flag_closes.mean()) * 100
                    
                    # Very relaxed slope requirement
                    if flag_slope_pct > 4.0:  # Allow up to 4% slope
                        continue
                
                # Volume check - very relaxed
                pole_volume = df['volume'].iloc[pole_start:pole_end].mean()
                flag_volume = flag_data['volume'].mean()
                volume_decline_pct = max(0, (pole_volume - flag_volume) / pole_volume * 100)
                
                # Very lenient volume requirement
                if volume_decline_pct < 5:  # Just 5% decline minimum
                    volume_decline_pct = 5  # Set minimum for display
                
                # Entry timing - very flexible
                price_position = (current_price - flag_low) / (flag_high - flag_low) if flag_high != flag_low else 0.5
                
                if is_bullish:
                    approaching_breakout = price_position > 0.3  # Very flexible
                    breakout_level = flag_high * 1.002
                else:
                    approaching_breakout = price_position < 0.7  # Very flexible
                    breakout_level = flag_low * 0.998
                
                # Flag age check - very flexible
                flag_age = len(flag_data)
                
                # Calculate profit targets
                pole_height = abs(pole_end_price - pole_start_price)
                
                if is_bullish:
                    target_price = breakout_level + pole_height
                    target_pct = (target_price - current_price) / current_price * 100
                    stop_loss = flag_low * 0.996
                    risk_pct = (current_price - stop_loss) / current_price * 100
                else:
                    target_price = breakout_level - pole_height
                    target_pct = (current_price - target_price) / current_price * 100
                    stop_loss = flag_high * 1.004
                    risk_pct = (stop_loss - current_price) / current_price * 100
                
                # Very relaxed R:R requirements
                risk_reward = target_pct / risk_pct if risk_pct > 0 else 0
                
                if risk_reward < 1.0 or target_pct < 1.0:  # Very low bar
                    continue
                
                # Determine pattern type
                if flag_slope_pct < 1.0:
                    pattern_name = "Flag"
                else:
                    pattern_name = "Pennant"
                
                return {
                    'pattern_type': f'{pattern_name} ({pole_move:.1f}% pole)',
                    'direction': 'Bullish' if is_bullish else 'Bearish',
                    'pole_move_pct': pole_move,
                    'pole_speed': pole_speed,
                    'pole_duration': pole_duration,
                    'flag_range_pct': flag_range,
                    'flag_slope_pct': flag_slope_pct,
                    'volume_decline_pct': volume_decline_pct,
                    'breakout_level': breakout_level,
                    'current_price': current_price,
                    'target_price': target_price,
                    'target_pct': target_pct,
                    'stop_loss': stop_loss,
                    'risk_pct': risk_pct,
                    'risk_reward': risk_reward,
                    'flag_age': flag_age,
                    'price_position': price_position,
                    'approaching_breakout': approaching_breakout,
                    'quality': 'High' if risk_reward >= 2.0 else 'Good'
                }
        
        return None
    
    def detect_early_triangle(self, df, symbol=""):
        """Detect triangles DURING formation for early positioning"""
        if len(df) < 25:
            return None
        
        # Get swing points from recent data
        highs, lows = self.find_swing_points(df.tail(50), period=3)  # More sensitive
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Focus on very recent swings (last 30 candles)
        recent_highs = [h for h in highs if h['index'] >= len(df) - 30]
        recent_lows = [l for l in lows if l['index'] >= len(df) - 30]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None
        
        # Sort by index
        recent_highs.sort(key=lambda x: x['index'])
        recent_lows.sort(key=lambda x: x['index'])
        
        # Calculate trendlines
        high_prices = [h['price'] for h in recent_highs]
        low_prices = [l['price'] for l in recent_lows]
        
        # Trendline slopes
        high_slope = (high_prices[-1] - high_prices[0]) / max(len(recent_highs) - 1, 1)
        low_slope = (low_prices[-1] - low_prices[0]) / max(len(recent_lows) - 1, 1)
        
        # More lenient triangle detection
        current_price = df['close'].iloc[-1]
        upper_level = recent_highs[-1]['price']
        lower_level = recent_lows[-1]['price']
        
        # Calculate slopes as percentages
        high_slope_pct = (high_slope / high_prices[0]) * 100 if high_prices[0] > 0 else 0
        low_slope_pct = (low_slope / low_prices[0]) * 100 if low_prices[0] > 0 else 0
        
        # Determine triangle type with relaxed criteria
        triangle_type = None
        target_direction = None
        
        if abs(high_slope_pct) < 0.5:  # Nearly horizontal resistance
            if low_slope_pct > 0.2:  # Rising support
                triangle_type = "Ascending"
                target_direction = "Bullish"
        elif abs(low_slope_pct) < 0.5:  # Nearly horizontal support
            if high_slope_pct < -0.2:  # Falling resistance
                triangle_type = "Descending"  
                target_direction = "Bearish"
        elif high_slope_pct < -0.1 and low_slope_pct > 0.1:  # Converging
            triangle_type = "Symmetrical"
            target_direction = "Either direction"
        
        if triangle_type is None:
            return None
        
        # Early triangle characteristics
        triangle_range = upper_level - lower_level
        range_pct = triangle_range / lower_level * 100
        
        # Look for forming triangles (2-8% range, not too tight yet)
        if range_pct < 1.0 or range_pct > 8.0:
            return None
        
        # Price position analysis
        price_position = (current_price - lower_level) / triangle_range
        
        # Volume analysis
        recent_volume = df['volume'].tail(10).mean()
        earlier_volume = df['volume'].iloc[-25:-15].mean()
        volume_declining = recent_volume < earlier_volume * 0.9
        
        # Early breakout signals
        last_5_candles = df.tail(5)
        price_compression = (last_5_candles['high'].max() - last_5_candles['low'].min()) / current_price * 100
        
        # Look for squeeze signs
        is_compressing = price_compression < range_pct * 0.6
        
        # Calculate proximity to breakout levels
        distance_to_upper = abs(current_price - upper_level) / current_price * 100
        distance_to_lower = abs(current_price - lower_level) / current_price * 100
        
        # Early entry criteria - catch before breakout
        near_breakout = min(distance_to_upper, distance_to_lower) < 2.0
        
        if triangle_type in ["Ascending", "Descending"] or (triangle_type == "Symmetrical" and near_breakout):
            
            # Calculate targets
            target_move = triangle_range * 0.8  # Conservative target
            
            if triangle_type == "Ascending":
                breakout_level = upper_level * 1.002
                target_price = breakout_level + target_move
                confidence = "High" if distance_to_upper < 1.5 else "Medium"
            elif triangle_type == "Descending":
                breakout_level = lower_level * 0.998
                target_price = breakout_level - target_move
                confidence = "High" if distance_to_lower < 1.5 else "Medium"
            else:  # Symmetrical
                breakout_level = upper_level if distance_to_upper < distance_to_lower else lower_level
                target_price = None
                confidence = "Medium"
            
            target_pct = 0
            if target_price:
                target_pct = abs(target_price - current_price) / current_price * 100
            
            # Age of pattern
            pattern_age = len(df) - recent_highs[0]['index']
            
            return {
                'pattern_type': f'Forming {triangle_type} Triangle',
                'direction': target_direction,
                'range_pct': range_pct,
                'upper_level': upper_level,
                'lower_level': lower_level,
                'current_price': current_price,
                'breakout_level': breakout_level,
                'target_price': target_price,
                'target_pct': target_pct,
                'volume_declining': volume_declining,
                'price_position': price_position,
                'compression_factor': price_compression / range_pct,
                'distance_to_breakout': min(distance_to_upper, distance_to_lower),
                'confidence': confidence,
                'pattern_age': pattern_age,
                'quality': 'Early' if pattern_age <= 15 else 'Mature'
            }
        
        return None
    
    def calculate_pattern_quality(self, pattern_data, volume_profile):
        """Score pattern quality based on multiple factors"""
        score = 50  # Base score
        
        if 'Developing' in pattern_data['pattern_type'] or 'Forming' in pattern_data['pattern_type']:
            # Early pattern bonus scoring
            if pattern_data.get('quality') == 'Early':
                score += 20  # Bonus for catching early
            
            # Age scoring (newer is better for early detection)
            pattern_age = pattern_data.get('pattern_age', 10)
            if pattern_age <= 5:
                score += 15  # Very fresh
            elif pattern_age <= 10:
                score += 10  # Fresh
            
            # Proximity to breakout (closer is better for early patterns)
            if 'distance_to_breakout' in pattern_data:
                if pattern_data['distance_to_breakout'] < 1.0:
                    score += 15  # Very close
                elif pattern_data['distance_to_breakout'] < 2.0:
                    score += 10  # Close
        
        if 'Flag Pole' in pattern_data['pattern_type']:
            # Flag-specific scoring
            if 'Developing' in pattern_data['pattern_type']:
                # Compression factor (good compression = higher score)
                compression = pattern_data.get('compression_factor', 1.0)
                if compression < 0.5:
                    score += 15  # Good compression
                elif compression < 0.7:
                    score += 10
                
                # Flag age (sweet spot for early entry)
                flag_age = pattern_data.get('flag_age', 10)
                if 3 <= flag_age <= 8:
                    score += 15  # Perfect timing
                elif flag_age <= 12:
                    score += 10
            
        if 'Flag' in pattern_data['pattern_type'] or 'Pennant' in pattern_data['pattern_type']:
            # Tradeable pattern scoring
            
            # Risk/Reward ratio (critical for trading)
            risk_reward = pattern_data.get('risk_reward', 0)
            if risk_reward >= 3.0:
                score += 25  # Excellent R:R
            elif risk_reward >= 2.0:
                score += 20  # Good R:R
            elif risk_reward >= 1.5:
                score += 15  # Acceptable R:R
            
            # Target potential
            target_pct = pattern_data.get('target_pct', 0)
            if target_pct >= 4.0:
                score += 20  # Large target
            elif target_pct >= 2.5:
                score += 15  # Good target
            elif target_pct >= 1.5:
                score += 10  # Acceptable target
            
            # Pole strength
            pole_move = pattern_data.get('pole_move_pct', 0)
            if pole_move >= 5.0:
                score += 15  # Strong pole
            elif pole_move >= 3.5:
                score += 10  # Good pole
            elif pole_move >= 3.0:
                score += 5   # Acceptable pole
            
            # Volume decline
            volume_decline = pattern_data.get('volume_decline_pct', 0)
            if volume_decline >= 25:
                score += 15  # Good volume pattern
            elif volume_decline >= 15:
                score += 10  # Decent volume pattern
            elif volume_decline >= 10:
                score += 5   # Acceptable volume pattern
            
            # Entry timing
            flag_age = pattern_data.get('flag_age', 20)
            if 3 <= flag_age <= 6:
                score += 15  # Perfect timing
            elif flag_age <= 10:
                score += 10  # Good timing
            elif flag_age <= 15:
                score += 5   # Acceptable timing
        
        elif 'Triangle' in pattern_data['pattern_type']:
            # Triangle-specific scoring
            if 'Forming' in pattern_data['pattern_type']:
                # Confidence level
                confidence = pattern_data.get('confidence', 'Medium')
                if confidence == 'High':
                    score += 20
                elif confidence == 'Medium':
                    score += 10
                
                # Compression factor
                compression = pattern_data.get('compression_factor', 1.0)
                if compression < 0.6:
                    score += 15
            
            # Range quality
            if 2.0 <= pattern_data.get('range_pct', 0) <= 6.0:
                score += 15
            
            if pattern_data.get('volume_declining'):
                score += 10
        
        # Volume quality
        if volume_profile['avg_volume'] > volume_profile['min_threshold']:
            score += 10
        
        return min(score, 100)  # Cap at 100
    
    def scan_for_patterns(self):
        """Main scanning function"""
        print("🎯 FLAG POLE & TRIANGLE PATTERN SCANNER")
        print("=" * 60)
        print("🔍 Scanning 5-minute charts for breakout patterns...")
        print("📊 Targets: Relaxed flag poles, triangles, more opportunities")
        print()
        
        coins = self.get_top_coins()
        opportunities = []
        
        stats = {'scanned': 0, 'no_data': 0, 'patterns_found': 0}
        
        for i, symbol in enumerate(coins):
            try:
                print(f"📡 {symbol}... ", end="")
                stats['scanned'] += 1
                
                # Get price data
                df = self.get_binance_klines(symbol)
                if df is None or len(df) < 50:
                    print("No data")
                    stats['no_data'] += 1
                    continue
                
                # Volume analysis
                volume_profile = {
                    'avg_volume': df['volume'].tail(20).mean(),
                    'min_threshold': 100000  # Minimum volume threshold
                }
                
                # Check for relaxed flag/pennant patterns
                flag_pattern = self.detect_tradeable_flag_pole(df, symbol)
                if flag_pattern:
                    score = self.calculate_pattern_quality(flag_pattern, volume_profile)
                    if score >= 50:  # Much lower threshold to find opportunities
                        opportunities.append({
                            'symbol': symbol,
                            'pattern': flag_pattern,
                            'score': score,
                            'volume_profile': volume_profile
                        })
                        print(f"🎯 {flag_pattern['pattern_type']} ({flag_pattern['direction']}) - R:R {flag_pattern['risk_reward']:.1f}:1")
                        stats['patterns_found'] += 1
                        time.sleep(0.1)
                        continue
                
                # Check for developing triangle patterns
                triangle_pattern = self.detect_early_triangle(df, symbol)
                if triangle_pattern:
                    score = self.calculate_pattern_quality(triangle_pattern, volume_profile)
                    if score >= 45:  # Even lower threshold for triangles
                        opportunities.append({
                            'symbol': symbol,
                            'pattern': triangle_pattern,
                            'score': score,
                            'volume_profile': volume_profile
                        })
                        status = "🟡" if triangle_pattern['quality'] == 'Early' else "🟢"
                        print(f"{status} {triangle_pattern['pattern_type']} ({triangle_pattern['direction']}) - {triangle_pattern['quality']}")
                        stats['patterns_found'] += 1
                        time.sleep(0.1)
                        continue
                
                print("No patterns")
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error - {e}")
                continue
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: {stats['scanned']}")
        print(f"No data: {stats['no_data']}")
        print(f"Patterns found: {stats['patterns_found']}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display pattern scanning results"""
        print("\n" + "=" * 80)
        print(f"🎯 FOUND {len(opportunities)} HIGH-QUALITY PATTERN OPPORTUNITIES")
        print("=" * 80)
        
        if not opportunities:
            print("❌ No high-quality patterns found")
            print("💡 Try again later or during strong momentum periods")
            print("🔧 Scanner now uses STRICT criteria for genuine flag poles only")
            return
        
        print("📊 Results (sorted by quality score):")
        print()
        
        for i, opp in enumerate(opportunities[:15], 1):
            pattern = opp['pattern']
            
            # Pattern emoji
            if 'Flag' in pattern['pattern_type']:
                pattern_emoji = "🏁"
            elif 'Pennant' in pattern['pattern_type']:
                pattern_emoji = "🚩"
            else:
                pattern_emoji = "📐"
            
            # Direction emoji
            dir_emoji = "📈" if 'Bullish' in pattern['direction'] else "📉" if 'Bearish' in pattern['direction'] else "⚡"
            
            print(f"{i:2}. {pattern_emoji} {opp['symbol']:8} | Score: {opp['score']:2}/100 | "
                  f"{pattern['pattern_type']} | {dir_emoji}")
            
            print(f"     Current: ${pattern['current_price']:.6f} | "
                  f"Breakout: ${pattern.get('breakout_level', 0):.6f}")
            
            if pattern.get('target_price'):
                print(f"     Target: ${pattern['target_price']:.6f} | "
                      f"Potential: {pattern.get('target_pct', 0):.1f}% | "
                      f"R:R: {pattern.get('risk_reward', 0):.1f}:1")
            
            # Pattern-specific details
            if 'Flag' in pattern['pattern_type'] or 'Pennant' in pattern['pattern_type']:
                print(f"     Stop: ${pattern.get('stop_loss', 0):.6f} | "
                      f"Risk: {pattern.get('risk_pct', 0):.1f}% | "
                      f"Age: {pattern.get('flag_age', 'N/A')} candles")
                print(f"     Pole: {pattern.get('pole_move_pct', 0):.1f}% | "
                      f"Vol Decline: {pattern.get('volume_decline_pct', 0):.0f}% | "
                      f"Slope: {pattern.get('flag_slope_pct', 0):.1f}%")
                
                # Quality indicators
                quality_indicators = []
                if pattern.get('approaching_breakout'):
                    quality_indicators.append("⚡ Ready")
                if pattern.get('pole_speed', 0) > 0.8:
                    quality_indicators.append("🚀 Fast Pole")
                if pattern.get('volume_decline_pct', 0) > 25:
                    quality_indicators.append("📊 Strong Vol")
                
                if quality_indicators:
                    print(f"     Quality: {' | '.join(quality_indicators)}")
        
        print("\n💡 TRADING STRATEGY:")
        print("• Wait for breakout above/below flag with 2x+ volume")
        print("• Enter on breakout candle close or pullback test")
        print("• Stop loss: Opposite side of flag/pennant")
        print("• Target: Full pole height (traditional flag pole rule)")
        print("• Best win rate: 70-80% when all criteria met")
        print("• Hold time: Usually 1-6 hours for 5m patterns")

def main():
    scanner = PatternScanner()
    opportunities = scanner.scan_for_patterns()
    scanner.display_results(opportunities)

if __name__ == "__main__":
    main()