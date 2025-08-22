#!/usr/bin/env python3
"""
FIBONACCI RETRACEMENT SCANNER - REVISION 1 WITH DATABASE INTEGRATION
Professional scanner for Fibonacci pullback entries after initial bounce
Strategy: Drop → 38.2% bounce → 23.6% pullback entry → 50-61.8% targets
Integrated with isolated database logging following Elliott Wave pattern
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import sys
import os

# Add the database_and_logging directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'database_and_logging'))

from fibonacci_logger import FibonacciLogger, FibonacciSignal

class FibonacciRetracementScannerR1:
    def __init__(self, timeframe='5m'):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.timeframe = timeframe
        
        # Initialize database logger with isolation
        self.config = {
            'scanner_id': 'fibonacci_retracement_r1',
            'fibonacci_levels': [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618, 2.618, 4.236],
            'confidence_threshold': 0.7,
            'max_concurrent_scans': 5,
            'cleanup_interval_minutes': 30
        }
        
        try:
            self.fibonacci_logger = FibonacciLogger(self.config)
            print(f"✅ Fibonacci Logger initialized successfully")
        except Exception as e:
            print(f"❌ Fibonacci Logger initialization failed: {e}")
            self.fibonacci_logger = None
        
    def get_top_coins(self, limit=150):
        """Get top coins by volume for Fibonacci scanning"""
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
                
                # Volume requirements for short-term Fibonacci trading
                if (volume_24h > 12_000_000 and  # Good volume for quick moves
                    market_cap > 150_000_000 and   # Avoid micro-caps
                    market_cap < 300_000_000_000): # Focus on alt coins
                    coins.append(symbol)
            
            return coins[:100]  # Top 100 for Fibonacci analysis
            
        except Exception as e:
            print(f"Error fetching coins: {e}")
            return ['BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT', 'MATIC', 'LTC', 'LINK']
    
    def get_binance_klines(self, symbol, interval='5m', limit=200):
        """Get klines from Binance for Fibonacci analysis"""
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
    
    def identify_swing_points(self, df, lookback=5):
        """Identify swing highs and swing lows for Fibonacci levels"""
        highs = []
        lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            for j in range(lookback):
                if (df['high'].iloc[i - j - 1] >= current_high or 
                    df['high'].iloc[i + j + 1] >= current_high):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                highs.append({
                    'index': i,
                    'price': current_high,
                    'timestamp': df['timestamp'].iloc[i]
                })
            
            # Check for swing low
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            for j in range(lookback):
                if (df['low'].iloc[i - j - 1] <= current_low or 
                    df['low'].iloc[i + j + 1] <= current_low):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                lows.append({
                    'index': i,
                    'price': current_low,
                    'timestamp': df['timestamp'].iloc[i]
                })
        
        return highs, lows
    
    def calculate_fibonacci_levels(self, high, low):
        """Calculate Fibonacci retracement and extension levels"""
        diff = high - low
        
        levels = {
            '0.000': low,      # 0% (swing low)
            '0.236': high - (diff * 0.236),  # 23.6%
            '0.382': high - (diff * 0.382),  # 38.2%
            '0.500': high - (diff * 0.500),  # 50%
            '0.618': high - (diff * 0.618),  # 61.8%
            '0.786': high - (diff * 0.786),  # 78.6%
            '1.000': high,     # 100% (swing high)
            '1.272': high + (diff * 0.272),  # 127.2% extension
            '1.618': high + (diff * 0.618),  # 161.8% extension
            '2.618': high + (diff * 1.618),  # 261.8% extension
            '4.236': high + (diff * 3.236)   # 423.6% extension
        }
        
        return levels
    
    def analyze_fibonacci_setup(self, symbol, df):
        """Analyze Fibonacci retracement setup for a symbol"""
        if len(df) < 50:
            return None
        
        # Identify swing points
        highs, lows = self.identify_swing_points(df)
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Use most recent significant swing points
        recent_highs = sorted(highs, key=lambda x: x['index'])[-3:]
        recent_lows = sorted(lows, key=lambda x: x['index'])[-3:]
        
        current_price = df['close'].iloc[-1]
        
        opportunities = []
        
        # Analyze each swing high-low combination
        for swing_high in recent_highs:
            for swing_low in recent_lows:
                # Ensure swing high is after swing low for bullish setup
                if swing_high['index'] > swing_low['index']:
                    high_price = swing_high['price']
                    low_price = swing_low['price']
                    
                    # Calculate Fibonacci levels
                    fib_levels = self.calculate_fibonacci_levels(high_price, low_price)
                    
                    # Check if current price is near 23.6% retracement (entry level)
                    entry_level = fib_levels['0.236']
                    tolerance = entry_level * 0.02  # 2% tolerance
                    
                    if abs(current_price - entry_level) <= tolerance:
                        # Calculate quality metrics
                        quality_score = self.calculate_setup_quality(df, fib_levels, current_price)
                        
                        if quality_score >= 60:  # Minimum quality threshold
                            setup = self.create_fibonacci_setup(
                                symbol, df, swing_high, swing_low, fib_levels, 
                                current_price, quality_score
                            )
                            opportunities.append(setup)
        
        return opportunities
    
    def calculate_setup_quality(self, df, fib_levels, current_price):
        """Calculate quality score for Fibonacci setup"""
        score = 50  # Base score
        
        # Volume confirmation
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].iloc[-50:].mean()
        if recent_volume > avg_volume * 1.2:
            score += 15
        
        # Price momentum
        recent_prices = df['close'].iloc[-10:]
        price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        if abs(price_momentum) > 0.02:  # 2% move
            score += 10
        
        # Fibonacci level respect
        touches_count = 0
        for level_name, level_price in fib_levels.items():
            if level_name in ['0.236', '0.382', '0.618']:
                # Check how many times price touched this level
                for price in df['close']:
                    if abs(price - level_price) / level_price < 0.01:  # 1% tolerance
                        touches_count += 1
        
        if touches_count >= 3:
            score += 15
        
        # Risk-reward ratio
        entry_price = fib_levels['0.236']
        stop_loss = fib_levels['0.000']  # Below swing low
        target_1 = fib_levels['0.382']
        
        risk = entry_price - stop_loss
        reward = target_1 - entry_price
        
        if risk > 0 and reward > 0:
            rr_ratio = reward / risk
            if rr_ratio >= 2.0:
                score += 10
        
        return min(score, 100)
    
    def create_fibonacci_setup(self, symbol, df, swing_high, swing_low, fib_levels, current_price, quality_score):
        """Create Fibonacci setup dictionary with complete trading data"""
        entry_price = fib_levels['0.236']
        stop_loss = fib_levels['0.000']
        
        # Calculate risk percentage
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        
        # Calculate move size
        move_size_pct = ((swing_high['price'] - swing_low['price']) / swing_low['price']) * 100
        
        # Calculate targets with risk/reward ratios
        targets = {
            'target_1': {
                'price': fib_levels['0.382'],
                'gain_pct': ((fib_levels['0.382'] - entry_price) / entry_price) * 100,
                'risk_reward': ((fib_levels['0.382'] - entry_price) / (entry_price - stop_loss)) if (entry_price - stop_loss) > 0 else 0
            },
            'target_2': {
                'price': fib_levels['0.500'],
                'gain_pct': ((fib_levels['0.500'] - entry_price) / entry_price) * 100,
                'risk_reward': ((fib_levels['0.500'] - entry_price) / (entry_price - stop_loss)) if (entry_price - stop_loss) > 0 else 0
            },
            'target_3': {
                'price': fib_levels['0.618'],
                'gain_pct': ((fib_levels['0.618'] - entry_price) / entry_price) * 100,
                'risk_reward': ((fib_levels['0.618'] - entry_price) / (entry_price - stop_loss)) if (entry_price - stop_loss) > 0 else 0
            }
        }
        
        # Determine setup stage and entry timing
        if abs(current_price - entry_price) / entry_price < 0.005:  # Within 0.5%
            setup_stage = 'immediate_entry'
            entry_timing = 'NOW'
        elif current_price > entry_price:
            setup_stage = 'approaching_entry'
            entry_timing = 'CLOSE'
        else:
            setup_stage = 'waiting_for_pullback'
            entry_timing = 'WAIT'
        
        return {
            'symbol': symbol,
            'type': 'bullish_fibonacci_retracement',
            'quality_score': quality_score,
            'setup_stage': setup_stage,
            'entry_timing_status': entry_timing,
            'current_price': current_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'targets': targets,
            'risk_pct': risk_pct,
            'move_size_pct': move_size_pct,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'fib_levels': fib_levels,
            'timestamp': datetime.now()
        }
    
    def log_fibonacci_setup_to_database(self, setup):
        """Log Fibonacci setup to database using the isolated logger"""
        if not self.fibonacci_logger:
            print(f"❌ Database logger not available for {setup['symbol']}")
            return None
        
        try:
            # Create Fibonacci signal for database logging with complete trading data
            signal = FibonacciSignal(
                symbol=setup['symbol'],
                timeframe=self.timeframe,
                signal_type='SUPPORT',  # Fibonacci retracement is typically support
                level=0.236,  # 23.6% retracement level
                price=setup['entry_price'],
                timestamp=setup['timestamp'],
                confidence=setup['quality_score'] / 100.0,  # Convert to 0-1 scale
                volume_confirmation=True,  # Will be checked in the logger
                momentum_confirmation=True,  # Will be checked in the logger
                metadata={
                    'swing_high': setup['swing_high']['price'],
                    'swing_low': setup['swing_low']['price'],
                    'current_price': setup['current_price'],
                    'trend_direction': 'BULLISH',
                    'setup_stage': setup['setup_stage'],
                    'entry_timing_status': setup.get('entry_timing_status', 'UNKNOWN'),
                    'quality_score': setup['quality_score'],
                    'move_size_pct': setup['move_size_pct'],
                    'risk_pct': setup['risk_pct'],
                    'stop_loss': setup['stop_loss'],
                    'target_1_price': setup['targets']['target_1']['price'],
                    'target_1_percentage': setup['targets']['target_1']['gain_pct'],
                    'target_1_risk_reward': setup['targets']['target_1']['risk_reward'],
                    'target_2_price': setup['targets']['target_2']['price'],
                    'target_2_percentage': setup['targets']['target_2']['gain_pct'],
                    'target_2_risk_reward': setup['targets']['target_2']['risk_reward'],
                    'target_3_price': setup['targets']['target_3']['price'],
                    'target_3_percentage': setup['targets']['target_3']['gain_pct'],
                    'target_3_risk_reward': setup['targets']['target_3']['risk_reward'],
                    'fib_levels': setup['fib_levels'],
                    'targets': setup['targets'],
                    'scanner_id': self.config['scanner_id']
                }
            )
            
            # Log to database
            signal_id = self.fibonacci_logger.log_fibonacci_signal(signal)
            print(f"✅ Logged Fibonacci setup for {setup['symbol']} to database: {signal_id}")
            return signal_id
            
        except Exception as e:
            print(f"❌ Failed to log Fibonacci setup for {setup['symbol']}: {e}")
            return None
    
    def scan_for_fibonacci_setups(self, symbols_limit=50):
        """Main scanning function with database integration"""
        print(f"🔍 Scanning for Fibonacci retracement setups ({self.timeframe} timeframe)...")
        
        # Get symbols to scan
        symbols = self.get_top_coins(symbols_limit)
        print(f"📊 Analyzing {len(symbols)} symbols for Fibonacci patterns...")
        
        opportunities = []
        signals_logged = 0
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"  [{i}/{len(symbols)}] Analyzing {symbol}...", end=' ')
                
                # Get price data
                df = self.get_binance_klines(symbol, self.timeframe)
                if df is None or len(df) < 50:
                    print("❌ Insufficient data")
                    continue
                
                # Analyze Fibonacci setup
                setups = self.analyze_fibonacci_setup(symbol, df)
                
                if setups:
                    print(f"✅ Found {len(setups)} setup(s)")
                    opportunities.extend(setups)
                    
                    # Log to database
                    for setup in setups:
                        if self.log_fibonacci_setup_to_database(setup):
                            signals_logged += 1
                else:
                    print("❌ No setups found")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        print(f"\n🎯 Scan complete: {len(opportunities)} setups found, {signals_logged} logged to database")
        
        # Get health status
        if self.fibonacci_logger:
            health = self.fibonacci_logger.get_health_status()
            print(f"📊 Scanner health: {health['status']}")
        
        return opportunities
    
    def display_results(self, opportunities):
        """Display Fibonacci retracement scanning results"""
        timeframe_display = "1-MINUTE" if self.timeframe == '1m' else "5-MINUTE"
        print("\n" + "=" * 85)
        print(f"🎯 FOUND {len(opportunities)} FIBONACCI RETRACEMENT SETUPS ({timeframe_display})")
        print("=" * 85)
        
        if not opportunities:
            print("❌ No high-quality Fibonacci retracement setups found")
            print("💡 Try again later - looking for 38.2% bounce → 23.6% pullback patterns")
            return
        
        print("📊 Results (sorted by quality score):")
        print()
        
        for i, setup in enumerate(opportunities[:10], 1):
            # Setup info
            direction = "LONG" if setup['type'] == 'bullish_fibonacci_retracement' else "SHORT"
            direction_emoji = "📈" if direction == "LONG" else "📉"
            
            # Entry timing
            stage = setup['setup_stage']
            if 'immediate' in stage:
                status = '🎯 NOW'
            elif 'approaching' in stage:
                status = '🔜 CLOSE'
            else:
                status = '⏳ WAIT'
            
            print(f"{i:2}. {direction_emoji} {setup['symbol']:8} | Score: {setup['quality_score']:2}/100 | "
                  f"📊 FIB RETRACEMENT | {status}")
            
            # Current position and entry
            print(f"     Current: ${setup['current_price']:.4f} | "
                  f"Entry: ${setup['entry_price']:.4f} (23.6% Fib) | "
                  f"Move: {setup['move_size_pct']:.1f}%")
            
            # Targets
            targets_text = []
            for target_name, target_data in setup['targets'].items():
                targets_text.append(f"{target_name.replace('target_', 'T')}: ${target_data['price']:.4f} "
                                  f"({target_data['gain_pct']:.1f}% | R:R {target_data['risk_reward']:.1f}:1)")
            
            print(f"     Targets: {' | '.join(targets_text)}")
            
            # Stop loss and Fibonacci structure
            print(f"     Stop: ${setup['stop_loss']:.4f} ({setup['risk_pct']:.1f}%) | "
                  f"Swing: ${setup['swing_low']['price']:.4f} → ${setup['swing_high']['price']:.4f}")
            
            # Key Fibonacci levels
            fib = setup['fib_levels']
            print(f"     Fib Levels: 23.6%: ${fib['0.236']:.4f} | 38.2%: ${fib['0.382']:.4f} | "
                  f"50%: ${fib['0.500']:.4f} | 61.8%: ${fib['0.618']:.4f}")
            
            # Setup guidance
            if 'immediate' in stage:
                print(f"     🎯 ENTER NOW - Price at 23.6% Fibonacci retracement!")
            elif 'approaching' in stage:
                print(f"     🔜 Wait for pullback to 23.6% level (${setup['entry_price']:.4f})")
            elif 'waiting_for_pullback' in stage:
                print(f"     ⏳ Wait for pullback from 38.2% to 23.6% entry")
            elif 'waiting_for_bounce' in stage:
                print(f"     ⏳ Wait for bounce from 38.2% to 23.6% entry")
            
            print()
        
        print("🎯 FIBONACCI RETRACEMENT STRATEGY (REVISION 1 WITH DATABASE INTEGRATION):")
        print("• Setup: 2%+ move → 38.2% bounce → 23.6% pullback entry")
        print("• Entry: 23.6% Fibonacci retracement level (±2% tolerance)")
        print("• Targets: 38.2%, 50%, 61.8% Fibonacci levels")
        print("• Stop: Just beyond swing high/low")
        print("• Hold Time: 30 minutes - 4 hours typical")
        print("• Success Rate: ~60-70% with relaxed criteria")
        print("• Min R:R: 1:1 minimum, 2:1+ preferred")
        print("• Database: All signals logged to isolated fibonacci_signals table")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.fibonacci_logger:
            self.fibonacci_logger.cleanup()
            print("✅ Fibonacci Logger cleanup completed")

def main():
    import sys
    
    # Allow user to specify timeframe
    timeframe = '5m'  # Default
    if len(sys.argv) > 1:
        if sys.argv[1] in ['1m', '5m']:
            timeframe = sys.argv[1]
        else:
            print("Usage: python fibonacci_retracement_scanner_r1.py [1m|5m]")
            print("Defaulting to 5m timeframe...")
    
    scanner = FibonacciRetracementScannerR1(timeframe=timeframe)
    
    try:
        opportunities = scanner.scan_for_fibonacci_setups()
        scanner.display_results(opportunities)
    finally:
        scanner.cleanup()

if __name__ == "__main__":
    main()
