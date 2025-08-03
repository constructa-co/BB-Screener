#!/usr/bin/env python3
"""
ICT FVG Scanner 4H - Production Version (Fixed)
Based on R21 parameters: 84-90% win rate with 3.5-4.5% avg return per trade

Key Features:
- Fair Value Gap detection (proven 80%+ win rate)
- Real-time scanning of top 500 crypto pairs
- Smart Fibonacci targets with safety constraints
- Partial profit management (50% T1, 30% T2, 20% T3)
- Quality scoring based on R21 backtest results
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
import logging
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_fvg_scanner_4h.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ICTFVGScanner:
    """Production FVG Scanner based on R21 proven parameters"""
    
    def __init__(self, exchange_id='binance', test_mode=False):
        """Initialize scanner with R21 winning configuration"""
        # R21 Proven Parameters
        self.config = {
            # FVG Detection (from R21)
            'min_fvg_size': 0.007,          # 0.7% minimum gap (tightened from 0.6%)
            'max_fvg_age': 30,              # Maximum age in bars
            'min_quality_score': 70,         # Minimum quality threshold
            'max_distance_to_fvg': 0.10,    # Max 10% distance from current price
            
            # Targets (R21 optimized)
            'targets': {
                'T1': 0.05,                 # 5% (84-90% hit rate)
                'T2': 0.07,                 # 7% (for partial profits)
                'T3': 0.10                  # 10% (runners)
            },
            
            # Position Management
            'position_sizing': {
                'T1_exit': 0.5,             # Exit 50% at T1
                'T2_exit': 0.3,             # Exit 30% at T2
                'T3_exit': 0.2              # Exit final 20% at T3
            },
            
            # Risk Management
            'stop_loss': 0.02,              # 2% stop loss
            'min_risk_reward': 0.3,         # Very low for FVGs (proven to work)
            'max_risk_per_trade': 0.01,     # 1% account risk
            
            # Market Filters
            'min_volume_24h': 1000000,      # $1M daily volume
            'max_spread_pct': 0.002,        # 0.2% max spread
            
            # Quality Scoring
            'volume_surge_threshold': 1.5,   # 50% above average
            'category_weights': {
                'Layer 2/Scaling': 1.3,
                'Meme/Community': 1.2,
                'Major Cryptos': 1.1,
                'Infrastructure': 1.0,
                'Others': 1.0,
                'Altcoins': 1.0,
                'Layer 1s': 0.9,
                'DeFi': 0.9,
                'Gaming/NFTs': 0.8
            }
        }
        
        # Exchange setup
        self.exchange_id = exchange_id
        self.test_mode = test_mode
        self.exchange = None
        self.setup_exchange()
        
        # Tracking
        self.last_scan_time = {}
        self.active_setups = {}
        self.performance_stats = {
            'total_scans': 0,
            'setups_found': 0,
            'alerts_sent': 0
        }
        
    def setup_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            logger.info(f"Connected to {self.exchange_id}")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
            
    def get_top_symbols(self, limit=500) -> List[str]:
        """Get top symbols by 24h volume"""
        try:
            markets = self.exchange.load_markets()
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs with sufficient volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > self.config['min_volume_24h']:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': ticker.get('quoteVolume', 0)
                    })
            
            # Sort by volume and take top N
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            symbols = [pair['symbol'] for pair in usdt_pairs[:limit]]
            
            logger.info(f"Scanning {len(symbols)} symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
            
    def fetch_candles(self, symbol: str, timeframe='4h', limit=200) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add technical indicators
            df = self.add_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
            
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for quality scoring"""
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Volume metrics
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MFI (Money Flow Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        mfi_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
        
    def detect_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Detect Fair Value Gaps using R21 logic"""
        fvgs = []
        
        # Need at least 3 candles
        if len(df) < 3:
            return fvgs
            
        current_price = df.iloc[-1]['close']
        
        # Iterate through candles looking for gaps
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev_prev = df.iloc[i-2]
            
            # Bullish FVG: gap up between prev_prev.high and current.low
            bullish_gap = current['low'] - prev_prev['high']
            bullish_gap_pct = bullish_gap / prev_prev['high']
            
            if bullish_gap_pct >= self.config['min_fvg_size']:
                # Check if gap is unfilled
                remaining_bars = df.iloc[i+1:] if i < len(df)-1 else pd.DataFrame()
                
                if len(remaining_bars) == 0 or remaining_bars['low'].min() > prev_prev['high']:
                    fvg_age = len(df) - i - 1
                    
                    if fvg_age <= self.config['max_fvg_age']:
                        # Calculate distance from current price to FVG
                        gap_midpoint = (prev_prev['high'] + current['low']) / 2
                        distance_to_gap = abs(current_price - gap_midpoint) / current_price
                        
                        # Only include FVGs within reasonable distance (10% default)
                        if distance_to_gap <= self.config['max_distance_to_fvg']:
                            # For bullish FVG, price should be below or near the gap
                            if current_price <= gap_midpoint * 1.02:  # Allow 2% above midpoint
                                # Check volume surge on the gap-creating candle
                                volume_surge = False
                                if i < len(df) and 'volume_ratio' in df.columns:
                                    # Check volume on all three candles that form the FVG
                                    vol_surge_current = df.iloc[i]['volume_ratio'] > self.config['volume_surge_threshold']
                                    vol_surge_prev = df.iloc[i-1]['volume_ratio'] > self.config['volume_surge_threshold']
                                    vol_surge_prev_prev = df.iloc[i-2]['volume_ratio'] > self.config['volume_surge_threshold']
                                    # Volume surge on any of the three candles counts
                                    volume_surge = vol_surge_current or vol_surge_prev or vol_surge_prev_prev
                                
                                fvg = {
                                    'type': 'bullish',
                                    'index': i,
                                    'timestamp': df.index[i],
                                    'gap_low': prev_prev['high'],
                                    'gap_high': current['low'],
                                    'gap_size': bullish_gap,
                                    'gap_size_pct': bullish_gap_pct,
                                    'age': fvg_age,
                                    'current_price': current_price,
                                    'distance_to_gap': distance_to_gap,
                                    'volume_surge': volume_surge
                                }
                                
                                # Calculate quality score
                                fvg['quality_score'] = self.calculate_fvg_quality(fvg, df.iloc[i])
                                
                                if fvg['quality_score'] >= self.config['min_quality_score']:
                                    fvgs.append(fvg)
            
            # Bearish FVG: gap down between prev_prev.low and current.high
            bearish_gap = prev_prev['low'] - current['high']
            bearish_gap_pct = bearish_gap / current['high']
            
            if bearish_gap_pct >= self.config['min_fvg_size']:
                # Check if gap is unfilled
                remaining_bars = df.iloc[i+1:] if i < len(df)-1 else pd.DataFrame()
                
                if len(remaining_bars) == 0 or remaining_bars['high'].max() < prev_prev['low']:
                    fvg_age = len(df) - i - 1
                    
                    if fvg_age <= self.config['max_fvg_age']:
                        # Calculate distance from current price to FVG
                        gap_midpoint = (prev_prev['low'] + current['high']) / 2
                        distance_to_gap = abs(current_price - gap_midpoint) / current_price
                        
                        # Only include FVGs within reasonable distance (10% default)
                        if distance_to_gap <= self.config['max_distance_to_fvg']:
                            # For bearish FVG, price should be above or near the gap
                            if current_price >= gap_midpoint * 0.98:  # Allow 2% below midpoint
                                # Check volume surge on the gap-creating candle
                                volume_surge = False
                                if i < len(df) and 'volume_ratio' in df.columns:
                                    # Check volume on all three candles that form the FVG
                                    vol_surge_current = df.iloc[i]['volume_ratio'] > self.config['volume_surge_threshold']
                                    vol_surge_prev = df.iloc[i-1]['volume_ratio'] > self.config['volume_surge_threshold']
                                    vol_surge_prev_prev = df.iloc[i-2]['volume_ratio'] > self.config['volume_surge_threshold']
                                    # Volume surge on any of the three candles counts
                                    volume_surge = vol_surge_current or vol_surge_prev or vol_surge_prev_prev
                                
                                fvg = {
                                    'type': 'bearish',
                                    'index': i,
                                    'timestamp': df.index[i],
                                    'gap_high': prev_prev['low'],
                                    'gap_low': current['high'],
                                    'gap_size': bearish_gap,
                                    'gap_size_pct': bearish_gap_pct,
                                    'age': fvg_age,
                                    'current_price': current_price,
                                    'distance_to_gap': distance_to_gap,
                                    'volume_surge': volume_surge
                                }
                                
                                # Calculate quality score
                                fvg['quality_score'] = self.calculate_fvg_quality(fvg, df.iloc[i])
                                
                                if fvg['quality_score'] >= self.config['min_quality_score']:
                                    fvgs.append(fvg)
                            
        # Sort by quality score and distance (closest and highest quality first)
        fvgs.sort(key=lambda x: (x['quality_score'], -x['distance_to_gap']), reverse=True)
        
        # Return only the best FVG per symbol (if any)
        return fvgs[:1] if fvgs else []
        
    def calculate_fvg_quality(self, fvg: Dict, candle_data: pd.Series) -> float:
        """Calculate quality score based on R21 parameters"""
        score = 50.0  # Base score
        
        # Gap size bonus (larger gaps score higher)
        if fvg['gap_size_pct'] > 0.01:  # > 1%
            score += 10
        if fvg['gap_size_pct'] > 0.015:  # > 1.5%
            score += 10
            
        # Freshness bonus (newer gaps score higher)
        if fvg['age'] < 5:
            score += 15
        elif fvg['age'] < 10:
            score += 10
        elif fvg['age'] < 20:
            score += 5
            
        # Volume confirmation
        if fvg['volume_surge']:
            score += 15
            
        # Technical indicator confluence
        if fvg['type'] == 'bullish':
            if candle_data['rsi'] < 40:
                score += 10
            if candle_data['mfi'] < 30:
                score += 10
        else:  # bearish
            if candle_data['rsi'] > 60:
                score += 10
            if candle_data['mfi'] > 70:
                score += 10
                
        return min(score, 100)  # Cap at 100
        
    def find_swing_high(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find the most significant resistance ZONE, not just a swing point"""
        if len(df) < lookback:
            lookback = len(df)
        
        recent_df = df.tail(lookback)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        # Method 1: Find all local highs first
        local_highs = []
        for i in range(2, len(highs)-2):
            # Basic swing high structure
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                local_highs.append({'index': i, 'price': highs[i]})
        
        if not local_highs:
            return recent_df['high'].max()
        
        # Method 2: Create resistance zones by clustering nearby highs
        zones = []
        zone_threshold = 0.03  # 3% to be in same zone
        
        for high in local_highs:
            added_to_zone = False
            
            # Check if this high belongs to an existing zone
            for zone in zones:
                zone_center = sum(h['price'] for h in zone['highs']) / len(zone['highs'])
                if abs(high['price'] - zone_center) / zone_center < zone_threshold:
                    zone['highs'].append(high)
                    added_to_zone = True
                    break
            
            # Create new zone if not added
            if not added_to_zone:
                zones.append({
                    'highs': [high],
                    'first_index': high['index'],
                    'last_index': high['index']
                })
        
        # Update zone properties
        for zone in zones:
            zone['max_price'] = max(h['price'] for h in zone['highs'])
            zone['min_price'] = min(h['price'] for h in zone['highs'])
            zone['avg_price'] = sum(h['price'] for h in zone['highs']) / len(zone['highs'])
            zone['touches'] = len(zone['highs'])
            zone['first_index'] = min(h['index'] for h in zone['highs'])
            zone['last_index'] = max(h['index'] for h in zone['highs'])
            zone['span'] = zone['last_index'] - zone['first_index']
            
            # Count rejections from this zone
            rejections = 0
            for i in range(zone['first_index'], min(zone['last_index'] + 5, len(highs))):
                if i < len(highs) - 2:
                    # Check if price was rejected (fell 3% or more)
                    if highs[i] > zone['min_price'] * 0.97:
                        next_low = min(lows[i:min(i+3, len(lows))])
                        if next_low < highs[i] * 0.97:
                            rejections += 1
            zone['rejections'] = rejections
        
        # Score each zone
        best_zone = None
        best_score = -999
        
        # Debug logging
        symbol_name = recent_df.index.name if hasattr(recent_df.index, 'name') else "Unknown"
        
        for zone in zones:
            # Skip very recent zones (last 2 bars)
            if zone['last_index'] > len(highs) - 3:
                continue
                
            # Calculate score
            score = 0
            
            # Multiple touches are very important
            score += zone['touches'] * 10
            
            # Rejections show it's real resistance
            score += zone['rejections'] * 8
            
            # Zones that span time are stronger
            if zone['span'] > 3:
                score += min(zone['span'], 15) * 2
            
            # Prefer zones in the middle of the range (not extremes)
            price_position = (zone['avg_price'] - min(lows)) / (max(highs) - min(lows))
            if 0.4 < price_position < 0.9:  # Not at the very top
                score += 20
            
            # Penalty for being the absolute high
            if zone['max_price'] == max(highs):
                score -= 15
                
            zone['score'] = score
            
            # Debug print for SPK
            if 'SPK' in str(symbol_name) or zone['avg_price'] > 0.10 and zone['avg_price'] < 0.12:
                logger.debug(f"Zone at ${zone['avg_price']:.4f}: touches={zone['touches']}, "
                            f"rejections={zone['rejections']}, span={zone['span']}, score={score}")
            
            if score > best_score:
                best_score = score
                best_zone = zone
        
        # If we found a good zone, return its highest point
        if best_zone and best_zone['score'] > 10:
            return best_zone['max_price']
        
        # Fallback: Find the price level with most concentration
        # This catches consolidation zones that might not have perfect swing highs
        price_histogram = {}
        bin_size = 0.002  # 0.2% bins
        
        for high in highs:
            bin_price = round(high / bin_size) * bin_size
            price_histogram[bin_price] = price_histogram.get(bin_price, 0) + 1
        
        # Find the bin with most touches (excluding recent prices)
        best_bin = None
        max_count = 0
        current_price = highs[-1]
        
        for bin_price, count in price_histogram.items():
            # Skip bins too close to current price
            if abs(bin_price - current_price) / current_price < 0.05:
                continue
            # Skip the absolute high
            if bin_price >= max(highs) * 0.98:
                continue
                
            if count > max_count:
                max_count = count
                best_bin = bin_price
        
        if best_bin and max_count >= 3:  # At least 3 touches
            # Find the actual high near this bin
            for i, high in enumerate(highs):
                if abs(high - best_bin) / best_bin < 0.01:
                    return high
        
        # Final fallback
        return recent_df['high'].max()
        
    def find_swing_low(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find the most significant support ZONE, not just a swing point"""
        if len(df) < lookback:
            lookback = len(df)
            
        recent_df = df.tail(lookback)
        lows = recent_df['low'].values
        highs = recent_df['high'].values
        
        # Method 1: Find all local lows first
        local_lows = []
        for i in range(2, len(lows)-2):
            # Basic swing low structure
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                local_lows.append({'index': i, 'price': lows[i]})
        
        if not local_lows:
            return recent_df['low'].min()
        
        # Method 2: Create support zones by clustering nearby lows
        zones = []
        zone_threshold = 0.03  # 3% to be in same zone
        
        for low in local_lows:
            added_to_zone = False
            
            # Check if this low belongs to an existing zone
            for zone in zones:
                zone_center = sum(l['price'] for l in zone['lows']) / len(zone['lows'])
                if abs(low['price'] - zone_center) / zone_center < zone_threshold:
                    zone['lows'].append(low)
                    added_to_zone = True
                    break
            
            # Create new zone if not added
            if not added_to_zone:
                zones.append({
                    'lows': [low],
                    'first_index': low['index'],
                    'last_index': low['index']
                })
        
        # Update zone properties
        for zone in zones:
            zone['min_price'] = min(l['price'] for l in zone['lows'])
            zone['max_price'] = max(l['price'] for l in zone['lows'])
            zone['avg_price'] = sum(l['price'] for l in zone['lows']) / len(zone['lows'])
            zone['touches'] = len(zone['lows'])
            zone['first_index'] = min(l['index'] for l in zone['lows'])
            zone['last_index'] = max(l['index'] for l in zone['lows'])
            zone['span'] = zone['last_index'] - zone['first_index']
            
            # Count bounces from this zone
            bounces = 0
            for i in range(zone['first_index'], min(zone['last_index'] + 5, len(lows))):
                if i < len(lows) - 2:
                    # Check if price bounced (rose 3% or more)
                    if lows[i] < zone['max_price'] * 1.03:
                        next_high = max(highs[i:min(i+3, len(highs))])
                        if next_high > lows[i] * 1.03:
                            bounces += 1
            zone['bounces'] = bounces
        
        # Score each zone
        best_zone = None
        best_score = -999
        
        for zone in zones:
            # Skip very recent zones (last 2 bars)
            if zone['last_index'] > len(lows) - 3:
                continue
                
            # Calculate score
            score = 0
            
            # Multiple touches are very important
            score += zone['touches'] * 10
            
            # Bounces show it's real support
            score += zone['bounces'] * 8
            
            # Zones that span time are stronger
            if zone['span'] > 3:
                score += min(zone['span'], 15) * 2
            
            # Prefer zones in the middle of the range (not extremes)
            price_position = (zone['avg_price'] - min(lows)) / (max(highs) - min(lows))
            if 0.1 < price_position < 0.6:  # Not at the very bottom
                score += 20
            
            # Penalty for being the absolute low
            if zone['min_price'] == min(lows):
                score -= 15
                
            zone['score'] = score
            
            if score > best_score:
                best_score = score
                best_zone = zone
        
        # If we found a good zone, return its lowest point
        if best_zone and best_zone['score'] > 10:
            return best_zone['min_price']
        
        # Fallback: Find the price level with most concentration
        price_histogram = {}
        bin_size = 0.002  # 0.2% bins
        
        for low in lows:
            bin_price = round(low / bin_size) * bin_size
            price_histogram[bin_price] = price_histogram.get(bin_price, 0) + 1
        
        # Find the bin with most touches (excluding recent prices)
        best_bin = None
        max_count = 0
        current_price = lows[-1]
        
        for bin_price, count in price_histogram.items():
            # Skip bins too close to current price
            if abs(bin_price - current_price) / current_price < 0.05:
                continue
            # Skip the absolute low
            if bin_price <= min(lows) * 1.02:
                continue
                
            if count > max_count:
                max_count = count
                best_bin = bin_price
        
        if best_bin and max_count >= 3:  # At least 3 touches
            # Find the actual low near this bin
            for i, low in enumerate(lows):
                if abs(low - best_bin) / best_bin < 0.01:
                    return low
        
        # Final fallback
        return recent_df['low'].min()
        
    def calculate_smart_fibonacci_targets(self, df: pd.DataFrame, fvg: Dict, entry_price: float, stop_loss: float) -> Dict:
        """
        Calculate proper Fibonacci-based targets using market structure
        """
        # Find recent swing points
        swing_high = self.find_swing_high(df, lookback=50)
        swing_low = self.find_swing_low(df, lookback=50)
        fib_range = abs(swing_high - swing_low)
        
        # Current price for reference
        current_price = df.iloc[-1]['close']
        
        targets = {}
        fib_levels = {}
        
        if fvg['type'] == 'bullish':
            # Calculate Fibonacci retracement levels from swing
            fib_levels['236'] = swing_low + (fib_range * 0.236)
            fib_levels['382'] = swing_low + (fib_range * 0.382)
            fib_levels['500'] = swing_low + (fib_range * 0.500)
            fib_levels['618'] = swing_low + (fib_range * 0.618)
            fib_levels['786'] = swing_low + (fib_range * 0.786)
            fib_levels['1000'] = swing_high  # 100% level
            fib_levels['1272'] = swing_high + (fib_range * 0.272)  # Extension
            fib_levels['1618'] = swing_high + (fib_range * 0.618)  # Extension
            
            # Use proper Fibonacci extensions for targets
            # T1: First significant Fib level above entry
            if entry_price < fib_levels['382']:
                targets['T1'] = fib_levels['382']
            elif entry_price < fib_levels['500']:
                targets['T1'] = fib_levels['500']
            elif entry_price < fib_levels['618']:
                targets['T1'] = fib_levels['618']
            elif entry_price < fib_levels['786']:
                targets['T1'] = fib_levels['786']
            elif entry_price < fib_levels['1000']:
                targets['T1'] = fib_levels['1000']
            else:
                targets['T1'] = fib_levels['1272']  # Extension
            
            # T2: Next Fibonacci level
            if targets['T1'] == fib_levels['382']:
                targets['T2'] = fib_levels['500']
            elif targets['T1'] == fib_levels['500']:
                targets['T2'] = fib_levels['618']
            elif targets['T1'] == fib_levels['618']:
                targets['T2'] = fib_levels['786']
            elif targets['T1'] == fib_levels['786']:
                targets['T2'] = fib_levels['1000']
            elif targets['T1'] == fib_levels['1000']:
                targets['T2'] = fib_levels['1272']
            else:
                targets['T2'] = fib_levels['1618']
            
            # T3: Extended Fibonacci target
            if targets['T2'] == fib_levels['500']:
                targets['T3'] = fib_levels['786']
            elif targets['T2'] == fib_levels['618']:
                targets['T3'] = fib_levels['1000']
            elif targets['T2'] == fib_levels['786']:
                targets['T3'] = fib_levels['1272']
            elif targets['T2'] == fib_levels['1000']:
                targets['T3'] = fib_levels['1618']
            else:
                targets['T3'] = swing_high + fib_range  # 200% extension
                
        else:  # bearish
            # Calculate Fibonacci retracement levels from swing
            fib_levels['236'] = swing_high - (fib_range * 0.236)
            fib_levels['382'] = swing_high - (fib_range * 0.382)
            fib_levels['500'] = swing_high - (fib_range * 0.500)
            fib_levels['618'] = swing_high - (fib_range * 0.618)
            fib_levels['786'] = swing_high - (fib_range * 0.786)
            fib_levels['1000'] = swing_low  # 100% level
            fib_levels['1272'] = swing_low - (fib_range * 0.272)  # Extension
            fib_levels['1618'] = swing_low - (fib_range * 0.618)  # Extension
            
            # Use proper Fibonacci extensions for targets
            # T1: First significant Fib level below entry
            if entry_price > fib_levels['382']:
                targets['T1'] = fib_levels['382']
            elif entry_price > fib_levels['500']:
                targets['T1'] = fib_levels['500']
            elif entry_price > fib_levels['618']:
                targets['T1'] = fib_levels['618']
            elif entry_price > fib_levels['786']:
                targets['T1'] = fib_levels['786']
            elif entry_price > fib_levels['1000']:
                targets['T1'] = fib_levels['1000']
            else:
                targets['T1'] = fib_levels['1272']  # Extension
            
            # T2: Next Fibonacci level
            if targets['T1'] == fib_levels['382']:
                targets['T2'] = fib_levels['500']
            elif targets['T1'] == fib_levels['500']:
                targets['T2'] = fib_levels['618']
            elif targets['T1'] == fib_levels['618']:
                targets['T2'] = fib_levels['786']
            elif targets['T1'] == fib_levels['786']:
                targets['T2'] = fib_levels['1000']
            elif targets['T1'] == fib_levels['1000']:
                targets['T2'] = fib_levels['1272']
            else:
                targets['T2'] = fib_levels['1618']
            
            # T3: Extended Fibonacci target
            if targets['T2'] == fib_levels['500']:
                targets['T3'] = fib_levels['786']
            elif targets['T2'] == fib_levels['618']:
                targets['T3'] = fib_levels['1000']
            elif targets['T2'] == fib_levels['786']:
                targets['T3'] = fib_levels['1272']
            elif targets['T2'] == fib_levels['1000']:
                targets['T3'] = fib_levels['1618']
            else:
                targets['T3'] = swing_low - fib_range  # 200% extension
        
        # Calculate risk/reward with proper Fibonacci targets
        risk = abs(entry_price - stop_loss) / entry_price
        reward_t1 = abs(targets['T1'] - entry_price) / entry_price
        risk_reward = reward_t1 / risk if risk > 0 else 0
        
        # Quality scoring based on Fibonacci alignment
        fib_quality = 0
        
        # Check if entry is near a Fibonacci level (good confluence)
        for level_name, level_price in fib_levels.items():
            if abs(entry_price - level_price) / entry_price < 0.02:  # Within 2%
                fib_quality += 20
                break
                
        # Check if current price respects Fibonacci structure
        if fvg['type'] == 'bullish' and swing_low < entry_price < swing_high:
            fib_quality += 10
        elif fvg['type'] == 'bearish' and swing_low < entry_price < swing_high:
            fib_quality += 10
            
        return {
            'targets': targets,
            'risk_reward': risk_reward,
            'fib_quality': fib_quality,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'fib_levels': fib_levels
        }
        
    def get_token_category(self, symbol: str) -> str:
        """Categorize token based on R21 categories"""
        categories = {
            'Major Cryptos': ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE'],
            'Layer 1s': ['DOT', 'AVAX', 'ATOM', 'NEAR', 'ALGO', 'ICP', 'FLOW'],
            'Layer 2/Scaling': ['MATIC', 'ARB', 'OP', 'IMX', 'LRC'],
            'DeFi': ['UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'CRV', 'YFI'],
            'Meme/Community': ['SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF'],
            'Gaming/NFTs': ['MANA', 'SAND', 'AXS', 'ENJ', 'GALA', 'CHZ'],
            'Infrastructure': ['LINK', 'GRT', 'FIL', 'AR', 'OCEAN'],
            'Altcoins': ['VET', 'XLM', 'XTZ', 'EOS', 'NEO', 'QTUM']
        }
        
        for category, tokens in categories.items():
            if symbol in tokens:
                return category
                
        return 'Others'
        
    def calculate_setup_details(self, symbol: str, fvg: Dict, current_price: float) -> Dict:
        """Calculate entry, stop, targets, and position sizing"""
        # Entry at FVG midpoint (R21 proven entry)
        entry_price = (fvg['gap_high'] + fvg['gap_low']) / 2
        
        # Calculate distance from current price to entry
        distance_to_entry = (entry_price - current_price) / current_price
        
        # Determine action required
        if fvg['type'] == 'bullish':
            if current_price < entry_price * 0.99:  # Price below entry
                action_required = "Set limit buy order at entry"
            elif current_price <= entry_price * 1.01:  # Price at entry (1% tolerance)
                action_required = "Enter LONG now (at FVG midpoint)"
            else:
                action_required = "Wait for pullback to entry"
        else:  # bearish
            if current_price > entry_price * 1.01:  # Price above entry
                action_required = "Set limit sell order at entry"
            elif current_price >= entry_price * 0.99:  # Price at entry (1% tolerance)
                action_required = "Enter SHORT now (at FVG midpoint)"
            else:
                action_required = "Wait for rally to entry"
        
        # Calculate stop loss based on FVG structure
        gap_range = abs(fvg['gap_high'] - fvg['gap_low'])
        
        if fvg['type'] == 'bullish':
            # Stop loss just below the gap
            stop_loss = fvg['gap_low'] - (gap_range * 0.236)  # 23.6% below gap
        else:  # bearish
            # Stop loss just above the gap
            stop_loss = fvg['gap_high'] + (gap_range * 0.236)  # 23.6% above gap
            
        # Get dataframe for Fibonacci calculation
        # Note: In production, you'd pass the df to this method
        df = self.fetch_candles(symbol)  # Fetch fresh data for swing analysis
        
        # Calculate smart Fibonacci targets
        fib_result = self.calculate_smart_fibonacci_targets(df, fvg, entry_price, stop_loss)
        
        # Get token category for weighting
        base_symbol = symbol.split('/')[0]
        category = self.get_token_category(base_symbol)
        category_weight = self.config['category_weights'].get(category, 1.0)
        
        # Final quality with category weight and Fibonacci quality
        final_quality = (fvg['quality_score'] + fib_result['fib_quality']) * category_weight
        final_quality = min(final_quality, 100)  # Cap at 100
        
        return {
            'symbol': symbol,
            'type': fvg['type'],
            'entry_price': entry_price,
            'current_price': current_price,
            'distance_to_entry': distance_to_entry * 100,
            'action_required': action_required,
            'stop_loss': stop_loss,
            'targets': fib_result['targets'],
            'fvg_zone': {
                'high': fvg['gap_high'],
                'low': fvg['gap_low']
            },
            'swing_levels': {
                'high': fib_result['swing_high'],
                'low': fib_result['swing_low']
            },
            'risk_pct': abs(entry_price - stop_loss) / entry_price * 100,
            'reward_pct': abs(fib_result['targets']['T1'] - entry_price) / entry_price * 100,
            'risk_reward': fib_result['risk_reward'],
            'quality_score': fvg['quality_score'],
            'fib_quality': fib_result['fib_quality'],
            'final_quality': final_quality,
            'category': category,
            'gap_size_pct': fvg['gap_size_pct'] * 100,
            'distance_pct': fvg['distance_to_gap'] * 100,
            'fvg_age': fvg['age'],
            'volume_surge': fvg['volume_surge'],
            'timestamp': datetime.now()
        }
        
    def format_alert(self, setup: Dict) -> str:
        """Format setup for alerts"""
        direction = "🟢 LONG" if setup['type'] == 'bullish' else "🔴 SHORT"
        
        # Calculate target percentages
        t1_pct = abs(setup['targets']['T1'] - setup['entry_price']) / setup['entry_price'] * 100
        t2_pct = abs(setup['targets']['T2'] - setup['entry_price']) / setup['entry_price'] * 100
        t3_pct = abs(setup['targets']['T3'] - setup['entry_price']) / setup['entry_price'] * 100
        
        # Determine which Fibonacci level each target represents
        def get_fib_level(price, swing_high, swing_low, is_bullish):
            fib_range = swing_high - swing_low
            if is_bullish:
                level = (price - swing_low) / fib_range
            else:
                level = (swing_high - price) / fib_range
            
            if level < 0.3:
                return "23.6%"
            elif level < 0.4:
                return "38.2%"
            elif level < 0.55:
                return "50%"
            elif level < 0.65:
                return "61.8%"
            elif level < 0.8:
                return "78.6%"
            elif level < 1.1:
                return "100%"
            elif level < 1.35:
                return "127.2%"
            elif level < 1.7:
                return "161.8%"
            else:
                return "200%+"
        
        is_bullish = setup['type'] == 'bullish'
        t1_basis = get_fib_level(setup['targets']['T1'], setup['swing_levels']['high'], setup['swing_levels']['low'], is_bullish)
        t2_basis = get_fib_level(setup['targets']['T2'], setup['swing_levels']['high'], setup['swing_levels']['low'], is_bullish)
        t3_basis = get_fib_level(setup['targets']['T3'], setup['swing_levels']['high'], setup['swing_levels']['low'], is_bullish)
        
        alert = f"""
{direction} Signal: {setup['symbol']}
━━━━━━━━━━━━━━━━━━━━━━━━

📊 FVG ZONE:
- Gap High: ${setup['fvg_zone']['high']:.4f}
- Gap Low: ${setup['fvg_zone']['low']:.4f}
- Entry (Midpoint): ${setup['entry_price']:.4f}

📈 MARKET STRUCTURE:
- Swing High: ${setup['swing_levels']['high']:.4f}
- Swing Low: ${setup['swing_levels']['low']:.4f}
- Swing Range: {abs(setup['swing_levels']['high'] - setup['swing_levels']['low']) / setup['swing_levels']['low'] * 100:.1f}%

💹 CURRENT STATUS:
- Price Now: ${setup['current_price']:.4f}
- Distance to Entry: {setup['distance_to_entry']:.1f}%
- Action: {setup['action_required']}

🛑 RISK MANAGEMENT:
- Stop Loss: ${setup['stop_loss']:.4f} (-{setup['risk_pct']:.1f}%)
- Risk/Reward: {setup['risk_reward']:.1f}:1

🎯 SMART FIBONACCI TARGETS:
- T1: ${setup['targets']['T1']:.4f} (+{t1_pct:.1f}%) [{t1_basis}] - Exit 50%
- T2: ${setup['targets']['T2']:.4f} (+{t2_pct:.1f}%) [{t2_basis}] - Exit 30%
- T3: ${setup['targets']['T3']:.4f} (+{t3_pct:.1f}%) [{t3_basis}] - Exit 20%

📈 SETUP QUALITY:
- FVG Score: {setup['quality_score']:.0f}/100
- Fib Confluence: {setup['fib_quality']:.0f}/100
- Final Score: {setup['final_quality']:.0f}/100
- Gap Size: {setup['gap_size_pct']:.1f}%
- FVG Age: {setup['fvg_age']} bars
- Volume Surge: {'Yes ✅' if setup['volume_surge'] else 'No ❌'}
- Category: {setup['category']}

💡 Strategy: R21 FVG + Smart Fibonacci
⏰ Time: {setup['timestamp'].strftime('%Y-%m-%d %H:%M')}
"""
        return alert
        
    def scan_all_symbols(self, top_n=500, min_quality=75):
        """Main scanning loop"""
        logger.info(f"Starting scan of top {top_n} symbols...")
        self.performance_stats['total_scans'] += 1
        
        symbols = self.get_top_symbols(limit=top_n)
        high_quality_setups = []
        
        for i, symbol in enumerate(symbols):
            try:
                # Rate limiting
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(symbols)} symbols scanned")
                    
                # Skip recently scanned symbols (within 15 minutes)
                last_scan = self.last_scan_time.get(symbol, datetime.min)
                if datetime.now() - last_scan < timedelta(minutes=15):
                    continue
                    
                # Fetch and analyze
                df = self.fetch_candles(symbol)
                if df.empty or len(df) < 50:
                    continue
                    
                # Detect FVGs
                fvgs = self.detect_fvg(df)
                
                for fvg in fvgs:
                    setup = self.calculate_setup_details(
                        symbol, 
                        fvg, 
                        df.iloc[-1]['close']
                    )
                    
                    # Only alert high quality setups
                    if setup['final_quality'] >= min_quality:
                        high_quality_setups.append(setup)
                        self.performance_stats['setups_found'] += 1
                        
                        # Send alert (implement your notification method)
                        alert_text = self.format_alert(setup)
                        logger.info(alert_text)
                        self.send_alert(alert_text)
                        
                self.last_scan_time[symbol] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
                
        # Summary
        logger.info(f"Scan complete. Found {len(high_quality_setups)} high-quality setups")
        return high_quality_setups
        
    def send_alert(self, message: str):
        """Send alert via Telegram/Discord/etc"""
        # TODO: Implement your notification method
        # For now, just log it
        self.performance_stats['alerts_sent'] += 1
        
        # Example Telegram implementation:
        # bot.send_message(chat_id=CHAT_ID, text=message)
        
        # Example Discord webhook:
        # webhook.send(content=message)
        
    def run_continuous(self, scan_interval=900):  # 15 minutes default
        """Run scanner continuously"""
        logger.info(f"Starting continuous scanner (interval: {scan_interval}s)")
        
        while True:
            try:
                # Run scan
                setups = self.scan_all_symbols()
                
                # Log performance
                logger.info(f"Performance Stats: {self.performance_stats}")
                
                # Wait for next scan
                logger.info(f"Next scan in {scan_interval} seconds...")
                time.sleep(scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Scanner stopped by user")
                break
            except Exception as e:
                logger.error(f"Scanner error: {e}")
                time.sleep(60)  # Wait 1 minute on error
                
    def backtest_setup(self, setup: Dict) -> Dict:
        """Quick backtest of setup (optional)"""
        # This would check historical performance of similar setups
        # For now, return expected stats based on R21
        return {
            'expected_win_rate': 0.85,  # 85% based on R21
            'expected_return': 0.045,    # 4.5% based on R21
            'similar_setups_30d': 'N/A'  # Would need historical data
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT FVG Scanner 4H')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuous')
    parser.add_argument('--symbols', type=int, default=500, help='Number of symbols to scan')
    parser.add_argument('--quality', type=int, default=75, help='Minimum quality threshold')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = ICTFVGScanner(test_mode=args.test)
    
    if args.once:
        # Run single scan
        setups = scanner.scan_all_symbols(top_n=args.symbols, min_quality=args.quality)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Found {len(setups)} high-quality FVG setups")
        print(f"{'='*60}")
        
        for setup in setups[:10]:  # Show top 10
            print(f"\n{setup['symbol']} - {setup['type'].upper()}")
            print(f"Quality: {setup['final_quality']:.0f}/100")
            print(f"Entry: ${setup['entry_price']:.4f}")
            print(f"Risk/Reward: {setup['risk_reward']:.1f}:1")
            
    else:
        # Run continuous scanning
        scanner.run_continuous()


if __name__ == "__main__":
    main()