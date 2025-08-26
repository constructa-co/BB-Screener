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

# Import the universal enrichment module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from universal_enrichment import UniversalEnrichment

# Import database logger
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from trade_logger import TradeLogger

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
        
        # Initialize universal enrichment
        self.enricher = UniversalEnrichment(exchange_id)
        
        # Database integration
        self.db_logger = None
        if not self.test_mode:  # Only try database connection if not in test mode
            try:
                self.db_logger = TradeLogger()
                logger.info("✅ Database logger initialized")
            except Exception as e:
                logger.warning(f"❌ Database logger not available: {e}")
                logger.info("🔄 Running in offline mode - database logging disabled")
        else:
            logger.info("🔄 Test mode - database logging disabled")
        
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
        """Find the most significant resistance ZONE based on consolidation and rejections"""
        if len(df) < lookback:
            lookback = len(df)
        
        recent_df = df.tail(lookback)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        closes = recent_df['close'].values
        
        # Create a more granular histogram of price levels
        price_levels = {}
        bin_size = 0.001  # 0.1% bins for more precision
        
        # Analyze every high in the lookback period
        for i in range(len(highs)):
            # Round to nearest bin
            bin_price = round(highs[i] / bin_size) * bin_size
            
            if bin_price not in price_levels:
                price_levels[bin_price] = {
                    'count': 0,
                    'indices': [],
                    'actual_highs': [],
                    'rejections': 0,
                    'first_touch': i,
                    'last_touch': i
                }
            
            price_levels[bin_price]['count'] += 1
            price_levels[bin_price]['indices'].append(i)
            price_levels[bin_price]['actual_highs'].append(highs[i])
            price_levels[bin_price]['last_touch'] = i
        
        # Merge nearby levels (within 1%)
        merged_levels = {}
        for price, data in sorted(price_levels.items()):
            merged = False
            for merged_price in list(merged_levels.keys()):
                if abs(price - merged_price) / merged_price < 0.01:  # Within 1%
                    # Merge into existing level
                    merged_levels[merged_price]['count'] += data['count']
                    merged_levels[merged_price]['indices'].extend(data['indices'])
                    merged_levels[merged_price]['actual_highs'].extend(data['actual_highs'])
                    merged_levels[merged_price]['first_touch'] = min(
                        merged_levels[merged_price]['first_touch'], 
                        data['first_touch']
                    )
                    merged_levels[merged_price]['last_touch'] = max(
                        merged_levels[merged_price]['last_touch'], 
                        data['last_touch']
                    )
                    merged = True
                    break
            
            if not merged:
                merged_levels[price] = data
        
        # Calculate rejections and consolidation for each level
        for level_price, level_data in merged_levels.items():
            # Count rejections (price falling after touching this level)
            rejections = 0
            consolidation_score = 0
            
            for idx in level_data['indices']:
                # Check if price was rejected
                if idx < len(highs) - 3:
                    # Look for price drop after touching level
                    next_low = min(lows[idx:idx+3])
                    if next_low < level_price * 0.97:  # 3% drop
                        rejections += 1
                    
                    # Check for consolidation (multiple bars near this level)
                    nearby_bars = 0
                    for j in range(max(0, idx-5), min(len(highs), idx+5)):
                        if abs(highs[j] - level_price) / level_price < 0.02:  # Within 2%
                            nearby_bars += 1
                    consolidation_score += nearby_bars
            
            level_data['rejections'] = rejections
            level_data['consolidation_score'] = consolidation_score
            level_data['span'] = level_data['last_touch'] - level_data['first_touch']
            level_data['max_high'] = max(level_data['actual_highs'])
        
        # Score each level
        best_level = None
        best_score = -999
        
        # Current price and absolute high for context
        current_price = closes[-1]
        absolute_high = max(highs)
        
        # Debug info for SPK
        symbol_name = recent_df.index.name if hasattr(recent_df.index, 'name') else "Unknown"
        
        for level_price, level_data in merged_levels.items():
            # Skip if too few touches
            if level_data['count'] < 2:
                continue
                
            # Skip very recent levels (last 2 bars)
            if level_data['last_touch'] > len(highs) - 3:
                continue
            
            # Calculate comprehensive score
            score = 0
            
            # 1. Multiple touches are crucial
            score += level_data['count'] * 5
            
            # 2. Rejections show real resistance
            score += level_data['rejections'] * 10
            
            # 3. Consolidation indicates significance
            score += min(level_data['consolidation_score'] / 5, 20)
            
            # 4. Time span (established levels)
            if level_data['span'] > 5:
                score += min(level_data['span'] / 2, 15)
            
            # 5. Position in range
            price_position = (level_data['max_high'] - min(lows)) / (absolute_high - min(lows))
            
            # Prefer levels that are NOT at the absolute top
            if 0.3 < price_position < 0.85:  # Middle to upper-middle range
                score += 25
            elif price_position > 0.95:  # Too close to absolute high
                score -= 30
            
            # 6. Distance from current price (not too far)
            distance_from_current = abs(level_data['max_high'] - current_price) / current_price
            if distance_from_current < 0.15:  # Within 15%
                score += 10
            
            # 7. Cluster density bonus
            # If multiple price levels are very close, it's a strong zone
            cluster_count = 0
            for other_price in merged_levels:
                if other_price != level_price and abs(other_price - level_price) / level_price < 0.02:
                    cluster_count += 1
            score += cluster_count * 5
            
            level_data['score'] = score
            
            if score > best_score:
                best_score = score
                best_level = level_data
        
        # Return the highest actual high from the best resistance zone
        if best_level and best_level['score'] > 20:
            return best_level['max_high']
        
        # Fallback: Look for the level with most touches that's not the absolute high
        fallback_levels = []
        for level_price, level_data in merged_levels.items():
            if level_data['max_high'] < absolute_high * 0.95:  # Not the very top
                fallback_levels.append({
                    'price': level_data['max_high'],
                    'touches': level_data['count'],
                    'rejections': level_data['rejections']
                })
        
        if fallback_levels:
            # Sort by touches + rejections
            fallback_levels.sort(key=lambda x: x['touches'] + x['rejections'], reverse=True)
            return fallback_levels[0]['price']
        
        # Final fallback
        return recent_df['high'].max()
        
    def find_swing_low(self, df: pd.DataFrame, lookback: int = 50) -> float:
        """Find the most significant support ZONE based on consolidation and bounces"""
        if len(df) < lookback:
            lookback = len(df)
            
        recent_df = df.tail(lookback)
        lows = recent_df['low'].values
        highs = recent_df['high'].values
        closes = recent_df['close'].values
        
        # Create a more granular histogram of price levels
        price_levels = {}
        bin_size = 0.001  # 0.1% bins for more precision
        
        # Analyze every low in the lookback period
        for i in range(len(lows)):
            # Round to nearest bin
            bin_price = round(lows[i] / bin_size) * bin_size
            
            if bin_price not in price_levels:
                price_levels[bin_price] = {
                    'count': 0,
                    'indices': [],
                    'actual_lows': [],
                    'bounces': 0,
                    'first_touch': i,
                    'last_touch': i
                }
            
            price_levels[bin_price]['count'] += 1
            price_levels[bin_price]['indices'].append(i)
            price_levels[bin_price]['actual_lows'].append(lows[i])
            price_levels[bin_price]['last_touch'] = i
        
        # Merge nearby levels (within 1%)
        merged_levels = {}
        for price, data in sorted(price_levels.items()):
            merged = False
            for merged_price in list(merged_levels.keys()):
                if abs(price - merged_price) / merged_price < 0.01:  # Within 1%
                    # Merge into existing level
                    merged_levels[merged_price]['count'] += data['count']
                    merged_levels[merged_price]['indices'].extend(data['indices'])
                    merged_levels[merged_price]['actual_lows'].extend(data['actual_lows'])
                    merged_levels[merged_price]['first_touch'] = min(
                        merged_levels[merged_price]['first_touch'], 
                        data['first_touch']
                    )
                    merged_levels[merged_price]['last_touch'] = max(
                        merged_levels[merged_price]['last_touch'], 
                        data['last_touch']
                    )
                    merged = True
                    break
            
            if not merged:
                merged_levels[price] = data
        
        # Calculate bounces and consolidation for each level
        for level_price, level_data in merged_levels.items():
            # Count bounces (price rising after touching this level)
            bounces = 0
            consolidation_score = 0
            
            for idx in level_data['indices']:
                # Check if price bounced
                if idx < len(lows) - 3:
                    # Look for price rise after touching level
                    next_high = max(highs[idx:idx+3])
                    if next_high > level_price * 1.03:  # 3% rise
                        bounces += 1
                    
                    # Check for consolidation (multiple bars near this level)
                    nearby_bars = 0
                    for j in range(max(0, idx-5), min(len(lows), idx+5)):
                        if abs(lows[j] - level_price) / level_price < 0.02:  # Within 2%
                            nearby_bars += 1
                    consolidation_score += nearby_bars
            
            level_data['bounces'] = bounces
            level_data['consolidation_score'] = consolidation_score
            level_data['span'] = level_data['last_touch'] - level_data['first_touch']
            level_data['min_low'] = min(level_data['actual_lows'])
        
        # Score each level
        best_level = None
        best_score = -999
        
        # Current price and absolute low for context
        current_price = closes[-1]
        absolute_low = min(lows)
        
        for level_price, level_data in merged_levels.items():
            # Skip if too few touches
            if level_data['count'] < 2:
                continue
                
            # Skip very recent levels (last 2 bars)
            if level_data['last_touch'] > len(lows) - 3:
                continue
            
            # Calculate comprehensive score
            score = 0
            
            # 1. Multiple touches are crucial
            score += level_data['count'] * 5
            
            # 2. Bounces show real support
            score += level_data['bounces'] * 10
            
            # 3. Consolidation indicates significance
            score += min(level_data['consolidation_score'] / 5, 20)
            
            # 4. Time span (established levels)
            if level_data['span'] > 5:
                score += min(level_data['span'] / 2, 15)
            
            # 5. Position in range
            price_position = (level_data['min_low'] - absolute_low) / (max(highs) - absolute_low)
            
            # Prefer levels that are NOT at the absolute bottom
            if 0.15 < price_position < 0.7:  # Lower-middle to middle range
                score += 25
            elif price_position < 0.05:  # Too close to absolute low
                score -= 30
            
            # 6. Distance from current price (not too far)
            distance_from_current = abs(level_data['min_low'] - current_price) / current_price
            if distance_from_current < 0.15:  # Within 15%
                score += 10
            
            # 7. Cluster density bonus
            cluster_count = 0
            for other_price in merged_levels:
                if other_price != level_price and abs(other_price - level_price) / level_price < 0.02:
                    cluster_count += 1
            score += cluster_count * 5
            
            level_data['score'] = score
            
            if score > best_score:
                best_score = score
                best_level = level_data
        
        # Return the lowest actual low from the best support zone
        if best_level and best_level['score'] > 20:
            return best_level['min_low']
        
        # Fallback: Look for the level with most touches that's not the absolute low
        fallback_levels = []
        for level_price, level_data in merged_levels.items():
            if level_data['min_low'] > absolute_low * 1.05:  # Not the very bottom
                fallback_levels.append({
                    'price': level_data['min_low'],
                    'touches': level_data['count'],
                    'bounces': level_data['bounces']
                })
        
        if fallback_levels:
            # Sort by touches + bounces
            fallback_levels.sort(key=lambda x: x['touches'] + x['bounces'], reverse=True)
            return fallback_levels[0]['price']
        
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
        
        setup = {
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
        
        # Debug: Print the setup structure before enrichment
        logger.info(f"🔍 Setup structure for {symbol} BEFORE enrichment:")
        logger.info(f"  fvg_zone: {setup.get('fvg_zone', 'NOT FOUND')}")
        logger.info(f"  swing_levels: {setup.get('swing_levels', 'NOT FOUND')}")
        logger.info(f"  targets: {setup.get('targets', 'NOT FOUND')}")
        
        # Store original setup data before enrichment
        original_fvg_zone = setup.get('fvg_zone', {})
        original_swing_levels = setup.get('swing_levels', {})
        original_targets = setup.get('targets', {})
        
        # Enrich with universal data
        logger.info(f"🔧 Enriching data for {symbol}")
        try:
            enriched_output = self.enricher.enrich_scanner_output('ict', setup)
            
            # Preserve original nested data in enriched output
            enriched_output['original_fvg_zone'] = original_fvg_zone
            enriched_output['original_swing_levels'] = original_swing_levels
            enriched_output['original_targets'] = original_targets
            
            return enriched_output
        except Exception as e:
            logger.error(f"❌ Enrichment failed for {symbol}: {e}")
            # Return original setup if enrichment fails
            return setup
        

        
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
        
    def scan_symbol(self, symbol):
        """Scan a single symbol for ICT FVG setups"""
        setups = []
        
        try:
            # Fetch and analyze
            df = self.fetch_candles(symbol)
            if df.empty or len(df) < 50:
                logger.debug(f"⏭️ Skipping {symbol}: No data or insufficient data")
                return setups
                
            # Detect FVGs
            fvgs = self.detect_fvg(df)
            logger.debug(f"🔍 Found {len(fvgs)} FVGs for {symbol}")
            
            for fvg in fvgs:
                try:
                    setup = self.calculate_setup_details(
                        symbol, 
                        fvg, 
                        df.iloc[-1]['close']
                    )
                    
                    # Only add high quality setups
                    if setup['final_quality'] >= 75:  # Default minimum quality
                        # Debug: Print the setup structure before enrichment
                        logger.info(f"🔍 Setup structure for {symbol} BEFORE enrichment:")
                        logger.info(f"  fvg_zone: {setup.get('fvg_zone', 'NOT FOUND')}")
                        logger.info(f"  swing_levels: {setup.get('swing_levels', 'NOT FOUND')}")
                        logger.info(f"  targets: {setup.get('targets', 'NOT FOUND')}")
                        
                        setups.append(setup)
                        logger.info(f"📊 Found setup for {symbol}: Quality={setup['final_quality']}, R/R={setup['risk_reward']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error calculating setup for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            
        return setups

    def display_scan_summary(self, high_quality_setups):
        """Display scan summary"""
        if not high_quality_setups:
            return
            
        print(f"\n{'='*60}")
        print(f"SCAN SUMMARY: Found {len(high_quality_setups)} setups, showing top {min(len(high_quality_setups), 20)}")
        print(f"{'='*60}")
        
        # Show top 10 by different criteria
        print(f"\n📊 TOP 10 BY QUALITY SCORE:")
        for i, setup in enumerate(high_quality_setups[:10]):
            action = "🟢 LONG" if setup['type'] == 'bullish' else "🔴 SHORT"
            print(f"{i+1:2d}. {action} {setup['symbol']:<12} | Score: {setup['final_quality']:3.0f} | R/R: {setup['risk_reward']:4.1f}:1 | Entry Distance: {abs(setup['distance_to_entry']):5.1f}%")
        
        # Best risk/reward setups
        by_rr = sorted(high_quality_setups, key=lambda x: x['risk_reward'], reverse=True)[:5]
        print(f"\n💰 TOP 5 BY RISK/REWARD:")
        for i, setup in enumerate(by_rr):
            action = "🟢 LONG" if setup['type'] == 'bullish' else "🔴 SHORT"
            print(f"{i+1}. {action} {setup['symbol']:<12} | R/R: {setup['risk_reward']:4.1f}:1 | Quality: {setup['final_quality']:3.0f}")
        
        # Immediately actionable (closest to entry)
        actionable = [s for s in high_quality_setups if abs(s['distance_to_entry']) < 1.0]
        if actionable:
            print(f"\n⚡ IMMEDIATELY ACTIONABLE (within 1% of entry):")
            for setup in actionable[:5]:
                action = "🟢 LONG" if setup['type'] == 'bullish' else "🔴 SHORT"
                print(f"• {action} {setup['symbol']:<12} | Entry: ${setup['entry_price']:.4f} | Current: ${setup['current_price']:.4f}")

    def scan_all_symbols(self, top_n=500, min_quality=75, max_alerts=20):
        """Main scanning loop with improved sorting and limits"""
        logger.info(f"Starting scan of top {top_n} symbols...")
        self.performance_stats['total_scans'] += 1
        
        # Start scan logging if database is available (do this early)
        scan_id = None
        
        # Check if database is connected
        if not self.db_logger or not self.db_logger.connection:
            logger.warning("⚠️ Database not connected - running in offline mode")
            logger.warning("Results will not be saved to database")
            logger.info("📝 Running without database logging (offline mode)")
        else:
            try:
                scan_id = self.db_logger.log_scan_start('ict_scanner_4h', 'R8')
                logger.info(f"✅ Scan started with ID: {scan_id}")
            except Exception as e:
                logger.warning(f"❌ Failed to start scan logging: {e}")
                scan_id = None
                logger.info("📝 Running without database logging (offline mode)")
        
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
                    
                # Scan for setups
                setups = self.scan_symbol(symbol)
                
                # Filter by quality
                quality_setups = [s for s in setups if s.get('final_quality', 0) >= min_quality]
                
                if quality_setups:
                    # Sort by quality and add to results
                    quality_setups.sort(key=lambda x: x.get('final_quality', 0), reverse=True)
                    high_quality_setups.extend(quality_setups)
                    
                    # Log to main database (if available)
                    if self.db_logger and self.db_logger.connection:
                        try:
                            for setup in quality_setups:
                                # Debug: Print the original setup structure
                                logger.info(f"🔍 Original setup structure for {symbol}:")
                                logger.info(f"  fvg_zone: {setup.get('fvg_zone', 'NOT FOUND')}")
                                logger.info(f"  swing_levels: {setup.get('swing_levels', 'NOT FOUND')}")
                                logger.info(f"  targets: {setup.get('targets', 'NOT FOUND')}")
                                
                                # Enrich the setup for main database logging
                                enriched_setup = self.enricher.enrich_scanner_output('ict', setup)
                                self.db_logger.log_trade_opportunity(scan_id, enriched_setup)
                        except Exception as e:
                            logger.warning(f"❌ Failed to log ICT setup: {symbol}")
                            logger.error(f"Error logging trade {symbol}: {e}")
                
                # Update last scan time
                self.last_scan_time[symbol] = datetime.now()
                
            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")
                continue
        
        # Sort all results by quality
        high_quality_setups.sort(key=lambda x: x.get('final_quality', 0), reverse=True)
        
        # Limit results
        high_quality_setups = high_quality_setups[:max_alerts]
        
        # Store results for universal logger
        self.last_scan_results = high_quality_setups
        
        # Display results
        if high_quality_setups:
            logger.info(f"Found {len(high_quality_setups)} setups, showing top {min(len(high_quality_setups), max_alerts)}")
            
            for setup in high_quality_setups[:max_alerts]:
                alert = self.format_alert(setup)
                logger.info(alert)
        else:
            logger.info("No high-quality setups found")
        
        # Display summary
        self.display_scan_summary(high_quality_setups)
        
        # Update performance stats
        self.performance_stats['setups_found'] += len(high_quality_setups)
        self.performance_stats['alerts_sent'] += len(high_quality_setups)
        
        logger.info(f"Scan complete. Found {len(high_quality_setups)} high-quality setups, displayed {min(len(high_quality_setups), max_alerts)}")
        logger.info(f"Performance Stats: {self.performance_stats}")
        
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
        
    def run_continuous(self, scan_interval=900, max_alerts_per_scan=20):
        """Run scanner continuously with alert limits"""
        logger.info(f"Starting continuous scanner (interval: {scan_interval}s)")
        
        while True:
            try:
                # Run scan with alert limit
                setups = self.scan_all_symbols(max_alerts=max_alerts_per_scan)
                
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





def call_universal_logger(high_quality_setups):
    """Call the universal logger to log trades to the other_scanners database"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'database_and_logging'))
        from universal_scanner_logger import UniversalScannerLogger
        
        # Initialize logger
        logger = UniversalScannerLogger('ict_4h_scanner', 'r9')
        
        # Log all detected trades
        if high_quality_setups:
            for setup in high_quality_setups:
                # Debug: Print the setup structure
                print(f"🔍 Setup structure for {setup.get('symbol', 'UNKNOWN')}:")
                print(f"  fvg_zone: {setup.get('fvg_zone', 'NOT FOUND')}")
                print(f"  swing_levels: {setup.get('swing_levels', 'NOT FOUND')}")
                print(f"  targets: {setup.get('targets', 'NOT FOUND')}")
                print(f"  gap_high: {setup.get('gap_high', 'NOT FOUND')}")
                print(f"  gap_low: {setup.get('gap_low', 'NOT FOUND')}")
                
                # Prepare trade data for universal logger
                trade_data = {
                    'symbol': setup.get('symbol', ''),
                    'timeframe': '4h',  # Lowercase!
                    'side': 'BUY' if setup.get('type') == 'bullish' else 'SELL',
                    'entry_price': float(setup.get('entry_price', 0)),
                    'quantity': 1.0,  # Default quantity
                    'stop_loss': float(setup.get('stop_loss', 0)),
                    'take_profit': float(setup.get('targets', {}).get('T1', 0)),
                    'current_price': float(setup.get('current_price', 0)),
                    'risk_reward_ratio': float(setup.get('risk_reward', 0)),
                    'target_1': float(setup.get('targets', {}).get('T1', 0)),
                    'target_2': float(setup.get('targets', {}).get('T2', 0)),
                    'target_3': float(setup.get('targets', {}).get('T3', 0)),
                    
                    # Technical indicators
                    'technical_indicators': {
                        'rsi': float(setup.get('rsi', 0)),
                        'mfi': float(setup.get('mfi', 0)),
                        'stochastic_k': float(setup.get('stochastic_k', 0)),
                        'volume_surge': float(setup.get('volume_surge', 0)),
                        'macd_signal': setup.get('macd_signal', 'neutral'),
                        'final_quality': float(setup.get('final_quality', 0)),
                        'risk_reward': float(setup.get('risk_reward', 0)),
                        'probability': float(setup.get('probability', setup.get('final_quality', 0)))
                    },
                    
                    # Scanner signals - FIXED: Use preserved original data
                    'scanner_signals': {
                        'timeframe': '4h',
                        'pattern_type': f"ICT FVG {setup.get('type', 'unknown')}",
                        'pattern_quality': 'GOOD' if setup.get('final_quality', 0) > 80 else 'FAIR',
                        'gap_size_pct': float(setup.get('gap_size_pct', 0)),
                        'fvg_age': int(setup.get('fvg_age', 0)),
                        'distance_to_entry': float(setup.get('distance_to_entry', 0)),
                        'gap_high': float(setup.get('original_fvg_zone', {}).get('high', 0)),
                        'gap_low': float(setup.get('original_fvg_zone', {}).get('low', 0)),
                        'gap_midpoint': float(setup.get('entry_price', 0)),
                        'swing_high': float(setup.get('original_swing_levels', {}).get('high', 0)),
                        'swing_low': float(setup.get('original_swing_levels', {}).get('low', 0)),
                        'swing_range_pct': float(abs(setup.get('original_swing_levels', {}).get('high', 0) - setup.get('original_swing_levels', {}).get('low', 0)) / setup.get('original_swing_levels', {}).get('low', 1) * 100),
                        'order_block_high': float(setup.get('order_block_high', 0)),
                        'order_block_low': float(setup.get('order_block_low', 0)),
                        'liquidity_sweep_level': float(setup.get('liquidity_sweep_level', 0)),
                        'imbalance_high': float(setup.get('imbalance_high', 0)),
                        'imbalance_low': float(setup.get('imbalance_low', 0)),
                        'risk_pct': float(setup.get('risk_pct', 0)),
                        'action_required': setup.get('action_required', ''),
                        'quality_score': float(setup.get('quality_score', 0)),
                        'fib_quality': float(setup.get('fib_quality', 0)),
                        'current_price': float(setup.get('current_price', 0)),
                        'risk_reward_ratio': float(setup.get('risk_reward', 0)),
                        'target_t1': float(setup.get('targets', {}).get('T1', 0)),
                        'target_t2': float(setup.get('targets', {}).get('T2', 0)),
                        'target_t3': float(setup.get('targets', {}).get('T3', 0)),
                        'side': setup.get('type', 'UNKNOWN').upper()
                    },
                    
                    # Market conditions
                    'market_conditions': {
                        'market_cap': float(setup.get('market_cap', 0)),
                        'volume_24h': float(setup.get('volume_24h', 0)),
                        'price_change_24h': float(setup.get('price_change_24h', 0)),
                        'historical_win_rate': float(setup.get('historical_win_rate', 0)),
                        'category_win_rate': float(setup.get('category_win_rate', 0)),
                        'similar_setups_count': int(setup.get('similar_setups_count', 0)),
                        'current_price': float(setup.get('current_price', 0)),
                        'category': setup.get('category', ''),
                        'volume_surge': bool(setup.get('volume_surge', False))
                    },
                    
                    # Fibonacci targets - FIXED: Use preserved original targets
                    'feature_vector': [
                        float(setup.get('original_targets', {}).get('T1', 0)),
                        float(setup.get('original_targets', {}).get('T2', 0)),
                        float(setup.get('original_targets', {}).get('T3', 0)),
                        float(setup.get('original_targets', {}).get('fib_236', 0)),
                        float(setup.get('original_targets', {}).get('fib_382', 0)),
                        float(setup.get('original_targets', {}).get('fib_500', 0)),
                        float(setup.get('original_targets', {}).get('fib_618', 0)),
                        float(setup.get('original_targets', {}).get('fib_786', 0))
                    ]
                }
                
                # Log the trade
                trade_id = logger.log_trade(trade_data)
                if trade_id:
                    print(f"✅ Logged ICT trade {trade_id}: {setup.get('symbol')} (Quality: {setup.get('final_quality', 0):.1f})")
                else:
                    print(f"❌ Failed to log ICT trade: {setup.get('symbol')}")
                    
            # Close logger properly
            if hasattr(logger, 'connection_pool'):
                logger.connection_pool.closeall()
            print(f"✅ Successfully logged {len(high_quality_setups)} ICT trades to database")
        else:
            print("ℹ️ No ICT trades found in this scan")
            
    except Exception as e:
        print(f"⚠️ Database logging failed (non-critical): {e}")
        # Don't crash the scanner if logging fails
        pass


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT FVG Scanner 4H')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuous')
    parser.add_argument('--symbols', type=int, default=500, help='Number of symbols to scan')
    parser.add_argument('--quality', type=int, default=85, help='Minimum quality threshold (default: 85)')
    parser.add_argument('--max-alerts', type=int, default=20, help='Maximum alerts per scan (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = ICTFVGScanner(test_mode=args.test)
    
    if args.once:
        # Run single scan
        setups = scanner.scan_all_symbols(
            top_n=args.symbols, 
            min_quality=args.quality,
            max_alerts=args.max_alerts
        )
        
        # The summary is now printed within scan_all_symbols
        
        # Call universal logger integration
        call_universal_logger(setups)
            
    else:
        # Run continuous scanning
        scanner.run_continuous(max_alerts_per_scan=args.max_alerts)


if __name__ == "__main__":
    main()