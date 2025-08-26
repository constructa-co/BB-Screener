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

# Add TradeLogger for database integration
try:
    from database_and_logging.trade_logger import TradeLogger
    trade_logger = TradeLogger()
    DB_LOGGING_AVAILABLE = True
except ImportError:
    print("Warning: TradeLogger not available - database logging disabled")
    DB_LOGGING_AVAILABLE = False

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
    """Production FVG Scanner based on 15M R2 proven parameters"""
    
    def __init__(self, exchange_id='binance', test_mode=False, elite_mode=True):
        """Initialize scanner with 15M R2 winning configuration"""
        # Elite mode: True for only 97%+ setups, False for all FVGs
        self.elite_mode = elite_mode
        
        # 15M R2 Proven Parameters (from successful backtest)
        self.config = {
            # FVG Detection (15M optimized)
            'min_fvg_size': 0.002,          # 0.2% for 15M volatility (from 0.4%)
            'max_fvg_age': 40,              # 40 bars for 15M (from 72)
            'min_quality_score': 65,         # Minimum quality threshold (from 70)
            'max_distance_to_fvg': 0.10,    # Max 10% distance from current price
            
            # Order Block Detection (15M optimized)
            'min_order_block_size': 0.5,    # 0.5% for 15M (from 1.0)
            'min_volume_ratio': 1.2,        # 1.2x for 15M (from 1.5)
            'max_ob_age': 40,               # 40 bars for 15M (from 72)
            
            # Targets (15M optimized)
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
            
            # Risk Management (15M R2 optimized with commission adjustment)
            'stop_loss_pct': 0.003,         # 0.3% stop loss (commission-adjusted from 0.5%)
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
            
    def fetch_candles(self, symbol: str, timeframe='15m', limit=200) -> pd.DataFrame:
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
                                    # NEW: Check for RSI/MFI confluence
                                    fvg_index = fvg['index']
                                    fvg_type = fvg['type']  # 'bullish' or 'bearish'
                                    
                                    rsi_signal, rsi_desc = self.check_rsi_confluence(df, fvg_index, fvg_type)
                                    mfi_signal, mfi_desc = self.check_mfi_confluence(df, fvg_index, fvg_type)
                                    
                                    # Debug logging for confluence detection
                                    if logger.level <= logging.DEBUG:
                                        logger.debug(f"Bullish FVG at index {fvg_index}: RSI={df['rsi'].iloc[fvg_index]:.1f}, MFI={df['mfi'].iloc[fvg_index]:.1f}")
                                        logger.debug(f"RSI Signal: {rsi_signal} ({rsi_desc}), MFI Signal: {mfi_signal} ({mfi_desc})")
                                    
                                    # Check elite mode filtering
                                    if self.elite_mode:
                                        # Only show FVGs with RSI/MFI confluence
                                        if not (rsi_signal or mfi_signal):
                                            continue
                                    
                                    # Add confluence information to the signal
                                    fvg['has_elite_confluence'] = rsi_signal or mfi_signal
                                    fvg['confluence_factors'] = []
                                    
                                    if rsi_signal:
                                        fvg['confluence_factors'].append(rsi_desc)
                                    if mfi_signal:
                                        fvg['confluence_factors'].append(mfi_desc)
                                        
                                    fvg['expected_win_rate'] = '97%+' if (rsi_signal and mfi_signal) else '97%' if (rsi_signal or mfi_signal) else '85%'
                                    
                                    # Now add to signals
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
                                    # NEW: Check for RSI/MFI confluence
                                    fvg_index = fvg['index']
                                    fvg_type = fvg['type']  # 'bullish' or 'bearish'
                                    
                                    rsi_signal, rsi_desc = self.check_rsi_confluence(df, fvg_index, fvg_type)
                                    mfi_signal, mfi_desc = self.check_mfi_confluence(df, fvg_index, fvg_type)
                                    
                                    # Check elite mode filtering
                                    if self.elite_mode:
                                        # Only show FVGs with RSI/MFI confluence
                                        if not (rsi_signal or mfi_signal):
                                            continue
                                    
                                    # Add confluence information to the signal
                                    fvg['has_elite_confluence'] = rsi_signal or mfi_signal
                                    fvg['confluence_factors'] = []
                                    
                                    if rsi_signal:
                                        fvg['confluence_factors'].append(rsi_desc)
                                    if mfi_signal:
                                        fvg['confluence_factors'].append(mfi_desc)
                                        
                                    fvg['expected_win_rate'] = '97%+' if (rsi_signal and mfi_signal) else '97%' if (rsi_signal or mfi_signal) else '85%'
                                    
                                    # Now add to signals
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
        swing_high = recent_df['high'].max()
        
        # Validate swing high makes sense
        current_price = recent_df['close'].iloc[-1]
        if swing_high < current_price:
            # If swing high is below current price, use recent max
            swing_high = recent_df['high'].rolling(20).max().iloc[-1]
        
        return swing_high
        
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
        swing_low = recent_df['low'].min()
        
        # Validate swing low makes sense
        current_price = recent_df['close'].iloc[-1]
        if swing_low > current_price:
            # If swing low is above current price, use recent min
            swing_low = recent_df['low'].rolling(20).min().iloc[-1]
        
        return swing_low
        
    def calculate_smart_fibonacci_targets(self, df: pd.DataFrame, fvg: Dict, entry_price: float, stop_loss: float) -> Dict:
        """
        Calculate proper Fibonacci-based targets using market structure
        """
        # Find recent swing points
        swing_high = self.find_swing_high(df, lookback=50)
        swing_low = self.find_swing_low(df, lookback=50)
        
        # Error handling for None values
        if swing_high is None or swing_low is None:
            # Fallback to percentage-based targets
            if fvg['type'] == 'bullish':
                targets = {
                    'T1': entry_price * 1.03,
                    'T2': entry_price * 1.05,
                    'T3': entry_price * 1.07
                }
                return {'targets': targets, 'note': 'Fallback +3/5/7% (swing error)', 'fib_quality': 50}
            else:
                targets = {
                    'T1': entry_price * 0.97,
                    'T2': entry_price * 0.95,
                    'T3': entry_price * 0.93
                }
                return {'targets': targets, 'note': 'Fallback -3/-5/-7% (swing error)', 'fib_quality': 50}
        
        # Ensure swing levels are valid
        if swing_high <= swing_low:
            # Swap if inverted
            swing_high, swing_low = swing_low, swing_high
            
        fib_range = abs(swing_high - swing_low)
        # Fallback for tiny swing range
        if fib_range < entry_price * 0.01:
            if fvg['type'] == 'bullish':
                targets = {
                    'T1': entry_price * 1.03,
                    'T2': entry_price * 1.05,
                    'T3': entry_price * 1.07
                }
                return {'targets': targets, 'note': 'Fallback +3/5/7%', 'fib_quality': 50}
            else:
                targets = {
                    'T1': entry_price * 0.97,
                    'T2': entry_price * 0.95,
                    'T3': entry_price * 0.93
                }
                return {'targets': targets, 'note': 'Fallback -3/-5/-7%', 'fib_quality': 50}
        
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
        """Categorize token based on R2 categories"""
        categories = {
            'Major Cryptos': ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'TRX', 'LTC', 'BCH'],
            'Layer 1s': ['DOT', 'AVAX', 'ATOM', 'NEAR', 'ALGO', 'ICP', 'FLOW', 'EGLD', 'XTZ', 'THETA', 'HBAR'],
            'Layer 2/Scaling': ['MATIC', 'ARB', 'OP', 'IMX', 'LRC', 'BLUR'],
            'DeFi': ['UNI', 'AAVE', 'SUSHI', 'COMP', 'MKR', 'SNX', 'CRV', 'YFI', 'BAL', '1INCH', 'LDO'],
            'Meme/Community': ['SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', '1000CAT', '1000CHEEMS'],
            'Gaming/NFTs': ['MANA', 'SAND', 'AXS', 'ENJ', 'GALA', 'CHZ', 'ALICE', 'SLP', 'JUV', 'PIXEL'],
            'Infrastructure': ['LINK', 'GRT', 'FIL', 'AR', 'OCEAN', 'STORJ', 'LPT', 'API3', 'BAND'],
            'Altcoins': ['VET', 'XLM', 'XTZ', 'EOS', 'NEO', 'QTUM', 'ZIL', 'HOT', 'IOTA', 'ZEC', 'DASH', 'KSM', 'RUNE', 'LUNA', 'FET', 'APE', 'GMT', 'CELO', 'BAT', 'ONE', 'TON', 'ACM', 'AIXBT', 'VIRTUAL', 'CVX', 'FXS', 'ETHFI', 'ETC', 'TREE', 'ASR', 'OM', 'TAO', 'AMP']
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
        
        # Calculate stop loss using R2 parameter
        stop_loss_pct = self.config.get('stop_loss_pct', 0.003)  # 0.3% default (commission-adjusted)
        
        if fvg['type'] == 'bullish':
            # Stop loss below entry using R2 parameter
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:  # bearish
            # Stop loss above entry using R2 parameter
            stop_loss = entry_price * (1 + stop_loss_pct)
            
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
        # Validate targets are in correct direction
        if setup['type'] == 'bearish':
            if setup['targets']['T1'] > setup['entry_price']:
                print(f"ERROR: Bearish trade {setup['symbol']} has targets above entry!")
                return None
        if setup['type'] == 'bullish':
            if setup['targets']['T1'] < setup['entry_price']:
                print(f"ERROR: Bullish trade {setup['symbol']} has targets below entry!")
                return None
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
- Elite Confluence: {'Yes ✅' if setup.get('has_elite_confluence') else 'No ❌'}
- Confluence Factors: {', '.join(setup.get('confluence_factors', [])) if setup.get('confluence_factors') else 'None'}
- Expected Win Rate: {setup.get('expected_win_rate', '85% (standard)')}
- Category: {setup['category']}

💡 Strategy: R21 FVG + Smart Fibonacci
⏰ Time: {setup['timestamp'].strftime('%Y-%m-%d %H:%M')}
"""
        return alert
        
    def scan_all_symbols(self, top_n=500, min_quality=65, max_alerts=20):
        """Main scanning loop with improved sorting and limits"""
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
                
                # Debug indicators if debug mode is enabled
                if logger.level <= logging.DEBUG:
                    self.debug_indicators(df, symbol)
                    
                # Detect FVGs
                fvgs = self.detect_fvg(df)
                
                for fvg in fvgs:
                    setup = self.calculate_setup_details(
                        symbol, 
                        fvg, 
                        df.iloc[-1]['close']
                    )
                    
                    # Only add high quality setups
                    if setup['final_quality'] >= min_quality:
                        high_quality_setups.append(setup)
                        
                self.last_scan_time[symbol] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by multiple criteria for best setups first
        high_quality_setups.sort(
            key=lambda x: (
                x['final_quality'],              # Highest quality first
                x['risk_reward'],                # Best R/R first
                -abs(x['distance_to_entry']),    # Closest to entry first
                -x['fvg_age'],                   # Freshest FVGs first
                1 if x['volume_surge'] else 0    # Volume surge preferred
            ), 
            reverse=True
        )
        
        # Limit to top setups if too many
        total_found = len(high_quality_setups)
        if total_found > max_alerts:
            logger.info(f"Found {total_found} setups, showing top {max_alerts}")
            high_quality_setups = high_quality_setups[:max_alerts]
        
        # Update stats
        self.performance_stats['setups_found'] = total_found
        
        # Send alerts for selected setups and log to database
        for setup in high_quality_setups:
            alert_text = self.format_alert(setup)
            if alert_text: # Only send if format_alert didn't return None
                logger.info(alert_text)
                self.send_alert(alert_text)
                self.performance_stats['alerts_sent'] += 1
                
            # Log to database if TradeLogger is available
            if DB_LOGGING_AVAILABLE:
                try:
                    # Create unique ID for the signal
                    signal_id = f"ICT_15M_{setup['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Prepare opportunity data
                    opportunity = {
                        'symbol': setup['symbol'],
                        'timeframe': '15m',
                        'pattern_type': setup.get('pattern_type', 'ICT_FVG'),
                        'entry_price': setup.get('entry_price'),
                        'stop_loss': setup.get('stop_loss'),
                        'target_1': setup.get('target_1'),
                        'target_2': setup.get('target_2'),
                        'target_3': setup.get('target_3'),
                        'current_price': setup.get('current_price'),
                        'risk_reward_ratio': setup.get('risk_reward', 0),
                        'confidence_score': setup.get('final_quality', 50),
                        'scanner_specific_data': {
                            'timeframe': '15m',
                            'pattern_type': setup.get('pattern_type', 'ICT_FVG'),
                            'fvg_age': setup.get('fvg_age', 0),
                            'distance_to_entry': setup.get('distance_to_entry', 0),
                            'gap_midpoint': setup.get('gap_midpoint', 0),
                            'swing_range_pct': setup.get('swing_range_pct', 0),
                            'gap_high': setup.get('gap_high', 0),
                            'gap_low': setup.get('gap_low', 0),
                            'gap_size_pct': setup.get('gap_size_pct', 0),
                            'swing_high': setup.get('swing_high', 0),
                            'swing_low': setup.get('swing_low', 0),
                            'risk_pct': setup.get('risk_pct', 0),
                            'action_required': setup.get('action_required', 'MONITOR'),
                            'quality_score': setup.get('final_quality', 0),
                            'fib_quality': setup.get('fib_quality', 0),
                            'category': setup.get('category', ''),
                            'volume_surge': float(setup.get('volume_surge', False)),
                            'target_t1': setup.get('target_1', 0),
                            'target_t2': setup.get('target_2', 0),
                            'target_t3': setup.get('target_3', 0),
                            'current_price': setup.get('current_price', 0),
                            'risk_reward_ratio': setup.get('risk_reward', 0),
                            'side': setup.get('type', 'UNKNOWN').upper()
                        }
                    }
                    
                    # Log to database
                    trade_logger.log_scan(
                        scan_id=signal_id,
                        symbol=setup['symbol'],
                        opportunities=[opportunity]
                    )
                    print(f"✅ Logged ICT 15M signal for {setup['symbol']}")
                    
                except Exception as e:
                    print(f"❌ Failed to log 15M signal for {setup['symbol']}: {e}")
                
        # Print summary at the end
        if high_quality_setups:
            print(f"\n{'='*60}")
            print(f"SCAN SUMMARY: Found {total_found} setups, showing top {len(high_quality_setups)}")
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
            
            # Elite setups summary
            if self.elite_mode:
                elite_setups = [s for s in high_quality_setups if s.get('has_elite_confluence', False)]
                if elite_setups:
                    print(f"\n🏆 ELITE SETUPS (97%+ WIN RATE):")
                    for i, setup in enumerate(elite_setups[:5], 1):
                        confluence = ', '.join(setup.get('confluence_factors', []))
                        action = "🟢 LONG" if setup['type'] == 'bullish' else "🔴 SHORT"
                        print(f"{i}. {action} {setup['symbol']:<12} | Win Rate: {setup.get('expected_win_rate', 'N/A')} | {confluence}")
                else:
                    print(f"\n⚠️ No elite setups found (97%+ win rate)")
            
            # Immediately actionable (closest to entry)
            actionable = [s for s in high_quality_setups if abs(s['distance_to_entry']) < 1.0]
            if actionable:
                print(f"\n⚡ IMMEDIATELY ACTIONABLE (within 1% of entry):")
                for setup in actionable[:5]:
                    action = "🟢 LONG" if setup['type'] == 'bullish' else "🔴 SHORT"
                    print(f"• {action} {setup['symbol']:<12} | Entry: ${setup['entry_price']:.4f} | Current: ${setup['current_price']:.4f}")
        
        # Summary
        logger.info(f"Scan complete. Found {total_found} high-quality setups, displayed {len(high_quality_setups)}")
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
        
    def run_continuous(self, scan_interval=300, max_alerts_per_scan=20):
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
                
    def check_rsi_confluence(self, df: pd.DataFrame, fvg_index: int, fvg_type: str) -> Tuple[bool, Optional[str]]:
        """
        Check for RSI confluence at FVG formation - 97.2% win rate
        """
        rsi = df['rsi'].iloc[fvg_index]
        
        if pd.isna(rsi):
            return False, None
            
        # For BULLISH FVGs
        if fvg_type == 'bullish':
            if rsi < 30:  # Classic oversold
                return True, f"RSI Oversold ({rsi:.1f})"
                
            # Check for bullish divergence
            if self._check_rsi_bullish_divergence(df, fvg_index):
                return True, f"RSI Bullish Divergence ({rsi:.1f})"
                
        # For BEARISH FVGs
        elif fvg_type == 'bearish':
            if rsi > 70:  # Classic overbought
                return True, f"RSI Overbought ({rsi:.1f})"
                
            # Check for bearish divergence
            if self._check_rsi_bearish_divergence(df, fvg_index):
                return True, f"RSI Bearish Divergence ({rsi:.1f})"
                
        return False, None

    def check_mfi_confluence(self, df: pd.DataFrame, fvg_index: int, fvg_type: str) -> Tuple[bool, Optional[str]]:
        """
        Check for MFI confluence at FVG formation - 97.1% win rate
        """
        mfi = df['mfi'].iloc[fvg_index]
        
        if pd.isna(mfi):
            return False, None
            
        # For BULLISH FVGs
        if fvg_type == 'bullish':
            if mfi < 20:  # Strong oversold
                return True, f"MFI Oversold ({mfi:.1f})"
                
            # Check for bullish divergence
            if self._check_mfi_bullish_divergence(df, fvg_index):
                return True, f"MFI Bullish Divergence ({mfi:.1f})"
                
        # For BEARISH FVGs
        elif fvg_type == 'bearish':
            if mfi > 80:  # Strong overbought
                return True, f"MFI Overbought ({mfi:.1f})"
                
            # Check for bearish divergence
            if self._check_mfi_bearish_divergence(df, fvg_index):
                return True, f"MFI Bearish Divergence ({mfi:.1f})"
                
        return False, None

    def _check_rsi_bullish_divergence(self, df: pd.DataFrame, current_index: int, lookback: int = 10) -> bool:
        """Check for bullish divergence: price lower low, RSI higher low"""
        if current_index < lookback:
            return False
            
        # Get recent data
        recent_prices = df['low'].iloc[current_index-lookback:current_index+1]
        recent_rsi = df['rsi'].iloc[current_index-lookback:current_index+1]
        
        # Find swing lows
        price_lows = []
        rsi_lows = []
        
        for i in range(1, len(recent_prices)-1):
            # Price swing low
            if (recent_prices.iloc[i] < recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] < recent_prices.iloc[i+1]):
                price_lows.append((i, recent_prices.iloc[i]))
                
            # RSI swing low
            if (recent_rsi.iloc[i] < recent_rsi.iloc[i-1] and 
                recent_rsi.iloc[i] < recent_rsi.iloc[i+1]):
                rsi_lows.append((i, recent_rsi.iloc[i]))
        
        # Need at least 2 lows to compare
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Check most recent two lows
            if (price_lows[-1][1] < price_lows[-2][1] and  # Price lower low
                rsi_lows[-1][1] > rsi_lows[-2][1]):         # RSI higher low
                return True
                
        return False

    def _check_rsi_bearish_divergence(self, df: pd.DataFrame, current_index: int, lookback: int = 10) -> bool:
        """Check for bearish divergence: price higher high, RSI lower high"""
        if current_index < lookback:
            return False
            
        recent_prices = df['high'].iloc[current_index-lookback:current_index+1]
        recent_rsi = df['rsi'].iloc[current_index-lookback:current_index+1]
        
        price_highs = []
        rsi_highs = []
        
        for i in range(1, len(recent_prices)-1):
            if (recent_prices.iloc[i] > recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] > recent_prices.iloc[i+1]):
                price_highs.append((i, recent_prices.iloc[i]))
                
            if (recent_rsi.iloc[i] > recent_rsi.iloc[i-1] and 
                recent_rsi.iloc[i] > recent_rsi.iloc[i+1]):
                rsi_highs.append((i, recent_rsi.iloc[i]))
        
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if (price_highs[-1][1] > price_highs[-2][1] and  # Price higher high
                rsi_highs[-1][1] < rsi_highs[-2][1]):         # RSI lower high
                return True
                
        return False

    def _check_mfi_bullish_divergence(self, df: pd.DataFrame, current_index: int, lookback: int = 10) -> bool:
        """Check for MFI bullish divergence"""
        if current_index < lookback:
            return False
            
        recent_prices = df['low'].iloc[current_index-lookback:current_index+1]
        recent_mfi = df['mfi'].iloc[current_index-lookback:current_index+1]
        
        price_lows = []
        mfi_lows = []
        
        for i in range(1, len(recent_prices)-1):
            if (recent_prices.iloc[i] < recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] < recent_prices.iloc[i+1]):
                price_lows.append((i, recent_prices.iloc[i]))
                
            if (recent_mfi.iloc[i] < recent_mfi.iloc[i-1] and 
                recent_mfi.iloc[i] < recent_mfi.iloc[i+1]):
                mfi_lows.append((i, recent_mfi.iloc[i]))
        
        if len(price_lows) >= 2 and len(mfi_lows) >= 2:
            if (price_lows[-1][1] < price_lows[-2][1] and
                mfi_lows[-1][1] > mfi_lows[-2][1]):
                return True
                
        return False

    def _check_mfi_bearish_divergence(self, df: pd.DataFrame, current_index: int, lookback: int = 10) -> bool:
        """Check for MFI bearish divergence"""
        if current_index < lookback:
            return False
            
        recent_prices = df['high'].iloc[current_index-lookback:current_index+1]
        recent_mfi = df['mfi'].iloc[current_index-lookback:current_index+1]
        
        price_highs = []
        mfi_highs = []
        
        for i in range(1, len(recent_prices)-1):
            if (recent_prices.iloc[i] > recent_prices.iloc[i-1] and 
                recent_prices.iloc[i] > recent_prices.iloc[i+1]):
                price_highs.append((i, recent_prices.iloc[i]))
                
            if (recent_mfi.iloc[i] > recent_mfi.iloc[i-1] and 
                recent_mfi.iloc[i] > recent_mfi.iloc[i+1]):
                mfi_highs.append((i, recent_mfi.iloc[i]))
        
        if len(price_highs) >= 2 and len(mfi_highs) >= 2:
            if (price_highs[-1][1] > price_highs[-2][1] and
                mfi_highs[-1][1] < mfi_highs[-2][1]):
                return True
                
        return False

    def debug_indicators(self, df, symbol):
        """Debug method to check if indicators are calculated"""
        if 'rsi' in df.columns and 'mfi' in df.columns:
            last_5_rsi = df['rsi'].tail(5).values
            last_5_mfi = df['mfi'].tail(5).values
            logger.debug(f"{symbol} - Last 5 RSI: {last_5_rsi}")
            logger.debug(f"{symbol} - Last 5 MFI: {last_5_mfi}")
        else:
            logger.error(f"{symbol} - Missing indicators! Columns: {df.columns.tolist()}")

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
    
    parser = argparse.ArgumentParser(description='ICT FVG Scanner 15M R2')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuous')
    parser.add_argument('--symbols', type=int, default=500, help='Number of symbols to scan')
    parser.add_argument('--quality', type=int, default=65, help='Minimum quality threshold (default: 65)')
    parser.add_argument('--max-alerts', type=int, default=20, help='Maximum alerts per scan (default: 20)')
    parser.add_argument('--elite-mode', action='store_true', default=True, help='Only show 97%+ elite setups (default: True)')
    parser.add_argument('--all-fvgs', action='store_true', help='Show all FVGs, not just elite ones')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Determine elite mode based on arguments
    elite_mode = args.elite_mode and not args.all_fvgs
    
    # Initialize scanner
    scanner = ICTFVGScanner(test_mode=args.test, elite_mode=elite_mode)
    
    if args.once:
        # Run single scan
        setups = scanner.scan_all_symbols(
            top_n=args.symbols, 
            min_quality=args.quality,
            max_alerts=args.max_alerts
        )
        
        # The summary is now printed within scan_all_symbols
            
    else:
        # Run continuous scanning
        scanner.run_continuous(max_alerts_per_scan=args.max_alerts)


if __name__ == "__main__":
    main()