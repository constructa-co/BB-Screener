#!/usr/bin/env python3
"""
ICT 4H Scanner - Order Blocks & Liquidity Grabs
Professional ICT (Inner Circle Trader) concepts scanner

Features:
- Order Block Detection (Bullish & Bearish)
- Liquidity Grab Identification (Buy-Side & Sell-Side)
- Integration with existing FVG scanner
- Multi-confluence scoring system
- Institutional-level precision entries

Based on Michael Huddleston's ICT concepts
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class ICTScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'rateLimit': 50,
            'enableRateLimit': True,
        })
        
        # ICT 4H Parameters - TIGHTENED FOR QUALITY
        self.lookback_periods = 30    # 30 days for 4H analysis
        self.min_order_block_size = 3.0  # 3% minimum move (vs 2%) - more significant
        self.min_liquidity_sweep = 1.0   # 1.0% minimum (vs 0.5%) - stronger reversal
        self.min_volume_ratio = 2.0      # 2.0x average volume (vs 1.5x) - institutional level
        self.max_order_block_age = 15    # Maximum 15 candles (vs 20) - fresher setups
        self.min_quality_score = 85      # 85+ quality score (vs 65) - elite setups only
        
    def get_symbols(self):
        """Get major trading pairs for ICT analysis"""
        try:
            markets = self.exchange.load_markets()
            
            # Get USDT pairs
            usdt_pairs = []
            for symbol, market in markets.items():
                if (symbol.endswith('USDT') and 
                    market['active'] and 
                    market['spot'] and
                    market.get('type') == 'spot'):
                    usdt_pairs.append(symbol)
            
            print(f"📊 Found {len(usdt_pairs)} total USDT pairs")
            
            # Filter by volume for 4H swing trading
            viable_pairs = []
            for symbol in usdt_pairs:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume_24h = ticker.get('quoteVolume', 0)
                    price = ticker.get('last', 0)
                    
                    # 4H swing trading criteria
                    min_volume_24h = 100000    # $100K minimum for 4H swings
                    min_price = 0.0001
                    max_price = 100000
                    
                    if (volume_24h >= min_volume_24h and 
                        min_price <= price <= max_price):
                        viable_pairs.append({
                            'symbol': symbol,
                            'volume_24h': volume_24h,
                            'price': price
                        })
                        
                except Exception:
                    continue
            
            # Sort by volume and take top 100 for 4H analysis
            viable_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            top_pairs = viable_pairs[:100]
            
            print(f"📊 Filtered to {len(top_pairs)} high-volume pairs for 4H ICT analysis")
            return [pair['symbol'] for pair in top_pairs]
            
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            # Fallback to major pairs
            return [
                'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT',
                'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT', 'UNI/USDT',
                'LTC/USDT', 'BCH/USDT', 'BNB/USDT', 'ATOM/USDT', 'NEAR/USDT'
            ]

    def get_4h_data(self, symbol):
        """Get 4H OHLCV data for ICT analysis"""
        try:
            since = self.exchange.milliseconds() - (self.lookback_periods * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '4h', since=since, limit=self.lookback_periods * 6)
            
            if len(ohlcv) < 50:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['atr'] = self.calculate_atr(df, 14)
            df['price_change_pct'] = df['close'].pct_change() * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching 4H data for {symbol}: {e}")
            return None

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(window=period).mean()

    def detect_order_blocks(self, df):
        """
        Detect ICT Order Blocks
        
        Order Block = Last opposing candle before strong directional move
        - Bullish OB: Last red candle before strong green move
        - Bearish OB: Last green candle before strong red move
        """
        order_blocks = []
        
        for i in range(5, len(df) - 5):
            current_candle = df.iloc[i]
            
            # Check for strong moves (3%+ in 4H timeframe) - TIGHTENED
            move_threshold = self.min_order_block_size
            
            # Check for Bullish Order Block
            # Last red candle before strong bullish move
            if (current_candle['close'] < current_candle['open'] and  # Red candle
                current_candle['volume_ratio'] >= self.min_volume_ratio):  # High volume (2x+)
                
                # Check for strong bullish move in next 1-5 candles
                future_moves = []
                for j in range(1, 6):
                    if i + j < len(df):
                        future_price = df.iloc[i + j]['close']
                        move_pct = (future_price - current_candle['close']) / current_candle['close'] * 100
                        future_moves.append(move_pct)
                
                max_future_move = max(future_moves) if future_moves else 0
                
                if max_future_move >= move_threshold:  # 3%+ move required
                    order_blocks.append({
                        'type': 'bullish',
                        'index': i,
                        'timestamp': current_candle['timestamp'],
                        'high': current_candle['high'],
                        'low': current_candle['low'],
                        'open': current_candle['open'],
                        'close': current_candle['close'],
                        'volume_ratio': current_candle['volume_ratio'],
                        'future_move': max_future_move,
                        'age': len(df) - i - 1
                    })
            
            # Check for Bearish Order Block
            # Last green candle before strong bearish move
            elif (current_candle['close'] > current_candle['open'] and  # Green candle
                  current_candle['volume_ratio'] >= self.min_volume_ratio):  # High volume (2x+)
                
                # Check for strong bearish move in next 1-5 candles
                future_moves = []
                for j in range(1, 6):
                    if i + j < len(df):
                        future_price = df.iloc[i + j]['close']
                        move_pct = (current_candle['close'] - future_price) / current_candle['close'] * 100
                        future_moves.append(move_pct)
                
                max_future_move = max(future_moves) if future_moves else 0
                
                if max_future_move >= move_threshold:  # 3%+ move required
                    order_blocks.append({
                        'type': 'bearish',
                        'index': i,
                        'timestamp': current_candle['timestamp'],
                        'high': current_candle['high'],
                        'low': current_candle['low'],
                        'open': current_candle['open'],
                        'close': current_candle['close'],
                        'volume_ratio': current_candle['volume_ratio'],
                        'future_move': max_future_move,
                        'age': len(df) - i - 1
                    })
        
        # Filter out old order blocks
        valid_order_blocks = [ob for ob in order_blocks 
                             if ob['age'] <= self.max_order_block_age]
        
        return valid_order_blocks

    def detect_liquidity_grabs(self, df):
        """
        Detect ICT Liquidity Grabs
        
        Liquidity Grab = False breakout that hunts stops then reverses
        - Buy-Side: Break above highs, then reverse down
        - Sell-Side: Break below lows, then reverse up
        """
        liquidity_grabs = []
        
        for i in range(10, len(df) - 5):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_close = df.iloc[i]['close']
            
            # Look for recent highs/lows in past 10 candles
            lookback_data = df.iloc[i-10:i]
            recent_high = lookback_data['high'].max()
            recent_low = lookback_data['low'].min()
            
            # Buy-Side Liquidity Grab
            # Price breaks above recent high, then reverses down
            if current_high > recent_high * 1.001:  # Break above with 0.1% buffer
                
                # Check for reversal in next 1-5 candles
                reversal_found = False
                reversal_move = 0
                
                for j in range(1, 6):
                    if i + j < len(df):
                        future_low = df.iloc[i + j]['low']
                        move_down = (current_high - future_low) / current_high * 100
                        
                        if move_down >= self.min_liquidity_sweep:  # 1.0%+ reversal required
                            reversal_found = True
                            reversal_move = move_down
                            break
                
                if reversal_found:
                    liquidity_grabs.append({
                        'type': 'buy_side_grab',
                        'index': i,
                        'timestamp': df.iloc[i]['timestamp'],
                        'grab_level': current_high,
                        'recent_high': recent_high,
                        'reversal_move': reversal_move,
                        'volume_ratio': df.iloc[i]['volume_ratio']
                    })
            
            # Sell-Side Liquidity Grab  
            # Price breaks below recent low, then reverses up
            elif current_low < recent_low * 0.999:  # Break below with 0.1% buffer
                
                # Check for reversal in next 1-5 candles
                reversal_found = False
                reversal_move = 0
                
                for j in range(1, 6):
                    if i + j < len(df):
                        future_high = df.iloc[i + j]['high']
                        move_up = (future_high - current_low) / current_low * 100
                        
                        if move_up >= self.min_liquidity_sweep:  # 1.0%+ reversal required
                            reversal_found = True
                            reversal_move = move_up
                            break
                
                if reversal_found:
                    liquidity_grabs.append({
                        'type': 'sell_side_grab',
                        'index': i,
                        'timestamp': df.iloc[i]['timestamp'],
                        'grab_level': current_low,
                        'recent_low': recent_low,
                        'reversal_move': reversal_move,
                        'volume_ratio': df.iloc[i]['volume_ratio']
                    })
        
        return liquidity_grabs

    def detect_fair_value_gaps(self, df):
        """
        Detect Fair Value Gaps (integrate with existing FVG logic)
        
        FVG = Gap between candle highs/lows showing imbalance
        - Bullish FVG: Gap between candle 1 high and candle 3 low
        - Bearish FVG: Gap between candle 1 low and candle 3 high
        """
        fvgs = []
        
        for i in range(2, len(df)):
            candle_1 = df.iloc[i-2]
            candle_2 = df.iloc[i-1]  
            candle_3 = df.iloc[i]
            
            # Bullish FVG
            if (candle_1['high'] < candle_3['low'] and
                candle_2['high'] < candle_3['low']):
                
                gap_size = (candle_3['low'] - candle_1['high']) / candle_1['high'] * 100
                
                if gap_size >= 0.2:  # Minimum 0.2% gap for 4H
                    fvgs.append({
                        'type': 'bullish',
                        'index': i,
                        'timestamp': candle_3['timestamp'],
                        'gap_high': candle_3['low'],
                        'gap_low': candle_1['high'],
                        'gap_mid': (candle_3['low'] + candle_1['high']) / 2,
                        'gap_size': gap_size,
                        'volume_ratio': candle_3['volume_ratio']
                    })
            
            # Bearish FVG
            elif (candle_1['low'] > candle_3['high'] and
                  candle_2['low'] > candle_3['high']):
                
                gap_size = (candle_1['low'] - candle_3['high']) / candle_3['high'] * 100
                
                if gap_size >= 0.2:  # Minimum 0.2% gap for 4H
                    fvgs.append({
                        'type': 'bearish',
                        'index': i,
                        'timestamp': candle_3['timestamp'],
                        'gap_high': candle_1['low'],
                        'gap_low': candle_3['high'],
                        'gap_mid': (candle_1['low'] + candle_3['high']) / 2,
                        'gap_size': gap_size,
                        'volume_ratio': candle_3['volume_ratio']
                    })
        
        return fvgs

    def identify_market_structure(self, df):
        """Identify key market structure levels for ICT analysis"""
        structure = {
            'recent_highs': [],
            'recent_lows': [],
            'equal_highs': [],
            'equal_lows': [],
            'swing_highs': [],
            'swing_lows': []
        }
        
        # Find significant highs and lows (swing points)
        for i in range(5, len(df) - 5):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for swing high
            window_highs = df.iloc[i-5:i+6]['high']
            if current_high == window_highs.max():
                structure['swing_highs'].append({
                    'price': current_high,
                    'index': i,
                    'timestamp': df.iloc[i]['timestamp']
                })
            
            # Check for swing low  
            window_lows = df.iloc[i-5:i+6]['low']
            if current_low == window_lows.min():
                structure['swing_lows'].append({
                    'price': current_low,
                    'index': i,
                    'timestamp': df.iloc[i]['timestamp']
                })
        
        # Identify equal highs/lows (liquidity zones)
        structure['equal_highs'] = self.find_equal_levels(structure['swing_highs'], tolerance=0.5)
        structure['equal_lows'] = self.find_equal_levels(structure['swing_lows'], tolerance=0.5)
        
        # Get recent significant levels
        if structure['swing_highs']:
            structure['recent_highs'] = sorted(structure['swing_highs'], 
                                             key=lambda x: x['index'], reverse=True)[:5]
        if structure['swing_lows']:
            structure['recent_lows'] = sorted(structure['swing_lows'], 
                                            key=lambda x: x['index'], reverse=True)[:5]
        
        return structure

    def find_equal_levels(self, levels, tolerance=0.5):
        """Find equal highs/lows within tolerance percentage"""
        equal_groups = []
        
        for i, level1 in enumerate(levels):
            group = [level1]
            for j, level2 in enumerate(levels[i+1:], i+1):
                price_diff = abs(level1['price'] - level2['price']) / level1['price'] * 100
                if price_diff <= tolerance:
                    group.append(level2)
            
            if len(group) >= 2:  # At least 2 equal levels
                equal_groups.append(group)
        
        return equal_groups

    def find_ict_confluences(self, df, order_blocks, liquidity_grabs, fvgs):
        """
        Find high-probability ICT confluences using PROPER ICT PRINCIPLES
        
        ICT Target Logic:
        1. Entry: Order Block level (institutional order placement)
        2. Stop: Beyond Order Block invalidation 
        3. Targets: Next opposing Order Blocks, FVGs, Liquidity levels
        4. Structure: Previous highs/lows, equal highs/lows
        """
        confluences = []
        current_price = df.iloc[-1]['close']
        
        # Identify key market structure levels for proper ICT analysis
        structure_levels = self.identify_market_structure(df)
        
        for ob in order_blocks:
            setup_score = 50  # Base score
            confluence_factors = []
            
            # Order Block scoring
            if ob['volume_ratio'] >= 3.0:
                setup_score += 25
                confluence_factors.append("Institutional Volume Order Block")
            elif ob['volume_ratio'] >= 2.0:
                setup_score += 15
                confluence_factors.append("High Volume Order Block")
            
            # Age factor (fresher is better)
            if ob['age'] <= 5:
                setup_score += 20
                confluence_factors.append("Fresh Order Block")
            elif ob['age'] <= 10:
                setup_score += 15
                confluence_factors.append("Recent Order Block")
            
            # Future move strength
            if ob['future_move'] >= 8.0:
                setup_score += 25
                confluence_factors.append("Strong Institutional Validation")
            elif ob['future_move'] >= 5.0:
                setup_score += 20
                confluence_factors.append("Good Institutional Validation")
            elif ob['future_move'] >= 3.0:
                setup_score += 10
                confluence_factors.append("Moderate Validation")
            
            # Check for liquidity grab confluence
            liquidity_confluence = self.analyze_liquidity_confluence(ob, liquidity_grabs)
            if liquidity_confluence:
                setup_score += liquidity_confluence['score']
                confluence_factors.extend(liquidity_confluence['factors'])
            
            # Check for FVG confluence
            fvg_confluence = self.analyze_fvg_confluence(ob, fvgs)
            if fvg_confluence:
                setup_score += fvg_confluence['score']
                confluence_factors.extend(fvg_confluence['factors'])
            
            # Current price proximity to Order Block
            distance_analysis = self.analyze_price_distance(ob, current_price)
            setup_score += distance_analysis['score']
            confluence_factors.extend(distance_analysis['factors'])
            
            # ICT TARGET CALCULATION - Using Market Structure
            targets_analysis = self.calculate_ict_targets(ob, df, structure_levels, fvgs, order_blocks)
            
            # Only include high-quality setups
            if setup_score >= self.min_quality_score:
                confluences.append({
                    'order_block': ob,
                    'quality_score': min(setup_score, 100),  # Cap at 100
                    'confluence_factors': confluence_factors,
                    'entry_price': targets_analysis['entry'],
                    'stop_loss': targets_analysis['stop'],
                    'targets': targets_analysis['targets'],
                    'target_reasoning': targets_analysis['reasoning'],
                    'risk_reward': targets_analysis['risk_reward'],
                    'current_price': current_price,
                    'distance_pct': abs(current_price - targets_analysis['entry']) / current_price * 100,
                    'market_context': targets_analysis['context']
                })
        
        return confluences

    def analyze_liquidity_confluence(self, order_block, liquidity_grabs):
        """Analyze liquidity grab confluence for order block"""
        confluence = {'score': 0, 'factors': []}
        
        for lg in liquidity_grabs:
            time_diff = abs((lg['timestamp'] - order_block['timestamp']).total_seconds() / 3600)
            
            if time_diff <= 48:  # Within 48 hours
                if ((order_block['type'] == 'bullish' and lg['type'] == 'sell_side_grab') or
                    (order_block['type'] == 'bearish' and lg['type'] == 'buy_side_grab')):
                    
                    if lg['reversal_move'] >= 3.0:
                        confluence['score'] += 25
                        confluence['factors'].append(f"Strong Liquidity Hunt ({lg['reversal_move']:.1f}%)")
                    elif lg['reversal_move'] >= 1.5:
                        confluence['score'] += 15
                        confluence['factors'].append(f"Liquidity Grab Confluence ({lg['reversal_move']:.1f}%)")
        
        return confluence if confluence['score'] > 0 else None

    def analyze_fvg_confluence(self, order_block, fvgs):
        """Analyze Fair Value Gap confluence"""
        confluence = {'score': 0, 'factors': []}
        
        for fvg in fvgs:
            if order_block['type'] == fvg['type']:  # Same direction
                # Check if FVG overlaps with Order Block
                ob_range = [order_block['low'], order_block['high']]
                fvg_range = [fvg['gap_low'], fvg['gap_high']]
                
                if (ob_range[0] <= fvg_range[1] and ob_range[1] >= fvg_range[0]):
                    if fvg['gap_size'] >= 1.0:
                        confluence['score'] += 20
                        confluence['factors'].append(f"Significant FVG Confluence ({fvg['gap_size']:.1f}%)")
                    else:
                        confluence['score'] += 10
                        confluence['factors'].append(f"FVG Confluence ({fvg['gap_size']:.1f}%)")
        
        return confluence if confluence['score'] > 0 else None

    def analyze_price_distance(self, order_block, current_price):
        """Analyze current price distance to order block"""
        analysis = {'score': 0, 'factors': []}
        
        if order_block['type'] == 'bullish':
            ob_center = (order_block['low'] + order_block['high']) / 2
        else:
            ob_center = (order_block['low'] + order_block['high']) / 2
        
        distance_pct = abs(current_price - ob_center) / current_price * 100
        
        if distance_pct <= 0.5:
            analysis['score'] += 30
            analysis['factors'].append("Price at Order Block")
        elif distance_pct <= 1.0:
            analysis['score'] += 25
            analysis['factors'].append("Price near Order Block")
        elif distance_pct <= 2.0:
            analysis['score'] += 15
            analysis['factors'].append("Price approaching Order Block")
        elif distance_pct <= 5.0:
            analysis['score'] += 5
            analysis['factors'].append("Price within range")
        
        return analysis

    def calculate_ict_targets(self, order_block, df, structure_levels, fvgs, all_order_blocks):
        """Calculate ICT targets using proper ICT methodology"""
        
        if order_block['type'] == 'bullish':
            entry_price = order_block['low']  # Enter at Order Block low
            stop_loss = order_block['low'] - (order_block['high'] - order_block['low']) * 0.2  # 20% beyond OB
            
            # Find BULLISH targets above current price
            targets = self.find_bullish_targets(entry_price, structure_levels, fvgs, all_order_blocks, df)
            
        else:  # Bearish Order Block
            entry_price = order_block['high']  # Enter at Order Block high
            stop_loss = order_block['high'] + (order_block['high'] - order_block['low']) * 0.2  # 20% beyond OB
            
            # Find BEARISH targets below current price
            targets = self.find_bearish_targets(entry_price, structure_levels, fvgs, all_order_blocks, df)
        
        # Ensure we always return the required keys
        if not targets.get('targets', []):
            # Fallback targets if no structural levels found
            if order_block['type'] == 'bullish':
                targets = {
                    'targets': [
                        {'price': entry_price * 1.02, 'type': 'Conservative Target'},
                        {'price': entry_price * 1.05, 'type': 'Moderate Target'},
                        {'price': entry_price * 1.08, 'type': 'Extended Target'}
                    ],
                    'reasoning': [
                        f"Conservative target @ ${entry_price * 1.02:.4f} (2% gain)",
                        f"Moderate target @ ${entry_price * 1.05:.4f} (5% gain)", 
                        f"Extended target @ ${entry_price * 1.08:.4f} (8% gain)"
                    ],
                    'context': {'note': 'No structural levels found - using percentage targets'}
                }
            else:
                targets = {
                    'targets': [
                        {'price': entry_price * 0.98, 'type': 'Conservative Target'},
                        {'price': entry_price * 0.95, 'type': 'Moderate Target'},
                        {'price': entry_price * 0.92, 'type': 'Extended Target'}
                    ],
                    'reasoning': [
                        f"Conservative target @ ${entry_price * 0.98:.4f} (2% decline)",
                        f"Moderate target @ ${entry_price * 0.95:.4f} (5% decline)",
                        f"Extended target @ ${entry_price * 0.92:.4f} (8% decline)"
                    ],
                    'context': {'note': 'No structural levels found - using percentage targets'}
                }
        
        # Ensure reasoning always exists
        if 'reasoning' not in targets or not targets['reasoning']:
            targets['reasoning'] = [f"Target analysis for {order_block['type']} order block"]
        
        # Calculate risk/reward
        risk_amount = abs(entry_price - stop_loss)
        target_rewards = [abs(target['price'] - entry_price) for target in targets['targets']]
        risk_rewards = [reward / risk_amount if risk_amount > 0 else 0 for reward in target_rewards]
        
        return {
            'entry': entry_price,
            'stop': stop_loss,
            'targets': [t['price'] for t in targets['targets']],
            'reasoning': targets['reasoning'],
            'risk_reward': risk_rewards[0] if risk_rewards else 0,
            'context': targets['context']
        }

    def find_bullish_targets(self, entry_price, structure, fvgs, order_blocks, df):
        """Find bullish targets using ICT methodology"""
        targets = []
        reasoning = []
        context = {}
        
        try:
            current_price = df.iloc[-1]['close']
            
            # 1. Look for BEARISH Order Blocks above (opposing institutional levels)
            bearish_obs = [ob for ob in order_blocks if ob['type'] == 'bearish' and ob['high'] > entry_price]
            bearish_obs.sort(key=lambda x: x['high'])  # Closest first
            
            for ob in bearish_obs[:3]:  # Top 3 closest
                targets.append({
                    'price': ob['high'],
                    'type': 'Bearish Order Block',
                    'distance': abs(ob['high'] - entry_price) / entry_price * 100
                })
                reasoning.append(f"Bearish OB @ ${ob['high']:.4f} (institutional resistance)")
            
            # 2. Look for Equal Highs above (liquidity targets)
            for equal_group in structure.get('equal_highs', []):
                avg_price = sum([level['price'] for level in equal_group]) / len(equal_group)
                if avg_price > entry_price:
                    targets.append({
                        'price': avg_price,
                        'type': 'Equal Highs Liquidity',
                        'distance': abs(avg_price - entry_price) / entry_price * 100
                    })
                    reasoning.append(f"Equal Highs @ ${avg_price:.4f} ({len(equal_group)} touches - liquidity pool)")
            
            # 3. Look for Bearish FVGs above (imbalance fills)
            bearish_fvgs = [fvg for fvg in fvgs if fvg['type'] == 'bearish' and fvg['gap_mid'] > entry_price]
            bearish_fvgs.sort(key=lambda x: x['gap_mid'])  # Closest first
            
            for fvg in bearish_fvgs[:2]:  # Top 2 closest
                targets.append({
                    'price': fvg['gap_mid'],
                    'type': 'Bearish FVG',
                    'distance': abs(fvg['gap_mid'] - entry_price) / entry_price * 100
                })
                reasoning.append(f"Bearish FVG @ ${fvg['gap_mid']:.4f} ({fvg['gap_size']:.1f}% gap)")
            
            # 4. Major swing highs
            for swing in structure.get('recent_highs', [])[:3]:
                if swing['price'] > entry_price:
                    targets.append({
                        'price': swing['price'],
                        'type': 'Swing High',
                        'distance': abs(swing['price'] - entry_price) / entry_price * 100
                    })
                    reasoning.append(f"Swing High @ ${swing['price']:.4f} (previous structure)")
            
            # Sort targets by distance and take best 3
            targets.sort(key=lambda x: x['distance'])
            final_targets = targets[:3]
            
            # If we don't have 3 targets, add percentage-based targets
            while len(final_targets) < 3:
                if len(final_targets) == 0:
                    target_price = entry_price * 1.02  # 2%
                    reasoning.append(f"Conservative target @ ${target_price:.4f} (2% gain)")
                elif len(final_targets) == 1:
                    target_price = entry_price * 1.05  # 5%
                    reasoning.append(f"Moderate target @ ${target_price:.4f} (5% gain)")
                else:
                    target_price = entry_price * 1.08  # 8%
                    reasoning.append(f"Extended target @ ${target_price:.4f} (8% gain)")
                
                final_targets.append({
                    'price': target_price,
                    'type': 'Percentage Target',
                    'distance': abs(target_price - entry_price) / entry_price * 100
                })
            
            # Context for display
            context = {
                'bearish_obs_above': len(bearish_obs),
                'equal_highs_above': len([g for g in structure.get('equal_highs', []) 
                                        if sum([l['price'] for l in g])/len(g) > entry_price]),
                'bearish_fvgs_above': len(bearish_fvgs),
                'swing_highs_above': len([s for s in structure.get('recent_highs', []) if s['price'] > entry_price])
            }
            
        except Exception as e:
            # Fallback if anything fails
            final_targets = [
                {'price': entry_price * 1.02, 'type': 'Conservative Target', 'distance': 2.0},
                {'price': entry_price * 1.05, 'type': 'Moderate Target', 'distance': 5.0},
                {'price': entry_price * 1.08, 'type': 'Extended Target', 'distance': 8.0}
            ]
            reasoning = [
                f"Conservative target @ ${entry_price * 1.02:.4f} (2% gain)",
                f"Moderate target @ ${entry_price * 1.05:.4f} (5% gain)",
                f"Extended target @ ${entry_price * 1.08:.4f} (8% gain)"
            ]
            context = {'note': f'Fallback targets due to error: {str(e)[:30]}'}
        
        return {
            'targets': final_targets,
            'reasoning': reasoning[:3],  # Top 3 reasons
            'context': context
        }

    def find_bearish_targets(self, entry_price, structure, fvgs, order_blocks, df):
        """Find bearish targets using ICT methodology"""
        targets = []
        reasoning = []
        context = {}
        
        try:
            # 1. Look for BULLISH Order Blocks below
            bullish_obs = [ob for ob in order_blocks if ob['type'] == 'bullish' and ob['low'] < entry_price]
            bullish_obs.sort(key=lambda x: x['low'], reverse=True)  # Closest first
            
            for ob in bullish_obs[:3]:
                targets.append({
                    'price': ob['low'],
                    'type': 'Bullish Order Block',
                    'distance': abs(entry_price - ob['low']) / entry_price * 100
                })
                reasoning.append(f"Bullish OB @ ${ob['low']:.4f} (institutional support)")
            
            # 2. Look for Equal Lows below
            for equal_group in structure.get('equal_lows', []):
                avg_price = sum([level['price'] for level in equal_group]) / len(equal_group)
                if avg_price < entry_price:
                    targets.append({
                        'price': avg_price,
                        'type': 'Equal Lows Liquidity',
                        'distance': abs(entry_price - avg_price) / entry_price * 100
                    })
                    reasoning.append(f"Equal Lows @ ${avg_price:.4f} ({len(equal_group)} touches - liquidity pool)")
            
            # 3. Look for Bullish FVGs below
            bullish_fvgs = [fvg for fvg in fvgs if fvg['type'] == 'bullish' and fvg['gap_mid'] < entry_price]
            bullish_fvgs.sort(key=lambda x: x['gap_mid'], reverse=True)  # Closest first
            
            for fvg in bullish_fvgs[:2]:
                targets.append({
                    'price': fvg['gap_mid'],
                    'type': 'Bullish FVG',
                    'distance': abs(entry_price - fvg['gap_mid']) / entry_price * 100
                })
                reasoning.append(f"Bullish FVG @ ${fvg['gap_mid']:.4f} ({fvg['gap_size']:.1f}% gap)")
            
            # 4. Major swing lows
            for swing in structure.get('recent_lows', [])[:3]:
                if swing['price'] < entry_price:
                    targets.append({
                        'price': swing['price'],
                        'type': 'Swing Low',
                        'distance': abs(entry_price - swing['price']) / entry_price * 100
                    })
                    reasoning.append(f"Swing Low @ ${swing['price']:.4f} (previous structure)")
            
            # Sort and finalize
            targets.sort(key=lambda x: x['distance'])
            final_targets = targets[:3]
            
            # Add percentage targets if needed
            while len(final_targets) < 3:
                if len(final_targets) == 0:
                    target_price = entry_price * 0.98  # 2% down
                    reasoning.append(f"Conservative target @ ${target_price:.4f} (2% decline)")
                elif len(final_targets) == 1:
                    target_price = entry_price * 0.95  # 5% down
                    reasoning.append(f"Moderate target @ ${target_price:.4f} (5% decline)")
                else:
                    target_price = entry_price * 0.92  # 8% down
                    reasoning.append(f"Extended target @ ${target_price:.4f} (8% decline)")
                
                final_targets.append({
                    'price': target_price,
                    'type': 'Percentage Target',
                    'distance': abs(entry_price - target_price) / entry_price * 100
                })
            
            context = {
                'bullish_obs_below': len(bullish_obs),
                'equal_lows_below': len([g for g in structure.get('equal_lows', []) 
                                       if sum([l['price'] for l in g])/len(g) < entry_price]),
                'bullish_fvgs_below': len(bullish_fvgs),
                'swing_lows_below': len([s for s in structure.get('recent_lows', []) if s['price'] < entry_price])
            }
            
        except Exception as e:
            # Fallback if anything fails
            final_targets = [
                {'price': entry_price * 0.98, 'type': 'Conservative Target', 'distance': 2.0},
                {'price': entry_price * 0.95, 'type': 'Moderate Target', 'distance': 5.0},
                {'price': entry_price * 0.92, 'type': 'Extended Target', 'distance': 8.0}
            ]
            reasoning = [
                f"Conservative target @ ${entry_price * 0.98:.4f} (2% decline)",
                f"Moderate target @ ${entry_price * 0.95:.4f} (5% decline)",
                f"Extended target @ ${entry_price * 0.92:.4f} (8% decline)"
            ]
            context = {'note': f'Fallback targets due to error: {str(e)[:30]}'}
        
        return {
            'targets': final_targets,
            'reasoning': reasoning[:3],
            'context': context
        }

    def analyze_symbol(self, symbol):
        """Analyze single symbol for ICT setups"""
        try:
            print(f"🔍 {symbol.replace('/USDT', '')}...", end=' ')
            
            df = self.get_4h_data(symbol)
            if df is None or len(df) < 50:
                print("Insufficient data")
                return []
            
            # Detect ICT components
            order_blocks = self.detect_order_blocks(df)
            liquidity_grabs = self.detect_liquidity_grabs(df)
            fvgs = self.detect_fair_value_gaps(df)
            
            if not order_blocks:
                print("No order blocks found")
                return []
            
            # Find confluences
            confluences = self.find_ict_confluences(df, order_blocks, liquidity_grabs, fvgs)
            
            if not confluences:
                print("No quality confluences")
                return []
            
            # Format results
            results = []
            for conf in confluences:
                ob = conf['order_block']
                setup = {
                    'symbol': symbol,
                    'type': 'ICT_' + ob['type'].upper(),
                    'direction': 'BULLISH' if ob['type'] == 'bullish' else 'BEARISH',
                    'quality_score': conf['quality_score'],
                    'current_price': conf['current_price'],
                    'entry_price': conf['entry_price'],
                    'stop_loss': conf['stop_loss'],
                    'targets': conf['targets'],
                    'target_reasoning': conf['target_reasoning'],
                    'risk_reward': conf['risk_reward'],
                    'distance_pct': conf['distance_pct'],
                    'confluence_factors': conf['confluence_factors'],
                    'order_block_age': ob['age'],
                    'validation_move': ob['future_move'],
                    'volume_ratio': ob['volume_ratio'],
                    'setup_time': ob['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'market_context': conf['market_context']
                }
                results.append(setup)
            
            print(f"Found {len(results)} ICT setups")
            return results
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def scan_all_symbols(self):
        """Scan all symbols for ICT setups"""
        print("🔥 ICT SCANNER - 4H ORDER BLOCKS & LIQUIDITY GRABS")
        print("=" * 80)
        print("📈 Scanning for institutional-level ICT setups...")
        print("🎯 Strategy: Order Blocks + Liquidity Grabs + FVG confluence")
        print("💰 Filtering: $100K+ daily volume, top 100 liquid pairs")
        print()
        
        symbols = self.get_symbols()
        all_setups = []
        
        print(f"\n🔍 Analyzing {len(symbols)} symbols for ICT patterns...")
        print()
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"🔍 {symbol.replace('/USDT', '')}...", end=' ')
                patterns = self.analyze_symbol(symbol)
                all_setups.extend(patterns)
                
                if i % 10 == 0:
                    print(f"\n📊 Progress: {i}/{len(symbols)} symbols analyzed...")
                
                time.sleep(0.1)  # Rate limiting
                
            except KeyError as e:
                print(f"KeyError: {e}")
                continue
            except Exception as e:
                print(f"Error: {str(e)[:50]}...")
                continue
        
        return all_setups

    def display_results(self, setups):
        """Display top ICT scan results - LIMIT TO TOP 15 ELITE SETUPS"""
        if not setups:
            print(f"\n📊 SCAN RESULTS:")
            print("Total scanned: 100")
            print("ICT setups found: 0")
            print("=" * 80)
            print("🔥 FOUND 0 ELITE ICT SETUPS")
            print("=" * 80)
            print("❌ No elite ICT confluences found")
            print("💡 Try running during London/NY sessions for better institutional flow")
            return
        
        # Sort by quality score and distance (prioritize actionable setups)
        def setup_priority(setup):
            # Prioritize high quality + close to entry price
            quality_bonus = setup['quality_score']
            distance_penalty = setup['distance_pct'] * 2  # Penalize distant setups
            return quality_bonus - distance_penalty
        
        setups.sort(key=setup_priority, reverse=True)
        
        # LIMIT TO TOP 15 ELITE SETUPS ONLY
        elite_setups = setups[:15]
        
        print(f"\n📊 SCAN RESULTS:")
        print(f"Total scanned: 100")
        print(f"Elite ICT setups found: {len(elite_setups)} (from {len(setups)} total)")
        print("=" * 80)
        print(f"🔥 TOP {len(elite_setups)} ELITE ICT ORDER BLOCK SETUPS")
        print("=" * 80)
        print("📊 Filtered for: 85+ quality score, 3%+ validation moves, 2x+ volume")
        print("🎯 Focus: Institutional-grade confluences only")
        print()
        
        for i, setup in enumerate(elite_setups, 1):
            direction_icon = "📈" if setup['direction'] == 'BULLISH' else "📉"
            
            # Determine action based on distance
            if setup['distance_pct'] <= 1.0:
                action_icon = "🎯 ACTIONABLE"
                action_color = "🟢"
            elif setup['distance_pct'] <= 3.0:
                action_icon = "⏳ WATCH"
                action_color = "🟡"  
            else:
                action_icon = "📊 TRACK"
                action_color = "🔵"
            
            print(f"{i}. {action_color} {direction_icon} {setup['symbol'].replace('/USDT', '')} | "
                  f"Score: {setup['quality_score']:.0f}/100 | {action_icon}")
            
            print(f"   Current: ${setup['current_price']:.4f} | "
                  f"Entry: ${setup['entry_price']:.4f} | "
                  f"Distance: {setup['distance_pct']:.1f}% | "
                  f"R:R: {setup['risk_reward']:.1f}:1")
            
            print(f"   Stop: ${setup['stop_loss']:.4f} | "
                  f"Age: {setup['order_block_age']}h | "
                  f"Vol: {setup['volume_ratio']:.1f}x | "
                  f"Move: {setup['validation_move']:.1f}%")
            
            print(f"   Targets: T1: ${setup['targets'][0]:.4f} | "
                  f"T2: ${setup['targets'][1]:.4f} | "
                  f"T3: ${setup['targets'][2]:.4f}")
            
            # Show ICT target reasoning  
            if 'target_reasoning' in setup and setup['target_reasoning']:
                print(f"   🎯 ICT Target Logic:")
                for reason in setup['target_reasoning']:
                    print(f"   • {reason}")
            
            # Show market context
            if 'market_context' in setup and setup['market_context']:
                context = setup['market_context']
                context_items = []
                if setup['direction'] == 'BULLISH':
                    if context.get('bearish_obs_above', 0) > 0:
                        context_items.append(f"{context['bearish_obs_above']} bearish OBs above")
                    if context.get('equal_highs_above', 0) > 0:
                        context_items.append(f"{context['equal_highs_above']} equal highs above")
                    if context.get('bearish_fvgs_above', 0) > 0:
                        context_items.append(f"{context['bearish_fvgs_above']} bearish FVGs above")
                else:
                    if context.get('bullish_obs_below', 0) > 0:
                        context_items.append(f"{context['bullish_obs_below']} bullish OBs below")
                    if context.get('equal_lows_below', 0) > 0:
                        context_items.append(f"{context['equal_lows_below']} equal lows below")
                    if context.get('bullish_fvgs_below', 0) > 0:
                        context_items.append(f"{context['bullish_fvgs_below']} bullish FVGs below")
                
                if context_items:
                    print(f"   📊 Market Structure: {' | '.join(context_items)}")
                elif 'note' in context:
                    print(f"   📊 Market Structure: {context['note']}")
            
            # Show top 3 confluence factors only
            top_factors = setup['confluence_factors'][:3]
            print(f"   🎯 Key Confluences: {' • '.join(top_factors)}")
            
            # Simplified trading action
            if setup['distance_pct'] <= 1.0:
                print(f"   💡 ACTION: ENTER NOW at Order Block level")
                print(f"   ⏰ TIMEFRAME: 1-7 days (4H swing)")
            elif setup['distance_pct'] <= 3.0:
                print(f"   💡 ACTION: SET ALERT at ${setup['entry_price']:.4f}")
                print(f"   📊 MONITOR: High-probability institutional level")
            else:
                print(f"   💡 ACTION: LONG-TERM TRACKING")
                print(f"   📈 NOTE: Valid until Order Block mitigated")
            
            print("-" * 78)

        # Summary statistics
        actionable = len([s for s in elite_setups if s['distance_pct'] <= 1.0])
        watchlist = len([s for s in elite_setups if 1.0 < s['distance_pct'] <= 3.0])
        tracking = len([s for s in elite_setups if s['distance_pct'] > 3.0])
        
        print(f"\n📊 ELITE SETUP BREAKDOWN:")
        print(f"🟢 ACTIONABLE (≤1% away): {actionable} setups")
        print(f"🟡 WATCHLIST (1-3% away): {watchlist} setups") 
        print(f"🔵 TRACKING (>3% away): {tracking} setups")
        print(f"\n💡 Focus on ACTIONABLE setups for immediate entries!")
        print(f"📱 Set alerts for WATCHLIST setups approaching entry zones")
        print(f"📊 Monitor TRACKING setups for future opportunities")

def main():
    """Main execution function"""
    try:
        scanner = ICTScanner()
        setups = scanner.scan_all_symbols()
        scanner.display_results(setups)
        
    except KeyboardInterrupt:
        print("\n❌ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan error: {e}")

if __name__ == "__main__":
    main()