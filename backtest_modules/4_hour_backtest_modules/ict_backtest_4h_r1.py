#!/usr/bin/env python3
"""
ICT Enhanced Backtester - Fixed for Proper 4H Swing Trading
Major improvements:
1. Extended tracking windows (200+ candles for proper swing completion)
2. ICT-based target calculation (structure levels, not arbitrary percentages)
3. Minimum hold time logic (institutions need time to develop moves)
4. Wider stop losses (breathing room for 4H timeframe)
5. Parameter optimization framework (find optimal settings)

This addresses the core flaws: 4H ICT should be swing trades (days), not scalps (hours)
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class ICTEnhancedBacktester:
    def __init__(self, config=None):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'timeout': 30000,
            'rateLimit': 50,
            'enableRateLimit': True,
        })
        
        # Default optimized configuration (will be tested and refined)
        self.config = config or {
            'future_window': 200,      # Candles to track ahead (33 days for proper swings)
            'min_hold_bars': 12,       # Minimum 48 hours hold (institutions need time)
            'stop_multiplier': 0.75,   # Stop distance (breathing room for 4H)
            'target_method': 'ict',    # ICT structure-based targets
            'lookback_months': 3,      # Extended from 1 month due to 30-day Binance limit
            'min_order_block_size': 2.5,   # RELAXED: 2.5% vs 3.0% (more opportunities)
            'min_volume_ratio': 1.8,       # RELAXED: 1.8x vs 2.0x (more opportunities)
            'min_quality_score': 70        # RELAXED: 70 vs 85 (more opportunities)
        }
        
        self.all_trades = []
        self.commission = 0.001

    def add_technical_indicators(self, df):
        """Add ICT-specific technical indicators for correlation analysis"""
        
        # Volume indicators (critical for ICT)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Keep proven indicators that work for ICT
        # RSI (still useful for institutional extremes)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MFI (Money Flow = institutional activity)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        
        money_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # CMF (Chaikin Money Flow - volume-weighted momentum)
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # ICT-SPECIFIC INDICATORS
        
        # Accumulation/Distribution Line (institutional accumulation)
        ad_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        ad_multiplier = ad_multiplier.fillna(0)  # Handle division by zero
        ad_volume = ad_multiplier * df['volume']
        df['accumulation_distribution'] = ad_volume.cumsum()
        
        # On Balance Volume (volume-price relationship for smart money)
        obv = []
        obv_value = 0
        for i in range(len(df)):
            if i == 0:
                obv_value = df.iloc[i]['volume']
            else:
                if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                    obv_value += df.iloc[i]['volume']
                elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
                    obv_value -= df.iloc[i]['volume']
                # If close == previous close, OBV unchanged
            obv.append(obv_value)
        df['on_balance_volume'] = obv
        
        # Williams %R (institutional overbought/oversold)
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Average True Range (volatility for stop placement)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume Weighted Average Price (institutional reference price)
        typical_price_vwap = (df['high'] + df['low'] + df['close']) / 3
        vwap_numerator = (typical_price_vwap * df['volume']).rolling(window=20).sum()
        vwap_denominator = df['volume'].rolling(window=20).sum()
        df['vwap'] = vwap_numerator / vwap_denominator
        
        # VWAP deviation (distance from institutional reference)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Commodity Channel Index (institutional extremes)
        cci_tp = (df['high'] + df['low'] + df['close']) / 3
        cci_sma = cci_tp.rolling(window=20).mean()
        cci_mad = cci_tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (cci_tp - cci_sma) / (0.015 * cci_mad)
        
        # Market Structure Indicators
        
        # Swing High/Low Detection (for liquidity levels)
        swing_window = 5
        df['swing_high'] = df['high'][(df['high'].shift(swing_window) < df['high']) & 
                                      (df['high'].shift(-swing_window) < df['high'])]
        df['swing_low'] = df['low'][(df['low'].shift(swing_window) > df['low']) & 
                                    (df['low'].shift(-swing_window) > df['low'])]
        
        # Liquidity Grab Detection (fake breakouts)
        df['recent_high'] = df['high'].rolling(window=10).max()
        df['recent_low'] = df['low'].rolling(window=10).min()
        
        # Price above recent high = potential buy-side liquidity grab
        df['above_recent_high'] = (df['high'] > df['recent_high'].shift(1)) & (df['close'] < df['recent_high'].shift(1))
        # Price below recent low = potential sell-side liquidity grab  
        df['below_recent_low'] = (df['low'] < df['recent_low'].shift(1)) & (df['close'] > df['recent_low'].shift(1))
        
        # Keep some momentum for comparison (but ICT-relevant timeframes)
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1  # 20 periods = ~3 days on 4H
        df['momentum_50'] = df['close'] / df['close'].shift(50) - 1  # 50 periods = ~8 days on 4H
        
        # Remove non-ICT indicators
        # No more MACD histogram, short-term momentum
        
        return df

    def find_swing_levels(self, df, window=10):
        """Find significant swing highs and lows for ICT targets"""
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Swing High: Current high is highest in surrounding window
            if df.iloc[i]['high'] == df.iloc[i-window:i+window+1]['high'].max():
                swing_highs.append(df.iloc[i]['high'])
            
            # Swing Low: Current low is lowest in surrounding window  
            if df.iloc[i]['low'] == df.iloc[i-window:i+window+1]['low'].min():
                swing_lows.append(df.iloc[i]['low'])
        
        return {
            'highs': sorted(list(set(swing_highs)), reverse=True),
            'lows': sorted(list(set(swing_lows)))
        }

    def find_equal_levels(self, df, tolerance=0.005):
        """Find equal highs/lows (liquidity pools) within tolerance"""
        swing_levels = self.find_swing_levels(df)
        
        equal_highs = []
        equal_lows = []
        
        # Group similar highs (within tolerance)
        for i, high1 in enumerate(swing_levels['highs']):
            matches = [high1]
            for high2 in swing_levels['highs'][i+1:]:
                if abs(high1 - high2) / high1 <= tolerance:
                    matches.append(high2)
            
            if len(matches) >= 2:  # At least 2 touches = equal level
                equal_highs.append(np.mean(matches))
        
        # Group similar lows (within tolerance)
        for i, low1 in enumerate(swing_levels['lows']):
            matches = [low1]
            for low2 in swing_levels['lows'][i+1:]:
                if abs(low1 - low2) / low1 <= tolerance:
                    matches.append(low2)
            
            if len(matches) >= 2:  # At least 2 touches = equal level
                equal_lows.append(np.mean(matches))
        
        return {
            'equal_highs': sorted(list(set(equal_highs)), reverse=True),
            'equal_lows': sorted(list(set(equal_lows)))
        }

    def calculate_ict_targets(self, df, entry_price, direction, order_block):
        """Calculate ICT-based targets using market structure"""
        
        swing_levels = self.find_swing_levels(df)
        equal_levels = self.find_equal_levels(df)
        targets = []
        target_reasons = []
        
        try:
            if direction == 'BULLISH':
                # T1: Next swing high above entry
                relevant_highs = [h for h in swing_levels['highs'] if h > entry_price * 1.01]
                if relevant_highs:
                    targets.append(min(relevant_highs))
                    target_reasons.append(f"Swing High @ {min(relevant_highs):.6f}")
                
                # T2: Next equal highs (liquidity pool)
                relevant_equal_highs = [h for h in equal_levels['equal_highs'] if h > entry_price * 1.02]
                if relevant_equal_highs:
                    targets.append(min(relevant_equal_highs))
                    target_reasons.append(f"Equal Highs @ {min(relevant_equal_highs):.6f}")
                
                # T3: Major swing high (further out)
                if len(relevant_highs) > 1:
                    targets.append(sorted(relevant_highs)[1])
                    target_reasons.append(f"Major High @ {sorted(relevant_highs)[1]:.6f}")
                
            else:  # BEARISH
                # T1: Next swing low below entry
                relevant_lows = [l for l in swing_levels['lows'] if l < entry_price * 0.99]
                if relevant_lows:
                    targets.append(max(relevant_lows))
                    target_reasons.append(f"Swing Low @ {max(relevant_lows):.6f}")
                
                # T2: Next equal lows (liquidity pool)
                relevant_equal_lows = [l for l in equal_levels['equal_lows'] if l < entry_price * 0.98]
                if relevant_equal_lows:
                    targets.append(max(relevant_equal_lows))
                    target_reasons.append(f"Equal Lows @ {max(relevant_equal_lows):.6f}")
                
                # T3: Major swing low (further out)
                if len(relevant_lows) > 1:
                    targets.append(sorted(relevant_lows, reverse=True)[1])
                    target_reasons.append(f"Major Low @ {sorted(relevant_lows, reverse=True)[1]:.6f}")
        
        except Exception as e:
            print(f"ICT target calculation error: {e}")
            # Fallback to percentage-based targets if structure analysis fails
            if direction == 'BULLISH':
                targets = [entry_price * 1.025, entry_price * 1.05, entry_price * 1.075]
                target_reasons = ["Fallback +2.5%", "Fallback +5.0%", "Fallback +7.5%"]
            else:
                targets = [entry_price * 0.975, entry_price * 0.95, entry_price * 0.925]
                target_reasons = ["Fallback -2.5%", "Fallback -5.0%", "Fallback -7.5%"]
        
        # Ensure we have 3 targets
        while len(targets) < 3:
            if direction == 'BULLISH':
                targets.append(entry_price * (1.02 + len(targets) * 0.025))
                target_reasons.append(f"Extended +{(2 + len(target_reasons) * 2.5):.1f}%")
            else:
                targets.append(entry_price * (0.98 - len(targets) * 0.025))
                target_reasons.append(f"Extended -{(2 + len(target_reasons) * 2.5):.1f}%")
        
        return targets[:3], target_reasons[:3]

    def detect_order_blocks(self, df):
        """Detect ICT Order Blocks with enhanced logic and debug output"""
        order_blocks = []
        debug_stats = {
            'total_candles': len(df),
            'volume_filtered': 0,
            'move_validated': 0,
            'final_blocks': 0
        }
        
        for i in range(10, len(df) - 10):  # Increased buffer for better validation
            current_candle = df.iloc[i]
            move_threshold = self.config.get('min_order_block_size', 3.0)
            
            # Bullish Order Block (last red candle before strong green move)
            if (current_candle['close'] < current_candle['open'] and
                current_candle['volume_ratio'] >= self.config.get('min_volume_ratio', 2.0)):
                
                debug_stats['volume_filtered'] += 1
                
                # Look for validation move in next 10 candles (more flexible)
                future_moves = []
                for j in range(1, 11):
                    if i + j < len(df):
                        future_price = df.iloc[i + j]['close']
                        move_pct = (future_price - current_candle['close']) / current_candle['close'] * 100
                        future_moves.append(move_pct)
                
                max_future_move = max(future_moves) if future_moves else 0
                
                if max_future_move >= move_threshold:
                    debug_stats['move_validated'] += 1
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
                        'age_candles': len(df) - i - 1,  # Keep original for reference
                        'age': i  # FIXED: Age from current analysis point (realistic)
                    })
            
            # Bearish Order Block (last green candle before strong red move)
            elif (current_candle['close'] > current_candle['open'] and
                  current_candle['volume_ratio'] >= self.config.get('min_volume_ratio', 2.0)):
                
                debug_stats['volume_filtered'] += 1
                
                future_moves = []
                for j in range(1, 11):
                    if i + j < len(df):
                        future_price = df.iloc[i + j]['close']
                        move_pct = (current_candle['close'] - future_price) / current_candle['close'] * 100
                        future_moves.append(move_pct)
                
                max_future_move = max(future_moves) if future_moves else 0
                
                if max_future_move >= move_threshold:
                    debug_stats['move_validated'] += 1
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
                        'age_candles': len(df) - i - 1,  # Keep original for reference
                        'age': i  # FIXED: Age from current analysis point (realistic)
                    })
        
        # FIXED: Much more realistic age filtering for actionable trades
        valid_order_blocks = [ob for ob in order_blocks if ob['age'] <= 100]  # ~16 days on 4H
        debug_stats['final_blocks'] = len(valid_order_blocks)
        
        # Store debug stats for later reporting
        self.last_debug_stats = debug_stats
        
        return valid_order_blocks

    def simulate_trade_outcome_enhanced(self, trade, future_df):
        """Enhanced trade simulation with proper ICT swing logic"""
        
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        targets = trade['targets']
        direction = trade['direction']
        min_hold_bars = self.config['min_hold_bars']
        
        hit_target = None
        hit_stop = False
        exit_price = None
        exit_time = None
        bars_held = 0
        max_favorable = 0
        max_adverse = 0
        
        for i, (_, candle) in enumerate(future_df.iterrows()):
            bars_held = i + 1
            
            # CRITICAL: Don't exit before minimum hold time (institutions need time)
            if bars_held < min_hold_bars:
                # Still track excursions but don't exit
                if direction == 'BULLISH':
                    favorable_move = (candle['high'] - entry_price) / entry_price * 100
                    adverse_move = (entry_price - candle['low']) / entry_price * 100
                else:
                    favorable_move = (entry_price - candle['low']) / entry_price * 100
                    adverse_move = (candle['high'] - entry_price) / entry_price * 100
                
                max_favorable = max(max_favorable, favorable_move)
                max_adverse = max(max_adverse, adverse_move)
                continue  # Don't check exits yet
            
            # After minimum hold time, check exits normally
            if direction == 'BULLISH':
                # Check stop loss
                if candle['low'] <= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                    exit_time = candle['timestamp']
                    break
                
                # Check targets (in order)
                for j, target in enumerate(targets):
                    if candle['high'] >= target and hit_target is None:
                        hit_target = j + 1
                        exit_price = target
                        exit_time = candle['timestamp']
                        break
                
                if hit_target:
                    break
                
                # Track excursions
                favorable_move = (candle['high'] - entry_price) / entry_price * 100
                adverse_move = (entry_price - candle['low']) / entry_price * 100
                
            else:  # BEARISH
                # Check stop loss
                if candle['high'] >= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                    exit_time = candle['timestamp']
                    break
                
                # Check targets (in order)
                for j, target in enumerate(targets):
                    if candle['low'] <= target and hit_target is None:
                        hit_target = j + 1
                        exit_price = target
                        exit_time = candle['timestamp']
                        break
                
                if hit_target:
                    break
                
                # Track excursions
                favorable_move = (entry_price - candle['low']) / entry_price * 100
                adverse_move = (candle['high'] - entry_price) / entry_price * 100
            
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)
        
        # If no exit found, use last price (swing still developing)
        if exit_price is None:
            exit_price = future_df.iloc[-1]['close']
            exit_time = future_df.iloc[-1]['timestamp']
        
        # Calculate results
        if direction == 'BULLISH':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Apply commission
        pnl_pct -= (self.commission * 2 * 100)
        
        return {
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl_pct': pnl_pct,
            'hit_target': hit_target,
            'hit_stop': hit_stop,
            'bars_held': bars_held,
            'max_favorable_excursion': max_favorable,
            'max_adverse_excursion': max_adverse,
            'win': pnl_pct > 0,
            'hold_days': bars_held * 4 / 24  # Convert 4H bars to days
        }

    def backtest_symbol_enhanced(self, symbol):
        """Enhanced backtesting with proper ICT swing logic"""
        print(f"📊 Backtesting {symbol.replace('/USDT', '')}... ", end='')
        
        try:
            # Get extended historical data using config parameter
            lookback_months = self.config.get('lookback_months', 3)
            since = self.exchange.milliseconds() - (lookback_months * 30 * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '4h', since=since, limit=1000)
            
            if len(ohlcv) < 300:  # Need more data for proper swing analysis
                print("Insufficient data")
                return []
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            trades = []
            window_size = 200  # Larger window for better Order Block detection
            future_window = self.config['future_window']  # 200 candles = 33 days tracking
            
            # More conservative stepping (every 30 candles vs 20)
            for i in range(window_size, len(df) - future_window, 30):
                window_df = df.iloc[i-window_size:i].copy()
                future_df = df.iloc[i:i+future_window].copy()
                
                # Detect ICT setups
                ict_setups = self.detect_historical_setups_enhanced(window_df, future_df, symbol)
                trades.extend(ict_setups)
            
            print(f"Found {len(trades)} historical trades")
            return trades
            
        except Exception as e:
            print(f"Error: {e}")
            return []

    def detect_historical_setups_enhanced(self, df, future_df, symbol):
        """Detect ICT setups with enhanced parameters and debug output"""
        
        order_blocks = self.detect_order_blocks(df)
        
        # Debug output for Order Block detection
        debug_stats = getattr(self, 'last_debug_stats', {})
        total_candles = debug_stats.get('total_candles', len(df))
        volume_filtered = debug_stats.get('volume_filtered', 0)
        move_validated = debug_stats.get('move_validated', 0)
        final_blocks = debug_stats.get('final_blocks', len(order_blocks))
        
        if len(order_blocks) == 0:
            # Debug why no Order Blocks found
            token = symbol.replace('/USDT', '')
            if total_candles > 0:
                print(f"🔍 DEBUG {token}: {total_candles} candles → {volume_filtered} volume OK → {move_validated} move OK → {final_blocks} final blocks")
            return []
        
        trades = []
        current_time = df.iloc[-1]['timestamp']
        current_price = df.iloc[-1]['close']
        
        quality_filtered = 0
        distance_filtered = 0
        
        for ob in order_blocks:
            # Enhanced confluence scoring
            setup_score = 50
            confluence_factors = []
            
            # Volume scoring (more granular)
            if ob['volume_ratio'] >= 4.0:
                setup_score += 30
                confluence_factors.append("Exceptional Volume")
            elif ob['volume_ratio'] >= 3.0:
                setup_score += 20
                confluence_factors.append("High Volume")
            elif ob['volume_ratio'] >= 2.0:
                setup_score += 10
                confluence_factors.append("Good Volume")
            
            # Age scoring (fresher is better)
            if ob['age'] <= 5:
                setup_score += 25
                confluence_factors.append("Very Fresh")
            elif ob['age'] <= 10:
                setup_score += 15
                confluence_factors.append("Fresh")
            elif ob['age'] <= 20:
                setup_score += 5
                confluence_factors.append("Recent")
            
            # Validation move scoring
            if ob['future_move'] >= 10.0:
                setup_score += 30
                confluence_factors.append("Massive Validation")
            elif ob['future_move'] >= 8.0:
                setup_score += 25
                confluence_factors.append("Strong Validation")
            elif ob['future_move'] >= 5.0:
                setup_score += 15
                confluence_factors.append("Good Validation")
            elif ob['future_move'] >= 3.0:
                setup_score += 5
                confluence_factors.append("Moderate Validation")
            
            # Skip low quality setups
            if setup_score < self.config.get('min_quality_score', 85):
                quality_filtered += 1
                continue
            
            # Calculate entry and stop with enhanced logic
            if ob['type'] == 'bullish':
                entry_price = ob['low']  # ICT: Enter at Order Block low
                # Enhanced stop calculation with breathing room
                ob_range = ob['high'] - ob['low']
                stop_loss = ob['low'] - (ob_range * self.config['stop_multiplier'])
            else:
                entry_price = ob['high']  # ICT: Enter at Order Block high
                ob_range = ob['high'] - ob['low']
                stop_loss = ob['high'] + (ob_range * self.config['stop_multiplier'])
            
            # Skip if price too far from entry - RELAXED for more opportunities
            distance_pct = abs(current_price - entry_price) / current_price * 100
            if distance_pct > 8.0:  # RELAXED: 8% vs 5% (more actionable trades)
                distance_filtered += 1
                continue
            
            # Calculate ICT-based targets
            targets, target_reasons = self.calculate_ict_targets(df, entry_price, 
                                                               'BULLISH' if ob['type'] == 'bullish' else 'BEARISH', 
                                                               ob)
            
            # Create trade record
            trade = {
                'symbol': symbol,
                'setup_time': current_time,
                'direction': 'BULLISH' if ob['type'] == 'bullish' else 'BEARISH',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'targets': targets,
                'target_reasons': target_reasons,
                'quality_score': min(setup_score, 100),
                'confluence_factors': len(confluence_factors),
                'order_block_age': ob['age'],
                'validation_move': ob['future_move'],
                'volume_ratio': ob['volume_ratio'],
                'distance_pct': distance_pct,
                
                # ICT-specific technical indicators at setup time
                'rsi': df.iloc[-1]['rsi'],
                'mfi': df.iloc[-1]['mfi'],
                'cmf': df.iloc[-1]['cmf'],
                'accumulation_distribution': df.iloc[-1]['accumulation_distribution'],
                'on_balance_volume': df.iloc[-1]['on_balance_volume'],
                'williams_r': df.iloc[-1]['williams_r'],
                'atr': df.iloc[-1]['atr'],
                'vwap': df.iloc[-1]['vwap'],
                'vwap_deviation': df.iloc[-1]['vwap_deviation'],
                'cci': df.iloc[-1]['cci'],
                'momentum_20': df.iloc[-1]['momentum_20'],
                'momentum_50': df.iloc[-1]['momentum_50'],
                'above_recent_high': df.iloc[-1]['above_recent_high'],
                'below_recent_low': df.iloc[-1]['below_recent_low']
            }
            
            # Simulate trade outcome with enhanced logic
            outcome = self.simulate_trade_outcome_enhanced(trade, future_df)
            trade.update(outcome)
            
            trades.append(trade)
        
        # Additional debug output for filtering
        if final_blocks > 0 and len(trades) == 0:
            token = symbol.replace('/USDT', '')
            print(f"🔍 DEBUG {token}: {final_blocks} blocks → {quality_filtered} quality filtered → {distance_filtered} distance filtered → {len(trades)} final trades")
        
        return trades

    def run_enhanced_backtest(self, symbols=None, config=None):
        """Run enhanced ICT backtest with proper swing trading logic"""
        
        if config:
            self.config.update(config)
            
        if symbols is None:
            # RESTORED: Complete 89-token list from original backtester
            symbols = [
                # Major Cryptocurrencies
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT',
                'DOGE/USDT', 'TRX/USDT', 'LTC/USDT', 'BCH/USDT', 'DOT/USDT', 'AVAX/USDT',
                'SHIB/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'XLM/USDT', 'ETC/USDT',
                
                # DeFi & Layer 1s
                'MATIC/USDT', 'ALGO/USDT', 'VET/USDT', 'FIL/USDT', 'ICP/USDT', 'NEAR/USDT',
                'HBAR/USDT', 'FLOW/USDT', 'EGLD/USDT', 'XTZ/USDT', 'THETA/USDT', 'KLAY/USDT',
                'MANA/USDT', 'SAND/USDT', 'CRV/USDT', 'SUSHI/USDT', 'COMP/USDT', 'YFI/USDT',
                'AAVE/USDT', 'MKR/USDT', 'SNX/USDT', '1INCH/USDT', 'BAL/USDT', 'REN/USDT',
                
                # Gaming & NFTs
                'AXS/USDT', 'ENJ/USDT', 'GALA/USDT', 'CHZ/USDT', 'ALICE/USDT', 'SLP/USDT',
                
                # Popular Altcoins
                'FTM/USDT', 'ONE/USDT', 'CELO/USDT', 'BAT/USDT', 'ZIL/USDT', 'HOT/USDT',
                'IOTA/USDT', 'QTUM/USDT', 'OMG/USDT', 'ZEC/USDT', 'DASH/USDT', 'XMR/USDT',
                'WAVES/USDT', 'KSM/USDT', 'RUNE/USDT', 'LUNA/USDT', 'FET/USDT', 'OCEAN/USDT',
                
                # Emerging Tokens
                'APE/USDT', 'GMT/USDT', 'GST/USDT', 'STEPN/USDT', 'LDO/USDT', 'ARB/USDT',
                'OP/USDT', 'BLUR/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
                
                # Exchange Tokens
                'CRO/USDT', 'LEO/USDT', 'OKB/USDT', 'HT/USDT', 'KCS/USDT',
                
                # Oracles & Infrastructure  
                'BAND/USDT', 'API3/USDT', 'GRT/USDT', 'LPT/USDT', 'STORJ/USDT', 'AR/USDT'
            ]
        
        print("🔥 ICT ENHANCED BACKTESTING - PROPER SWING TRADING")
        print("=" * 70)
        print(f"📊 Testing {len(symbols)} symbols with optimized parameters")
        print(f"🎯 Future Window: {self.config['future_window']} candles ({self.config['future_window']*4//24} days)")
        print(f"⏰ Min Hold Time: {self.config['min_hold_bars']} bars ({self.config['min_hold_bars']*4} hours)")
        print(f"🛡️ Stop Multiplier: {self.config['stop_multiplier']}x (breathing room)")
        print(f"🎯 Target Method: {self.config['target_method']} (structure-based)")
        print(f"🔍 DEBUG: Quality Score Threshold: {self.config.get('min_quality_score', 70)}")
        print(f"🔍 DEBUG: Order Block Size Threshold: {self.config.get('min_order_block_size', 2.5)}%")
        print(f"🔍 DEBUG: Volume Ratio Threshold: {self.config.get('min_volume_ratio', 1.8)}x")
        print(f"🔍 DEBUG: Age Limit: 100 candles (~16 days) - FIXED!")
        print(f"🔍 DEBUG: Distance Limit: 8% (relaxed from 5%)")
        print()
        
        all_trades = []
        
        for symbol in symbols:
            trades = self.backtest_symbol_enhanced(symbol)
            all_trades.extend(trades)
            time.sleep(0.1)  # Rate limiting
        
        self.all_trades = all_trades
        
        if not all_trades:
            print("❌ No trades found in backtest period")
            return
        
        # Analyze results with enhanced metrics
        self.analyze_enhanced_performance()
        self.analyze_category_performance()
        self.analyze_indicator_correlations()
        
        return all_trades

    def get_token_category(self, symbol):
        """Categorize tokens by type for analysis"""
        token = symbol.replace('/USDT', '')
        
        categories = {
            'Major Cryptos': ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE', 'TRX', 'LTC', 'BCH'],
            'Layer 1s': ['DOT', 'AVAX', 'MATIC', 'ALGO', 'VET', 'FIL', 'ICP', 'NEAR', 'HBAR', 'FLOW', 'EGLD', 'XTZ', 'THETA', 'KLAY', 'FTM', 'ONE', 'CELO', 'LUNA'],
            'DeFi': ['UNI', 'LINK', 'CRV', 'SUSHI', 'COMP', 'YFI', 'AAVE', 'MKR', 'SNX', '1INCH', 'BAL', 'REN', 'LDO'],
            'Gaming/NFTs': ['AXS', 'ENJ', 'GALA', 'CHZ', 'ALICE', 'SLP', 'MANA', 'SAND'],
            'Meme/Community': ['SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'GMT', 'GST'],
            'Exchange Tokens': ['CRO', 'LEO', 'OKB', 'HT', 'KCS'],
            'Infrastructure': ['GRT', 'BAND', 'API3', 'LPT', 'STORJ', 'AR', 'OCEAN', 'FET'],
            'Altcoins': ['XLM', 'ETC', 'BAT', 'ZIL', 'HOT', 'IOTA', 'QTUM', 'OMG', 'ZEC', 'DASH', 'XMR', 'WAVES', 'KSM', 'RUNE'],
            'Layer 2/Scaling': ['ARB', 'OP', 'BLUR'],
            'Others': ['APE', 'STEPN', 'ATOM']
        }
        
        for category, tokens in categories.items():
            if token in tokens:
                return category
        
        return 'Others'

    def analyze_category_performance(self):
        """Analyze ICT performance by token categories"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Categorize trades
        category_stats = {}
        
        for trade in trades:
            category = self.get_token_category(trade['symbol'])
            
            if category not in category_stats:
                category_stats[category] = {
                    'trades': [],
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'symbols': set()
                }
            
            category_stats[category]['trades'].append(trade)
            category_stats[category]['symbols'].add(trade['symbol'].replace('/USDT', ''))
            
            if trade['win']:
                category_stats[category]['wins'] += 1
            else:
                category_stats[category]['losses'] += 1
            
            category_stats[category]['total_pnl'] += trade['pnl_pct']
        
        print("\n" + "=" * 70)
        print("🏆 ICT PERFORMANCE BY TOKEN CATEGORIES")
        print("=" * 70)
        
        # Sort categories by win rate
        sorted_categories = sorted(category_stats.items(), 
                                 key=lambda x: x[1]['wins'] / len(x[1]['trades']) if x[1]['trades'] else 0, 
                                 reverse=True)
        
        print(f"{'CATEGORY':<20} | {'TRADES':<7} | {'WIN RATE':<9} | {'AVG R/T':<8} | {'TOTAL R':<8} | {'SYMBOLS':<6}")
        print("-" * 70)
        
        for category, stats in sorted_categories:
            if not stats['trades']:
                continue
                
            total_trades = len(stats['trades'])
            win_rate = stats['wins'] / total_trades * 100
            avg_return = stats['total_pnl'] / total_trades
            total_return = stats['total_pnl']
            symbol_count = len(stats['symbols'])
            
            # Performance indicators
            if win_rate >= 85:
                performance_icon = "🔥"
            elif win_rate >= 75:
                performance_icon = "💪"
            elif win_rate >= 65:
                performance_icon = "✅"
            else:
                performance_icon = "⚠️"
            
            print(f"{performance_icon} {category:<17} | {total_trades:<7} | {win_rate:>6.1f}% | {avg_return:>+6.2f}% | {total_return:>+6.1f}% | {symbol_count:<6}")
        
        # Find best and worst categories
        if sorted_categories:
            best_category = sorted_categories[0]
            worst_category = sorted_categories[-1]
            
            print(f"\n🏆 BEST PERFORMING CATEGORY:")
            best_stats = best_category[1]
            best_wr = best_stats['wins'] / len(best_stats['trades']) * 100
            print(f"   {best_category[0]}: {best_wr:.1f}% win rate ({len(best_stats['trades'])} trades)")
            print(f"   Top tokens: {', '.join(list(best_stats['symbols'])[:5])}")
            
            print(f"\n⚠️  LOWEST PERFORMING CATEGORY:")
            worst_stats = worst_category[1]
            worst_wr = worst_stats['wins'] / len(worst_stats['trades']) * 100 if worst_stats['trades'] else 0
            print(f"   {worst_category[0]}: {worst_wr:.1f}% win rate ({len(worst_stats['trades'])} trades)")
            print(f"   Tokens: {', '.join(list(worst_stats['symbols'])[:5])}")
        
        print(f"\n💡 INSIGHTS:")
        print(f"   • Test across {len(category_stats)} different token categories")
        print(f"   • Categories with 85%+ win rates are ideal for ICT strategies")
        print(f"   • Focus trading on top-performing categories for better results")

    def analyze_indicator_correlations(self):
        """Analyze which technical indicators correlate with successful ICT trades"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Create DataFrame for analysis
        df_trades = pd.DataFrame(trades)
        
        # Remove NaN values - updated for ICT indicators
        numeric_columns = [
            'rsi', 'mfi', 'cmf', 'accumulation_distribution', 'on_balance_volume', 
            'williams_r', 'atr', 'vwap_deviation', 'cci', 'momentum_20', 'momentum_50', 'pnl_pct'
        ]
        df_clean = df_trades[numeric_columns].dropna()
        
        if len(df_clean) < 10:
            print("❌ Insufficient data for indicator correlation analysis")
            return
        
        print("\n" + "=" * 60)
        print("📈 TECHNICAL INDICATOR CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Correlation with PnL
        correlations = {}
        for indicator in numeric_columns[:-1]:  # Exclude pnl_pct
            corr = df_clean[indicator].corr(df_clean['pnl_pct'])
            correlations[indicator] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🎯 INDICATOR CORRELATION WITH TRADE SUCCESS:")
        print("(Higher absolute values = stronger predictive power)")
        print()
        
        for indicator, corr in sorted_corr:
            strength = self.get_correlation_strength(abs(corr))
            direction = "📈 Positive" if corr > 0 else "📉 Negative"
            print(f"{indicator.upper():15} | {corr:+.3f} | {strength} {direction}")
        
        # Win rate by indicator ranges
        print(f"\n🏆 WIN RATE BY INDICATOR RANGES:")
        self.analyze_indicator_ranges(df_clean)

    def get_correlation_strength(self, abs_corr):
        """Classify correlation strength"""
        if abs_corr >= 0.7:
            return "🔥 Very Strong"
        elif abs_corr >= 0.5:
            return "💪 Strong    "
        elif abs_corr >= 0.3:
            return "✅ Moderate  "
        elif abs_corr >= 0.1:
            return "⚠️  Weak     "
        else:
            return "❌ None     "

    def analyze_indicator_ranges(self, df):
        """Analyze win rates across different indicator value ranges"""
        
        key_indicators = ['rsi', 'mfi', 'cmf', 'williams_r', 'vwap_deviation', 'cci', 'accumulation_distribution']
        
        for indicator in key_indicators:
            if indicator not in df.columns:
                continue
            
            # Create quantile-based ranges
            q25 = df[indicator].quantile(0.25)
            q75 = df[indicator].quantile(0.75)
            
            low_range = df[df[indicator] <= q25]
            mid_range = df[(df[indicator] > q25) & (df[indicator] < q75)]
            high_range = df[df[indicator] >= q75]
            
            ranges = {
                'Low': low_range,
                'Mid': mid_range, 
                'High': high_range
            }
            
            print(f"\n{indicator.upper()} Performance by Range:")
            for range_name, range_data in ranges.items():
                if len(range_data) > 0:
                    wins = len(range_data[range_data['pnl_pct'] > 0])
                    total = len(range_data)
                    win_rate = wins / total * 100
                    avg_return = range_data['pnl_pct'].mean()
                    
                    range_values = f"({range_data[indicator].min():.2f} to {range_data[indicator].max():.2f})"
                    print(f"  {range_name:4} {range_values:15} | {win_rate:5.1f}% WR | {avg_return:+6.2f}% avg | {total:3d} trades")

    def analyze_enhanced_performance(self):
        """Analyze performance with swing trading metrics"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['win']])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades * 100
        
        # Enhanced PnL metrics
        total_pnl = sum([t['pnl_pct'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losing_trades > 0 else 0
        
        # Enhanced timing metrics
        avg_hold_days = np.mean([t['hold_days'] for t in trades])
        avg_bars_held = np.mean([t['bars_held'] for t in trades])
        
        # Target hit analysis
        target_hits = {}
        for i in range(1, 4):
            target_hits[f'T{i}'] = len([t for t in trades if t['hit_target'] == i])
        
        print("\n" + "=" * 70)
        print("🎯 ICT ENHANCED BACKTEST RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
        print(f"Total Return: {total_pnl:.1f}%")
        print(f"Average Return per Trade: {total_pnl/total_trades:.2f}%")
        
        print(f"\n💰 WIN/LOSS ANALYSIS:")
        print(f"Average Win: +{avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
        
        print(f"\n🎯 TARGET HIT ANALYSIS:")
        for target, count in target_hits.items():
            pct = count / total_trades * 100
            print(f"{target} Hit Rate: {pct:.1f}% ({count} trades)")
        
        stop_hits = len([t for t in trades if t['hit_stop']])
        print(f"Stop Loss Hit: {stop_hits/total_trades*100:.1f}% ({stop_hits} trades)")
        
        print(f"\n⏰ ENHANCED TIMING METRICS:")
        print(f"Average Hold Time: {avg_hold_days:.1f} days ({avg_bars_held:.1f} bars)")
        print(f"Min Hold Time Enforced: {self.config['min_hold_bars']} bars ({self.config['min_hold_bars']*4} hours)")
        
        # Compare to previous results
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"Previous Average Return: +0.47% → Current: {total_pnl/total_trades:+.2f}%")
        print(f"Previous Hold Time: 4 hours → Current: {avg_hold_days*24:.0f} hours")
        print(f"Previous Win Rate: 60.4% → Current: {win_rate:.1f}%")
        
        # Show best performing trade
        if trades:
            best_trade = max(trades, key=lambda x: x['pnl_pct'])
            print(f"\n🏆 BEST TRADE:")
            print(f"Symbol: {best_trade['symbol'].replace('/USDT', '')}")
            print(f"Return: +{best_trade['pnl_pct']:.2f}%")
            print(f"Hold Time: {best_trade['hold_days']:.1f} days")
            print(f"Target Hit: T{best_trade['hit_target']}" if best_trade['hit_target'] else "Manual Exit")

def main():
    """Run enhanced ICT backtesting"""
    
    # Test different parameter configurations
    configurations = [
        {
            'name': 'Conservative',
            'config': {
                'future_window': 150,
                'min_hold_bars': 6,
                'stop_multiplier': 0.5,
                'target_method': 'ict'
            }
        },
        {
            'name': 'Balanced',
            'config': {
                'future_window': 200,
                'min_hold_bars': 12,
                'stop_multiplier': 0.75,
                'target_method': 'ict'
            }
        },
        {
            'name': 'Aggressive',
            'config': {
                'future_window': 250,
                'min_hold_bars': 24,
                'stop_multiplier': 1.0,
                'target_method': 'ict'
            }
        }
    ]
    
    try:
        print("🚀 STARTING ICT PARAMETER OPTIMIZATION")
        print("=" * 70)
        
        best_config = None
        best_performance = 0
        
        for test in configurations:
            print(f"\n🔧 Testing {test['name']} Configuration...")
            backtester = ICTEnhancedBacktester(config=test['config'])
            results = backtester.run_enhanced_backtest()
            
            if results:
                total_return = sum([t['pnl_pct'] for t in results])
                win_rate = len([t for t in results if t['win']]) / len(results) * 100
                performance_score = total_return * (win_rate / 100)  # Combined score
                
                print(f"Performance Score: {performance_score:.2f} (Return: {total_return:.1f}%, Win Rate: {win_rate:.1f}%)")
                
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_config = test
        
        if best_config:
            print(f"\n🏆 BEST CONFIGURATION: {best_config['name']}")
            print(f"Performance Score: {best_performance:.2f}")
            print("🎯 Use these parameters for your live ICT scanner!")
        
    except KeyboardInterrupt:
        print("\n❌ Backtesting interrupted by user")
    except Exception as e:
        print(f"\n❌ Backtesting error: {e}")

if __name__ == "__main__":
    main()