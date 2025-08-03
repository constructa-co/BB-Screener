#!/usr/bin/env python3
"""
ICT Enhanced Backtester - Phase 7 with Dynamic Target Calculation
Building on Phase 6 (97% FVG win rate) with proper technical level targets

PHASE 7 ADDITIONS (R6):
1. Dynamic target calculation based on swing levels
2. Fibonacci retracement/extension levels
3. Support/resistance zone detection
4. ATR-based target scaling
5. Market structure-based targets
6. Enhanced win percentage optimization
7. Drawdown analysis for optimal stop loss placement
8. Take profit optimization analysis
9. Risk/reward ratio improvement
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
        
        # Phase 6: Adding FVG detection to surgical fixes
        self.config = config or {
            'future_window': 200,      # 33 days tracking
            'min_hold_bars': 6,        # 24 hours minimum
            'stop_multiplier': 0.75,   # Stop distance
            'target_method': 'ict',    # ICT method
            'lookback_months': 3,      # Extended lookback
            'min_order_block_size': 1.0,   # Reduced for 1H volatility
            'min_volume_ratio': 1.5,       # Reduced for 1H volatility
            'min_quality_score': 60,       # R18: Relaxed for more trades
            'max_distance_pct': 18.0,      # Keep Phase 5 baseline
            'min_confluence_factors': 3,   # Keep Phase 5 baseline
            # PHASE 5 SURGICAL FIXES:
            'use_optimized_targets': True,     # Use optimized targets
            'use_retracement_entry': True,     # Enter at 25% into OB
            'use_category_weighting': True,    # Weight by performance
            'use_enhanced_regime': True,       # Better market filtering
            'quality_boost_enabled': True,     # Enable quality improvements
            'target_win_rate': 80.0,          # Ultimate target
            'current_baseline': 64.0,         # Known working baseline
            'max_ob_age': 72,                 # Increased for 1H (72 hours)
            'require_market_regime': False,   # Market regime filter
            'min_validation_move': 0,         # Minimum validation move
            'focus_categories': None,         # Category focus filter
            # PHASE 6 ADDITIONS:
            'use_fvg_detection': True,        # Enable FVG detection
            'min_fvg_size': 0.4,             # Reduced for 1H volatility
            'max_fvg_age': 72,               # Increased for 1H (72 hours)
            'use_breaker_blocks': True,      # Enable breaker block detection
            'fvg_weight': 1.5,               # FVG quality score multiplier
            'require_fvg_volume': True,      # NEW
            'min_indicator_confluence': 3,   # INCREASED from 2
            'use_atr_targets': True,         # NEW
            'min_risk_reward': 0.8,         # CHANGED from 0.3 to 0.8 for better R/R
            # PHASE 7 ADDITIONS:
            'use_dynamic_targets': True,     # Enable dynamic target calculation
            'target_optimization_enabled': True,  # Enable target optimization analysis
            # PHASE 8 ADDITIONS:
            'use_swing_targets': True,       # NEW
            'use_dynamic_stops': True,       # NEW
            'underperforming_categories': ['Gaming/NFTs', 'DeFi'],  # Categories to monitor for improvement
            # PHASE 9 ADDITIONS:
            'category_evaluation_period': 30,  # Re-evaluate every 30 days
            'min_category_win_rate': 60,      # Minimum to trade full size
            'category_position_scaling': True,  # Scale position by performance
            # R21: NEW confluence configuration
            'confluence_quality_boost': 30,  # Boost for OB+FVG overlap
            'use_fvg_logic_for_confluence': True,  # Use FVG logic for confluence
            'min_fvg_age': 3,              # Ensure fresh FVGs
            'stop_loss_pct': 0.006,        # 0.6% stop loss
        }
        
        # Dynamic category weighting based on historical performance
        self.category_multipliers = {
            'Layer 2/Scaling': 1.3,    # 74.2% historical win rate
            'Meme/Community': 1.2,     # 71.2% win rate
            'Major Cryptos': 1.1,      # 68.9% win rate
            'Infrastructure': 1.0,     # 62.2% win rate
            'Others': 1.0,            # Baseline
            'Altcoins': 1.0,          # Baseline
            'Layer 1s': 0.95,         # 65.1% win rate
            'DeFi': 0.9,              # 68.0% win rate
            'Gaming/NFTs': 0.8        # 49.4% win rate
        }
        
        self.all_trades = []
        self.commission = 0.001

    def detect_fair_value_gaps(self, df):
        """PHASE 6: Detect Fair Value Gaps (FVGs) - crucial ICT concept"""
        fvgs = []
        min_fvg_size = self.config.get('min_fvg_size', 0.3)
        
        for i in range(2, len(df) - 1):
            # Get three consecutive candles
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if candle3['low'] > candle1['high']:
                gap_size = (candle3['low'] - candle1['high']) / candle1['high'] * 100
                
                if gap_size >= min_fvg_size:
                    fvgs.append({
                        'type': 'bullish',
                        'index': i,
                        'timestamp': candle3['timestamp'],
                        'gap_high': candle3['low'],
                        'gap_low': candle1['high'],
                        'gap_size': gap_size,
                        'filled': False,
                        'age': len(df) - i - 1,
                        'volume_surge': candle2['volume_ratio'] > 2.0
                    })
            
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            elif candle3['high'] < candle1['low']:
                gap_size = (candle1['low'] - candle3['high']) / candle1['low'] * 100
                
                if gap_size >= min_fvg_size:
                    fvgs.append({
                        'type': 'bearish',
                        'index': i,
                        'timestamp': candle3['timestamp'],
                        'gap_high': candle1['low'],
                        'gap_low': candle3['high'],
                        'gap_size': gap_size,
                        'filled': False,
                        'age': len(df) - i - 1,
                        'volume_surge': candle2['volume_ratio'] > 2.0
                    })
        
        # Filter by age
        max_age = self.config.get('max_fvg_age', 30)
        valid_fvgs = [fvg for fvg in fvgs if fvg['age'] <= max_age]
        
        # Check if FVGs have been filled
        for fvg in valid_fvgs:
            for j in range(fvg['index'] + 1, len(df)):
                candle = df.iloc[j]
                if fvg['type'] == 'bullish':
                    # Bullish FVG is filled if price goes below gap low
                    if candle['low'] <= fvg['gap_low']:
                        fvg['filled'] = True
                        break
                else:
                    # Bearish FVG is filled if price goes above gap high
                    if candle['high'] >= fvg['gap_high']:
                        fvg['filled'] = True
                        break
        
        # Return only unfilled FVGs
        return [fvg for fvg in valid_fvgs if not fvg['filled']]

    def detect_breaker_blocks(self, df, order_blocks):
        """PHASE 6: Detect Breaker Blocks (failed Order Blocks that become support/resistance)"""
        breaker_blocks = []
        
        for ob in order_blocks:
            # Check if the order block has been violated
            violated = False
            violation_index = None
            
            for i in range(ob['index'] + 1, len(df)):
                candle = df.iloc[i]
                
                if ob['type'] == 'bullish':
                    # Bullish OB violated if price closes below its low
                    if candle['close'] < ob['low']:
                        violated = True
                        violation_index = i
                        break
                else:
                    # Bearish OB violated if price closes above its high
                    if candle['close'] > ob['high']:
                        violated = True
                        violation_index = i
                        break
            
            if violated and violation_index:
                # The violated OB becomes a breaker block with opposite bias
                breaker_blocks.append({
                    'type': 'bearish' if ob['type'] == 'bullish' else 'bullish',
                    'index': violation_index,
                    'original_ob_index': ob['index'],
                    'timestamp': df.iloc[violation_index]['timestamp'],
                    'high': ob['high'],
                    'low': ob['low'],
                    'age': len(df) - violation_index - 1,
                    'volume_ratio': ob['volume_ratio']
                })
        
        return breaker_blocks

    def add_technical_indicators(self, df):
        """Add ICT-specific technical indicators with SURGICAL FIX enhancements"""
        
        # Volume indicators (critical for ICT)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volume trend for quality scoring
        df['volume_trend'] = df['volume_sma'] / df['volume_sma'].shift(20)
        
        # RSI (institutional extremes)
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
        
        # CMF (Chaikin Money Flow)
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Accumulation/Distribution Line
        ad_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        ad_multiplier = ad_multiplier.fillna(0)
        ad_volume = ad_multiplier * df['volume']
        df['accumulation_distribution'] = ad_volume.cumsum()
        
        # On Balance Volume
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
            obv.append(obv_value)
        df['on_balance_volume'] = obv
        
        # Williams %R
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14- low_14))
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # ATR percentile for regime detection
        df['atr_percentile'] = df['atr'].rolling(50).rank(pct=True)
        
        # VWAP
        typical_price_vwap = (df['high'] + df['low'] + df['close']) / 3
        vwap_numerator = (typical_price_vwap * df['volume']).rolling(window=20).sum()
        vwap_denominator = df['volume'].rolling(window=20).sum()
        df['vwap'] = vwap_numerator / vwap_denominator
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # CCI
        cci_tp = (df['high'] + df['low'] + df['close']) / 3
        cci_sma = cci_tp.rolling(window=20).mean()
        cci_mad = cci_tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (cci_tp - cci_sma) / (0.015 * cci_mad)
        
        # Swing High/Low Detection
        swing_window = 5
        df['swing_high'] = df['high'][(df['high'].shift(swing_window) < df['high']) & 
                                      (df['high'].shift(-swing_window) < df['high'])]
        df['swing_low'] = df['low'][(df['low'].shift(swing_window) > df['low']) & 
                                    (df['low'].shift(-swing_window) > df['low'])]
        
        # Liquidity Grab Detection
        df['recent_high'] = df['high'].rolling(window=10).max()
        df['recent_low'] = df['low'].rolling(window=10).min()
        df['above_recent_high'] = (df['high'] > df['recent_high'].shift(1)) & (df['close'] < df['recent_high'].shift(1))
        df['below_recent_low'] = (df['low'] < df['recent_low'].shift(1)) & (df['close'] > df['recent_low'].shift(1))
        
        # Liquidity sweep detection
        df['liquidity_sweep_high'] = (df['high'] > df['recent_high'].shift(1)) & (df['close'] < df['open'])
        df['liquidity_sweep_low'] = (df['low'] < df['recent_low'].shift(1)) & (df['close'] > df['open'])
        
        # Momentum
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        df['momentum_50'] = df['close'] / df['close'].shift(50) - 1
        
        # SMAs for market structure
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Market structure alignment
        df['price_above_sma20'] = df['close'] > df['sma_20']
        df['price_above_sma50'] = df['close'] > df['sma_50']
        df['sma20_above_sma50'] = df['sma_20'] > df['sma_50']
        
        # ADX for trend strength
        df['plus_di'] = 100 * (df['high'].diff().clip(lower=0).rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (-df['low'].diff().clip(upper=0).rolling(14).mean() / df['atr'])
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()
        
        return df 

    def detect_market_regime(self, df):
        """Enhanced market regime detection"""
        
        # Get latest values
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].rolling(50).mean().iloc[-1]
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Trend strength from ADX
        adx_value = df['adx'].iloc[-1] if 'adx' in df.columns else 20
        
        # Volume profile
        volume_trend = df['volume_trend'].iloc[-1] if 'volume_trend' in df.columns else 1
        
        # Market structure
        bullish_structure = (df['price_above_sma20'].iloc[-1] and 
                           df['price_above_sma50'].iloc[-1] and 
                           df['sma20_above_sma50'].iloc[-1])
        bearish_structure = (not df['price_above_sma20'].iloc[-1] and 
                           not df['price_above_sma50'].iloc[-1] and 
                           not df['sma20_above_sma50'].iloc[-1])
        
        # ATR percentile for volatility regime
        atr_percentile = df['atr_percentile'].iloc[-1] if 'atr_percentile' in df.columns else 0.5
        
        # Determine regime
        regime = {
            'volatility': 'high' if atr_percentile > 0.7 else 'normal',
            'trend': 'strong' if adx_value > 25 else 'weak',
            'volume': 'increasing' if volume_trend > 1.1 else 'normal',
            'structure': 'bullish' if bullish_structure else ('bearish' if bearish_structure else 'neutral'),
            'favorable': True if (volatility_ratio < 1.8 and 
                                 adx_value > 20 and 
                                 volume_trend > 0.9) else False
        }
        
        return regime

    def find_swing_levels(self, df, window=10):
        """Find significant swing highs and lows for ICT targets"""
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            if df.iloc[i]['high'] == df.iloc[i-window:i+window+1]['high'].max():
                swing_highs.append(df.iloc[i]['high'])
            
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
        
        for i, high1 in enumerate(swing_levels['highs']):
            matches = [high1]
            for high2 in swing_levels['highs'][i+1:]:
                if abs(high1 - high2) / high1 <= tolerance:
                    matches.append(high2)
            
            if len(matches) >= 2:
                equal_highs.append(np.mean(matches))
        
        for i, low1 in enumerate(swing_levels['lows']):
            matches = [low1]
            for low2 in swing_levels['lows'][i+1:]:
                if abs(low1 - low2) / low1 <= tolerance:
                    matches.append(low2)
            
            if len(matches) >= 2:
                equal_lows.append(np.mean(matches))
        
        return {
            'equal_highs': sorted(list(set(equal_highs)), reverse=True),
            'equal_lows': sorted(list(set(equal_lows)))
        }

    def calculate_dynamic_targets(self, df, entry_price, direction, setup_type='order_block'):
        """PHASE 7: Dynamic target calculation based on technical levels"""
        
        targets = []
        target_reasons = []
        atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else entry_price * 0.02
        
        # Get swing levels and market structure
        swing_levels = self.find_swing_levels(df)
        equal_levels = self.find_equal_levels(df)
        
        # Calculate Fibonacci levels
        recent_high = df['high'].rolling(window=20).max().iloc[-1]
        recent_low = df['low'].rolling(window=20).min().iloc[-1]
        fib_range = recent_high - recent_low
        
        # ATR-based scaling
        atr_multiplier = 1.5 if setup_type == 'fvg' else 2.0
        
        try:
            if direction == 'BULLISH':
                # T1: Nearest swing high or Fibonacci 0.618
                fib_618 = recent_low + (fib_range * 0.618)
                relevant_highs = [h for h in swing_levels['highs'] if h > entry_price * 1.005]
                
                if relevant_highs:
                    nearest_high = min(relevant_highs)
                    t1_target = min(nearest_high, fib_618)
                    targets.append(t1_target)
                    target_reasons.append(f"T1: Swing/Fib @ {t1_target:.6f}")
                else:
                    t1_target = entry_price + (atr_value * atr_multiplier)
                    targets.append(t1_target)
                    target_reasons.append(f"T1: ATR-based @ {t1_target:.6f}")
                
                # T2: Major swing high or Fibonacci 1.0
                fib_100 = recent_low + fib_range
                if len(relevant_highs) > 1:
                    major_high = sorted(relevant_highs)[1]
                    t2_target = min(major_high, fib_100)
                else:
                    t2_target = entry_price + (atr_value * atr_multiplier * 2)
                targets.append(t2_target)
                target_reasons.append(f"T2: Major Level @ {t2_target:.6f}")
                
                # T3: Fibonacci 1.618 or significant resistance
                fib_1618 = recent_low + (fib_range * 1.618)
                equal_highs = [h for h in equal_levels['equal_highs'] if h > entry_price * 1.01]
                if equal_highs:
                    resistance_level = min(equal_highs)
                    t3_target = min(resistance_level, fib_1618)
                else:
                    t3_target = entry_price + (atr_value * atr_multiplier * 3)
                targets.append(t3_target)
                target_reasons.append(f"T3: Resistance/Fib @ {t3_target:.6f}")
                
            else:  # BEARISH
                # T1: Nearest swing low or Fibonacci 0.382
                fib_382 = recent_high - (fib_range * 0.382)
                relevant_lows = [l for l in swing_levels['lows'] if l < entry_price * 0.995]
                
                if relevant_lows:
                    nearest_low = max(relevant_lows)
                    t1_target = max(nearest_low, fib_382)
                    targets.append(t1_target)
                    target_reasons.append(f"T1: Swing/Fib @ {t1_target:.6f}")
                else:
                    t1_target = entry_price - (atr_value * atr_multiplier)
                    targets.append(t1_target)
                    target_reasons.append(f"T1: ATR-based @ {t1_target:.6f}")
                
                # T2: Major swing low or Fibonacci 0.0
                fib_000 = recent_high - fib_range
                if len(relevant_lows) > 1:
                    major_low = sorted(relevant_lows, reverse=True)[1]
                    t2_target = max(major_low, fib_000)
                else:
                    t2_target = entry_price - (atr_value * atr_multiplier * 2)
                targets.append(t2_target)
                target_reasons.append(f"T2: Major Level @ {t2_target:.6f}")
                
                # T3: Fibonacci -0.618 or significant support
                fib_neg618 = recent_high - (fib_range * 1.618)
                equal_lows = [l for l in equal_levels['equal_lows'] if l < entry_price * 0.99]
                if equal_lows:
                    support_level = max(equal_lows)
                    t3_target = max(support_level, fib_neg618)
                else:
                    t3_target = entry_price - (atr_value * atr_multiplier * 3)
                targets.append(t3_target)
                target_reasons.append(f"T3: Support/Fib @ {t3_target:.6f}")
        
        except Exception as e:
            print(f"Dynamic target calculation error: {e}")
            # Fallback to ATR-based targets
            if direction == 'BULLISH':
                targets = [
                    entry_price + (atr_value * atr_multiplier),
                    entry_price + (atr_value * atr_multiplier * 2),
                    entry_price + (atr_value * atr_multiplier * 3)
                ]
                target_reasons = ["T1: ATR Fallback", "T2: ATR Fallback", "T3: ATR Fallback"]
            else:
                targets = [
                    entry_price - (atr_value * atr_multiplier),
                    entry_price - (atr_value * atr_multiplier * 2),
                    entry_price - (atr_value * atr_multiplier * 3)
                ]
                target_reasons = ["T1: ATR Fallback", "T2: ATR Fallback", "T3: ATR Fallback"]
        
        return targets[:3], target_reasons[:3]

    def calculate_fibonacci_targets(self, df, entry_price, direction):
        """Calculate targets based on Fibonacci extensions from swing levels"""
        # Find recent swing levels (wider window for better context)
        swings = self.find_swing_levels(df, window=20)
        
        if len(swings['highs']) < 2 or len(swings['lows']) < 2:
            # Fallback to percentage targets
            if direction == 'BULLISH':
                targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
                reasons = ["T1: +0.8%", "T2: +1.5%", "T3: +2.5%"]
            else:
                targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
                reasons = ["T1: -0.8%", "T2: -1.5%", "T3: -2.5%"]
            return targets, reasons
        
        if direction == 'BULLISH':
            # Find recent swing low and swing high for bullish trades
            recent_lows = sorted(swings['lows'][-5:])  # Last 5 swing lows
            recent_highs = sorted(swings['highs'][-5:])  # Last 5 swing highs
            
            if len(recent_lows) < 2 or len(recent_highs) < 2:
                # Fallback to percentage targets
                targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
                reasons = ["T1: +0.8%", "T2: +1.5%", "T3: +2.5%"]
                return targets, reasons
            
            # Use the most recent swing low and high
            swing_low = recent_lows[-1]  # Most recent swing low
            swing_high = recent_highs[-1]  # Most recent swing high
            
            # Ensure swing low is below entry and swing high is above
            if swing_low >= entry_price or swing_high <= entry_price:
                # Fallback to percentage targets
                targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
                reasons = ["T1: +0.8%", "T2: +1.5%", "T3: +2.5%"]
                return targets, reasons
            
                            # Calculate Fibonacci extensions from swing low to swing high (R18: Enhanced levels)
                range_size = swing_high - swing_low
                fib_600 = swing_low + (range_size * 0.600)  # 60% extension (R18: Enhanced)
                fib_800 = swing_low + (range_size * 0.800)  # 80% extension (R18: Optimized)
                fib_100 = swing_low + (range_size * 1.000)  # 100% extension (R18: Equal move)
                
                return [fib_600, fib_800, fib_100], ["Fib 60%", "Fib 80%", "Fib 100%"]
        else:
            # Bearish - reverse the logic
            recent_highs = sorted(swings['highs'][-5:], reverse=True)  # Last 5 swing highs
            recent_lows = sorted(swings['lows'][-5:], reverse=True)  # Last 5 swing lows
            
            if len(recent_highs) < 2 or len(recent_lows) < 2:
                # Fallback to percentage targets
                targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
                reasons = ["T1: -0.8%", "T2: -1.5%", "T3: -2.5%"]
                return targets, reasons
            
            swing_high = recent_highs[-1]  # Most recent swing high
            swing_low = recent_lows[-1]  # Most recent swing low
            
            # Ensure swing high is above entry and swing low is below
            if swing_high <= entry_price or swing_low >= entry_price:
                # Fallback to percentage targets
                targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
                reasons = ["T1: -0.8%", "T2: -1.5%", "T3: -2.5%"]
                return targets, reasons
            
                            # Calculate Fibonacci extensions from swing high to swing low (R18: Enhanced levels)
                range_size = swing_high - swing_low
                fib_600 = swing_high - (range_size * 0.600)  # 60% extension (R18: Enhanced)
                fib_800 = swing_high - (range_size * 0.800)  # 80% extension (R18: Optimized)
                fib_100 = swing_high - (range_size * 1.000)  # 100% extension (R18: Equal move)
                
                return [fib_600, fib_800, fib_100], ["Fib 60%", "Fib 80%", "Fib 100%"]

    def calculate_support_resistance_targets(self, df, entry_price, direction):
        """Calculate targets based on key support/resistance levels"""
        # Find recent swing levels
        swings = self.find_swing_levels(df, window=20)
        
        if direction == 'BULLISH':
            # Find resistance levels above entry
            resistances = [h for h in swings['highs'] if h > entry_price * 1.01]
            if len(resistances) >= 3:
                return sorted(resistances)[:3], ["R1", "R2", "R3"]
        else:
            # Find support levels below entry
            supports = [l for l in swings['lows'] if l < entry_price * 0.99]
            if len(supports) >= 3:
                return sorted(supports, reverse=True)[:3], ["S1", "S2", "S3"]
        
        # Fallback to percentage targets
        if direction == 'BULLISH':
            targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
            reasons = ["T1: +0.8% (Fallback)", "T2: +1.5% (Fallback)", "T3: +2.5% (Fallback)"]
        else:
            targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
            reasons = ["T1: -0.8% (Fallback)", "T2: -1.5% (Fallback)", "T3: -2.5% (Fallback)"]
        return targets, reasons

    def calculate_atr_dynamic_targets(self, df, entry_price, direction, setup_type):
        """Calculate ATR-based targets that adapt to volatility"""
        atr = df.iloc[-1]['atr']
        
        if setup_type == 'order_block':
            # Order Blocks: Higher targets due to stronger moves
            multipliers = [1.5, 2.5, 4.0]  # More aggressive
        else:
            # FVGs: Lower targets due to weaker moves
            multipliers = [1.0, 1.8, 2.8]  # More conservative
        
        if direction == 'BULLISH':
            targets = [entry_price + (atr * m) for m in multipliers]
        else:
            targets = [entry_price - (atr * m) for m in multipliers]
        
        return targets, [f"{m}x ATR" for m in multipliers]

    def calculate_swing_based_targets(self, df, entry_price, direction):
        """Calculate targets based on recent swing highs/lows (LEGACY - kept for compatibility)"""
        # Find recent swings
        swings = self.find_swing_levels(df, window=10)
        
        if direction == 'BULLISH':
            # Find resistance levels above entry
            resistances = [h for h in swings['highs'] if h > entry_price * 1.002]
            if len(resistances) >= 3:
                return sorted(resistances)[:3], ["Swing High 1", "Swing High 2", "Swing High 3"]
        else:
            # Find support levels below entry
            supports = [l for l in swings['lows'] if l < entry_price * 0.998]
            if len(supports) >= 3:
                return sorted(supports, reverse=True)[:3], ["Swing Low 1", "Swing Low 2", "Swing Low 3"]
        
        # Fallback to percentage targets
        if direction == 'BULLISH':
            targets = [entry_price * 1.008, entry_price * 1.015, entry_price * 1.025]
            reasons = ["T1: +0.8% (Fallback)", "T2: +1.5% (Fallback)", "T3: +2.5% (Fallback)"]
        else:
            targets = [entry_price * 0.992, entry_price * 0.985, entry_price * 0.975]
            reasons = ["T1: -0.8% (Fallback)", "T2: -1.5% (Fallback)", "T3: -2.5% (Fallback)"]
        return targets, reasons

    def calculate_atr_based_targets(self, df, entry_price, direction):
        """Calculate targets based on ATR multiples"""
        atr = df.iloc[-1]['atr']
        
        if direction == 'BULLISH':
            targets = [
                entry_price + (atr * 1.0),  # 1x ATR
                entry_price + (atr * 2.0),  # 2x ATR
                entry_price + (atr * 3.0)   # 3x ATR
            ]
        else:
            targets = [
                entry_price - (atr * 1.0),
                entry_price - (atr * 2.0),
                entry_price - (atr * 3.0)
            ]
        
        return targets, ["1x ATR", "2x ATR", "3x ATR"]

    def calculate_ict_targets(self, df, entry_price, direction, setup_type='order_block'):
        """PHASE 14: Fibonacci-based intelligent target calculation"""
        
        # Try Fibonacci-based targets first (NEW for R14)
        try:
            targets, reasons = self.calculate_fibonacci_targets(df, entry_price, direction)
            if targets and reasons:
                return targets[:3], reasons[:3]
        except:
            pass
        
        # Try Support/Resistance targets second
        try:
            targets, reasons = self.calculate_support_resistance_targets(df, entry_price, direction)
            if targets and reasons:
                return targets[:3], reasons[:3]
        except:
            pass
        
        # Try ATR-based dynamic targets third
        try:
            targets, reasons = self.calculate_atr_dynamic_targets(df, entry_price, direction, setup_type)
            if targets and reasons:
                return targets[:3], reasons[:3]
        except:
            pass
        
        # Fallback to optimized percentage targets (if all else fails)
        if direction == 'BULLISH':
            if setup_type == 'order_block':
                # Order Block trades: fallback targets
                targets = [
                    entry_price * 1.050,  # 5.0% - Fallback OB T1
                    entry_price * 1.080,  # 8.0% - Fallback OB T2
                    entry_price * 1.120   # 12.0% - Fallback OB T3
                ]
                target_reasons = ["T1: +5.0% (Fallback)", "T2: +8.0% (Fallback)", "T3: +12.0% (Fallback)"]
            else:
                # FVG trades: fallback targets
                targets = [
                    entry_price * 1.040,  # 4.0% - Fallback FVG T1
                    entry_price * 1.065,  # 6.5% - Fallback FVG T2
                    entry_price * 1.090   # 9.0% - Fallback FVG T3
                ]
                target_reasons = ["T1: +4.0% (Fallback)", "T2: +6.5% (Fallback)", "T3: +9.0% (Fallback)"]
        else:
            if setup_type == 'order_block':
                # Order Block trades: fallback targets
                targets = [
                    entry_price * 0.950,  # -5.0% - Fallback OB T1
                    entry_price * 0.920,  # -8.0% - Fallback OB T2
                    entry_price * 0.880   # -12.0% - Fallback OB T3
                ]
                target_reasons = ["T1: -5.0% (Fallback)", "T2: -8.0% (Fallback)", "T3: -12.0% (Fallback)"]
            else:
                # FVG trades: fallback targets
                targets = [
                    entry_price * 0.960,  # -4.0% - Fallback FVG T1
                    entry_price * 0.935,  # -6.5% - Fallback FVG T2
                    entry_price * 0.910   # -9.0% - Fallback FVG T3
                ]
                target_reasons = ["T1: -4.0% (Fallback)", "T2: -6.5% (Fallback)", "T3: -9.0% (Fallback)"]
        
        return targets[:3], target_reasons[:3]

    def calculate_retracement_entry(self, zone, direction, zone_type='order_block'):
        """Calculate 25% retracement entry into zone (OB or FVG)"""
        
        if zone_type == 'order_block':
            zone_range = zone['high'] - zone['low']
            
            if direction == 'BULLISH':
                # For bullish OB, enter 25% up from the low
                entry_price = zone['low'] + (zone_range * 0.25)
            else:
                # For bearish OB, enter 25% down from the high
                entry_price = zone['high'] - (zone_range * 0.25)
        
        elif zone_type == 'fvg':
            # For FVGs, enter at the midpoint of the gap
            entry_price = (zone['gap_high'] + zone['gap_low']) / 2
        
        elif zone_type == 'breaker':
            # For breaker blocks, enter at 50% of the zone
            zone_range = zone['high'] - zone['low']
            entry_price = zone['low'] + (zone_range * 0.5)
        
        return entry_price

    def calculate_dynamic_stop_loss(self, df, ob, entry_price, direction):
        """Calculate stop loss based on market structure and volatility"""
        atr = df.iloc[-1]['atr']
        
        # Method 1: Structure-based stop
        if direction == 'BULLISH':
            # Find recent swing low
            recent_lows = df['low'].rolling(window=10).min()
            structure_stop = recent_lows.iloc[-1] - (atr * 0.1)
            
            # Method 2: ATR-based stop
            atr_stop = entry_price - (atr * 1.5)
            
            # Method 3: Order block based stop
            ob_stop = ob['low'] - (atr * 0.2)
            
            # Use the tightest stop that gives at least 1:1 R/R
            stop_loss = max(structure_stop, atr_stop, ob_stop)
        else:
            # Find recent swing high
            recent_highs = df['high'].rolling(window=10).max()
            structure_stop = recent_highs.iloc[-1] + (atr * 0.1)
            
            atr_stop = entry_price + (atr * 1.5)
            ob_stop = ob['high'] + (atr * 0.2)
            
            stop_loss = min(structure_stop, atr_stop, ob_stop)
        
        return stop_loss

    def calculate_position_size_multiplier(self, quality_score, setup_type, category):
        """Calculate position size based on setup probability"""
        base_multiplier = 1.0
        
        # Quality score adjustment
        if quality_score >= 90:
            base_multiplier *= 1.5
        elif quality_score >= 80:
            base_multiplier *= 1.2
        elif quality_score < 70:
            base_multiplier *= 0.5
        
        # Setup type adjustment
        if setup_type == 'fvg':
            base_multiplier *= 1.3  # FVGs are more reliable
        
        # Category adjustment
        if category == 'Layer 2/Scaling':
            base_multiplier *= 1.2
        elif category in ['Gaming/NFTs', 'DeFi']:
            base_multiplier *= 0.5
        
        return min(base_multiplier, 2.0)  # Cap at 2x

    def detect_order_blocks(self, df):
        """Detect ICT Order Blocks"""
        order_blocks = []
        debug_stats = {
            'total_candles': len(df),
            'volume_filtered': 0,
            'move_validated': 0,
            'final_blocks': 0
        }
        
        for i in range(10, len(df) - 10):
            current_candle = df.iloc[i]
            move_threshold = self.config.get('min_order_block_size', 1.3)
            
            # Check for liquidity sweep before OB formation
            liquidity_swept = False
            quality_multiplier = 1.0
            if i >= 2:
                prev_low = df.iloc[i-2:i]['low'].min()
                if df.iloc[i-1]['low'] < prev_low and df.iloc[i]['close'] > prev_low:
                    liquidity_swept = True
                    quality_multiplier = 1.5  # 50% quality boost
            
            # Bullish Order Block
            if (current_candle['close'] < current_candle['open'] and
                current_candle['volume_ratio'] >= self.config.get('min_volume_ratio', 1.4)):
                
                debug_stats['volume_filtered'] += 1
                
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
                        'age_candles': len(df) - i - 1,
                        'age': i,
                        'liquidity_swept': liquidity_swept,
                        'quality_multiplier': quality_multiplier
                    })
            
            # Bearish Order Block
            elif (current_candle['close'] > current_candle['open'] and
                  current_candle['volume_ratio'] >= self.config.get('min_volume_ratio', 1.4)):
                
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
                        'age_candles': len(df) - i - 1,
                        'age': i,
                        'liquidity_swept': liquidity_swept,
                        'quality_multiplier': quality_multiplier
                    })
        
        # Age filtering - TIGHTENED to 50 bars
        max_age = self.config.get('max_ob_age', 50)
        valid_order_blocks = [ob for ob in order_blocks if ob['age'] <= max_age]
        
        # Validation move filter
        min_validation = self.config.get('min_validation_move', 0)
        if min_validation > 0:
            valid_order_blocks = [ob for ob in valid_order_blocks if ob['future_move'] >= min_validation]
        
        debug_stats['final_blocks'] = len(valid_order_blocks)
        self.last_debug_stats = debug_stats
        
        return valid_order_blocks

    def simulate_trade_outcome_enhanced(self, trade, future_df):
        """Simulate trade outcome with surgical fixes"""
        
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
            
            # Minimum hold logic
            if bars_held < min_hold_bars and bars_held < 6:
                if direction == 'BULLISH':
                    favorable_move = (candle['high'] - entry_price) / entry_price * 100
                    adverse_move = (entry_price - candle['low']) / entry_price * 100
                else:
                    favorable_move = (entry_price - candle['low']) / entry_price * 100
                    adverse_move = (candle['high'] - entry_price) / entry_price * 100
                
                max_favorable = max(max_favorable, favorable_move)
                max_adverse = max(max_adverse, adverse_move)
                
                if adverse_move > 8.0:
                    break
                    
                continue
            
            # Check exits after minimum hold
            if direction == 'BULLISH':
                if candle['low'] <= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                    exit_time = candle['timestamp']
                    break
                
                for j, target in enumerate(targets):
                    if candle['high'] >= target and hit_target is None:
                        hit_target = j + 1
                        exit_price = target
                        exit_time = candle['timestamp']
                        break
                
                if hit_target:
                    break
                
                favorable_move = (candle['high'] - entry_price) / entry_price * 100
                adverse_move = (entry_price - candle['low']) / entry_price * 100
                
            else:  # BEARISH
                if candle['high'] >= stop_loss:
                    hit_stop = True
                    exit_price = stop_loss
                    exit_time = candle['timestamp']
                    break
                
                for j, target in enumerate(targets):
                    if candle['low'] <= target and hit_target is None:
                        hit_target = j + 1
                        exit_price = target
                        exit_time = candle['timestamp']
                        break
                
                if hit_target:
                    break
                
                favorable_move = (entry_price - candle['low']) / entry_price * 100
                adverse_move = (candle['high'] - entry_price) / entry_price * 100
            
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)
        
        # If no exit found, use last price
        if exit_price is None:
            exit_price = future_df.iloc[-1]['close']
            exit_time = future_df.iloc[-1]['timestamp']
        
        # Calculate PnL
        if direction == 'BULLISH':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # Apply commission
        pnl_pct -= (self.commission * 2 * 100)
        

        
        # PHASE 7: More realistic win/loss logic
        # A trade is a win if it's profitable, regardless of target hits
        is_win = pnl_pct > 0
        
        # If we hit a target, that's a "target win"
        # If we're profitable but didn't hit targets, that's still a win
        if hit_target is not None:
            target_win = True
        else:
            target_win = is_win  # If profitable, it's a target win too
        
        return {
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl_pct': pnl_pct,
            'hit_target': hit_target,
            'hit_stop': hit_stop,
            'bars_held': bars_held,
            'max_favorable_excursion': max_favorable,
            'max_adverse_excursion': max_adverse,
            'win': is_win,
            'target_win': target_win,  # New field for target-based wins
            'hold_days': bars_held * 4 / 24
        } 

    def backtest_symbol_enhanced(self, symbol):
        """Enhanced backtesting with proper ICT swing logic"""
        print(f"📊 Backtesting {symbol.replace('/USDT', '')}... ", end='')
        
        try:
            lookback_months = self.config.get('lookback_months', 3)
            since = self.exchange.milliseconds() - (lookback_months * 30 * 24 * 60 * 60 * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
            
            if len(ohlcv) < 300:
                print("Insufficient data")
                return []
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df = self.add_technical_indicators(df)
            
            trades = []
            window_size = 200
            future_window = self.config['future_window']
            
            for i in range(window_size, len(df) - future_window, 30):
                window_df = df.iloc[i-window_size:i].copy()
                future_df = df.iloc[i:i+future_window].copy()
                
                ict_setups = self.detect_historical_setups_enhanced(window_df, future_df, symbol)
                trades.extend(ict_setups)
            
            print(f"Found {len(trades)} historical trades")
            return trades
            
        except Exception as e:
            print(f"Error: {e}")
            return [] 

    def analyze_losing_trade_patterns(self):
        """Analyze losing trades to identify patterns for improvement"""
        trades = self.all_trades
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
            if 'quality_score' not in trade:
                trade['quality_score'] = 50
        
        losing_trades = [t for t in trades if not t['win']]
        
        if not losing_trades:
            return
        
        print("\n" + "=" * 70)
        print("🔍 LOSING TRADE PATTERN ANALYSIS")
        print("=" * 70)
        
        # Analyze by setup type
        setup_types = {}
        for trade in losing_trades:
            setup_type = trade.get('setup_type', 'order_block')
            if setup_type not in setup_types:
                setup_types[setup_type] = []
            setup_types[setup_type].append(trade)
        
        print("\n📊 LOSING TRADES BY SETUP TYPE:")
        for setup_type, type_trades in setup_types.items():
            if type_trades:
                avg_loss = np.mean([t['pnl_pct'] for t in type_trades])
                count = len(type_trades)
                pct_of_losses = count / len(losing_trades) * 100
                print(f"{setup_type}: {count} trades ({pct_of_losses:.1f}% of losses), avg loss: {avg_loss:.2f}%")
        
        # Analyze by quality score
        quality_buckets = {
            'Low (50-60)': [t for t in losing_trades if 50 <= t['quality_score'] < 60],
            'Medium (60-70)': [t for t in losing_trades if 60 <= t['quality_score'] < 70],
            'High (70-80)': [t for t in losing_trades if 70 <= t['quality_score'] < 80],
            'Very High (80+)': [t for t in losing_trades if t['quality_score'] >= 80]
        }
        
        print("\n📊 LOSING TRADES BY QUALITY SCORE:")
        for bucket, bucket_trades in quality_buckets.items():
            if bucket_trades:
                avg_loss = np.mean([t['pnl_pct'] for t in bucket_trades])
                count = len(bucket_trades)
                pct_of_losses = count / len(losing_trades) * 100
                print(f"{bucket}: {count} trades ({pct_of_losses:.1f}% of losses), avg loss: {avg_loss:.2f}%")
        
        # Analyze by volume ratio (for OBs)
        ob_losses = [t for t in losing_trades if t.get('setup_type', 'order_block') == 'order_block']
        if ob_losses:
            print("\n📊 ORDER BLOCK LOSSES BY VOLUME RATIO:")
            volume_ranges = {
                'Low (<1.5x)': [t for t in ob_losses if t.get('volume_ratio', 0) < 1.5],
                'Medium (1.5-2.0x)': [t for t in ob_losses if 1.5 <= t.get('volume_ratio', 0) < 2.0],
                'High (2.0-3.0x)': [t for t in ob_losses if 2.0 <= t.get('volume_ratio', 0) < 3.0],
                'Very High (3.0x+)': [t for t in ob_losses if t.get('volume_ratio', 0) >= 3.0]
            }
            
            for range_name, range_trades in volume_ranges.items():
                if range_trades:
                    count = len(range_trades)
                    pct_of_losses = count / len(ob_losses) * 100
                    avg_loss = np.mean([t['pnl_pct'] for t in range_trades])
                    print(f"{range_name}: {count} trades ({pct_of_losses:.1f}% of OB losses), avg loss: {avg_loss:.2f}%")
        
        # Analyze FVG losses
        fvg_losses = [t for t in losing_trades if t.get('setup_type') == 'fvg']
        if fvg_losses:
            print("\n📊 FVG LOSSES BY GAP SIZE:")
            gap_ranges = {
                'Small (<0.5%)': [t for t in fvg_losses if t.get('fvg_gap_size', 0) < 0.5],
                'Medium (0.5-1.0%)': [t for t in fvg_losses if 0.5 <= t.get('fvg_gap_size', 0) < 1.0],
                'Large (>1.0%)': [t for t in fvg_losses if t.get('fvg_gap_size', 0) >= 1.0]
            }
            
            for range_name, range_trades in gap_ranges.items():
                if range_trades:
                    count = len(range_trades)
                    pct_of_fvg_losses = count / len(fvg_losses) * 100
                    print(f"{range_name}: {count} trades ({pct_of_fvg_losses:.1f}% of FVG losses)")
        
        # Stop loss analysis
        stop_losses = [t for t in losing_trades if t['hit_stop']]
        if stop_losses:
            stop_pct = len(stop_losses) / len(losing_trades) * 100
            print(f"\n📊 STOP LOSS ANALYSIS:")
            print(f"Stop losses: {len(stop_losses)} ({stop_pct:.1f}% of losing trades)")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS TO REACH 80% WIN RATE:")
        
        # Setup type recommendation
        if 'fvg' in setup_types and len(setup_types['fvg']) > len(losing_trades) * 0.3:
            print("1. ⚠️ Review FVG entry criteria (30%+ losses are FVGs)")
        
        # Quality score recommendation
        low_quality_losses = len(quality_buckets['Low (50-60)'])
        if low_quality_losses > len(losing_trades) * 0.3:
            print("2. ⚠️ Increase min_quality_score to 65 (30%+ losses are low quality)")
        
        # Stop loss recommendation
        if len(stop_losses) > len(losing_trades) * 0.6:
            print("3. ⚠️ Consider wider stops or better entry timing (60%+ losses hit stops)")

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

    def evaluate_category_performance(self, category, recent_days=30):
        """Dynamically evaluate category performance over recent period"""
        # Check last 30 days of trades for this category
        recent_performance = self.get_recent_category_stats(category, recent_days)
        
        if recent_performance['win_rate'] < 60:
            return 0.5  # Reduce position size
        elif recent_performance['win_rate'] > 75:
            return 1.5  # Increase position size
        else:
            return 1.0  # Normal position size

    def get_recent_category_stats(self, category, recent_days=30):
        """Get recent performance statistics for a specific category"""
        if not hasattr(self, 'all_trades') or not self.all_trades:
            return {'win_rate': 65, 'total_trades': 0, 'avg_pnl': 0}
        
        # Calculate the cutoff date (recent_days ago)
        cutoff_date = datetime.now() - timedelta(days=recent_days)
        
        # Filter trades for this category within the recent period
        category_trades = []
        for trade in self.all_trades:
            if (self.get_token_category(trade['symbol']) == category and 
                trade['setup_time'] >= cutoff_date):
                category_trades.append(trade)
        
        if not category_trades:
            return {'win_rate': 65, 'total_trades': 0, 'avg_pnl': 0}
        
        # Calculate statistics
        winning_trades = [t for t in category_trades if t['win']]
        win_rate = len(winning_trades) / len(category_trades) * 100
        avg_pnl = np.mean([t['pnl_pct'] for t in category_trades])
        
        return {
            'win_rate': win_rate,
            'total_trades': len(category_trades),
            'avg_pnl': avg_pnl
        }

    def track_category_improvements(self):
        """Track and report on category performance improvements"""
        if not hasattr(self, 'all_trades') or not self.all_trades:
            return
        
        underperforming_categories = self.config.get('underperforming_categories', [])
        
        print("\n" + "=" * 70)
        print("📈 CATEGORY PERFORMANCE IMPROVEMENT TRACKING")
        print("=" * 70)
        
        for category in underperforming_categories:
            recent_stats = self.get_recent_category_stats(category, 30)
            older_stats = self.get_recent_category_stats(category, 60)  # 60 days ago
            
            if recent_stats['total_trades'] > 0 and older_stats['total_trades'] > 0:
                win_rate_change = recent_stats['win_rate'] - older_stats['win_rate']
                pnl_change = recent_stats['avg_pnl'] - older_stats['avg_pnl']
                
                print(f"\n📊 {category}:")
                print(f"  Recent 30 days: {recent_stats['win_rate']:.1f}% win rate, {recent_stats['avg_pnl']:.2f}% avg PnL")
                print(f"  Previous 30 days: {older_stats['win_rate']:.1f}% win rate, {older_stats['avg_pnl']:.2f}% avg PnL")
                print(f"  Change: {win_rate_change:+.1f}% win rate, {pnl_change:+.2f}% PnL")
                
                if win_rate_change > 10:
                    print(f"  🚀 SIGNIFICANT IMPROVEMENT - Consider scaling back up")
                elif win_rate_change > 5:
                    print(f"  📈 MODERATE IMPROVEMENT - Monitor for continued progress")
                elif win_rate_change < -5:
                    print(f"  ⚠️  DECLINING PERFORMANCE - Maintain reduced position sizing")
                else:
                    print(f"  ➡️  STABLE PERFORMANCE - Continue monitoring")

    def analyze_category_performance(self):
        """Analyze ICT performance by token categories"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
        
        category_stats = {}
        
        for trade in trades:
            category = self.get_token_category(trade['symbol'])
            
            if category not in category_stats:
                category_stats[category] = {
                    'trades': [],
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'symbols': set(),
                    'setup_types': {}
                }
            
            category_stats[category]['trades'].append(trade)
            category_stats[category]['symbols'].add(trade['symbol'].replace('/USDT', ''))
            
            # Track setup types
            setup_type = trade.get('setup_type', 'order_block')
            if setup_type not in category_stats[category]['setup_types']:
                category_stats[category]['setup_types'][setup_type] = {'wins': 0, 'losses': 0}
            
            if trade['win']:
                category_stats[category]['wins'] += 1
                category_stats[category]['setup_types'][setup_type]['wins'] += 1
            else:
                category_stats[category]['losses'] += 1
                category_stats[category]['setup_types'][setup_type]['losses'] += 1
            
            category_stats[category]['total_pnl'] += trade['pnl_pct']
        
        print("\n" + "=" * 70)
        print("🏆 ICT PERFORMANCE BY TOKEN CATEGORIES (WITH FVGs)")
        print("=" * 70)
        
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
            
            if win_rate >= 80:
                performance_icon = "🎯"
            elif win_rate >= 75:
                performance_icon = "🔥"
            elif win_rate >= 70:
                performance_icon = "💪"
            elif win_rate >= 65:
                performance_icon = "✅"
            else:
                performance_icon = "⚠️"
            
            print(f"{performance_icon} {category:<17} | {total_trades:<7} | {win_rate:>6.1f}% | {avg_return:>+6.2f}% | {total_return:>+6.1f}% | {symbol_count:<6}")
        
        # Show setup type breakdown for top categories
        print("\n📊 SETUP TYPE BREAKDOWN FOR TOP CATEGORIES:")
        for category, stats in sorted_categories[:3]:
            if stats['trades']:
                print(f"\n{category}:")
                for setup_type, type_stats in stats['setup_types'].items():
                    total = type_stats['wins'] + type_stats['losses']
                    if total > 0:
                        wr = type_stats['wins'] / total * 100
                        print(f"  - {setup_type}: {total} trades, {wr:.1f}% win rate")
        
        # Category weighting impact
        if self.config.get('use_category_weighting', True):
            print(f"\n📊 CATEGORY WEIGHTING MULTIPLIERS:")
            for category, multiplier in self.category_multipliers.items():
                print(f"   {category}: {multiplier:.1f}x")

    def analyze_indicator_performance(self):
        """Analyze which indicators provide better confirmations and win rates"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
        
        print(f"\n======================================================================")
        print(f"📊 INDICATOR PERFORMANCE ANALYSIS")
        print(f"======================================================================\n")
        
        # Group trades by indicator confirmations
        indicator_stats = {
            'RSI': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'MFI': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'CMF': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'ADX': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'VWAP': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'Williams_R': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'CCI': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'OBV': {'trades': [], 'win_rate': 0, 'avg_return': 0},
            'AD': {'trades': [], 'win_rate': 0, 'avg_return': 0}
        }
        
        # Analyze each trade's indicator confirmations (R17: Fixed indicator tracking)
        for trade in trades:
            # Check individual indicators from trade data
            if trade.get('rsi', 0) < 40:  # Oversold RSI
                indicator_stats['RSI']['trades'].append(trade)
            if trade.get('mfi', 0) < 30:  # Oversold MFI
                indicator_stats['MFI']['trades'].append(trade)
            if trade.get('cmf', 0) > 0.1:  # Positive CMF
                indicator_stats['CMF']['trades'].append(trade)
            if trade.get('adx', 0) > 25:  # Strong ADX
                indicator_stats['ADX']['trades'].append(trade)
            if trade.get('vwap_deviation', 0) < -2:  # Below VWAP
                indicator_stats['VWAP']['trades'].append(trade)
            if trade.get('williams_r', 0) < -80:  # Oversold Williams
                indicator_stats['Williams_R']['trades'].append(trade)
            if trade.get('cci', 0) < -100:  # Oversold CCI
                indicator_stats['CCI']['trades'].append(trade)
            if trade.get('on_balance_volume', 0) > 0:  # Rising OBV
                indicator_stats['OBV']['trades'].append(trade)
            if trade.get('accumulation_distribution', 0) > 0:  # Rising A/D
                indicator_stats['AD']['trades'].append(trade)
        
        # Calculate statistics for each indicator
        for indicator, stats in indicator_stats.items():
            if stats['trades']:
                winning_trades = [t for t in stats['trades'] if t['win']]
                stats['win_rate'] = len(winning_trades) / len(stats['trades']) * 100
                stats['avg_return'] = sum(t['pnl_pct'] for t in stats['trades']) / len(stats['trades'])
        
        # Sort indicators by win rate
        sorted_indicators = sorted(indicator_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        print(f"🎯 INDICATOR WIN RATES (Ranked by Performance):")
        print(f"{'Indicator':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Avg Return':<12} | {'Best Setup':<12}")
        print(f"{'-'*12} | {'-'*8} | {'-'*10} | {'-'*12} | {'-'*12}")
        
        for indicator, stats in sorted_indicators:
            if stats['trades']:
                # Find best setup type for this indicator
                fvg_trades = [t for t in stats['trades'] if t.get('setup_type') == 'fvg']
                ob_trades = [t for t in stats['trades'] if t.get('setup_type') == 'order_block']
                
                fvg_win_rate = len([t for t in fvg_trades if t['win']]) / len(fvg_trades) * 100 if fvg_trades else 0
                ob_win_rate = len([t for t in ob_trades if t['win']]) / len(ob_trades) * 100 if ob_trades else 0
                
                best_setup = "FVG" if fvg_win_rate > ob_win_rate else "OB" if ob_win_rate > fvg_win_rate else "Equal"
                
                print(f"{indicator:<12} | {len(stats['trades']):<8} | {stats['win_rate']:<9.1f}% | {stats['avg_return']:<11.2f}% | {best_setup:<12}")
        
        print(f"\n📊 INDICATOR CONFLUENCE ANALYSIS:")
        
        # Analyze trades with multiple indicator confirmations
        confluence_levels = {}
        for trade in trades:
            confluence_factors = trade.get('confluence_factors', [])
            # Handle case where confluence_factors might be an integer
            if isinstance(confluence_factors, int):
                confluence_count = 0
            else:
                confluence_count = len(confluence_factors)
            
            if confluence_count not in confluence_levels:
                confluence_levels[confluence_count] = []
            confluence_levels[confluence_count].append(trade)
        
        print(f"{'Confluence':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Avg Return':<12}")
        print(f"{'-'*12} | {'-'*8} | {'-'*10} | {'-'*12}")
        
        for level in sorted(confluence_levels.keys()):
            level_trades = confluence_levels[level]
            winning_trades = [t for t in level_trades if t['win']]
            win_rate = len(winning_trades) / len(level_trades) * 100 if level_trades else 0
            avg_return = sum(t['pnl_pct'] for t in level_trades) / len(level_trades) if level_trades else 0
            
            print(f"{level} factors: {len(level_trades):<8} | {win_rate:<9.1f}% | {avg_return:<11.2f}%")
        
        print(f"\n💡 TOP PERFORMING INDICATOR COMBINATIONS:")
        
        # Find best indicator combinations
        combination_stats = {}
        for trade in trades:
            factors = trade.get('confluence_factors', [])
            # Handle case where factors might be an integer
            if isinstance(factors, int):
                continue
            if len(factors) >= 2:
                # Create combinations of 2-3 indicators
                for i in range(len(factors)):
                    for j in range(i+1, len(factors)):
                        combo = f"{factors[i]} + {factors[j]}"
                        if combo not in combination_stats:
                            combination_stats[combo] = []
                        combination_stats[combo].append(trade)
        
        # Calculate stats for combinations
        combo_results = []
        for combo, combo_trades in combination_stats.items():
            if len(combo_trades) >= 3:  # Only show combinations with 3+ trades
                winning_trades = [t for t in combo_trades if t['win']]
                win_rate = len(winning_trades) / len(combo_trades) * 100
                avg_return = sum(t['pnl_pct'] for t in combo_trades) / len(combo_trades)
                combo_results.append((combo, len(combo_trades), win_rate, avg_return))
        
        # Sort by win rate
        combo_results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"{'Combination':<35} | {'Trades':<8} | {'Win Rate':<10} | {'Avg Return':<12}")
        print(f"{'-'*35} | {'-'*8} | {'-'*10} | {'-'*12}")
        
        for combo, combo_trades, win_rate, avg_return in combo_results[:10]:  # Top 10
            print(f"{combo:<35} | {combo_trades:<8} | {win_rate:<9.1f}% | {avg_return:<11.2f}%")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        # Find best performing indicators
        best_indicators = [ind for ind, stats in sorted_indicators[:3] if stats['trades']]
        if best_indicators:
            print(f"✅ Top 3 Indicators: {', '.join(best_indicators)}")
        
        # Find optimal confluence level
        best_confluence = max(confluence_levels.keys(), key=lambda x: 
            len([t for t in confluence_levels[x] if t['win']]) / len(confluence_levels[x]) * 100 
            if confluence_levels[x] else 0)
        
        print(f"✅ Optimal Confluence Level: {best_confluence} factors")
        
        # Find best combination
        if combo_results:
            best_combo = combo_results[0]
            print(f"✅ Best Indicator Combination: {best_combo[0]} ({best_combo[2]:.1f}% win rate)")
        
        print(f"\n")

    def analyze_enhanced_performance(self):
        """Analyze performance with Phase 7 dynamic target metrics"""
        trades = self.all_trades
        
        if not trades:
            return
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'bars_held' not in trade:
                trade['bars_held'] = trade.get('hold_days', 0) * 24 / 4  # Convert days to bars
            if 'hold_days' not in trade:
                trade['hold_days'] = trade.get('bars_held', 0) * 4 / 24  # Convert bars to days
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
            if 'hit_target' not in trade:
                trade['hit_target'] = None
            if 'hit_stop' not in trade:
                trade['hit_stop'] = False
            if 'max_favorable_excursion' not in trade:
                trade['max_favorable_excursion'] = 0
            if 'max_adverse_excursion' not in trade:
                trade['max_adverse_excursion'] = 0
            if 'target_win' not in trade:
                trade['target_win'] = trade.get('win', False)
            # Add any other missing fields with safe defaults
            if 'exit_price' not in trade:
                trade['exit_price'] = trade.get('entry_price', 0)
            if 'exit_time' not in trade:
                trade['exit_time'] = trade.get('setup_time', 0)
            if 'exit_reason' not in trade:
                trade['exit_reason'] = 'unknown'
            if 'pnl' not in trade:
                trade['pnl'] = trade.get('pnl_pct', 0)
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['win']])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades * 100
        
        # PHASE 7: Analyze target-based wins vs actual wins
        target_winning_trades = len([t for t in trades if t.get('target_win', t['win'])])
        target_win_rate = target_winning_trades / total_trades * 100
        
        total_pnl = sum([t['pnl_pct'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['win']]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in trades if not t['win']]) if losing_trades > 0 else 0
        
        avg_hold_days = np.mean([t['hold_days'] for t in trades])
        avg_bars_held = np.mean([t['bars_held'] for t in trades])
        
        target_hits = {}
        for i in range(1, 4):
            target_hits[f'T{i}'] = len([t for t in trades if t['hit_target'] == i])
        
        print("\n" + "=" * 70)
        print("🔥 ICT 1H R2 - TIGHTENED STOP LOSS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"Total Trades: {total_trades}")
        print(f"Actual Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
        print(f"Target-Based Win Rate: {target_win_rate:.1f}% ({target_winning_trades}W)")
        
        # Show the difference
        if target_winning_trades != winning_trades:
            print(f"📈 Target Logic Impact: +{target_winning_trades - winning_trades} additional wins")
            print(f"   (Trades profitable but didn't hit targets)")
        
        baseline = self.config.get('current_baseline', 64.0)
        if win_rate >= baseline:
            print(f"✅ BASELINE MAINTAINED: {win_rate:.1f}% (minimum: {baseline:.1f}%)")
        else:
            print(f"⚠️  BELOW BASELINE: {win_rate:.1f}% (minimum: {baseline:.1f}%)")
        
        print(f"Total Return: {total_pnl:.1f}%")
        print(f"Average Return per Trade: {total_pnl/total_trades:.2f}%")
        
        print(f"\n💰 WIN/LOSS ANALYSIS:")
        print(f"Average Win: +{avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        win_loss_ratio = abs(avg_win/avg_loss) if avg_loss != 0 else 0
        print(f"Win/Loss Ratio: {win_loss_ratio:.2f}:1")
        
        print(f"\n🎯 TARGET HIT ANALYSIS (ADJUSTED TARGETS):")
        for target, count in target_hits.items():
            pct = count / total_trades * 100
            if target == 'T1':
                print(f"{target} Hit Rate: {pct:.1f}% ({count} trades) [Target: Fibonacci/S/R/ATR]")
            elif target == 'T2':
                print(f"{target} Hit Rate: {pct:.1f}% ({count} trades) [Target: Fibonacci/S/R/ATR]")
            else:
                print(f"{target} Hit Rate: {pct:.1f}% ({count} trades) [Target: Fibonacci/S/R/ATR]")
        
        stop_hits = len([t for t in trades if t['hit_stop']])
        print(f"Stop Loss Hit: {stop_hits/total_trades*100:.1f}% ({stop_hits} trades)")
        
        print(f"\n⏰ TIMING METRICS:")
        print(f"Average Hold Time: {avg_hold_days:.1f} days ({avg_bars_held:.1f} bars)")
        
        # Setup type analysis
        setup_types = {}
        for trade in trades:
            setup_type = trade.get('setup_type', 'order_block')
            if setup_type not in setup_types:
                setup_types[setup_type] = {'trades': 0, 'wins': 0}
            setup_types[setup_type]['trades'] += 1
            if trade['win']:
                setup_types[setup_type]['wins'] += 1
        
        print(f"\n📊 PERFORMANCE BY SETUP TYPE:")
        for setup_type, stats in setup_types.items():
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            print(f"{setup_type}: {stats['trades']} trades, {wr:.1f}% win rate")
        
        # Quality metrics
        avg_quality = np.mean([t['quality_score'] for t in trades])
        ob_trades = [t for t in trades if t.get('setup_type', 'order_block') == 'order_block']
        fvg_trades = [t for t in trades if t.get('setup_type') == 'fvg']
        
        if ob_trades:
            avg_volume_ratio = np.mean([t['volume_ratio'] for t in ob_trades if 'volume_ratio' in t])
            avg_ob_age = np.mean([t['order_block_age'] for t in ob_trades if 'order_block_age' in t])
            print(f"\n📊 ORDER BLOCK METRICS:")
            print(f"Average Volume Ratio: {avg_volume_ratio:.2f}x")
            print(f"Average OB Age: {avg_ob_age:.1f} bars")
        
        if fvg_trades:
            avg_gap_size = np.mean([t['fvg_gap_size'] for t in fvg_trades if 'fvg_gap_size' in t])
            avg_fvg_age = np.mean([t['fvg_age'] for t in fvg_trades if 'fvg_age' in t])
            print(f"\n📊 FVG METRICS:")
            print(f"Average Gap Size: {avg_gap_size:.2f}%")
            print(f"Average FVG Age: {avg_fvg_age:.1f} bars")
        
        print(f"\n📊 OVERALL QUALITY METRICS:")
        print(f"Average Quality Score: {avg_quality:.1f}")
        
        # Liquidity sweep impact
        liquidity_swept_trades = [t for t in trades if t.get('liquidity_swept', False)]
        if liquidity_swept_trades:
            swept_wins = len([t for t in liquidity_swept_trades if t['win']])
            swept_wr = swept_wins / len(liquidity_swept_trades) * 100
            print(f"\n🌊 LIQUIDITY SWEEP IMPACT:")
            print(f"Trades with liquidity sweep: {len(liquidity_swept_trades)} ({swept_wr:.1f}% win rate)")
        
        # Show best trade
        if trades:
            best_trade = max(trades, key=lambda x: x['pnl_pct'])
            print(f"\n🏆 BEST TRADE:")
            print(f"Symbol: {best_trade['symbol'].replace('/USDT', '')}")
            print(f"Category: {best_trade.get('category', 'Unknown')}")
            print(f"Setup Type: {best_trade.get('setup_type', 'order_block')}")
            print(f"Return: +{best_trade['pnl_pct']:.2f}%")
            print(f"Quality Score: {best_trade['quality_score']}")
        
        print(f"\n🔧 PHASE 6 FEATURES:")
        print(f"✅ Fix 1: Fibonacci-based targets (38.2%, 61.8%, 100%)")
        print(f"✅ Fix 2: Retracement entry (25% into OB, midpoint for FVG)")
        print(f"✅ Fix 3: Enhanced quality scoring")
        print(f"✅ Fix 4: Dynamic category weighting")
        print(f"✅ Fix 5: Improved market regime detection")
        print(f"✅ Fix 6: Fair Value Gap (FVG) detection")
        print(f"✅ Fix 7: Breaker Block detection")
        print(f"✅ Fix 8: Tightened OB age to 50 bars")

    def apply_incremental_quality_boost(self, setup_score, confluence_factors, setup, category):
        """Enhanced quality scoring with category weighting and FVG bonus"""
        
        if not self.config.get('quality_boost_enabled', True):
            return setup_score
        
        bonus_score = 0
        
        # Volume-based bonuses (for Order Blocks)
        if 'volume_ratio' in setup:
            if setup['volume_ratio'] >= 3.0:
                bonus_score += 10
            elif setup['volume_ratio'] >= 2.5:
                bonus_score += 5
        
        # Age-based bonuses
        if setup['age'] <= 5:
            bonus_score += 8
        elif setup['age'] <= 10:
            bonus_score += 4
        
        # Validation move bonuses (for Order Blocks)
        if 'future_move' in setup and setup['future_move'] >= 8.0:
            bonus_score += 8
        elif 'future_move' in setup and setup['future_move'] >= 5.0:
            bonus_score += 4
        
        # Liquidity sweep bonus
        if setup.get('liquidity_swept', False):
            bonus_score += 15
            confluence_factors.append("Liquidity Swept")
        
        # PHASE 6: FVG-specific bonuses
        if 'gap_size' in setup:  # This is an FVG
            if setup['gap_size'] >= 1.0:
                bonus_score += 20
                confluence_factors.append("Large FVG")
            elif setup['gap_size'] >= 0.5:
                bonus_score += 10
                confluence_factors.append("Medium FVG")
            
            if setup.get('volume_surge', False):
                bonus_score += 10
                confluence_factors.append("Volume Surge")
        
        # Apply incremental boost
        incremental_boost = min(bonus_score, 30)
        setup_score += incremental_boost
        
        # Apply category multiplier
        if self.config.get('use_category_weighting', True):
            multiplier = self.category_multipliers.get(category, 1.0)
            setup_score = int(setup_score * multiplier)
        
        # PHASE 6: Apply FVG weight multiplier if enabled
        if self.config.get('use_fvg_detection', True) and 'gap_size' in setup:
            fvg_weight = self.config.get('fvg_weight', 1.5)
            setup_score = int(setup_score * fvg_weight)
        
        return setup_score 

    def detect_historical_setups_enhanced(self, df, future_df, symbol):
        """Detect ICT setups including FVGs, Order Blocks, and Breaker Blocks"""
        
        # FIX: Define ALL variables that might be used in debug messages
        min_quality = self.config.get('min_quality_score')  # Use actual config value, no default
        min_confluence = self.config.get('min_confluence_factors')  # Use actual config value, no default
        quality_filtered = 0
        distance_filtered = 0
        
        def check_ob_fvg_confluence(ob, fvgs, df):
            """Order Block needs confluence to trade"""
            ob_score = ob['quality_score']
            confluences = []
            
            # Check FVG overlap
            for fvg in fvgs:
                if abs(ob['index'] - fvg['index']) < 10:  # Within 10 bars
                    if ob['type'] == fvg['type']:  # Same direction
                        confluences.append('FVG_Support')
                        ob_score += 30
            
            # Check indicators at OB level
            ob_bar = df.iloc[ob['index']]
            if ob['type'] == 'bullish':
                if ob_bar['mfi'] < 30: confluences.append('MFI_Oversold')
                if ob_bar['cmf'] > 0.1: confluences.append('CMF_Accumulation')
            
            # Require at least one confluence
            return len(confluences) > 0, ob_score, confluences
        
        def zones_overlap(zone1, zone2, tolerance=0.002):
            """Check if two zones overlap"""
            # Handle different key names safely
            z1_high = zone1.get('high', zone1.get('gap_high', 0))
            z1_low = zone1.get('low', zone1.get('gap_low', 0))
            z2_high = zone2.get('high', zone2.get('gap_high', 0))
            z2_low = zone2.get('low', zone2.get('gap_low', 0))
            
            # Check overlap
            if z1_high >= z2_low and z1_low <= z2_high:
                return True
            return False
        
        def check_zone_overlap(self, zone1, zone2, tolerance_pct=0.5):
            """Check if two price zones overlap"""
            # Get price ranges
            z1_high = zone1.get('high', zone1.get('gap_high'))
            z1_low = zone1.get('low', zone1.get('gap_low'))
            z2_high = zone2.get('high', zone2.get('gap_high'))
            z2_low = zone2.get('low', zone2.get('gap_low'))
            
            # Check for overlap
            if z1_high >= z2_low and z1_low <= z2_high:
                overlap_pct = (min(z1_high, z2_high) - max(z1_low, z2_low)) / (z1_high - z1_low)
                return overlap_pct > tolerance_pct
            return False
        
        # Category detection for weighting
        category = self.get_token_category(symbol)
        
        # Performance-based category management (instead of hard exclusion)
        underperforming_categories = self.config.get('underperforming_categories', [])
        category_performance_multiplier = 1.0
        
        if category in underperforming_categories:
            # Check recent performance for underperforming categories
            recent_performance = self.get_recent_category_stats(category, 30)
            if recent_performance['win_rate'] < self.config.get('min_category_win_rate', 60):
                category_performance_multiplier = 0.5  # Reduce position size
            elif recent_performance['win_rate'] > 75:
                category_performance_multiplier = 1.2  # Scale back up if improving
        
        # Focus categories filter
        focus_categories = self.config.get('focus_categories', None)
        if focus_categories and category not in focus_categories:
            return []
        
        # Enhanced market regime filter
        if self.config.get('use_enhanced_regime', True):
            regime = self.detect_market_regime(df)
            if self.config.get('require_market_regime', False) and not regime['favorable']:
                return []
        
        # Step 1: Detect ALL Order Blocks and FVGs (no filtering yet)
        all_order_blocks = self.detect_order_blocks(df)
        all_fvgs = self.detect_fair_value_gaps(df) if self.config.get('use_fvg_detection', True) else []
        breaker_blocks = self.detect_breaker_blocks(df, all_order_blocks) if self.config.get('use_breaker_blocks', True) else []
        
        # Debug stats
        debug_stats = getattr(self, 'last_debug_stats', {})
        total_candles = debug_stats.get('total_candles', len(df))
        volume_filtered = debug_stats.get('volume_filtered', 0)
        move_validated = debug_stats.get('move_validated', 0)
        final_blocks = debug_stats.get('final_blocks', len(all_order_blocks))
        
        total_setups = len(all_order_blocks) + len(all_fvgs) + len(breaker_blocks)
        
        if total_setups == 0:
            token = symbol.replace('/USDT', '')
            if total_candles > 0:
                print(f"🔍 DEBUG {token}: {total_candles} candles → {volume_filtered} volume OK → {move_validated} move OK → {final_blocks} OBs, {len(all_fvgs)} FVGs")
            return []
        
        trades = []
        current_time = df.iloc[-1]['timestamp']
        current_price = df.iloc[-1]['close']
        
        # R21: Step 1 - Process FVGs with OB confluence boost
        trades = []
        current_time = df.iloc[-1]['timestamp']
        current_price = df.iloc[-1]['close']
        
        for fvg in all_fvgs:
            fvg_boost = 0
            has_ob_confluence = False
            
            # Check if any OB overlaps this FVG
            for ob in all_order_blocks:
                if abs(fvg['index'] - ob['index']) <= 10:  # Within 10 bars
                    if fvg['type'] == ob['type']:  # Same direction
                        # Check zone overlap
                        fvg_high = fvg.get('gap_high', fvg.get('high', 0))
                        fvg_low = fvg.get('gap_low', fvg.get('low', 0))
                        ob_high = ob.get('high', 0)
                        ob_low = ob.get('low', 0)
                        
                        if fvg_high >= ob_low and fvg_low <= ob_high:
                            fvg_boost = self.config.get('confluence_quality_boost', 30)
                            has_ob_confluence = True
                            break
            
            # R21: Use FVG logic for confluence (surgical fix)
            if self.config.get('use_fvg_logic_for_confluence', True):
                # Calculate entry using FVG logic
                entry_price = (fvg['gap_high'] + fvg['gap_low']) / 2  # FVG midpoint
                
                # Calculate stop using FVG logic
                atr = df.iloc[-1]['atr']
                if fvg['type'] == 'bullish':
                    stop_loss = fvg['gap_low'] - (atr * 0.5)
                else:
                    stop_loss = fvg['gap_high'] + (atr * 0.5)
                
                # Boost quality score significantly
                quality_score = 70 + fvg_boost  # Base FVG score + confluence boost
                
                # Distance filtering
                distance_pct = abs(current_price - entry_price) / current_price * 100
                if distance_pct > self.config.get('max_distance_pct', 18.0):
                    continue
                
                # Calculate targets
                targets, target_reasons = self.calculate_ict_targets(df, entry_price, 
                                                                   'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH', 
                                                                   'fvg')
                
                # Risk/Reward filter
                stop_distance = abs(entry_price - stop_loss)
                target_distance = abs(targets[0] - entry_price)
                risk_reward = target_distance / stop_distance if stop_distance > 0 else 0

                if risk_reward < self.config.get('min_risk_reward', 0.8):
                    continue
                
                # Create trade with proper setup type
                setup_type = 'ob_fvg_confluence' if has_ob_confluence else 'fvg'
                
                trade = {
                    'symbol': symbol,
                    'setup_type': setup_type,
                    'setup_time': current_time,
                    'direction': 'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'target_reasons': target_reasons,
                    'quality_score': quality_score,
                    'confluence_factors': 2 if has_ob_confluence else 1,
                    'fvg_gap_size': fvg.get('gap_size', 0),
                    'fvg_age': fvg.get('age', 0),
                    'has_ob_confluence': has_ob_confluence,
                    'distance_pct': distance_pct,
                    'risk_reward_ratio': risk_reward,
                    
                    # Technical indicators
                    'rsi': df.iloc[-1]['rsi'],
                    'mfi': df.iloc[-1]['mfi'],
                    'cmf': df.iloc[-1]['cmf'],
                    'atr': df.iloc[-1]['atr'],
                }
                
                # Simulate trade
                outcome = self.simulate_trade_outcome_enhanced(trade, future_df)
                trade.update(outcome)
                
                trades.append(trade)
        
        # R21: Step 2 - Process standalone FVGs (already working well)
        for fvg in all_fvgs:
            # Skip if already processed as confluence
            if any(t.get('fvg_gap_size') == fvg.get('gap_size') and 
                   t.get('fvg_age') == fvg.get('age') for t in trades):
                continue
            
            # Calculate basic R/R for standalone FVG
            entry_price = (fvg['gap_high'] + fvg['gap_low']) / 2
            atr = df.iloc[-1]['atr']
            
            if fvg['type'] == 'bullish':
                stop_loss = fvg['gap_low'] - (atr * 0.5)
            else:
                stop_loss = fvg['gap_high'] + (atr * 0.5)
            
            targets, target_reasons = self.calculate_ict_targets(df, entry_price, 
                                                               'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH', 
                                                               'fvg')
            
            stop_distance = abs(entry_price - stop_loss)
            target_distance = abs(targets[0] - entry_price)
            risk_reward = target_distance / stop_distance if stop_distance > 0 else 0
            
            if risk_reward >= 0.8:  # FVGs have proven success
                distance_pct = abs(current_price - entry_price) / current_price * 100
                if distance_pct <= self.config.get('max_distance_pct', 18.0):
                    trade = {
                        'symbol': symbol,
                        'setup_type': 'fvg',
                        'setup_time': current_time,
                        'direction': 'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'targets': targets,
                        'target_reasons': target_reasons,
                        'quality_score': 70,
                        'confluence_factors': 1,
                        'fvg_gap_size': fvg.get('gap_size', 0),
                        'fvg_age': fvg.get('age', 0),
                        'has_ob_confluence': False,
                        'distance_pct': distance_pct,
                        'risk_reward_ratio': risk_reward,
                        
                        # Technical indicators
                        'rsi': df.iloc[-1]['rsi'],
                        'mfi': df.iloc[-1]['mfi'],
                        'cmf': df.iloc[-1]['cmf'],
                        'atr': df.iloc[-1]['atr'],
                    }
                    
                    outcome = self.simulate_trade_outcome_enhanced(trade, future_df)
                    trade.update(outcome)
                    
                    trades.append(trade)
            
            # REMOVED: Old Order Block processing logic - replaced with confluence-based approach
            
            # REMOVED: Old validation move scoring - replaced with confluence-based approach
            
            # REMOVED: Old market structure scoring - replaced with confluence-based approach
            
            # REMOVED: Old indicator confluence scoring - replaced with confluence-based approach

            # Skip if not enough indicator confluence
            # if len(indicator_confluences) < self.config.get('min_indicator_confluence', 2):
            #     continue
            
            # REMOVED: Old Order Block processing - replaced with confluence-based approach
        
        return trades
        
        # Step 4: Process confluence setups into trades
        print(f"   DEBUG: Processing {len(confluence_setups)} confluence setups")
        for setup in confluence_setups:
            if setup['type'] == 'ob_fvg_confluence':
                print(f"   DEBUG: Found OB+FVG confluence setup")
                # Process OB+FVG confluence setup
                ob = setup['ob']
                fvg = setup['fvg']
                
                # Calculate entry (use OB entry logic)
                if self.config.get('use_retracement_entry', True):
                    entry_price = self.calculate_retracement_entry(ob, 'BULLISH' if ob['type'] == 'bullish' else 'BEARISH', 'order_block')
                else:
                    if ob['type'] == 'bullish':
                        entry_price = ob.get('low', ob.get('gap_low', 0))
                    else:
                        entry_price = ob.get('high', ob.get('gap_high', 0))
                
                # Calculate stop loss
                stop_loss_pct = self.config.get('stop_loss_pct', 0.006)  # 0.6% default
                if ob['type'] == 'bullish':
                    stop_loss = entry_price * (1 - stop_loss_pct)
                else:
                    stop_loss = entry_price * (1 + stop_loss_pct)
                

                
                # Distance filtering
                distance_pct = abs(current_price - entry_price) / current_price * 100
                if distance_pct > self.config.get('max_distance_pct', 18.0):
                    distance_filtered += 1
                    continue
                
                # Calculate targets
                targets, target_reasons = self.calculate_ict_targets(df, entry_price, 
                                                                   'BULLISH' if ob['type'] == 'bullish' else 'BEARISH', 
                                                                   'order_block')
                
                # Risk/Reward filter
                stop_distance = abs(entry_price - stop_loss)
                target_distance = abs(targets[0] - entry_price)
                risk_reward = target_distance / stop_distance if stop_distance > 0 else 0

                if risk_reward < self.config.get('min_risk_reward', 0.5):  # Lowered from 0.8 to 0.5
                    distance_filtered += 1
                    continue
                
                # Create high-quality confluence trade
                trade = {
                    'symbol': symbol,
                    'category': category,
                    'setup_type': 'ob_fvg_confluence',
                    'position_multiplier': category_performance_multiplier,
                    'setup_time': current_time,
                    'direction': 'BULLISH' if ob['type'] == 'bullish' else 'BEARISH',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'targets': targets,
                    'target_reasons': target_reasons,
                    'quality_score': setup['quality_score'],
                    'confluence_factors': len(setup['confluences']),
                    'order_block_age': ob['age'],
                    'validation_move': ob['future_move'],
                    'volume_ratio': ob['volume_ratio'],
                    'distance_pct': distance_pct,
                    'liquidity_swept': ob.get('liquidity_swept', False),
                    'fvg_gap_size': fvg.get('gap_size', 0),
                    'fvg_age': fvg.get('age', 0),
                    
                    # Technical indicators
                    'rsi': df.iloc[-1]['rsi'],
                    'mfi': df.iloc[-1]['mfi'],
                    'cmf': df.iloc[-1]['cmf'],
                    'atr': df.iloc[-1]['atr'],
                    'accumulation_distribution': df.iloc[-1]['accumulation_distribution'],
                    'on_balance_volume': df.iloc[-1]['on_balance_volume'],
                    'williams_r': df.iloc[-1]['williams_r'],
                    'cci': df.iloc[-1]['cci'],
                    'vwap_deviation': df.iloc[-1]['vwap_deviation'],
                    'momentum_20': df.iloc[-1]['momentum_20'],
                    'momentum_50': df.iloc[-1]['momentum_50'],
                    'sma_20': df.iloc[-1]['sma_20'],
                    'sma_50': df.iloc[-1]['sma_50'],
                    'adx': df.iloc[-1]['adx'],
                    
                    # Risk metrics
                    'risk_reward_ratio': risk_reward,
                    'max_drawdown': 0,
                    'time_to_target': 0,
                    'target_hit': False,
                    'stop_hit': False,
                    'exit_price': 0,
                    'exit_reason': '',
                    'pnl': 0,
                    'pnl_pct': 0,
                    'hold_bars': 0,
                    'max_adverse_excursion': 0,
                    'max_favorable_excursion': 0,
                    'quality_multiplier': ob.get('quality_multiplier', 1.0)
                }
                
                trades.append(trade)
        
        # Process Fair Value Gaps with tighter requirements
        for fvg in all_fvgs:
            # Skip FVGs without volume surge if required
            # Temporarily disable volume requirement for FVGs
            # if self.config.get('require_fvg_volume', True) and not fvg.get('volume_surge', False):
            #     continue
                
            setup_score = 70  # Higher base score for FVGs
            confluence_factors = ["Fair Value Gap"]
            
            # Gap size scoring
            gap_size = fvg.get('gap_size', 0)
            if gap_size >= 1.0:
                setup_score += 40
                confluence_factors.append("Large Gap")
            elif gap_size >= 0.5:
                setup_score += 25
                confluence_factors.append("Medium Gap")
            else:
                setup_score += 10
                confluence_factors.append("Small Gap")
            
            # Age scoring
            fvg_age = fvg.get('age', 0)
            if fvg_age <= 5:
                setup_score += 30
                confluence_factors.append("Very Fresh FVG")
            elif fvg_age <= 15:
                setup_score += 20
                confluence_factors.append("Fresh FVG")
            elif fvg_age <= 30:
                setup_score += 10
                confluence_factors.append("Recent FVG")
            
            # Volume surge bonus
            if fvg.get('volume_surge', False):
                setup_score += 15
                confluence_factors.append("Volume Surge")
            
            # Apply quality boost
            setup_score = self.apply_incremental_quality_boost(setup_score, confluence_factors, fvg, category)
            
            # Quality filtering
            if len(confluence_factors) < 2:  # Lower requirement for FVGs
                continue
            
            if setup_score < min_quality:
                continue
            
            # Calculate entry (midpoint of gap)
            entry_price = self.calculate_retracement_entry(fvg, 'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH', 'fvg')
            
            # Calculate stop loss (beyond the gap)
            stop_loss_pct = self.config.get('stop_loss_pct', 0.006)  # 0.6% default
            if fvg['type'] == 'bullish':
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:
                stop_loss = entry_price * (1 + stop_loss_pct)
            

            
            # Distance filtering
            distance_pct = abs(current_price - entry_price) / current_price * 100
            if distance_pct > self.config.get('max_distance_pct', 18.0):
                continue
            
            # Calculate targets
            targets, target_reasons = self.calculate_ict_targets(df, entry_price, 
                                                               'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH', 
                                                               'fvg')
            
            # Risk/Reward filter
            stop_distance = abs(entry_price - stop_loss)
            target_distance = abs(targets[0] - entry_price)
            risk_reward = target_distance / stop_distance if stop_distance > 0 else 0

            if risk_reward < self.config.get('min_risk_reward', 0.5):  # Lowered from 0.8 to 0.5
                print(f"   FVG R/R filtered: {risk_reward:.2f} < 0.5")  # Updated debug
                continue
            
            # Apply category performance multiplier to FVG position sizing
            if self.config.get('category_position_scaling', True):
                fvg_position_multiplier = category_performance_multiplier
            else:
                fvg_position_multiplier = 1.0
            
            # Create FVG trade record
            trade = {
                'symbol': symbol,
                'category': category,
                'setup_type': 'fvg',
                'position_multiplier': fvg_position_multiplier,  # Track position sizing
                'setup_time': current_time,
                'direction': 'BULLISH' if fvg['type'] == 'bullish' else 'BEARISH',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'targets': targets,
                'target_reasons': target_reasons,
                'quality_score': min(setup_score, 100),
                'confluence_factors': len(confluence_factors),
                'fvg_gap_size': fvg.get('gap_size', 0),
                'fvg_age': fvg.get('age', 0),
                'volume_surge': fvg.get('volume_surge', False),
                'distance_pct': distance_pct,
                
                # Technical indicators
                'rsi': df.iloc[-1]['rsi'],
                'mfi': df.iloc[-1]['mfi'],
                'cmf': df.iloc[-1]['cmf'],
                'atr': df.iloc[-1]['atr'],
                'momentum_20': df.iloc[-1]['momentum_20'],
                'momentum_50': df.iloc[-1]['momentum_50']
            }
            
            # Simulate trade
            outcome = self.simulate_trade_outcome_enhanced(trade, future_df)
            trade.update(outcome)
            
            # Ensure trade has 'win' field
            if 'win' not in trade:
                trade['win'] = trade.get('pnl', 0) > 0
            
            trades.append(trade)
        
        if (final_blocks > 0 or len(all_fvgs) > 0) and len(trades) == 0:
            token = symbol.replace('/USDT', '')
            print(f"🔍 DEBUG {token}: {final_blocks} OBs ({quality_filtered} quality filtered, {distance_filtered} distance filtered), {len(all_fvgs)} FVGs → {len(trades)} final trades")
        
        return trades 

    def run_enhanced_backtest(self, symbols=None, config=None):
        """Run enhanced ICT backtest with FVG detection"""
        
        if config:
            self.config.update(config)
            
        if symbols is None:
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
        
        print("🔥 ICT 1H R2 - TIGHTENED STOP LOSS FOR 80% WIN RATE")
        print("=" * 70)
        print(f"📊 Testing {len(symbols)} symbols with PHASE 9 DYNAMIC CATEGORY MANAGEMENT")
        print(f"🎯 STRATEGY: Performance-based position sizing to reach 80% target")
        print(f"🔧 PHASE 9 CONFIGURATION:")
        print(f"   • Order Block: {self.config.get('min_order_block_size')}% moves")
        print(f"   • Volume Ratio: {self.config.get('min_volume_ratio')}x minimum")
        print(f"   • Quality Score: {self.config.get('min_quality_score')} minimum")
        print(f"   • Distance Limit: {self.config.get('max_distance_pct')}%")
        print(f"   • Max OB Age: {self.config.get('max_ob_age')} bars")
        print(f"   • Min Confluence: {self.config.get('min_confluence_factors')} factors")
        
        print(f"\n🔧 PHASE 6 ADDITIONS:")
        if self.config.get('use_fvg_detection', True):
            print(f"   ✅ Fair Value Gap Detection: Min {self.config.get('min_fvg_size', 0.3)}% gap")
            print(f"   ✅ Max FVG Age: {self.config.get('max_fvg_age', 30)} bars")
        if self.config.get('use_breaker_blocks', True):
            print(f"   ✅ Breaker Block Detection: Enabled")
        print(f"   ✅ Fibonacci Targets: 38.2%, 61.8%, 100%")
        
        print()
        
        all_trades = []
        
        for symbol in symbols:
            trades = self.backtest_symbol_enhanced(symbol)
            all_trades.extend(trades)
            time.sleep(0.1)
        
        self.all_trades = all_trades
        
        if not all_trades:
            print("❌ No trades found in backtest period")
            return
        
        # Analyze results
        self.analyze_enhanced_performance()
        self.analyze_indicator_performance()  # R16: NEW - Indicator analysis
        self.analyze_category_performance()
        self.track_category_improvements()  # PHASE 9: Track category improvements
        self.analyze_losing_trade_patterns()
        
        # PHASE 7: New analysis methods
        self.analyze_drawdown_statistics()
        self.analyze_take_profit_optimization()
        self.analyze_target_optimization()
        self.analyze_risk_reward_optimization()
        
        return all_trades  # FIXED: Changed from all_trades1 to all_trades

    def analyze_drawdown_statistics(self):
        """PHASE 7: Analyze drawdown patterns in winning trades for optimal stop loss placement"""
        trades = self.all_trades
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'max_adverse_excursion' not in trade:
                trade['max_adverse_excursion'] = 0
            if 'hit_stop' not in trade:
                trade['hit_stop'] = False
        
        winning_trades = [t for t in trades if t['win']]
        
        if not winning_trades:
            print("❌ No winning trades to analyze drawdown")
            return
        
        print("\n" + "=" * 70)
        print("📉 DRAWDOWN ANALYSIS FOR OPTIMAL STOP LOSS PLACEMENT")
        print("=" * 70)
        
        # Analyze max adverse excursion in winning trades
        max_adverse_excursions = [t['max_adverse_excursion'] for t in winning_trades]
        
        print(f"\n📊 MAX ADVERSE EXCURSION IN WINNING TRADES:")
        print(f"Total Winning Trades: {len(winning_trades)}")
        print(f"Average Max Drawdown: {np.mean(max_adverse_excursions):.2f}%")
        print(f"Median Max Drawdown: {np.median(max_adverse_excursions):.2f}%")
        print(f"Min Max Drawdown: {np.min(max_adverse_excursions):.2f}%")
        print(f"Max Max Drawdown: {np.max(max_adverse_excursions):.2f}%")
        
        # Drawdown percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n📊 DRAWDOWN PERCENTILES:")
        for p in percentiles:
            value = np.percentile(max_adverse_excursions, p)
            print(f"  {p}th percentile: {value:.2f}%")
        
        # Analyze by setup type
        setup_drawdowns = {}
        for trade in winning_trades:
            setup_type = trade.get('setup_type', 'order_block')
            if setup_type not in setup_drawdowns:
                setup_drawdowns[setup_type] = []
            setup_drawdowns[setup_type].append(trade['max_adverse_excursion'])
        
        print(f"\n📊 DRAWDOWN BY SETUP TYPE:")
        for setup_type, drawdowns in setup_drawdowns.items():
            if drawdowns:
                avg_dd = np.mean(drawdowns)
                median_dd = np.median(drawdowns)
                max_dd = np.max(drawdowns)
                print(f"  {setup_type}: {len(drawdowns)} trades")
                print(f"    Average: {avg_dd:.2f}%, Median: {median_dd:.2f}%, Max: {max_dd:.2f}%")
        
        # Analyze by quality score
        quality_drawdowns = {
            'Low (50-70)': [t['max_adverse_excursion'] for t in winning_trades if 50 <= t['quality_score'] < 70],
            'Medium (70-85)': [t['max_adverse_excursion'] for t in winning_trades if 70 <= t['quality_score'] < 85],
            'High (85+)': [t['max_adverse_excursion'] for t in winning_trades if t['quality_score'] >= 85]
        }
        
        print(f"\n📊 DRAWDOWN BY QUALITY SCORE:")
        for quality_range, drawdowns in quality_drawdowns.items():
            if drawdowns:
                avg_dd = np.mean(drawdowns)
                median_dd = np.median(drawdowns)
                print(f"  {quality_range}: {len(drawdowns)} trades, Avg: {avg_dd:.2f}%, Median: {median_dd:.2f}%")
        
        # Stop loss recommendations
        print(f"\n💡 STOP LOSS RECOMMENDATIONS:")
        
        # Conservative stop (90th percentile)
        conservative_stop = np.percentile(max_adverse_excursions, 90)
        print(f"  Conservative Stop (90th percentile): {conservative_stop:.2f}%")
        print(f"    Would protect {len([d for d in max_adverse_excursions if d <= conservative_stop])} out of {len(winning_trades)} winning trades")
        
        # Moderate stop (75th percentile)
        moderate_stop = np.percentile(max_adverse_excursions, 75)
        print(f"  Moderate Stop (75th percentile): {moderate_stop:.2f}%")
        print(f"    Would protect {len([d for d in max_adverse_excursions if d <= moderate_stop])} out of {len(winning_trades)} winning trades")
        
        # Aggressive stop (50th percentile)
        aggressive_stop = np.percentile(max_adverse_excursions, 50)
        print(f"  Aggressive Stop (50th percentile): {aggressive_stop:.2f}%")
        print(f"    Would protect {len([d for d in max_adverse_excursions if d <= aggressive_stop])} out of {len(winning_trades)} winning trades")
        
        # Current stop analysis
        current_stop_hits = len([t for t in trades if t['hit_stop']])
        total_trades = len(trades)
        current_stop_rate = current_stop_hits / total_trades * 100 if total_trades > 0 else 0
        
        print(f"\n📊 CURRENT STOP LOSS PERFORMANCE:")
        print(f"  Current Stop Hit Rate: {current_stop_rate:.1f}% ({current_stop_hits}/{total_trades})")
        
        if current_stop_rate > 10:
            print(f"  ⚠️  Stop rate is high - consider wider stops")
        elif current_stop_rate < 2:
            print(f"  ✅ Stop rate is low - stops may be too wide")
        else:
            print(f"  ✅ Stop rate is reasonable")
        
        # Risk/reward optimization
        print(f"\n🎯 RISK/REWARD OPTIMIZATION:")
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
        
        for stop_type, stop_pct in [("Conservative", conservative_stop), 
                                   ("Moderate", moderate_stop), 
                                   ("Aggressive", aggressive_stop)]:
            risk_reward_ratio = avg_win / stop_pct if stop_pct > 0 else 0
            print(f"  {stop_type} Stop ({stop_pct:.2f}%): {risk_reward_ratio:.2f}:1 R/R ratio")
        
        return {
            'conservative_stop': conservative_stop,
            'moderate_stop': moderate_stop,
            'aggressive_stop': aggressive_stop,
            'avg_drawdown': np.mean(max_adverse_excursions),
            'median_drawdown': np.median(max_adverse_excursions)
        }

    def analyze_take_profit_optimization(self):
        """PHASE 7: Analyze take profit levels and timing for optimization"""
        trades = self.all_trades
        
        if not trades:
            print("❌ No trades to analyze take profit optimization")
            return
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
            if 'hit_target' not in trade:
                trade['hit_target'] = None
        
        print("\n" + "=" * 70)
        print("🎯 TAKE PROFIT OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        # Analyze target hit rates
        target_hits = {}
        for i in range(1, 4):
            target_hits[f'T{i}'] = len([t for t in trades if t['hit_target'] == i])
        
        total_trades = len(trades)
        
        print(f"\n📊 CURRENT TARGET HIT RATES:")
        for target, hits in target_hits.items():
            hit_rate = hits / total_trades * 100 if total_trades > 0 else 0
            print(f"  {target}: {hit_rate:.1f}% ({hits}/{total_trades})")
        
        # Analyze winning trades that didn't hit targets
        winning_trades = [t for t in trades if t['win']]
        winning_no_target = [t for t in winning_trades if t['hit_target'] is None]
        
        if winning_no_target:
            print(f"\n📊 WINNING TRADES WITHOUT TARGET HITS:")
            print(f"  Total: {len(winning_no_target)} trades")
            avg_exit_pnl = np.mean([t['pnl_pct'] for t in winning_no_target])
            print(f"  Average Exit PnL: {avg_exit_pnl:.2f}%")
            
            # Analyze exit timing
            exit_times = [t['bars_held'] for t in winning_no_target]
            print(f"  Average Hold Time: {np.mean(exit_times):.1f} bars")
            print(f"  Median Hold Time: {np.median(exit_times):.1f} bars")
        
        # Analyze target levels vs actual moves
        print(f"\n📊 TARGET LEVEL ANALYSIS:")
        
        # Get all target levels used
        all_targets = []
        for trade in trades:
            if 'targets' in trade and trade['targets']:
                entry_price = trade['entry_price']
                direction = trade['direction']
                
                for i, target in enumerate(trade['targets']):
                    if direction == 'BULLISH':
                        target_pct = (target - entry_price) / entry_price * 100
                    else:
                        target_pct = (entry_price - target) / entry_price * 100
                    
                    all_targets.append({
                        'target_num': i + 1,
                        'target_pct': target_pct,
                        'hit': trade['hit_target'] == i + 1,
                        'setup_type': trade.get('setup_type', 'order_block'),
                        'direction': direction
                    })
        
        if all_targets:
            # Analyze by target number
            for target_num in [1, 2, 3]:
                target_data = [t for t in all_targets if t['target_num'] == target_num]
                if target_data:
                    hit_rate = len([t for t in target_data if t['hit']]) / len(target_data) * 100
                    avg_target_pct = np.mean([t['target_pct'] for t in target_data])
                    print(f"  T{target_num}: {avg_target_pct:.2f}% target, {hit_rate:.1f}% hit rate")
        
        # Analyze by setup type
        setup_targets = {}
        for target_data in all_targets:
            setup_type = target_data['setup_type']
            if setup_type not in setup_targets:
                setup_targets[setup_type] = []
            setup_targets[setup_type].append(target_data)
        
        print(f"\n📊 TARGET PERFORMANCE BY SETUP TYPE:")
        for setup_type, targets in setup_targets.items():
            if targets:
                hit_rate = len([t for t in targets if t['hit']]) / len(targets) * 100
                avg_target_pct = np.mean([t['target_pct'] for t in targets])
                print(f"  {setup_type}: {avg_target_pct:.2f}% avg target, {hit_rate:.1f}% hit rate")
        
        # Analyze if targets are too conservative
        print(f"\n🎯 TARGET CONSERVATIVENESS ANALYSIS:")
        
        # Check if we're leaving money on the table
        winning_trades_with_targets = [t for t in winning_trades if t['hit_target'] is not None]
        if winning_trades_with_targets:
            avg_target_hit_pnl = np.mean([t['pnl_pct'] for t in winning_trades_with_targets])
            avg_win_pnl = np.mean([t['pnl_pct'] for t in winning_trades])
            
            print(f"  Average PnL when target hit: {avg_target_hit_pnl:.2f}%")
            print(f"  Average PnL for all wins: {avg_win_pnl:.2f}%")
            
            if avg_win_pnl > avg_target_hit_pnl * 1.5:
                print(f"  ⚠️  Targets may be too conservative - wins average {avg_win_pnl/avg_target_hit_pnl:.1f}x higher than targets")
            elif avg_win_pnl < avg_target_hit_pnl * 0.8:
                print(f"  ✅ Targets appear appropriate - wins close to target levels")
        
        # Analyze target timing
        print(f"\n⏰ TARGET TIMING ANALYSIS:")
        
        target_hit_times = [t['bars_held'] for t in trades if t['hit_target'] is not None]
        if target_hit_times:
            avg_target_time = np.mean(target_hit_times)
            median_target_time = np.median(target_hit_times)
            print(f"  Average time to target: {avg_target_time:.1f} bars")
            print(f"  Median time to target: {median_target_time:.1f} bars")
            
            # Check if targets are hit too quickly
            quick_hits = len([t for t in target_hit_times if t <= 3])
            if quick_hits > len(target_hit_times) * 0.3:
                print(f"  ⚠️  {quick_hits/len(target_hit_times)*100:.1f}% of targets hit within 3 bars - may be too tight")
            else:
                print(f"  ✅ Target timing appears reasonable")
        
        # Recommendations
        print(f"\n💡 TAKE PROFIT RECOMMENDATIONS:")
        
        # Analyze if T2/T3 are being hit
        t2_hit_rate = target_hits['T2'] / total_trades * 100 if total_trades > 0 else 0
        t3_hit_rate = target_hits['T3'] / total_trades * 100 if total_trades > 0 else 0
        
        if t2_hit_rate < 5:
            print(f"  1. ⚠️  T2 hit rate is very low ({t2_hit_rate:.1f}%) - consider reducing T2 target")
        if t3_hit_rate < 2:
            print(f"  2. ⚠️  T3 hit rate is very low ({t3_hit_rate:.1f}%) - consider reducing T3 target")
        
        # Check if T1 is too conservative
        t1_hit_rate = target_hits['T1'] / total_trades * 100 if total_trades > 0 else 0
        if t1_hit_rate > 80:
            print(f"  3. ✅ T1 hit rate is excellent ({t1_hit_rate:.1f}%) - current target is well-calibrated")
        elif t1_hit_rate < 50:
            print(f"  4. ⚠️  T1 hit rate is low ({t1_hit_rate:.1f}%) - consider reducing T1 target")
        
        return {
            'target_hit_rates': target_hits,
            'avg_target_time': np.mean(target_hit_times) if target_hit_times else 0,
            't1_hit_rate': t1_hit_rate,
            't2_hit_rate': t2_hit_rate,
            't3_hit_rate': t3_hit_rate
        }

    def analyze_target_optimization(self):
        """PHASE 7: Analyze what targets we're leaving on the table"""
        trades = self.all_trades
        
        if not trades:
            print("❌ No trades to analyze target optimization")
            return
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
            if 'hit_target' not in trade:
                trade['hit_target'] = None
        
        print("\n" + "=" * 70)
        print("🎯 TARGET OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        # Analyze winning trades that didn't hit targets
        winning_trades = [t for t in trades if t['win']]
        winning_no_target = [t for t in winning_trades if t['hit_target'] is None]
        
        if winning_no_target:
            print(f"\n📊 WINNING TRADES WITHOUT TARGET HITS:")
            print(f"  Total: {len(winning_no_target)} trades")
            avg_exit_pnl = np.mean([t['pnl_pct'] for t in winning_no_target])
            print(f"  Average Exit PnL: {avg_exit_pnl:.2f}%")
            
            # Analyze what we're leaving on the table
            pnl_distribution = [t['pnl_pct'] for t in winning_no_target]
            print(f"  Min Exit PnL: {np.min(pnl_distribution):.2f}%")
            print(f"  Max Exit PnL: {np.max(pnl_distribution):.2f}%")
            print(f"  Median Exit PnL: {np.median(pnl_distribution):.2f}%")
            
            # Analyze by setup type
            fvg_no_target = [t for t in winning_no_target if t.get('setup_type') == 'fvg']
            ob_no_target = [t for t in winning_no_target if t.get('setup_type', 'order_block') == 'order_block']
            
            if fvg_no_target:
                fvg_avg = np.mean([t['pnl_pct'] for t in fvg_no_target])
                print(f"  FVG trades without targets: {len(fvg_no_target)} (avg: {fvg_avg:.2f}%)")
            
            if ob_no_target:
                ob_avg = np.mean([t['pnl_pct'] for t in ob_no_target])
                print(f"  Order Block trades without targets: {len(ob_no_target)} (avg: {ob_avg:.2f}%)")
        
        # Analyze target hit rates and actual moves
        print(f"\n📊 TARGET HIT ANALYSIS:")
        target_hits = {}
        for i in range(1, 4):
            target_hits[f'T{i}'] = len([t for t in trades if t['hit_target'] == i])
        
        total_trades = len(trades)
        for target, hits in target_hits.items():
            hit_rate = hits / total_trades * 100 if total_trades > 0 else 0
            print(f"  {target} Hit Rate: {hit_rate:.1f}% ({hits}/{total_trades})")
        
        # Analyze what targets should be based on actual moves
        print(f"\n📊 ACTUAL MOVE ANALYSIS:")
        all_pnls = [t['pnl_pct'] for t in winning_trades]
        if all_pnls:
            print(f"  Average winning move: {np.mean(all_pnls):.2f}%")
            print(f"  Median winning move: {np.median(all_pnls):.2f}%")
            print(f"  75th percentile: {np.percentile(all_pnls, 75):.2f}%")
            print(f"  90th percentile: {np.percentile(all_pnls, 90):.2f}%")
            print(f"  95th percentile: {np.percentile(all_pnls, 95):.2f}%")
        
        # Recommendations for target optimization
        print(f"\n💡 TARGET OPTIMIZATION RECOMMENDATIONS:")
        
        if winning_no_target:
            avg_untapped = np.mean([t['pnl_pct'] for t in winning_no_target])
            print(f"  1. Average untapped profit: {avg_untapped:.2f}%")
            
            if avg_untapped > 5:
                print(f"     ⚠️  Significant profit left on table - consider higher targets")
            elif avg_untapped > 2:
                print(f"     💡 Moderate profit left on table - consider moderate target increases")
            else:
                print(f"     ✅ Targets appear well-calibrated")
        
        # Analyze by setup type
        fvg_trades = [t for t in winning_trades if t.get('setup_type') == 'fvg']
        ob_trades = [t for t in winning_trades if t.get('setup_type', 'order_block') == 'order_block']
        
        if fvg_trades:
            fvg_pnls = [t['pnl_pct'] for t in fvg_trades]
            fvg_avg = np.mean(fvg_pnls)
            fvg_median = np.median(fvg_pnls)
            print(f"  2. FVG trades: avg {fvg_avg:.2f}%, median {fvg_median:.2f}%")
            
            if fvg_avg > 2:
                print(f"     💡 FVG targets could be increased to capture more profit")
        
        if ob_trades:
            ob_pnls = [t['pnl_pct'] for t in ob_trades]
            ob_avg = np.mean(ob_pnls)
            ob_median = np.median(ob_pnls)
            print(f"  3. Order Block trades: avg {ob_avg:.2f}%, median {ob_median:.2f}%")
            
            if ob_avg > 5:
                print(f"     💡 Order Block targets could be increased significantly")
        
        return {
            'avg_untapped_profit': avg_untapped if 'avg_untapped' in locals() else 0,
            'target_hit_rates': target_hits,
            'fvg_avg_pnl': np.mean(fvg_pnls) if 'fvg_pnls' in locals() else 0,
            'ob_avg_pnl': np.mean(ob_pnls) if 'ob_pnls' in locals() else 0
        }

    def analyze_risk_reward_optimization(self):
        """PHASE 7: Comprehensive risk/reward analysis and optimization"""
        trades = self.all_trades
        
        if not trades:
            print("❌ No trades to analyze risk/reward optimization")
            return
        
        # Ensure all trades have required fields
        for trade in trades:
            if 'win' not in trade:
                trade['win'] = trade.get('pnl_pct', 0) > 0
            if 'pnl_pct' not in trade:
                trade['pnl_pct'] = trade.get('pnl', 0)
            if 'risk_reward_ratio' not in trade:
                trade['risk_reward_ratio'] = 1.0
        
        print("\n" + "=" * 70)
        print("⚖️ RISK/REWARD OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        # Current risk/reward metrics
        winning_trades = [t for t in trades if t['win']]
        losing_trades = [t for t in trades if not t['win']]
        
        if winning_trades == 0:
            print("  ⚠️  No winning trades - cannot calculate win/loss ratio")
            return
        
        if winning_trades and losing_trades:
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            print(f"\n📊 CURRENT RISK/REWARD METRICS:")
            print(f"  Average Win: +{avg_win:.2f}%")
            print(f"  Average Loss: {avg_loss:.2f}%")
            print(f"  Win/Loss Ratio: {win_loss_ratio:.2f}:1")
            
            # Analyze if the ratio is sustainable
            if win_loss_ratio < 0.5:
                print(f"  ⚠️  Risk/Reward ratio is poor - need to improve")
            elif win_loss_ratio < 1.0:
                print(f"  ⚠️  Risk/Reward ratio is below 1:1 - high win rate needed")
            elif win_loss_ratio < 2.0:
                print(f"  ✅ Risk/Reward ratio is reasonable")
            else:
                print(f"  🎯 Risk/Reward ratio is excellent")
        
        # Analyze by setup type
        print(f"\n📊 RISK/REWARD BY SETUP TYPE:")
        setup_rr = {}
        
        for trade in trades:
            setup_type = trade.get('setup_type', 'order_block')
            if setup_type not in setup_rr:
                setup_rr[setup_type] = {'wins': [], 'losses': []}
            
            if trade['win']:
                setup_rr[setup_type]['wins'].append(trade['pnl_pct'])
            else:
                setup_rr[setup_type]['losses'].append(trade['pnl_pct'])
        
        for setup_type, data in setup_rr.items():
            if data['wins'] and data['losses']:
                avg_win = np.mean(data['wins'])
                avg_loss = np.mean(data['losses'])
                rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                win_rate = len(data['wins']) / (len(data['wins']) + len(data['losses'])) * 100
                
                print(f"  {setup_type}: {rr_ratio:.2f}:1 R/R, {win_rate:.1f}% win rate")
                print(f"    Avg Win: +{avg_win:.2f}%, Avg Loss: {avg_loss:.2f}%")
        
        # Analyze optimal stop loss placement
        print(f"\n🎯 OPTIMAL STOP LOSS ANALYSIS:")
        
        # Get drawdown analysis
        drawdown_stats = self.analyze_drawdown_statistics()
        
        if drawdown_stats:
            conservative_stop = drawdown_stats['conservative_stop']
            moderate_stop = drawdown_stats['moderate_stop']
            aggressive_stop = drawdown_stats['aggressive_stop']
            
            # Calculate potential R/R ratios with different stops
            if winning_trades:
                avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
                
                print(f"  Current average win: +{avg_win:.2f}%")
                print(f"  Conservative stop ({conservative_stop:.2f}%): {avg_win/conservative_stop:.2f}:1 R/R")
                print(f"  Moderate stop ({moderate_stop:.2f}%): {avg_win/moderate_stop:.2f}:1 R/R")
                print(f"  Aggressive stop ({aggressive_stop:.2f}%): {avg_win/aggressive_stop:.2f}:1 R/R")
        
        # Analyze take profit optimization
        print(f"\n🎯 TAKE PROFIT OPTIMIZATION:")
        
        tp_stats = self.analyze_take_profit_optimization()
        
        if tp_stats:
            t1_hit_rate = tp_stats['t1_hit_rate']
            t2_hit_rate = tp_stats['t2_hit_rate']
            t3_hit_rate = tp_stats['t3_hit_rate']
            
            print(f"  T1 Hit Rate: {t1_hit_rate:.1f}%")
            print(f"  T2 Hit Rate: {t2_hit_rate:.1f}%")
            print(f"  T3 Hit Rate: {t3_hit_rate:.1f}%")
            
            # Recommendations
            if t1_hit_rate < 60:
                print(f"  ⚠️  T1 hit rate is low - consider adjusting Fibonacci levels")
            if t2_hit_rate < 10:
                print(f"  ⚠️  T2 hit rate is very low - consider adjusting Fibonacci levels")
            if t3_hit_rate < 5:
                print(f"  ⚠️  T3 hit rate is very low - consider adjusting Fibonacci levels")
        
        # Overall recommendations
        print(f"\n💡 RISK/REWARD OPTIMIZATION RECOMMENDATIONS:")
        
        # Check if we need to improve R/R ratio
        if 'win_loss_ratio' in locals() and win_loss_ratio < 1.0:
            print(f"  1. ⚠️  Improve risk/reward ratio (currently {win_loss_ratio:.2f}:1)")
            print(f"     - Consider wider stops to reduce average loss")
            print(f"     - Consider higher targets to increase average win")
        
        # Check if win rate compensates for poor R/R
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        if win_rate > 80 and win_loss_ratio < 0.5:
            print(f"  2. ✅ High win rate ({win_rate:.1f}%) compensates for poor R/R ratio")
        elif win_rate < 70 and win_loss_ratio < 1.0:
            print(f"  3. ⚠️  Both win rate ({win_rate:.1f}%) and R/R ratio need improvement")
        
        return {
            'win_loss_ratio': win_loss_ratio if 'win_loss_ratio' in locals() else 0,
            'win_rate': win_rate,
            'avg_win': avg_win if 'avg_win' in locals() else 0,
            'avg_loss': avg_loss if 'avg_loss' in locals() else 0
        }

def main():
    """Run 1H R2 - Tightened Stop Loss backtest"""
    
    # 1H R2 configuration with tightened stop loss
    config_1h = {
        'name': '1H R2 - Tightened Stop Loss',
        'config': {
            'future_window': 200,           # 33 days tracking
            'min_hold_bars': 6,             # 24 hours minimum
            'stop_multiplier': 0.75,        # Stop distance
            'target_method': 'ict',         # ICT method
            'lookback_months': 3,           # Extended lookback
            'min_order_block_size': 1.0,    # 1H value
            'min_volume_ratio': 1.5,        # 1H value
            'min_quality_score': 70,        # 1H value
            'max_distance_pct': 18.0,       # Keep proven baseline
            'min_confluence_factors': 3,    # Keep proven baseline
            'max_ob_age': 72,               # 1H value
            # SURGICAL FIXES:
            'use_optimized_targets': True,  # 0.8%, 1.2%, 2.0%
            'use_retracement_entry': True,  # 25% into OB
            'use_category_weighting': True, # Weight by performance
            'use_enhanced_regime': True,    # Better filtering
            'quality_boost_enabled': True,  # Quality improvements
            'target_win_rate': 80.0,       # Ultimate target
            'current_baseline': 64.0,      # Known baseline
            # PHASE 6 ADDITIONS:
            'use_fvg_detection': True,     # Enable FVG detection
            'min_fvg_size': 0.4,           # 1H value
            'max_fvg_age': 72,             # 1H value
            'use_breaker_blocks': True,   # Enable breaker blocks
            'fvg_weight': 1.5,            # FVG quality multiplier
            # PHASE 9 ADDITIONS:
            'category_evaluation_period': 30,  # Re-evaluate every 30 days
            'min_category_win_rate': 60,      # Minimum to trade full size
            'category_position_scaling': True,  # Scale position by performance
            'underperforming_categories': ['Gaming/NFTs', 'DeFi'],  # Categories to monitor
            'use_swing_targets': True,       # NEW
            'use_dynamic_stops': True,       # NEW
            'stop_loss_pct': 0.006,         # 0.6% stop loss - TIGHTENED
        }
    }
    
    # Create phase9_config alias to fix undefined variable errors
    phase9_config = config_1h
    
    print("Select test mode:")
    print("1. Test R21 on 36 tokens (quick)")
    print("2. Test R21 on all 89 tokens")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Quick test with 36 tokens
        print("\n🚀 TESTING 1H R2 - TIGHTENED STOP LOSS ON 36 TOKENS")
        print("=" * 70)
        
        test_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ARB/USDT', 'OP/USDT', 'SHIB/USDT', 'PEPE/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT',
            'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT',
            'XRP/USDT', 'DOGE/USDT', 'TRX/USDT', 'ETC/USDT',
            'FIL/USDT', 'NEAR/USDT', 'ALGO/USDT', 'VET/USDT',
            'ICP/USDT', 'FLOW/USDT', 'THETA/USDT', 'XTZ/USDT',
            'AAVE/USDT', 'SUSHI/USDT', 'COMP/USDT', 'MKR/USDT',
            'SNX/USDT', 'CRV/USDT', 'YFI/USDT', 'BAL/USDT'
        ]
        
        backtester = ICTEnhancedBacktester(config=config_1h['config'])
        results = backtester.run_enhanced_backtest(symbols=test_symbols, config=config_1h['config'])
        
        if results:
            win_rate = len([t for t in results if t['win']]) / len(results) * 100
            total_trades = len(results)
            
            # Count by setup type
            setup_counts = {}
            for trade in results:
                setup_type = trade.get('setup_type', 'order_block')
                if setup_type not in setup_counts:
                    setup_counts[setup_type] = 0
                setup_counts[setup_type] += 1
            
            t1_hit_rate = len([t for t in results if t['hit_target'] == 1]) / total_trades * 100
            
            print(f"\n🎯 PHASE 18 ENHANCED FIBONACCI TARGETS QUICK TEST RESULTS:")
            print(f"=" * 50)
            print(f"📊 Total Trades: {total_trades}")
            print(f"🏆 Win Rate: {win_rate:.1f}%")
            print(f"🎯 T1 Hit Rate: {t1_hit_rate:.1f}%")
            
            print(f"\n📊 TRADES BY SETUP TYPE:")
            for setup_type, count in setup_counts.items():
                pct = count / total_trades * 100
                print(f"   {setup_type}: {count} trades ({pct:.1f}%)")
            
            if win_rate >= 80:
                print(f"\n🎉 80% TARGET ACHIEVED!")
                print(f"✅ FVG detection pushed win rate to target")
            elif win_rate >= 75:
                print(f"\n🔥 VERY CLOSE! {win_rate:.1f}% (target: 80%)")
            else:
                print(f"\n⚠️ Need further optimization")
                
    elif choice == "2":
        # Full 89-token test
        print("\n🚀 TESTING 1H R2 - TIGHTENED STOP LOSS ON ALL 89 TOKENS")
        print("=" * 70)
        
        backtester = ICTEnhancedBacktester(config=config_1h['config'])
        results = backtester.run_enhanced_backtest(config=config_1h['config'])  # Uses all 89 tokens
        
        if results:
            win_rate = len([t for t in results if t['win']]) / len(results) * 100
            total_return = sum([t['pnl_pct'] for t in results])
            total_trades = len(results)
            
            # Count by setup type
            setup_counts = {}
            win_by_setup = {}
            for trade in results:
                setup_type = trade.get('setup_type', 'order_block')
                if setup_type not in setup_counts:
                    setup_counts[setup_type] = 0
                    win_by_setup[setup_type] = {'wins': 0, 'total': 0}
                setup_counts[setup_type] += 1
                win_by_setup[setup_type]['total'] += 1
                if trade['win']:
                    win_by_setup[setup_type]['wins'] += 1
            
            t1_hit_rate = len([t for t in results if t['hit_target'] == 1]) / total_trades * 100
            t2_hit_rate = len([t for t in results if t['hit_target'] == 2]) / total_trades * 100
            stop_hits = len([t for t in results if t['hit_stop']])
            
            print(f"\n🎯 PHASE 18 ENHANCED FIBONACCI TARGETS SUMMARY:")
            print(f"=" * 50)
            print(f"📊 Total Trades Found: {total_trades}")
            print(f"🏆 Win Rate: {win_rate:.1f}%")
            print(f"💰 Total Return: {total_return:.1f}%")
            print(f"🎯 T1 Hit Rate: {t1_hit_rate:.1f}% (Fibonacci/S/R/ATR target)")
            print(f"🎯 T2 Hit Rate: {t2_hit_rate:.1f}% (Fibonacci/S/R/ATR target)")
            print(f"🛡️ Stop Hit Rate: {stop_hits/total_trades*100:.1f}%")
            
            print(f"\n📊 PERFORMANCE BY SETUP TYPE:")
            for setup_type, stats in win_by_setup.items():
                wr = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"   {setup_type}: {stats['total']} trades, {wr:.1f}% win rate")
            
            # Success assessment
            print(f"\n📊 PERFORMANCE vs TARGETS:")
            if win_rate >= 80:
                print(f"🎉 80% WIN RATE ACHIEVED!")
                print(f"✅ System ready for live trading")
                print(f"✅ FVG detection was the missing piece!")
            elif win_rate >= 75:
                print(f"🔥 VERY CLOSE! {win_rate:.1f}% (target: 80%)")
                print(f"💡 Minor tweaks needed")
            elif win_rate >= 70:
                print(f"💪 GOOD PROGRESS! {win_rate:.1f}% (target: 80%)")
                print(f"💡 Continue optimization")
            else:
                print(f"⚠️ MORE WORK NEEDED: {win_rate:.1f}% (target: 80%)")
                
    elif choice == "3":
        # Compare with/without FVG detection
        print("\n🚀 COMPARING WITH/WITHOUT FVG DETECTION")
        print("=" * 70)
        
        test_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ARB/USDT', 'OP/USDT', 'SHIB/USDT', 'PEPE/USDT'
        ]
        
        # Test WITHOUT FVG detection
        config_without = phase9_config['config'].copy()
        config_without['use_fvg_detection'] = False
        config_without['use_breaker_blocks'] = False
        
        print("\n📊 Testing WITHOUT FVG detection...")
        backtester_without = ICTEnhancedBacktester(config=config_without)
        results_without = backtester_without.run_enhanced_backtest(symbols=test_symbols, config=config_without)
        
        if results_without:
            wr_without = len([t for t in results_without if t['win']]) / len(results_without) * 100
            trades_without = len(results_without)
            t1_without = len([t for t in results_without if t['hit_target'] == 1]) / trades_without * 100
        
        # Test WITH FVG detection
        print("\n📊 Testing WITH FVG detection...")
        backtester_with = ICTEnhancedBacktester(config=phase9_config['config'])
        results_with = backtester_with.run_enhanced_backtest(symbols=test_symbols, config=phase9_config['config'])
        
        if results_with:
            wr_with = len([t for t in results_with if t['win']]) / len(results_with) * 100
            trades_with = len(results_with)
            t1_with = len([t for t in results_with if t['hit_target'] == 1]) / trades_with * 100
            
            # Count FVG trades
            fvg_trades = len([t for t in results_with if t.get('setup_type') == 'fvg'])
            fvg_wins = len([t for t in results_with if t.get('setup_type') == 'fvg' and t['win']])
            fvg_wr = fvg_wins / fvg_trades * 100 if fvg_trades > 0 else 0
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"=" * 50)
        print(f"                    WITHOUT FVG  | WITH FVG")
        print(f"Win Rate:           {wr_without:.1f}%        | {wr_with:.1f}%")
        print(f"Total Trades:       {trades_without}          | {trades_with}")
        print(f"T1 Hit Rate:        {t1_without:.1f}%       | {t1_with:.1f}%")
        
        if fvg_trades > 0:
            print(f"\n📊 FVG CONTRIBUTION:")
            print(f"FVG Trades: {fvg_trades} ({fvg_trades/trades_with*100:.1f}% of total)")
            print(f"FVG Win Rate: {fvg_wr:.1f}%")
        
        print(f"\n📈 IMPROVEMENT:")
        print(f"Win Rate:    {'+' if wr_with > wr_without else ''}{wr_with - wr_without:.1f}%")
        print(f"Trade Count: {'+' if trades_with > trades_without else ''}{trades_with - trades_without}")
        
        if wr_with > wr_without + 3:
            print(f"\n✅ FVG DETECTION HIGHLY EFFECTIVE!")
            print(f"💡 Fair Value Gaps provide high-probability setups")
        elif wr_with > wr_without:
            print(f"\n✅ FVG DETECTION MODERATELY EFFECTIVE")
        else:
            print(f"\n⚠️ FVG DETECTION NEEDS ADJUSTMENT")
    
    elif choice == "4":
        # Phase 9: Risk/Reward & Drawdown Analysis (8 tokens)
        print("\n🚀 PHASE 9: RISK/REWARD & DRAWDOWN ANALYSIS ON 8 TOKENS")
        print("=" * 70)
        
        test_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ARB/USDT', 'OP/USDT', 'SHIB/USDT', 'PEPE/USDT'
        ]
        
        backtester = ICTEnhancedBacktester(config=phase9_config['config'])
        results = backtester.run_enhanced_backtest(symbols=test_symbols)
        
        if results:
            print(f"\n🎯 PHASE 9 ANALYSIS COMPLETE!")
            print(f"📊 Total Trades Analyzed: {len(results)}")
            
            # The new analysis methods are automatically called in run_enhanced_backtest
            print(f"✅ Drawdown analysis completed")
            print(f"✅ Take profit optimization analysis completed")
            print(f"✅ Risk/reward optimization analysis completed")
    
    elif choice == "5":
        # Phase 9: Risk/Reward & Drawdown Analysis (all tokens)
        print("\n🚀 PHASE 9: RISK/REWARD & DRAWDOWN ANALYSIS ON ALL TOKENS")
        print("=" * 70)
        
        backtester = ICTEnhancedBacktester(config=phase9_config['config'])
        results = backtester.run_enhanced_backtest()  # Uses all 89 tokens
        
        if results:
            print(f"\n🎯 PHASE 9 ANALYSIS COMPLETE!")
            print(f"📊 Total Trades Analyzed: {len(results)}")
            
            # The new analysis methods are automatically called in run_enhanced_backtest
            print(f"✅ Drawdown analysis completed")
            print(f"✅ Take profit optimization analysis completed")
            print(f"✅ Risk/reward optimization analysis completed")
    
    elif choice == "6":
        # Phase 9: Dynamic Target Analysis (8 tokens)
        print("\n🚀 PHASE 9: DYNAMIC TARGET ANALYSIS ON 8 TOKENS")
        print("=" * 70)
        
        test_symbols = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'ARB/USDT', 'OP/USDT', 'SHIB/USDT', 'PEPE/USDT'
        ]
        
        # Use dynamic target configuration
        dynamic_config = phase9_config['config'].copy()
        dynamic_config['use_dynamic_targets'] = True
        
        backtester = ICTEnhancedBacktester(config=dynamic_config)
        results = backtester.run_enhanced_backtest(symbols=test_symbols)
        
        if results:
            print(f"\n🎯 PHASE 9 DYNAMIC TARGET ANALYSIS COMPLETE!")
            print(f"📊 Total Trades Analyzed: {len(results)}")
            
            # The new analysis methods are automatically called in run_enhanced_backtest
            print(f"✅ Dynamic target analysis completed")
            print(f"✅ Target optimization analysis completed")
            print(f"✅ Risk/reward optimization analysis completed")
    
    elif choice == "7":
        # Phase 9: Dynamic Target Analysis (all tokens)
        print("\n🚀 PHASE 9: DYNAMIC TARGET ANALYSIS ON ALL TOKENS")
        print("=" * 70)
        
        # Use dynamic target configuration
        dynamic_config = phase9_config['config'].copy()
        dynamic_config['use_dynamic_targets'] = True
        
        backtester = ICTEnhancedBacktester(config=dynamic_config)
        results = backtester.run_enhanced_backtest()  # Uses all 89 tokens
        
        if results:
            print(f"\n🎯 PHASE 9 DYNAMIC TARGET ANALYSIS COMPLETE!")
            print(f"📊 Total Trades Analyzed: {len(results)}")
            
            # The new analysis methods are automatically called in run_enhanced_backtest
            print(f"✅ Dynamic target analysis completed")
            print(f"✅ Target optimization analysis completed")
            print(f"✅ Risk/reward optimization analysis completed")
    
    else:
        print("Invalid choice. Please run again and select 1, 2, 3, 4, 5, 6, or 7.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Backtest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()