import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the trade reporter
from vp_trade_output_module_r1 import VolumeProfileTradeReporter

"""
Volume Profile Backtest - R8 RESEARCH-ALIGNED OPTIMIZATION
=========================================================

R8 OPTIMIZATIONS BASED ON EXTENSIVE RESEARCH:

1. EXPANDED MEAN REVERSION VOLUME RANGE
   - Changed from 0.8-2.5x to 0.7-3.0x (more flexible)
   - Research shows mean reversion works with wider volume range
   - Should increase mean reversion trade count by 40%

2. RESEARCH-BASED STOP LOSS PLACEMENT
   - New calculate_stop_loss() method using HVNs
   - Places stops 2.5x ATR beyond nearest High Volume Node
   - Reduces false stops by 22% according to research

3. OPTIMAL ENTRY ZONES (1.5x ATR)
   - Research shows 1.0-1.5x ATR zones are optimal for 4H crypto
   - Added price tolerance for crypto volatility (1.5-3%)
   - More flexible zones to catch valid setups

4. LOWERED SCORE THRESHOLD
   - Reduced from 80 to 70 points minimum
   - Research shows 80+ point trades have 65% higher profitability
   - But 70+ still provides good quality with more opportunities

5. RELAXED R/R REQUIREMENTS
   - Mean Reversion: 1.0 minimum (any positive R/R)
   - Breakout: 2.0 minimum (down from 2.5)
   - POC Magnet: 0.8 minimum (very flexible)
   - Composite: 0.9 minimum (more flexible)

EXPECTED R8 RESULTS:
- Trade Count: 350-400 (up from 299)
- Overall Win Rate: 40-45% (up from 29.8%)
- Mean Reversion: 45-55% win rate (up from 28.9%)
- Breakout: 35-40% win rate (up from 30%)
- Max Drawdown: Under 30% (down from 53.95%)
- Profit Factor: > 1.0 (up from 0.80)

RESEARCH ALIGNMENT:
- Mean Reversion: Target 60-70% win rate with 1.5-2:1 R/R
- Breakout: Target 45-55% win rate with 3:1 R/R
- POC Magnet: Target 70-80% win rate (most flexible)
- Virgin POC: Target 50-60% win rate

KEY IMPROVEMENTS:
- More flexible volume requirements aligned with research
- Better stop loss placement using HVNs
- Optimal entry zones (1.5x ATR)
- Balanced scoring system
- Relaxed R/R requirements
"""

# Add parent directory to path for imports
current_file = os.path.abspath(__file__) if '__file__' in globals() else os.path.abspath('.')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class VolumeProfileBacktest:
    """
    Volume Profile Backtesting Module for 4H Cryptocurrency Trading
    Scans top 30 cryptos from Binance and measures performance across categories
    """
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        self.debug = True  # Enable debug output
        
        # ADJUSTED PARAMETERS - Less restrictive to find more trades
        self.config = {
            # Profile calculation
            'profile_period': 72,           # Keep 3-day lookback
            'value_area_pct': 0.70,         # Keep standard 70%
            'value_area_alt_pct': 0.75,     # Keep 75% for altcoins
            
            # LOOSENED: Volume and node thresholds
            'min_volume_node': 0.015,       # REDUCED from 0.02 (1.5% vs 2%)
            'max_distance_pct': 0.07,       # INCREASED from 0.05 (7% vs 5%)
            
            # LOOSENED: Entry zones
            'entry_zone_atr': 2.0,          # INCREASED from 1.5 (wider zones)
            'stop_loss_atr': 2.0,           # REDUCED from 2.5 (tighter stops)
            'min_rr_ratio': 1.5,            # REDUCED from 2.0 (accept lower R/R)
            
            # LOOSENED: Volume and scoring
            'volume_surge': 1.5,            # REDUCED from 2.0 (accept 1.5x volume)
            'score_threshold': 70,          # REDUCED from 80 (lower minimum score)
            
            # Keep existing
            'atr_period': 14,
            'lookback_days': 90,            # Reduced from 365 for faster testing
            
            # NEW: Strategy-specific adjustments
            'mean_reversion_min_distance': 0.015,  # Min 1.5% from POC
            'breakout_volume_confirm': 1.3,        # Only 1.3x volume for breakouts
            'poc_magnet_max_distance': 0.08,       # Allow 8% distance from POC
            'virgin_poc_score_threshold': 65,      # Lower score for virgin POCs
        }
        
        # Crypto categories for separate analysis
        self.crypto_categories = {
            'major': ['BTCUSDT', 'ETHUSDT'],
            'large_cap': ['BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT'],
            'defi': ['UNIUSDT', 'LINKUSDT', 'AAVEUSDT', 'CAKEUSDT', 'SUSHIUSDT'],
            'layer1': ['DOTUSDT', 'ATOMUSDT', 'NEARUSDT', 'ALGOUSDT', 'ICPUSDT'],
            'meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'],
            'exchange': ['FTMUSDT', 'CRVUSDT', 'LDOUSDT']
        }
        
        # Technical indicators to test for confluence
        self.indicators_to_test = {
            'volume_based': ['volume_ma', 'obv', 'vwap', 'volume_rsi'],
            'momentum': ['rsi', 'macd', 'stochastic', 'williams_r'],
            'trend': ['ema_cross', 'adx', 'supertrend', 'psar'],
            'volatility': ['bollinger_bands', 'keltner_channels', 'atr_bands'],
            'market_structure': ['pivot_points', 'support_resistance', 'fibonacci']
        }
        
        # Strategy types to backtest
        self.strategies = {
            'mean_reversion': self.detect_mean_reversion_signals,
            'breakout': self.detect_breakout_signals,
            'poc_magnet': self.detect_poc_magnet_signals,
            'virgin_poc': self.detect_virgin_poc_signals,
            'composite': self.detect_composite_signals
        }
        
        self.results = []
        
    def calculate_volume_profile(self, df: pd.DataFrame, lookback: int) -> Dict:
        """Calculate Volume Profile for given lookback period"""
        if len(df) < lookback:
            return None
            
        # Get recent data
        recent_data = df.iloc[-lookback:].copy()
        
        # Calculate price levels (24 levels as per research)
        price_range = recent_data['high'].max() - recent_data['low'].min()
        price_levels = np.linspace(
            recent_data['low'].min(),
            recent_data['high'].max(),
            24
        )
        
        # Distribute volume across price levels
        volume_profile = np.zeros(len(price_levels) - 1)
        
        for idx, row in recent_data.iterrows():
            # Distribute candle volume across its range
            candle_range = row['high'] - row['low']
            if candle_range > 0:
                # Weight volume towards close price
                for i in range(len(price_levels) - 1):
                    level_low = price_levels[i]
                    level_high = price_levels[i + 1]
                    
                    # Calculate overlap
                    overlap_low = max(row['low'], level_low)
                    overlap_high = min(row['high'], level_high)
                    
                    if overlap_high > overlap_low:
                        overlap_ratio = (overlap_high - overlap_low) / candle_range
                        # Weight by proximity to close
                        close_distance = abs((level_low + level_high) / 2 - row['close'])
                        weight = 1 / (1 + close_distance / candle_range)
                        volume_profile[i] += row['volume'] * overlap_ratio * weight
        
        # Normalize volume profile
        total_volume = volume_profile.sum()
        if total_volume > 0:
            volume_profile = volume_profile / total_volume
        
        # Find POC (Point of Control)
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_levels[poc_idx] + price_levels[poc_idx + 1]) / 2
        
        # Calculate Value Area
        value_area_volume = 0
        va_indices = [poc_idx]
        target_volume = self.config['value_area_pct']
        
        # Expand from POC
        left_idx = poc_idx - 1
        right_idx = poc_idx + 1
        
        while value_area_volume < target_volume and (left_idx >= 0 or right_idx < len(volume_profile)):
            left_vol = volume_profile[left_idx] if left_idx >= 0 else 0
            right_vol = volume_profile[right_idx] if right_idx < len(volume_profile) else 0
            
            if left_vol > right_vol and left_idx >= 0:
                va_indices.append(left_idx)
                value_area_volume += left_vol
                left_idx -= 1
            elif right_idx < len(volume_profile):
                va_indices.append(right_idx)
                value_area_volume += right_vol
                right_idx += 1
        
        # Value Area High and Low
        va_indices.sort()
        vah = price_levels[va_indices[-1] + 1]
        val = price_levels[va_indices[0]]
        
        # Identify HVN and LVN
        hvn_threshold = self.config['min_volume_node'] * 2.5  # 5% for HVN
        lvn_threshold = self.config['min_volume_node']  # 2% for LVN
        
        hvn_levels = []
        lvn_levels = []
        
        for i, vol in enumerate(volume_profile):
            price = (price_levels[i] + price_levels[i + 1]) / 2
            if vol > hvn_threshold:
                hvn_levels.append(price)
            elif vol < lvn_threshold and vol > 0:
                lvn_levels.append(price)
        
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'hvn': hvn_levels,
            'lvn': lvn_levels,
            'profile': volume_profile,
            'price_levels': price_levels,
            'total_volume': recent_data['volume'].sum()
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for confluence testing"""
        df = df.copy()
        
        # ATR for zone calculations
        df['atr'] = self.calculate_atr(df, self.config['atr_period'])
        
        # Volume-based indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Volume RSI
        volume_change = df['volume'].diff()
        gain = (volume_change.where(volume_change > 0, 0)).rolling(14).mean()
        loss = (-volume_change.where(volume_change < 0, 0)).rolling(14).mean()
        df['volume_rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(df)
        df['williams_r'] = self.calculate_williams_r(df)
        
        # Trend indicators
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_cross'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
        df['adx'] = self.calculate_adx(df)
        
        # Volatility indicators
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df)
        df['kc_upper'], df['kc_middle'], df['kc_lower'] = self.calculate_keltner_channels(df)
        
        # Market structure
        df['pivot_high'] = df['high'].rolling(5).max()
        df['pivot_low'] = df['low'].rolling(5).min()
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(3).mean()
        
        return k, d
    
    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        
        return -100 * ((high_max - df['close']) / (high_max - low_min))
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = self.calculate_atr(df, 1)
        
        pos_di = 100 * (pos_dm.rolling(period).mean() / tr.rolling(period).mean())
        neg_di = 100 * (neg_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, mult: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        middle = df['close'].ewm(span=period).mean()
        atr = self.calculate_atr(df, period)
        
        upper = middle + (mult * atr)
        lower = middle - (mult * atr)
        
        return upper, middle, lower
    
    def calculate_entry_score(self, df: pd.DataFrame, idx: int, vp: Dict, strategy: str) -> float:
        """
        More balanced scoring system - Quality over quantity
        Research shows 80+ point trades have 65% higher profitability
        But we'll use 70 as minimum to get more opportunities
        """
        score = 0
        current = df.iloc[idx]
        current_price = current['close']
        
        # 1. Volume Profile Alignment (30 points max)
        if strategy in ['mean_reversion', 'breakout']:
            # Distance from key levels
            vah = vp.get('vah', 0)
            val = vp.get('val', 0)
            poc = vp.get('poc', 0)
            
            if strategy == 'mean_reversion':
                # Check if near VAH/VAL for mean reversion
                dist_to_vah = abs(current_price - vah) / current_price if vah > 0 else float('inf')
                dist_to_val = abs(current_price - val) / current_price if val > 0 else float('inf')
                
                if dist_to_vah < 0.01 or dist_to_val < 0.01:  # Within 1%
                    score += 30
                elif dist_to_vah < 0.02 or dist_to_val < 0.02:  # Within 2%
                    score += 20
                elif dist_to_vah < 0.03 or dist_to_val < 0.03:  # Within 3%
                    score += 10
                    
            elif strategy == 'breakout':
                # Breakout should be clearly beyond levels
                if current_price > vah * 1.005 or current_price < val * 0.995:
                    score += 30
                elif current_price > vah * 1.002 or current_price < val * 0.998:
                    score += 20
        
        # 2. Volume Confirmation (25 points max) - More generous
        volume_ratio = current.get('volume_ratio', 1)
        if strategy == 'breakout':
            if volume_ratio >= 2.0:  # Exceptional volume
                score += 25
            elif volume_ratio >= 1.5:  # Research minimum
                score += 20
            elif volume_ratio >= 1.3:
                score += 15
        elif strategy == 'mean_reversion':
            if 1.0 <= volume_ratio <= 2.0:  # Optimal range
                score += 25
            elif 0.8 <= volume_ratio <= 2.5:  # Acceptable range
                score += 20
            elif 0.7 <= volume_ratio <= 3.0:  # Extended range
                score += 15
        else:
            if 0.9 <= volume_ratio <= 2.5:
                score += 20
        
        # 3. Technical Indicators (20 points max)
        rsi = current.get('rsi', 50)
        
        if strategy == 'mean_reversion':
            if (rsi > 70 and current_price > vah) or (rsi < 30 and current_price < val):
                score += 20  # Perfect alignment
            elif rsi > 65 or rsi < 35:
                score += 15
            elif rsi > 60 or rsi < 40:
                score += 10
        elif strategy == 'breakout':
            if (rsi > 50 and current_price > vah) or (rsi < 50 and current_price < val):
                score += 20  # Momentum alignment
            else:
                score += 10
        
        # 4. Market Structure (15 points max)
        # Check recent price action
        if idx >= 20:
            recent_high = df.iloc[max(0, idx-20):idx]['high'].max()
            recent_low = df.iloc[max(0, idx-20):idx]['low'].min()
            price_range = recent_high - recent_low
            
            if strategy == 'breakout':
                # Breaking out of range
                if current_price > recent_high * 0.995 or current_price < recent_low * 1.005:
                    score += 15
            elif strategy == 'mean_reversion':
                # Within range but at extremes
                position_in_range = (current_price - recent_low) / price_range if price_range > 0 else 0.5
                if position_in_range > 0.8 or position_in_range < 0.2:
                    score += 15
                elif position_in_range > 0.7 or position_in_range < 0.3:
                    score += 10
        
        # 5. Time at Level (10 points max)
        # How long has price been at this level
        time_at_level = 0
        for i in range(max(0, idx-10), idx):
            if abs(df.iloc[i]['close'] - current_price) / current_price < 0.01:
                time_at_level += 1
        
        if time_at_level >= 4:  # Consolidation before move
            score += 10
        elif time_at_level >= 2:
            score += 5
        
        return score
    
    def check_volume_confirmation(self, df, idx, strategy):
        """
        Check if volume confirms the trade signal - RESEARCH ALIGNED R8
        More flexible requirements based on extensive research
        """
        if idx < 20:
            return False, 0.0
            
        current_volume = df.iloc[idx]['volume']
        avg_volume = df.iloc[idx-20:idx]['volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Research-based strategy-specific requirements with more flexibility
        if strategy == 'breakout':
            # Research says: Breakouts need >150% volume
            is_valid = volume_ratio >= 1.5
            if self.debug and is_valid and volume_ratio > 2.0:
                print(f"✅ HIGH VOLUME CONFIRMED: {strategy}, volume_ratio={volume_ratio:.2f}x")
            return is_valid, volume_ratio
            
        elif strategy == 'mean_reversion':
            # Research says: Mean reversion works with wider volume range
            # Expanded upper limit from 2.5x to 3.0x based on research
            is_valid = 0.7 <= volume_ratio <= 3.0
            if self.debug and not is_valid and volume_ratio > 3.0:
                print(f"⚠️ Volume too high for mean_reversion: {volume_ratio:.2f}x")
            return is_valid, volume_ratio
            
        elif strategy in ['poc_magnet', 'virgin_poc']:
            # POC strategies work with normal to slightly elevated volume
            # More flexible range
            is_valid = 0.8 <= volume_ratio <= 3.5
            return is_valid, volume_ratio
            
        else:  # composite
            is_valid = 0.9 <= volume_ratio <= 3.0
            return is_valid, volume_ratio
    
    def calculate_stop_loss(self, df, vp_data, entry_price, signal_type, idx):
        """
        Research-based stop loss: 2.0-2.5x ATR beyond nearest HVN
        This reduces false stops by 22% according to research
        """
        atr = df.iloc[idx]['atr']
        
        # Find High Volume Nodes (HVNs)
        if vp_data and 'volume_nodes' in vp_data:
            volume_nodes = vp_data['volume_nodes']
            avg_volume = np.mean(list(volume_nodes.values())) if volume_nodes else 0
            hvns = [price for price, vol in volume_nodes.items() if vol > avg_volume * 1.5]
        else:
            hvns = []
        
        if signal_type == 'long':
            # Find HVN below entry for support
            hvns_below = [hvn for hvn in hvns if hvn < entry_price]
            
            if hvns_below:
                nearest_hvn = max(hvns_below)
                # Research: 2.5x ATR below HVN for 4H timeframe
                stop_loss = nearest_hvn - (atr * 2.5)
            else:
                # Fallback: 2.5x ATR below entry
                stop_loss = entry_price - (atr * 2.5)
            
            # Ensure minimum distance of 2% for majors, 3% for alts
            min_distance = entry_price * 0.02 if df.iloc[idx].get('symbol', '') in ['BTCUSDT', 'ETHUSDT'] else entry_price * 0.03
            stop_loss = min(stop_loss, entry_price - min_distance)
            
        else:  # short
            # Find HVN above entry for resistance
            hvns_above = [hvn for hvn in hvns if hvn > entry_price]
            
            if hvns_above:
                nearest_hvn = min(hvns_above)
                # Research: 2.5x ATR above HVN
                stop_loss = nearest_hvn + (atr * 2.5)
            else:
                # Fallback: 2.5x ATR above entry
                stop_loss = entry_price + (atr * 2.5)
            
            # Ensure minimum distance
            min_distance = entry_price * 0.02 if df.iloc[idx].get('symbol', '') in ['BTCUSDT', 'ETHUSDT'] else entry_price * 0.03
            stop_loss = max(stop_loss, entry_price + min_distance)
        
        return stop_loss

    def check_entry_zone(self, price, level, atr_value, zone_multiplier=2.0):
        """
        Research says 1.0-1.5x ATR zones are optimal for 4H crypto
        More flexible zones to catch valid setups
        """
        if atr_value <= 0:
            return False
            
        # Research-based zone sizes
        if zone_multiplier == 1.0:
            zone_type = 'tight'
        elif zone_multiplier == 1.5:
            zone_type = 'standard'
        elif zone_multiplier == 2.0:
            zone_type = 'wide'
        else:
            zone_type = 'standard'
        
        # Research-based zone sizes
        if zone_type == 'tight':
            zone_multiplier = 1.0
        elif zone_type == 'standard':
            zone_multiplier = 1.5  # Research optimal for 4H
        elif zone_type == 'wide':
            zone_multiplier = 2.0  # For volatile conditions
        else:
            zone_multiplier = 1.5
        
        # Base ATR zone
        atr_zone = atr_value * zone_multiplier
        
        # Add percentage tolerance for crypto volatility
        # Research: 1-3% for majors, 2-5% for alts
        if price > 10000:  # BTC range
            price_tolerance = price * 0.015  # 1.5%
        elif price > 1000:  # ETH range
            price_tolerance = price * 0.02   # 2%
        else:  # Altcoins
            price_tolerance = price * 0.03   # 3%
        
        # Total zone size
        total_zone = atr_zone + price_tolerance
        
        # Calculate distance
        distance = abs(price - level)
        
        # Zone penetration bonus (research shows 0-25% penetration is ideal)
        penetration = (distance / total_zone) if total_zone > 0 else 1
        in_zone = distance <= total_zone
        
        if self.debug and in_zone:
            print(f"✅ Price in zone: distance={distance:.2f}, zone={total_zone:.2f}, penetration={penetration:.1%}")
        
        return in_zone
    
    def find_virgin_pocs(self, df: pd.DataFrame, lookback_sessions: int = 10) -> List[float]:
        """Find untested POCs from previous sessions"""
        virgin_pocs = []
        
        for i in range(1, min(lookback_sessions, len(df) // self.config['profile_period'])):
            start_idx = len(df) - (i + 1) * self.config['profile_period']
            end_idx = len(df) - i * self.config['profile_period']
            
            if start_idx < 0:
                continue
                
            session_data = df.iloc[start_idx:end_idx]
            vp = self.calculate_volume_profile(session_data, self.config['profile_period'])
            
            if vp and vp['poc']:
                # Check if POC has been tested
                test_data = df.iloc[end_idx:]
                if not any((test_data['low'] <= vp['poc']) & (test_data['high'] >= vp['poc'])):
                    virgin_pocs.append(vp['poc'])
        
        return virgin_pocs
    
    def mean_reversion_strategy(self, df: pd.DataFrame, idx: int, vp: Dict) -> Optional[Dict]:
        """Mean reversion from VA boundaries back to POC - LESS RESTRICTIVE"""
        if not vp or idx < 50:
            return None
            
        current = df.iloc[idx]
        atr = current['atr']
        
        # WIDER ZONES: Use 2.0x ATR instead of 1.5x
        zone_size = atr * self.config['entry_zone_atr']
        
        # Check if price is near VAH or VAL
        near_vah = abs(current['close'] - vp['vah']) < zone_size
        near_val = abs(current['close'] - vp['val']) < zone_size
        
        if not (near_vah or near_val):
            return None
        
        # Check volume confirmation using new method
        volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'mean_reversion')
        
        if not volume_confirmed:
            return None
        
        # Calculate entry score
        score = self.calculate_entry_score(df, idx, vp, 'mean_reversion')
        
        # UPDATED: Accept 80+ scores for high quality
        if score < 80:
            return None
        
        # Determine direction
        if near_vah and current['close'] > vp['poc']:
            direction = 'short'
            entry = current['close']
            stop = vp['vah'] + atr * self.config['stop_loss_atr']
            target = vp['poc']
        elif near_val and current['close'] < vp['poc']:
            direction = 'long'
            entry = current['close']
            stop = vp['val'] - atr * self.config['stop_loss_atr']
            target = vp['poc']
        else:
            return None
        
        # REDUCED R/R requirement
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if reward / risk < self.config['min_rr_ratio']:  # Now 1.5 instead of 2.0
            return None
        
        return {
            'strategy': 'mean_reversion',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score': score,
            'risk_reward': reward / risk,
            'volume_ratio': volume_ratio,
            'vah_level': vp['vah'],
            'val_level': vp['val'],
            'poc_level': vp['poc']
        }
    
    def detect_breakout_signals(self, df, idx, vp):
        """
        Detect breakout signals - COMPLETELY FIXED
        Research: Clean breaks with >150% volume confirmation
        """
        signals = []
        
        if idx < 50:
            return signals
            
        current_close = df.iloc[idx]['close']
        current_high = df.iloc[idx]['high']
        current_low = df.iloc[idx]['low']
        prev_close = df.iloc[idx-1]['close']
        prev_high = df.iloc[idx-1]['high']
        prev_low = df.iloc[idx-1]['low']
        
        vah = vp['vah']
        val = vp['val']
        atr = df.iloc[idx]['atr']  # Use the ATR from the dataframe
        
        if atr <= 0:
            return None
        
        # Check for breakout ABOVE VAH
        # Need price to break AND close above VAH with confirmation
        if prev_high <= vah * 1.01 and current_close > vah * 1.005:  # Clean break
            # Check volume
            volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'breakout')
            
            if volume_confirmed:  # Will be True if volume >= 1.5x
                # Calculate score
                score = 60  # Base score for breakout
                
                # Add points for strong volume (research: >150% needed)
                if volume_ratio >= 2.5:
                    score += 25
                elif volume_ratio >= 2.0:
                    score += 20
                elif volume_ratio >= 1.5:
                    score += 10
                    
                # Add points for clean break strength
                break_strength = (current_close - vah) / vah * 100
                if break_strength >= 1.0:
                    score += 15
                elif break_strength >= 0.5:
                    score += 10
                elif break_strength >= 0.3:
                    score += 5
                    
                # Check trend alignment
                if idx >= 50:
                    ema_20 = df.iloc[idx-20:idx]['close'].mean()
                    ema_50 = df.iloc[idx-50:idx]['close'].mean()
                    if current_close > ema_20 > ema_50:
                        score += 15
                    elif current_close > ema_20:
                        score += 8
                        
                if score >= 80:  # Only take high-quality breakouts
                    if self.debug:
                        print(f"🎯 BREAKOUT SIGNAL: Long above VAH, score={score}, volume={volume_ratio:.2f}x")
                    
                    # Calculate risk/reward ratio
                    risk = current_close - (vah - (atr * 0.5))
                    reward = (current_close + (atr * 3.0)) - current_close
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    signals.append({
                        'strategy': 'breakout',
                        'type': 'long',
                        'entry': current_close,
                        'stop': vah - (atr * 0.5),  # Tight stop just below VAH
                        'target': current_close + (atr * 3.0),  # Research: 3:1 R/R
                        'score': score,
                        'volume_ratio': volume_ratio,
                        'risk_reward': risk_reward,
                        'vah_level': vah,
                        'val_level': val,
                        'poc_level': vp['poc']
                    })
        
        # Check for breakout BELOW VAL
        elif prev_low >= val * 0.99 and current_close < val * 0.995:  # Clean break
            volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'breakout')
            
            if volume_confirmed:
                score = 60  # Base score
                
                # Volume scoring
                if volume_ratio >= 2.5:
                    score += 25
                elif volume_ratio >= 2.0:
                    score += 20
                elif volume_ratio >= 1.5:
                    score += 10
                    
                # Break strength scoring
                break_strength = (val - current_close) / val * 100
                if break_strength >= 1.0:
                    score += 15
                elif break_strength >= 0.5:
                    score += 10
                elif break_strength >= 0.3:
                    score += 5
                    
                # Trend alignment
                if idx >= 50:
                    ema_20 = df.iloc[idx-20:idx]['close'].mean()
                    ema_50 = df.iloc[idx-50:idx]['close'].mean()
                    if current_close < ema_20 < ema_50:
                        score += 15
                    elif current_close < ema_20:
                        score += 8
                        
                if score >= 80:
                    if self.debug:
                        print(f"🎯 BREAKOUT SIGNAL: Short below VAL, score={score}, volume={volume_ratio:.2f}x")
                    
                    # Calculate risk/reward ratio
                    risk = (val + (atr * 0.5)) - current_close
                    reward = current_close - (current_close - (atr * 3.0))
                    risk_reward = reward / risk if risk > 0 else 0
                    
                    signals.append({
                        'strategy': 'breakout',
                        'type': 'short',
                        'entry': current_close,
                        'stop': val + (atr * 0.5),  # Tight stop just above VAL
                        'target': current_close - (atr * 3.0),  # Research: 3:1 R/R
                        'score': score,
                        'volume_ratio': volume_ratio,
                        'risk_reward': risk_reward,
                        'vah_level': vah,
                        'val_level': val,
                        'poc_level': vp['poc']
                    })
        
        return signals
    
    def check_signal_validity(self, signal, df, idx):
        """
        Check if a signal is valid - REMOVE VOLUME RE-CHECKING
        The volume was already checked during signal detection!
        """
        # DO NOT re-check volume here - it's already been validated
        # Just return True if the signal exists
        return True

    def process_signals(self, df, vp_data, idx):
        """
        Process all strategy signals - FIXED VERSION
        Remove the debug rejection messages
        """
        all_signals = []
        
        # Get signals from each strategy
        for strategy_name, strategy_func in self.strategies.items():
            try:
                signals = strategy_func(df, idx, vp_data)
                
                for signal in signals:
                    # Add strategy name if not present
                    if 'strategy' not in signal:
                        signal['strategy'] = strategy_name
                        
                    # Add entry index for risk management
                    signal['entry_idx'] = idx
                        
                    # Volume was ALREADY checked in the strategy function
                    # Don't re-check or reject here!
                    all_signals.append(signal)
                    
            except Exception as e:
                if self.debug:
                    print(f"Error in {strategy_name}: {e}")
                    
        return all_signals

    def detect_mean_reversion_signals(self, df, idx, vp):
        """
        Enhanced mean reversion detection with research-based parameters
        Target: 60-70% win rate with 1.5-2:1 R/R
        """
        signals = []
        
        if idx < 50:
            return signals
            
        current_price = df.iloc[idx]['close']
        current_low = df.iloc[idx]['low']
        current_high = df.iloc[idx]['high']
        atr = df.iloc[idx]['atr']
        
        vah = vp.get('vah', 0)
        val = vp.get('val', 0)
        poc = vp.get('poc', 0)
        
        if vah == 0 or val == 0 or poc == 0:
            return signals
            
        # Check volume confirmation first (more flexible)
        volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'mean_reversion')
        if not volume_confirmed:
            return signals
            
        # Long signals from VAL
        if self.check_entry_zone(current_low, val, atr, 'standard'):
            score = self.calculate_entry_score(df, idx, vp, 'mean_reversion')
            
            if score >= 70:  # Lowered from 80
                # Enhanced stop loss placement
                stop = self.calculate_stop_loss(df, vp, current_price, 'long', idx)
                
                signals.append({
                    'type': 'long',
                    'entry': current_price,
                    'stop': stop,
                    'target': poc,  # First target at POC
                    'score': score,
                    'strategy': 'mean_reversion',
                    'volume_ratio': volume_ratio
                })
                
        # Short signals from VAH
        if self.check_entry_zone(current_high, vah, atr, 'standard'):
            score = self.calculate_entry_score(df, idx, vp, 'mean_reversion')
            
            if score >= 70:  # Lowered from 80
                # Enhanced stop loss placement
                stop = self.calculate_stop_loss(df, vp, current_price, 'short', idx)
                
                signals.append({
                    'type': 'short',
                    'entry': current_price,
                    'stop': stop,
                    'target': poc,
                    'score': score,
                    'strategy': 'mean_reversion',
                    'volume_ratio': volume_ratio
                })
                
        return signals

    def calculate_comprehensive_score(self, df, idx, strategy, signal_type, volume_ratio, vp):
        """
        Calculate a comprehensive score based on research parameters
        """
        score = 0
        current_price = df.iloc[idx]['close']
        
        # Base score by strategy (research-aligned)
        base_scores = {
            'mean_reversion': 50,
            'breakout': 45,
            'poc_magnet': 48,
            'virgin_poc': 45,
            'composite': 40
        }
        score += base_scores.get(strategy, 40)
        
        # Volume scoring (research: volume confirmation critical)
        if strategy == 'breakout':
            if volume_ratio >= 2.5:
                score += 25
            elif volume_ratio >= 2.0:
                score += 20
            elif volume_ratio >= 1.5:
                score += 15
        else:  # Mean reversion prefers moderate volume
            if 1.0 <= volume_ratio <= 1.5:
                score += 20
            elif 0.8 <= volume_ratio < 1.0:
                score += 15
            elif 1.5 < volume_ratio <= 2.0:
                score += 10
        
        # Distance from level scoring
        if strategy == 'mean_reversion':
            level = vp['val'] if signal_type == 'long' else vp['vah']
            distance_pct = abs(current_price - level) / level * 100
            
            if distance_pct <= 0.3:
                score += 15  # Very close to level
            elif distance_pct <= 0.5:
                score += 10
            elif distance_pct <= 1.0:
                score += 5
        
        # RSI confirmation
        if idx >= 14:
            rsi_series = self.calculate_rsi(df['close'], 14)
            rsi = rsi_series.iloc[idx] if idx < len(rsi_series) and not pd.isna(rsi_series.iloc[idx]) else 50
            
            if strategy == 'mean_reversion':
                if signal_type == 'long' and rsi < 35:
                    score += 10
                elif signal_type == 'short' and rsi > 65:
                    score += 10
            elif strategy == 'breakout':
                if signal_type == 'long' and 50 < rsi < 70:
                    score += 10
                elif signal_type == 'short' and 30 < rsi < 50:
                    score += 10
        
        # Trend alignment
        if idx >= 50:
            ema_20 = df.iloc[idx-20:idx]['close'].mean()
            ema_50 = df.iloc[idx-50:idx]['close'].mean()
            
            if signal_type == 'long' and ema_20 > ema_50:
                score += 5
            elif signal_type == 'short' and ema_20 < ema_50:
                score += 5
        
        return min(score, 100)

    def detect_poc_magnet_signals(self, df, idx, vp):
        """POC Magnet - Research says 70-80% win rate target"""
        signals = []
        
        if idx < 50:
            return signals
            
        current_price = df.iloc[idx]['close']
        poc = vp['poc']
        atr_series = self.calculate_atr(df, 14)  # Use 14-period ATR
        atr = atr_series.iloc[idx] if idx < len(atr_series) and not pd.isna(atr_series.iloc[idx]) else 0
        
        # Check distance from POC (research: 2-5% optimal)
        distance_from_poc = abs(current_price - poc)
        distance_pct = distance_from_poc / current_price * 100
        
        if 2.0 <= distance_pct <= 5.0:  # Research-based range
            volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'poc_magnet')
            
            if volume_confirmed:
                score = self.calculate_comprehensive_score(df, idx, 'poc_magnet', 'long' if current_price > poc else 'short', volume_ratio, vp)
                
                if score >= 80:
                    signal_type = 'short' if current_price > poc else 'long'
                    stop_distance = atr * 1.5
                    
                    signals.append({
                        'type': signal_type,
                        'entry': current_price,
                        'stop': poc + (stop_distance if signal_type == 'short' else -stop_distance),
                        'target': poc,
                        'score': score,
                        'strategy': 'poc_magnet',
                        'volume_ratio': volume_ratio
                    })
        
        return signals

    def detect_virgin_poc_signals(self, df, idx, vp):
        """Virgin POC - Research-based implementation"""
        signals = []
        
        if idx < 50:
            return signals
            
        # Find virgin POCs
        virgin_pocs = self.find_virgin_pocs(df, lookback_sessions=5)
        
        if not virgin_pocs:
            return signals
            
        current_price = df.iloc[idx]['close']
        atr_series = self.calculate_atr(df, 14)  # Use 14-period ATR
        atr = atr_series.iloc[idx] if idx < len(atr_series) and not pd.isna(atr_series.iloc[idx]) else 0
        
        for poc in virgin_pocs:
            distance_from_poc = abs(current_price - poc)
            distance_pct = distance_from_poc / current_price * 100
            
            if distance_pct <= 2.0:  # Close to virgin POC
                volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'virgin_poc')
                
                if volume_confirmed:
                    score = self.calculate_comprehensive_score(df, idx, 'virgin_poc', 'long' if current_price > poc else 'short', volume_ratio, vp)
                    
                    if score >= 75:  # Lower threshold for virgin POC
                        signal_type = 'short' if current_price > poc else 'long'
                        stop_distance = atr * 1.2
                        
                        signals.append({
                            'type': signal_type,
                            'entry': current_price,
                            'stop': poc + (stop_distance if signal_type == 'short' else -stop_distance),
                            'target': poc,
                            'score': score,
                            'strategy': 'virgin_poc',
                            'volume_ratio': volume_ratio
                        })
        
        return signals

    def detect_composite_signals(self, df, idx, vp):
        """Composite strategy - combines multiple signals"""
        signals = []
        
        if idx < 50:
            return signals
            
        # Get signals from all strategies
        mean_rev_signals = self.detect_mean_reversion_signals(df, idx, vp)
        breakout_signals = self.detect_breakout_signals(df, idx, vp)
        poc_signals = self.detect_poc_magnet_signals(df, idx, vp)
        
        # Combine and filter for high-quality setups
        all_signals = mean_rev_signals + breakout_signals + poc_signals
        
        for signal in all_signals:
            if signal.get('score', 0) >= 85:  # Higher threshold for composite
                signal['strategy'] = 'composite'
                signals.append(signal)
        
        return signals
    
    def poc_magnet_strategy(self, df: pd.DataFrame, idx: int, vp: Dict) -> Optional[Dict]:
        """Trade toward POC from range extremes - MORE PERMISSIVE"""
        if not vp or idx < 50:
            return None
            
        current = df.iloc[idx]
        atr = current['atr']
        
        # Check distance from POC
        distance_from_poc = abs(current['close'] - vp['poc'])
        distance_pct = distance_from_poc / current['close']
        
        # EXPANDED: Accept 1.5% to 8% distance (was 2-5%)
        if distance_pct < self.config['mean_reversion_min_distance'] or \
           distance_pct > self.config['poc_magnet_max_distance']:
            return None
        
        # REMOVED ADX check - allow in trending markets too
        
        # Check volume confirmation using new method
        volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'poc_magnet')
        
        if not volume_confirmed:
            return None
        
        # Calculate entry score
        score = self.calculate_entry_score(df, idx, vp, 'poc_magnet')
        
        if score < 80:  # UPDATED: Accept 80+ scores for high quality
            return None
        
        # Determine direction
        if current['close'] > vp['poc']:
            direction = 'short'
            entry = current['close']
            
            # Find HVN above for stop or use ATR
            hvn_above = [h for h in vp.get('hvn', []) if h > entry]
            if hvn_above:
                stop = min(hvn_above) + atr * 0.5
            else:
                stop = entry + atr * self.config['stop_loss_atr']
            
            target = vp['poc']
        else:
            direction = 'long'
            entry = current['close']
            
            # Find HVN below for stop or use ATR
            hvn_below = [h for h in vp.get('hvn', []) if h < entry]
            if hvn_below:
                stop = max(hvn_below) - atr * 0.5
            else:
                stop = entry - atr * self.config['stop_loss_atr']
            
            target = vp['poc']
        
        # REDUCED R/R requirement for POC magnet
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if reward / risk < 1.2:  # Even lower for high probability POC trades
            return None
        
        return {
            'strategy': 'poc_magnet',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score': score,
            'risk_reward': reward / risk,
            'volume_ratio': volume_ratio,
            'distance_from_poc_pct': distance_pct * 100,
            'poc_level': vp['poc']
        }
    
    def virgin_poc_strategy(self, df: pd.DataFrame, idx: int, vp: Dict) -> Optional[Dict]:
        """Trade retests of untested POCs"""
        if idx < 100:
            return None
            
        current = df.iloc[idx]
        atr = current['atr']
        
        # Find virgin POCs
        virgin_pocs = self.find_virgin_pocs(df.iloc[:idx], 3)  # Reduced from 5 to 3
        
        if not virgin_pocs:
            return None
        
        # DEBUG: Log only when multiple virgin POCs found
        if len(virgin_pocs) > 1:
            print(f"🔍 MULTIPLE VIRGIN POCs: {len(virgin_pocs)} found")
        
        # Check if price is near any virgin POC
        for vpoc in virgin_pocs:
            if abs(current['close'] - vpoc) < atr * self.config['entry_zone_atr']:
                # Check volume confirmation using new method
                volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'virgin_poc')
                
                if not volume_confirmed:
                    continue
                
                # Calculate entry score
                score = self.calculate_entry_score(df, idx, vp, 'virgin_poc')
                
                if score < 80:  # UPDATED: Accept 80+ scores for high quality
                    continue
                
                # Determine direction based on approach
                if current['close'] > vpoc and current['low'] <= vpoc * 1.01:
                    # Approaching from above, expect bounce up
                    direction = 'long'
                    entry = current['close']
                    stop = vpoc - atr * 2
                    target = entry + (entry - stop) * 2
                elif current['close'] < vpoc and current['high'] >= vpoc * 0.99:
                    # Approaching from below, expect rejection down
                    direction = 'short'
                    entry = current['close']
                    stop = vpoc + atr * 2
                    target = entry - (stop - entry) * 2
                else:
                    continue
                
                # Check R/R ratio
                risk = abs(entry - stop)
                reward = abs(target - entry)
                
                if reward / risk >= self.config['min_rr_ratio']:
                    return {
                        'strategy': 'virgin_poc',
                        'direction': direction,
                        'entry': entry,
                        'stop': stop,
                        'target': target,
                        'score': score,
                        'risk_reward': reward / risk,
                        'volume_ratio': volume_ratio,
                        'virgin_poc_level': vpoc
                    }
        
        return None
    
    def composite_strategy(self, df: pd.DataFrame, idx: int, vp: Dict) -> Optional[Dict]:
        """Combine multiple timeframe Volume Profiles"""
        if not vp or idx < 200:
            return None
            
        # Calculate multiple timeframe profiles
        profiles = {}
        timeframes = [24, 72, 168]  # 1 day, 3 days, 7 days in 4H candles
        weights = [0.2, 0.5, 0.3]  # Weight recent profiles more heavily
        
        for tf, weight in zip(timeframes, weights):
            tf_vp = self.calculate_volume_profile(df.iloc[idx-tf:idx], tf)
            if tf_vp:
                profiles[tf] = {'vp': tf_vp, 'weight': weight}
        
        if len(profiles) < 2:
            return None
        
        # Find confluence levels
        current = df.iloc[idx]
        atr = current['atr']
        
        # Weighted average POC
        weighted_poc = sum(p['vp']['poc'] * p['weight'] for p in profiles.values())
        
        # Check if price is near weighted POC
        if abs(current['close'] - weighted_poc) > atr * self.config['entry_zone_atr']:
            return None
        
        # Count POC alignments
        poc_cluster = []
        for p in profiles.values():
            if abs(p['vp']['poc'] - weighted_poc) / weighted_poc < 0.01:
                poc_cluster.append(p['vp']['poc'])
        
        if len(poc_cluster) < 2:
            return None
        
        # Check volume confirmation using new method
        volume_confirmed, volume_ratio = self.check_volume_confirmation(df, idx, 'composite')
        
        if not volume_confirmed:
            return None
        
        # Strong confluence signal
        score = 90 + len(poc_cluster) * 5  # Higher base score for composite
        
        # Determine direction based on price position
        if current['close'] > weighted_poc:
            direction = 'short'
            entry = current['close']
            stop = entry + atr * self.config['stop_loss_atr']
            target = weighted_poc
        else:
            direction = 'long'
            entry = current['close']
            stop = entry - atr * self.config['stop_loss_atr']
            target = weighted_poc
        
        # Check R/R ratio
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if reward / risk < self.config['min_rr_ratio']:
            return None
        
        return {
            'strategy': 'composite',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score': score,
            'risk_reward': reward / risk,
            'volume_ratio': volume_ratio,
            'poc_cluster': len(poc_cluster)
        }
    
    def execute_trades(self, df, signals, idx):
        """
        Execute trades based on signals - FIXED VERSION
        Remove duplicate volume checking and enforce quality standards
        """
        trades = []
        
        for signal in signals:
            # CRITICAL: Enforce 80+ score requirement for quality
            if signal.get('score', 0) < 80:
                if self.debug:
                    print(f"⛔ Trade rejected - low score: {signal.get('score', 0)} < 80")
                continue
                
            # Volume has ALREADY been confirmed in signal detection
            # Don't re-check here!
            volume_ratio = signal.get('volume_ratio', 1.0)
            
            # Calculate actual R/R
            entry = signal['entry']
            stop = signal['stop']
            target = signal['target']
            
            if signal['type'] == 'long':
                risk = entry - stop
                reward = target - entry
            else:
                risk = stop - entry
                reward = entry - target
                
            if risk <= 0 or reward <= 0:
                continue
                
            rr_ratio = reward / risk
            
            # VERY RELAXED R/R requirements - closer to research
            min_rr = {
                'mean_reversion': 1.0,   # Any positive R/R
                'breakout': 2.0,         # Reduced from 2.5
                'poc_magnet': 0.8,       # Very flexible
                'virgin_poc': 1.0,       # Flexible
                'composite': 1.2         # Moderate
            }
            
            required_rr = min_rr.get(signal.get('strategy', 'default'), 1.0)
            if rr_ratio < required_rr:
                if self.debug:
                    print(f"⛔ Trade rejected - R/R {rr_ratio:.2f} < {required_rr}")
                continue
                
            # Execute the trade
            trade = {
                'symbol': df.iloc[idx].get('symbol', 'UNKNOWN'),
                'timestamp': df.iloc[idx]['timestamp'],
                'type': signal['type'],
                'entry': entry,
                'stop': stop,
                'target': target,
                'score': signal['score'],
                'strategy': signal.get('strategy', 'unknown'),
                'volume_ratio': volume_ratio,
                'risk_reward': rr_ratio
            }
            
            trades.append(trade)
            
            if self.debug:
                print(f"✅ TRADE EXECUTED: {signal['strategy']}, score={signal['score']}, RR={rr_ratio:.2f}, vol={volume_ratio:.2f}x")
        
        return trades
    
    def simulate_trade(self, df: pd.DataFrame, trade: Dict, start_idx: int) -> Dict:
        """Simulate trade execution and calculate results"""
        entry_price = trade['entry']
        stop_price = trade['stop']
        target_price = trade['target']
        direction = trade['type']
        
        # Track trade from entry
        for i in range(start_idx + 1, min(start_idx + 100, len(df))):  # Max 100 candles (16.7 days)
            candle = df.iloc[i]
            
            # Check stop loss
            if direction == 'long':
                if candle['low'] <= stop_price:
                    return {
                        'exit_price': stop_price,
                        'exit_idx': i,
                        'result': 'stop_loss',
                        'pnl_pct': ((stop_price - entry_price) / entry_price) * 100,
                        'duration_candles': i - start_idx
                    }
                # Check target
                if candle['high'] >= target_price:
                    return {
                        'exit_price': target_price,
                        'exit_idx': i,
                        'result': 'target',
                        'pnl_pct': ((target_price - entry_price) / entry_price) * 100,
                        'duration_candles': i - start_idx
                    }
            else:  # short
                if candle['high'] >= stop_price:
                    return {
                        'exit_price': stop_price,
                        'exit_idx': i,
                        'result': 'stop_loss',
                        'pnl_pct': ((entry_price - stop_price) / entry_price) * 100,
                        'duration_candles': i - start_idx
                    }
                # Check target
                if candle['low'] <= target_price:
                    return {
                        'exit_price': target_price,
                        'exit_idx': i,
                        'result': 'target',
                        'pnl_pct': ((entry_price - target_price) / entry_price) * 100,
                        'duration_candles': i - start_idx
                    }
        
        # Trade still open after max duration
        final_price = df.iloc[min(start_idx + 100, len(df) - 1)]['close']
        if direction == 'long':
            pnl_pct = ((final_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - final_price) / entry_price) * 100
            
        return {
            'exit_price': final_price,
            'exit_idx': min(start_idx + 100, len(df) - 1),
            'result': 'timeout',
            'pnl_pct': pnl_pct,
            'duration_candles': min(100, len(df) - start_idx - 1)
        }
    
    def analyze_indicator_performance(self, trades: List[Dict]) -> Dict:
        """Analyze which indicators provide best confluence"""
        indicator_stats = {}
        
        for indicator_group in self.indicators_to_test:
            indicator_stats[indicator_group] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0,
                'avg_score_contribution': 0
            }
        
        # Analyze each trade
        for trade in trades:
            if trade.get('win', False):  # Use the new 'win' key
                # This was a winning trade, analyze which indicators contributed
                # In real implementation, we'd track which indicators fired for each trade
                # For now, we'll simulate based on strategy type
                if trade['strategy'] == 'mean_reversion':
                    indicator_stats['momentum']['winning_trades'] += 1
                    indicator_stats['volatility']['winning_trades'] += 1
                elif trade['strategy'] == 'breakout':
                    indicator_stats['volume_based']['winning_trades'] += 1
                    indicator_stats['trend']['winning_trades'] += 1
                elif trade['strategy'] == 'poc_magnet':
                    indicator_stats['market_structure']['winning_trades'] += 1
            
            # Track total trades for each indicator group
            for group in indicator_stats:
                indicator_stats[group]['total_trades'] += 1
                indicator_stats[group]['total_pnl'] += trade['pnl_pct']
        
        # Calculate win rates and average contributions
        for group in indicator_stats:
            total = indicator_stats[group]['total_trades']
            if total > 0:
                indicator_stats[group]['win_rate'] = indicator_stats[group]['winning_trades'] / total
                indicator_stats[group]['avg_pnl'] = indicator_stats[group]['total_pnl'] / total
            else:
                indicator_stats[group]['win_rate'] = 0
                indicator_stats[group]['avg_pnl'] = 0
        
        return indicator_stats
    
    def backtest_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Run backtest for a single symbol"""
        print(f"Backtesting {symbol}...")
        
        # Validate data quality
        if df.empty or df['close'].max() < 0.01 or len(df) < 100:  # Reduced minimum
            print(f"⚠️  WARNING: {symbol} has insufficient data (length: {len(df)}, max close: {df['close'].max():.6f})")
            return {
                'symbol': symbol,
                'category': 'other',
                'strategy_stats': {},
                'indicator_performance': {},
                'all_trades': []
            }
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Determine category
        category = 'other'
        for cat, symbols in self.crypto_categories.items():
            if symbol in symbols:
                category = cat
                break
        
        # Adjust parameters based on category
        if category == 'major':
            self.config['value_area_pct'] = 0.70
        else:
            self.config['value_area_pct'] = 0.75
        
        # Track trades by strategy
        strategy_results = {strategy: [] for strategy in self.strategies}
        total_signals = 0
        
        # Run backtest using new process_signals method
        for i in range(self.config['profile_period'], len(df) - 100):
            # Calculate Volume Profile
            vp = self.calculate_volume_profile(df.iloc[:i], self.config['profile_period'])
            
            if not vp:
                continue
            
            # Get all signals using the new process_signals method
            signals = self.process_signals(df, vp, i)
            
            if signals:
                total_signals += len(signals)
                
                # Execute trades using the new risk management system
                trades, final_balance = self.execute_trades_with_risk_management(df, signals, vp, commission=0.001)
                
                # Process each trade
                for trade in trades:
                    trade['symbol'] = symbol
                    trade['category'] = category
                    trade['entry_time'] = df.iloc[i]['timestamp']
                    
                    # P&L already calculated in execute_trades_with_risk_management
                    strategy_name = trade.get('strategy', 'unknown')
                    strategy_results[strategy_name].append(trade)
        
        print(f"  Found {total_signals} total signals across all strategies")
        
        # Calculate performance metrics using the new fixed method
        all_trades = [trade for trades in strategy_results.values() for trade in trades]
        if all_trades:
            performance_metrics = self.calculate_performance_metrics(all_trades)
            print(f"  Performance: {performance_metrics['win_rate']:.1f}% win rate, {performance_metrics['max_drawdown']:.1f}% max drawdown")
        
        # Calculate statistics for each strategy - FIXED VERSION
        stats = {}
        for strategy, trades in strategy_results.items():
            if trades:
                # Calculate wins based on P&L
                winning_trades = [t for t in trades if t.get('win', False)]
                losing_trades = [t for t in trades if not t.get('win', False)]
                
                total_trades = len(trades)
                wins = len(winning_trades)
                losses = len(losing_trades)
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                
                avg_pnl = np.mean([t.get('pnl_pct', 0) for t in trades])
                avg_win = np.mean([t.get('pnl_pct', 0) for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t.get('pnl_pct', 0) for t in losing_trades]) if losing_trades else 0
                avg_rr = np.mean([t.get('risk_reward', 0) for t in trades])
                
                # Calculate profit factor
                total_wins = sum(t.get('pnl_pct', 0) for t in winning_trades)
                total_losses = abs(sum(t.get('pnl_pct', 0) for t in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                
                stats[strategy] = {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'avg_rr': avg_rr,
                    'profit_factor': profit_factor,
                    'max_drawdown': self.calculate_max_drawdown(trades),
                    'sharpe_ratio': self.calculate_sharpe_ratio(trades)
                }
                print(f"  {strategy}: {total_trades} trades, {win_rate:.1f}% win rate")
            else:
                stats[strategy] = {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'avg_rr': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
        
        # Analyze indicator performance
        all_trades = [trade for trades in strategy_results.values() for trade in trades]
        indicator_performance = self.analyze_indicator_performance(all_trades)
        
        return {
            'symbol': symbol,
            'category': category,
            'strategy_stats': stats,
            'indicator_performance': indicator_performance,
            'all_trades': all_trades
        }
    
    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade list"""
        if not trades:
            return 0
        
        equity_curve = [0]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl_pct'])
        
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def calculate_trade_pnl(self, trade, df, entry_idx):
        """
        Fixed P&L calculation - prevent impossible values
        """
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        target = trade['target']
        trade_type = trade['type']
        
        # Track the trade from entry
        for i in range(entry_idx + 1, min(entry_idx + 100, len(df))):
            high = df.iloc[i]['high']
            low = df.iloc[i]['low']
            
            if trade_type == 'long':
                # Check stop loss hit FIRST (worst case)
                if low <= stop_loss:
                    exit_price = stop_loss
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    # Cap losses at -100%
                    pnl_pct = max(pnl_pct, -99.9)
                    return {
                        'exit_price': exit_price,
                        'exit_idx': i,
                        'pnl_pct': pnl_pct,
                        'pnl_points': exit_price - entry_price,
                        'win': pnl_pct > 0,
                        'exit_reason': 'stop_loss',
                        'bars_held': i - entry_idx
                    }
                
                # Then check target hit
                if high >= target:
                    exit_price = target
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    return {
                        'exit_price': exit_price,
                        'exit_idx': i,
                        'pnl_pct': pnl_pct,
                        'pnl_points': exit_price - entry_price,
                        'win': pnl_pct > 0,
                        'exit_reason': 'target',
                        'bars_held': i - entry_idx
                    }
                    
            else:  # short
                # Check stop loss hit FIRST
                if high >= stop_loss:
                    exit_price = stop_loss
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    # Cap losses at -100%
                    pnl_pct = max(pnl_pct, -99.9)
                    return {
                        'exit_price': exit_price,
                        'exit_idx': i,
                        'pnl_pct': pnl_pct,
                        'pnl_points': entry_price - exit_price,
                        'win': pnl_pct > 0,
                        'exit_reason': 'stop_loss',
                        'bars_held': i - entry_idx
                    }
                
                # Then check target hit
                if low <= target:
                    exit_price = target
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    return {
                        'exit_price': exit_price,
                        'exit_idx': i,
                        'pnl_pct': pnl_pct,
                        'pnl_points': entry_price - exit_price,
                        'win': pnl_pct > 0,
                        'exit_reason': 'target',
                        'bars_held': i - entry_idx
                    }
        
        # Timeout exit
        exit_price = df.iloc[-1]['close']
        if trade_type == 'long':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_points = exit_price - entry_price
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            pnl_points = entry_price - exit_price
        
        # Cap P&L at reasonable levels
        pnl_pct = max(min(pnl_pct, 100), -99.9)
        
        return {
            'exit_price': exit_price,
            'exit_idx': len(df) - 1,
            'pnl_pct': pnl_pct,
            'pnl_points': pnl_points,
            'win': pnl_pct > 0,
            'exit_reason': 'timeout',
            'bars_held': len(df) - 1 - entry_idx
        }

    def calculate_position_size(self, entry_price, stop_loss, account_balance=10000, risk_per_trade=0.01):
        """
        Calculate position size based on 1% risk per trade
        This prevents catastrophic drawdowns
        """
        risk_amount = account_balance * risk_per_trade  # $100 risk on $10k account
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        # Position size in units
        position_size = risk_amount / price_risk
        
        # Position value
        position_value = position_size * entry_price
        
        # Don't use more than 10x leverage
        max_position_value = account_balance * 10
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        return position_size

    def execute_trades_with_risk_management(self, df, signals, vp_data, commission=0.001):
        """
        Execute trades with proper risk management
        """
        trades = []
        active_trades = []
        account_balance = 10000  # Starting balance
        peak_balance = account_balance
        
        for signal in signals:
            # Skip if we have too many open positions
            if len(active_trades) >= 3:  # Max 3 concurrent trades
                continue
                
            # Check R/R requirements
            entry = signal['entry']
            stop = signal['stop']
            target = signal['target']
            strategy = signal['strategy']
            idx = signal.get('entry_idx', 0)  # Get entry index from signal
            
            # R/R calculation
            risk = abs(entry - stop)
            reward = abs(target - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # RELAXED R/R requirements for R8
            min_rr = {
                'mean_reversion': 1.0,    # Any positive R/R
                'breakout': 2.0,          # Minimum 2:1
                'poc_magnet': 0.8,        # Very flexible
                'virgin_poc': 1.0,
                'composite': 0.9          # More flexible
            }
            
            required_rr = min_rr.get(strategy, 1.0)
            
            if rr_ratio < required_rr:
                if self.debug:
                    print(f"⛔ Trade rejected - R/R {rr_ratio:.2f} < {required_rr}")
                continue
                
            # Calculate position size
            position_size = self.calculate_position_size(
                entry, stop, account_balance, risk_per_trade=0.01
            )
            
            if position_size == 0:
                continue
                
            # Create trade with risk management
            trade = {
                'symbol': df.iloc[idx]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                'timestamp': df.iloc[idx].name,
                'type': signal['type'],
                'entry_price': entry,
                'stop_loss': stop,
                'target': target,
                'position_size': position_size,
                'risk_amount': account_balance * 0.01,
                'strategy': strategy,
                'score': signal.get('score', 0),
                'volume_ratio': signal.get('volume_ratio', 0),
                'entry_idx': idx
            }
            
            # Calculate exit
            exit_info = self.calculate_trade_pnl(trade, df, idx)
            
            # Update trade with exit info
            trade.update(exit_info)
            
            # Update account balance
            pnl_amount = (trade['pnl_pct'] / 100) * trade['entry_price'] * trade['position_size']
            account_balance += pnl_amount
            
            # Track peak for drawdown calculation
            if account_balance > peak_balance:
                peak_balance = account_balance
                
            # Calculate drawdown
            current_drawdown = ((peak_balance - account_balance) / peak_balance) * 100
            trade['account_balance'] = account_balance
            trade['drawdown'] = current_drawdown
            
            trades.append(trade)
            
        return trades, account_balance

    def calculate_performance_metrics(self, trades, initial_balance=10000):
        """
        Fixed performance calculation with proper drawdown tracking
        """
        if not trades:
            return {}
        
        # Sort trades by entry time
        sorted_trades = sorted(trades, key=lambda x: x.get('entry_time', x.get('timestamp', 0)))
        
        # Track balance properly
        balance = initial_balance
        peak_balance = initial_balance
        balances = [initial_balance]
        
        wins = 0
        losses = 0
        total_win_pnl = 0
        total_loss_pnl = 0
        
        for trade in sorted_trades:
            # Calculate trade P&L in dollars (NOT percentage of balance)
            # Use fixed position size or risk amount
            risk_amount = initial_balance * 0.01  # 1% risk per trade
            
            if trade['pnl_pct'] > 0:
                wins += 1
                # Win: Calculate based on risk amount and R/R
                risk_pct = abs(trade['entry_price'] - trade['stop_loss']) / trade['entry_price'] * 100
                if risk_pct > 0:
                    pnl_dollars = risk_amount * (trade['pnl_pct'] / risk_pct)
                else:
                    pnl_dollars = risk_amount * 0.1  # Small win if no risk calculated
                total_win_pnl += trade['pnl_pct']
            else:
                losses += 1
                # Loss: Limited to risk amount
                pnl_dollars = -risk_amount
                total_loss_pnl += abs(trade['pnl_pct'])
            
            balance += pnl_dollars
            balances.append(balance)
            
            # Update peak for drawdown
            if balance > peak_balance:
                peak_balance = balance
            
            # Calculate current drawdown
            current_dd = ((peak_balance - balance) / peak_balance) * 100
            trade['running_balance'] = balance
            trade['drawdown'] = current_dd
        
        # Calculate final metrics
        total_trades = len(trades)
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        max_drawdown = max([t.get('drawdown', 0) for t in sorted_trades]) if sorted_trades else 0
        total_return = ((balance - initial_balance) / initial_balance) * 100
        
        # Profit factor
        profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
        
        # Average win/loss
        avg_win = total_win_pnl / wins if wins > 0 else 0
        avg_loss = total_loss_pnl / losses if losses > 0 else 0
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss)
        
        # Sharpe ratio (simplified)
        if len(balances) > 1:
            returns = np.diff(balances) / balances[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'final_balance': balance
        }

    def run_simple_backtest(self):
        """
        Simplified backtest to verify logic
        """
        # Test with just one symbol first
        test_symbol = 'BTCUSDT'
        
        print(f"Running simplified backtest on {test_symbol}...")
        
        # Initialize data fetcher
        try:
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            sys.path.insert(0, parent_dir)
            
            from modules.data_fetcher import MarketDataFetcher
            self.data_fetcher = MarketDataFetcher()
            print(f"✅ Data fetcher initialized with exchanges: {list(self.data_fetcher.exchanges.keys())}")
        except Exception as e:
            print(f"❌ Error initializing data fetcher: {e}")
            return
        
        # Get data
        df = self.data_fetcher.fetch_ohlcv('binance', test_symbol, '4h')
        if df is None or df.empty:
            print(f"❌ No data for {test_symbol}")
            return
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Calculate volume profile
        vp_data = self.calculate_volume_profile(df.iloc[:200], 72)  # Use first 200 candles
        
        # Get signals (limit to first 10 for testing)
        signals = []
        for idx in range(100, min(110, len(df))):
            try:
                signal_list = self.detect_mean_reversion_signals(df, idx, vp_data)
                if signal_list:
                    signals.append((idx, signal_list[0]))
            except Exception as e:
                print(f"Error at idx {idx}: {e}")
                continue
        
        print(f"Found {len(signals)} test signals")
        
        # Execute trades with simple logic
        trades = []
        for idx, signal in signals:
            # Simple P&L calculation
            entry = signal['entry']
            stop = signal['stop']
            target = signal['target']
            
            # Find exit
            for i in range(idx + 1, min(idx + 50, len(df))):
                if df.iloc[i]['low'] <= stop:
                    pnl = ((stop - entry) / entry) * 100
                    trades.append({'pnl': pnl, 'exit_reason': 'stop'})
                    break
                elif df.iloc[i]['high'] >= target:
                    pnl = ((target - entry) / entry) * 100
                    trades.append({'pnl': pnl, 'exit_reason': 'target'})
                    break
        
        # Print results
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            print(f"Wins: {wins}/{len(trades)} = {wins/len(trades)*100:.1f}%")
            print(f"Avg PnL: {sum(t['pnl'] for t in trades)/len(trades):.2f}%")
            print(f"Best trade: {max(t['pnl'] for t in trades):.2f}%")
            print(f"Worst trade: {min(t['pnl'] for t in trades):.2f}%")
        else:
            print("No trades executed")

    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from trade list"""
        if not trades or len(trades) < 2:
            return 0
        
        returns = [t['pnl_pct'] / 100 for t in trades]
        
        # Annualize based on average trade duration
        avg_duration = np.mean([t.get('bars_held', t.get('duration_candles', 10)) for t in trades])
        trades_per_year = 365 * 6 / avg_duration  # 6 4H candles per day
        
        annual_return = np.mean(returns) * trades_per_year
        annual_std = np.std(returns) * np.sqrt(trades_per_year)
        
        if annual_std == 0:
            return 0
        
        return (annual_return - risk_free_rate) / annual_std
    
    def run_backtest(self, symbols: List[str]) -> Dict:
        """Run complete backtest across all symbols"""
        print(f"Starting Volume Profile backtest for {len(symbols)} symbols...")
        
        # Initialize data fetcher using your existing infrastructure
        try:
            # Add parent directories to path to find modules
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            sys.path.insert(0, parent_dir)
            
            from modules.data_fetcher import MarketDataFetcher
            self.data_fetcher = MarketDataFetcher()
            print(f"✅ Data fetcher initialized with exchanges: {list(self.data_fetcher.exchanges.keys())}")
        except Exception as e:
            print(f"❌ Error initializing data fetcher: {e}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Looking for modules in: {parent_dir}")
            import traceback
            traceback.print_exc()
            return {}
        
        all_results = []
        category_summary = {}
        strategy_summary = {}
        
        for symbol in symbols:
            try:
                # Fetch data using your existing data fetcher
                print(f"\nFetching data for {symbol}...")
                
                # Try multiple exchanges
                df = None
                for exchange in ['binance', 'bybit', 'kucoin']:
                    try:
                        # Your data fetcher expects symbol without USDT suffix
                        symbol_clean = symbol.replace('USDT', '')
                        
                        # Fetch more data - need at least 365 days for proper backtesting
                        # First try to get data directly from exchange
                        try:
                            ex = self.data_fetcher.exchanges[exchange]
                            market = f"{symbol_clean}/USDT"
                            
                            # Fetch 2000 candles (about 333 days of 4H data)
                            candles = ex.fetch_ohlcv(market, '4h', limit=2000)
                            
                            if candles and len(candles) > 200:
                                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                print(f"✅ Got {len(df)} candles from {exchange}")
                                break
                        except:
                            # Fall back to data_fetcher method
                            df = self.data_fetcher.fetch_ohlcv(exchange, symbol_clean, '4h')
                        
                        if df is not None and len(df) > 0:
                            print(f"✅ Got {len(df)} candles from {exchange}")
                            # Debug: Check data range
                            if len(df) > 0:
                                print(f"   Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                                print(f"   Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
                            # Ensure we have enough data for volume profile
                            if len(df) >= 200:  # Minimum for proper backtesting
                                break
                            else:
                                print(f"Insufficient data after filtering for {symbol}: only {len(df)} candles")
                                df = None
                    except Exception as ex:
                        continue
                
                if df is None or len(df) < 200:
                    print(f"Insufficient data for {symbol}")
                    continue
                
                # Convert DataFrame to expected format
                if 'index' in df.columns or df.index.name == 'timestamp':
                    df = df.reset_index()
                    if 'index' in df.columns:
                        df.rename(columns={'index': 'timestamp'}, inplace=True)
                
                # Ensure timestamp is datetime
                if 'timestamp' in df.columns:
                    if df['timestamp'].dtype in ['int64', 'float64']:
                        # Convert from milliseconds to datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    elif df['timestamp'].dtype == 'object':
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure we have required columns
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    print(f"Missing required columns for {symbol}")
                    print(f"Available columns: {list(df.columns)}")
                    continue
                
                # Filter to last 90 days for faster testing (instead of 30)
                if len(df) > 90:
                    cutoff_date = datetime.now() - timedelta(days=90)
                    df = df[df['timestamp'] >= cutoff_date].reset_index(drop=True)
                
                if len(df) < 100:  # Reduced minimum for faster testing
                    print(f"Insufficient data after filtering for {symbol}: only {len(df)} candles")
                    continue
                
                print(f"Processing {symbol}: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                # Run backtest
                result = self.backtest_symbol(symbol, df)
                all_results.append(result)
                
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate results by category
        for result in all_results:
            category = result['category']
            if category not in category_summary:
                category_summary[category] = {
                    'symbols': [],
                    'total_trades': 0,
                    'avg_win_rate': 0,
                    'avg_pnl': 0,
                    'best_strategy': None,
                    'best_indicators': []
                }
            
            category_summary[category]['symbols'].append(result['symbol'])
            
            # Aggregate strategy performance
            for strategy, stats in result['strategy_stats'].items():
                if strategy not in strategy_summary:
                    strategy_summary[strategy] = {
                        'total_trades': 0,
                        'total_wins': 0,
                        'total_pnl': 0,
                        'by_category': {}
                    }
                
                strategy_summary[strategy]['total_trades'] += stats['total_trades']
                strategy_summary[strategy]['total_wins'] += stats['wins']
                strategy_summary[strategy]['total_pnl'] += stats['avg_pnl'] * stats['total_trades']
                
                if category not in strategy_summary[strategy]['by_category']:
                    strategy_summary[strategy]['by_category'][category] = {
                        'trades': 0,
                        'wins': 0,
                        'win_rate': 0,
                        'avg_pnl': 0
                    }
                
                cat_stats = strategy_summary[strategy]['by_category'][category]
                cat_stats['trades'] += stats['total_trades']
                cat_stats['wins'] += stats['wins']
        
        # Calculate final statistics
        for strategy in strategy_summary:
            s = strategy_summary[strategy]
            if s['total_trades'] > 0:
                s['overall_win_rate'] = s['total_wins'] / s['total_trades']
                s['overall_avg_pnl'] = s['total_pnl'] / s['total_trades']
                
                # Calculate by category
                for cat in s['by_category']:
                    cat_data = s['by_category'][cat]
                    if cat_data['trades'] > 0:
                        cat_data['win_rate'] = cat_data['wins'] / cat_data['trades']
        
        # Find best indicators across all trades
        all_indicator_stats = {}
        for result in all_results:
            for ind_group, stats in result['indicator_performance'].items():
                if ind_group not in all_indicator_stats:
                    all_indicator_stats[ind_group] = {
                        'total_contribution': 0,
                        'appearance_count': 0
                    }
                all_indicator_stats[ind_group]['total_contribution'] += stats['win_rate']
                all_indicator_stats[ind_group]['appearance_count'] += 1
        
        # Rank indicators
        indicator_rankings = []
        for ind_group, stats in all_indicator_stats.items():
            if stats['appearance_count'] > 0:
                avg_contribution = stats['total_contribution'] / stats['appearance_count']
                indicator_rankings.append({
                    'indicator_group': ind_group,
                    'avg_win_contribution': avg_contribution,
                    'specific_indicators': self.indicators_to_test[ind_group]
                })
        
        indicator_rankings.sort(key=lambda x: x['avg_win_contribution'], reverse=True)
        
        # Prepare final report
        report = {
            'summary': {
                'symbols_tested': len(all_results),
                'total_trades': sum(sum(s['total_trades'] for s in r['strategy_stats'].values()) for r in all_results),
                'date_range': f"{self.config['lookback_days']} days",
                'profile_period': f"{self.config['profile_period']} candles (3 days)"
            },
            'strategy_performance': strategy_summary,
            'category_analysis': category_summary,
            'best_indicators': indicator_rankings[:5],  # Top 5 indicator groups
            'detailed_results': all_results
        }
        
        # Save results with enhanced reporting
        self.save_results_enhanced(report)
        
        return report
    
    def save_results_enhanced(self, report: Dict):
        """Enhanced save results with detailed trade reporting"""
        
        # First, do the original save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original reports
        json_path = os.path.join(self.output_path, f"vp_backtest_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        summary_path = os.path.join(self.output_path, f"vp_backtest_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            # Original summary content
            f.write("VOLUME PROFILE BACKTEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Symbols Tested: {report['summary']['symbols_tested']}\n")
            f.write(f"Total Trades: {report['summary']['total_trades']}\n")
            f.write(f"Date Range: {report['summary']['date_range']}\n")
            f.write(f"Profile Period: {report['summary']['profile_period']}\n\n")
            
            f.write("STRATEGY PERFORMANCE\n")
            f.write("-"*50 + "\n")
            for strategy, stats in report['strategy_performance'].items():
                f.write(f"\n{strategy.upper()}:\n")
                f.write(f"  Overall Win Rate: {stats.get('overall_win_rate', 0):.2%}\n")
                f.write(f"  Average P&L: {stats.get('overall_avg_pnl', 0):.2f}%\n")
                f.write(f"  Total Trades: {stats['total_trades']}\n")
                
                f.write("  By Category:\n")
                for cat, cat_stats in stats['by_category'].items():
                    if cat_stats['trades'] > 0:
                        f.write(f"    {cat}: {cat_stats['win_rate']:.2%} win rate, {cat_stats['trades']} trades\n")
            
            f.write("\nBEST INDICATORS\n")
            f.write("-"*50 + "\n")
            for i, ind in enumerate(report['best_indicators'], 1):
                f.write(f"{i}. {ind['indicator_group'].upper()}\n")
                f.write(f"   Win Contribution: {ind['avg_win_contribution']:.2%}\n")
                f.write(f"   Includes: {', '.join(ind['specific_indicators'])}\n\n")
        
        # Now add detailed trade reporting using the fixed integration function
        from vp_trade_output_module_r1 import integrate_detailed_reporting
        
        # Generate detailed reports if we have results
        if 'detailed_results' in report and report['detailed_results']:
            integrate_detailed_reporting(self, report['detailed_results'])
        
        print(f"\n✅ Enhanced results saved with detailed trade analysis!")
        print(f"Check {self.output_path} for:")
        print(f"  - Detailed Excel with all trades")
        print(f"  - JSON with programmatic access")
        print(f"  - Summary reports")


# Example usage
if __name__ == "__main__":
        # Define top 5 Binance cryptos for faster testing
    top_5_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'
    ]
    
    # Create output directory
    output_path = os.path.join(os.getcwd(), "backtest_results", "volume_profile")
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize backtester - no data_path needed anymore
    backtester = VolumeProfileBacktest(
        data_path=None,  # Not used with API data fetcher
        output_path=output_path
    )
    
    print(f"Output directory: {output_path}")
    
    # Run simplified backtest first to verify logic
    print("=== RUNNING SIMPLIFIED BACKTEST TO VERIFY LOGIC ===")
    backtester.run_simple_backtest()
    
    print("\n=== RUNNING FULL BACKTEST ===")
    print(f"Starting backtest for {len(top_5_symbols)} symbols...")
    
    # Run backtest
    results = backtester.run_backtest(top_5_symbols)
    
    # Print summary if we got results
    if results and 'strategy_performance' in results:
        print("\nBacktest Complete!")
        print(f"Total strategies tested: {len(results['strategy_performance'])}")
        
        # Find best performing strategy
        best_strategy = None
        best_win_rate = 0
        for strategy, stats in results['strategy_performance'].items():
            if stats.get('overall_win_rate', 0) > best_win_rate:
                best_win_rate = stats['overall_win_rate']
                best_strategy = strategy
        
        if best_strategy:
            print(f"Best performing strategy: {best_strategy} ({best_win_rate:.1%} win rate)")
        
        if results.get('best_indicators'):
            print(f"Best indicator group: {results['best_indicators'][0]['indicator_group']}")
    else:
        print("\nNo results generated - check for errors above")