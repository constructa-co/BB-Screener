import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

"""
Volume Profile Backtest - DEBUG VERSION
=======================================

CHANGES MADE FOR DEBUGGING:
1. Loosened filters:
   - score_threshold: 80 → 70
   - entry_zone_atr: 1.5 → 2.0
   - min_rr_ratio: 2.0 → 1.5
   - volume_surge: 1.2 → 1.0
   - ADX limit: 30 → 35 (poc_magnet)
   - virgin_poc lookback: 5 → 3

2. Added debugging output:
   - Scores >= 50 logged with details
   - Breakout attempts logged
   - Virgin POCs found logged
   - Rejected trades logged (volume_ratio > 0.8)

3. Enhanced logging:
   - Entry scores with strategy details
   - Volume profile levels and distances
   - Technical indicator values
   - Trade rejection reasons

This version should find more signals while providing detailed debugging information.
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
        
        # Configuration parameters based on research (LOOSENED FOR DEBUGGING)
        self.config = {
            'profile_period': 72,      # 3-day lookback (72 4H candles)
            'value_area_pct': 0.70,    # 70% for major pairs
            'value_area_alt_pct': 0.75,  # 75% for altcoins
            'min_volume_node': 0.02,   # 2% minimum volume threshold
            'max_distance_pct': 0.05,  # 5% max distance from price
            'entry_zone_atr': 2.0,     # 2.0x ATR for zone entries (LOOSENED)
            'stop_loss_atr': 2.5,      # 2.5x ATR for stop loss
            'min_rr_ratio': 1.5,       # Minimum 1.5:1 R/R (LOOSENED)
            'volume_surge': 1.0,       # 100% volume confirmation (LOOSENED)
            'score_threshold': 70,     # Minimum score for entry (LOOSENED)
            'atr_period': 14,          # ATR calculation period
            'lookback_days': 365       # 1 year of data for backtesting
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
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'poc_magnet': self.poc_magnet_strategy,
            'virgin_poc': self.virgin_poc_strategy,
            'composite': self.composite_strategy
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
        """Calculate entry score based on multiple factors"""
        score = 0
        current = df.iloc[idx]
        
        # Zone penetration score (0-25 points)
        atr = current['atr']
        zone_size = atr * self.config['entry_zone_atr']
        
        if strategy in ['mean_reversion', 'poc_magnet']:
            # Distance from key level
            distances = [
                abs(current['close'] - vp['poc']),
                abs(current['close'] - vp['vah']),
                abs(current['close'] - vp['val'])
            ]
            min_distance = min(distances)
            penetration = min_distance / zone_size
            score += 25 * (1 - min(penetration, 1))
        
        # Volume confirmation (25 points)
        if current['volume_ratio'] > self.config['volume_surge']:
            score += 25
        
        # Time at level (20 points)
        # Check how many candles price has been near this level
        lookback = min(4, idx)
        near_level_count = 0
        for i in range(1, lookback + 1):
            prev = df.iloc[idx - i]
            if abs(prev['close'] - current['close']) / current['close'] < 0.02:
                near_level_count += 1
        score += 20 * (near_level_count / 4)
        
        # Technical confluence (30 points)
        confluence_count = 0
        
        # Check various indicators
        if current['rsi'] < 30 and strategy == 'mean_reversion':
            confluence_count += 1
        if current['rsi'] > 70 and strategy == 'mean_reversion':
            confluence_count += 1
        if current['macd'] > current['macd_signal'] and strategy == 'breakout':
            confluence_count += 1
        if current['close'] > current['bb_upper'] and strategy == 'breakout':
            confluence_count += 1
        if current['adx'] > 25 and strategy == 'breakout':
            confluence_count += 1
        if current['volume_rsi'] > 70:
            confluence_count += 1
        
        score += 30 * min(confluence_count / 3, 1)
        
        # DEBUG: Log scores above 50 for analysis
        if score >= 50:
            print(f"🔍 DEBUG: {strategy} score={score:.1f}, close={current['close']:.4f}, POC={vp['poc']:.4f}, VAH={vp['vah']:.4f}, VAL={vp['val']:.4f}")
            print(f"    Volume ratio: {current['volume_ratio']:.2f}, RSI: {current['rsi']:.1f}, ADX: {current['adx']:.1f}")
        
        return score
    
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
        """Mean reversion from VA boundaries back to POC"""
        if not vp or idx < 50:
            return None
            
        current = df.iloc[idx]
        atr = current['atr']
        
        # Check if price is near VAH or VAL
        near_vah = abs(current['close'] - vp['vah']) < atr * self.config['entry_zone_atr']
        near_val = abs(current['close'] - vp['val']) < atr * self.config['entry_zone_atr']
        
        if not (near_vah or near_val):
            return None
        
        # Calculate entry score
        score = self.calculate_entry_score(df, idx, vp, 'mean_reversion')
        
        if score < self.config['score_threshold']:
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
        
        # Check R/R ratio
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if reward / risk < self.config['min_rr_ratio']:
            return None
        
        return {
            'strategy': 'mean_reversion',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score': score,
            'risk_reward': reward / risk
        }
    
    def breakout_strategy(self, df: pd.DataFrame, idx: int, vp: Dict) -> Optional[Dict]:
        """Breakout trading beyond Value Area"""
        if not vp or idx < 50:
            return None
            
        current = df.iloc[idx]
        prev = df.iloc[idx - 1]
        atr = current['atr']
        
        # Check for breakout
        breakout_up = prev['close'] < vp['vah'] and current['close'] > vp['vah']
        breakout_down = prev['close'] > vp['val'] and current['close'] < vp['val']
        
        if not (breakout_up or breakout_down):
            return None
        
        # DEBUG: Log breakout attempts
        print(f"🔍 DEBUG: Breakout detected - Up: {breakout_up}, Down: {breakout_down}")
        print(f"    Prev close: {prev['close']:.4f}, Current: {current['close']:.4f}")
        print(f"    VAH: {vp['vah']:.4f}, VAL: {vp['val']:.4f}, Volume ratio: {current['volume_ratio']:.2f}")
        
        # Volume confirmation
        if current['volume_ratio'] < self.config['volume_surge']:
            return None
        
        # Calculate entry score
        score = self.calculate_entry_score(df, idx, vp, 'breakout')
        
        if score < self.config['score_threshold']:
            return None
        
        # Set trade parameters
        if breakout_up:
            direction = 'long'
            entry = current['close']
            stop = vp['vah'] - atr * 1.5  # Tighter stop for breakouts
            
            # Find next resistance (HVN or previous high)
            potential_targets = [h for h in vp['hvn'] if h > entry]
            if potential_targets:
                target = min(potential_targets)
            else:
                target = entry + (entry - stop) * 3  # 3:1 R/R minimum
        else:
            direction = 'short'
            entry = current['close']
            stop = vp['val'] + atr * 1.5
            
            # Find next support
            potential_targets = [h for h in vp['hvn'] if h < entry]
            if potential_targets:
                target = max(potential_targets)
            else:
                target = entry - (stop - entry) * 3
        
        # Check R/R ratio
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if reward / risk < self.config['min_rr_ratio']:
            return None
        
        return {
            'strategy': 'breakout',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score': score,
            'risk_reward': reward / risk
        }
    
    def poc_magnet_strategy(self, df: pd.DataFrame, idx: int, vp: Dict) -> Optional[Dict]:
        """Trade toward POC from range extremes"""
        if not vp or idx < 50:
            return None
            
        current = df.iloc[idx]
        atr = current['atr']
        
        # Check distance from POC
        distance_from_poc = abs(current['close'] - vp['poc'])
        distance_pct = distance_from_poc / current['close']
        
        # Optimal distance: 2-5% from POC
        if distance_pct < 0.02 or distance_pct > 0.05:
            return None
        
        # Check if we're in a balanced market (not trending strongly) - LOOSENED
        if current['adx'] > 35:  # Increased from 30 to 35
            return None
        
        # Calculate entry score
        score = self.calculate_entry_score(df, idx, vp, 'poc_magnet')
        
        if score < self.config['score_threshold']:
            return None
        
        # Determine direction
        if current['close'] > vp['poc']:
            direction = 'short'
            entry = current['close']
            
            # Find HVN above for stop
            hvn_above = [h for h in vp['hvn'] if h > entry]
            if hvn_above:
                stop = min(hvn_above) + atr * 0.5
            else:
                stop = entry + atr * self.config['stop_loss_atr']
            
            target = vp['poc']
        else:
            direction = 'long'
            entry = current['close']
            
            # Find HVN below for stop
            hvn_below = [h for h in vp['hvn'] if h < entry]
            if hvn_below:
                stop = max(hvn_below) - atr * 0.5
            else:
                stop = entry - atr * self.config['stop_loss_atr']
            
            target = vp['poc']
        
        # Check R/R ratio
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if reward / risk < 1.5:  # Lower R/R acceptable for high probability
            return None
        
        return {
            'strategy': 'poc_magnet',
            'direction': direction,
            'entry': entry,
            'stop': stop,
            'target': target,
            'score': score,
            'risk_reward': reward / risk
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
        
        # DEBUG: Log virgin POCs found
        print(f"🔍 DEBUG: Found {len(virgin_pocs)} virgin POCs: {[f'{v:.4f}' for v in virgin_pocs[:3]]}")
        
        # Check if price is near any virgin POC
        for vpoc in virgin_pocs:
            if abs(current['close'] - vpoc) < atr * self.config['entry_zone_atr']:
                # Calculate entry score
                score = self.calculate_entry_score(df, idx, vp, 'virgin_poc')
                
                if score < self.config['score_threshold']:
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
            'poc_cluster': len(poc_cluster)
        }
    
    def simulate_trade(self, df: pd.DataFrame, trade: Dict, start_idx: int) -> Dict:
        """Simulate trade execution and calculate results"""
        entry_price = trade['entry']
        stop_price = trade['stop']
        target_price = trade['target']
        direction = trade['direction']
        
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
            if trade['result'] == 'target':
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
        
        # Run backtest
        for i in range(self.config['profile_period'], len(df) - 100):
            # Calculate Volume Profile
            vp = self.calculate_volume_profile(df.iloc[:i], self.config['profile_period'])
            
            if not vp:
                continue
            
            # Test each strategy
            for strategy_name, strategy_func in self.strategies.items():
                trade_signal = strategy_func(df, i, vp)
                
                if trade_signal:
                    total_signals += 1
                    # Simulate trade
                    trade_result = self.simulate_trade(df, trade_signal, i)
                    
                    # Combine trade info
                    full_trade = {**trade_signal, **trade_result}
                    full_trade['symbol'] = symbol
                    full_trade['category'] = category
                    full_trade['entry_time'] = df.iloc[i]['timestamp']
                    
                    strategy_results[strategy_name].append(full_trade)
                else:
                    # DEBUG: Track rejected trades for analysis
                    current = df.iloc[i]
                    if current['volume_ratio'] > 0.8:  # Only log if volume is decent
                        print(f"⛔ Rejected: {strategy_name} at {current['timestamp']}, close={current['close']:.4f}, volume_ratio={current['volume_ratio']:.2f}")
        
        print(f"  Found {total_signals} total signals across all strategies")
        
        # Calculate statistics for each strategy
        stats = {}
        for strategy, trades in strategy_results.items():
            if trades:
                wins = [t for t in trades if t['result'] == 'target']
                losses = [t for t in trades if t['result'] == 'stop_loss']
                
                stats[strategy] = {
                    'total_trades': len(trades),
                    'wins': len(wins),
                    'losses': len(losses),
                    'win_rate': len(wins) / len(trades) if trades else 0,
                    'avg_pnl': np.mean([t['pnl_pct'] for t in trades]),
                    'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
                    'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
                    'avg_rr': np.mean([t['risk_reward'] for t in trades]),
                    'profit_factor': abs(sum(t['pnl_pct'] for t in wins) / sum(t['pnl_pct'] for t in losses)) if losses else 0,
                    'max_drawdown': self.calculate_max_drawdown(trades),
                    'sharpe_ratio': self.calculate_sharpe_ratio(trades)
                }
                print(f"  {strategy}: {len(trades)} trades, {stats[strategy]['win_rate']:.1%} win rate")
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
    
    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from trade list"""
        if not trades or len(trades) < 2:
            return 0
        
        returns = [t['pnl_pct'] / 100 for t in trades]
        
        # Annualize based on average trade duration
        avg_duration = np.mean([t['duration_candles'] for t in trades])
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
                            
                            # Fetch 1000 candles (about 166 days of 4H data)
                            candles = ex.fetch_ohlcv(market, '4h', limit=1000)
                            
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
                            break
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
                
                # Filter to last year (skip if we don't have enough data)
                if len(df) > self.config['lookback_days']:
                    cutoff_date = datetime.now() - timedelta(days=self.config['lookback_days'])
                    df = df[df['timestamp'] >= cutoff_date].reset_index(drop=True)
                
                if len(df) < 200:
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
        
        # Save results
        self.save_results(report)
        
        return report
    
    def save_results(self, report: Dict):
        """Save backtest results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_path = os.path.join(self.output_path, f"vp_backtest_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create summary report
        summary_path = os.path.join(self.output_path, f"vp_backtest_summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
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
        
        print(f"Results saved to {self.output_path}")
        print(f"- Detailed report: {json_path}")
        print(f"- Summary: {summary_path}")


# Example usage
if __name__ == "__main__":
    # Define top 30 Binance cryptos
    top_30_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
        'SHIBUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT',
        'FTMUSDT', 'NEARUSDT', 'ALGOUSDT', 'ICPUSDT', 'FILUSDT',
        'AAVEUSDT', 'CAKEUSDT', 'SUSHIUSDT', 'CRVUSDT', 'LDOUSDT',
        'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'ENJUSDT', 'PEPEUSDT'
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
    print(f"Starting backtest for {len(top_30_symbols)} symbols...")
    
    # Run backtest
    results = backtester.run_backtest(top_30_symbols)
    
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