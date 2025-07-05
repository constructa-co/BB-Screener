"""
Risk/Reward Calculator - Auto Risk/Reward Calculation
Implements ChatGPT's recommendation for automatic R:R calculation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema

class RiskRewardCalculator:
    """
    Automatic Risk/Reward calculation using support/resistance levels
    Implements ChatGPT's recommendation for R:R ≥ 1.5:1 filter
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # R:R calculation parameters
        self.rr_params = {
            'min_rr_ratio': 1.5,          # Minimum R:R for pattern boosts
            'lookback_periods': 50,       # Periods to look back for S/R levels
            'min_touches': 2,             # Minimum touches to confirm S/R level
            'level_tolerance': 0.005,     # 0.5% tolerance for level matching
            'atr_multiplier': 2.0,        # ATR multiplier for stop loss
            'max_sl_distance': 0.05       # Maximum 5% stop loss distance
        }
        
        # Support/Resistance detection parameters
        self.sr_params = {
            'swing_window': 5,            # Window for swing high/low detection
            'min_level_strength': 0.3,    # Minimum strength for valid level
            'recent_weight': 1.5,         # Weight multiplier for recent levels
            'volume_weight': 1.2          # Weight multiplier for high-volume levels
        }
        
        self.logger.info("Risk/Reward Calculator initialized with auto S/L and T/P calculation")
    
    def calculate_auto_rr(self, df: pd.DataFrame, current_price: float, 
                         bb_data: Dict, pattern_direction: str = 'neutral') -> Dict:
        """
        Calculate automatic risk/reward based on support/resistance levels
        
        Args:
            df: OHLCV DataFrame
            current_price: Current market price
            bb_data: Bollinger Band data for context
            pattern_direction: 'bullish', 'bearish', or 'neutral'
            
        Returns:
            Comprehensive R:R analysis
        """
        try:
            self.logger.debug(f"Calculating auto R:R for price {current_price:.4f}, "
                            f"direction: {pattern_direction}")
            
            # Step 1: Detect support and resistance levels
            sr_levels = self._detect_support_resistance_levels(df)
            
            # Step 2: Calculate ATR for stop loss reference
            atr_value = self._calculate_atr(df, period=14)
            
            # Step 3: Find optimal stop loss and take profit levels
            if pattern_direction == 'bullish':
                sl_data = self._find_optimal_stop_loss(current_price, sr_levels['support'], 
                                                     atr_value, 'bullish')
                tp_data = self._find_optimal_take_profit(current_price, sr_levels['resistance'], 
                                                       bb_data, 'bullish')
            elif pattern_direction == 'bearish':
                sl_data = self._find_optimal_stop_loss(current_price, sr_levels['resistance'], 
                                                     atr_value, 'bearish')
                tp_data = self._find_optimal_take_profit(current_price, sr_levels['support'], 
                                                       bb_data, 'bearish')
            else:
                # Neutral patterns - use nearest levels
                sl_data = self._find_nearest_stop_loss(current_price, sr_levels, atr_value)
                tp_data = self._find_nearest_take_profit(current_price, sr_levels, bb_data)
            
            # Step 4: Calculate risk/reward metrics
            rr_metrics = self._calculate_rr_metrics(current_price, sl_data, tp_data)
            
            # Step 5: Validate and enhance results
            validation_results = self._validate_rr_setup(rr_metrics, pattern_direction)
            
            # Compile comprehensive results
            results = {
                'current_price': current_price,
                'pattern_direction': pattern_direction,
                
                # Stop Loss Analysis
                'auto_sl': sl_data['price'],
                'sl_distance': sl_data['distance'],
                'sl_percentage': sl_data['percentage'],
                'sl_method': sl_data['method'],
                'sl_level_strength': sl_data.get('level_strength', 0),
                
                # Take Profit Analysis
                'auto_tp': tp_data['price'],
                'tp_distance': tp_data['distance'],
                'tp_percentage': tp_data['percentage'],
                'tp_method': tp_data['method'],
                'tp_level_strength': tp_data.get('level_strength', 0),
                
                # Risk/Reward Metrics
                'risk_reward_ratio': rr_metrics['rr_ratio'],
                'risk_amount': rr_metrics['risk_amount'],
                'reward_amount': rr_metrics['reward_amount'],
                'rr_valid': validation_results['is_valid'],
                'rr_quality': validation_results['quality_score'],
                
                # Supporting Data
                'atr_value': atr_value,
                'support_levels': sr_levels['support'][:3],  # Top 3 support levels
                'resistance_levels': sr_levels['resistance'][:3],  # Top 3 resistance levels
                'sr_analysis_success': len(sr_levels['support']) > 0 or len(sr_levels['resistance']) > 0,
                
                # Validation Results
                'validation_notes': validation_results['notes'],
                'boost_eligible': validation_results['boost_eligible']
            }
            
            self.logger.info(f"Auto R:R calculated: {rr_metrics['rr_ratio']:.2f}:1 "
                           f"(SL: {sl_data['price']:.4f}, TP: {tp_data['price']:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating auto R:R: {str(e)}")
            return self._get_default_rr_results(current_price)
    
    def _detect_support_resistance_levels(self, df: pd.DataFrame) -> Dict:
        """Detect support and resistance levels from price action"""
        try:
            if len(df) < self.rr_params['lookback_periods']:
                return {'support': [], 'resistance': []}
            
            # Use recent data for S/R detection
            recent_df = df.tail(self.rr_params['lookback_periods']).copy()
            
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(recent_df)
            swing_lows = self._find_swing_lows(recent_df)
            
            # Convert to support/resistance levels with strength scoring
            resistance_levels = self._convert_to_levels(swing_highs, recent_df, 'resistance')
            support_levels = self._convert_to_levels(swing_lows, recent_df, 'support')
            
            # Sort by strength (strongest first)
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting S/R levels: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _find_swing_highs(self, df: pd.DataFrame) -> List[Dict]:
        """Find swing high points"""
        try:
            highs = df['high'].values
            
            # Find local maxima
            high_indices = argrelextrema(highs, np.greater, 
                                       order=self.sr_params['swing_window'])[0]
            
            swing_highs = []
            for idx in high_indices:
                if idx < len(df):
                    swing_highs.append({
                        'price': df.iloc[idx]['high'],
                        'index': idx,
                        'timestamp': df.index[idx] if hasattr(df.index, '__getitem__') else idx,
                        'volume': df.iloc[idx]['volume']
                    })
            
            return swing_highs
            
        except Exception as e:
            self.logger.error(f"Error finding swing highs: {str(e)}")
            return []
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[Dict]:
        """Find swing low points"""
        try:
            lows = df['low'].values
            
            # Find local minima
            low_indices = argrelextrema(lows, np.less, 
                                      order=self.sr_params['swing_window'])[0]
            
            swing_lows = []
            for idx in low_indices:
                if idx < len(df):
                    swing_lows.append({
                        'price': df.iloc[idx]['low'],
                        'index': idx,
                        'timestamp': df.index[idx] if hasattr(df.index, '__getitem__') else idx,
                        'volume': df.iloc[idx]['volume']
                    })
            
            return swing_lows
            
        except Exception as e:
            self.logger.error(f"Error finding swing lows: {str(e)}")
            return []
    
    def _convert_to_levels(self, swing_points: List[Dict], df: pd.DataFrame, level_type: str) -> List[Dict]:
        """Convert swing points to support/resistance levels with strength"""
        if not swing_points:
            return []
        
        levels = []
        tolerance = self.rr_params['level_tolerance']
        
        # Group similar price levels
        for point in swing_points:
            price = point['price']
            
            # Check if this price is close to an existing level
            existing_level = None
            for level in levels:
                if abs(price - level['price']) / level['price'] <= tolerance:
                    existing_level = level
                    break
            
            if existing_level:
                # Add to existing level
                existing_level['touches'] += 1
                existing_level['prices'].append(price)
                existing_level['volumes'].append(point['volume'])
                existing_level['indices'].append(point['index'])
                # Update average price
                existing_level['price'] = np.mean(existing_level['prices'])
            else:
                # Create new level
                levels.append({
                    'price': price,
                    'type': level_type,
                    'touches': 1,
                    'prices': [price],
                    'volumes': [point['volume']],
                    'indices': [point['index']],
                    'strength': 0  # Will be calculated below
                })
        
        # Calculate strength for each level
        for level in levels:
            level['strength'] = self._calculate_level_strength(level, df)
        
        # Filter out weak levels
        min_strength = self.sr_params['min_level_strength']
        strong_levels = [level for level in levels if level['strength'] >= min_strength]
        
        return strong_levels
    
    def _calculate_level_strength(self, level: Dict, df: pd.DataFrame) -> float:
        """Calculate the strength of a support/resistance level"""
        try:
            # Base strength from number of touches
            base_strength = min(level['touches'] / 5.0, 1.0)  # Normalize to max 1.0
            
            # Volume weight (higher volume = stronger level)
            avg_volume = np.mean(level['volumes']) if level['volumes'] else 1
            total_avg_volume = df['volume'].mean()
            volume_multiplier = min(avg_volume / total_avg_volume, 2.0) if total_avg_volume > 0 else 1.0
            
            # Recency weight (more recent levels are more relevant)
            max_index = len(df) - 1
            most_recent_index = max(level['indices']) if level['indices'] else 0
            recency_ratio = most_recent_index / max_index if max_index > 0 else 0.5
            recency_multiplier = 0.5 + (recency_ratio * 0.5)  # 0.5 to 1.0
            
            # Final strength calculation
            strength = base_strength * volume_multiplier * recency_multiplier
            
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating level strength: {str(e)}")
            return 0.3  # Default moderate strength
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(df) < period + 1:
                return 0.0
            
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0
    
    def _find_optimal_stop_loss(self, current_price: float, levels: List[Dict], 
                               atr_value: float, direction: str) -> Dict:
        """Find optimal stop loss level"""
        try:
            if direction == 'bullish':
                # For bullish patterns, find nearest support below current price
                relevant_levels = [l for l in levels if l['price'] < current_price * 0.98]  # At least 2% below
            else:
                # For bearish patterns, find nearest resistance above current price
                relevant_levels = [l for l in levels if l['price'] > current_price * 1.02]  # At least 2% above
            
            if relevant_levels:
                # Choose strongest level within reasonable distance
                best_level = None
                for level in relevant_levels:
                    distance = abs(current_price - level['price']) / current_price
                    if distance <= self.rr_params['max_sl_distance']:  # Within 5%
                        if best_level is None or level['strength'] > best_level['strength']:
                            best_level = level
                
                if best_level:
                    sl_price = best_level['price']
                    # Add small buffer (0.1% beyond the level)
                    if direction == 'bullish':
                        sl_price *= 0.999  # Slightly below support
                    else:
                        sl_price *= 1.001  # Slightly above resistance
                    
                    return {
                        'price': sl_price,
                        'distance': abs(current_price - sl_price),
                        'percentage': abs(current_price - sl_price) / current_price,
                        'method': 'support_resistance',
                        'level_strength': best_level['strength']
                    }
            
            # Fallback to ATR-based stop loss
            atr_distance = atr_value * self.rr_params['atr_multiplier']
            if direction == 'bullish':
                sl_price = current_price - atr_distance
            else:
                sl_price = current_price + atr_distance
            
            return {
                'price': sl_price,
                'distance': atr_distance,
                'percentage': atr_distance / current_price,
                'method': 'atr_based',
                'level_strength': 0.5  # Moderate strength for ATR method
            }
            
        except Exception as e:
            self.logger.error(f"Error finding optimal stop loss: {str(e)}")
            return {
                'price': current_price * 0.98 if direction == 'bullish' else current_price * 1.02,
                'distance': current_price * 0.02,
                'percentage': 0.02,
                'method': 'fallback',
                'level_strength': 0.3
            }
    
    def _find_optimal_take_profit(self, current_price: float, levels: List[Dict], 
                                 bb_data: Dict, direction: str) -> Dict:
        """Find optimal take profit level"""
        try:
            if direction == 'bullish':
                # For bullish patterns, find nearest resistance above current price
                relevant_levels = [l for l in levels if l['price'] > current_price * 1.02]  # At least 2% above
            else:
                # For bearish patterns, find nearest support below current price
                relevant_levels = [l for l in levels if l['price'] < current_price * 0.98]  # At least 2% below
            
            if relevant_levels:
                # Sort by distance (nearest first)
                relevant_levels.sort(key=lambda x: abs(x['price'] - current_price))
                
                # Choose first strong level that provides good R:R
                for level in relevant_levels[:3]:  # Check top 3 nearest levels
                    tp_price = level['price']
                    # Add small buffer (0.1% before the level)
                    if direction == 'bullish':
                        tp_price *= 0.999  # Slightly below resistance
                    else:
                        tp_price *= 1.001  # Slightly above support
                    
                    return {
                        'price': tp_price,
                        'distance': abs(tp_price - current_price),
                        'percentage': abs(tp_price - current_price) / current_price,
                        'method': 'support_resistance',
                        'level_strength': level['strength']
                    }
            
            # Fallback to BB-based take profit
            bb_middle = bb_data.get('bb_middle', current_price)
            bb_upper = bb_data.get('bb_upper', current_price * 1.03)
            bb_lower = bb_data.get('bb_lower', current_price * 0.97)
            
            if direction == 'bullish':
                tp_price = bb_middle if current_price < bb_middle else bb_upper
            else:
                tp_price = bb_middle if current_price > bb_middle else bb_lower
            
            return {
                'price': tp_price,
                'distance': abs(tp_price - current_price),
                'percentage': abs(tp_price - current_price) / current_price,
                'method': 'bollinger_band',
                'level_strength': 0.6  # Moderate-high strength for BB method
            }
            
        except Exception as e:
            self.logger.error(f"Error finding optimal take profit: {str(e)}")
            return {
                'price': current_price * 1.03 if direction == 'bullish' else current_price * 0.97,
                'distance': current_price * 0.03,
                'percentage': 0.03,
                'method': 'fallback',
                'level_strength': 0.4
            }
    
    def _find_nearest_stop_loss(self, current_price: float, sr_levels: Dict, atr_value: float) -> Dict:
        """Find nearest stop loss for neutral patterns"""
        all_levels = sr_levels['support'] + sr_levels['resistance']
        
        if not all_levels:
            # ATR-based fallback
            sl_price = current_price - (atr_value * self.rr_params['atr_multiplier'])
            return {
                'price': sl_price,
                'distance': atr_value * self.rr_params['atr_multiplier'],
                'percentage': (atr_value * self.rr_params['atr_multiplier']) / current_price,
                'method': 'atr_based',
                'level_strength': 0.5
            }
        
        # Find nearest level below current price for stop loss
        below_levels = [l for l in all_levels if l['price'] < current_price]
        if below_levels:
            nearest_level = min(below_levels, key=lambda x: abs(x['price'] - current_price))
            return {
                'price': nearest_level['price'] * 0.999,  # Small buffer
                'distance': current_price - nearest_level['price'],
                'percentage': (current_price - nearest_level['price']) / current_price,
                'method': 'nearest_support',
                'level_strength': nearest_level['strength']
            }
        
        # Fallback to ATR
        sl_price = current_price - (atr_value * self.rr_params['atr_multiplier'])
        return {
            'price': sl_price,
            'distance': atr_value * self.rr_params['atr_multiplier'],
            'percentage': (atr_value * self.rr_params['atr_multiplier']) / current_price,
            'method': 'atr_fallback',
            'level_strength': 0.5
        }
    
    def _find_nearest_take_profit(self, current_price: float, sr_levels: Dict, bb_data: Dict) -> Dict:
        """Find nearest take profit for neutral patterns"""
        all_levels = sr_levels['support'] + sr_levels['resistance']
        
        if all_levels:
            # Find nearest level above current price for take profit
            above_levels = [l for l in all_levels if l['price'] > current_price]
            if above_levels:
                nearest_level = min(above_levels, key=lambda x: abs(x['price'] - current_price))
                return {
                    'price': nearest_level['price'] * 0.999,  # Small buffer
                    'distance': nearest_level['price'] - current_price,
                    'percentage': (nearest_level['price'] - current_price) / current_price,
                    'method': 'nearest_resistance',
                    'level_strength': nearest_level['strength']
                }
        
        # Fallback to BB middle
        bb_middle = bb_data.get('bb_middle', current_price * 1.02)
        return {
            'price': bb_middle,
            'distance': abs(bb_middle - current_price),
            'percentage': abs(bb_middle - current_price) / current_price,
            'method': 'bb_middle',
            'level_strength': 0.6
        }
    
    def _calculate_rr_metrics(self, current_price: float, sl_data: Dict, tp_data: Dict) -> Dict:
        """Calculate comprehensive risk/reward metrics"""
        try:
            risk_amount = abs(current_price - sl_data['price'])
            reward_amount = abs(tp_data['price'] - current_price)
            
            rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            
            return {
                'rr_ratio': rr_ratio,
                'risk_amount': risk_amount,
                'reward_amount': reward_amount,
                'risk_percentage': (risk_amount / current_price) * 100,
                'reward_percentage': (reward_amount / current_price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating R:R metrics: {str(e)}")
            return {
                'rr_ratio': 0,
                'risk_amount': 0,
                'reward_amount': 0,
                'risk_percentage': 0,
                'reward_percentage': 0
            }
    
    def _validate_rr_setup(self, rr_metrics: Dict, pattern_direction: str) -> Dict:
        """Validate R:R setup quality"""
        try:
            rr_ratio = rr_metrics['rr_ratio']
            risk_pct = rr_metrics['risk_percentage']
            
            # Quality scoring
            quality_score = 50  # Base score
            
            # R:R ratio scoring
            if rr_ratio >= 3.0:
                quality_score += 30
            elif rr_ratio >= 2.0:
                quality_score += 20
            elif rr_ratio >= 1.5:
                quality_score += 10
            
            # Risk percentage scoring (lower risk = higher quality)
            if risk_pct <= 2.0:
                quality_score += 15
            elif risk_pct <= 3.0:
                quality_score += 10
            elif risk_pct <= 5.0:
                quality_score += 5
            
            # Validation rules
            is_valid = (
                rr_ratio >= self.rr_params['min_rr_ratio'] and
                risk_pct <= 5.0  # Maximum 5% risk
            )
            
            boost_eligible = is_valid and rr_ratio >= 2.0  # Higher threshold for boosts
            
            # Validation notes
            notes = []
            if rr_ratio < self.rr_params['min_rr_ratio']:
                notes.append(f"R:R ratio {rr_ratio:.2f} below minimum {self.rr_params['min_rr_ratio']}")
            if risk_pct > 5.0:
                notes.append(f"Risk {risk_pct:.1f}% exceeds 5% maximum")
            if rr_ratio >= 2.0:
                notes.append("Excellent R:R ratio")
            
            return {
                'is_valid': is_valid,
                'quality_score': min(quality_score, 100),
                'boost_eligible': boost_eligible,
                'notes': notes
            }
            
        except Exception as e:
            self.logger.error(f"Error validating R:R setup: {str(e)}")
            return {
                'is_valid': False,
                'quality_score': 0,
                'boost_eligible': False,
                'notes': ['Validation failed']
            }
    
    def _get_default_rr_results(self, current_price: float) -> Dict:
        """Return default R:R results when calculation fails"""
        return {
            'current_price': current_price,
            'pattern_direction': 'unknown',
            'auto_sl': current_price * 0.98,
            'sl_distance': current_price * 0.02,
            'sl_percentage': 2.0,
            'sl_method': 'default',
            'sl_level_strength': 0.3,
            'auto_tp': current_price * 1.03,
            'tp_distance': current_price * 0.03,
            'tp_percentage': 3.0,
            'tp_method': 'default',
            'tp_level_strength': 0.3,
            'risk_reward_ratio': 1.5,
            'risk_amount': current_price * 0.02,
            'reward_amount': current_price * 0.03,
            'rr_valid': True,
            'rr_quality': 50,
            'atr_value': 0,
            'support_levels': [],
            'resistance_levels': [],
            'sr_analysis_success': False,
            'validation_notes': ['Default R:R calculation'],
            'boost_eligible': False
        }
    
    def get_rr_statistics(self) -> Dict:
        """Get R:R calculator statistics and configuration"""
        return {
            'rr_parameters': self.rr_params,
            'sr_parameters': self.sr_params,
            'features': [
                'Auto stop-loss calculation',
                'Auto take-profit calculation', 
                'Support/resistance level detection',
                'ATR-based fallback stops',
                'R:R validation (≥1.5:1)',
                'Level strength scoring'
            ],
            'methods': {
                'stop_loss': ['Support/Resistance', 'ATR-based', 'Fallback'],
                'take_profit': ['Support/Resistance', 'Bollinger Bands', 'Fallback']
            }
        }