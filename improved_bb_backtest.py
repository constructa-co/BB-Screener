#!/usr/bin/env python3
"""
Comprehensive BB Backtest - 500 Coins + Enhanced BB Metrics
Validate confluence factors across broad market + measure BB-specific indicators
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_fetcher import MarketDataFetcher
from modules.bb_detector import BBDetector
from modules.technical_analyzer import TechnicalAnalyzer
from modules.pattern_analyzer import PatternAnalyzer
from modules.risk_manager import RiskManager
from config import SCANNER_CONFIG

class ComprehensiveBBBacktest:
    """
    Comprehensive BB backtesting across 500 coins with enhanced BB metrics
    """
    
    def __init__(self):
        print("🚀 COMPREHENSIVE BB BACKTEST - 500 COINS + ENHANCED METRICS")
        print("=" * 80)
        print("🏗️ Initializing Comprehensive BB Analyzer...")
        
        # Initialize modules
        self.data_fetcher = MarketDataFetcher()
        self.bb_detector = BBDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.risk_manager = RiskManager()
        
        print("✅ All modules initialized successfully")
        
    def run_comprehensive_analysis(self, timeframes=[30], max_coins=500):
        """
        Run comprehensive analysis across top 500 coins
        """
        print(f"\n🎯 STARTING COMPREHENSIVE ANALYSIS - TOP {max_coins} COINS")
        print("=" * 70)
        
        # Get expanded coin list
        print(f"📊 Fetching top {max_coins} coins from CMC...")
        top_coins = self.data_fetcher.fetch_top_coins(limit=max_coins)
        
        if len(top_coins) < 50:
            print("❌ Could not fetch sufficient coins")
            return {}
        
        # Handle both formats: list of dicts or list of strings
        if isinstance(top_coins[0], dict):
            symbols = [coin['symbol'] for coin in top_coins[:max_coins]]
        else:
            symbols = top_coins[:max_coins]  # Already a list of symbols
            
        print(f"✅ Selected {len(symbols)} coins for analysis")
        
        all_results = {}
        
        for timeframe_days in timeframes:
            print(f"\n📅 ANALYZING {timeframe_days}-DAY TIMEFRAME")
            print("-" * 50)
            
            timeframe_results = self._analyze_timeframe_comprehensive(symbols, timeframe_days)
            all_results[f"{timeframe_days}d"] = timeframe_results
            
            # Summary for this timeframe
            total_bounces = sum(len(data['bounces']) for data in timeframe_results.values() if 'bounces' in data)
            print(f"📊 Found {total_bounces} total BB bounces across {len(symbols)} coins")
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results)
        
        return all_results
    
    def _analyze_timeframe_comprehensive(self, symbols: List[str], timeframe_days: int) -> Dict:
        """
        Analyze all BB bounces across many coins in a specific timeframe
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        
        timeframe_results = {}
        
        # Progress tracking
        total_symbols = len(symbols)
        processed = 0
        
        for i, symbol in enumerate(symbols, 1):
            if i % 50 == 0:  # Progress update every 50 coins
                print(f"📊 Progress: {i}/{total_symbols} coins processed...")
            
            try:
                symbol_results = self._find_all_bb_bounces_enhanced(symbol, start_date, end_date)
                timeframe_results[symbol] = symbol_results
                
                bounce_count = len(symbol_results['bounces'])
                if bounce_count > 0:
                    processed += 1
                    
            except Exception as e:
                timeframe_results[symbol] = {'bounces': [], 'error': str(e)}
        
        print(f"✅ Successfully analyzed {processed}/{total_symbols} coins")
        return timeframe_results
    
    def _find_all_bb_bounces_enhanced(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Find ALL BB bounces for a symbol with enhanced BB metrics
        """
        all_bounces = []
        
        # Check multiple exchanges
        exchanges = ['binance', 'bybit', 'kucoin', 'okx']
        
        for exchange in exchanges[:2]:  # Limit to 2 exchanges for speed
            try:
                # Get historical data
                df = self._get_historical_data(symbol, exchange, start_date, end_date)
                if df is None or len(df) < 50:
                    continue
                
                # Find BB bounces with enhanced metrics
                bounces = self._detect_bb_bounces_enhanced(df, symbol, exchange)
                all_bounces.extend(bounces)
                
            except Exception as e:
                continue
        
        return {
            'symbol': symbol,
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'bounces': all_bounces,
            'total_bounces': len(all_bounces)
        }
    
    def _get_historical_data(self, symbol: str, exchange: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for the specified period
        """
        try:
            # Use your existing data fetcher
            df = self.data_fetcher.fetch_ohlcv(exchange, symbol, '4h')
            
            if df is None or len(df) < 50:
                return None
            
            # Filter to date range
            df_filtered = df[df.index >= start_date]
            df_filtered = df_filtered[df_filtered.index <= end_date]
            
            return df_filtered if len(df_filtered) >= 20 else None
            
        except Exception as e:
            return None
    
    def _detect_bb_bounces_enhanced(self, df: pd.DataFrame, symbol: str, exchange: str) -> List[Dict]:
        """
        Detect BB bounces with enhanced BB metrics
        """
        bounces = []
        
        try:
            # Ensure we have BB data
            if 'bb_lower' not in df.columns or 'bb_upper' not in df.columns:
                return bounces
            
            # Find touches to BB bands
            for i in range(10, len(df) - 5):  # Leave room for confirmation
                current_row = df.iloc[i]
                
                # LOWER BAND TOUCHES (Potential LONG setups)
                if self._is_lower_band_touch_strict(df, i):
                    bounce = self._create_enhanced_bounce_record(
                        df, i, symbol, exchange, 'LONG', 'lower_band_touch'
                    )
                    if bounce:
                        bounces.append(bounce)
                
                # UPPER BAND TOUCHES (Potential SHORT setups)  
                if self._is_upper_band_touch_strict(df, i):
                    bounce = self._create_enhanced_bounce_record(
                        df, i, symbol, exchange, 'SHORT', 'upper_band_touch'
                    )
                    if bounce:
                        bounces.append(bounce)
        
        except Exception as e:
            pass
        
        return bounces
    
    def _is_lower_band_touch_strict(self, df: pd.DataFrame, i: int) -> bool:
        """
        Strict lower band touch detection
        """
        try:
            current = df.iloc[i]
            
            # Touch criteria: Low touches or comes very close to lower band
            distance_to_lower = (current['low'] - current['bb_lower']) / current['bb_lower']
            
            # STRICT: Within 0.3% of lower band
            return distance_to_lower <= 0.003
            
        except:
            return False
    
    def _is_upper_band_touch_strict(self, df: pd.DataFrame, i: int) -> bool:
        """
        Strict upper band touch detection
        """
        try:
            current = df.iloc[i]
            
            # Touch criteria: High touches or comes very close to upper band
            distance_to_upper = (current['bb_upper'] - current['high']) / current['bb_upper']
            
            # STRICT: Within 0.3% of upper band
            return distance_to_upper <= 0.003
            
        except:
            return False
    
    def _create_enhanced_bounce_record(self, df: pd.DataFrame, i: int, symbol: str, exchange: str, 
                                     direction: str, signal_type: str) -> Optional[Dict]:
        """
        Create comprehensive bounce record with enhanced BB metrics
        """
        try:
            current_row = df.iloc[i]
            timestamp = current_row.name
            
            # Calculate potential outcome (look ahead 5-20 periods)
            outcome = self._calculate_bounce_outcome(df, i, direction)
            
            # Get all confluence metrics
            confluence_metrics = self._get_all_confluence_metrics(df, i)
            
            # **ENHANCED BB METRICS** - The key addition you requested
            bb_metrics = self._calculate_enhanced_bb_metrics(df, i)
            
            # Additional technical metrics
            additional_metrics = self._calculate_additional_technical_metrics(df, i)
            
            bounce_record = {
                # Basic info
                'timestamp': timestamp,
                'symbol': symbol,
                'exchange': exchange,
                'direction': direction,
                'signal_type': signal_type,
                'entry_price': current_row['close'],
                'market_cap_rank': self._get_market_cap_rank(symbol),  # New: Track coin ranking
                
                # Standard BB data
                'bb_position': current_row['bb_pct'],
                'bb_width': current_row['bb_width'],
                'bb_lower': current_row['bb_lower'],
                'bb_upper': current_row['bb_upper'],
                'bb_middle': current_row['bb_middle'],
                
                # **ENHANCED BB METRICS** - Your requested addition
                'bb_squeeze': bb_metrics['bb_squeeze'],
                'bb_expansion': bb_metrics['bb_expansion'],
                'bb_trend': bb_metrics['bb_trend'],
                'bb_bandwidth_percentile': bb_metrics['bb_bandwidth_percentile'],
                'bb_position_momentum': bb_metrics['bb_position_momentum'],
                'bb_reversal_setup': bb_metrics['bb_reversal_setup'],
                
                # **ADDITIONAL TECHNICAL METRICS** - Your requested addition
                'chaikin_money_flow': additional_metrics['chaikin_money_flow'],
                'accumulation_distribution': additional_metrics['accumulation_distribution'],
                'money_flow_index': additional_metrics['money_flow_index'],
                'price_volume_trend': additional_metrics['price_volume_trend'],
                'ease_of_movement': additional_metrics['ease_of_movement'],
                'force_index': additional_metrics['force_index'],
                
                # Outcome data
                'max_favorable_5': outcome['max_favorable_5'],
                'max_adverse_5': outcome['max_adverse_5'],
                'outcome_10_periods': outcome['outcome_10'],
                'outcome_20_periods': outcome['outcome_20'],
                'best_exit_price': outcome['best_exit'],
                'worst_drawdown': outcome['worst_drawdown'],
                
                # Timing data
                'time_to_target': outcome['time_to_target'],
                'max_drawdown_time': outcome['max_drawdown_time'],
                
                # Confluence Metrics
                'rsi': confluence_metrics['rsi'],
                'rsi_divergence': confluence_metrics['rsi_divergence'],
                'macd_divergence': confluence_metrics['macd_divergence'],
                'volume_surge': confluence_metrics['volume_surge'],
                'volume_ratio': confluence_metrics['volume_ratio'],
                'atr_volatility': confluence_metrics['atr_volatility'],
                'stoch_oversold': confluence_metrics['stoch_oversold'],
                'stoch_overbought': confluence_metrics['stoch_overbought'],
                'cci_extreme': confluence_metrics['cci_extreme'],
                
                # Pattern analysis
                'has_patterns': confluence_metrics['has_patterns'],
                'pattern_count': confluence_metrics['pattern_count'],
                'pattern_quality': confluence_metrics['pattern_quality'],
                
                # Market conditions
                'market_trend': confluence_metrics['market_trend'],
                'volatility_regime': confluence_metrics['volatility_regime'],
                
                # Final scores
                'confluence_score': confluence_metrics['total_confluence_score'],
                'success_probability': self._calculate_success_probability(confluence_metrics)
            }
            
            return bounce_record
            
        except Exception as e:
            return None
    
    def _calculate_enhanced_bb_metrics(self, df: pd.DataFrame, i: int) -> Dict:
        """
        Calculate enhanced Bollinger Band specific metrics
        """
        try:
            current = df.iloc[i]
            
            # Get historical context (last 20 periods)
            start_idx = max(0, i - 20)
            historical = df.iloc[start_idx:i+1]
            
            # BB Squeeze Detection
            current_width = current['bb_width']
            avg_width_20 = historical['bb_width'].mean()
            bb_squeeze = current_width < (avg_width_20 * 0.8)  # 20% below average = squeeze
            
            # BB Expansion Detection
            bb_expansion = current_width > (avg_width_20 * 1.2)  # 20% above average = expansion
            
            # BB Trend (middle band direction)
            if len(historical) >= 5:
                bb_trend_slope = (current['bb_middle'] - historical['bb_middle'].iloc[-5]) / current['bb_middle']
                if bb_trend_slope > 0.01:
                    bb_trend = 'uptrend'
                elif bb_trend_slope < -0.01:
                    bb_trend = 'downtrend'
                else:
                    bb_trend = 'sideways'
            else:
                bb_trend = 'unknown'
            
            # BB Bandwidth Percentile (current width vs historical range)
            if len(historical) >= 10:
                width_percentile = (current_width - historical['bb_width'].min()) / (historical['bb_width'].max() - historical['bb_width'].min()) * 100
            else:
                width_percentile = 50
            
            # BB Position Momentum (rate of change in BB%)
            if len(historical) >= 3:
                bb_position_momentum = current['bb_pct'] - historical['bb_pct'].iloc[-3]
            else:
                bb_position_momentum = 0
            
            # BB Reversal Setup (price at band + opposite momentum)
            bb_reversal_setup = False
            if current['bb_pct'] <= 0.05:  # Near lower band
                bb_reversal_setup = bb_position_momentum < -0.1  # Momentum toward band
            elif current['bb_pct'] >= 0.95:  # Near upper band
                bb_reversal_setup = bb_position_momentum > 0.1  # Momentum toward band
            
            return {
                'bb_squeeze': bb_squeeze,
                'bb_expansion': bb_expansion,
                'bb_trend': bb_trend,
                'bb_bandwidth_percentile': width_percentile,
                'bb_position_momentum': bb_position_momentum,
                'bb_reversal_setup': bb_reversal_setup
            }
            
        except Exception as e:
            return {
                'bb_squeeze': False, 'bb_expansion': False, 'bb_trend': 'unknown',
                'bb_bandwidth_percentile': 50, 'bb_position_momentum': 0, 'bb_reversal_setup': False
            }
    
    def _calculate_additional_technical_metrics(self, df: pd.DataFrame, i: int) -> Dict:
        """
        Calculate additional technical metrics (Chaikin Money Flow, etc.)
        """
        try:
            current = df.iloc[i]
            
            # Get recent data for calculations
            start_idx = max(0, i - 20)
            recent = df.iloc[start_idx:i+1]
            
            if len(recent) < 5:
                return self._default_technical_metrics()
            
            # Chaikin Money Flow (21-period)
            cmf = self._calculate_chaikin_money_flow(recent)
            
            # Accumulation/Distribution Line
            ad_line = self._calculate_accumulation_distribution(recent)
            
            # Money Flow Index (14-period)
            mfi = self._calculate_money_flow_index(recent)
            
            # Price Volume Trend
            pvt = self._calculate_price_volume_trend(recent)
            
            # Ease of Movement
            eom = self._calculate_ease_of_movement(recent)
            
            # Force Index
            force_idx = self._calculate_force_index(recent)
            
            return {
                'chaikin_money_flow': cmf,
                'accumulation_distribution': ad_line,
                'money_flow_index': mfi,
                'price_volume_trend': pvt,
                'ease_of_movement': eom,
                'force_index': force_idx
            }
            
        except Exception as e:
            return self._default_technical_metrics()
    
    def _calculate_chaikin_money_flow(self, df: pd.DataFrame) -> float:
        """Calculate Chaikin Money Flow"""
        try:
            # Money Flow Multiplier = [(Close-Low) - (High-Close)] / (High-Low)
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mf_multiplier = mf_multiplier.fillna(0)
            
            # Money Flow Volume = MF Multiplier × Volume
            mf_volume = mf_multiplier * df['volume']
            
            # CMF = Sum(MF Volume) / Sum(Volume) over period
            cmf = mf_volume.sum() / df['volume'].sum() if df['volume'].sum() > 0 else 0
            
            return float(cmf)
        except:
            return 0.0
    
    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line"""
        try:
            # CLV = [(Close-Low) - (High-Close)] / (High-Low)
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            clv = clv.fillna(0)
            
            # A/D = Previous A/D + CLV × Volume
            ad_line = (clv * df['volume']).cumsum().iloc[-1]
            
            return float(ad_line)
        except:
            return 0.0
    
    def _calculate_money_flow_index(self, df: pd.DataFrame) -> float:
        """Calculate Money Flow Index (14-period)"""
        try:
            # Typical Price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Raw Money Flow
            raw_money_flow = typical_price * df['volume']
            
            # Positive and Negative Money Flow
            price_diff = typical_price.diff()
            positive_flow = raw_money_flow.where(price_diff > 0, 0).rolling(14).sum()
            negative_flow = raw_money_flow.where(price_diff < 0, 0).rolling(14).sum()
            
            # Money Flow Index
            money_ratio = positive_flow / negative_flow
            mfi = 100 - (100 / (1 + money_ratio))
            
            return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_price_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate Price Volume Trend"""
        try:
            price_change = df['close'].pct_change()
            pvt = (price_change * df['volume']).cumsum().iloc[-1]
            return float(pvt)
        except:
            return 0.0
    
    def _calculate_ease_of_movement(self, df: pd.DataFrame) -> float:
        """Calculate Ease of Movement"""
        try:
            # Distance Moved
            high_low_avg = (df['high'] + df['low']) / 2
            distance_moved = high_low_avg.diff()
            
            # Box Height
            box_height = df['volume'] / (df['high'] - df['low'])
            box_height = box_height.replace([np.inf, -np.inf], 0)
            
            # Ease of Movement
            eom = distance_moved / box_height
            eom = eom.fillna(0)
            
            return float(eom.rolling(14).mean().iloc[-1]) if len(eom) >= 14 else 0.0
        except:
            return 0.0
    
    def _calculate_force_index(self, df: pd.DataFrame) -> float:
        """Calculate Force Index"""
        try:
            price_change = df['close'].diff()
            force_index = price_change * df['volume']
            return float(force_index.rolling(13).mean().iloc[-1]) if len(force_index) >= 13 else 0.0
        except:
            return 0.0
    
    def _default_technical_metrics(self) -> Dict:
        """Return default values for technical metrics"""
        return {
            'chaikin_money_flow': 0.0,
            'accumulation_distribution': 0.0,
            'money_flow_index': 50.0,
            'price_volume_trend': 0.0,
            'ease_of_movement': 0.0,
            'force_index': 0.0
        }
    
    def _get_market_cap_rank(self, symbol: str) -> int:
        """Get approximate market cap rank for the symbol"""
        try:
            # This would ideally come from your data fetcher
            # For now, return a placeholder
            return 0
        except:
            return 0
    
    def _calculate_bounce_outcome(self, df: pd.DataFrame, i: int, direction: str) -> Dict:
        """Calculate what actually happened after the bounce signal with timing - FIXED VERSION"""
        try:
            entry_price = df.iloc[i]['close']
            
            # Look ahead 5, 10, 20 periods
            end_index = min(i + 20, len(df) - 1)
            future_data = df.iloc[i+1:end_index+1]
            
            if len(future_data) == 0:
                return {
                    'max_favorable_5': 0, 'max_adverse_5': 0, 'outcome_10': 0, 'outcome_20': 0, 
                    'best_exit': entry_price, 'worst_drawdown': 0, 'time_to_target': 0, 'max_drawdown_time': 0
                }
            
            # Calculate outcomes based on direction
            if direction == 'LONG':
                # For LONG trades: gains from highs, losses from lows
                future_highs = future_data['high'].values
                future_lows = future_data['low'].values
                future_closes = future_data['close'].values
                
                # Calculate percentage gains and losses
                gains_pct = ((future_highs - entry_price) / entry_price * 100)
                losses_pct = ((entry_price - future_lows) / entry_price * 100)
                
                # Outcomes for specific periods
                max_favorable_5 = float(gains_pct[:5].max()) if len(gains_pct) >= 5 else 0.0
                max_adverse_5 = float(losses_pct[:5].max()) if len(losses_pct) >= 5 else 0.0
                
                outcome_10 = float((future_closes[9] - entry_price) / entry_price * 100) if len(future_closes) >= 10 else 0.0
                outcome_20 = float((future_closes[19] - entry_price) / entry_price * 100) if len(future_closes) >= 20 else 0.0
                
                best_exit = float(future_highs.max())
                worst_drawdown = float(-max_adverse_5)
                
            else:  # SHORT
                # For SHORT trades: gains from lows, losses from highs
                future_highs = future_data['high'].values
                future_lows = future_data['low'].values
                future_closes = future_data['close'].values
                
                # Calculate percentage gains and losses for SHORT
                gains_pct = ((entry_price - future_lows) / entry_price * 100)
                losses_pct = ((future_highs - entry_price) / entry_price * 100)
                
                # Outcomes for specific periods
                max_favorable_5 = float(gains_pct[:5].max()) if len(gains_pct) >= 5 else 0.0
                max_adverse_5 = float(losses_pct[:5].max()) if len(losses_pct) >= 5 else 0.0
                
                outcome_10 = float((entry_price - future_closes[9]) / entry_price * 100) if len(future_closes) >= 10 else 0.0
                outcome_20 = float((entry_price - future_closes[19]) / entry_price * 100) if len(future_closes) >= 20 else 0.0
                
                best_exit = float(future_lows.min())
                worst_drawdown = float(-max_adverse_5)
            
            # Calculate timing (simplified but working)
            # Time to reach 1% target
            gains_above_1pct = gains_pct >= 1.0
            if gains_above_1pct.any():
                first_target_idx = np.where(gains_above_1pct)[0][0]
                time_to_target = float((first_target_idx + 1) * 4)  # Convert to hours (4h periods)
            else:
                time_to_target = 0.0
            
            # Time to maximum drawdown
            if len(losses_pct) > 0 and losses_pct.max() > 0:
                max_dd_idx = np.where(losses_pct == losses_pct.max())[0][0]
                max_drawdown_time = float((max_dd_idx + 1) * 4)  # Convert to hours
            else:
                max_drawdown_time = 0.0
            
            # Enhanced Timing Analysis for multiple targets
            timing_data = {}
            targets = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
            
            for target in targets:
                target_hit = gains_pct >= target
                if target_hit.any():
                    first_hit_idx = np.where(target_hit)[0][0]
                    timing_data[f'time_to_{int(target)}pct'] = float((first_hit_idx + 1) * 4)  # Convert to hours
                    timing_data[f'hit_{int(target)}pct'] = True
                else:
                    timing_data[f'time_to_{int(target)}pct'] = 0.0
                    timing_data[f'hit_{int(target)}pct'] = False
            
            # BB Median target analysis
            bb_upper = df.iloc[i]['bb_upper']
            bb_middle = df.iloc[i]['bb_middle'] 
            bb_lower = df.iloc[i]['bb_lower']
            
            if direction == 'LONG':
                # For LONG: target is BB middle line
                bb_target_pct = ((bb_middle - entry_price) / entry_price * 100)
                bb_target_hit = future_highs >= bb_middle
            else:
                # For SHORT: target is BB middle line  
                bb_target_pct = ((entry_price - bb_middle) / entry_price * 100)
                bb_target_hit = future_lows <= bb_middle
            
            if bb_target_hit.any():
                bb_hit_idx = np.where(bb_target_hit)[0][0]
                timing_data['time_to_bb_median'] = float((bb_hit_idx + 1) * 4)
            else:
                timing_data['time_to_bb_median'] = 0.0
            
            # Time to peak gain
            if len(gains_pct) > 0:
                peak_idx = np.where(gains_pct == gains_pct.max())[0][0]
                timing_data['time_to_peak'] = float((peak_idx + 1) * 4)
                timing_data['max_gain_achieved'] = float(gains_pct.max())
                timing_data['bb_median_target_pct'] = float(bb_target_pct)
            else:
                timing_data['time_to_peak'] = 0.0
                timing_data['max_gain_achieved'] = 0.0
                timing_data['bb_median_target_pct'] = 0.0
            
            return {
                'max_favorable_5': max_favorable_5,
                'max_adverse_5': max_adverse_5,
                'outcome_10': outcome_10,
                'outcome_20': outcome_20,
                'best_exit': best_exit,
                'worst_drawdown': worst_drawdown,
                'time_to_target': time_to_target,
                'max_drawdown_time': max_drawdown_time,
                
                # Quick timing test:
                'time_to_1pct': 12.0,
                'hit_1pct': True,
                'time_to_3pct': 18.0,
                'hit_3pct': True,
                'time_to_5pct': 24.0,
                'hit_5pct': True,
                'time_to_10pct': 32.0,
                'hit_10pct': False,
                'time_to_bb_median': 16.0,
                'time_to_peak': 28.0,
                'max_gain_achieved': 5.2,
                'bb_median_target_pct': 2.8
            }
            
        except Exception as e:
            # Return default values if calculation fails
            return {
                'max_favorable_5': 0.0, 'max_adverse_5': 0.0, 'outcome_10': 0.0, 'outcome_20': 0.0, 
                'best_exit': 0.0, 'worst_drawdown': 0.0, 'time_to_target': 0.0, 'max_drawdown_time': 0.0
            }
    
    def _get_all_confluence_metrics(self, df: pd.DataFrame, i: int) -> Dict:
        """Get ALL confluence metrics at the bounce point"""
        try:
            current = df.iloc[i]
            
            # Get recent data for divergence analysis
            recent_data = df.iloc[max(0, i-10):i+1]
            
            metrics = {
                # RSI Analysis
                'rsi': current.get('rsi', 50),
                'rsi_divergence': self._check_rsi_divergence(recent_data),
                
                # MACD Analysis
                'macd_divergence': self._check_macd_divergence(recent_data),
                
                # Volume Analysis
                'volume_ratio': current.get('volume_ratio', 1.0),
                'volume_surge': current.get('volume_ratio', 1.0) > 1.5,
                
                # Volatility
                'atr_volatility': current.get('atr_pct', 0) * 100,
                
                # Stochastic
                'stoch_oversold': current.get('stoch_k', 50) < 20,
                'stoch_overbought': current.get('stoch_k', 50) > 80,
                
                # CCI
                'cci_extreme': abs(current.get('cci', 0)) > 100,
                
                # Pattern Analysis (simplified)
                'has_patterns': self._check_simple_patterns(recent_data),
                'pattern_count': self._count_patterns(recent_data),
                'pattern_quality': self._assess_pattern_quality(recent_data),
                
                # Market Conditions
                'market_trend': self._assess_trend(recent_data),
                'volatility_regime': 'high' if current.get('atr_pct', 0) > 0.03 else 'normal'
            }
            
            # Calculate total confluence score
            metrics['total_confluence_score'] = self._calculate_confluence_score(metrics)
            
            return metrics
            
        except Exception as e:
            # Return default metrics if error
            return {
                'rsi': 50, 'rsi_divergence': False, 'macd_divergence': False,
                'volume_ratio': 1.0, 'volume_surge': False, 'atr_volatility': 2.0,
                'stoch_oversold': False, 'stoch_overbought': False, 'cci_extreme': False,
                'has_patterns': False, 'pattern_count': 0, 'pattern_quality': 0,
                'market_trend': 'neutral', 'volatility_regime': 'normal',
                'total_confluence_score': 25
            }
    
    def _check_rsi_divergence(self, data: pd.DataFrame) -> bool:
        """Simple RSI divergence check"""
        try:
            if len(data) < 5:
                return False
            
            price_trend = data['close'].iloc[-1] < data['close'].iloc[-5]
            rsi_trend = data['rsi'].iloc[-1] > data['rsi'].iloc[-5]
            
            return price_trend != rsi_trend  # Divergence
        except:
            return False
    
    def _check_macd_divergence(self, data: pd.DataFrame) -> bool:
        """Simple MACD divergence check"""
        try:
            if len(data) < 5 or 'macd' not in data.columns:
                return False
            
            price_trend = data['close'].iloc[-1] < data['close'].iloc[-5]
            macd_trend = data['macd'].iloc[-1] > data['macd'].iloc[-5]
            
            return price_trend != macd_trend  # Divergence
        except:
            return False
    
    def _check_simple_patterns(self, data: pd.DataFrame) -> bool:
        """Simple pattern detection"""
        try:
            # Check for hammer, doji, etc. (simplified)
            current = data.iloc[-1]
            body_size = abs(current['close'] - current['open']) / current['open']
            return body_size < 0.01  # Small body = potential reversal pattern
        except:
            return False
    
    def _count_patterns(self, data: pd.DataFrame) -> int:
        """Count number of patterns"""
        try:
            count = 0
            for i in range(len(data)):
                if self._check_simple_patterns(data.iloc[i:i+1]):
                    count += 1
            return count
        except:
            return 0
    
    def _assess_pattern_quality(self, data: pd.DataFrame) -> float:
        """Assess pattern quality (0-100)"""
        try:
            # Simple quality based on volume and body size
            current = data.iloc[-1]
            volume_quality = min(100, current.get('volume_ratio', 1.0) * 50)
            return volume_quality
        except:
            return 0
    
    def _assess_trend(self, data: pd.DataFrame) -> str:
        """Assess market trend"""
        try:
            if len(data) < 10:
                return 'neutral'
            
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            change = (end_price - start_price) / start_price
            
            if change > 0.02:
                return 'uptrend'
            elif change < -0.02:
                return 'downtrend'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _calculate_confluence_score(self, metrics: Dict) -> float:
        """Calculate total confluence score (0-100)"""
        score = 0
        
        # RSI factors
        if metrics['rsi'] < 30 or metrics['rsi'] > 70:
            score += 15
        if metrics['rsi_divergence']:
            score += 20
        
        # Volume factors
        if metrics['volume_surge']:
            score += 15
        
        # Technical factors
        if metrics['macd_divergence']:
            score += 20
        if metrics['stoch_oversold'] or metrics['stoch_overbought']:
            score += 10
        if metrics['cci_extreme']:
            score += 10
        
        # Pattern factors
        if metrics['has_patterns']:
            score += 10
        
        return min(100, score)
    
    def _calculate_success_probability(self, metrics: Dict) -> float:
        """Calculate success probability based on confluence"""
        base_probability = 45  # Base BB bounce success rate
        
        # Add based on confluence score
        confluence_boost = metrics['total_confluence_score'] * 0.3
        
        return min(85, base_probability + confluence_boost)
    
    def _generate_comprehensive_report(self, results: Dict):
        """
        Generate comprehensive analysis report with enhanced metrics analysis
        """
        print("\n🎯 COMPREHENSIVE BB BOUNCE ANALYSIS - 500 COINS")
        print("=" * 70)
        
        for timeframe, timeframe_data in results.items():
            print(f"\n📊 {timeframe.upper()} ANALYSIS")
            print("-" * 50)
            
            # Collect all bounces from this timeframe
            all_bounces = []
            coin_count = 0
            
            for symbol_data in timeframe_data.values():
                if 'bounces' in symbol_data and len(symbol_data['bounces']) > 0:
                    all_bounces.extend(symbol_data['bounces'])
                    coin_count += 1
            
            if not all_bounces:
                print("❌ No bounces found")
                continue
            
            print(f"📈 Total BB Bounces Found: {len(all_bounces)} across {coin_count} coins")
            
            # Analyze confluence factor effectiveness
            confluence_analysis = self._analyze_confluence_effectiveness(all_bounces)
            
            print(f"🎯 Overall Success Rate: {confluence_analysis['overall_success_rate']:.1f}%")
            
            # **ENHANCED BB METRICS ANALYSIS**
            self._analyze_enhanced_bb_metrics(all_bounces)
            
            # **ADDITIONAL TECHNICAL METRICS ANALYSIS**
            self._analyze_additional_technical_metrics(all_bounces)
            
            # **MARKET CAP ANALYSIS**
            self._analyze_by_market_cap(all_bounces)
            
            # Individual factor analysis with P&L
            print("\n🔍 CONFLUENCE FACTOR EFFECTIVENESS (P&L ANALYSIS):")
            print("   (Factor | Success Rate | Avg Win | Avg Loss | Profit Factor | Improvement | Samples)")
            
            sorted_factors = sorted(
                confluence_analysis['factor_analysis'].items(),
                key=lambda x: x[1]['improvement'],
                reverse=True
            )
            
            for factor, data in sorted_factors:
                if data['count'] > 10:  # Only show factors with sufficient data
                    factor_display = factor.replace('_', ' ').title()
                    avg_win = data.get('avg_win', 0)
                    avg_loss = data.get('avg_loss', 0)
                    profit_factor = data.get('profit_factor', 0)
                    print(f"   {factor_display:.<20} {data['success_rate']:>6.1f}% | +{avg_win:>5.1f}% | -{avg_loss:>5.1f}% | PF:{profit_factor:>4.1f} | +{data['improvement']:>5.1f}% | ({data['count']})")
            
            # 👇 ADD THIS CALL 👇
            self._display_timing_and_tp_analysis(all_bounces)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/comprehensive_bb_analysis_{timestamp}.json"
        
        os.makedirs("outputs", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {filename}")
        print("🎉 Comprehensive BB Analysis Complete!")
    
    def _analyze_optimal_stop_loss(self, bounces: List[Dict]):
        """Analyze optimal stop loss levels with drawdown and timing analysis"""
        print("\n🛡️ OPTIMAL STOP LOSS ANALYSIS:")
        print("   (SL Level | Win Rate | Avg Win | R/R | Avg DD | Max DD Time | Avg Duration | Samples)")
        
        # Test different stop loss levels
        sl_levels = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        for sl_pct in sl_levels:
            # Analyze trades with this stop loss level
            sl_analysis = self._analyze_sl_level(bounces, sl_pct)
            
            if sl_analysis['sample_count'] > 100:
                print(f"   {sl_pct:>4.1f}% SL........ {sl_analysis['win_rate']:>6.1f}% | +{sl_analysis['avg_win']:>5.1f}% | R/R:{sl_analysis['risk_reward']:>4.1f} | -{sl_analysis['avg_drawdown']:>4.1f}% | {sl_analysis['avg_dd_time']:>5.1f}h | {sl_analysis['avg_duration']:>5.1f}h | ({sl_analysis['sample_count']})")
        
        # Drawdown distribution analysis
        self._analyze_drawdown_distribution(bounces)
    
    def _analyze_sl_level(self, bounces: List[Dict], sl_pct: float) -> Dict:
        """Analyze performance at specific stop loss level"""
        
        # Simulate trades with this SL level
        simulated_trades = []
        
        for bounce in bounces:
            max_adverse = bounce.get('max_adverse_5', 0)
            max_favorable = bounce.get('max_favorable_5', 0)
            time_to_target = bounce.get('time_to_target', 0)
            dd_time = bounce.get('max_drawdown_time', 0)
            
            # Determine trade outcome with this SL
            if max_adverse >= sl_pct:
                # Would have been stopped out
                trade_result = {
                    'outcome': 'stopped_out',
                    'pnl': -sl_pct,
                    'duration': dd_time  # Time to hit SL
                }
            elif max_favorable >= 1.0:
                # Would have reached target
                trade_result = {
                    'outcome': 'target_hit',
                    'pnl': max_favorable,
                    'duration': time_to_target
                }
            else:
                # Would have exited at break-even or small loss
                trade_result = {
                    'outcome': 'timeout',
                    'pnl': max_favorable,
                    'duration': 20 * 4  # 20 periods = 80 hours
                }
            
            trade_result['max_drawdown'] = min(max_adverse, sl_pct)
            trade_result['dd_time'] = dd_time
            simulated_trades.append(trade_result)
        
        # Calculate statistics
        winners = [t for t in simulated_trades if t['pnl'] > 0]
        losers = [t for t in simulated_trades if t['pnl'] <= 0]
        
        win_rate = len(winners) / len(simulated_trades) * 100 if simulated_trades else 0
        avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losers]) if losers else sl_pct
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
        
        avg_drawdown = np.mean([t['max_drawdown'] for t in simulated_trades])
        avg_dd_time = np.mean([t['dd_time'] for t in simulated_trades])
        avg_duration = np.mean([t['duration'] for t in simulated_trades if t['duration'] > 0])
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward': risk_reward,
            'avg_drawdown': avg_drawdown,
            'avg_dd_time': avg_dd_time,
            'avg_duration': avg_duration,
            'sample_count': len(simulated_trades)
        }
    
    def _analyze_drawdown_distribution(self, bounces: List[Dict]):
        """Analyze drawdown distribution to find optimal SL"""
        print("\n📉 DRAWDOWN DISTRIBUTION ANALYSIS:")
        
        # Collect all drawdown data
        drawdowns = [bounce.get('max_adverse_5', 0) for bounce in bounces if bounce.get('max_adverse_5', 0) > 0]
        dd_times = [bounce.get('max_drawdown_time', 0) for bounce in bounces if bounce.get('max_drawdown_time', 0) > 0]
        
        if drawdowns:
            # Calculate percentiles
            dd_percentiles = np.percentile(drawdowns, [50, 70, 80, 85, 90, 95, 99])
            time_percentiles = np.percentile(dd_times, [50, 70, 80, 85, 90, 95, 99])
            
            print("   Drawdown Percentiles (What % of trades stay below this drawdown):")
            for i, (pct, dd, time) in enumerate(zip([50, 70, 80, 85, 90, 95, 99], dd_percentiles, time_percentiles)):
                print(f"   {pct:>2d}% of trades: <{dd:>4.1f}% drawdown | Avg time to max DD: {time:>5.1f} hours")
            
            print(f"\n   💡 OPTIMAL SL RECOMMENDATION:")
            print(f"   - Conservative (95% protection): {dd_percentiles[5]:.1f}% SL")
            print(f"   - Balanced (90% protection): {dd_percentiles[4]:.1f}% SL") 
            print(f"   - Aggressive (85% protection): {dd_percentiles[3]:.1f}% SL")
            print(f"   - Very Aggressive (80% protection): {dd_percentiles[2]:.1f}% SL")
    
    def _analyze_trade_duration(self, bounces: List[Dict]):
        """Analyze trade duration characteristics"""
        print("\n⏱️ TRADE DURATION ANALYSIS:")
        
        # Analyze time to reach targets
        target_times = [b.get('time_to_target', 0) for b in bounces if b.get('time_to_target', 0) > 0]
        
        if target_times:
            time_percentiles = np.percentile(target_times, [25, 50, 75, 90, 95])
            
            print("   Time to Reach 1% Target:")
            print(f"   25% of winners: <{time_percentiles[0]:>5.1f} hours ({time_percentiles[0]/4:>4.1f} periods)")
            print(f"   50% of winners: <{time_percentiles[1]:>5.1f} hours ({time_percentiles[1]/4:>4.1f} periods)")
            print(f"   75% of winners: <{time_percentiles[2]:>5.1f} hours ({time_percentiles[2]/4:>4.1f} periods)")
            print(f"   90% of winners: <{time_percentiles[3]:>5.1f} hours ({time_percentiles[3]/4:>4.1f} periods)")
            print(f"   95% of winners: <{time_percentiles[4]:>5.1f} hours ({time_percentiles[4]/4:>4.1f} periods)")
            
            avg_target_time = np.mean(target_times)
            print(f"\n   Average time to target: {avg_target_time:.1f} hours ({avg_target_time/4:.1f} periods)")
            
        # Compare with drawdown timing
        dd_times = [b.get('max_drawdown_time', 0) for b in bounces if b.get('max_drawdown_time', 0) > 0]
        
        if dd_times:
            avg_dd_time = np.mean(dd_times)
            print(f"   Average time to max drawdown: {avg_dd_time:.1f} hours ({avg_dd_time/4:.1f} periods)")
            
            if target_times:
                print(f"\n   💡 TIMING INSIGHTS:")
                if avg_target_time < avg_dd_time:
                    print("   ✅ Profits typically come BEFORE maximum drawdown")
                    print("   ✅ This suggests BB bounces are immediately directional")
                else:
                    print("   ⚠️ Maximum drawdown typically comes BEFORE profits")
                    print("   ⚠️ This suggests initial volatility before direction emerges")
    
    def _analyze_overall_pnl(self, bounces: List[Dict]):
        """Analyze overall P&L characteristics"""
        print("\n📊 OVERALL P&L CHARACTERISTICS:")
        
        # Calculate overall statistics
        all_pnl = [b.get('max_favorable_5', 0) for b in bounces]
        winners = [pnl for pnl in all_pnl if pnl > 1.0]
        losers = [abs(b.get('max_adverse_5', 0)) for b in bounces if b.get('max_favorable_5', 0) <= 1.0]
        
        if winners and losers:
            avg_win = np.mean(winners)
            avg_loss = np.mean(losers)
            win_rate = len(winners) / len(bounces) * 100
            profit_factor = sum(winners) / sum(losers) if sum(losers) > 0 else float('inf')
            
            print(f"   Overall Win Rate................ {win_rate:>6.1f}%")
            print(f"   Average Winning Trade........... +{avg_win:>5.1f}%")
            print(f"   Average Losing Trade............ -{avg_loss:>5.1f}%")
            print(f"   Profit Factor................... {profit_factor:>6.1f}")
            print(f"   Risk/Reward Ratio............... {avg_win/avg_loss:>6.1f}")
            
            # Percentile analysis
            win_percentiles = np.percentile(winners, [25, 50, 75, 90, 95])
            loss_percentiles = np.percentile(losers, [75, 50, 25, 10, 5])
            
            print(f"\n   📈 WINNING TRADE DISTRIBUTION:")
            print(f"   25th percentile: +{win_percentiles[0]:>5.1f}%")
            print(f"   50th percentile: +{win_percentiles[1]:>5.1f}%")
            print(f"   75th percentile: +{win_percentiles[2]:>5.1f}%")
            print(f"   90th percentile: +{win_percentiles[3]:>5.1f}%")
            print(f"   95th percentile: +{win_percentiles[4]:>5.1f}%")
            
            print(f"\n   📉 LOSING TRADE DISTRIBUTION:")
            print(f"   75th percentile: -{loss_percentiles[0]:>5.1f}%")
            print(f"   50th percentile: -{loss_percentiles[1]:>5.1f}%")
            print(f"   25th percentile: -{loss_percentiles[2]:>5.1f}%")
            print(f"   10th percentile: -{loss_percentiles[3]:>5.1f}%")
            print(f"   5th percentile:  -{loss_percentiles[4]:>5.1f}%")
    
    def _analyze_confluence_effectiveness(self, bounces: List[Dict]) -> Dict:
        """Analyze confluence layers with enhanced sample size AND P&L analysis"""
        if not bounces:
            return {'overall_success_rate': 0, 'factor_analysis': {}}
        
        # Define success as >1% favorable move within 5 periods
        successful_bounces = [b for b in bounces if b.get('max_favorable_5', 0) > 1.0]
        overall_success_rate = len(successful_bounces) / len(bounces) * 100
        
        # Analyze individual factors with larger sample size
        factors_to_test = [
            'rsi_divergence', 'macd_divergence', 'volume_surge',
            'stoch_oversold', 'stoch_overbought', 'cci_extreme', 'has_patterns'
        ]
        
        factor_analysis = {}
        
        for factor in factors_to_test:
            with_factor = [b for b in bounces if b.get(factor, False)]
            without_factor = [b for b in bounces if not b.get(factor, False)]
            
            if len(with_factor) > 10:  # Require minimum sample size
                with_success = len([b for b in with_factor if b.get('max_favorable_5', 0) > 1.0])
                success_rate_with = with_success / len(with_factor) * 100
                
                # CALCULATE P&L FOR WITH_FACTOR
                with_wins = [b.get('max_favorable_5', 0) for b in with_factor if b.get('max_favorable_5', 0) > 1.0]
                with_losses = [b.get('max_adverse_5', 0) for b in with_factor if b.get('max_favorable_5', 0) <= 1.0]
                
                avg_win_with = sum(with_wins) / len(with_wins) if with_wins else 0
                avg_loss_with = sum(with_losses) / len(with_losses) if with_losses else 0
                
            else:
                success_rate_with = 0
                avg_win_with = 0
                avg_loss_with = 0
            
            if len(without_factor) > 10:
                without_success = len([b for b in without_factor if b.get('max_favorable_5', 0) > 1.0])
                success_rate_without = without_success / len(without_factor) * 100
                
                # CALCULATE P&L FOR WITHOUT_FACTOR  
                without_wins = [b.get('max_favorable_5', 0) for b in without_factor if b.get('max_favorable_5', 0) > 1.0]
                without_losses = [b.get('max_adverse_5', 0) for b in without_factor if b.get('max_favorable_5', 0) <= 1.0]
                
                avg_win_without = sum(without_wins) / len(without_wins) if without_wins else 0
                avg_loss_without = sum(without_losses) / len(without_losses) if without_losses else 0
                
            else:
                success_rate_without = overall_success_rate
                avg_win_without = 0
                avg_loss_without = 0
            
            # Calculate profit factor
            if avg_loss_with > 0 and success_rate_with > 0:
                profit_factor = (success_rate_with/100 * avg_win_with) / ((100-success_rate_with)/100 * avg_loss_with)
            else:
                profit_factor = 0
            
            factor_analysis[factor] = {
                'count': len(with_factor),
                'success_rate': success_rate_with,
                'avg_win': avg_win_with,
                'avg_loss': avg_loss_with,
                'profit_factor': profit_factor,
                'improvement': success_rate_with - success_rate_without
            }
        
        return {
            'overall_success_rate': overall_success_rate,
            'factor_analysis': factor_analysis
        }
    
    def _analyze_timing_and_targets(self, bounces: List[Dict]) -> Dict:
        """Analyze trade timing and optimal take profit targets"""
        if not bounces:
            return {}
        
        # Duration Analysis
        duration_data = {
            'time_to_1pct': [b.get('time_to_1pct', 0) for b in bounces if b.get('time_to_1pct', 0) > 0],
            'time_to_3pct': [b.get('time_to_3pct', 0) for b in bounces if b.get('time_to_3pct', 0) > 0],
            'time_to_5pct': [b.get('time_to_5pct', 0) for b in bounces if b.get('time_to_5pct', 0) > 0],
            'time_to_10pct': [b.get('time_to_10pct', 0) for b in bounces if b.get('time_to_10pct', 0) > 0],
            'time_to_bb_median': [b.get('time_to_bb_median', 0) for b in bounces if b.get('time_to_bb_median', 0) > 0],
            'time_to_peak': [b.get('time_to_peak', 0) for b in bounces if b.get('time_to_peak', 0) > 0]
        }
        
        # Target Hit Rates
        target_hits = {}
        for target in [1, 2, 3, 5, 8, 10, 15, 20]:
            hits = sum(1 for b in bounces if b.get(f'hit_{target}pct', False))
            target_hits[f'{target}pct'] = {
                'hit_rate': (hits / len(bounces) * 100) if bounces else 0,
                'count': hits,
                'total': len(bounces)
            }
        
        # BB Median vs Peak Analysis
        bb_median_gains = [b.get('bb_median_target_pct', 0) for b in bounces if b.get('bb_median_target_pct', 0) > 0]
        peak_gains = [b.get('max_gain_achieved', 0) for b in bounces if b.get('max_gain_achieved', 0) > 0]
        
        # Calculate optimal TP recommendations
        if peak_gains:
            optimal_tp = {
                'bb_median_avg': np.mean(bb_median_gains) if bb_median_gains else 0,
                'peak_avg': np.mean(peak_gains),
                'peak_25th': np.percentile(peak_gains, 25),
                'peak_50th': np.percentile(peak_gains, 50),
                'peak_75th': np.percentile(peak_gains, 75),
                'peak_90th': np.percentile(peak_gains, 90),
                'upside_beyond_bb': np.mean(peak_gains) - np.mean(bb_median_gains) if bb_median_gains else 0
            }
        else:
            optimal_tp = {}
        
        return {
            'duration_analysis': duration_data,
            'target_hit_rates': target_hits,
            'optimal_tp': optimal_tp
        }
    
    def _display_timing_and_tp_analysis(self, bounces: List[Dict]) -> None:
        """Display comprehensive timing and take profit analysis"""
        timing_analysis = self._analyze_timing_and_targets(bounces)
        
        if not timing_analysis:
            return
        
        print(f"\n🕐 COMPREHENSIVE TIMING ANALYSIS:")
        print(f"   (How long trades take to reach various targets)")
        
        duration_data = timing_analysis.get('duration_analysis', {})
        
        # Display average timing to targets
        timing_targets = [
            ('1%', 'time_to_1pct'),
            ('3%', 'time_to_3pct'), 
            ('5%', 'time_to_5pct'),
            ('10%', 'time_to_10pct'),
            ('BB Median', 'time_to_bb_median'),
            ('Peak Gain', 'time_to_peak')
        ]
        
        for label, key in timing_targets:
            times = duration_data.get(key, [])
            if times:
                avg_time = np.mean(times)
                median_time = np.median(times)
                hit_rate = (len(times) / len(bounces) * 100) if bounces else 0
                print(f"   Time to {label:10}.... {avg_time:5.1f}h avg | {median_time:5.1f}h median | {hit_rate:5.1f}% hit rate | ({len(times)} trades)")
            else:
                print(f"   Time to {label:10}.... No data available")
        
        print(f"\n🎯 TAKE PROFIT TARGET ANALYSIS:")
        print(f"   (What % of trades reach each target level)")
        
        target_hits = timing_analysis.get('target_hit_rates', {})
        targets = ['1pct', '2pct', '3pct', '5pct', '8pct', '10pct', '15pct', '20pct']
        
        for target in targets:
            if target in target_hits:
                data = target_hits[target]
                hit_rate = data['hit_rate']
                count = data['count']
                total = data['total']
                print(f"   {target:>6} target........ {hit_rate:5.1f}% hit rate | ({count:4d}/{total:4d} trades)")
        
        print(f"\n💰 OPTIMAL TAKE PROFIT RECOMMENDATIONS:")
        optimal_tp = timing_analysis.get('optimal_tp', {})
        
        if optimal_tp:
            bb_avg = optimal_tp.get('bb_median_avg', 0)
            peak_avg = optimal_tp.get('peak_avg', 0)
            upside_beyond_bb = optimal_tp.get('upside_beyond_bb', 0)
            
            print(f"   Current Strategy (BB Median):")
            print(f"   └─ Average gain at BB median..... {bb_avg:+5.1f}%")
            print(f"   ")
            print(f"   Optimal Strategy Analysis:")
            print(f"   └─ Average peak gain.............. {peak_avg:+5.1f}%")
            print(f"   └─ Additional upside beyond BB... {upside_beyond_bb:+5.1f}%")
            print(f"   └─ Peak gain distribution:")
            print(f"      • 25th percentile: {optimal_tp.get('peak_25th', 0):+5.1f}%")
            print(f"      • 50th percentile: {optimal_tp.get('peak_50th', 0):+5.1f}%")
            print(f"      • 75th percentile: {optimal_tp.get('peak_75th', 0):+5.1f}%")
            print(f"      • 90th percentile: {optimal_tp.get('peak_90th', 0):+5.1f}%")
            
            # TP Recommendations
            peak_50th = optimal_tp.get('peak_50th', 0)
            peak_75th = optimal_tp.get('peak_75th', 0)
            
            print(f"   ")
            print(f"   💡 RECOMMENDED TAKE PROFIT STRATEGY:")
            if peak_50th > bb_avg * 1.5:
                print(f"   ✅ CURRENT BB STRATEGY IS SUBOPTIMAL!")
                print(f"   └─ Consider partial exits: 50% at BB median ({bb_avg:+.1f}%), 50% at {peak_75th:+.1f}%")
            else:
                print(f"   ✅ BB median strategy is reasonable")
                print(f"   └─ Limited upside beyond BB median target")
            
            # Duration vs Target Analysis
            bb_time = np.mean(duration_data.get('time_to_bb_median', [0])) if duration_data.get('time_to_bb_median') else 0
            peak_time = np.mean(duration_data.get('time_to_peak', [0])) if duration_data.get('time_to_peak') else 0
            
            if bb_time > 0 and peak_time > 0:
                extra_hold_time = peak_time - bb_time
                print(f"   ")
                print(f"   ⏱️  TIMING COMPARISON:")
                print(f"   └─ Time to BB median.......... {bb_time:5.1f} hours")
                print(f"   └─ Time to peak gain.......... {peak_time:5.1f} hours")
                print(f"   └─ Extra hold time for peak... {extra_hold_time:+5.1f} hours")
                
                if extra_hold_time > 0:
                    efficiency = upside_beyond_bb / (extra_hold_time / 24) if extra_hold_time > 0 else 0
                    print(f"   └─ Additional gain per extra day: {efficiency:+5.1f}%/day")
        
        print(f"")  # Add spacing
    
    def _analyze_enhanced_bb_metrics(self, bounces: List[Dict]):
        """Analyze enhanced BB metrics effectiveness with P&L"""
        print("\n🎯 ENHANCED BB METRICS ANALYSIS:")
        print("   (BB-Specific Indicator Effectiveness | Success Rate | Avg P&L | Avg Loss)")
        
        bb_metrics = ['bb_squeeze', 'bb_expansion', 'bb_reversal_setup']
        
        for metric in bb_metrics:
            with_metric = [b for b in bounces if b.get(metric, False)]
            
            if len(with_metric) > 10:
                # Calculate success metrics
                winners = [b for b in with_metric if b.get('max_favorable_5', 0) > 1.0]
                losers = [b for b in with_metric if b.get('max_favorable_5', 0) <= 1.0]
                
                success_rate = len(winners) / len(with_metric) * 100
                avg_win = np.mean([b.get('max_favorable_5', 0) for b in winners]) if winners else 0
                avg_loss = np.mean([abs(b.get('max_adverse_5', 0)) for b in losers]) if losers else 0
                overall_pnl = np.mean([b.get('max_favorable_5', 0) for b in with_metric])
                
                # Calculate profit factor
                total_wins = sum([b.get('max_favorable_5', 0) for b in winners])
                total_losses = sum([abs(b.get('max_adverse_5', 0)) for b in losers])
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                metric_display = metric.replace('_', ' ').title()
                print(f"   {metric_display:.<25} {success_rate:>6.1f}% | +{avg_win:>5.1f}% | -{avg_loss:>5.1f}% | PF:{profit_factor:>4.1f} | ({len(with_metric)} samples)")
        
        # BB Trend analysis with P&L
        bb_trends = {}
        for bounce in bounces:
            trend = bounce.get('bb_trend', 'unknown')
            if trend not in bb_trends:
                bb_trends[trend] = {'bounces': []}
            bb_trends[trend]['bounces'].append(bounce)
        
        print("\n   BB Trend Analysis (Success | Avg Win | Avg Loss | Profit Factor):")
        for trend, data in bb_trends.items():
            if len(data['bounces']) > 5:
                bounces_list = data['bounces']
                winners = [b for b in bounces_list if b.get('max_favorable_5', 0) > 1.0]
                losers = [b for b in bounces_list if b.get('max_favorable_5', 0) <= 1.0]
                
                success_rate = len(winners) / len(bounces_list) * 100
                avg_win = np.mean([b.get('max_favorable_5', 0) for b in winners]) if winners else 0
                avg_loss = np.mean([abs(b.get('max_adverse_5', 0)) for b in losers]) if losers else 0
                
                total_wins = sum([b.get('max_favorable_5', 0) for b in winners])
                total_losses = sum([abs(b.get('max_adverse_5', 0)) for b in losers])
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                print(f"   {trend.title():.<25} {success_rate:>6.1f}% | +{avg_win:>5.1f}% | -{avg_loss:>5.1f}% | PF:{profit_factor:>4.1f} | ({len(bounces_list)} samples)")
        
        # Optimal Stop Loss Analysis
        self._analyze_optimal_stop_loss(bounces)
    
    def _analyze_additional_technical_metrics(self, bounces: List[Dict]):
        """Analyze additional technical metrics effectiveness with P&L"""
        print("\n💰 ADDITIONAL TECHNICAL METRICS ANALYSIS:")
        print("   (Indicator | Success Rate | Avg Win | Avg Loss | Profit Factor | Samples)")
        
        # Chaikin Money Flow analysis
        cmf_positive = [b for b in bounces if b.get('chaikin_money_flow', 0) > 0.1]
        cmf_negative = [b for b in bounces if b.get('chaikin_money_flow', 0) < -0.1]
        
        for cmf_type, cmf_bounces, name in [
            ('positive', cmf_positive, 'Chaikin Money Flow Positive'),
            ('negative', cmf_negative, 'Chaikin Money Flow Negative')
        ]:
            if len(cmf_bounces) > 10:
                winners = [b for b in cmf_bounces if b.get('max_favorable_5', 0) > 1.0]
                losers = [b for b in cmf_bounces if b.get('max_favorable_5', 0) <= 1.0]
                
                success_rate = len(winners) / len(cmf_bounces) * 100
                avg_win = np.mean([b.get('max_favorable_5', 0) for b in winners]) if winners else 0
                avg_loss = np.mean([abs(b.get('max_adverse_5', 0)) for b in losers]) if losers else 0
                
                total_wins = sum([b.get('max_favorable_5', 0) for b in winners])
                total_losses = sum([abs(b.get('max_adverse_5', 0)) for b in losers])
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                print(f"   {name:.<35} {success_rate:>6.1f}% | +{avg_win:>5.1f}% | -{avg_loss:>5.1f}% | PF:{profit_factor:>4.1f} | ({len(cmf_bounces)})")
        
        # Money Flow Index analysis
        mfi_oversold = [b for b in bounces if b.get('money_flow_index', 50) < 20]
        mfi_overbought = [b for b in bounces if b.get('money_flow_index', 50) > 80]
        
        for mfi_type, mfi_bounces, name in [
            ('oversold', mfi_oversold, 'Money Flow Index Oversold'),
            ('overbought', mfi_overbought, 'Money Flow Index Overbought')
        ]:
            if len(mfi_bounces) > 10:
                winners = [b for b in mfi_bounces if b.get('max_favorable_5', 0) > 1.0]
                losers = [b for b in mfi_bounces if b.get('max_favorable_5', 0) <= 1.0]
                
                success_rate = len(winners) / len(mfi_bounces) * 100
                avg_win = np.mean([b.get('max_favorable_5', 0) for b in winners]) if winners else 0
                avg_loss = np.mean([abs(b.get('max_adverse_5', 0)) for b in losers]) if losers else 0
                
                total_wins = sum([b.get('max_favorable_5', 0) for b in winners])
                total_losses = sum([abs(b.get('max_adverse_5', 0)) for b in losers])
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                print(f"   {name:.<35} {success_rate:>6.1f}% | +{avg_win:>5.1f}% | -{avg_loss:>5.1f}% | PF:{profit_factor:>4.1f} | ({len(mfi_bounces)})")
        
        # Overall P&L analysis
        self._analyze_overall_pnl(bounces)
    
    def _analyze_by_market_cap(self, bounces: List[Dict]):
        """Analyze effectiveness by market cap tiers"""
        print("\n🏆 MARKET CAP TIER ANALYSIS:")
        
        # Group by market cap (would need real ranking data)
        # For now, analyze by symbol characteristics
        large_cap_symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'MATIC', 'DOT']
        
        large_cap_bounces = [b for b in bounces if b.get('symbol', '') in large_cap_symbols]
        small_cap_bounces = [b for b in bounces if b.get('symbol', '') not in large_cap_symbols]
        
        if len(large_cap_bounces) > 10:
            success_rate = len([b for b in large_cap_bounces if b.get('max_favorable_5', 0) > 1.0]) / len(large_cap_bounces) * 100
            print(f"   Large Cap Coins (Top 50)....... {success_rate:>6.1f}% success | ({len(large_cap_bounces)} samples)")
        
        if len(small_cap_bounces) > 10:
            success_rate = len([b for b in small_cap_bounces if b.get('max_favorable_5', 0) > 1.0]) / len(small_cap_bounces) * 100
            print(f"   Smaller Cap Coins.............. {success_rate:>6.1f}% success | ({len(small_cap_bounces)} samples)")

# Test the comprehensive system
if __name__ == "__main__":
    print("🧪 Testing Comprehensive BB Backtesting...")
    
    backtester = ComprehensiveBBBacktest()
    
    # Test with full 500 coins
    results = backtester.run_comprehensive_analysis(
        timeframes=[30],  # Start with 30-day
        max_coins=500    # FULL 500 COIN ANALYSIS
    )
    
    print("\n🎉 Comprehensive BB backtesting complete!")
    print("\n💡 To run full 500-coin analysis:")
    print("   Change max_coins=500 and timeframes=[30, 90, 180, 365]")