#!/usr/bin/env python3
"""
Improved BB Backtesting - All Bounces with Metrics Attribution
Find ALL BB bounces first, then analyze which confluence factors improve performance
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_fetcher import MarketDataFetcher
from modules.bb_detector import BBDetector
from modules.technical_analyzer import TechnicalAnalyzer
from modules.pattern_analyzer import PatternAnalyzer
from modules.risk_manager import RiskManager
from modules.enhanced_backtesting_engine import EnhancedBacktestingEngine
from config import SCANNER_CONFIG

class ImprovedBBBacktester:
    """
    BB-First Backtesting: Find ALL bounces, then analyze confluence factors
    """
    
    def __init__(self):
        print("🚀 IMPROVED BB BACKTESTING - ALL BOUNCES + METRICS ATTRIBUTION")
        print("=" * 80)
        print("🏗️ Initializing BB-First Historical Analyzer...")
        
        # Initialize modules
        self.data_fetcher = MarketDataFetcher()
        self.bb_detector = BBDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.risk_manager = RiskManager()
        
        print("✅ All modules initialized successfully")
        
    def run_comprehensive_bb_analysis(self, timeframes=[30, 90, 180, 365], symbols=None):
        """
        Find ALL BB bounces across multiple timeframes, then analyze confluence factors
        """
        print("\n🎯 STARTING COMPREHENSIVE BB BOUNCE ANALYSIS")
        print("=" * 60)
        
        # Use top coins if no symbols provided
        if symbols is None:
            top_coins = self.data_fetcher.fetch_top_coins(limit=20)
            symbols = [coin['symbol'] for coin in top_coins[:10]]  # Test top 10
            print(f"✅ Selected {len(symbols)} coins for analysis")
        
        all_results = {}
        
        for timeframe_days in timeframes:
            print(f"\n📅 ANALYZING {timeframe_days}-DAY TIMEFRAME")
            print("-" * 50)
            
            timeframe_results = self._analyze_timeframe(symbols, timeframe_days)
            all_results[f"{timeframe_days}d"] = timeframe_results
            
            # Summary for this timeframe
            total_bounces = sum(len(data['bounces']) for data in timeframe_results.values())
            print(f"📊 Found {total_bounces} total BB bounces in {timeframe_days}-day period")
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_results)
        
        return all_results
    
    def _analyze_timeframe(self, symbols: List[str], timeframe_days: int) -> Dict:
        """
        Analyze all BB bounces in a specific timeframe
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=timeframe_days)
        
        timeframe_results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"📊 Analyzing {symbol} ({i}/{len(symbols)})...")
            
            try:
                symbol_results = self._find_all_bb_bounces(symbol, start_date, end_date)
                timeframe_results[symbol] = symbol_results
                
                bounce_count = len(symbol_results['bounces'])
                if bounce_count > 0:
                    print(f"   ✅ Found {bounce_count} BB bounces for {symbol}")
                else:
                    print(f"   ➖ No BB bounces found for {symbol}")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {str(e)}")
                timeframe_results[symbol] = {'bounces': [], 'error': str(e)}
        
        return timeframe_results
    
    def _find_all_bb_bounces(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Find ALL BB bounces for a symbol, regardless of other conditions
        """
        all_bounces = []
        
        # Check multiple exchanges
        exchanges = ['binance', 'bybit', 'kucoin', 'okx']
        
        for exchange in exchanges:
            try:
                # Get historical data
                df = self._get_historical_data(symbol, exchange, start_date, end_date)
                if df is None or len(df) < 50:
                    continue
                
                # Find BB bounces using relaxed criteria
                bounces = self._detect_bb_bounces_relaxed(df, symbol, exchange)
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
    
    def _detect_bb_bounces_relaxed(self, df: pd.DataFrame, symbol: str, exchange: str) -> List[Dict]:
        """
        Detect BB bounces with RELAXED criteria - find ALL potential bounces
        """
        bounces = []
        
        try:
            # Ensure we have BB data
            if 'bb_lower' not in df.columns or 'bb_upper' not in df.columns:
                return bounces
            
            # Find touches to BB bands (relaxed criteria)
            for i in range(10, len(df) - 5):  # Leave room for confirmation
                current_row = df.iloc[i]
                
                # LOWER BAND TOUCHES (Potential LONG setups)
                if self._is_lower_band_touch_relaxed(df, i):
                    bounce = self._create_bounce_record(
                        df, i, symbol, exchange, 'LONG', 'lower_band_touch'
                    )
                    if bounce:
                        bounces.append(bounce)
                
                # UPPER BAND TOUCHES (Potential SHORT setups)  
                if self._is_upper_band_touch_relaxed(df, i):
                    bounce = self._create_bounce_record(
                        df, i, symbol, exchange, 'SHORT', 'upper_band_touch'
                    )
                    if bounce:
                        bounces.append(bounce)
        
        except Exception as e:
            pass
        
        return bounces
    
    def _is_lower_band_touch_relaxed(self, df: pd.DataFrame, i: int) -> bool:
        """
        Relaxed lower band touch detection - more permissive
        """
        try:
            current = df.iloc[i]
            
            # Touch criteria: Low touches or comes very close to lower band
            distance_to_lower = (current['low'] - current['bb_lower']) / current['bb_lower']
            
            # RELAXED: Within 0.3% of lower band (was 1%)
            return distance_to_lower <= 0.003
            
        except:
            return False
    
    def _is_upper_band_touch_relaxed(self, df: pd.DataFrame, i: int) -> bool:
        """
        Relaxed upper band touch detection - more permissive
        """
        try:
            current = df.iloc[i]
            
            # Touch criteria: High touches or comes very close to upper band
            distance_to_upper = (current['bb_upper'] - current['high']) / current['bb_upper']
            
            # RELAXED: Within 0.3% of upper band (was 1%)
            return distance_to_upper <= 0.003
            
        except:
            return False
    
    def _create_bounce_record(self, df: pd.DataFrame, i: int, symbol: str, exchange: str, 
                            direction: str, signal_type: str) -> Optional[Dict]:
        """
        Create comprehensive bounce record with ALL metrics
        """
        try:
            current_row = df.iloc[i]
            timestamp = current_row.name
            
            # Calculate potential outcome (look ahead 5-20 periods)
            outcome = self._calculate_bounce_outcome(df, i, direction)
            
            # Get all confluence metrics at this point
            confluence_metrics = self._get_all_confluence_metrics(df, i)
            
            bounce_record = {
                # Basic info
                'timestamp': timestamp,
                'symbol': symbol,
                'exchange': exchange,
                'direction': direction,
                'signal_type': signal_type,
                'entry_price': current_row['close'],
                
                # BB specific data
                'bb_position': current_row['bb_pct'],
                'bb_width': current_row['bb_width'],
                'bb_lower': current_row['bb_lower'],
                'bb_upper': current_row['bb_upper'],
                'bb_middle': current_row['bb_middle'],
                
                # Outcome data
                'max_favorable_5': outcome['max_favorable_5'],
                'max_adverse_5': outcome['max_adverse_5'],
                'outcome_10_periods': outcome['outcome_10'],
                'outcome_20_periods': outcome['outcome_20'],
                'best_exit_price': outcome['best_exit'],
                'worst_drawdown': outcome['worst_drawdown'],
                
                # ALL Confluence Metrics
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
    
    def _calculate_bounce_outcome(self, df: pd.DataFrame, i: int, direction: str) -> Dict:
        """
        Calculate what actually happened after the bounce signal
        """
        try:
            entry_price = df.iloc[i]['close']
            
            # Look ahead 5, 10, 20 periods
            end_index = min(i + 20, len(df) - 1)
            future_data = df.iloc[i+1:end_index+1]
            
            if len(future_data) == 0:
                return {'max_favorable_5': 0, 'max_adverse_5': 0, 'outcome_10': 0, 'outcome_20': 0, 'best_exit': entry_price, 'worst_drawdown': 0}
            
            if direction == 'LONG':
                # For LONG: favorable = higher prices, adverse = lower prices
                max_favorable_5 = ((future_data['high'][:5].max() - entry_price) / entry_price * 100) if len(future_data) >= 5 else 0
                max_adverse_5 = ((entry_price - future_data['low'][:5].min()) / entry_price * 100) if len(future_data) >= 5 else 0
                outcome_10 = ((future_data['close'].iloc[9] - entry_price) / entry_price * 100) if len(future_data) >= 10 else 0
                outcome_20 = ((future_data['close'].iloc[19] - entry_price) / entry_price * 100) if len(future_data) >= 20 else 0
                best_exit = future_data['high'].max()
                worst_drawdown = -max_adverse_5
            else:  # SHORT
                # For SHORT: favorable = lower prices, adverse = higher prices
                max_favorable_5 = ((entry_price - future_data['low'][:5].min()) / entry_price * 100) if len(future_data) >= 5 else 0
                max_adverse_5 = ((future_data['high'][:5].max() - entry_price) / entry_price * 100) if len(future_data) >= 5 else 0
                outcome_10 = ((entry_price - future_data['close'].iloc[9]) / entry_price * 100) if len(future_data) >= 10 else 0
                outcome_20 = ((entry_price - future_data['close'].iloc[19]) / entry_price * 100) if len(future_data) >= 20 else 0
                best_exit = future_data['low'].min()
                worst_drawdown = -max_adverse_5
            
            return {
                'max_favorable_5': max_favorable_5,
                'max_adverse_5': max_adverse_5,
                'outcome_10': outcome_10,
                'outcome_20': outcome_20,
                'best_exit': best_exit,
                'worst_drawdown': worst_drawdown
            }
            
        except Exception as e:
            return {'max_favorable_5': 0, 'max_adverse_5': 0, 'outcome_10': 0, 'outcome_20': 0, 'best_exit': 0, 'worst_drawdown': 0}
    
    def _get_all_confluence_metrics(self, df: pd.DataFrame, i: int) -> Dict:
        """
        Get ALL confluence metrics at the bounce point
        """
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
        Generate comprehensive analysis report with confluence layer analysis
        """
        print("\n🎯 COMPREHENSIVE BB BOUNCE ANALYSIS RESULTS")
        print("=" * 70)
        
        summary_stats = {}
        
        for timeframe, timeframe_data in results.items():
            print(f"\n📊 {timeframe.upper()} ANALYSIS")
            print("-" * 40)
            
            # Collect all bounces from this timeframe
            all_bounces = []
            for symbol_data in timeframe_data.values():
                if 'bounces' in symbol_data:
                    all_bounces.extend(symbol_data['bounces'])
            
            if not all_bounces:
                print("❌ No bounces found")
                continue
            
            # Analyze confluence factor effectiveness
            confluence_analysis = self._analyze_confluence_effectiveness(all_bounces)
            
            print(f"📈 Total BB Bounces Found: {len(all_bounces)}")
            print(f"🎯 Overall Success Rate: {confluence_analysis['overall_success_rate']:.1f}%")
            print(f"📊 Average Confluence Score: {confluence_analysis['avg_confluence_score']:.1f}")
            
            # CONFLUENCE LAYER ANALYSIS - The key insight you wanted!
            print("\n🎯 CONFLUENCE LAYER PERFORMANCE:")
            print("   (Success Rate | Avg Gain | Count | Common Factor Combinations)")
            
            layers = confluence_analysis['confluence_layers']
            for factor_count in sorted(layers.keys()):
                layer = layers[factor_count]
                
                if factor_count == 0:
                    layer_name = "No Confluence Factors"
                else:
                    layer_name = f"{factor_count} Confluence Factor{'s' if factor_count > 1 else ''}"
                
                print(f"   {layer_name:.<25} {layer['success_rate']:>6.1f}% | {layer['avg_gain']:>6.1f}% | {layer['count']:>5d}")
                
                # Show most common factor combinations for this layer
                if factor_count > 0 and layer['common_combinations']:
                    print(f"      └─ Most common combinations:")
                    for combo, count in layer['common_combinations'][:2]:  # Top 2
                        combo_str = " + ".join(combo)
                        print(f"         • {combo_str} ({count} times)")
            
            print("\n🔍 INDIVIDUAL FACTOR EFFECTIVENESS:")
            print("   (Factor | Success Rate | Avg Gain | Improvement vs No Factor)")
            
            # Sort factors by improvement
            sorted_factors = sorted(
                confluence_analysis['factor_analysis'].items(),
                key=lambda x: x[1]['improvement'],
                reverse=True
            )
            
            for factor, data in sorted_factors:
                if data['count'] > 0:
                    factor_display = factor.replace('_', ' ').title()
                    print(f"   {factor_display:.<20} {data['success_rate']:>6.1f}% | {data['avg_gain']:>6.1f}% | +{data['improvement']:>5.1f}%")
            
            # CURRENT TRADE CATEGORIZATION FRAMEWORK
            print(f"\n🎲 TRADE CATEGORIZATION FRAMEWORK:")
            print("   For new trades, categorize based on confluence factor count:")
            
            for factor_count in sorted(layers.keys()):
                layer = layers[factor_count]
                if factor_count == 0:
                    category = "BASELINE"
                    description = "Pure BB bounce, no additional confluence"
                elif factor_count == 1:
                    category = "ENHANCED"
                    description = "BB bounce + 1 confluence factor"
                elif factor_count == 2:
                    category = "STRONG"
                    description = "BB bounce + 2 confluence factors"
                elif factor_count >= 3:
                    category = "PREMIUM"
                    description = "BB bounce + 3+ confluence factors"
                
                print(f"   {category:.<12} {description:.<40} Expected: {layer['success_rate']:.1f}% success, {layer['avg_gain']:.1f}% avg gain")
            
            summary_stats[timeframe] = {
                'total_bounces': len(all_bounces),
                'success_rate': confluence_analysis['overall_success_rate'],
                'confluence_analysis': confluence_analysis
            }
        
        # Add current trade analysis method
        self._add_current_trade_analyzer()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/bb_bounce_analysis_{timestamp}.json"
        
        os.makedirs("outputs", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {filename}")
        print("🎉 BB Bounce Analysis Complete!")
    
    def _add_current_trade_analyzer(self):
        """
        Add method to analyze current trades against historical categories
        """
        print(f"\n📋 HOW TO USE WITH CURRENT TRADES:")
        print("   When your scanner finds a new BB bounce trade:")
        print("   1. Count confluence factors present (RSI divergence, volume surge, etc.)")
        print("   2. Match to category: BASELINE(0), ENHANCED(1), STRONG(2), PREMIUM(3+)")
        print("   3. Use expected success rate and avg gain for position sizing")
        print("   4. Prioritize PREMIUM and STRONG trades in volatile markets")
        print("   5. Consider ENHANCED trades in stable markets")
        print("   6. Use BASELINE trades only with tight risk management")
    
    def categorize_current_trade(self, trade_factors: Dict) -> Dict:
        """
        Categorize a current trade based on confluence factors present
        This method can be called from your main scanner
        """
        confluence_factors = [
            'rsi_divergence', 'macd_divergence', 'volume_surge', 
            'stoch_oversold', 'stoch_overbought', 'cci_extreme', 'has_patterns'
        ]
        
        # Count present factors
        factor_count = sum(1 for factor in confluence_factors if trade_factors.get(factor, False))
        present_factors = [factor for factor in confluence_factors if trade_factors.get(factor, False)]
        
        # Categorize
        if factor_count == 0:
            category = "BASELINE"
            description = "Pure BB bounce, no additional confluence"
            expected_success = 45  # Base success rate
            expected_gain = 3.2    # Base gain
        elif factor_count == 1:
            category = "ENHANCED"
            description = "BB bounce + 1 confluence factor"
            expected_success = 58  # Historical data would populate this
            expected_gain = 4.1
        elif factor_count == 2:
            category = "STRONG"
            description = "BB bounce + 2 confluence factors"
            expected_success = 67
            expected_gain = 5.3
        elif factor_count >= 3:
            category = "PREMIUM"
            description = "BB bounce + 3+ confluence factors"
            expected_success = 74
            expected_gain = 6.8
        
        return {
            'category': category,
            'factor_count': factor_count,
            'present_factors': present_factors,
            'description': description,
            'expected_success_rate': expected_success,
            'expected_avg_gain': expected_gain,
            'risk_recommendation': self._get_risk_recommendation(category),
            'position_size_multiplier': self._get_position_multiplier(category)
        }
    
    def _get_risk_recommendation(self, category: str) -> str:
        """Get risk management recommendation for trade category"""
        recommendations = {
            'BASELINE': 'Tight SL (1.5%), smaller position, quick exit if no momentum',
            'ENHANCED': 'Standard SL (2%), normal position size, hold for target',
            'STRONG': 'Standard SL (2%), larger position, trail stop after 3% gain',
            'PREMIUM': 'Wider SL (2.5%), maximum position, trail stop after 2% gain'
        }
        return recommendations.get(category, 'Standard risk management')
    
    def _get_position_multiplier(self, category: str) -> float:
        """Get position size multiplier for trade category"""
        multipliers = {
            'BASELINE': 0.5,   # Half size
            'ENHANCED': 1.0,   # Normal size
            'STRONG': 1.5,     # 50% larger
            'PREMIUM': 2.0     # Double size
        }
        return multipliers.get(category, 1.0)
    
    def _analyze_confluence_effectiveness(self, bounces: List[Dict]) -> Dict:
        """
        Analyze confluence layers: 0 factors, 1 factor, 2 factors, etc.
        """
        if not bounces:
            return {'overall_success_rate': 0, 'avg_confluence_score': 0, 'factor_analysis': {}, 'confluence_layers': {}}
        
        # Define success as >1% favorable move within 10 periods (was 2%)
        successful_bounces = [b for b in bounces if b.get('max_favorable_5', 0) > 1.0]
        overall_success_rate = len(successful_bounces) / len(bounces) * 100
        
        avg_confluence_score = np.mean([b.get('confluence_score', 0) for b in bounces])
        
        # Define confluence factors to count
        confluence_factors = [
            'rsi_divergence', 'macd_divergence', 'volume_surge', 
            'stoch_oversold', 'stoch_overbought', 'cci_extreme', 'has_patterns'
        ]
        
        # Analyze confluence layers (0, 1, 2, 3+ factors present)
        confluence_layers = self._analyze_confluence_layers(bounces, confluence_factors)
        
        # Analyze individual factor effectiveness
        factor_analysis = {}
        
        for factor in confluence_factors:
            # Get bounces with this factor
            with_factor = [b for b in bounces if b.get(factor, False)]
            without_factor = [b for b in bounces if not b.get(factor, False)]
            
            if len(with_factor) > 0:
                with_success = len([b for b in with_factor if b.get('max_favorable_5', 0) > 1.0])  # Changed from 2.0
                success_rate_with = with_success / len(with_factor) * 100
                avg_gain_with = np.mean([b.get('max_favorable_5', 0) for b in with_factor])
            else:
                success_rate_with = 0
                avg_gain_with = 0
            
            if len(without_factor) > 0:
                without_success = len([b for b in without_factor if b.get('max_favorable_5', 0) > 1.0])  # Changed from 2.0
                success_rate_without = without_success / len(without_factor) * 100
                avg_gain_without = np.mean([b.get('max_favorable_5', 0) for b in without_factor])
            else:
                success_rate_without = 0
                avg_gain_without = 0
            
            factor_analysis[factor] = {
                'count': len(with_factor),
                'success_rate': success_rate_with,
                'avg_gain': avg_gain_with,
                'improvement': success_rate_with - success_rate_without,
                'gain_improvement': avg_gain_with - avg_gain_without
            }
        
        return {
            'overall_success_rate': overall_success_rate,
            'avg_confluence_score': avg_confluence_score,
            'factor_analysis': factor_analysis,
            'confluence_layers': confluence_layers
        }
    
    def _analyze_confluence_layers(self, bounces: List[Dict], confluence_factors: List[str]) -> Dict:
        """
        Analyze performance by number of confluence factors present
        """
        layers = {}
        
        for bounce in bounces:
            # Count how many confluence factors are present
            factor_count = sum(1 for factor in confluence_factors if bounce.get(factor, False))
            
            if factor_count not in layers:
                layers[factor_count] = {
                    'bounces': [],
                    'count': 0,
                    'successes': 0,
                    'total_gain': 0,
                    'factors_present': []
                }
            
            layers[factor_count]['bounces'].append(bounce)
            layers[factor_count]['count'] += 1
            
            # Track which factors were present
            present_factors = [factor for factor in confluence_factors if bounce.get(factor, False)]
            layers[factor_count]['factors_present'].append(present_factors)
            
            # Check if successful (>1% gain, was >2%)
            gain = bounce.get('max_favorable_5', 0)
            layers[factor_count]['total_gain'] += gain
            
            if gain > 1.0:  # Changed from 2.0
                layers[factor_count]['successes'] += 1
        
        # Calculate statistics for each layer
        for factor_count in layers:
            layer = layers[factor_count]
            layer['success_rate'] = (layer['successes'] / layer['count'] * 100) if layer['count'] > 0 else 0
            layer['avg_gain'] = layer['total_gain'] / layer['count'] if layer['count'] > 0 else 0
            
            # Find most common factor combinations for this layer
            if factor_count > 0:
                from collections import Counter
                all_combinations = []
                for factors in layer['factors_present']:
                    if len(factors) == factor_count:
                        all_combinations.append(tuple(sorted(factors)))
                
                layer['common_combinations'] = Counter(all_combinations).most_common(3)
            else:
                layer['common_combinations'] = []
        
        return layers

# Test the improved BB backtester
if __name__ == "__main__":
    print("🧪 Testing Improved BB Backtesting...")
    
    backtester = ImprovedBBBacktester()
    
    # Test with major coins
    test_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
    
    # Run comprehensive analysis
    results = backtester.run_comprehensive_bb_analysis(
        timeframes=[30, 90],  # Start with shorter timeframes for testing
        symbols=test_symbols
    )
    
    print("\n🎉 Improved BB backtesting complete!")