"""
Pattern Analyzer - Main Pattern Detection Controller
Comprehensive candlestick pattern detection with ChatGPT improvements
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import our pattern modules
# Import our pattern modules
try:
    from .candlestick_patterns import CandlestickPatterns
    print("✅ CandlestickPatterns imported successfully")
except ImportError as e:
    print(f"❌ CandlestickPatterns import failed: {e}")
    CandlestickPatterns = None

try:
    from .pattern_quality_scorer import PatternQualityScorer
    print("✅ PatternQualityScorer imported successfully")
except ImportError as e:
    print(f"❌ PatternQualityScorer import failed: {e}")
    PatternQualityScorer = None

try:
    from .pattern_filters import PatternFilters
    print("✅ PatternFilters imported successfully")
except ImportError as e:
    print(f"❌ PatternFilters import failed: {e}")
    PatternFilters = None

try:
    from .risk_reward_calculator import RiskRewardCalculator
    print("✅ RiskRewardCalculator imported successfully")
except ImportError as e:
    print(f"❌ RiskRewardCalculator import failed: {e}")
    RiskRewardCalculator = None

try:
    from .chart_patterns import ChartPatterns
    print("✅ ChartPatterns imported successfully")
except ImportError as e:
    print(f"❌ ChartPatterns import failed: {e}")
    ChartPatterns = None

try:
    from .pattern_performance_tracker import PatternPerformanceTracker
    print("✅ PatternPerformanceTracker imported successfully")
except ImportError as e:
    print(f"❌ PatternPerformanceTracker import failed: {e}")
    PatternPerformanceTracker = None

class PatternAnalyzer:
    """
    Main pattern analysis controller that orchestrates comprehensive pattern detection
    with ChatGPT improvements and integrates seamlessly with existing BB system
    """
    
    def __init__(self):
        """Initialize all pattern analysis components"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all pattern analysis modules
        self.candlestick_detector = CandlestickPatterns() if CandlestickPatterns else None
        self.quality_scorer = PatternQualityScorer() if PatternQualityScorer else self._create_dummy_quality_scorer()
        self.pattern_filters = PatternFilters() if PatternFilters else None
        self.rr_calculator = RiskRewardCalculator() if RiskRewardCalculator else None
        self.chart_detector = ChartPatterns() if ChartPatterns else None
        self.performance_tracker = PatternPerformanceTracker() if PatternPerformanceTracker else None
        
        # Pattern tier definitions
        self.pattern_tiers = {
            'tier_1': ['hammer', 'inverted_hammer', 'shooting_star', 'hanging_man',
                      'bullish_engulfing', 'bearish_engulfing', 'doji', 'piercing_line', 'dark_cloud_cover'],
            'tier_2': ['morning_star', 'evening_star', 'tweezer_top', 'tweezer_bottom',
                      'bullish_harami', 'bearish_harami', 'three_white_soldiers', 'three_black_crows',
                      'spinning_top', 'marubozu_bullish', 'marubozu_bearish'],
            'tier_3': ['abandoned_baby', 'kicking_pattern', 'gravestone_doji', 'dragonfly_doji',
                      'high_wave', 'long_legged_doji', 'belt_hold_bullish', 'belt_hold_bearish']
        }
        # Add this boost_multipliers block here:
        self.boost_multipliers = {
            'tier_1': {'excellent': 15, 'good': 10, 'fair': 5},
            'tier_2': {'excellent': 8, 'good': 5, 'fair': 3},
            'tier_3': {'excellent': 0, 'good': 0, 'fair': 0}
        }


    def _create_dummy_quality_scorer(self):
        """Create dummy quality scorer when import fails"""
        class DummyQualityScorer:
            def calculate_quality(self, *args, **kwargs):
                return 50.0  # Default neutral quality score
        
        return DummyQualityScorer()
    
    def _log_multi_timeframe_analysis(self, symbol: str, patterns_4h: List[Dict], patterns_1h: List[Dict],
                                    filtered_4h: List[Dict], filtered_1h: List[Dict], bb_data: Dict):
        """Log multi-timeframe pattern analysis for performance tracking"""
        try:
            # Log 4H patterns
            if patterns_4h:
                self.performance_tracker.log_pattern_detection(
                    symbol=f"{symbol}_4H",
                    all_patterns=patterns_4h,
                    filtered_patterns=filtered_4h,
                    bb_data=bb_data
                )
            
            # Log 1H patterns
            if patterns_1h:
                self.performance_tracker.log_pattern_detection(
                    symbol=f"{symbol}_1H", 
                    all_patterns=patterns_1h,
                    filtered_patterns=filtered_1h,
                    bb_data=bb_data
                )
                
        except Exception as e:
            self.logger.warning(f"Error in multi-timeframe pattern analysis for {symbol}: {str(e)}")
            # Return default pattern data so analysis can continue
            return self._get_default_pattern_results(symbol)
    
    def _get_default_pattern_results(self, symbol: str) -> Dict:
        """Return default results when pattern analysis fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            
            # Empty results for both timeframes
            'patterns_4h': [],
            'patterns_1h': [],
            'significant_patterns_4h': [],
            'significant_patterns_1h': [],
            'clustering_4h': {'has_clustering': False, 'cluster_boost': 0, 'cluster_count': 0},
            'clustering_1h': {'has_clustering': False, 'cluster_boost': 0, 'cluster_count': 0},
            
            # No confluence
            'timeframe_confluence': {
                'confluence_type': 'NONE',
                'confluence_bonus': 0,
                'conflict_penalty': 0,
                'confluence_strength': 0,
                'confluence_description': 'No patterns detected',
                'trading_recommendation': 'Rely on BB analysis only'
            },
            
            # Zero boosts
            'total_pattern_boost': 0,
            'best_pattern_quality_4h': 0,
            'best_pattern_quality_1h': 0,
            'final_pattern_confidence': 0,
            
            # Combined empty results
            'all_patterns': [],
            'significant_patterns': [],
            'auto_risk_reward': {'risk_reward_ratio': 0, 'auto_sl': 0, 'auto_tp': 0, 'rr_valid': False},
            
            # Excel summary
            'excel_summary': {
                'all_patterns_detected': 'None',
                'patterns_4h': 'None',
                'patterns_1h': 'None',
                'significant_patterns': 'None',
                'timeframe_confluence': 'NONE',
                'confluence_description': 'No patterns detected',
                'trading_recommendation': 'Rely on BB analysis only',
                'total_pattern_boost': 0,
                'risk_reward_ratio': 0,
                'multi_timeframe_analysis': True
            },
            
            'analysis_success': False,
            'multi_timeframe_analysis': True
        }
    
    def get_enhanced_pattern_statistics(self) -> Dict:
        """Get enhanced pattern detection statistics for multi-timeframe analysis"""
        base_stats = self.get_pattern_statistics()
        
        enhanced_stats = {
            **base_stats,
            'multi_timeframe_analysis': True,
            'confluence_types': [
                'STRONG_BULLISH', 'STRONG_BEARISH', 'MODERATE_BULLISH', 'MODERATE_BEARISH',
                'CONFLICTING', 'MIXED', 'PRIMARY_ONLY', 'CONFIRMATION_ONLY', 'NONE'
            ],
            'boost_range': '-20% to +35% (enhanced for multi-timeframe)',
            'timeframe_weighting': {
                '4H_primary': '70% weight (setup detection)',
                '1H_confirmation': '30% weight (entry timing)'
            },
            'confluence_bonuses': {
                'strong_alignment': '+15% confidence',
                'moderate_alignment': '+10% confidence', 
                'mixed_signals': '+5% confidence',
                'conflicting_signals': '-15% penalty'
            }
        }
        
        return enhanced_stats
        
        # Conservative boost system (ChatGPT recommendation)

        
        self.logger.info("Pattern Analyzer initialized with comprehensive detection capabilities")
    
    def analyze_comprehensive_patterns(self, symbol: str, df_4h: pd.DataFrame, 
                                     df_1h: pd.DataFrame, atr_value: float, 
                                     bb_data: Dict) -> Dict:
        """
        Main pattern analysis function - detects patterns on both 4H and 1H timeframes
        with institutional-grade timeframe confluence analysis
        
        Args:
            symbol: Trading symbol
            df_4h: 4-hour OHLCV data (primary setup detection)
            df_1h: 1-hour OHLCV data (entry timing confirmation)
            atr_value: Average True Range value
            bb_data: Bollinger Band analysis results
            
        Returns:
            Comprehensive pattern analysis results with timeframe confluence
        """
        try:
            self.logger.info(f"Starting multi-timeframe pattern analysis for {symbol}")
            
            # Step 1: Primary 4H Pattern Detection
            patterns_4h = self._detect_all_patterns(df_4h, atr_value, timeframe='4H')
            
            # Step 2: 1H Pattern Confirmation Analysis  
            patterns_1h = self._detect_all_patterns(df_1h, atr_value, timeframe='1H')
            
            # Step 3: Apply ChatGPT filters to both timeframes
            filtered_patterns_4h = self._apply_chatgpt_filters(patterns_4h, df_4h, atr_value, '4H')
            filtered_patterns_1h = self._apply_chatgpt_filters(patterns_1h, df_1h, atr_value, '1H')
            
            # Step 4: Analyze Timeframe Confluence
            confluence_data = self._analyze_timeframe_confluence(filtered_patterns_4h, filtered_patterns_1h)
            
            # Step 5: Calculate pattern clustering (within each timeframe)
            clustering_4h = self._analyze_pattern_clustering(filtered_patterns_4h)
            clustering_1h = self._analyze_pattern_clustering(filtered_patterns_1h)
            
            # Step 6: Auto-calculate Risk/Reward
            rr_data = self._calculate_auto_risk_reward(df_4h, bb_data)
            
            # Step 6.5: Chart Pattern Detection (NEW)
            chart_patterns_4h = []
            if self.chart_detector:
                try:
                    chart_results = self.chart_detector.detect_chart_patterns(df_4h, df_1h, atr_value, bb_data)
                    if chart_results.get('analysis_success') and chart_results.get('total_patterns', 0) > 0:
                        chart_patterns_4h = chart_results.get('all_patterns', [])

                except Exception as e:
                    self.logger.warning(f"Chart pattern detection failed: {str(e)}")

            # Step 7: Calculate enhanced pattern boosts with timeframe confluence
            boost_data = self._calculate_enhanced_pattern_boosts(
                filtered_patterns_4h, filtered_patterns_1h, confluence_data, 
                clustering_4h, clustering_1h
            )
            
            # Step 8: Log for performance tracking
            self._log_multi_timeframe_analysis(symbol, patterns_4h, patterns_1h, 
                                             filtered_patterns_4h, filtered_patterns_1h, bb_data)
            
            # Compile comprehensive results
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                
                # 4H Primary Analysis
                'patterns_4h': patterns_4h,
                'significant_patterns_4h': filtered_patterns_4h,
                'clustering_4h': clustering_4h,
                
                # 1H Confirmation Analysis
                'patterns_1h': patterns_1h,
                'significant_patterns_1h': filtered_patterns_1h,
                'clustering_1h': clustering_1h,
                
                # Timeframe Confluence Analysis
                'timeframe_confluence': confluence_data,
                'confluence_bonus': confluence_data['confluence_bonus'],
                'conflict_penalty': confluence_data['conflict_penalty'],
                
                # Enhanced Confidence Scoring
                'total_pattern_boost': boost_data['total_boost'],
                'best_pattern_quality_4h': boost_data['best_quality_4h'],
                'best_pattern_quality_1h': boost_data['best_quality_1h'],
                'final_pattern_confidence': boost_data['final_confidence'],
                
                # Combined Pattern Intelligence
                'all_patterns': patterns_4h + patterns_1h,  # All detected patterns
                'significant_patterns': filtered_patterns_4h + filtered_patterns_1h,  # All significant
                
                # Auto Risk/Reward
                'auto_risk_reward': rr_data,
                
                # Excel-friendly summaries
                'excel_summary': self._create_enhanced_excel_summary(
                    patterns_4h, patterns_1h, filtered_patterns_4h, filtered_patterns_1h,
                    confluence_data, clustering_4h, clustering_1h, rr_data, boost_data, chart_patterns_4h
                ),
                
                # Analysis success
                'analysis_success': True,
                'multi_timeframe_analysis': True
            }
            
            self.logger.info(f"Multi-timeframe pattern analysis complete for {symbol}: "
                           f"4H patterns: {len(patterns_4h)}, 1H patterns: {len(patterns_1h)}, "
                           f"Confluence: {confluence_data['confluence_type']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe pattern analysis for {symbol}: {str(e)}")
            return self._get_default_pattern_results(symbol)
    
    def _detect_all_patterns(self, df: pd.DataFrame, atr_value: float, timeframe: str = '4H') -> List[Dict]:
        """Detect all candlestick patterns using comprehensive detection"""
          
        # CHECK IF CANDLESTICK DETECTOR IS AVAILABLE - ADD THIS CHECK HERE
        if self.candlestick_detector is None:
            self.logger.warning(f"Candlestick detector not available for {timeframe} analysis")
            return []
    
        all_patterns = []
        
        # Get the last few candles for pattern detection
        recent_candles = df.tail(10)  # Last 10 candles for pattern context
        
        # Detect each pattern type
        for tier, pattern_list in self.pattern_tiers.items():
            for pattern_name in pattern_list:
                try:
                    pattern_data = self.candlestick_detector.detect_pattern(
                        recent_candles, pattern_name, atr_value
                    )
                    
                    if pattern_data:
                        # Calculate quality score
                        quality_score = self.quality_scorer.calculate_quality(
                            pattern_data, recent_candles, atr_value
                        )
                        
                        pattern_info = {
                            'pattern_name': pattern_name,
                            'tier': tier,
                            'timeframe': timeframe,  # Track which timeframe detected this
                            'quality_score': quality_score,
                            'pattern_data': pattern_data,
                            'candle_index': pattern_data.get('candle_index', -1),
                            'detection_confidence': pattern_data.get('confidence', 0),
                            'direction': pattern_data.get('direction', 'neutral')
                        }
                        
                        all_patterns.append(pattern_info)
                        
                except Exception as e:
                    self.logger.warning(f"Error detecting {pattern_name} on {timeframe}: {str(e)}")
                    continue
        
        return all_patterns
    
    def _apply_chatgpt_filters(self, patterns: List[Dict], df: pd.DataFrame, 
                             atr_value: float, timeframe: str) -> List[Dict]:
        """Return all patterns without filtering - let scoring handle quality assessment"""
        
        # No filtering - return all detected patterns
        filtered_patterns = patterns
        
        # Add metadata to all patterns
        for pattern in filtered_patterns:
            pattern['filters_passed'] = True
            pattern['timeframe'] = timeframe
            pattern['atr_significance'] = pattern['pattern_data'].get('atr_multiple', 0)
            pattern['volume_confirmation'] = pattern['pattern_data'].get('volume_confirmed', False)
        
        self.logger.info(f"Pattern filtering disabled - showing all patterns for {timeframe}: {len(filtered_patterns)} patterns")
        
        return filtered_patterns
    
    def _analyze_timeframe_confluence(self, patterns_4h: List[Dict], patterns_1h: List[Dict]) -> Dict:
        """
        Analyze timeframe confluence between 4H and 1H patterns
        Professional multi-timeframe pattern analysis
        """
        try:
            # Extract pattern directions for analysis
            directions_4h = [p.get('direction', 'neutral') for p in patterns_4h if p.get('direction') != 'neutral']
            directions_1h = [p.get('direction', 'neutral') for p in patterns_1h if p.get('direction') != 'neutral']
            
            # Count directional patterns
            bullish_4h = directions_4h.count('bullish')
            bearish_4h = directions_4h.count('bearish')
            bullish_1h = directions_1h.count('bullish')
            bearish_1h = directions_1h.count('bearish')
            
            # Determine confluence type
            confluence_type = 'NONE'
            confluence_bonus = 0
            conflict_penalty = 0
            confluence_strength = 0
            
            # Strong Bullish Confluence
            if bullish_4h > 0 and bullish_1h > 0 and bearish_4h == 0 and bearish_1h == 0:
                confluence_type = 'STRONG_BULLISH'
                confluence_bonus = 15  # +15% for strong alignment
                confluence_strength = min(bullish_4h + bullish_1h, 5) * 3  # Up to 15 points
            
            # Strong Bearish Confluence  
            elif bearish_4h > 0 and bearish_1h > 0 and bullish_4h == 0 and bullish_1h == 0:
                confluence_type = 'STRONG_BEARISH'
                confluence_bonus = 15  # +15% for strong alignment
                confluence_strength = min(bearish_4h + bearish_1h, 5) * 3  # Up to 15 points
            
            # Moderate Bullish Confluence
            elif bullish_4h > bearish_4h and bullish_1h > bearish_1h:
                confluence_type = 'MODERATE_BULLISH'
                confluence_bonus = 10  # +10% for moderate alignment
                confluence_strength = 10
            
            # Moderate Bearish Confluence
            elif bearish_4h > bullish_4h and bearish_1h > bullish_1h:
                confluence_type = 'MODERATE_BEARISH'
                confluence_bonus = 10  # +10% for moderate alignment
                confluence_strength = 10
            
            # Conflicting Signals (4H vs 1H opposing)
            elif (bullish_4h > 0 and bearish_1h > 0) or (bearish_4h > 0 and bullish_1h > 0):
                confluence_type = 'CONFLICTING'
                conflict_penalty = -15  # -15% for opposing signals
                confluence_strength = -15
            
            # Weak or Mixed Signals
            elif len(directions_4h) > 0 and len(directions_1h) > 0:
                confluence_type = 'MIXED'
                confluence_bonus = 5  # Small +5% for having patterns on both timeframes
                confluence_strength = 5
            
            # Single Timeframe Only
            elif len(directions_4h) > 0 and len(directions_1h) == 0:
                confluence_type = 'PRIMARY_ONLY'
                confluence_strength = 0  # No bonus or penalty
            
            elif len(directions_4h) == 0 and len(directions_1h) > 0:
                confluence_type = 'CONFIRMATION_ONLY'
                confluence_strength = 0  # No bonus or penalty
            
            # Calculate pattern matching score
            pattern_matches = self._calculate_pattern_matches(patterns_4h, patterns_1h)
            
            confluence_data = {
                'confluence_type': confluence_type,
                'confluence_bonus': confluence_bonus,
                'conflict_penalty': conflict_penalty,
                'confluence_strength': confluence_strength,
                'pattern_matches': pattern_matches,
                
                # Detailed breakdown
                'patterns_4h_count': len(patterns_4h),
                'patterns_1h_count': len(patterns_1h),
                'bullish_4h': bullish_4h,
                'bearish_4h': bearish_4h,
                'bullish_1h': bullish_1h,
                'bearish_1h': bearish_1h,
                
                # Human-readable summary
                'confluence_description': self._get_confluence_description(confluence_type, confluence_strength),
                'trading_recommendation': self._get_trading_recommendation(confluence_type)
            }
            
            self.logger.info(f"Timeframe confluence: {confluence_type} "
                           f"(4H: {len(patterns_4h)} patterns, 1H: {len(patterns_1h)} patterns)")
            
            return confluence_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe confluence: {str(e)}")
            return {
                'confluence_type': 'ERROR',
                'confluence_bonus': 0,
                'conflict_penalty': 0,
                'confluence_strength': 0,
                'confluence_description': 'Confluence analysis failed',
                'trading_recommendation': 'Use caution'
            }
    
    def _calculate_pattern_matches(self, patterns_4h: List[Dict], patterns_1h: List[Dict]) -> Dict:
        """Calculate how many patterns match between timeframes"""
        matches = {
            'exact_matches': 0,  # Same pattern name on both timeframes
            'directional_matches': 0,  # Same direction but different patterns
            'tier_matches': 0  # Same tier patterns
        }
        
        for p4h in patterns_4h:
            for p1h in patterns_1h:
                # Exact pattern match
                if p4h['pattern_name'] == p1h['pattern_name']:
                    matches['exact_matches'] += 1
                
                # Directional match
                elif p4h.get('direction') == p1h.get('direction') and p4h.get('direction') != 'neutral':
                    matches['directional_matches'] += 1
                
                # Tier match (both high quality)
                elif p4h['tier'] == p1h['tier'] and p4h['tier'] == 'tier_1':
                    matches['tier_matches'] += 1
        
        return matches
    
    def _analyze_pattern_clustering(self, patterns: List[Dict]) -> Dict:
        """Analyze pattern clustering for additional confidence"""
        if len(patterns) < 2:
            return {'has_clustering': False, 'cluster_boost': 0, 'cluster_count': len(patterns)}
        
        # Check if patterns are within 3 candles of each other
        pattern_indices = [p['candle_index'] for p in patterns]
        max_spread = max(pattern_indices) - min(pattern_indices)
        
        clustering_data = {
            'has_clustering': max_spread <= 3,
            'cluster_boost': 10 if max_spread <= 3 else 0,  # ChatGPT's +10% clustering boost
            'cluster_count': len(patterns),
            'pattern_spread': max_spread,
            'clustered_patterns': [p['pattern_name'] for p in patterns] if max_spread <= 3 else []
        }
        
        return clustering_data
    
    def _calculate_auto_risk_reward(self, df: pd.DataFrame, bb_data: Dict) -> Dict:
        """Calculate automatic risk/reward based on support/resistance"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Use RR calculator to find nearest support/resistance
            rr_data = self.rr_calculator.calculate_auto_rr(df, current_price, bb_data)
            
            return rr_data
            
        except Exception as e:
            self.logger.error(f"Error calculating auto R/R: {str(e)}")
            return {'risk_reward_ratio': 0, 'auto_sl': 0, 'auto_tp': 0, 'rr_valid': False}
    
    
    def _calculate_enhanced_pattern_boosts(self, patterns_4h: List[Dict], patterns_1h: List[Dict], 
                                         confluence_data: Dict, clustering_4h: Dict, clustering_1h: Dict) -> Dict:
        """Calculate enhanced pattern boosts with timeframe confluence"""
        if not patterns_4h and not patterns_1h:
            return {'total_boost': 0, 'best_quality_4h': 0, 'best_quality_1h': 0, 'final_confidence': 0}
        
        # Calculate base boosts for each timeframe
        boost_4h = self._calculate_timeframe_boost(patterns_4h, clustering_4h, primary=True)
        boost_1h = self._calculate_timeframe_boost(patterns_1h, clustering_1h, primary=False)
        
        # Base pattern boost (weighted - 4H primary, 1H confirmation)
        base_boost = (boost_4h['boost'] * 0.7) + (boost_1h['boost'] * 0.3)
        
        # Add confluence bonus/penalty
        confluence_adjustment = confluence_data['confluence_bonus'] + confluence_data['conflict_penalty']
        
        # Calculate final boost
        total_boost = base_boost + confluence_adjustment
        
        # Cap total boost at reasonable level (increased from 25% to 35% for multi-timeframe)
        total_boost = min(max(total_boost, -20), 35)  # Range: -20% to +35%
        
        # Calculate final confidence score
        best_4h_quality = boost_4h['best_quality']
        best_1h_quality = boost_1h['best_quality']
        
        # Weighted confidence (4H primary)
        base_confidence = (best_4h_quality * 0.7) + (best_1h_quality * 0.3)
        final_confidence = min(base_confidence + confluence_data['confluence_strength'], 100)
        
        return {
            'total_boost': round(total_boost, 1),
            'base_boost_4h': boost_4h.get('boost', 0),
            'base_boost_1h': boost_1h.get('boost', 0),
            'confluence_adjustment': confluence_adjustment,
            'best_quality_4h': best_4h_quality,
            'best_quality_1h': best_1h_quality,
            'final_confidence': round(final_confidence, 1),
            'boost_breakdown': {
                '4h_patterns': boost_4h['boost'],
                '1h_patterns': boost_1h['boost'], 
                'confluence': confluence_adjustment,
                'total': total_boost
            }
        }
    
    def _calculate_timeframe_boost(self, patterns: List[Dict], clustering_data: Dict, primary: bool = True) -> Dict:
        """Calculate boost for a specific timeframe"""
        
        total_boost = 0
        best_quality = 0
        
        # Weight multiplier (4H gets full weight, 1H gets reduced weight)
        weight_multiplier = 1.0 if primary else 0.6
        
        # Calculate boost for each significant pattern
        for pattern in patterns:
            tier = pattern['tier']
            quality = pattern['quality_score']
            
            # Determine quality category
            if quality >= 85:
                quality_cat = 'excellent'
            elif quality >= 70:
                quality_cat = 'good'
            elif quality >= 60:
                quality_cat = 'fair'
            else:
                continue  # Below 60% quality gets no boost
            
            # Apply conservative boost with timeframe weighting
            pattern_boost = self.boost_multipliers[tier][quality_cat] * weight_multiplier
            total_boost += pattern_boost
            
            # Track best quality
            best_quality = max(best_quality, quality)
        
        # Add clustering boost
        total_boost += clustering_data['cluster_boost'] * weight_multiplier
        
        return {'boost': total_boost, 'best_quality': best_quality}
    
    def _get_confluence_description(self, confluence_type: str, strength: int) -> str:
        """Get human-readable confluence description"""
        descriptions = {
            'STRONG_BULLISH': f'Strong bullish alignment across timeframes (+{strength}% confidence)',
            'STRONG_BEARISH': f'Strong bearish alignment across timeframes (+{strength}% confidence)',
            'MODERATE_BULLISH': f'Moderate bullish confluence (+{strength}% confidence)',
            'MODERATE_BEARISH': f'Moderate bearish confluence (+{strength}% confidence)',
            'CONFLICTING': f'Conflicting signals between timeframes ({strength}% penalty)',
            'MIXED': f'Mixed signals with some alignment (+{strength}% confidence)',
            'PRIMARY_ONLY': 'Patterns only on 4H timeframe',
            'CONFIRMATION_ONLY': 'Patterns only on 1H timeframe',
            'NONE': 'No significant patterns detected',
            'ERROR': 'Confluence analysis failed'
        }
        return descriptions.get(confluence_type, 'Unknown confluence type')
    
    def _get_trading_recommendation(self, confluence_type: str) -> str:
        """Get trading recommendation based on confluence"""
        recommendations = {
            'STRONG_BULLISH': 'HIGH CONFIDENCE - Strong bullish setup',
            'STRONG_BEARISH': 'HIGH CONFIDENCE - Strong bearish setup', 
            'MODERATE_BULLISH': 'MODERATE CONFIDENCE - Bullish bias with confirmation',
            'MODERATE_BEARISH': 'MODERATE CONFIDENCE - Bearish bias with confirmation',
            'CONFLICTING': 'AVOID TRADE - Wait for timeframe alignment',
            'MIXED': 'CAUTION - Mixed signals, use smaller position size',
            'PRIMARY_ONLY': 'STANDARD CONFIDENCE - Single timeframe signal',
            'CONFIRMATION_ONLY': 'LOW CONFIDENCE - Only 1H patterns detected',
            'NONE': 'NO PATTERN SIGNAL - Rely on BB analysis only'
        }
        return recommendations.get(confluence_type, 'Use caution')
    
    def _create_enhanced_excel_summary(self, patterns_4h: List[Dict], patterns_1h: List[Dict],
                                     filtered_4h: List[Dict], filtered_1h: List[Dict],
                                     confluence_data: Dict, clustering_4h: Dict, clustering_1h: Dict,
                                     rr_data: Dict, boost_data: Dict, chart_patterns: List[Dict] = []) -> Dict:
        """Create enhanced Excel-friendly summary with multi-timeframe data"""
        return {
            # Overall pattern summary
            'all_patterns_detected': ', '.join([p['pattern_name'] for p in patterns_4h + patterns_1h]),
            'patterns_4h': ', '.join([p['pattern_name'] for p in patterns_4h]),
            'patterns_1h': ', '.join([p['pattern_name'] for p in patterns_1h]),
            
            # Significant patterns
            'significant_4h': ', '.join([p['pattern_name'] for p in filtered_4h]),
            'significant_1h': ', '.join([p['pattern_name'] for p in filtered_1h]),
            
            # Quality metrics
            'pattern_quality_best_4h': boost_data['best_quality_4h'],
            'pattern_quality_best_1h': boost_data['best_quality_1h'],
            'pattern_quality_overall': boost_data['final_confidence'],
            
            # Clustering information
            'pattern_clustering_4h': 'Yes' if clustering_4h['has_clustering'] else 'No',
            'pattern_clustering_1h': 'Yes' if clustering_1h['has_clustering'] else 'No',
            'clustered_patterns_4h': ', '.join(clustering_4h.get('clustered_patterns', [])),
            'clustered_patterns_1h': ', '.join(clustering_1h.get('clustered_patterns', [])),
            
            # Timeframe confluence
            'timeframe_confluence': confluence_data['confluence_type'],
            'confluence_strength': confluence_data['confluence_strength'],
            'confluence_description': confluence_data['confluence_description'],
            'trading_recommendation': confluence_data['trading_recommendation'],
            
            # Candlestick pattern data fixes (7 fields)
            'significant_patterns': ', '.join([p['pattern_name'] for p in filtered_4h + filtered_1h]) if (filtered_4h or filtered_1h) else 'None',
            'pattern_confidence': boost_data.get('final_confidence', 0),
            'pattern_boost': boost_data.get('total_boost', 0),
            'pattern_quality_best': max(boost_data.get('best_quality_4h', 0), boost_data.get('best_quality_1h', 0)),
            'auto_stop_loss': rr_data.get('auto_sl', 0),
            'auto_take_profit': rr_data.get('auto_tp', 0),
            'risk_reward_ratio': rr_data.get('risk_reward_ratio', 0),

            # Chart pattern data (add these 4 lines)
            'chart_patterns_detected': ', '.join([p['pattern_name'] for p in chart_patterns]) if chart_patterns else 'None',
            'best_chart_pattern': chart_patterns[0]['pattern_name'] if chart_patterns else 'None',
            'chart_pattern_confidence': chart_patterns[0].get('confidence', 0) if chart_patterns else 0,
            'chart_pattern_target': chart_patterns[0].get('target_price', 0) if chart_patterns else 0,

            # Enhanced boost calculation
            'total_pattern_boost': boost_data['total_boost'],
            'boost_4h': boost_data.get('base_boost_4h', 0),
            'boost_1h': boost_data.get('base_boost_1h', 0),
            'confluence_adjustment': boost_data.get('confluence_adjustment', 0),
            
            # Risk/Reward
            'auto_stop_loss': rr_data.get('auto_sl', 0),
            'auto_take_profit': rr_data.get('auto_tp', 0),
            'risk_reward_ratio': rr_data.get('risk_reward_ratio', 0),
            'rr_valid': rr_data.get('rr_valid', False),
            
            # Multi-timeframe flag
            'multi_timeframe_analysis': True,
            'best_quality': max(boost_data['best_quality_4h'], boost_data['best_quality_1h']),
            'confidence': min(max(boost_data['best_quality_4h'], boost_data['best_quality_1h']) + clustering_4h.get('cluster_boost', 0) + clustering_1h.get('cluster_boost', 0), 100),
            'chart_patterns_detected': ', '.join([p['pattern_name'] for p in chart_patterns]) if chart_patterns else 'None',
            'best_chart_pattern': chart_patterns[0]['pattern_name'] if chart_patterns else 'None',
            'chart_pattern_confidence': chart_patterns[0]['final_confidence'] if chart_patterns else 0,
            'chart_pattern_target': chart_patterns[0].get('target_price', 0) if chart_patterns else 0,
        }
    
    def _categorize_patterns_by_tier(self, patterns: List[Dict]) -> Dict:
        """Categorize patterns by tier for analysis"""
        tier_categorization = {'tier_1': [], 'tier_2': [], 'tier_3': []}
        
        for pattern in patterns:
            tier = pattern['tier']
            if tier in tier_categorization:
                tier_categorization[tier].append(pattern['pattern_name'])
        
        return tier_categorization
    
    def _create_excel_summary(self, all_patterns: List[Dict], filtered_patterns: List[Dict],
                            clustering_data: Dict, rr_data: Dict, boost_data: Dict) -> Dict:
        """Create Excel-friendly summary data"""
        return {
            'all_patterns_detected': ', '.join([p['pattern_name'] for p in all_patterns]),
            'tier_1_patterns': ', '.join([p['pattern_name'] for p in all_patterns if p['tier'] == 'tier_1']),
            'tier_2_patterns': ', '.join([p['pattern_name'] for p in all_patterns if p['tier'] == 'tier_2']),
            'tier_3_patterns': ', '.join([p['pattern_name'] for p in all_patterns if p['tier'] == 'tier_3']),
            'significant_patterns': ', '.join([p['pattern_name'] for p in filtered_patterns]),
            'pattern_quality_best': boost_data['best_quality'],
            'pattern_clustering': 'Yes' if clustering_data['has_clustering'] else 'No',
            'clustered_patterns': ', '.join(clustering_data.get('clustered_patterns', [])),
            'total_pattern_boost': boost_data['total_boost'],
            'auto_stop_loss': rr_data.get('auto_sl', 0),
            'auto_take_profit': rr_data.get('auto_tp', 0),
            'risk_reward_ratio': rr_data.get('risk_reward_ratio', 0),
            'rr_valid': rr_data.get('rr_valid', False)
        }
    
    def _log_pattern_analysis(self, symbol: str, all_patterns: List[Dict], 
                            filtered_patterns: List[Dict], bb_data: Dict):
        """Log pattern analysis for performance tracking"""
        try:
            self.performance_tracker.log_pattern_detection(
                symbol=symbol,
                all_patterns=all_patterns,
                filtered_patterns=filtered_patterns,
                bb_data=bb_data
            )
        except Exception as e:
            self.logger.warning(f"Error logging pattern analysis: {str(e)}")
    
    def _get_default_pattern_results(self, symbol: str) -> Dict:
        """Return default results when pattern analysis fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'all_patterns': [],
            'patterns_by_tier': {'tier_1': [], 'tier_2': [], 'tier_3': []},
            'significant_patterns': [],
            'pattern_clustering': {'has_clustering': False, 'cluster_boost': 0, 'cluster_count': 0},
            'total_pattern_boost': 0,
            'best_pattern_quality': 0,
            'pattern_confidence': 0,
            'auto_risk_reward': {'risk_reward_ratio': 0, 'auto_sl': 0, 'auto_tp': 0, 'rr_valid': False},
            'excel_summary': {
                'all_patterns_detected': 'None',
                'significant_patterns': 'None',
                'pattern_quality_best': 0,
                'pattern_clustering': 'No',
                'total_pattern_boost': 0,
                'risk_reward_ratio': 0
            },
            'analysis_success': False
        }
    
    def get_pattern_statistics(self) -> Dict:
        """Get pattern detection statistics for monitoring"""
        return {
            'total_patterns_detected': len(self.pattern_tiers['tier_1']) + 
                                     len(self.pattern_tiers['tier_2']) + 
                                     len(self.pattern_tiers['tier_3']),
            'tier_1_count': len(self.pattern_tiers['tier_1']),
            'tier_2_count': len(self.pattern_tiers['tier_2']),
            'tier_3_count': len(self.pattern_tiers['tier_3']),
            'boost_system': 'Conservative (ChatGPT optimized)',
            'filters_active': ['ATR Significance', 'Volume Confirmation', 'Minimum Range', 'Pattern Clustering']
        }