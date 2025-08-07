#!/usr/bin/env python3
"""
Crypto BB Bounce Scanner - Main Orchestrator
Modular architecture for clean, maintainable code
Enhanced with Market Regime Intelligence
"""

# ULTRA-AGGRESSIVE NUMPY 2.x COMPATIBILITY PATCH - MUST BE FIRST
import sys
import numpy as np

# Patch NumPy immediately before any other imports
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if 'NaN' not in np.__dict__:
    np.__dict__['NaN'] = np.nan
if hasattr(np, '__all__') and 'NaN' not in np.__all__:
    np.__all__.append('NaN')

print("✅ NumPy 2.x compatibility patch applied at startup")
from trade_logger import TradeLogger

# Telegram notifications
try:
    from telegram_alerts import TelegramNotifier
    TELEGRAM_AVAILABLE = True
except ImportError:
    print("⚠️ Telegram alerts not available - telegram_alerts.py not found")
    TELEGRAM_AVAILABLE = False

import argparse
import os
import sys
import logging
import warnings
import pandas as pd
from datetime import datetime

# Suppress pandas_ta deprecation warnings for cleaner output
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Add the current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.data_fetcher import MarketDataFetcher
    from modules.bb_detector import BBDetector
    from modules.technical_analyzer import TechnicalAnalyzer
    from modules.sentiment_analyzer import SentimentAnalyzer, MarketSentimentAnalyzer
    from modules.risk_manager import RiskManager
    from modules.output_generator import OutputGenerator
    # NEW: Market Regime Analyzer import (ONLY ADDITION)
    from modules.market_regime_analyzer import MarketRegimeAnalyzer
    from modules.market_regime_enhanced import create_enhanced_regime_analyzer, format_enhanced_regime_output
    # Add this import
    from modules.improved_bb_backtest import ComprehensiveBBBacktest
    # In imports section
    from historical_intelligence import HistoricalIntelligence, EnhancedOutputGenerator
    # Minimal Confidence Module import
    from modules.minimal_confidence_module import MinimalConfidenceModule, enhance_all_trades_with_confidence
    from modules.market_metadata_enricher import MarketMetadataEnricher
    from config import *
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all module files are in the 'modules' folder")
    sys.exit(1)

class ModularBBScanner:
    """Main orchestrator for the modular BB bounce scanner"""
    
    def __init__(self):
        # Initialize all modules (EXISTING - UNCHANGED)
        self.data_fetcher = MarketDataFetcher()
        self.bb_detector = BBDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_sentiment_analyzer = MarketSentimentAnalyzer()
        self.risk_manager = RiskManager()
        self.output_generator = OutputGenerator()
        
        # NEW: Market Regime Analyzer initialization (ONLY ADDITION)
        self.regime_analyzer = MarketRegimeAnalyzer(self.data_fetcher, self.sentiment_analyzer)
        
        # In __init__ method  
        self.historical_intelligence = HistoricalIntelligence(
            self.data_fetcher, self.technical_analyzer, self.bb_detector
        )
        self.enhanced_output = EnhancedOutputGenerator()
        
        # Add this line in your __init__ method
        self.market_analyzer = ComprehensiveBBBacktest()

        # New Market Metadata Enricher:
        self.metadata_enricher = MarketMetadataEnricher()

        # NEW: Confidence Module
        self.confidence_module = MinimalConfidenceModule()
        
        # NEW: Telegram Notifier
        if TELEGRAM_AVAILABLE:
            try:
                self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
                print("✅ Telegram notifications enabled")
            except Exception as e:
                print(f"⚠️ Telegram setup failed: {e}")
                self.telegram = None
        else:
            self.telegram = None
        
        # Setup logging (EXISTING - UNCHANGED)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup organized logging (EXISTING - UNCHANGED)"""
        # Create logs directory
        os.makedirs("outputs/logs", exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"outputs/logs/bb_scanner_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    async def run_market_regime_analysis(self):
        """NEW: Run comprehensive market regime analysis ONCE per scan"""
        try:
            print("\n" + "="*80)
            print("🌊 MARKET REGIME ANALYSIS - Comprehensive Market Intelligence")
            print("="*80)
            
            # Get ETH data as market proxy (ETH represents alt market best)
            print("📡 Fetching ETH data for market analysis...")
            eth_df = self.data_fetcher.fetch_ohlcv('binance', 'ETH', '4h')
            
            if eth_df is None or len(eth_df) < 50:
                print("⚠️  Could not fetch ETH data for market analysis - using defaults")
                return None
                
            # Run comprehensive market regime analysis
            print("🧠 Analyzing market regime with 9-layer intelligence (including futures data)...")
            enhanced_analyzer = await create_enhanced_regime_analyzer(self.regime_analyzer)
            market_regime = await enhanced_analyzer.analyze_enhanced_regime(eth_df)
            
            # Display the 6-line market intelligence summary
            print("\n📊 CURRENT MARKET REGIME:")
            print("-" * 80)
            regime_display = self.regime_analyzer.format_regime_display(market_regime)
            print(regime_display)
            print("-" * 80)
            
            # Extract key metrics for use in trades
            position_multiplier = market_regime.get('position_multiplier', 1.0)
            regime_confidence = market_regime.get('regime_confidence', 50)
            bb_suitability = market_regime.get('bb_suitability', 'FAIR')
            
            print(f"💰 Position Sizing: {position_multiplier:.2f}x multiplier will be applied to all trades")
            print(f"🎯 BB Strategy Suitability: {bb_suitability} ({regime_confidence:.1f}% confidence)")
            
            return market_regime
            
        except Exception as e:
            self.logger.error(f"Market regime analysis failed: {str(e)}")
            print("⚠️  Market regime analysis failed - continuing with standard analysis")
            return None
        
    def analyze_coin_comprehensive(self, symbol: str, market_regime=None) -> list:
        """Comprehensive analysis of a single coin across all exchanges (ENHANCED with regime data)"""
        all_analyses = []
        
        # NEW: Extract position multiplier from market regime
        position_multiplier = market_regime.get('position_multiplier', 1.0) if market_regime else 1.0
        
        for exchange_name in self.data_fetcher.get_available_exchanges():
            try:
                # Step 1: Fetch market data (EXISTING - UNCHANGED)
                df = self.data_fetcher.fetch_ohlcv(exchange_name, symbol, '4h')
                if df is None or len(df) < 50:
                    continue
                
                # Step 2: BB Detection (EXISTING - UNCHANGED)
                bb_analysis = self.bb_detector.analyze_bb_setup(df)
                # print(f"DEBUG: {symbol}/{exchange_name} BB setup: {bb_analysis['setup_type']}")
                # --- ML DATASET FIX: Always create a result for every coin ---
                last_candle = df.iloc[-1]

                # Initialize all variables to default values
                probability = 0
                risk_pct = 0
                gain_pct = 0
                div_info = {'detected': False, 'strength': '', 'confidence': 0, 'indicators': []}
                confirmations = {'volume_confirmation': False, 'momentum_alignment': False, 'risk_reward_acceptable': False}
                pattern_data = None
                rr_data = {}

                # Only calculate these if there is a BB setup
                if bb_analysis['setup_type'] != 'NONE':
                    # Step 2.5: Pattern Recognition Analysis (SUPPLEMENTARY DATA)
                    try:
                        from modules.pattern_analyzer import PatternAnalyzer
                        pattern_analyzer = PatternAnalyzer()
                        # Get 1H data for multi-timeframe analysis
                        df_1h = self.data_fetcher.fetch_ohlcv(exchange_name, symbol, '1h')
                        if df_1h is not None and len(df_1h) >= 50:
                            # Calculate ATR for pattern significance
                            atr_value = self._calculate_atr(df, period=14)
                            # Comprehensive multi-timeframe pattern analysis
                            pattern_data = pattern_analyzer.analyze_comprehensive_patterns(
                                symbol, df, df_1h, atr_value, bb_analysis
                            )
                            # Enhance BB result with pattern data
                            bb_analysis['pattern_analysis'] = pattern_data
                            bb_analysis['pattern_boost'] = pattern_data.get('excel_summary', {}).get('total_pattern_boost', 0)
                            bb_analysis['patterns_detected'] = pattern_data.get('excel_summary', {}).get('all_patterns_detected', 'None')
                            bb_analysis['pattern_confidence'] = pattern_data.get('excel_summary', {}).get('final_pattern_confidence', 0)
                            self.logger.info(f"Pattern analysis for {symbol}: {pattern_data.get('excel_summary', {}).get('all_patterns_detected', 'None')} "
                                           f"(Boost: {pattern_data.get('excel_summary', {}).get('total_pattern_boost', 0)}%)")
                        else:
                            # No 1H data available
                            bb_analysis['pattern_analysis'] = None
                            bb_analysis['pattern_boost'] = 0
                            bb_analysis['patterns_detected'] = 'No 1H data'
                            bb_analysis['pattern_confidence'] = 0
                    except Exception as e:
                        # Graceful fallback - BB analysis continues unaffected
                        self.logger.warning(f"Pattern analysis failed for {symbol}: {str(e)}")
                        bb_analysis['pattern_analysis'] = None
                        bb_analysis['pattern_boost'] = 0
                        bb_analysis['patterns_detected'] = 'Analysis failed'
                        bb_analysis['pattern_confidence'] = 0

                    # Step 3: Technical Analysis (EXISTING - UNCHANGED)
                    has_1h_confirmation = self.technical_analyzer.get_1h_confirmation(
                        exchange_name, symbol, bb_analysis['setup_type'], self.data_fetcher
                    )
                    # Validate current price (EXISTING - UNCHANGED)
                    price_valid, current_price = self.data_fetcher.validate_current_price(
                        exchange_name, symbol, bb_analysis['entry']
                    )
                    # Update entry price (EXISTING - UNCHANGED)
                    bb_analysis['entry'] = current_price if price_valid else bb_analysis['entry']

                    # UPDATED: Enhanced divergence detection (EXISTING - UNCHANGED)
                    bull_divergence = self.technical_analyzer.detect_enhanced_bullish_divergence(df)
                    bear_divergence = self.technical_analyzer.detect_enhanced_bearish_divergence(df)

                    # Step 4: Risk Assessment (EXISTING - UNCHANGED)
                    probability, confirmations = self.risk_manager.calculate_comprehensive_probability(
                        df, bb_analysis, bull_divergence, bear_divergence
                    )
                    # Calculate risk_pct and gain_pct
                    risk_pct = abs((bb_analysis['entry'] - bb_analysis['stop']) / bb_analysis['entry'] * 100) if bb_analysis['entry'] else 0
                    gain_pct = abs((bb_analysis['target1'] - bb_analysis['entry']) / bb_analysis['entry'] * 100) if bb_analysis['entry'] else 0

                    # Determine divergence info based on setup type
                    if bb_analysis['setup_type'] == 'LONG':
                        div_info = bull_divergence
                    else:
                        div_info = bear_divergence

                    # Extract risk/reward data from pattern analysis
                    rr_data = pattern_data.get('auto_risk_reward', {}) if pattern_data else {}

                # In analyze_coin_comprehensive method (after BB analysis)
                if bb_analysis['setup_type'] != 'NONE':
                    historical_data = self.historical_intelligence.analyze_historical_performance(
                        symbol, bb_analysis
                    )
                    self.enhanced_output.display_enhanced_trade_analysis(
                        symbol, bb_analysis, historical_data
                    )

                # Always create a basic result record for ML training
                result = {
                    # Basic coin data (ALWAYS COLLECTED)
                    'symbol': symbol,
                    'exchange': exchange_name,
                    'setup_type': bb_analysis.get('setup_type', 'NONE'),
                    'bb_score': bb_analysis.get('bb_score', 0),
                    'setup_quality': bb_analysis.get('setup_quality', 'None'),
                    'timestamp': datetime.now(),
                    # Debug: Check what bb_score is stored
                    # Technical data (ALWAYS AVAILABLE for ML)
                    'bb_pct': round(last_candle.get('bb_pct', 0), 3),
                    'rsi': round(last_candle.get('rsi', 0), 1),
                    'volume_ratio': round(last_candle.get('volume_ratio', 0), 2),
                    'atr_pct': round(last_candle.get('atr_pct', 0), 3),
                    # Market regime data (ALWAYS INCLUDED)
                    'regime_confidence': market_regime.get('regime_confidence', 50) if market_regime else 50,
                    'regime_type': market_regime.get('regime_type', 'MIXED') if market_regime else 'MIXED',
                    'bb_suitability': market_regime.get('bb_suitability', 'FAIR') if market_regime else 'FAIR',
                    'position_multiplier': position_multiplier,
                    'btc_health_score': market_regime.get('btc_health_score', 50) if market_regime else 50,
                    'alt_market_outlook': market_regime.get('alt_market_outlook', 'FAIR') if market_regime else 'FAIR',
                    'market_health_score': market_regime.get('market_health_score', 50) if market_regime else 50,
                    'alt_season_indicator': market_regime.get('alt_season_indicator', 'NEUTRAL') if market_regime else 'NEUTRAL',
                    # Enhanced scoring data (ALWAYS INCLUDED)
                    'bb_score_34': bb_analysis.get('bb_score', 0),
                    'setup_quality_enhanced': bb_analysis.get('setup_quality', 'None'),
                    'scoring_details': bb_analysis.get('scoring_details', {}),
                    # NEW: Add confidence fields here
                    'technical_confidence': 0.0,  # Will be populated by confidence module
                    'historical_confidence': 0.0,  # Will be populated by confidence module  
                    'sentiment_confidence': 0.0,   # Will be populated by confidence module
                    'composite_confidence': 0.0,   # Will be populated by confidence module
                    'confidence_tier': 'UNRATED', # Will be populated by confidence module
                    'confidence_rationale': 'No analysis available',  # Will be populated by confidence module
                    # Tier breakdown (defaults to 0 for "no setup")
                    'tier_base_bb': bb_analysis.get('scoring_details', {}).get('tier_scores', {}).get('base_bb', 0),
                    'tier_money_flow': bb_analysis.get('scoring_details', {}).get('tier_scores', {}).get('money_flow', 0),
                    'tier_bb_specific': bb_analysis.get('scoring_details', {}).get('tier_scores', {}).get('bb_specific', 0),
                    'tier_volume_momentum': bb_analysis.get('scoring_details', {}).get('tier_scores', {}).get('volume_momentum', 0),
                    'tier_divergence': bb_analysis.get('scoring_details', {}).get('tier_scores', {}).get('divergence', 0),
                    # Key indicator values (defaults to 0 for "no setup")
                    'mfi_value': bb_analysis.get('scoring_details', {}).get('indicator_values', {}).get('mfi', 0),
                    'cmf_value': bb_analysis.get('scoring_details', {}).get('indicator_values', {}).get('cmf', 0),
                    'bb_expansion_ratio': bb_analysis.get('scoring_details', {}).get('indicator_values', {}).get('bb_expansion', 0),
                    'volume_surge_detected': bb_analysis.get('scoring_details', {}).get('indicator_values', {}).get('volume_surge', False),
                    'bb_trend_direction': bb_analysis.get('scoring_details', {}).get('indicator_values', {}).get('bb_trend', ''),
                    # Component details (defaults to empty for "no setup")
                    'component_1': bb_analysis.get('scoring_details', {}).get('breakdown', [''])[0] if len(bb_analysis.get('scoring_details', {}).get('breakdown', [])) > 0 else '',
                    'component_2': bb_analysis.get('scoring_details', {}).get('breakdown', [''])[1] if len(bb_analysis.get('scoring_details', {}).get('breakdown', [])) > 1 else '',
                    'component_3': bb_analysis.get('scoring_details', {}).get('breakdown', [''])[2] if len(bb_analysis.get('scoring_details', {}).get('breakdown', [])) > 2 else '',
                    'component_4': bb_analysis.get('scoring_details', {}).get('breakdown', [''])[3] if len(bb_analysis.get('scoring_details', {}).get('breakdown', [])) > 3 else '',
                    'component_5': bb_analysis.get('scoring_details', {}).get('breakdown', [''])[4] if len(bb_analysis.get('scoring_details', {}).get('breakdown', [])) > 4 else '',
                    # Priority signals (defaults to False for "no setup")
                    'mfi_priority_signal': 'MFI' in str(bb_analysis.get('scoring_details', {}).get('breakdown', [])) and '4 pts' in str(bb_analysis.get('scoring_details', {}).get('breakdown', [])),
                    'total_components': len(bb_analysis.get('scoring_details', {}).get('breakdown', []))
                }

                # Market context and backtesting data
                try:
                    market_context = self.market_analyzer.get_market_context_for_trade(symbol, bb_analysis)
                    market_baselines = self.market_analyzer.get_market_baselines()
                    historical_data = self.historical_intelligence.analyze_historical_performance(symbol, bb_analysis)
                    
                    # Debug: Check if historical data is available
                    if symbol in ['BTCUSDT', 'ETHUSDT']:
                        self.logger.info(f"🔍 {symbol} HISTORICAL DATA STATUS:")
                        self.logger.info(f"   Has insufficient_data: {historical_data.get('insufficient_data', False)}")
                        self.logger.info(f"   Has error: {historical_data.get('error', False)}")
                        self.logger.info(f"   Historical data keys: {list(historical_data.keys())}")
                    
                    # Get real historical data from the historical intelligence module
                    trade_quality = historical_data.get('trade_quality_analysis', {})
                    timing_intelligence = historical_data.get('timing_intelligence', {})
                    
                    # Debug: Print the real values being used
                    if symbol in ['BTCUSDT', 'ETHUSDT']:  # Only debug for major coins to avoid spam
                        self.logger.info(f"🔍 {symbol} HISTORICAL DATA DEBUG:")
                        self.logger.info(f"   Trade Quality: {trade_quality}")
                        self.logger.info(f"   Timing Intelligence: {timing_intelligence}")
                        self.logger.info(f"   Market Baselines: {market_baselines}")
                        
                        # Check if real values exist
                        self.logger.info(f"🔍 {symbol} FIELD CHECK:")
                        self.logger.info(f"   win_rate_pct exists: {'win_rate_pct' in trade_quality}")
                        self.logger.info(f"   win_rate_pct value: {trade_quality.get('win_rate_pct', 'NOT FOUND')}")
                        self.logger.info(f"   avg_win_pct exists: {'avg_win_pct' in trade_quality}")
                        self.logger.info(f"   avg_win_pct value: {trade_quality.get('avg_win_pct', 'NOT FOUND')}")
                        self.logger.info(f"   avg_loss_pct exists: {'avg_loss_pct' in trade_quality}")
                        self.logger.info(f"   avg_loss_pct value: {trade_quality.get('avg_loss_pct', 'NOT FOUND')}")
                        self.logger.info(f"   avg_timing_3pct exists: {'avg_timing_3pct' in timing_intelligence}")
                        self.logger.info(f"   avg_timing_3pct value: {timing_intelligence.get('avg_timing_3pct', 'NOT FOUND')}")
                    
                    result.update({
                        'historical_probability': trade_quality.get('win_rate_pct', 0),  # Real asset-specific win rate
                        'historical_bb_baseline': trade_quality.get('win_rate_pct', 0),  # Real asset-specific BB baseline
                        'historical_component_success': market_context.get('component_success_rate', 0),  # BB + component combo win rate
                        'historical_avg_win': trade_quality.get('avg_win_pct', 0),  # Real average win percentage
                        'historical_avg_loss': trade_quality.get('avg_loss_pct', 0),  # Real average loss percentage
                        'historical_avg_duration': timing_intelligence.get('avg_timing_3pct', 0),  # Real average timing
                        'overall_success_rate': market_baselines.get('overall_success_rate'),  # Market-wide baseline
                        'market_health': market_baselines.get('market_health', 0),  # Market health score
                        'total_bounces': market_baselines.get('total_bounces', 0),  # Total bounces analyzed
                        'indicator_benchmark': market_context.get('indicator_benchmark', 0),
                        'relative_performance': market_context.get('relative_performance', 'UNKNOWN')
                    })
                    
                    # Debug: Show the final values for major coins
                    if symbol in ['BTCUSDT', 'ETHUSDT']:
                        self.logger.info(f"📊 {symbol} FINAL HISTORICAL VALUES:")
                        self.logger.info(f"   historical_probability: {result.get('historical_probability', 0)}%")
                        self.logger.info(f"   historical_bb_baseline: {result.get('historical_bb_baseline', 0)}%")
                        self.logger.info(f"   historical_avg_win: {result.get('historical_avg_win', 0)}%")
                        self.logger.info(f"   historical_avg_loss: {result.get('historical_avg_loss', 0)}%")
                        self.logger.info(f"   historical_avg_duration: {result.get('historical_avg_duration', 0)} hours")
                        self.logger.info(f"   market_baseline: {result.get('overall_success_rate', 0)}%")
                        self.logger.info(f"   market_health: {result.get('market_health', 0)}%")
                        self.logger.info(f"   total_bounces_analyzed: {result.get('total_bounces', 0)}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not get market context for {symbol}: {str(e)}")
                    result.update({
                        'historical_probability': 0,
                        'historical_bb_baseline': 0,
                        'historical_component_success': 0,
                        'historical_avg_win': 0,
                        'historical_avg_loss': 0,
                        'historical_avg_duration': 0,
                        'overall_success_rate': None,
                        'market_health': 0,
                        'total_bounces': 0,
                        'indicator_benchmark': 0,
                        'relative_performance': 'UNKNOWN'
                    })

                # IF there's a BB setup - add the trading-specific data
                if bb_analysis['setup_type'] != 'NONE':
                    # Add trading-specific fields to the result
                    result.update({
                        'probability': probability,  # From existing calculation
                        'entry': round(bb_analysis['entry'], 6) if bb_analysis['entry'] != 0 else 0,
                        'stop': round(bb_analysis['stop'], 6) if bb_analysis['stop'] != 0 else 0,
                        'target1': round(bb_analysis['target1'], 6) if bb_analysis['target1'] != 0 else 0,
                        'risk_reward': bb_analysis['risk_reward'],
                        'risk_pct': round(risk_pct, 2),
                        'gain_pct': round(gain_pct, 2),
                        'divergence_detected': div_info['detected'],
                        'divergence_strength': div_info['strength'],
                        'divergence_confidence': div_info['confidence'],
                        'divergence_indicators': ', '.join(div_info['indicators']) if div_info['indicators'] else 'None',
                        'volume_confirmation': confirmations['volume_confirmation'],
                        'momentum_alignment': confirmations['momentum_alignment'],
                        'rr_acceptable': confirmations['risk_reward_acceptable'],
                        'risk_acceptable': risk_pct <= 8.0,
                        'patterns_detected': bb_analysis.get('patterns_detected', 'None'),
                        'significant_patterns': bb_analysis.get('significant_patterns', 'None'),
                        'pattern_confidence': bb_analysis.get('pattern_confidence', 0),
                        'pattern_boost': bb_analysis.get('pattern_boost', 0),
                        # ... all other existing trading fields
                    })
                    
                    result = self.metadata_enricher.enrich_trade_data(symbol, result)
                    if bb_analysis['setup_type'] != 'NONE':
                        print(f"   📊 Market Intelligence:")
                        print(f"      Market Cap Rank: {result['market_cap_rank']}")
                        print(f"      Market Cap Tier: {result['market_cap_tier']}")
                        print(f"      24h Volume: ${result['volume_24h_usd']:,.0f}")
                        print(f"      Liquidity Tier: {result['liquidity_tier']}")
                        print(f"      Primary Sector: {result['primary_sector']}")
                        print(f"      Expected Success Rate: {result['expected_success_rate']:.1f}%")
                        print(f"      Position Multiplier: {result['market_cap_multiplier']:.1f}x")

                    # Display logic for quality setups (8+ BB score) - Updated for consistency
                    bb_score = bb_analysis.get('bb_score', 0)
                    if bb_score >= 8:
                        self.logger.info(f"Quality setup: {symbol} {bb_analysis['setup_type']} "
                                      f"({exchange_name}) - {probability}% probability, "
                                      f"Risk: {risk_pct:.1f}%, R:R: {bb_analysis['risk_reward']}")
                        
                        # ADD THIS AFTER A QUALITY SETUP IS FOUND
                        # (Find where you log quality setups and add this code right after)
                        if bb_analysis['setup_type'] != 'NONE' and bb_analysis.get('bb_score', 0) >= 8:
                            try:
                                # Get market context for this trade
                                market_context = self.market_analyzer.get_market_context_for_trade(symbol, bb_analysis)
                                market_baselines = self.market_analyzer.get_market_baselines()
                                
                                # Calculate this trade's performance vs market
                                trade_score_pct = (bb_analysis.get('bb_score', 0) / 34) * 100
                                market_baseline = market_baselines['overall_success_rate']
                                
                                # Display market context
                                print(f"📊 MARKET CONTEXT:")
                                print(f"   🎯 This Trade: {trade_score_pct:.1f}% | Market Baseline: {market_baseline}%")
                                print(f"   📈 Performance: {market_context.get('relative_performance', 'UNKNOWN')}")
                                print(f"   🔍 Key Driver: {market_context.get('indicator_name', 'Unknown')}")
                                
                                # Show indicator benchmark if available
                                if 'indicator_benchmark' in market_context:
                                    benchmark = market_context['indicator_benchmark']
                                    print(f"   ⭐ Indicator Benchmark: {benchmark}% historical success rate")
                                
                                print(f"   🏥 Market Health: {market_baselines.get('market_health', 0)}%")
                                print(f"   📊 Based on: {market_baselines.get('total_bounces', 0)} historical bounces")
                                
                            except Exception as e:
                                self.logger.warning(f"Market context analysis failed for {symbol}: {e}")
                                # Continue without market context - don't break the main analysis
                        
                        # Show detailed scoring breakdown
                        if bb_analysis.get('scoring_details'):
                            scoring_breakdown = self.output_generator.format_scoring_breakdown({
                                'symbol': symbol,
                                'setup_type': bb_analysis['setup_type'],
                                'bb_score': bb_analysis['bb_score'],
                                'scoring_details': bb_analysis['scoring_details']
                            })
                            print(scoring_breakdown)
                        
                        # NEW: Send Telegram alert for high-probability trades
                        if self.telegram and probability >= 75:  # High probability threshold
                            try:
                                # Get technical indicators for the alert
                                rsi_value = last_candle.get('rsi', 0)
                                mfi_value = last_candle.get('mfi', 0)
                                
                                # Prepare trade data for Telegram
                                trade_data = {
                                    'symbol': symbol,
                                    'scanner_type': 'bb_scanner',
                                    'timeframe': '4H',
                                    'probability': probability,
                                    'risk_reward_ratio': bb_analysis.get('risk_reward', 0),
                                    'entry_price': bb_analysis.get('entry', 0),
                                    'stop_loss': bb_analysis.get('stop', 0),
                                    'target_1': bb_analysis.get('target1', 0),
                                    'target_2': bb_analysis.get('target2', 0) if 'target2' in bb_analysis else bb_analysis.get('target1', 0),
                                    'rsi': rsi_value,
                                    'mfi': mfi_value,
                                    'pattern_type': f"{bb_analysis['setup_type']} Setup - {bb_analysis.get('patterns_detected', 'BB Bounce')}"
                                }
                                
                                # Send the alert
                                success = self.telegram.send_trade_alert(trade_data)
                                if success:
                                    self.logger.info(f"✅ Telegram alert sent for {symbol}")
                                else:
                                    self.logger.warning(f"❌ Failed to send Telegram alert for {symbol}")
                                    
                            except Exception as e:
                                self.logger.warning(f"Telegram alert failed for {symbol}: {e}")
                else:
                    # NO BB SETUP - add default values for trading fields
                    result.update({
                        'probability': 0,
                        'entry': 0,
                        'stop': 0,
                        'target1': 0,
                        'risk_reward': 0,
                        'risk_pct': 0,
                        'gain_pct': 0,
                        'divergence_detected': False,
                        'divergence_strength': '',
                        'divergence_confidence': 0,
                        'divergence_indicators': 'None',
                        'volume_confirmation': False,
                        'momentum_alignment': False,
                        'rr_acceptable': False,
                        'risk_acceptable': False,
                        'patterns_detected': 'None',
                        'significant_patterns': 'None',
                        'pattern_confidence': 0,
                        'pattern_boost': 0,
                        'pattern_quality_best': 0,
                        'auto_take_profit': 0,
                        'risk_reward_ratio': 0,
                        'chart_patterns_detected': 'None',
                        'best_chart_pattern': 'None',
                        'chart_pattern_confidence': 0,
                        'chart_pattern_target': 0,
                        'support_levels': 'None',
                        'resistance_levels': 'None',
                        'sr_analysis_success': False,
                        'sl_level_strength': 'None',
                        'tp_level_strength': 'None',
                        'validation_notes': 'None'
                    })
                # ALWAYS add the result (setup or no setup) - THIS IS THE KEY CHANGE!
                all_analyses.append(result)
                
            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol} on {exchange_name}: {e}")
                continue
                
        return all_analyses

    async def scan_all_coins_comprehensive(self, market_regime=None) -> list:
        """Comprehensive scan of all coins (ENHANCED with market regime)"""
        all_results = []
        
        # Get top coins (EXISTING - UNCHANGED)
        top_coins = self.data_fetcher.fetch_top_coins(limit=500)
        
        if not top_coins:
            self.logger.error("Failed to fetch top coins")
            return []
        
        print(f"✅ Selected {len(top_coins)} coins for analysis")
        print(f"🔗 Active exchanges: {', '.join(self.data_fetcher.get_available_exchanges())}")
        
        # Analyze each coin (ENHANCED with market regime)
        for i, symbol in enumerate(top_coins):
            try:
                # NEW: Pass market regime data to analysis
                analyses = self.analyze_coin_comprehensive(symbol, market_regime)
                all_results.extend(analyses)
                
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{len(top_coins)} coins analyzed")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue
                
        return all_results

    def enrich_with_sentiment(self, df_results, top_n: int = 10):
        """Add sentiment analysis to top trades (EXISTING - UNCHANGED)"""
        if df_results.empty:
            return df_results
        
        print(f"\n🧠 Adding sentiment analysis to top {top_n} trades...")
        print("⏳ This may take 1-2 minutes due to API rate limits...")
        
        # Get top trades
        top_trades = df_results.head(top_n).copy()
        
        # Initialize sentiment columns
        sentiment_columns = [
            'lunar_sentiment_score', 'lunar_social_score', 'lunar_galaxy_score', 
            'lunar_alt_rank', 'lunar_sentiment_rating', 'lunar_data_available',
            'tm_trader_grade', 'tm_ta_grade', 'tm_quant_grade', 
            'tm_grade_change_24h', 'tm_data_available',
            'sentiment_alignment_score', 'sentiment_overall_alignment', 
            'sentiment_alignment_factors'
        ]
        
        for col in sentiment_columns:
            df_results[col] = ''
        
        # Process top trades
        for idx, (row_idx, trade) in enumerate(top_trades.iterrows(), 1):
            symbol = trade['symbol']
            setup_type = trade['setup_type']
            
            print(f"  {idx}/{top_n}: {symbol} {setup_type} (Probability: {trade['probability']}%)")
            
            # Get sentiment data
            lunar_data = self.sentiment_analyzer.get_lunarcrush_sentiment(symbol)
            tm_data = self.sentiment_analyzer.get_tokenmetrics_sentiment(symbol)
            alignment = self.sentiment_analyzer.analyze_sentiment_alignment(setup_type, lunar_data, tm_data)
            
            # Update dataframe
            for key, value in lunar_data.items():
                df_results.at[row_idx, key] = value
            
            for key, value in tm_data.items():
                df_results.at[row_idx, key] = value
            
            df_results.at[row_idx, 'sentiment_alignment_score'] = alignment['alignment_score']
            df_results.at[row_idx, 'sentiment_overall_alignment'] = alignment['overall_alignment']
            df_results.at[row_idx, 'sentiment_alignment_factors'] = ', '.join(alignment['alignment_factors'])
        
        print(f"✅ Sentiment analysis complete for top {top_n} trades")
        return df_results

    def run(self):
        """Run the complete modular BB bounce scanner (ENHANCED with market regime)"""
        try:
            print("🚀 CRYPTO BB BOUNCE SCANNER - ENHANCED WITH MARKET INTELLIGENCE")
            print("="*80)
            print("📊 Timeframe: 4H setup detection + 1H confirmation")
            print("🎯 Target: Middle band (3-6% moves) | Stops: 2x ATR")
            print("📈 Analysis: 5-indicator divergence + sentiment analysis")
            print("🌊 NEW: Market regime intelligence + BTC context analysis")  # NEW LINE
            print("✅ Modular: Clean separation of concerns")
            
            # Create output directories (EXISTING - UNCHANGED)
            os.makedirs("outputs", exist_ok=True)
            os.makedirs("outputs/excel_reports", exist_ok=True)
            os.makedirs("outputs/logs", exist_ok=True) 
            os.makedirs("outputs/alerts", exist_ok=True)
            self.logger.info("Output directories created: outputs, outputs/excel_reports, outputs/logs, outputs/alerts")
            
            print(f"\nStep 1: Fetching top {SCANNER_CONFIG['top_coins_limit']} coins from CoinMarketCap...")
            
            # NEW: Market regime analysis (Step 1.5)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            market_regime = loop.run_until_complete(self.run_market_regime_analysis())
            
            # DEBUG: Check if market regime analysis succeeded
            if market_regime is None:
                print("❌ WARNING: Market regime analysis failed - no market regime data available")
                print("   This will cause the Market Regime Analysis sheet to be missing from Excel output")
            else:
                print("✅ Market regime analysis completed successfully")
            
            print(f"\nStep 2: Analyzing broader crypto market conditions...")
            market_sentiment = self.market_sentiment_analyzer.get_complete_market_sentiment()
            
            print(f"Step 3: Performing BB bounce detection with validation...")
            print("🔍 4H BB touch detection with balanced thresholds")
            print("⏱️  1H confirmation for entry timing")
            print("💲 Current price validation (max 1.5% drift)")
            print("🧮 Risk management with quality filtering")
            
            # Run comprehensive analysis (ENHANCED with market regime)
            all_results = loop.run_until_complete(self.scan_all_coins_comprehensive(market_regime))
            
            if not all_results:
                print("No analysis results found.")
                return
            
            # NEW: Add confidence enhancement
            print("🎯 Evaluating trade confidence...")
            
            try:
                enhanced_trades = enhance_all_trades_with_confidence(
                    all_results, market_regime, self.confidence_module
                )
                print(f"✅ Confidence enhancement complete: {len(enhanced_trades)} trades enhanced")
                
            except Exception as e:
                print(f"❌ Confidence enhancement failed: {e}")
                # Fallback to original data
                enhanced_trades = all_results

            # Get the global market summary data and update all trade results
            print("📊 Updating trade results with global market summary...")
            try:
                market_data = self.output_generator._run_market_overview_analysis()
                
                # Update each trade result with the global market summary stats
                for result in enhanced_trades:
                    result['total_bounces'] = market_data.get('total_bounces', 0)
                    result['overall_success_rate'] = market_data.get('overall_success_rate', 0)
                    result['market_health'] = market_data.get('market_health', 0)
                
                print(f"✅ Market summary updated for {len(enhanced_trades)} trades")
            except Exception as e:
                print(f"⚠️ Could not update market summary: {e}")

            # NEW: Analysis summary display
            quality_results = {}
            all_analysis_data = []  # Keep ALL data for Excel
            
            for result in enhanced_trades:
                # Save ALL data for Excel (ML training)
                all_analysis_data.append(result)
                
                # Only track quality setups for summary (8+ BB score)
                bb_score = result.get('bb_score', 0)
                if bb_score >= 8:  # Quality threshold
                    quality_results[result.get('symbol', 'Unknown')] = result
            
            # Display analysis summary
            print(f"\n🔍 ANALYSIS SUMMARY:")
            print(f"   • Total coins analyzed: {len(all_results)}")
            print(f"   • Quality setups found: {len(quality_results)}")
            print(f"   • Success rate: {len(quality_results)/len(all_results)*100:.1f}%")
            
            if quality_results:
                print(f"\n🎯 QUALITY SETUPS (8+ points):")
                print("=" * 80)
                for symbol, result in quality_results.items():
                    # Display quality setup details
                    setup_type = result.get('setup_type', 'Unknown')
                    bb_score = result.get('bb_score', 0)
                    probability = result.get('probability', 0)
                    risk_pct = result.get('risk_pct', 0)
                    risk_reward = result.get('risk_reward', 0)
                    
                    print(f"\n📊 {symbol} - {setup_type}")
                    print(f"   🎯 BB Score: {bb_score}/16 | Probability: {probability}%")
                    print(f"   💰 Risk: {risk_pct:.1f}% | R:R: {risk_reward}:1")
                    
                    # Show detailed scoring breakdown for quality setups
                    if result.get('scoring_details'):
                        scoring_breakdown = self.output_generator.format_scoring_breakdown({
                            'symbol': symbol,
                            'setup_type': setup_type,
                            'bb_score': bb_score,
                            'scoring_details': result.get('scoring_details')
                        })
                        print(scoring_breakdown)
            else:
                print(f"\n⏳ No quality setups found in current market conditions")
                print(f"   Waiting for institutional-grade opportunities (12+ points)")
            
            print(f"\nStep 4: Formatting and categorizing results...")
            # Format results inline
            import pandas as pd
            df_all = self.output_generator.format_comprehensive_results(all_analysis_data)  # Use ALL data for Excel
            
            # Sort and filter results
            # Results are already formatted and sorted by output_generator

            if df_all.empty:
                self.logger.info("No quality setups found after filtering.")
                print("\nNo quality setups found that meet the criteria.")
                return

            # Step 3: Enrich with sentiment data (EXISTING - UNCHANGED)
            df_enhanced = self.enrich_with_sentiment(df_all, top_n=10)

            # Step 4: Generate outputs (ENHANCED with market regime)
            try:
                # Get overall market sentiment (EXISTING - UNCHANGED)
                self.output_generator.display_market_sentiment(market_sentiment)

                # Display comprehensive summary (EXISTING - UNCHANGED)
                self.output_generator.display_terminal_summary(df_enhanced, market_sentiment)

                # Display sentiment analysis summary (EXISTING - UNCHANGED)
                self.output_generator.display_sentiment_summary(df_enhanced)
                
                # DATABASE LOGGING - Log all quality results to database
                print("💾 Logging results to database...")
                try:
                    logger = TradeLogger()
                    if logger.connection:
                        # Create scan record
                        scan_id = logger.log_scan_start('bb_scanner_4h', 'BB Scanner 4H')
                        print(f"✅ Created scan_id: {scan_id}")
                        
                        # Log each quality result
                        trades_logged = 0
                        for symbol, result in quality_results.items():
                            try:
                                # Only log trades that have actual BB setups (not NONE)
                                if result.get('setup_type') == 'NONE':
                                    print(f"SKIP: {symbol} - No BB setup")
                                    continue
                                
                                # FIX: Ensure symbol has USDT suffix for proper formatting
                                formatted_symbol = symbol
                                if not symbol.endswith('USDT') and not symbol.endswith('USD'):
                                    formatted_symbol = f"{symbol}USDT"
                                
                                # FIX: Extract cost data from result object
                                entry_price = result.get('entry', 0)
                                stop_loss = result.get('stop', 0)
                                target_1 = result.get('target1', 0)
                                current_price = result.get('current_price', 0)
                                
                                # DEBUG: Print the actual values being extracted
                                print(f"DEBUG: {symbol} - Entry: {entry_price}, Stop: {stop_loss}, Target: {target_1}, Current: {current_price}")
                                
                                # Prepare trade data for database
                                trade_data = {
                                    'symbol': formatted_symbol,
                                    'exchange': 'Binance',  # Default exchange
                                    'timeframe': '4H',
                                    'bb_score': result.get('bb_score', 0),
                                    'probability': result.get('probability', 0),
                                    'risk_reward_ratio': result.get('risk_reward', 0),
                                    'current_price': current_price,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'target_1': target_1,
                                    'target_2': result.get('target_2', 0),
                                    'target_3': result.get('target_3', 0),
                                    'rsi': result.get('rsi', 0),
                                    'mfi': result.get('mfi', 0),
                                    'stochastic_k': result.get('stochastic_k', 0),
                                    'volume_surge': result.get('volume_surge', 0),
                                    'macd_signal': result.get('macd_signal', 'neutral'),
                                    'pattern_type': result.get('setup_type', 'BB Bounce'),
                                    'pattern_quality': result.get('tier', 'GOOD'),
                                    'confluence_score': result.get('confluence_score', 0),
                                    'historical_win_rate': result.get('historical_win_rate', 0),
                                    'category_win_rate': result.get('category_win_rate', 0),
                                    'similar_setups_count': result.get('similar_setups_count', 0),
                                    'market_cap': result.get('market_cap', 0),
                                    'volume_24h': result.get('volume_24h', 0),
                                    'price_change_24h': result.get('price_change_24h', 0),
                                    'scanner_type': 'bb_scanner_4h'
                                }
                                
                                # Log to database
                                success = logger.log_trade_opportunity(scan_id, trade_data)
                                if success:
                                    trades_logged += 1
                                    print(f"✅ Logged trade: {symbol} -> {result.get('probability', 0)}%")
                                else:
                                    print(f"❌ Failed to log trade: {symbol}")
                                    
                            except Exception as e:
                                print(f"❌ Error logging trade {symbol}: {e}")
                                continue
                        
                        # Complete the scan
                        logger.complete_scan(scan_id, len(quality_results), trades_logged, 120)
                        print(f"✅ Database logging complete: {trades_logged} trades logged")
                        
                    else:
                        print("❌ Database connection failed")
                        
                except Exception as e:
                    print(f"❌ Database logging error: {e}")
                
                # NEW: Send Telegram summary report
                if self.telegram:
                    try:
                        # Count high-probability trades
                        high_prob_trades = [r for r in enhanced_trades if r.get('probability', 0) >= 75]
                        
                        # Prepare summary data
                        summary_data = {
                            'period': 'Last 4 Hours',
                            'total_scans': len(all_results),
                            'opportunities': len(quality_results),
                            'high_prob': len(high_prob_trades),
                            'by_scanner': {
                                'BB Scanner': {'opportunities': len(quality_results)}
                            },
                            'top_opportunities': [
                                {
                                    'symbol': r.get('symbol', 'Unknown'),
                                    'probability': r.get('probability', 0),
                                    'risk_reward_ratio': r.get('risk_reward', 0)
                                }
                                for r in high_prob_trades[:5]  # Top 5
                            ]
                        }
                        
                        # Send summary report
                        success = self.telegram.send_summary_report(summary_data)
                        if success:
                            self.logger.info("✅ Telegram summary report sent")
                        else:
                            self.logger.warning("❌ Failed to send Telegram summary report")
                            
                    except Exception as e:
                        self.logger.warning(f"Telegram summary report failed: {e}")
                
                print("✅ Analysis complete!")
                
                # Generate Excel output (ENHANCED with market regime) - ALL data for ML training
                # Use the enhanced DataFrame that includes sentiment data
                excel_filename = self.output_generator.generate_excel_output(df_enhanced, market_regime)
                print(f"📊 Excel results saved to: {excel_filename}")
                print(f"📁 Organized in: outputs/excel_reports/")
                print(f"💾 Excel contains ALL {len(all_analysis_data)} analyzed coins (for ML training)")

            except Exception as e:
                self.logger.error(f"Critical error in scanner: {e}")
                print(f"Error: {e}")

        except Exception as e:
            # NEW: Send error alert to Telegram
            if self.telegram:
                try:
                    self.telegram.send_error_alert(str(e), "BB Scanner 4H")
                except:
                    pass  # Don't let Telegram errors break the main error handling
            
            self.logger.error(f"Critical error in scanner: {e}")
            print(f"Error: {e}")

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
            """Calculate Average True Range for pattern analysis"""
            try:
                high = df['high']
                low = df['low']
                close = df['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=period).mean().iloc[-1]
                
                return atr if not pd.isna(atr) else 0.01
                
            except Exception as e:
                self.logger.error(f"ATR calculation error: {e}")
                return 0.01

def main():
    """Main entry point (EXISTING - UNCHANGED)"""
    scanner = ModularBBScanner()
    scanner.run()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crypto BB Bounce Scanner')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Run in quiet mode with minimal output')
    args = parser.parse_args()

    # Set logging level based on quiet mode
    if args.quiet:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')
        print("🤫 Quiet mode enabled - showing only results and warnings")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    main()