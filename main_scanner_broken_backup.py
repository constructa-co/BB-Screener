#!/usr/bin/env python3
"""
Crypto BB Bounce Scanner - Main Orchestrator
Modular architecture for clean, maintainable code
Enhanced with Market Regime Intelligence
"""

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
    # asyncio is already imported in your file, so no need to add it again
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
        
        print(f"DEBUG: Starting analysis for {symbol}")
        
        # NEW: Extract position multiplier from market regime
        position_multiplier = market_regime.get('position_multiplier', 1.0) if market_regime else 1.0
        print(f"DEBUG: {symbol} - Position multiplier: {position_multiplier}")
        
        for exchange_name in self.data_fetcher.get_available_exchanges():
            print(f"DEBUG: Analyzing {symbol} on {exchange_name}")
            try:
                print(f"DEBUG: {symbol}/{exchange_name} - Step 1: Getting data...")
                # Step 1: Fetch market data (EXISTING - UNCHANGED)
                df = self.data_fetcher.fetch_ohlcv(exchange_name, symbol, '4h')
                if df is None or len(df) < 50:
                    print(f"DEBUG: {symbol}/{exchange_name} - SKIP: Insufficient 4H data")
                    continue
                
                print(f"DEBUG: {symbol}/{exchange_name} - Step 2: BB analysis...")
                
                # Step 2: BB Detection (EXISTING - UNCHANGED)
                bb_analysis = self.bb_detector.analyze_bb_setup(df)
                if bb_analysis['setup_type'] == 'NONE':
                    print(f"DEBUG: {symbol}/{exchange_name} - SKIP: No BB setup (setup_type: {bb_analysis.get('setup_type', 'NONE')})")
                    continue
                
                print(f"DEBUG: {symbol}/{exchange_name} - Step 3: Pattern analysis...")
                
                # Step 2.5: Pattern Recognition Analysis (SUPPLEMENTARY DATA)
                pattern_data = None  # Initialize before try block
                try:
                    from modules.pattern_analyzer import PatternAnalyzer
                    pattern_analyzer = PatternAnalyzer()
                    print(f"DEBUG: {symbol}/{exchange_name} - Pattern analyzer loaded")
                    
                    # Get 1H data for multi-timeframe analysis
                    df_1h = self.data_fetcher.fetch_ohlcv(exchange_name, symbol, '1h')
                    if df_1h is not None and len(df_1h) >= 50:
                        
                        # Calculate ATR for pattern significance
                        atr_value = self._calculate_atr(df, period=14)
                        
                        # Comprehensive multi-timeframe pattern analysis
                        pattern_data = pattern_analyzer.analyze_comprehensive_patterns(
                            symbol, df, df_1h, atr_value, bb_analysis
                        )
                        print(f"DEBUG: {symbol}/{exchange_name} - Pattern analysis complete")
                        
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
                    print(f"DEBUG: {symbol}/{exchange_name} - Pattern analysis failed: {e}")
                    # Graceful fallback - BB analysis continues unaffected
                    self.logger.warning(f"Pattern analysis failed for {symbol}: {str(e)}")
                    pattern_data = None  # Explicitly set to None
                    bb_analysis['pattern_analysis'] = None
                    bb_analysis['pattern_boost'] = 0
                    bb_analysis['patterns_detected'] = 'Analysis failed'
                    bb_analysis['pattern_confidence'] = 0

                print(f"DEBUG: {symbol}/{exchange_name} - Step 4: 1H confirmation...")
                
                # Step 3: Technical Analysis (EXISTING - UNCHANGED)
                has_1h_confirmation = self.technical_analyzer.get_1h_confirmation(
                    exchange_name, symbol, bb_analysis['setup_type'], self.data_fetcher
                )
                
                # Skip if no 1H confirmation (EXISTING - UNCHANGED)
                if not has_1h_confirmation:
                    print(f"DEBUG: {symbol}/{exchange_name} - SKIP: No 1H confirmation")
                    self.logger.debug(f"{symbol} {bb_analysis['setup_type']} lacks 1H confirmation")
                    continue
                
                print(f"DEBUG: {symbol}/{exchange_name} - Step 5: Price validation...")
                
                # Validate current price (EXISTING - UNCHANGED)
                price_valid, current_price = self.data_fetcher.validate_current_price(
                    exchange_name, symbol, bb_analysis['entry']
                )
                
                if not price_valid:
                    print(f"DEBUG: {symbol}/{exchange_name} - SKIP: Price validation failed")
                    self.logger.debug(f"{symbol} price moved too much since scan")
                    continue
                
                # Update entry price (EXISTING - UNCHANGED)
                bb_analysis['entry'] = current_price
                
                # UPDATED: Enhanced divergence detection (EXISTING - UNCHANGED)
                bull_divergence = self.technical_analyzer.detect_enhanced_bullish_divergence(df)
                bear_divergence = self.technical_analyzer.detect_enhanced_bearish_divergence(df)
                
                # Step 4: Risk Assessment (EXISTING - UNCHANGED)
                probability, confirmations = self.risk_manager.calculate_comprehensive_probability(
                    df, bb_analysis, bull_divergence, bear_divergence
                )
                
                # Apply quality filters (EXISTING - UNCHANGED)
                risk_pct = abs((bb_analysis['entry'] - bb_analysis['stop']) / bb_analysis['entry'] * 100)
                
                # Skip poor quality setups (EXISTING - UNCHANGED)
                risk_metrics = {'risk_pct': risk_pct}
                if not self.risk_manager.apply_quality_filters(bb_analysis, risk_metrics):
                    continue
                
                # Calculate gain potential (EXISTING - UNCHANGED)
                gain_pct = abs((bb_analysis['target1'] - bb_analysis['entry']) / bb_analysis['entry'] * 100)
                
                # Get last candle data (EXISTING - UNCHANGED)
                last_candle = df.iloc[-1]
                
                # Determine divergence info based on setup type (EXISTING - UNCHANGED)
                if bb_analysis['setup_type'] == 'LONG':
                    div_info = bull_divergence
                else:
                    div_info = bear_divergence
                
                print(f"DEBUG: {symbol}/{exchange_name} - Step 6: About to create result...")
                
                # Create result record (ENHANCED with market regime data)
                
                print(f"DEBUG: About to create result for {symbol} on {exchange_name}")
                print(f"DEBUG: BB analysis setup_type: {bb_analysis['setup_type']}")
                print(f"DEBUG: Has 1H confirmation: {has_1h_confirmation}")
                print(f"DEBUG: Price valid: {price_valid}")
                
                # Extract risk/reward data from pattern analysis
                rr_data = pattern_data.get('auto_risk_reward', {}) if pattern_data else {}
                
                result = {
                    # EXISTING data (UNCHANGED)
                    'symbol': symbol,
                    'exchange': exchange_name,
                    'setup_type': bb_analysis['setup_type'],
                    'probability': probability,
                    'bb_score': bb_analysis['bb_score'],
                    'setup_quality': bb_analysis['setup_quality'],
                    
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
                    
                    'bb_pct': round(last_candle['bb_pct'], 3),
                    'rsi': round(last_candle['rsi'], 1),
                    'volume_ratio': round(last_candle['volume_ratio'], 2),
                    'atr_pct': round(last_candle['atr_pct'], 3),
                    
                    'volume_confirmation': confirmations['volume_confirmation'],
                    'momentum_alignment': confirmations['momentum_alignment'],
                    'rr_acceptable': confirmations['risk_reward_acceptable'],
                    'risk_acceptable': risk_pct <= 8.0,
                    
                    'timestamp': datetime.now(),
                    
                    # Pattern analysis data (add after line 260)
                    'patterns_detected': bb_analysis.get('patterns_detected', 'None'),
                    'significant_patterns': bb_analysis.get('significant_patterns', 'None'), 
                    'pattern_confidence': bb_analysis.get('pattern_confidence', 0),
                    'pattern_boost': bb_analysis.get('pattern_boost', 0),
                    'pattern_quality_best': bb_analysis.get('pattern_quality_best', 0),
                    'auto_take_profit': bb_analysis.get('auto_take_profit', 0),
                    'risk_reward_ratio': bb_analysis.get('risk_reward_ratio', 0),

                    # Chart pattern data (add these 4 lines)
                    'chart_patterns_detected': bb_analysis.get('chart_patterns_detected', 'None'),
                    'best_chart_pattern': bb_analysis.get('best_chart_pattern', 'None'),
                    'chart_pattern_confidence': bb_analysis.get('chart_pattern_confidence', 0),
                    'chart_pattern_target': bb_analysis.get('chart_pattern_target', 0),

                    # Support/Resistance data (add these 6 lines)
                    'support_levels': rr_data.get('support_levels', 'None'),
                    'resistance_levels': rr_data.get('resistance_levels', 'None'),
                    'sr_analysis_success': rr_data.get('sr_analysis_success', False),
                    'sl_level_strength': rr_data.get('sl_level_strength', 'None'),
                    'tp_level_strength': rr_data.get('tp_level_strength', 'None'),
                    'validation_notes': rr_data.get('validation_notes', 'None'),

                    # NEW: Market regime data (8 new columns)
                    'regime_confidence': market_regime.get('regime_confidence', 50) if market_regime else 50,
                    'regime_type': market_regime.get('regime_type', 'MIXED') if market_regime else 'MIXED',
                    'bb_suitability': market_regime.get('bb_suitability', 'FAIR') if market_regime else 'FAIR',
                    'position_multiplier': position_multiplier,
                    'btc_health_score': market_regime.get('btc_health_score', 50) if market_regime else 50,
                    'alt_market_outlook': market_regime.get('alt_market_outlook', 'FAIR') if market_regime else 'FAIR',
                    'market_health_score': market_regime.get('market_health_score', 50) if market_regime else 50,
                    'alt_season_indicator': market_regime.get('alt_season_indicator', 'NEUTRAL') if market_regime else 'NEUTRAL'
                }
                
                print(f"DEBUG: Result created successfully for {symbol} on {exchange_name}")
                print(f"DEBUG: Adding result for {symbol} on {exchange_name}")
                all_analyses.append(result)
                
                # EXISTING logging (UNCHANGED)
                if probability >= 65:
                    self.logger.info(f"Quality setup: {symbol} {bb_analysis['setup_type']} "
                                  f"({exchange_name}) - {probability}% probability, "
                                  f"Risk: {risk_pct:.1f}%, R:R: {bb_analysis['risk_reward']}")
                
            except Exception as e:
                print(f"DEBUG: Exception for {symbol} on {exchange_name}: {e}")
                self.logger.debug(f"Error analyzing {symbol} on {exchange_name}: {e}")
                continue
                
        print(f"DEBUG: {symbol} analysis complete. Found {len(all_analyses)} results")
        return all_analyses

    async def scan_all_coins_comprehensive(self, market_regime=None) -> list:
        """Comprehensive scan of all coins (ENHANCED with market regime)"""
        print("DEBUG: scan_all_coins_comprehensive started")
        all_results = []
        
        # Get top coins (EXISTING - UNCHANGED)
        top_coins = self.data_fetcher.fetch_top_coins(limit=100)
        
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
            
            print(f"\nStep 1: Fetching top 100 coins from CoinMarketCap...")
            
            # NEW: Market regime analysis (Step 1.5)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            market_regime = loop.run_until_complete(self.run_market_regime_analysis())
            
            print(f"\nStep 2: Analyzing broader crypto market conditions...")
            market_sentiment = self.market_sentiment_analyzer.get_complete_market_sentiment()
            
            print(f"Step 3: Performing BB bounce detection with validation...")
            print("🔍 4H BB touch detection with balanced thresholds")
            print("⏱️  1H confirmation for entry timing")
            print("💲 Current price validation (max 1.5% drift)")
            print("🧮 Risk management with quality filtering")
            
            # Run comprehensive analysis (ENHANCED with market regime)
            print(f"DEBUG: Starting scan_all_coins_comprehensive...")
            try:
                all_results = loop.run_until_complete(self.scan_all_coins_comprehensive(market_regime))
                print(f"DEBUG: Scan completed. Results count: {len(all_results) if all_results else 'None'}")
                print(f"DEBUG: Results type: {type(all_results)}")
            except Exception as e:
                print(f"ERROR in scan_all_coins_comprehensive: {e}")
                import traceback
                traceback.print_exc()
                return
            
            if not all_results:
                print("No analysis results found.")
                return
            
            print(f"Step 4: Formatting and categorizing results...")
            # Format results inline
            import pandas as pd
            df_all = self.output_generator.format_comprehensive_results(all_results)
            
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
                
                print("✅ Analysis complete!")
                
                # Generate Excel output (ENHANCED with market regime)
                excel_filename = self.output_generator.generate_excel_output(df_enhanced, market_regime)
                print(f"📊 Excel results saved to: {excel_filename}")
                print(f"📁 Organized in: outputs/excel_reports/")

            except Exception as e:
                self.logger.error(f"Critical error in scanner: {e}")
                print(f"Error: {e}")

        except Exception as e:
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