# output_generator.py - Output Generation Module
import pandas as pd
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
from config import *
from modules.market_regime_enhanced import format_enhanced_regime_output

logger = logging.getLogger(__name__)

class OutputGenerator:
    def __init__(self):
        # Create organized folder structure
        self.base_output_dir = "outputs"
        self.excel_dir = os.path.join(self.base_output_dir, "excel_reports")
        self.logs_dir = os.path.join(self.base_output_dir, "logs")
        self.alerts_dir = os.path.join(self.base_output_dir, "alerts")
        
        # Create directories if they don't exist
        self._create_output_directories()
        
    def _create_output_directories(self):
        """Create organized output directories"""
        try:
            directories = [self.base_output_dir, self.excel_dir, self.logs_dir, self.alerts_dir]
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            logger.info(f"Output directories created: {', '.join(directories)}")
        except Exception as e:
            logger.error(f"Error creating output directories: {e}")
            # Fallback to current directory
            self.excel_dir = "."
            self.logs_dir = "."
            self.alerts_dir = "."
        
    def format_results_dataframe(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Format all results into a comprehensive DataFrame"""
        try:
            if not all_results:
                return pd.DataFrame()
            
            # Remove duplicates - keep best exchange for each symbol
            unique_results = {}
            for result in all_results:
                key = f"{result['symbol']}_{result['setup_type']}" if result['setup_type'] != 'NONE' else result['symbol']
                if key not in unique_results or result['probability'] > unique_results[key]['probability']:
                    unique_results[key] = result
            
            all_formatted = list(unique_results.values())
            all_formatted.sort(key=lambda x: (x['probability'], x['risk_reward']), reverse=True)
            
            df = pd.DataFrame(all_formatted)
            
            if df.empty:
                return df
            
            # Add tier classification
            df['tier'] = df.apply(self._categorize_setup, axis=1)
            
            # Add action recommendations
            df['action'] = df.apply(self._recommend_action, axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error formatting results DataFrame: {e}")
            return pd.DataFrame()

    def _categorize_setup(self, row) -> str:
        """Categorize setup based on probability and risk"""
        try:
            prob = row['probability']
            risk_pct = row.get('risk_pct', 0)
            rr = row.get('risk_reward', 0)
            
            # Downgrade if excessive risk or poor R:R
            if risk_pct > 4.0 or rr < 1.0:
                prob = min(prob - 10, prob)
            
            if prob >= 75:
                return 'PREMIUM'
            elif prob >= 70:
                return 'HIGH'
            elif prob >= 65:
                return 'GOOD'
            elif prob >= 60:
                return 'FAIR'
            elif prob >= 55:
                return 'MARGINAL'
            else:
                return 'WEAK'
                
        except Exception as e:
            logger.error(f"Error categorizing setup: {e}")
            return 'WEAK'

    def _recommend_action(self, row) -> str:
        """Recommend action based on tier and setup type"""
        try:
            if row['tier'] in ['PREMIUM', 'HIGH'] and row['setup_type'] != 'NONE':
                return 'TAKE TRADE'
            elif row['tier'] == 'GOOD' and row['setup_type'] != 'NONE':
                return 'CONSIDER'
            elif row['tier'] == 'FAIR' and row['setup_type'] != 'NONE':
                return 'MONITOR'
            elif row['setup_type'] != 'NONE':
                return 'WATCH ONLY'
            else:
                return 'NO SETUP'
                
        except Exception as e:
            logger.error(f"Error recommending action: {e}")
            return 'NO SETUP'

    def generate_excel_output(self, df: pd.DataFrame, market_regime: Dict = None, filename: str = None) -> str:
        """Generate comprehensive Excel output with multiple sheets in organized folder (ENHANCED with market regime)"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bb_analysis_{timestamp}.xlsx"
            
            # Save to excel_reports folder
            filepath = os.path.join(self.excel_dir, filename)
            
            # Ensure market regime columns exist with actual data or defaults
            if not df.empty:
                # Format enhanced regime data if available
                if market_regime and market_regime.get('enhanced_analysis'):
                    enhanced_regime = format_enhanced_regime_output(market_regime)
                else:
                    enhanced_regime = market_regime
                
                regime_columns = {
                    'regime_confidence': enhanced_regime.get('regime_confidence', 50) if enhanced_regime else 50,
                    'regime_type': enhanced_regime.get('regime_type', 'MIXED') if enhanced_regime else 'MIXED',
                    'bb_suitability': enhanced_regime.get('bb_suitability', 'FAIR') if enhanced_regime else 'FAIR', 
                    'position_multiplier': enhanced_regime.get('position_multiplier', 1.0) if enhanced_regime else 1.0,
                    'btc_health_score': enhanced_regime.get('btc_health_score', 50) if enhanced_regime else 50,
                    'alt_market_outlook': enhanced_regime.get('alt_market_outlook', 'FAIR') if enhanced_regime else 'FAIR',
                    'market_health_score': enhanced_regime.get('market_health_score', 50) if enhanced_regime else 50,
                    'alt_season_indicator': enhanced_regime.get('alt_season_indicator', 'NEUTRAL') if enhanced_regime else 'NEUTRAL',
                    'spy_trend': enhanced_regime.get('spy_trend', 'UNKNOWN') if enhanced_regime else 'UNKNOWN',
                    'spy_change': enhanced_regime.get('spy_change', 0) if enhanced_regime else 0,
                    'vix_price': enhanced_regime.get('vix_price', 20) if enhanced_regime else 20,
                    'market_environment': enhanced_regime.get('market_environment', 'NEUTRAL') if enhanced_regime else 'NEUTRAL',
                    # Enhanced regime fields
                    'enhanced_regime_confidence': enhanced_regime.get('enhanced_regime_confidence', 50) if enhanced_regime else 50,
                    'futures_sentiment': enhanced_regime.get('futures_sentiment', 'NEUTRAL') if enhanced_regime else 'NEUTRAL',
                    'liquidation_pressure': enhanced_regime.get('liquidation_pressure', 'LOW') if enhanced_regime else 'LOW',
                    'synthetic_supply_ratio': enhanced_regime.get('synthetic_supply_ratio', 1.0) if enhanced_regime else 1.0,
                    'enhanced_timing_signal': enhanced_regime.get('enhanced_timing_signal', 'NEUTRAL') if enhanced_regime else 'NEUTRAL'
                }
                
                # Enhanced regime data with futures intelligence
                enhanced_regime_data = format_enhanced_regime_output(market_regime) if market_regime else {}
                
                # Add enhanced futures intelligence fields
                enhanced_regime_columns = {
                    # Enhanced regime metrics
                    'enhanced_regime_confidence': enhanced_regime_data.get('enhanced_regime_confidence', 50),
                    'enhanced_regime_classification': enhanced_regime_data.get('enhanced_regime_classification', 'UNKNOWN'),
                    'intelligence_layers': enhanced_regime_data.get('intelligence_layers', 6),
                    
                    # Futures intelligence (6 new columns)
                    'funding_sentiment_signal': enhanced_regime_data.get('funding_sentiment_signal', 'UNKNOWN'),
                    'funding_rate': enhanced_regime_data.get('funding_rate', 0.0),
                    'liquidation_pressure': enhanced_regime_data.get('liquidation_pressure', 'UNKNOWN'),
                    'squeeze_risk_level': enhanced_regime_data.get('squeeze_risk_level', 'UNKNOWN'),
                    'market_timing_signal': enhanced_regime_data.get('market_timing_signal', 'NEUTRAL'),
                    'enhanced_risk_level': enhanced_regime_data.get('enhanced_risk_level', 'MEDIUM')
                }
                
                # Merge enhanced columns with existing regime columns
                regime_columns.update(enhanced_regime_columns)
                
                # Add columns if they don't exist - but ONLY if market_regime data isn't already in df
                for col, default in regime_columns.items():
                    if col not in df.columns:
                        # Use actual market regime data if available, otherwise use defaults
                        if market_regime and col in ['regime_confidence', 'regime_type', 'bb_suitability', 'position_multiplier',
                                                   'btc_health_score', 'alt_market_outlook', 'market_health_score', 'alt_season_indicator',
                                                   'spy_trend', 'spy_change', 'vix_price', 'market_environment']:
                            df[col] = regime_columns[col]
                        else:
                            df[col] = default
            
            # Use ExcelWriter with better error handling
            try:
                with pd.ExcelWriter(filepath, engine='openpyxl', mode='w') as writer:
                    # Sheet 1: All results (ENHANCED with market regime columns)
                    if not df.empty:
                        df.to_excel(writer, sheet_name='All_Analysis', index=False)
                    
                    # Sheet 2: Premium and High probability only (FIXED)
                    premium_high = df[df['tier'].isin(['PREMIUM', 'HIGH'])] if not df.empty else pd.DataFrame()
                    if not premium_high.empty:
                        premium_high.to_excel(writer, sheet_name='Premium_High_Only', index=False)
                    else:
                        # Create empty sheet with message if no premium/high trades
                        empty_premium = pd.DataFrame([
                            ['No Premium or High probability trades found'],
                            ['Try running scanner when market conditions improve'],
                            ['Current market regime may not be suitable for BB bounces']
                        ], columns=['Message'])
                        empty_premium.to_excel(writer, sheet_name='Premium_High_Only', index=False)
                    
                    # Sheet 3: Trade recommendations  
                    trade_recs = df[df['action'].isin(['TAKE TRADE', 'CONSIDER'])] if not df.empty else pd.DataFrame()
                    if not trade_recs.empty:
                        trade_recs.to_excel(writer, sheet_name='Trade_Recommendations', index=False)
                    
                    # Sheet 4: Low risk trades (≤3% risk)
                    low_risk = df[(df['risk_pct'] <= 3.0) & (df['setup_type'] != 'NONE')] if not df.empty else pd.DataFrame()
                    if not low_risk.empty:
                        low_risk.to_excel(writer, sheet_name='Low_Risk_Trades', index=False)
                    
                    # Sheet 5: Monitoring list
                    monitor_list = df[df['action'].isin(['MONITOR', 'WATCH ONLY'])] if not df.empty else pd.DataFrame()
                    if not monitor_list.empty:
                        monitor_list.to_excel(writer, sheet_name='Monitoring_List', index=False)
                    
                    # Sheet 6: Top 10 with sentiment (if sentiment data exists)
                    sentiment_cols = ['lunar_data_available', 'tm_data_available']
                    if not df.empty and any(col in df.columns for col in sentiment_cols):
                        top_10_sentiment = df.head(10)
                        if not top_10_sentiment.empty:
                            top_10_sentiment.to_excel(writer, sheet_name='Top_10_Sentiment', index=False)
                    
                    # Sheet 7: Market Regime Analysis Dashboard (NEW)
                    if market_regime:
                        self._create_market_regime_sheet(writer, market_regime)
                    else:
                        # Create empty regime sheet with message
                        regime_df = pd.DataFrame([
                            ['Market Regime Analysis', 'Not Available'],
                            ['Status', 'No regime data provided'],
                            ['Recommendation', 'Enable market regime analysis for enhanced intelligence']
                        ], columns=['Metric', 'Value'])
                        regime_df.to_excel(writer, sheet_name='Market_Regime_Analysis', index=False)
            
            except PermissionError:
                # File might be open - try a different filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"bb_analysis_{timestamp}_alt.xlsx"
                filepath = os.path.join(self.excel_dir, new_filename)
                
                with pd.ExcelWriter(filepath, engine='openpyxl', mode='w') as writer:
                    if not df.empty:
                        df.to_excel(writer, sheet_name='All_Analysis', index=False)
                    if market_regime:
                        self._create_market_regime_sheet(writer, market_regime)
            
            logger.info(f"Excel output saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating Excel output: {e}")
            return ""

    def _create_market_regime_sheet(self, writer, market_regime: Dict):
        """Create comprehensive market regime analysis sheet (RESTORED FULL VERSION)"""
        try:
            # Create market regime summary data
            regime_data = []
            
            # Section 1: 6-Line Market Intelligence Summary
            regime_data.append(['MARKET REGIME SUMMARY', '', '', ''])
            regime_data.append(['=' * 50, '', '', ''])
            
            # Format the 6-line display for Excel (Import here to avoid circular imports)
            try:
                from modules.market_regime_analyzer import MarketRegimeAnalyzer
                temp_analyzer = MarketRegimeAnalyzer()
                regime_display = temp_analyzer.format_regime_display(market_regime)
                
                # Split the 6-line display and add to Excel
                for line in regime_display.split('\n'):
                    regime_data.append([line, '', '', ''])
            except Exception as e:
                logger.warning(f"Could not format regime display: {str(e)}")
                regime_data.append(['Market regime display unavailable', '', '', ''])
            
            regime_data.append(['', '', '', ''])
            regime_data.append(['DETAILED TECHNICAL METRICS', '', '', ''])
            regime_data.append(['=' * 50, '', '', ''])
            
            # Section 2: Alt Technical Analysis Details
            regime_data.append(['Alt Technical Analysis', '', '', ''])
            regime_data.append(['Metric', 'Value', 'Description', 'Impact'])
            regime_data.append(['Trend Strength (ADX)', market_regime.get('alt_trend_strength', 'N/A'), 'Average Directional Index', 'Higher = Stronger Trend'])
            regime_data.append(['Trend Direction', market_regime.get('alt_trend_direction', 'N/A'), 'SMA 20 vs SMA 50 Direction', 'UP/DOWN/NEUTRAL'])
            regime_data.append(['Volatility Regime', market_regime.get('alt_volatility_regime', 'N/A'), 'Current ATR vs Historical', 'HIGH/NORMAL/LOW'])
            regime_data.append(['Volume Trend', market_regime.get('alt_volume_trend', 'N/A'), 'Volume Momentum', 'STRONG/AVERAGE/WEAK'])
            regime_data.append(['BB Squeeze Phase', market_regime.get('bb_squeeze_phase', 'N/A'), 'Bollinger Band Width', 'True = Breakout Pending'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 3: BTC Intelligence
            regime_data.append(['BTC Technical & Sentiment Analysis', '', '', ''])
            regime_data.append(['Metric', 'Value', 'Description', 'Impact'])
            regime_data.append(['BTC Trend', market_regime.get('btc_trend', 'N/A'), 'BTC Technical Direction', 'BULLISH/BEARISH/NEUTRAL'])
            regime_data.append(['BTC Technical Confidence', f"{market_regime.get('btc_technical_confidence', 50):.1f}%", 'Technical Analysis Confidence', '0-100% Score'])
            regime_data.append(['BTC Sentiment Confidence', f"{market_regime.get('btc_sentiment_confidence', 50):.1f}%", 'Sentiment Analysis Confidence', '0-100% Score'])
            regime_data.append(['BTC Health Score', f"{market_regime.get('btc_health_score', 50):.1f}", 'Composite BTC Health', 'Technical 60% + Sentiment 40%'])
            regime_data.append(['BTC ADX', f"{market_regime.get('btc_adx', 20):.1f}", 'BTC Trend Strength', 'Higher = Stronger Trend'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 4: BTC Sentiment Breakdown
            regime_data.append(['BTC Sentiment Breakdown', '', '', ''])
            regime_data.append(['Source', 'Score', 'Description', 'Range'])
            regime_data.append(['LunarCrush Galaxy', market_regime.get('btc_galaxy_score', 50), 'Social + Price Intelligence', '0-100 (50+ Bullish)'])
            regime_data.append(['TokenMetrics TM Grade', f"{market_regime.get('btc_tm_grade', 50):.1f}", 'AI Trader Grade', '0-100 (70+ Strong Buy)'])
            regime_data.append(['TokenMetrics TA Grade', f"{market_regime.get('btc_ta_grade', 50):.1f}", 'Technical Analysis Grade', '0-100 (70+ Strong)'])
            regime_data.append(['TokenMetrics Quant Grade', f"{market_regime.get('btc_quant_grade', 50):.1f}", 'Quantitative Analysis', '0-100 (70+ Strong)'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 5: Wider Market Context
            regime_data.append(['Wider Market Intelligence', '', '', ''])
            regime_data.append(['Metric', 'Value', 'Description', 'Interpretation'])
            regime_data.append(['Fear & Greed Index', market_regime.get('fear_greed_index', 50), 'Market Sentiment 0-100', '<30 Fear, >70 Greed'])
            regime_data.append(['Fear & Greed Class', market_regime.get('fear_greed_classification', 'Neutral'), 'Sentiment Classification', 'Extreme Fear to Extreme Greed'])
            regime_data.append(['BTC Dominance', f"{market_regime.get('btc_dominance', 50):.1f}%", 'BTC Market Share', '<40% Alt Season, >60% BTC Season'])
            regime_data.append(['BTC Dominance Trend', market_regime.get('btc_dominance_trend', 'STABLE'), 'Dominance Direction', 'INCREASING/DECREASING/STABLE'])
            regime_data.append(['Market Health Score', f"{market_regime.get('market_health_score', 50):.1f}", 'Composite Market Health', '0-100 (Higher = Healthier)'])
            regime_data.append(['Alt Season Indicator', market_regime.get('alt_season_indicator', 'NEUTRAL'), 'Alt Market Conditions', 'ALT_SEASON/ALT_FAVORABLE/BTC_SEASON/NEUTRAL'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 6: Alt Market Analysis
            regime_data.append(['Alt Market Trends', '', '', ''])
            regime_data.append(['Metric', 'Value', 'Description', 'Impact'])
            regime_data.append(['Alt Market Cap Trend', market_regime.get('alt_market_cap_trend', 'NEUTRAL'), 'Alt vs BTC Market Cap', 'RISING/DECLINING/STABLE'])
            regime_data.append(['Alt Correlation Index', f"{market_regime.get('alt_correlation_index', 0.75):.2f}", 'Alt Synchronization', '0.6-0.9 (Higher = More Correlated)'])
            regime_data.append(['Alt Volatility Index', market_regime.get('alt_volatility_index', 'NORMAL'), 'Alt Market Volatility', 'HIGH/NORMAL/LOW'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 7: Traditional Markets
            regime_data.append(['Traditional Markets', '', '', ''])
            regime_data.append(['Metric', 'Value', 'Description', 'Impact'])
            regime_data.append(['SPY Trend', market_regime.get('spy_trend', 'UNKNOWN'), 'S&P 500 Direction', 'RISING/FALLING/STABLE'])
            regime_data.append(['SPY Change', f"{market_regime.get('spy_change', 0):.2f}%", 'Daily Change %', 'Market Performance'])
            regime_data.append(['VIX Level', f"{market_regime.get('vix_price', 20):.1f}", 'Fear Gauge', '<20 Low Fear, >30 High Fear'])
            regime_data.append(['DXY Change', f"{market_regime.get('dxy_change', 0):.2f}%", 'Dollar Index Change', 'Strong Dollar = Risk Off'])
            regime_data.append(['QQQ Change', f"{market_regime.get('qqq_change', 0):.2f}%", 'Tech Sector Performance', 'Nasdaq Tech Health'])
            regime_data.append(['Market Environment', market_regime.get('market_environment', 'NEUTRAL'), 'Risk Environment', 'RISK_ON/RISK_OFF/NEUTRAL'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 8: Position Sizing & Risk
            regime_data.append(['Position Sizing & Risk Management', '', '', ''])
            regime_data.append(['Metric', 'Value', 'Description', 'Application'])
            regime_data.append(['Regime Confidence', f"{market_regime.get('regime_confidence', 50):.1f}%", 'Overall Regime Confidence', 'Higher = More Confident'])
            regime_data.append(['BB Strategy Suitability', market_regime.get('bb_suitability', 'FAIR'), 'BB Bounce Strategy Fit', 'EXCELLENT/GOOD/FAIR/POOR'])
            regime_data.append(['Position Multiplier', f"{market_regime.get('position_multiplier', 1.0):.2f}x", 'Dynamic Position Sizing', '0.5x-1.5x (Applied to All Trades)'])
            regime_data.append(['Alt Market Outlook', market_regime.get('alt_market_outlook', 'FAIR'), 'BTC Impact on Alts', 'EXCELLENT/GOOD/FAIR/POOR'])
            
            regime_data.append(['', '', '', ''])
            
            # Section 9: Analysis Metadata
            regime_data.append(['Analysis Information', '', '', ''])
            regime_data.append(['Timestamp', market_regime.get('analysis_timestamp', 'N/A'), 'Analysis Time', ''])
            regime_data.append(['Symbol Analyzed', market_regime.get('symbol_analyzed', 'N/A'), 'Market Proxy Used', 'ETH as Alt Market Representative'])
            
            # Convert to DataFrame and save
            regime_df = pd.DataFrame(regime_data, columns=['Metric', 'Value', 'Description', 'Notes'])
            regime_df.to_excel(writer, sheet_name='Market_Regime_Analysis', index=False)
            
            logger.info("Market regime analysis sheet created successfully")
            
        except Exception as e:
            logger.error(f"Error creating market regime sheet: {str(e)}")
            # Create minimal sheet on error
            try:
                error_df = pd.DataFrame([
                    ['Error', 'Market regime sheet creation failed'],
                    ['Details', str(e)[:100]]  # Truncate error message
                ], columns=['Status', 'Message'])
                error_df.to_excel(writer, sheet_name='Market_Regime_Analysis', index=False)
            except:
                pass  # If even error sheet fails, just skip

    def format_comprehensive_results(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Format comprehensive results - alias for format_results_dataframe"""
        return self.format_results_dataframe(all_results)

    def display_market_sentiment(self, market_sentiment: Dict[str, Any]):
        """Display market sentiment before main analysis - public method"""
        try:
            print(f"\n" + "="*80)
            print("🌍 CRYPTO MARKET SENTIMENT CHECK")
            print("="*80)
            
            if not market_sentiment:
                print("Market sentiment data unavailable")
                return
            
            # Fear & Greed Index
            fng = market_sentiment.get('fear_greed', {})
            if fng.get('available'):
                fng_emoji = "🟢" if fng['signal'] in ['Bullish Signal', 'Positive'] else \
                           "🟡" if fng['signal'] == 'Neutral' else \
                           "🟠" if fng['signal'] == 'Caution' else "🔴"
                print(f"😨 Fear & Greed Index: {fng['value']}/100 ({fng['classification']}) {fng_emoji}")
                print(f"   Signal: {fng['signal']}")
            
            # BTC Dominance
            btc_dom = market_sentiment.get('btc_dominance', {})
            if btc_dom.get('available'):
                dom_emoji = "🟢" if btc_dom['signal'] in ['Bullish Signal', 'Positive'] else \
                           "🟡" if btc_dom['signal'] == 'Neutral' else \
                           "🟠" if btc_dom['signal'] == 'Caution' else "🔴"
                change_dir = "↗️" if btc_dom['change_24h'] > 0 else "↘️" if btc_dom['change_24h'] < 0 else "→"
                print(f"₿ BTC Dominance: {btc_dom['current']}% ({btc_dom['change_24h']:+.1f}% 24h) {change_dir} {dom_emoji}")
                print(f"   Signal: {btc_dom['signal']}")
            
            # BTC Technical
            btc_tech = market_sentiment.get('btc_technical', {})
            if btc_tech.get('available'):
                tech_emoji = "🟢" if btc_tech['signal'] in ['Bullish Signal', 'Positive'] else \
                            "🟡" if btc_tech['signal'] == 'Neutral' else \
                            "🟠" if btc_tech['signal'] == 'Caution' else "🔴"
                ma_status = "Above" if btc_tech.get('above_sma20', True) else "Below"
                print(f"📊 BTC Technical: ${btc_tech.get('price', 0):,.0f} | RSI: {btc_tech.get('rsi', 50)} | {ma_status} SMA20 {tech_emoji}")
                print(f"   Signal: {btc_tech['signal']}")
            
            # Overall Risk
            risk = market_sentiment.get('overall_risk', 'NEUTRAL')
            if risk == "HIGH RISK":
                risk_emoji = "🚨"
                risk_message = "Consider waiting or using smaller positions"
            elif risk == "MEDIUM RISK":
                risk_emoji = "⚠️"
                risk_message = "Use caution - reduced position sizes recommended"
            elif risk == "LOW RISK":
                risk_emoji = "✅"
                risk_message = "Good environment for normal position sizes"
            else:
                risk_emoji = "➖"
                risk_message = "Mixed signals - use standard risk management"
            
            print(f"\n🎯 OVERALL MARKET RISK: {risk} {risk_emoji}")
            print(f"💡 Recommendation: {risk_message}")
            
        except Exception as e:
            logger.error(f"Error displaying market sentiment: {e}")
            print("Market sentiment analysis unavailable")

    def display_terminal_summary(self, df: pd.DataFrame, market_sentiment: Dict[str, Any] = None):
        """Display comprehensive terminal summary"""
        try:
            print("\n" + "="*80)
            print("CRYPTO BB BOUNCE SCANNER - ANALYSIS SUMMARY")
            print("="*80)
            print(f"Total coins analyzed: {len(df)}")
            
            if df.empty:
                print("No analysis results found.")
                return
            
            # Tier breakdown
            tier_counts = df['tier'].value_counts()
            for tier in ['PREMIUM', 'HIGH', 'GOOD', 'FAIR', 'MARGINAL', 'WEAK']:
                count = tier_counts.get(tier, 0)
                percentage = (count / len(df) * 100) if len(df) > 0 else 0
                print(f"{tier}: {count} coins ({percentage:.1f}%)")
            
            # Risk analysis
            if len(df) > 0:
                low_risk_count = len(df[df['risk_pct'] <= 2.5])
                med_risk_count = len(df[(df['risk_pct'] > 2.5) & (df['risk_pct'] <= 4.0)])
                high_risk_count = len(df[df['risk_pct'] > 4.0])
                
                print(f"\nRisk Analysis:")
                print(f"Low Risk (≤2.5%): {low_risk_count} trades")
                print(f"Medium Risk (2.5-4%): {med_risk_count} trades")
                print(f"High Risk (>4%): {high_risk_count} trades")
            
            # Show premium trades
            self._display_premium_trades(df)
            
            # Show good trades
            self._display_good_trades(df)
            
            # Display market sentiment if available
            if market_sentiment:
                self._display_market_sentiment(market_sentiment)
            
            # Enhanced recommendations
            self._display_recommendations(df)
            
        except Exception as e:
            logger.error(f"Error displaying terminal summary: {e}")

    def _display_premium_trades(self, df: pd.DataFrame):
        """Display premium trades (70%+ probability)"""
        try:
            premium_trades = df[df['probability'] >= 70]
            print(f"\n" + "="*80)
            print(f"PREMIUM TRADES (70%+ Probability): {len(premium_trades)}")
            print("="*80)
            
            if len(premium_trades) > 0:
                for i, (_, trade) in enumerate(premium_trades.head(10).iterrows(), 1):
                    div_info = f" | Div: {trade['divergence_indicators']}" if trade.get('divergence_detected', False) else " | No divergence"
                    
                    # Risk indicators
                    if trade['risk_pct'] <= 3.0:
                        risk_flag = "✅"
                    elif trade['risk_pct'] <= 6.0:
                        risk_flag = "⚠️"
                    else:
                        risk_flag = "🔴"
                    
                    print(f"\n{i}. {trade['symbol']} - {trade['setup_type']} ({trade['exchange']})")
                    print(f"   🎯 Probability: {trade['probability']}% ({trade['tier']}) | BB Score: {trade['bb_score']}/11")
                    if trade['entry'] > 0:
                        print(f"   💰 Entry: ${trade['entry']:.6f} | Stop: ${trade['stop']:.6f} | Target: ${trade['target1']:.6f}")
                        print(f"   📊 R:R: {trade['risk_reward']}:1 | Risk: {trade['risk_pct']}% {risk_flag} | Gain: {trade['gain_pct']}%")
                    
                    # Get pattern info if available
                    if trade.get('patterns_detected') and trade['patterns_detected'] != 'None':
                        patterns = trade['patterns_detected'].split(', ')[:2]  # Show max 2 patterns
                        pattern_info = f" | 🕯️ {', '.join(patterns)}"
                    else:
                        pattern_info = f" | 🕯️ N/A"

                    # Get chart pattern info if available
                    chart_pattern_info = ""
                    chart_patterns = trade.get('chart_patterns_detected', 'None')
                    if chart_patterns and chart_patterns != 'None':
                        chart_pattern_info = f" | 📈 {chart_patterns}"
                    else:
                        chart_pattern_info = f" | 📈 N/A"

                    print(f"🔍 RSI: {trade['rsi']:.1f} | BB%: {trade['bb_pct']:.3f} | Vol: {trade['volume_ratio']:.1f}x{chart_pattern_info} | {div_info}{pattern_info}")
            else:
                print("No premium trades found.")
                
        except Exception as e:
            logger.error(f"Error displaying premium trades: {e}")

    def _display_good_trades(self, df: pd.DataFrame):
        """Display good trades (65-69% probability)"""
        try:
            good_trades = df[(df['probability'] >= 65) & (df['probability'] < 70)]
            if len(good_trades) > 0:
                print(f"\n" + "="*80)
                print(f"GOOD TRADES (65-69% Probability): {len(good_trades)}")
                print("="*80)
                
                for i, (_, trade) in enumerate(good_trades.head(5).iterrows(), 1):
                    div_info = f" | Div: {trade['divergence_indicators']}" if trade.get('divergence_detected', False) else ""
                    
                    if trade['risk_pct'] <= 3.0:
                        risk_flag = "✅"
                    elif trade['risk_pct'] <= 6.0:
                        risk_flag = "⚠️"
                    else:
                        risk_flag = "🔴"
                        
                    print(f"{i}. {trade['symbol']} - {trade['setup_type']} | {trade['probability']}% | Risk: {trade['risk_pct']}% {risk_flag}{div_info}")
                    
        except Exception as e:
            logger.error(f"Error displaying good trades: {e}")

    def _display_market_sentiment(self, market_sentiment: Dict[str, Any]):
        """Display market sentiment analysis"""
        try:
            print(f"\n" + "="*80)
            print("🌍 CRYPTO MARKET SENTIMENT")
            print("="*80)
            
            # Fear & Greed Index
            fng = market_sentiment.get('fear_greed', {})
            if fng.get('available'):
                fng_emoji = "🟢" if fng['signal'] in ['Bullish Signal', 'Positive'] else \
                           "🟡" if fng['signal'] == 'Neutral' else \
                           "🟠" if fng['signal'] == 'Caution' else "🔴"
                print(f"😨 Fear & Greed Index: {fng['value']}/100 ({fng['classification']}) {fng_emoji}")
                print(f"   Signal: {fng['signal']}")
            
            # BTC Dominance
            btc_dom = market_sentiment.get('btc_dominance', {})
            if btc_dom.get('available'):
                dom_emoji = "🟢" if btc_dom['signal'] in ['Bullish Signal', 'Positive'] else \
                           "🟡" if btc_dom['signal'] == 'Neutral' else \
                           "🟠" if btc_dom['signal'] == 'Caution' else "🔴"
                change_dir = "↗️" if btc_dom['change_24h'] > 0 else "↘️" if btc_dom['change_24h'] < 0 else "→"
                print(f"₿ BTC Dominance: {btc_dom['current']}% ({btc_dom['change_24h']:+.1f}% 24h) {change_dir} {dom_emoji}")
                print(f"   Signal: {btc_dom['signal']}")
                
        except Exception as e:
            logger.error(f"Error displaying market sentiment: {e}")

    def _display_recommendations(self, df: pd.DataFrame):
        """Display trading recommendations"""
        try:
            print(f"\n" + "="*80)
            print("TRADING RECOMMENDATIONS")
            print("="*80)
            
            take_trades = len(df[df['action'] == 'TAKE TRADE'])
            consider_trades = len(df[df['action'] == 'CONSIDER'])
            
            if take_trades > 0:
                print(f"🚀 {take_trades} PREMIUM trades ready")
                print("   - Use 1.5-2% position size")
                print("   - Expected duration: 2-7 days")
                print("   - Stops: 2x ATR (fewer false exits)")
                
            if consider_trades > 0:
                print(f"⭐ {consider_trades} GOOD trades to consider")
                print("   - Use 1% position size")
                print("   - Review risk levels carefully")
                
            if take_trades + consider_trades == 0:
                print("🔍 No high-probability trades found")
                print("   - Market may be trending strongly")
                print("   - Consider waiting for better market conditions")
            
            print(f"\n📋 EXECUTION PRIORITY:")
            print("1. Premium trades with divergence confirmation")
            print("2. High probability trades with good R:R")
            print("3. Consider risk tolerance: ✅≤3% ⚠️3-6% 🔴>6%")
            print("4. All trades have 1H confirmation + current price validation")
            
        except Exception as e:
            logger.error(f"Error displaying recommendations: {e}")

    def display_sentiment_summary(self, df: pd.DataFrame):
        """Display sentiment analysis summary for trades with sentiment data"""
        try:
            # Check if sentiment columns exist
            sentiment_cols = ['lunar_data_available', 'tm_data_available']
            if not any(col in df.columns for col in sentiment_cols):
                return
            
            # Get trades with sentiment data
            sentiment_trades = df[
                (df.get('lunar_data_available', False) == True) | 
                (df.get('tm_data_available', False) == True)
            ].head(10)
            
            if sentiment_trades.empty:
                print("\nNo sentiment data available for trades")
                return
            
            print(f"\n" + "="*80)
            print("SENTIMENT ANALYSIS SUMMARY")
            print("="*80)
            
            for i, (_, trade) in enumerate(sentiment_trades.iterrows(), 1):
                alignment = trade.get('sentiment_overall_alignment', 'No Data')
                sentiment_status = "📈" if alignment in ['Strong Positive', 'Positive'] else \
                                 "📉" if alignment in ['Strong Negative', 'Negative'] else "➖"
                
                print(f"\n{i}. {trade['symbol']} - {trade['setup_type']} | {trade['probability']}% Probability")
                
                # Show LunarCrush data if available
                if trade.get('lunar_data_available'):
                    lunar_rating = trade.get('lunar_sentiment_rating', 'No Data')
                    lunar_score = trade.get('lunar_sentiment_score', 0)
                    print(f"   🌙 LunarCrush: {lunar_rating} (Score: {lunar_score})")
                
                # Show TokenMetrics data if available (ENHANCED VERSION)
                if trade.get('tm_data_available'):
                    tm_grade = trade.get('tm_trader_grade', 0)
                    tm_ta_grade = trade.get('tm_ta_grade', 0)
                    tm_quant_grade = trade.get('tm_quant_grade', 0)
                    tm_change = trade.get('tm_grade_change_24h', 0)
                    change_arrow = "↗️" if tm_change > 0 else "↘️" if tm_change < 0 else "→"
                    
                    # Get enhanced descriptions using the scoring ranges
                    # TM Trader Grade descriptions
                    if tm_grade >= 80:
                        tm_desc = "✅ Excellent trading opportunity"
                    elif tm_grade >= 60:
                        tm_desc = "📈 Good trading opportunity"
                    elif tm_grade >= 40:
                        tm_desc = "📊 Average/Neutral"
                    elif tm_grade >= 20:
                        tm_desc = "⚠️ Below average"
                    else:
                        tm_desc = "❌ Poor trading opportunity"
                    
                    # TA Grade descriptions
                    if tm_ta_grade >= 80:
                        ta_desc = "✅ Strong technical signals"
                    elif tm_ta_grade >= 60:
                        ta_desc = "📈 Good technical setup"
                    elif tm_ta_grade >= 40:
                        ta_desc = "📊 Neutral technical picture"
                    elif tm_ta_grade >= 20:
                        ta_desc = "⚠️ Weak technicals"
                    else:
                        ta_desc = "❌ Poor technical setup"
                    
                    # Quant Grade descriptions
                    if tm_quant_grade >= 60:
                        quant_desc = "✅ High"
                    elif tm_quant_grade >= 40:
                        quant_desc = "📊 Neutral"
                    else:
                        quant_desc = "⚠️ Low"
                    
                    print(f"   📊 TokenMetrics: Trader Grade {tm_desc} {tm_grade} | TA Grade {ta_desc} {tm_ta_grade} | Quant Grade {quant_desc} {tm_quant_grade} | 24h: {tm_change:+.1f}% {change_arrow}")
                
                # Show alignment if available
                if alignment != 'No Data':
                    print(f"   {sentiment_status} Sentiment Alignment: {alignment}")
                    if trade.get('sentiment_alignment_factors'):
                        print(f"   💡 Factors: {trade['sentiment_alignment_factors']}")
                        
        except Exception as e:
            logger.error(f"Error displaying sentiment summary: {e}")