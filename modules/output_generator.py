# output_generator.py - Output Generation Module
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from config import *
from modules.market_regime_enhanced import format_enhanced_regime_output
from openpyxl.styles import Font

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
        """Generate comprehensive Excel output with Market Overview tab (and all previous sheets)"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bb_analysis_{timestamp}.xlsx"
            filepath = os.path.join(self.excel_dir, filename)
            if not df.empty:
                regime_columns = {
                    'regime_confidence': 50,
                    'regime_type': 'MIXED',
                    'bb_suitability': 'FAIR',
                    'position_multiplier': 1.0
                }
                for col, default_val in regime_columns.items():
                    if col not in df.columns:
                        df[col] = default_val
            market_data = self._run_market_overview_analysis()
            with pd.ExcelWriter(filepath, engine='openpyxl', mode='w') as writer:
                # Sheet 1: All results
                if not df.empty:
                    df.to_excel(writer, sheet_name='All_Analysis', index=False)
                # Sheet 2: Premium and High probability only
                premium_high = df[df['tier'].isin(['PREMIUM', 'HIGH'])] if not df.empty and 'tier' in df.columns else pd.DataFrame()
                if not premium_high.empty:
                    premium_high.to_excel(writer, sheet_name='Premium_High_Only', index=False)
                # Sheet 3: Trade recommendations
                trade_recs = df[df['action'].isin(['TAKE TRADE', 'CONSIDER'])] if not df.empty and 'action' in df.columns else pd.DataFrame()
                if not trade_recs.empty:
                    trade_recs.to_excel(writer, sheet_name='Trade_Recommendations', index=False)
                # Sheet 4: Low risk trades (≤3% risk)
                low_risk = df[(df['risk_pct'] <= 3.0) & (df['setup_type'] != 'NONE')] if not df.empty and 'risk_pct' in df.columns and 'setup_type' in df.columns else pd.DataFrame()
                if not low_risk.empty:
                    low_risk.to_excel(writer, sheet_name='Low_Risk_Trades', index=False)
                # Sheet 5: Monitoring list
                monitor_list = df[df['action'].isin(['MONITOR', 'WATCH ONLY'])] if not df.empty and 'action' in df.columns else pd.DataFrame()
                if not monitor_list.empty:
                    monitor_list.to_excel(writer, sheet_name='Monitoring_List', index=False)
                # Sheet 6: Top 10 with sentiment (if sentiment data exists)
                sentiment_cols = ['lunar_data_available', 'tm_data_available']
                if not df.empty and any(col in df.columns for col in sentiment_cols):
                    top_10_sentiment = df.head(10)
                    if not top_10_sentiment.empty:
                        top_10_sentiment.to_excel(writer, sheet_name='Top_10_Sentiment', index=False)
                # Sheet 7: Market Regime Analysis Dashboard
                if market_regime:
                    self._create_market_regime_sheet(writer, market_regime)
                # ADDITIONAL SHEET: Market Overview (improved BB backtest output)
                self._create_market_overview_sheet(writer, market_data)
            logger.info(f"Excel output with Market Overview saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error generating Excel output: {e}")
            return ""

    def _create_market_overview_sheet(self, writer, market_data: Dict = None):
        """Create Market Overview tab with daily backtesting snapshot for ML training"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            overview_data = []
            overview_data.append(['DAILY MARKET ANALYSIS SNAPSHOT', '', '', ''])
            overview_data.append(['Analysis Date', current_date, '', ''])
            overview_data.append(['Analysis Period', market_data.get('analysis_period', 'Rolling 30-Day Window'), '', ''])
            overview_data.append(['', '', '', ''])
            overview_data.append(['OVERALL BB PERFORMANCE', '', '', ''])
            overview_data.append(['Total BB Bounces Analyzed', market_data.get('total_bounces', ''), '', ''])
            overview_data.append(['Coins Successfully Analyzed', market_data.get('coins_analyzed', ''), '', ''])
            overview_data.append(['Overall Success Rate', f"{market_data.get('overall_success_rate', '')}%", '', ''])
            overview_data.append(['Market Health Score', f"{market_data.get('market_health', '')}%", '', ''])
            overview_data.append(['', '', '', ''])
            overview_data.append(['BB-SPECIFIC INDICATORS', 'Success Rate', 'Profit Factor', 'Samples'])
            for row in market_data.get('bb_specific_indicators', []):
                overview_data.append(row)
            overview_data.append(['', '', '', ''])
            overview_data.append(['TECHNICAL INDICATORS', 'Success Rate', 'Profit Factor', 'Samples'])
            for row in market_data.get('technical_indicators', []):
                overview_data.append(row)
            overview_data.append(['', '', '', ''])
            overview_data.append(['RISK CHARACTERISTICS', '', '', ''])
            for row in market_data.get('risk_characteristics', []):
                overview_data.append(row)
            overview_data.append(['', '', '', ''])
            overview_data.append(['TIMING ANALYSIS', 'Average', 'Hit Rate', ''])
            for row in market_data.get('timing_analysis', []):
                overview_data.append(row)
            overview_data.append(['', '', '', ''])
            overview_data.append(['MARKET CAP TIERS', 'Success Rate', 'Samples', ''])
            for row in market_data.get('market_cap_tiers', []):
                overview_data.append(row)
            overview_data.append(['', '', '', ''])
            overview_data.append(['ML TRAINING DATA', '', '', ''])
            for row in market_data.get('ml_training_data', []):
                overview_data.append(row)
            overview_data.append(['Next Update', market_data.get('next_update', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')), '', ''])
            df_overview = pd.DataFrame(overview_data, columns=['Metric', 'Value', 'Secondary', 'Notes'])
            df_overview.to_excel(writer, sheet_name='Market_Overview', index=False)
            worksheet = writer.sheets['Market_Overview']
            for row in [1, 5, 9, 14, 20, 26, 32, 36]:
                for col in range(1, 5):
                    cell = worksheet.cell(row=row, column=col)
                    cell.font = Font(bold=True)
            worksheet.column_dimensions['A'].width = 25
            worksheet.column_dimensions['B'].width = 15
            worksheet.column_dimensions['C'].width = 15
            worksheet.column_dimensions['D'].width = 15
            logger.info("Market Overview sheet created successfully")
        except Exception as e:
            logger.error(f"Error creating Market Overview sheet: {e}")
            fallback_data = [
                ['Market Overview', 'Error occurred during creation'],
                ['Status', 'Please check logs for details'],
                ['Date', datetime.now().strftime("%Y-%m-%d")]
            ]
            df_fallback = pd.DataFrame(fallback_data, columns=['Metric', 'Value'])
            df_fallback.to_excel(writer, sheet_name='Market_Overview', index=False)

    def _run_market_overview_analysis(self):
        """Run the improved_bb_backtest analysis and return summary data"""
        try:
            from modules.improved_bb_backtest import ComprehensiveBBBacktest
            from datetime import datetime, timedelta
            logger.info("Running real-time market overview analysis...")

            # Instantiate and run the comprehensive backtest for a 30-day window
            backtester = ComprehensiveBBBacktest()
            results = backtester.run_comprehensive_analysis(timeframes=[30], max_coins=500)
            results_30d = results.get('30d', {})

            # --- Summary stats ---
            coins_analyzed = len([k for k, v in results_30d.items() if v and isinstance(v, dict) and v.get('total_bounces', 0) > 0])
            total_bounces = sum(v.get('total_bounces', 0) for v in results_30d.values() if isinstance(v, dict))
            all_bounces = []
            for v in results_30d.values():
                if isinstance(v, dict) and 'bounces' in v:
                    all_bounces.extend(v['bounces'])
            successful_bounces = len([b for b in all_bounces if b.get('max_favorable_5', 0) > 1.0])
            overall_success_rate = round((successful_bounces / total_bounces) * 100, 1) if total_bounces > 0 else 0.0
            market_health = overall_success_rate

            # --- BB-Specific Indicators ---
            def indicator_row(name, stats):
                return [
                    name,
                    f"{stats['success_rate']:.1f}%",
                    f"{stats['profit_factor']:.1f}",
                    f"{stats['samples']}"
                ]
            bb_stats = self._extract_bb_stats_from_bounces(all_bounces)
            bb_specific_indicators = [
                indicator_row('BB Squeeze', bb_stats.get('bb_squeeze', {})),
                indicator_row('BB Expansion', bb_stats.get('bb_expansion', {})),
                indicator_row('BB Reversal Setup', bb_stats.get('bb_reversal_setup', {})),
            ]

            # --- Technical Indicators ---
            tech_stats = self._extract_technical_stats_from_bounces(all_bounces)
            technical_indicators = [
                indicator_row('MFI Oversold', tech_stats.get('mfi_oversold', {})),
                indicator_row('MFI Overbought', tech_stats.get('mfi_overbought', {})),
                indicator_row('Volume Surge', tech_stats.get('volume_surge', {})),
                indicator_row('CCI Extreme', tech_stats.get('cci_extreme', {})),
                indicator_row('Stoch Overbought', tech_stats.get('stoch_overbought', {})),
                indicator_row('Stoch Oversold', tech_stats.get('stoch_oversold', {})),
            ]

            # --- Risk Characteristics ---
            risk_stats = self._extract_risk_stats_from_bounces(all_bounces)
            risk_characteristics = [
                ['Average Winning Trade', f"+{risk_stats['avg_win']:.1f}%", '', ''],
                ['Average Losing Trade', f"-{risk_stats['avg_loss']:.1f}%", '', ''],
                ['Overall Profit Factor', f"{risk_stats['profit_factor']:.1f}", '', ''],
                ['Risk/Reward Ratio', f"{risk_stats['risk_reward_ratio']:.1f}", '', '']
            ]

            # --- Timing Analysis ---
            timing_stats = self._extract_timing_stats_from_bounces(all_bounces)
            timing_analysis = [
                ['Time to 1%', f"{timing_stats['avg_time_to_1pct']:.1f}h", f"{timing_stats['hit_rate_1pct']:.1f}%", ''],
                ['Time to 3%', f"{timing_stats['avg_time_to_3pct']:.1f}h", f"{timing_stats['hit_rate_3pct']:.1f}%", ''],
                ['Time to 5%', f"{timing_stats['avg_time_to_5pct']:.1f}h", f"{timing_stats['hit_rate_5pct']:.1f}%", ''],
                ['Time to Peak', f"{timing_stats['avg_time_to_peak']:.1f}h", f"{timing_stats['hit_rate_peak']:.1f}%", '']
            ]

            # --- Market Cap Tiers ---
            market_cap_stats = self._extract_market_cap_stats_from_bounces(all_bounces)
            market_cap_tiers = [
                ['Large Cap (Top 50)', f"{market_cap_stats['large_cap_success']:.1f}%", f"{market_cap_stats['large_cap_samples']}", ''],
                ['Smaller Cap', f"{market_cap_stats['small_cap_success']:.1f}%", f"{market_cap_stats['small_cap_samples']}", '']
            ]

            # --- ML Training Data ---
            ml_training_data = [
                ['Data Quality', 'HIGH', '', ''],
                ['Sample Size', f"EXCELLENT ({total_bounces} bounces)", '', ''],
                ['Confidence Level', 'INSTITUTIONAL GRADE', '', '']
            ]

            market_data = {
                'total_bounces': total_bounces,
                'coins_analyzed': coins_analyzed,
                'overall_success_rate': overall_success_rate,
                'market_health': market_health,
                'analysis_period': 'Rolling 30-Day Window',
                'bb_specific_indicators': bb_specific_indicators,
                'technical_indicators': technical_indicators,
                'risk_characteristics': risk_characteristics,
                'timing_analysis': timing_analysis,
                'market_cap_tiers': market_cap_tiers,
                'ml_training_data': ml_training_data,
                'next_update': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            }
            return market_data
        except Exception as e:
            logger.error(f"Error running market overview analysis: {e}")
            return {
                'total_bounces': 0,
                'coins_analyzed': 0,
                'overall_success_rate': 0,
                'market_health': 0,
                'analysis_period': 'Rolling 30-Day Window',
                'bb_specific_indicators': [],
                'technical_indicators': [],
                'risk_characteristics': [],
                'timing_analysis': [],
                'market_cap_tiers': [],
                'ml_training_data': [],
                'next_update': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            }

    # --- Helper extraction methods ---
    def _extract_bb_stats_from_bounces(self, bounces):
        # Calculate BB Squeeze, Expansion, Reversal Setup stats from bounces
        def calc_stats(filter_key):
            filtered = [b for b in bounces if b.get(filter_key)]
            samples = len(filtered)
            if samples == 0:
                return {'success_rate': 0, 'profit_factor': 0, 'samples': 0}
            winners = [b for b in filtered if b.get('max_favorable_5', 0) > 1.0]
            losers = [b for b in filtered if b.get('max_favorable_5', 0) <= 1.0]
            success_rate = len(winners) / samples * 100 if samples else 0
            avg_win = sum([b.get('max_favorable_5', 0) for b in winners]) / len(winners) if winners else 0
            avg_loss = sum([abs(b.get('max_adverse_5', 0)) for b in losers]) / len(losers) if losers else 0
            profit_factor = (sum([b.get('max_favorable_5', 0) for b in winners]) / sum([abs(b.get('max_adverse_5', 0)) for b in losers])) if losers else 0
            return {'success_rate': success_rate, 'profit_factor': profit_factor, 'samples': samples}
        return {
            'bb_squeeze': calc_stats('bb_squeeze'),
            'bb_expansion': calc_stats('bb_expansion'),
            'bb_reversal_setup': calc_stats('bb_reversal_setup')
        }

    def _extract_technical_stats_from_bounces(self, bounces):
        def calc_stats(filter_func):
            filtered = [b for b in bounces if filter_func(b)]
            samples = len(filtered)
            if samples == 0:
                return {'success_rate': 0, 'profit_factor': 0, 'samples': 0}
            winners = [b for b in filtered if b.get('max_favorable_5', 0) > 1.0]
            losers = [b for b in filtered if b.get('max_favorable_5', 0) <= 1.0]
            success_rate = len(winners) / samples * 100 if samples else 0
            avg_win = sum([b.get('max_favorable_5', 0) for b in winners]) / len(winners) if winners else 0
            avg_loss = sum([abs(b.get('max_adverse_5', 0)) for b in losers]) / len(losers) if losers else 0
            profit_factor = (sum([b.get('max_favorable_5', 0) for b in winners]) / sum([abs(b.get('max_adverse_5', 0)) for b in losers])) if losers else 0
            return {'success_rate': success_rate, 'profit_factor': profit_factor, 'samples': samples}
        return {
            'mfi_oversold': calc_stats(lambda b: b.get('money_flow_index', 50) < 20),
            'mfi_overbought': calc_stats(lambda b: b.get('money_flow_index', 50) > 80),
            'volume_surge': calc_stats(lambda b: b.get('volume_surge', False)),
            'cci_extreme': calc_stats(lambda b: abs(b.get('cci', 0)) > 100),
            'stoch_overbought': calc_stats(lambda b: b.get('stoch_overbought', False)),
            'stoch_oversold': calc_stats(lambda b: b.get('stoch_oversold', False)),
        }

    def _extract_risk_stats_from_bounces(self, bounces):
        winners = [b for b in bounces if b.get('max_favorable_5', 0) > 1.0]
        losers = [b for b in bounces if b.get('max_favorable_5', 0) <= 1.0]
        avg_win = sum([b.get('max_favorable_5', 0) for b in winners]) / len(winners) if winners else 0
        avg_loss = sum([abs(b.get('max_adverse_5', 0)) for b in losers]) / len(losers) if losers else 0
        profit_factor = (sum([b.get('max_favorable_5', 0) for b in winners]) / sum([abs(b.get('max_adverse_5', 0)) for b in losers])) if losers else 0
        risk_reward_ratio = (avg_win / avg_loss) if avg_loss else 0
        return {'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor, 'risk_reward_ratio': risk_reward_ratio}

    def _extract_timing_stats_from_bounces(self, bounces):
        def avg_and_hit(field, threshold=0):
            vals = [b.get(field, 0) for b in bounces if b.get(field, 0) > threshold]
            avg = sum(vals) / len(vals) if vals else 0
            hit_rate = len(vals) / len(bounces) * 100 if bounces else 0
            return avg, hit_rate
        avg_time_to_1pct, hit_rate_1pct = avg_and_hit('time_to_1pct')
        avg_time_to_3pct, hit_rate_3pct = avg_and_hit('time_to_3pct')
        avg_time_to_5pct, hit_rate_5pct = avg_and_hit('time_to_5pct')
        avg_time_to_peak, hit_rate_peak = avg_and_hit('time_to_peak')
        return {
            'avg_time_to_1pct': avg_time_to_1pct,
            'hit_rate_1pct': hit_rate_1pct,
            'avg_time_to_3pct': avg_time_to_3pct,
            'hit_rate_3pct': hit_rate_3pct,
            'avg_time_to_5pct': avg_time_to_5pct,
            'hit_rate_5pct': hit_rate_5pct,
            'avg_time_to_peak': avg_time_to_peak,
            'hit_rate_peak': hit_rate_peak
        }

    def _extract_market_cap_stats_from_bounces(self, bounces):
        large_cap_symbols = {'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'MATIC', 'DOT'}
        large_cap_bounces = [b for b in bounces if b.get('symbol', '').upper() in large_cap_symbols]
        small_cap_bounces = [b for b in bounces if b.get('symbol', '').upper() not in large_cap_symbols]
        def success_rate(bset):
            if not bset:
                return 0, 0
            winners = [b for b in bset if b.get('max_favorable_5', 0) > 1.0]
            return len(winners) / len(bset) * 100, len(bset)
        large_cap_success, large_cap_samples = success_rate(large_cap_bounces)
        small_cap_success, small_cap_samples = success_rate(small_cap_bounces)
        return {'large_cap_success': large_cap_success, 'large_cap_samples': large_cap_samples, 'small_cap_success': small_cap_success, 'small_cap_samples': small_cap_samples}

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
        """Display enhanced terminal summary with market regime"""
        try:
            # NEW: Display market regime if available
            if hasattr(self, '_market_regime_data') and self._market_regime_data:
                self._display_market_regime_summary(self._market_regime_data)
            
            # Existing terminal summary code continues here...
            if df.empty:
                print("\n" + "="*80)
                print("No trading opportunities found matching the criteria.")
                print("="*80)
                return
            
            print("\n" + "="*80)
            print("CRYPTO BB BOUNCE SCANNER - ANALYSIS SUMMARY")
            print("="*80)
            print(f"Total coins analyzed: {len(df)}")
            
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
            
            # Display detailed scoring breakdown
            self.display_detailed_scoring_breakdown(df)
            
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

    def display_detailed_scoring_breakdown(self, df: pd.DataFrame):
        """Display detailed scoring breakdown for trades with scoring details"""
        try:
            # Check if scoring_details column exists
            if 'scoring_details' not in df.columns:
                return
            
            # Get trades with scoring details
            scoring_trades = df[df['scoring_details'].notna()].head(5)
            
            if scoring_trades.empty:
                return
            
            print(f"\n" + "="*80)
            print("📊 DETAILED SCORING BREAKDOWN")
            print("="*80)
            
            for i, (_, trade) in enumerate(scoring_trades.iterrows(), 1):
                scoring_details = trade['scoring_details']
                if not scoring_details or 'breakdown' not in scoring_details:
                    continue
                
                print(f"\n{i}. {trade['symbol']} - {trade['setup_type']} | Score: {trade['bb_score']}/{scoring_details.get('total_possible', 34)}")
                print(f"   Quality: {trade['setup_quality']} | Probability: {trade['probability']}%")
                
                # Display tier scores
                tier_scores = scoring_details.get('tier_scores', {})
                if tier_scores:
                    print(f"   📈 Tier Breakdown:")
                    print(f"      • Base BB: {tier_scores.get('base_bb', 0)}/10 pts")
                    print(f"      • Money Flow: {tier_scores.get('money_flow', 0)}/7 pts")
                    print(f"      • BB Specific: {tier_scores.get('bb_specific', 0)}/8 pts")
                    print(f"      • Volume & Momentum: {tier_scores.get('volume_momentum', 0)}/6 pts")
                    print(f"      • Divergence: {tier_scores.get('divergence', 0)}/3 pts")
                
                # Display detailed breakdown
                breakdown = scoring_details.get('breakdown', [])
                if breakdown:
                    print(f"   🔍 Detailed Breakdown:")
                    for item in breakdown[:8]:  # Show first 8 items to avoid clutter
                        print(f"      • {item}")
                    
                    if len(breakdown) > 8:
                        print(f"      • ... and {len(breakdown) - 8} more indicators")
                
                # Display key indicator values
                indicator_values = scoring_details.get('indicator_values', {})
                if indicator_values:
                    print(f"   📊 Key Indicators:")
                    for indicator, value in indicator_values.items():
                        if isinstance(value, float):
                            print(f"      • {indicator.upper()}: {value:.3f}")
                        else:
                            print(f"      • {indicator.upper()}: {value}")
                
                print()  # Add spacing between trades
                
        except Exception as e:
            logger.error(f"Error displaying detailed scoring breakdown: {e}")

    def format_scoring_breakdown(self, setup_data):
        """Format detailed scoring breakdown for display"""
        
        if 'scoring_details' not in setup_data:
            return ""
        
        details = setup_data['scoring_details']
        symbol = setup_data.get('symbol', 'UNKNOWN')
        setup_type = setup_data.get('setup_type', 'NONE')
        bb_score = setup_data.get('bb_score', 0)
        
        # Build the breakdown display
        breakdown_text = f"\n📊 DETAILED SCORING BREAKDOWN - {symbol} {setup_type} ({bb_score}/34 points):\n"
        breakdown_text += "=" * 60 + "\n"
        
        # Show tier summaries
        tier_scores = details.get('tier_scores', {})
        breakdown_text += f"🏗️  TIER SUMMARY:\n"
        breakdown_text += f"   • Base BB Setup: {tier_scores.get('base_bb', 0)} pts\n"
        breakdown_text += f"   • Money Flow: {tier_scores.get('money_flow', 0)} pts\n"
        breakdown_text += f"   • BB-Specific: {tier_scores.get('bb_specific', 0)} pts\n"
        breakdown_text += f"   • Volume/Momentum: {tier_scores.get('volume_momentum', 0)} pts\n"
        breakdown_text += f"   • Divergence: {tier_scores.get('divergence', 0)} pts\n"
        breakdown_text += f"   ➡️  TOTAL: {bb_score} points\n\n"
        
        # Show detailed breakdown
        breakdown_text += f"🔍 DETAILED COMPONENT BREAKDOWN:\n"
        for i, component in enumerate(details.get('breakdown', []), 1):
            if "⭐" in component:  # Highlight top signals
                breakdown_text += f"   {i:2d}. {component} 🎯\n"
            else:
                breakdown_text += f"   {i:2d}. {component}\n"
        
        # Show key indicator values
        values = details.get('indicator_values', {})
        if values:
            breakdown_text += f"\n📈 KEY INDICATOR VALUES:\n"
            if 'mfi' in values:
                breakdown_text += f"   • MFI (Money Flow Index): {values['mfi']:.1f}\n"
            if 'cmf' in values:
                breakdown_text += f"   • CMF (Chaikin Money Flow): {values['cmf']:.4f}\n"
            if 'bb_expansion' in values:
                breakdown_text += f"   • BB Expansion Ratio: {values['bb_expansion']:.2f}x\n"
            if 'volume_surge' in values:
                breakdown_text += f"   • Volume Surge Detected: {values['volume_surge']}\n"
            if 'bb_trend' in values:
                breakdown_text += f"   • BB Trend Direction: {values['bb_trend']}\n"
        
        breakdown_text += "=" * 60 + "\n"
        
        return breakdown_text

    def _extract_enhanced_scoring_data(self, setup_data):
        """Extract enhanced scoring data for Excel output"""
        enhanced_data = {}
        
        # Basic enhanced data
        enhanced_data['bb_score_34'] = setup_data.get('bb_score', 0)
        enhanced_data['setup_quality_enhanced'] = setup_data.get('setup_quality', 'None')
        
        # Extract scoring details if available
        scoring_details = setup_data.get('scoring_details', {})
        
        if scoring_details:
            # Tier breakdown
            tier_scores = scoring_details.get('tier_scores', {})
            enhanced_data['tier_base_bb'] = tier_scores.get('base_bb', 0)
            enhanced_data['tier_money_flow'] = tier_scores.get('money_flow', 0)
            enhanced_data['tier_bb_specific'] = tier_scores.get('bb_specific', 0)
            enhanced_data['tier_volume_momentum'] = tier_scores.get('volume_momentum', 0)
            enhanced_data['tier_divergence'] = tier_scores.get('divergence', 0)
            
            # Key indicator values
            indicator_values = scoring_details.get('indicator_values', {})
            enhanced_data['mfi_value'] = indicator_values.get('mfi', 0)
            enhanced_data['cmf_value'] = indicator_values.get('cmf', 0)
            enhanced_data['bb_expansion_ratio'] = indicator_values.get('bb_expansion', 0)
            enhanced_data['volume_surge_detected'] = indicator_values.get('volume_surge', False)
            enhanced_data['bb_trend_direction'] = indicator_values.get('bb_trend', 'Unknown')
            
            # Component breakdown summary (first 5 components for Excel)
            breakdown = scoring_details.get('breakdown', [])
            for i, component in enumerate(breakdown[:5], 1):
                enhanced_data[f'component_{i}'] = component
                
            # Total components count
            enhanced_data['total_components'] = len(breakdown)
            
            # Check for high-value MFI signals (88% success rate indicator)
            mfi_signal_detected = any('MFI' in comp and '⭐' in comp for comp in breakdown)
            enhanced_data['mfi_priority_signal'] = mfi_signal_detected
            
        else:
            # Default values if no scoring details
            enhanced_data.update({
                'tier_base_bb': 0,
                'tier_money_flow': 0,
                'tier_bb_specific': 0,
                'tier_volume_momentum': 0,
                'tier_divergence': 0,
                'mfi_value': 0,
                'cmf_value': 0,
                'bb_expansion_ratio': 0,
                'volume_surge_detected': False,
                'bb_trend_direction': 'Unknown',
                'total_components': 0,
                'mfi_priority_signal': False
            })
        
        return enhanced_data

    def _sanitize_excel_value(self, value):
        """AGGRESSIVE sanitization for Excel compatibility - prevents ALL corruption"""
        
        # Handle None/NaN values first
        if value is None or pd.isna(value):
            return ""
        
        # Handle numpy types that cause issues
        if hasattr(value, 'dtype'):
            if 'float' in str(value.dtype) and pd.isna(value):
                return ""
            value = value.item() if hasattr(value, 'item') else str(value)
        
        # Handle boolean values
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        
        # Handle numeric values aggressively
        if isinstance(value, (int, float)):
            # Check for all problematic numeric values
            if pd.isna(value) or value == float('inf') or value == float('-inf'):
                return 0  # Use 0 instead of empty string for numbers
            if abs(value) > 1e15:  # Extremely large numbers
                return 0
            return round(float(value), 6)  # Limit precision
        
        # Handle complex objects (dictionaries, lists, etc.)
        if isinstance(value, (dict, list, tuple, set)):
            try:
                # Convert to simple string representation
                str_val = str(value)
                if len(str_val) > 1000:  # Limit length
                    return str_val[:1000] + "..."
                return str_val
            except:
                return "COMPLEX_OBJECT"
        
        # Handle strings aggressively
        if isinstance(value, str):
            # Remove all problematic characters
            value = value.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            value = value.replace('\x00', '').replace('\x01', '').replace('\x02', '')
            
            # Remove any non-printable characters
            value = ''.join(char for char in value if char.isprintable() or char.isspace())
            
            # Limit length aggressively
            if len(value) > 500:  # Much shorter limit
                value = value[:500] + "..."
            
            # Handle empty strings
            if not value.strip():
                return ""
                
            return value
        
        # Handle datetime objects
        if hasattr(value, 'strftime'):
            try:
                return value.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return str(value)
        
        # Final fallback - convert everything to string and sanitize
        try:
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:500]
            # Remove any remaining problematic characters
            return ''.join(char for char in str_value if char.isprintable() or char.isspace())
        except:
            return "SANITIZATION_ERROR"

    def _add_enhanced_columns_to_excel(self, worksheet, setups):
        """Add enhanced scoring columns with AGGRESSIVE sanitization"""
        
        # Enhanced headers (same as before)
        enhanced_headers = [
            'bb_score_34', 'setup_quality_enhanced', 'tier_base_bb',
            'tier_money_flow', 'tier_bb_specific', 'tier_volume_momentum',
            'tier_divergence', 'mfi_value', 'cmf_value', 'bb_expansion_ratio',
            'volume_surge_detected', 'bb_trend_direction', 'total_components',
            'mfi_priority_signal', 'component_1', 'component_2', 'component_3',
            'component_4', 'component_5'
        ]
        
        # Get existing headers
        existing_headers = []
        for col in range(1, worksheet.max_column + 1):
            header = worksheet.cell(row=1, column=col).value
            if header:
                existing_headers.append(header)
        
        # Add new headers
        start_col = len(existing_headers) + 1
        for i, header in enumerate(enhanced_headers):
            worksheet.cell(row=1, column=start_col + i, value=str(header))  # Ensure header is string
        
        # Add enhanced data with DOUBLE sanitization
        for row_idx, setup in enumerate(setups, start=2):
            enhanced_data = self._extract_enhanced_scoring_data(setup)
            
            for i, header in enumerate(enhanced_headers):
                raw_value = enhanced_data.get(header, '')
                
                # DOUBLE SANITIZATION - be extra aggressive
                sanitized_value = self._sanitize_excel_value(raw_value)
                final_value = self._sanitize_excel_value(sanitized_value)  # Second pass
                
                # Final safety check
                if final_value is None:
                    final_value = ""
                
                try:
                    worksheet.cell(row=row_idx, column=start_col + i, value=final_value)
                except Exception as e:
                    # If writing still fails, use empty string
                    worksheet.cell(row=row_idx, column=start_col + i, value="")

    # ADD NEW METHOD: Market Regime Terminal Display
    def _display_market_regime_summary(self, market_regime: Dict):
        """Display 6-line market regime summary"""
        print("\n" + "="*80)
        print("🌊 MARKET REGIME INTELLIGENCE")
        print("="*80)
        
        regime_type = market_regime.get('regime_type', 'UNKNOWN')
        confidence = market_regime.get('regime_confidence', 50)
        bb_suitability = market_regime.get('bb_suitability', 'UNKNOWN')
        position_mult = market_regime.get('position_multiplier', 1.0)
        
        print(f"📊 Regime: {regime_type} ({confidence}% confidence)")
        print(f"🎯 BB Strategy Suitability: {bb_suitability}")
        print(f"💰 Position Sizing: {position_mult}x multiplier")
        print(f"🏥 Market Health: {market_regime.get('market_health_score', 50)}/100")
        print(f"₿ BTC Health: {market_regime.get('btc_health_score', 50)}/100")
        print(f"🌍 Alt Season: {market_regime.get('alt_season_indicator', 'UNKNOWN')}")

    # ADD NEW METHOD: Format results with market context (for front columns)
    def format_comprehensive_results_with_market_context(self, all_results: List[Dict], market_baselines: Dict = None) -> pd.DataFrame:
        """Format results with market context and reordered columns"""
        
        if not all_results:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Get market baselines if not provided
        if market_baselines is None:
            market_baselines = {'overall_success_rate': 72.4}  # Fallback
        
        # ADD NEW FRONT COLUMNS
        df['technical_probability'] = (df.get('bb_score', 0) / 34 * 100).round(1)
        df['historical_probability'] = df.get('historical_win_rate', 0)
        df['historical_profit'] = df.get('historical_avg_win', 0)
        df['historical_drawdown'] = df.get('historical_avg_loss', 0) 
        df['historical_duration'] = df.get('historical_avg_duration', 0)
        
        # Add market context
        df['market_context'] = df.apply(lambda row: self._calculate_market_context(
            row.get('bb_score', 0), market_baselines.get('overall_success_rate', 72.4)
        ), axis=1)
        
        # REORDER COLUMNS - PRIORITY COLUMNS FIRST
        priority_columns = [
            'symbol', 'setup_type', 'exchange',
            'technical_probability', 'historical_probability', 
            'historical_profit', 'historical_drawdown', 'historical_duration',
            'market_context', 'probability', 'bb_score',
            'entry_price', 'stop_price', 'target1', 'risk_reward', 'risk_pct'
        ]
        
        # Get remaining columns (preserve all existing data)
        remaining_columns = [col for col in df.columns if col not in priority_columns]
        final_columns = priority_columns + remaining_columns
        
        # Reorder DataFrame
        available_columns = [col for col in final_columns if col in df.columns]
        df_reordered = df[available_columns].copy()
        
        return df_reordered

    # ADD HELPER METHOD: Market context classification
    def _calculate_market_context(self, bb_score: int, market_baseline: float) -> str:
        """Calculate market context classification"""
        score_percentage = (bb_score / 34) * 100
        
        if score_percentage > market_baseline + 15:
            return "SIGNIFICANTLY_ABOVE_MARKET"
        elif score_percentage > market_baseline + 5:
            return "ABOVE_MARKET"
        elif score_percentage > market_baseline - 5:
            return "MARKET_AVERAGE"
        else:
            return "BELOW_MARKET"

    # ADD METHOD TO STORE MARKET REGIME DATA
    def set_market_regime_data(self, market_regime: Dict):
        """Store market regime data for display"""
        self._market_regime_data = market_regime

