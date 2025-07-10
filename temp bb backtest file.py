def _create_market_overview_sheet(self, writer, market_data: Dict = None):
    """Create Market Overview tab with daily backtesting snapshot for ML training"""
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        overview_data = []
        overview_data.append(['DAILY MARKET ANALYSIS SNAPSHOT', '', '', '', '', '', '', ''])
        overview_data.append(['Analysis Date', current_date, '', '', '', '', '', ''])
        overview_data.append(['Analysis Period', market_data.get('analysis_period', 'Rolling 30-Day Window'), '', '', '', '', '', ''])
        overview_data.append(['', '', '', '', '', '', '', ''])
        overview_data.append(['OVERALL BB PERFORMANCE', '', '', '', '', '', '', ''])
        overview_data.append(['Total BB Bounces Analyzed', market_data.get('total_bounces', ''), '', '', '', '', '', ''])
        overview_data.append(['Coins Successfully Analyzed', market_data.get('coins_analyzed', ''), '', '', '', '', '', ''])
        overview_data.append(['Overall Success Rate', f"{market_data.get('overall_success_rate', '')}%", '', '', '', '', '', ''])
        overview_data.append(['Market Health Score', f"{market_data.get('market_health', '')}%", '', '', '', '', '', ''])
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # BB-SPECIFIC INDICATORS
        overview_data.append(['ENHANCED BB METRICS ANALYSIS', 'Indicator', 'Success Rate', 'Avg P&L', 'Avg Loss', 'Profit Factor', 'Samples'])
        for row in market_data.get('bb_specific_indicators', []):
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # BB TREND ANALYSIS (add the missing trend breakdown)
        overview_data.append(['', '', '', '', '', '', '', ''])
        overview_data.append(['BB Trend Analysis', 'Trend', 'Success Rate', 'Avg Win', 'Avg Loss', 'Profit Factor', 'Samples'])
        bb_trend_data = market_data.get('bb_trend_analysis', [])
        for row in bb_trend_data:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        # TECHNICAL INDICATORS  
        overview_data.append(['ADDITIONAL TECHNICAL METRICS ANALYSIS', 'Indicator', 'Success Rate', 'Avg Win', 'Avg Loss', 'Profit Factor', 'Samples'])
        for row in market_data.get('technical_indicators', []):
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 🛡️ OPTIMAL STOP LOSS ANALYSIS
        overview_data.append(['OPTIMAL STOP LOSS ANALYSIS', 'SL Level', 'Win Rate', 'Avg Win', 'R/R', 'Avg DD', 'Max DD Time', 'Avg Duration', 'Samples'])
        stop_loss_data = market_data.get('optimal_stop_loss', [])
        for row in stop_loss_data:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 📉 DRAWDOWN DISTRIBUTION
        overview_data.append(['DRAWDOWN DISTRIBUTION ANALYSIS', 'Coverage', 'Drawdown Limit', 'Avg Time to Max DD', '', '', '', ''])
        drawdown_data = market_data.get('drawdown_distribution', [])
        for row in drawdown_data:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 💡 OPTIMAL SL RECOMMENDATIONS
        overview_data.append(['OPTIMAL SL RECOMMENDATIONS', 'Strategy', 'Protection Level', 'Recommended SL', '', '', '', ''])
        sl_recommendations = market_data.get('sl_recommendations', [])
        for row in sl_recommendations:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 📊 P&L CHARACTERISTICS
        overview_data.append(['OVERALL P&L CHARACTERISTICS', 'Metric', 'Value', '', '', '', '', ''])
        pnl_data = market_data.get('pnl_characteristics', [])
        for row in pnl_data:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 📈 WINNING TRADE DISTRIBUTION
        overview_data.append(['WINNING TRADE DISTRIBUTION', 'Percentile', 'Value', '', '', '', '', ''])
        winning_dist = market_data.get('winning_distribution', [])
        for row in winning_dist:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 📉 LOSING TRADE DISTRIBUTION
        overview_data.append(['LOSING TRADE DISTRIBUTION', 'Percentile', 'Value', '', '', '', '', ''])
        losing_dist = market_data.get('losing_distribution', [])
        for row in losing_dist:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 🔍 CONFLUENCE FACTOR EFFECTIVENESS
        overview_data.append(['CONFLUENCE FACTOR EFFECTIVENESS', 'Factor', 'Success Rate', 'Avg Win', 'Avg Loss', 'Profit Factor', 'Improvement', 'Samples'])
        confluence_data = market_data.get('confluence_analysis', [])
        for row in confluence_data:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 🕐 COMPREHENSIVE TIMING ANALYSIS
        overview_data.append(['COMPREHENSIVE TIMING ANALYSIS', 'Target', 'Average', 'Median', 'Hit Rate', 'Trades', '', ''])
        timing_data = market_data.get('comprehensive_timing', [])
        for row in timing_data:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 🎯 TAKE PROFIT TARGET ANALYSIS
        overview_data.append(['TAKE PROFIT TARGET ANALYSIS', 'Target', 'Hit Rate', 'Hits/Total Trades', '', '', '', ''])
        tp_targets = market_data.get('take_profit_targets', [])
        for row in tp_targets:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 💰 OPTIMAL TAKE PROFIT RECOMMENDATIONS
        overview_data.append(['OPTIMAL TAKE PROFIT RECOMMENDATIONS', 'Current Strategy', 'Metric', 'Value', '', '', '', ''])
        tp_recommendations = market_data.get('tp_recommendations', [])
        for row in tp_recommendations:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])

        # 💡 OPTIMAL STRATEGY ANALYSIS
        overview_data.append(['OPTIMAL STRATEGY ANALYSIS', 'Strategy Type', 'Metric', 'Value', '', '', '', ''])
        optimal_strategy = market_data.get('optimal_strategy_analysis', [])
        for row in optimal_strategy:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 💰 PEAK GAIN DISTRIBUTION
        overview_data.append(['PEAK GAIN DISTRIBUTION', 'Percentile', 'Value', '', '', '', '', ''])
        peak_dist = market_data.get('peak_distribution', [])
        for row in peak_dist:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # ⏱️ TIMING COMPARISON
        overview_data.append(['TIMING COMPARISON', 'Timing Metric', 'Value', 'Units', '', '', '', ''])
        timing_comparison = market_data.get('timing_comparison', [])
        for row in timing_comparison:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # 💡 RECOMMENDED TAKE PROFIT STRATEGY
        overview_data.append(['RECOMMENDED TAKE PROFIT STRATEGY', 'Recommendation', 'Details', '', '', '', '', ''])
        overview_data.append(['✅ CURRENT BB STRATEGY IS SUBOPTIMAL!', '', '', '', '', '', '', ''])
        strategy_recommendations = market_data.get('strategy_recommendations', [])
        for row in strategy_recommendations:
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # MARKET CAP TIERS
        overview_data.append(['MARKET CAP TIER ANALYSIS', 'Tier', 'Success Rate', 'Samples', '', '', '', ''])
        for row in market_data.get('market_cap_tiers', []):
            overview_data.append(row)
        overview_data.append(['', '', '', '', '', '', '', ''])
        
        # ML TRAINING DATA
        overview_data.append(['ML TRAINING DATA', '', '', '', '', '', '', ''])
        for row in market_data.get('ml_training_data', []):
            overview_data.append(row)
        
        overview_data.append(['Next Update', market_data.get('next_update', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')), '', '', '', '', '', ''])
        
        df_overview = pd.DataFrame(overview_data, columns=['Metric', 'Value', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8'])
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