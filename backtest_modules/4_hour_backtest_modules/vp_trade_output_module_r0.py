"""
Volume Profile Trade Output Module
Provides detailed trade reporting for Volume Profile backtest results
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict, List
import os

class VolumeProfileTradeReporter:
    """Generates detailed trade reports from Volume Profile backtest results"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
    def generate_detailed_report(self, all_results: List[Dict], strategy_summary: Dict, 
                                category_summary: Dict, timestamp: str = None) -> None:
        """Generate comprehensive trade report with all details"""
        
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Create detailed trades DataFrame
        all_trades_data = []
        for result in all_results:
            symbol = result['symbol']
            category = result['category']
            
            for trade in result.get('all_trades', []):
                trade_data = {
                    # Basic Info
                    'symbol': symbol,
                    'category': category,
                    'strategy': trade['strategy'],
                    'direction': trade['direction'],
                    
                    # Entry Details
                    'entry_time': trade.get('entry_time', ''),
                    'entry_price': trade['entry'],
                    'stop_loss': trade['stop'],
                    'target': trade['target'],
                    'risk_reward': trade.get('risk_reward', 0),
                    'entry_score': trade.get('score', 0),
                    
                    # Exit Details
                    'exit_price': trade.get('exit_price', 0),
                    'exit_time': self._calculate_exit_time(trade),
                    'result': trade.get('result', 'unknown'),
                    'pnl_pct': trade.get('pnl_pct', 0),
                    'duration_hours': trade.get('duration_candles', 0) * 4,  # 4H candles
                    
                    # Volume Profile Specifics
                    'poc_level': trade.get('poc_level', 0),
                    'vah_level': trade.get('vah_level', 0),
                    'val_level': trade.get('val_level', 0),
                    'virgin_poc_level': trade.get('virgin_poc_level', 0),
                    'poc_cluster_count': trade.get('poc_cluster', 0),
                    
                    # Risk Metrics
                    'risk_pct': abs(trade['entry'] - trade['stop']) / trade['entry'] * 100,
                    'reward_pct': abs(trade['target'] - trade['entry']) / trade['entry'] * 100,
                    'actual_rr': abs(trade.get('pnl_pct', 0)) / (abs(trade['entry'] - trade['stop']) / trade['entry'] * 100) if trade.get('result') == 'target' else 0
                }
                all_trades_data.append(trade_data)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(all_trades_data)
        
        # 2. Generate Summary Statistics
        summary_stats = self._generate_summary_statistics(trades_df, strategy_summary)
        
        # 3. Create Performance by Category
        category_performance = self._analyze_category_performance(trades_df)
        
        # 4. Create Time Analysis
        time_analysis = self._analyze_trade_timing(trades_df)
        
        # 5. Save to Excel with multiple sheets
        excel_path = os.path.join(self.output_path, f"vp_detailed_trades_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: All Trades
            trades_df.to_excel(writer, sheet_name='All_Trades', index=False)
            
            # Sheet 2: Summary Statistics
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 3: Strategy Performance
            strategy_df = self._create_strategy_dataframe(strategy_summary)
            strategy_df.to_excel(writer, sheet_name='Strategy_Performance', index=False)
            
            # Sheet 4: Category Analysis
            category_performance.to_excel(writer, sheet_name='Category_Analysis', index=False)
            
            # Sheet 5: Timing Analysis
            time_analysis.to_excel(writer, sheet_name='Timing_Analysis', index=False)
            
            # Sheet 6: Top Winners and Losers
            self._create_top_trades_sheet(trades_df, writer)
        
        # 6. Generate Console Output
        self._print_console_summary(trades_df, summary_stats, strategy_summary)
        
        # 7. Save JSON for programmatic access
        json_path = os.path.join(self.output_path, f"vp_trades_{timestamp}.json")
        self._save_json_report(all_results, summary_stats, json_path)
        
        print(f"\n📊 Detailed reports saved to:")
        print(f"   Excel: {excel_path}")
        print(f"   JSON: {json_path}")
    
    def _calculate_exit_time(self, trade: Dict) -> str:
        """Calculate exit time based on entry time and duration"""
        if 'entry_time' in trade and 'duration_candles' in trade:
            try:
                entry = pd.to_datetime(trade['entry_time'])
                exit_time = entry + pd.Timedelta(hours=trade['duration_candles'] * 4)
                return exit_time.strftime('%Y-%m-%d %H:%M')
            except:
                return ''
        return ''
    
    def _generate_summary_statistics(self, trades_df: pd.DataFrame, strategy_summary: Dict) -> Dict:
        """Generate comprehensive summary statistics"""
        
        if len(trades_df) == 0:
            return {'total_trades': 0, 'message': 'No trades found'}
        
        winning_trades = trades_df[trades_df['result'] == 'target']
        losing_trades = trades_df[trades_df['result'] == 'stop_loss']
        
        return {
            # Overall Performance
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'timeout_trades': len(trades_df[trades_df['result'] == 'timeout']),
            'overall_win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            
            # P&L Statistics
            'total_pnl_pct': trades_df['pnl_pct'].sum(),
            'avg_pnl_pct': trades_df['pnl_pct'].mean(),
            'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'best_trade_pnl': trades_df['pnl_pct'].max(),
            'worst_trade_pnl': trades_df['pnl_pct'].min(),
            
            # Risk/Reward
            'avg_risk_reward': trades_df['risk_reward'].mean(),
            'avg_actual_rr': trades_df['actual_rr'].mean(),
            'profit_factor': abs(winning_trades['pnl_pct'].sum() / losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_pct'].sum() != 0 else 0,
            
            # Duration
            'avg_duration_hours': trades_df['duration_hours'].mean(),
            'avg_win_duration': winning_trades['duration_hours'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_duration': losing_trades['duration_hours'].mean() if len(losing_trades) > 0 else 0,
            
            # Score Analysis
            'avg_entry_score': trades_df['entry_score'].mean(),
            'winning_avg_score': winning_trades['entry_score'].mean() if len(winning_trades) > 0 else 0,
            'losing_avg_score': losing_trades['entry_score'].mean() if len(losing_trades) > 0 else 0,
        }
    
    def _analyze_category_performance(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by crypto category"""
        
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        category_stats = []
        for category in trades_df['category'].unique():
            cat_trades = trades_df[trades_df['category'] == category]
            wins = cat_trades[cat_trades['result'] == 'target']
            
            category_stats.append({
                'category': category,
                'total_trades': len(cat_trades),
                'wins': len(wins),
                'win_rate': len(wins) / len(cat_trades) * 100 if len(cat_trades) > 0 else 0,
                'avg_pnl': cat_trades['pnl_pct'].mean(),
                'total_pnl': cat_trades['pnl_pct'].sum(),
                'avg_duration': cat_trades['duration_hours'].mean(),
                'avg_score': cat_trades['entry_score'].mean()
            })
        
        return pd.DataFrame(category_stats)
    
    def _analyze_trade_timing(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze trade timing patterns"""
        
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        timing_stats = []
        
        # Group by duration buckets
        duration_buckets = [(0, 24), (24, 48), (48, 96), (96, 168), (168, float('inf'))]
        bucket_names = ['0-24h', '24-48h', '48-96h', '96-168h', '168h+']
        
        for i, (min_h, max_h) in enumerate(duration_buckets):
            bucket_trades = trades_df[(trades_df['duration_hours'] >= min_h) & 
                                     (trades_df['duration_hours'] < max_h)]
            wins = bucket_trades[bucket_trades['result'] == 'target']
            
            if len(bucket_trades) > 0:
                timing_stats.append({
                    'duration_bucket': bucket_names[i],
                    'trade_count': len(bucket_trades),
                    'win_rate': len(wins) / len(bucket_trades) * 100,
                    'avg_pnl': bucket_trades['pnl_pct'].mean(),
                    'avg_score': bucket_trades['entry_score'].mean()
                })
        
        return pd.DataFrame(timing_stats)
    
    def _create_strategy_dataframe(self, strategy_summary: Dict) -> pd.DataFrame:
        """Create DataFrame from strategy summary"""
        
        strategy_data = []
        for strategy, stats in strategy_summary.items():
            strategy_data.append({
                'strategy': strategy,
                'total_trades': stats.get('total_trades', 0),
                'wins': stats.get('total_wins', 0),
                'win_rate': stats.get('overall_win_rate', 0) * 100,
                'avg_pnl': stats.get('overall_avg_pnl', 0),
                **{f"{cat}_trades": cat_stats['trades'] 
                   for cat, cat_stats in stats.get('by_category', {}).items()},
                **{f"{cat}_win_rate": cat_stats['win_rate'] * 100 
                   for cat, cat_stats in stats.get('by_category', {}).items()}
            })
        
        return pd.DataFrame(strategy_data)
    
    def _create_top_trades_sheet(self, trades_df: pd.DataFrame, writer) -> None:
        """Create sheet with top winners and losers"""
        
        if len(trades_df) == 0:
            return
        
        # Top 10 Winners
        top_winners = trades_df.nlargest(10, 'pnl_pct')[
            ['symbol', 'strategy', 'entry_time', 'entry_price', 'exit_price', 
             'pnl_pct', 'duration_hours', 'entry_score']
        ]
        
        # Top 10 Losers
        top_losers = trades_df.nsmallest(10, 'pnl_pct')[
            ['symbol', 'strategy', 'entry_time', 'entry_price', 'exit_price', 
             'pnl_pct', 'duration_hours', 'entry_score']
        ]
        
        # Write to Excel
        top_winners.to_excel(writer, sheet_name='Top_Trades', index=False, startrow=1)
        top_losers.to_excel(writer, sheet_name='Top_Trades', index=False, 
                           startrow=len(top_winners) + 4)
        
        # Add headers
        worksheet = writer.sheets['Top_Trades']
        worksheet.cell(row=1, column=1, value='TOP 10 WINNERS')
        worksheet.cell(row=len(top_winners) + 4, column=1, value='TOP 10 LOSERS')
    
    def _print_console_summary(self, trades_df: pd.DataFrame, summary_stats: Dict, 
                              strategy_summary: Dict) -> None:
        """Print detailed console summary"""
        
        print("\n" + "="*80)
        print("📊 VOLUME PROFILE BACKTEST - DETAILED TRADE ANALYSIS")
        print("="*80)
        
        if len(trades_df) == 0:
            print("❌ No trades found in backtest")
            return
        
        # Overall Performance
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Total Trades: {summary_stats['total_trades']}")
        print(f"   Win Rate: {summary_stats['overall_win_rate']:.1f}%")
        print(f"   Average P&L: {summary_stats['avg_pnl_pct']:.2f}%")
        print(f"   Profit Factor: {summary_stats['profit_factor']:.2f}")
        print(f"   Average R/R: {summary_stats['avg_risk_reward']:.2f}")
        
        # Strategy Breakdown
        print(f"\n🎯 STRATEGY PERFORMANCE:")
        for strategy, stats in strategy_summary.items():
            if stats.get('total_trades', 0) > 0:
                print(f"   {strategy.upper()}:")
                print(f"      Trades: {stats.get('total_trades', 0)}")
                print(f"      Win Rate: {stats.get('overall_win_rate', 0)*100:.1f}%")
                print(f"      Avg P&L: {stats.get('overall_avg_pnl', 0):.2f}%")
        
        # Top 5 Trades
        print(f"\n💰 TOP 5 WINNERS:")
        top_5 = trades_df.nlargest(5, 'pnl_pct')
        for idx, trade in top_5.iterrows():
            print(f"   {trade['symbol']} ({trade['strategy']}): +{trade['pnl_pct']:.2f}% in {trade['duration_hours']:.0f}h")
        
        print(f"\n📉 TOP 5 LOSERS:")
        bottom_5 = trades_df.nsmallest(5, 'pnl_pct')
        for idx, trade in bottom_5.iterrows():
            print(f"   {trade['symbol']} ({trade['strategy']}): {trade['pnl_pct']:.2f}% in {trade['duration_hours']:.0f}h")
        
        # Entry Score Analysis
        print(f"\n📊 ENTRY SCORE ANALYSIS:")
        print(f"   Average Score (All): {summary_stats['avg_entry_score']:.1f}")
        print(f"   Average Score (Winners): {summary_stats['winning_avg_score']:.1f}")
        print(f"   Average Score (Losers): {summary_stats['losing_avg_score']:.1f}")
        
        # Duration Analysis
        print(f"\n⏱️ DURATION ANALYSIS:")
        print(f"   Average Duration: {summary_stats['avg_duration_hours']:.1f} hours")
        print(f"   Winners Avg: {summary_stats['avg_win_duration']:.1f} hours")
        print(f"   Losers Avg: {summary_stats['avg_loss_duration']:.1f} hours")
    
    def _save_json_report(self, all_results: List[Dict], summary_stats: Dict, 
                         json_path: str) -> None:
        """Save comprehensive JSON report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary_stats,
            'detailed_results': all_results
        }
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


# Integration function to add to volume_profile_backtest_4h_r1.py
def integrate_detailed_reporting(backtest_instance, results):
    """
    Add this function call at the end of run_backtest() in volume_profile_backtest_4h_r1.py
    
    Example:
    results = backtester.run_backtest(symbols)
    integrate_detailed_reporting(backtester, results)
    """
    
    # Create reporter instance
    reporter = VolumeProfileTradeReporter(backtest_instance.output_path)
    
    # Generate detailed reports
    if 'detailed_results' in results:
        reporter.generate_detailed_report(
            all_results=results['detailed_results'],
            strategy_summary=results.get('strategy_performance', {}),
            category_summary=results.get('category_analysis', {})
        )
    
    print("\n✅ Detailed trade analysis complete!")