import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List

class VolumeProfileTradeReporter:
    """
    Enhanced trade reporter for Volume Profile backtesting
    Generates detailed Excel reports with multiple analysis sheets
    """
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def generate_detailed_report(self, all_results: List[Dict], strategy_summary: Dict, 
                               category_summary: Dict, timestamp: str = None) -> None:
        """
        Generate comprehensive detailed report with multiple analysis sheets
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Collect all trades data
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
                    'type': trade['type'],  # FIXED: Use 'type' instead of 'direction'
                    
                    # Entry Details
                    'entry_time': trade.get('entry_time', ''),
                    'entry_price': trade['entry_price'],
                    'stop_loss': trade['stop_loss'],
                    'target': trade['target'],
                    'risk_reward': trade.get('risk_reward', 0),
                    'entry_score': trade.get('score', 0),
                    
                    # Exit Details
                    'exit_price': trade.get('exit_price', 0),
                    'exit_time': self._calculate_exit_time(trade),
                    'result': trade.get('result', 'unknown'),
                    'pnl_pct': trade.get('pnl_pct', 0),
                    'duration_hours': trade.get('duration_candles', 0) * 4,  # 4H candles
                    
                    # Win/Loss Status - ADD THIS
                    'win': trade.get('win', False),
                    
                    # Volume Profile Specifics
                    'poc_level': trade.get('poc_level', 0),
                    'vah_level': trade.get('vah_level', 0),
                    'val_level': trade.get('val_level', 0),
                    'virgin_poc_level': trade.get('virgin_poc_level', 0),
                    'poc_cluster_count': trade.get('poc_cluster', 0),
                    
                    # Risk Metrics
                    'risk_pct': abs(trade['entry_price'] - trade['stop_loss']) / trade['entry_price'] * 100,
                    'reward_pct': abs(trade['target'] - trade['entry_price']) / trade['entry_price'] * 100,
                    'actual_rr': abs(trade.get('pnl_pct', 0)) / (abs(trade['entry_price'] - trade['stop_loss']) / trade['entry_price'] * 100) if trade.get('win') else 0
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
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['win'] == True]
        losing_trades = trades_df[trades_df['win'] == False]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_pnl = trades_df['pnl_pct'].mean()
        total_pnl = trades_df['pnl_pct'].sum()
        
        # Risk metrics
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        best_trade = trades_df['pnl_pct'].max()
        worst_trade = trades_df['pnl_pct'].min()
        
        # Calculate profit factor
        total_wins = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Calculate max drawdown
        cumulative_pnl = trades_df['pnl_pct'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns_std = trades_df['pnl_pct'].std()
        sharpe_ratio = avg_pnl / returns_std if returns_std > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }
    
    def _analyze_category_performance(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by crypto category"""
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        category_stats = []
        for category in trades_df['category'].unique():
            cat_trades = trades_df[trades_df['category'] == category]
            
            if len(cat_trades) == 0:
                continue
                
            wins = cat_trades[cat_trades['win'] == True]
            losses = cat_trades[cat_trades['win'] == False]
            
            stats = {
                'category': category,
                'total_trades': len(cat_trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(cat_trades),
                'avg_pnl': cat_trades['pnl_pct'].mean(),
                'total_pnl': cat_trades['pnl_pct'].sum(),
                'avg_win': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
                'avg_loss': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
                'best_trade': cat_trades['pnl_pct'].max(),
                'worst_trade': cat_trades['pnl_pct'].min()
            }
            category_stats.append(stats)
        
        return pd.DataFrame(category_stats)
    
    def _analyze_trade_timing(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze trade timing patterns"""
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        # Convert entry_time to datetime for analysis
        trades_df['entry_datetime'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['hour'] = trades_df['entry_datetime'].dt.hour
        trades_df['day_of_week'] = trades_df['entry_datetime'].dt.day_name()
        
        # Hourly analysis
        hourly_stats = []
        for hour in range(24):
            hour_trades = trades_df[trades_df['hour'] == hour]
            if len(hour_trades) > 0:
                wins = hour_trades[hour_trades['result'] == 'target']
                stats = {
                    'hour': hour,
                    'total_trades': len(hour_trades),
                    'wins': len(wins),
                    'win_rate': len(wins) / len(hour_trades),
                    'avg_pnl': hour_trades['pnl_pct'].mean()
                }
                hourly_stats.append(stats)
        
        return pd.DataFrame(hourly_stats)
    
    def _create_strategy_dataframe(self, strategy_summary: Dict) -> pd.DataFrame:
        """Create strategy performance DataFrame"""
        strategy_data = []
        for strategy, stats in strategy_summary.items():
            if isinstance(stats, dict):
                # Handle different possible data structures
                strategy_data.append({
                    'strategy': strategy,
                    'total_trades': stats.get('total_trades', 0),
                    'wins': stats.get('wins', 0),
                    'losses': stats.get('losses', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'avg_pnl': stats.get('avg_pnl', 0),
                    'avg_win': stats.get('avg_win', 0),
                    'avg_loss': stats.get('avg_loss', 0),
                    'profit_factor': stats.get('profit_factor', 0),
                    'max_drawdown': stats.get('max_drawdown', 0),
                    'sharpe_ratio': stats.get('sharpe_ratio', 0)
                })
        
        return pd.DataFrame(strategy_data)
    
    def _create_top_trades_sheet(self, trades_df: pd.DataFrame, writer) -> None:
        """Create sheet with top winners and losers"""
        if len(trades_df) == 0:
            return
        
        # Top 10 winners
        top_winners = trades_df[trades_df['result'] == 'target'].nlargest(10, 'pnl_pct')
        
        # Top 10 losers
        top_losers = trades_df[trades_df['result'] == 'stop_loss'].nsmallest(10, 'pnl_pct')
        
        # Combine and save
        top_trades = pd.concat([top_winners, top_losers])
        top_trades.to_excel(writer, sheet_name='Top_Trades', index=False)
    
    def _print_console_summary(self, trades_df: pd.DataFrame, summary_stats: Dict, 
                             strategy_summary: Dict) -> None:
        """Print comprehensive console summary"""
        
        print("\n" + "="*80)
        print("📊 VOLUME PROFILE BACKTEST - DETAILED TRADE ANALYSIS")
        print("="*80)
        
        if len(trades_df) == 0:
            print("❌ No trades found in backtest")
            return
        
        # Overall Performance
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Total Trades: {summary_stats['total_trades']}")
        print(f"   Win Rate: {summary_stats['win_rate']:.1%}")
        print(f"   Total PnL: {summary_stats['total_pnl']:.2f}%")
        print(f"   Average PnL: {summary_stats['avg_pnl']:.2f}%")
        print(f"   Best Trade: {summary_stats['best_trade']:.2f}%")
        print(f"   Worst Trade: {summary_stats['worst_trade']:.2f}%")
        print(f"   Profit Factor: {summary_stats['profit_factor']:.2f}")
        print(f"   Max Drawdown: {summary_stats['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {summary_stats['sharpe_ratio']:.2f}")
        
        # Strategy Performance
        print(f"\n📈 STRATEGY PERFORMANCE:")
        for strategy, stats in strategy_summary.items():
            if isinstance(stats, dict):
                total_trades = stats.get('total_trades', 0)
                win_rate = stats.get('win_rate', 0)  # Use pre-calculated win rate
                if total_trades > 0:
                    print(f"   {strategy}: {total_trades} trades, {win_rate:.1%} win rate")
        
        # Category Performance
        print(f"\n🏷️  CATEGORY PERFORMANCE:")
        category_perf = self._analyze_category_performance(trades_df)
        for _, row in category_perf.iterrows():
            print(f"   {row['category']}: {row['total_trades']} trades, {row['win_rate']:.1%} win rate, {row['total_pnl']:.2f}% PnL")
        
        # Top Trades
        if len(trades_df) > 0:
            print(f"\n🏆 TOP PERFORMERS:")
            top_winners = trades_df[trades_df['win'] == True].nlargest(3, 'pnl_pct')
            for _, trade in top_winners.iterrows():
                print(f"   {trade['symbol']} ({trade['strategy']}): {trade['pnl_pct']:.2f}%")
            
            print(f"\n💥 BIGGEST LOSERS:")
            top_losers = trades_df[trades_df['win'] == False].nsmallest(3, 'pnl_pct')
            for _, trade in top_losers.iterrows():
                print(f"   {trade['symbol']} ({trade['strategy']}): {trade['pnl_pct']:.2f}%")
    
    def _save_json_report(self, all_results: List[Dict], summary_stats: Dict, json_path: str) -> None:
        """Save comprehensive JSON report"""
        report = {
            'summary': summary_stats,
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

def integrate_detailed_reporting(backtest_instance, results):
    """Integrate detailed reporting into backtest instance"""
    reporter = VolumeProfileTradeReporter(backtest_instance.output_path)
    
    # FIX: Calculate strategy summary from actual trade data
    strategy_summary = {}
    category_summary = {}
    
    # Collect all trades
    all_trades = []
    for result in results:
        all_trades.extend(result.get('all_trades', []))
    
    # Calculate strategy stats from actual trade data
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        
        for strategy in trades_df['strategy'].unique():
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            
            if not strategy_trades.empty:
                total_trades = len(strategy_trades)
                # Use the new 'win' key instead of 'result'
                winning_trades = strategy_trades[strategy_trades['win'] == True]
                losing_trades = strategy_trades[strategy_trades['win'] == False]
                
                wins = len(winning_trades)
                losses = len(losing_trades)
                win_rate = wins / total_trades if total_trades > 0 else 0
                
                avg_pnl = strategy_trades['pnl_pct'].mean() if 'pnl_pct' in strategy_trades.columns else 0
                avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 and 'pnl_pct' in winning_trades.columns else 0
                avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 and 'pnl_pct' in losing_trades.columns else 0
                
                # Calculate profit factor
                total_wins = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 and 'pnl_pct' in winning_trades.columns else 0
                total_losses = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 and 'pnl_pct' in losing_trades.columns else 0
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                
                strategy_summary[strategy] = {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'max_drawdown': 0,  # Would need to calculate from equity curve
                    'sharpe_ratio': 0   # Would need to calculate from returns
                }
    
    # Generate detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporter.generate_detailed_report(results, strategy_summary, category_summary, timestamp)
    
    print("\n✅ Enhanced results saved with detailed trade analysis!")
    print("Check /Users/robertsmith/Documents/BB Screener/backtest_modules/4_hour_backtest_modules/backtest_results/volume_profile for:")
    print("  - Detailed Excel with all trades")
    print("  - JSON with programmatic access")
    print("  - Summary reports") 