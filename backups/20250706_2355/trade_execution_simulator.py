"""
Trade Simulation Engine - Using WORKING patterns from your files
================================================================
Copies the exact working patterns from improved_bb_backtest.py and real_market_test.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your working modules - FIXED for modules folder structure
try:
    from modules.data_fetcher import MarketDataFetcher
    from modules.technical_analyzer import TechnicalAnalyzer
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing core modules: {e}")
    sys.exit(1)

class TradeSimulationEngine:
    def __init__(self):
        """Initialize using EXACT patterns from your working files"""
        self.data_fetcher = MarketDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Test symbols - same as your working files
        self.test_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
        self.timeframe = '1h'
        self.days_back = 30
        
        self.results = {}
        self.all_trades = []

    def run_parameter_optimization(self):
        """Run complete parameter optimization backtest"""
        print("🚀 TRADE SIMULATION ENGINE - PARAMETER OPTIMIZATION")
        print("=" * 70)
        print("🎯 Objective: Find optimal entry, stop-loss, and take-profit parameters")
        print("📊 Method: Realistic trade execution simulation with fees and slippage")
        print("⚡ Focus: Parameter optimization for maximum risk-adjusted returns")
        print("=" * 70)
        
        # Parameters to test
        stop_loss_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]  # ATR multipliers
        take_profit_strategies = ['bb_middle', 'bb_opposite', 'risk_reward_2_1']
        entry_conditions = [0.0, 0.05, 0.1]  # % inside BB bands
        
        print(f"📊 Testing {len(stop_loss_multipliers)} SL levels × {len(take_profit_strategies)} TP strategies")
        print(f"💰 Execution costs: 0.15% slippage + 0.15% fees")
        
        all_results = []
        
        # Test all parameter combinations
        for sl_mult in stop_loss_multipliers:
            for tp_strategy in take_profit_strategies:
                for entry_cond in entry_conditions:
                    
                    config = {
                        'stop_loss_atr_mult': sl_mult,
                        'take_profit_strategy': tp_strategy,
                        'entry_condition': entry_cond,
                        'slippage': 0.0015,
                        'fees': 0.0015
                    }
                    
                    result = self._test_parameter_set(config)
                    if result:
                        all_results.append(result)
        
        # Analyze and display results
        self._analyze_optimization_results(all_results)
        return all_results

    def _test_parameter_set(self, config):
        """Test a specific parameter configuration"""
        trades = []
        
        print(f"\n🎯 STARTING PARAMETER OPTIMIZATION BACKTEST")
        print(f"📊 Symbols: {self.test_symbols}")
        print(f"📅 Timeframe: {self.days_back}d")
        print("=" * 70)
        
        for symbol in self.test_symbols:
            print(f"🔍 ANALYZING {symbol}...")
            
            try:
                # Get historical data - EXACT same pattern as your working files
                df = self.data_fetcher.fetch_ohlcv('binance', symbol, self.timeframe)
                
                if df is None or len(df) < 100:
                    print(f"❌ Insufficient data for {symbol}")
                    continue
                
                # Add technical indicators - EXACT same pattern
                df = self._add_essential_indicators(df)
                
                # Find BB setups and simulate trades
                symbol_trades = self._find_and_simulate_trades(df, symbol, config)
                trades.extend(symbol_trades)
                
                print(f"✅ Found {len(symbol_trades)} trades for {symbol}")
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        if not trades:
            return None
        
        # Calculate performance metrics
        return self._calculate_performance(trades, config)

    def _add_essential_indicators(self, df):
        """Add essential indicators using working patterns from your files"""
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # BB percentage position
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR for stop losses
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def _find_and_simulate_trades(self, df, symbol, config):
        """Find BB setups and simulate trade execution"""
        trades = []
        
        for i in range(50, len(df) - 20):  # Leave room for forward analysis
            current = df.iloc[i]
            
            # Skip if missing essential data
            if pd.isna(current['atr']) or pd.isna(current['bb_pct']):
                continue
            
            # Check for BB setup
            setup_type = None
            
            # LONG setup - near lower band
            if current['bb_pct'] <= (0.1 + config['entry_condition']):
                if current['rsi'] < 35:  # Oversold
                    setup_type = 'LONG'
            
            # SHORT setup - near upper band  
            elif current['bb_pct'] >= (0.9 - config['entry_condition']):
                if current['rsi'] > 65:  # Overbought
                    setup_type = 'SHORT'
            
            if setup_type:
                # Simulate the trade
                trade_result = self._simulate_trade(df, i, setup_type, config)
                if trade_result:
                    trade_result['symbol'] = symbol
                    trades.append(trade_result)
        
        return trades

    def _simulate_trade(self, df, entry_idx, setup_type, config):
        """Simulate a single trade execution"""
        try:
            entry_row = df.iloc[entry_idx]
            entry_price = entry_row['close']
            atr = entry_row['atr']
            
            # Calculate stop loss and take profit
            if setup_type == 'LONG':
                stop_loss = entry_price - (atr * config['stop_loss_atr_mult'])
                
                if config['take_profit_strategy'] == 'bb_middle':
                    take_profit = entry_row['bb_middle']
                elif config['take_profit_strategy'] == 'bb_opposite':
                    take_profit = entry_row['bb_upper']
                else:  # risk_reward_2_1
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * 2)
            
            else:  # SHORT
                stop_loss = entry_price + (atr * config['stop_loss_atr_mult'])
                
                if config['take_profit_strategy'] == 'bb_middle':
                    take_profit = entry_row['bb_middle']
                elif config['take_profit_strategy'] == 'bb_opposite':
                    take_profit = entry_row['bb_lower']
                else:  # risk_reward_2_1
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * 2)
            
            # Apply execution costs
            slippage_cost = config['slippage']
            fees_cost = config['fees']
            
            if setup_type == 'LONG':
                actual_entry = entry_price * (1 + slippage_cost)
            else:
                actual_entry = entry_price * (1 - slippage_cost)
            
            # Simulate forward price action
            max_hold_periods = 168  # 7 days in hours
            
            for hours_ahead in range(1, min(max_hold_periods, len(df) - entry_idx)):
                future_idx = entry_idx + hours_ahead
                if future_idx >= len(df):
                    break
                
                future_row = df.iloc[future_idx]
                
                # Check exit conditions
                if setup_type == 'LONG':
                    # Stop loss hit
                    if future_row['low'] <= stop_loss:
                        exit_price = stop_loss * (1 - slippage_cost)
                        gross_return = (exit_price - actual_entry) / actual_entry
                        net_return = gross_return - fees_cost
                        
                        return {
                            'setup_type': setup_type,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'exit_reason': 'STOP_LOSS',
                            'hours_held': hours_ahead,
                            'gross_return': gross_return * 100,
                            'net_return': net_return * 100,
                            'config': config.copy()
                        }
                    
                    # Take profit hit
                    if future_row['high'] >= take_profit:
                        exit_price = take_profit * (1 - slippage_cost)
                        gross_return = (exit_price - actual_entry) / actual_entry
                        net_return = gross_return - fees_cost
                        
                        return {
                            'setup_type': setup_type,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'exit_reason': 'TARGET_HIT',
                            'hours_held': hours_ahead,
                            'gross_return': gross_return * 100,
                            'net_return': net_return * 100,
                            'config': config.copy()
                        }
                
                else:  # SHORT
                    # Stop loss hit
                    if future_row['high'] >= stop_loss:
                        exit_price = stop_loss * (1 + slippage_cost)
                        gross_return = (actual_entry - exit_price) / actual_entry
                        net_return = gross_return - fees_cost
                        
                        return {
                            'setup_type': setup_type,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'exit_reason': 'STOP_LOSS',
                            'hours_held': hours_ahead,
                            'gross_return': gross_return * 100,
                            'net_return': net_return * 100,
                            'config': config.copy()
                        }
                    
                    # Take profit hit
                    if future_row['low'] <= take_profit:
                        exit_price = take_profit * (1 + slippage_cost)
                        gross_return = (actual_entry - exit_price) / actual_entry
                        net_return = gross_return - fees_cost
                        
                        return {
                            'setup_type': setup_type,
                            'entry_price': actual_entry,
                            'exit_price': exit_price,
                            'exit_reason': 'TARGET_HIT',
                            'hours_held': hours_ahead,
                            'gross_return': gross_return * 100,
                            'net_return': net_return * 100,
                            'config': config.copy()
                        }
            
            # Time-based exit if no other exit triggered
            final_row = df.iloc[min(entry_idx + max_hold_periods - 1, len(df) - 1)]
            final_price = final_row['close']
            
            if setup_type == 'LONG':
                exit_price = final_price * (1 - slippage_cost)
                gross_return = (exit_price - actual_entry) / actual_entry
            else:
                exit_price = final_price * (1 + slippage_cost)
                gross_return = (actual_entry - exit_price) / actual_entry
            
            net_return = gross_return - fees_cost
            
            return {
                'setup_type': setup_type,
                'entry_price': actual_entry,
                'exit_price': exit_price,
                'exit_reason': 'TIME_EXIT',
                'hours_held': max_hold_periods,
                'gross_return': gross_return * 100,
                'net_return': net_return * 100,
                'config': config.copy()
            }
            
        except Exception as e:
            print(f"Error simulating trade: {e}")
            return None

    def _calculate_performance(self, trades, config):
        """Calculate performance metrics for a parameter set"""
        if not trades:
            return None
        
        df_trades = pd.DataFrame(trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['net_return'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        avg_win = df_trades[df_trades['net_return'] > 0]['net_return'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['net_return'] <= 0]['net_return'].mean() if total_trades > winning_trades else 0
        
        total_return = df_trades['net_return'].sum()
        avg_return_per_trade = total_return / total_trades
        avg_hold_time = df_trades['hours_held'].mean()
        
        # Profit factor
        total_wins = df_trades[df_trades['net_return'] > 0]['net_return'].sum() if winning_trades > 0 else 0
        total_losses = abs(df_trades[df_trades['net_return'] <= 0]['net_return'].sum()) if total_trades > winning_trades else 0.01
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = df_trades['net_return'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns - running_max
        max_drawdown = drawdowns.min()
        
        return {
            'config': config,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_hold_time': avg_hold_time,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': trades
        }

    def _analyze_optimization_results(self, results):
        """Analyze and display optimization results"""
        print(f"\n📊 PARAMETER OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        if not results:
            print("❌ No results to analyze")
            return
        
        # Sort by profit factor (risk-adjusted returns)
        results.sort(key=lambda x: x['profit_factor'], reverse=True)
        
        print(f"🎯 TESTED {len(results)} PARAMETER COMBINATIONS\n")
        
        print("🏆 TOP 5 CONFIGURATIONS BY PROFIT FACTOR:")
        print("-" * 70)
        
        for i, result in enumerate(results[:5], 1):
            config = result['config']
            print(f"\n#{i} CONFIGURATION:")
            print(f"   Stop Loss: {config['stop_loss_atr_mult']}x ATR")
            print(f"   Take Profit: {config['take_profit_strategy']}")
            print(f"   Entry: {config['entry_condition']*100:.0f}% inside BB bands")
            print(f"   📊 Results: {result['total_trades']} trades | {result['win_rate']:.1f}% win rate")
            print(f"   💰 Returns: {result['total_return']:.1f}% total | {result['avg_return_per_trade']:.2f}% per trade")
            print(f"   🎯 Profit Factor: {result['profit_factor']:.2f}")
            print(f"   📉 Max Drawdown: {result['max_drawdown']:.1f}%")
            print(f"   ⏱️  Avg Hold: {result['avg_hold_time']:.1f} hours")
        
        # Analysis by parameter
        print(f"\n📈 ANALYSIS BY STOP LOSS LEVEL:")
        print("-" * 50)
        
        # Group by stop loss multiplier
        sl_analysis = {}
        for result in results:
            sl_mult = result['config']['stop_loss_atr_mult']
            if sl_mult not in sl_analysis:
                sl_analysis[sl_mult] = []
            sl_analysis[sl_mult].append(result)
        
        for sl_mult in sorted(sl_analysis.keys()):
            group = sl_analysis[sl_mult]
            avg_win_rate = sum(r['win_rate'] for r in group) / len(group)
            avg_profit_factor = sum(r['profit_factor'] for r in group) / len(group)
            print(f"   {sl_mult}x ATR: {avg_win_rate:.1f}% win rate | {avg_profit_factor:.2f} profit factor")
        
        print(f"\n💡 KEY INSIGHTS:")
        best_config = results[0]['config']
        print(f"   🎯 Optimal Stop Loss: {best_config['stop_loss_atr_mult']}x ATR")
        print(f"   🎯 Optimal Take Profit: {best_config['take_profit_strategy']}")
        print(f"   🎯 Optimal Entry: {best_config['entry_condition']*100:.0f}% inside bands")
        
        # Risk analysis
        conservative_results = [r for r in results if r['max_drawdown'] > -10]  # Less than 10% drawdown
        if conservative_results:
            best_conservative = max(conservative_results, key=lambda x: x['profit_factor'])
            print(f"\n🛡️  BEST LOW-RISK CONFIG (< 10% drawdown):")
            config = best_conservative['config']
            print(f"   Stop Loss: {config['stop_loss_atr_mult']}x ATR | Take Profit: {config['take_profit_strategy']}")
            print(f"   Win Rate: {best_conservative['win_rate']:.1f}% | Profit Factor: {best_conservative['profit_factor']:.2f}")
            print(f"   Max Drawdown: {best_conservative['max_drawdown']:.1f}%")

def main():
    """Main execution function"""
    engine = TradeSimulationEngine()
    results = engine.run_parameter_optimization()
    
    print(f"\n🎉 PARAMETER OPTIMIZATION COMPLETE!")
    return results

if __name__ == "__main__":
    main()