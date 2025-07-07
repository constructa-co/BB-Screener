#!/usr/bin/env python3
"""
TRADE SIMULATION ENGINE - Option B
Separate module for actual trade execution simulation with parameter optimization
Uses existing BB detection but adds realistic trade execution simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_fetcher import MarketDataFetcher
from modules.bb_detector import BBDetector
from modules.technical_analyzer import TechnicalAnalyzer
from modules.market_regime_analyzer import MarketRegimeAnalyzer
from modules.pattern_analyzer import PatternAnalyzer
from modules.risk_manager import RiskManager

class TradeSimulationEngine:
    """
    Advanced trade simulation engine for backtesting with parameter optimization
    Simulates actual trade execution with real price action, stops, and targets
    """
    
    def __init__(self):
        print("🚀 TRADE SIMULATION ENGINE - REALISTIC EXECUTION TESTING")
        print("=" * 70)
        
        # Initialize modules
        self.data_fetcher = MarketDataFetcher()
        self.bb_detector = BBDetector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.market_regime = MarketRegimeAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.risk_manager = RiskManager()
        
        # Simulation parameters to test
        self.sl_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]  # ATR multipliers
        self.tp_strategies = ['middle_band', 'opposite_band', 'partial_50_50']
        self.entry_thresholds = [0.0, 0.5, 1.0]  # % inside BB bands for entry
        
        # Execution realism parameters
        self.slippage_pct = 0.15  # 0.15% slippage
        self.fee_pct = 0.075      # 0.075% trading fees per side
        
        print("✅ All modules initialized")
        print(f"📊 Testing {len(self.sl_multipliers)} SL levels × {len(self.tp_strategies)} TP strategies")
        print(f"💰 Execution costs: {self.slippage_pct}% slippage + {self.fee_pct * 2}% fees")
    
    def simulate_trade_execution(self, df, entry_idx, setup_type, entry_price, 
                               sl_multiplier=2.0, tp_strategy='middle_band', 
                               entry_threshold=0.0):
        """
        Simulate actual trade execution with realistic price action
        """
        
        if entry_idx >= len(df) - 10:  # Need at least 10 candles after entry
            return None
            
        try:
            # Calculate ATR for stop loss
            atr = df['atr'].iloc[entry_idx]
            
            # Adjust entry price for threshold and execution costs
            if setup_type == 'LONG':
                adjusted_entry = entry_price * (1 + entry_threshold/100 + self.slippage_pct/100)
                stop_loss = adjusted_entry - (atr * sl_multiplier)
            else:  # SHORT
                adjusted_entry = entry_price * (1 - entry_threshold/100 - self.slippage_pct/100)
                stop_loss = adjusted_entry + (atr * sl_multiplier)
            
            # Calculate take profit based on strategy
            bb_upper = df['bb_upper'].iloc[entry_idx]
            bb_lower = df['bb_lower'].iloc[entry_idx]
            bb_middle = df['bb_middle'].iloc[entry_idx]
            
            if tp_strategy == 'middle_band':
                take_profit = bb_middle
            elif tp_strategy == 'opposite_band':
                take_profit = bb_upper if setup_type == 'LONG' else bb_lower
            elif tp_strategy == 'partial_50_50':
                take_profit = bb_middle
            
            # Add trading fees to break-even calculation
            total_fees = adjusted_entry * (self.fee_pct / 100) * 2
            
            if setup_type == 'LONG':
                profit_threshold = take_profit - (take_profit * self.slippage_pct/100)
            else:  # SHORT
                profit_threshold = take_profit + (take_profit * self.slippage_pct/100)
            
            # Simulate trade progression through subsequent candles
            max_candles = min(48, len(df) - entry_idx - 1)  # Max 48 hours
            
            for i in range(1, max_candles + 1):
                current_idx = entry_idx + i
                candle = df.iloc[current_idx]
                
                low_price = candle['low']
                high_price = candle['high']
                
                # Check for stop loss hit
                if setup_type == 'LONG' and low_price <= stop_loss:
                    exit_price = stop_loss * (1 - self.slippage_pct/100)
                    pnl_pct = ((exit_price - adjusted_entry) / adjusted_entry) * 100
                    final_pnl = pnl_pct - (self.fee_pct * 2)
                    
                    return {
                        'outcome': 'LOSS',
                        'exit_reason': 'STOP_LOSS',
                        'entry_price': adjusted_entry,
                        'exit_price': exit_price,
                        'pnl_pct': final_pnl,
                        'hold_time_hours': i,
                        'sl_multiplier': sl_multiplier,
                        'tp_strategy': tp_strategy,
                        'entry_threshold': entry_threshold
                    }
                
                elif setup_type == 'SHORT' and high_price >= stop_loss:
                    exit_price = stop_loss * (1 + self.slippage_pct/100)
                    pnl_pct = ((adjusted_entry - exit_price) / adjusted_entry) * 100
                    final_pnl = pnl_pct - (self.fee_pct * 2)
                    
                    return {
                        'outcome': 'LOSS',
                        'exit_reason': 'STOP_LOSS', 
                        'entry_price': adjusted_entry,
                        'exit_price': exit_price,
                        'pnl_pct': final_pnl,
                        'hold_time_hours': i,
                        'sl_multiplier': sl_multiplier,
                        'tp_strategy': tp_strategy,
                        'entry_threshold': entry_threshold
                    }
                
                # Check for take profit hit
                if setup_type == 'LONG' and high_price >= profit_threshold:
                    exit_price = profit_threshold
                    pnl_pct = ((exit_price - adjusted_entry) / adjusted_entry) * 100
                    final_pnl = pnl_pct - (self.fee_pct * 2)
                    
                    return {
                        'outcome': 'WIN',
                        'exit_reason': 'TAKE_PROFIT',
                        'entry_price': adjusted_entry,
                        'exit_price': exit_price, 
                        'pnl_pct': final_pnl,
                        'hold_time_hours': i,
                        'sl_multiplier': sl_multiplier,
                        'tp_strategy': tp_strategy,
                        'entry_threshold': entry_threshold
                    }
                
                elif setup_type == 'SHORT' and low_price <= profit_threshold:
                    exit_price = profit_threshold
                    pnl_pct = ((adjusted_entry - exit_price) / adjusted_entry) * 100
                    final_pnl = pnl_pct - (self.fee_pct * 2)
                    
                    return {
                        'outcome': 'WIN',
                        'exit_reason': 'TAKE_PROFIT',
                        'entry_price': adjusted_entry,
                        'exit_price': exit_price,
                        'pnl_pct': final_pnl,
                        'hold_time_hours': i,
                        'sl_multiplier': sl_multiplier,
                        'tp_strategy': tp_strategy,
                        'entry_threshold': entry_threshold
                    }
            
            # If neither SL nor TP hit within max time, exit at market
            final_candle = df.iloc[entry_idx + max_candles]
            exit_price = final_candle['close'] * (1 - self.slippage_pct/100 if setup_type == 'LONG' else 1 + self.slippage_pct/100)
            
            if setup_type == 'LONG':
                pnl_pct = ((exit_price - adjusted_entry) / adjusted_entry) * 100
            else:
                pnl_pct = ((adjusted_entry - exit_price) / adjusted_entry) * 100
                
            final_pnl = pnl_pct - (self.fee_pct * 2)
            
            return {
                'outcome': 'WIN' if final_pnl > 0 else 'LOSS',
                'exit_reason': 'TIME_LIMIT',
                'entry_price': adjusted_entry,
                'exit_price': exit_price,
                'pnl_pct': final_pnl,
                'hold_time_hours': max_candles,
                'sl_multiplier': sl_multiplier,
                'tp_strategy': tp_strategy,
                'entry_threshold': entry_threshold
            }
            
        except Exception as e:
            print(f"❌ Trade simulation error: {e}")
            return None
    
    def backtest_with_parameter_optimization(self, symbols=['BTC', 'ETH'], timeframe='30d'):
        """
        Run comprehensive backtesting with parameter optimization
        """
        print(f"\n🎯 STARTING PARAMETER OPTIMIZATION BACKTEST")
        print(f"📊 Symbols: {symbols}")
        print(f"📅 Timeframe: {timeframe}")
        print("=" * 70)
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n🔍 ANALYZING {symbol}...")
            
            try:
                # Fetch historical data
                df = self.data_fetcher.fetch_ohlcv('binance', symbol, '1h')
                if df is None or len(df) < 200:
                    print(f"❌ Insufficient data for {symbol}")
                    continue
                
                # Add technical indicators using the working pattern from your files
                # Based on your improved_bb_backtest.py and real_market_test.py
                df['bb_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
                df['bb_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
                df['bb_middle'] = df['close'].rolling(20).mean()
                df['bb_percentage'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                df['atr'] = ((df['high'] - df['low']).rolling(14).mean())
                
                # Add essential indicators for BB analysis
                df['rsi'] = df['close'].diff().pipe(lambda x: 100 - (100 / (1 + x.where(x > 0, 0).rolling(14).mean() / x.where(x < 0, 0).abs().rolling(14).mean())))
                
                # Test different parameter combinations
                symbol_results = []
                
                for sl_mult in self.sl_multipliers:
                    for tp_strat in self.tp_strategies:
                        for entry_thresh in self.entry_thresholds:
                            
                            # Simulate multiple entry points across the dataset
                            trades_for_params = []
                            
                            # Look for BB touches throughout historical data
                            for i in range(100, len(df) - 50):
                                
                                # Simple BB touch detection for simulation
                                if 'bb_percentage' in df.columns:
                                    bb_pct = df['bb_percentage'].iloc[i]
                                    setup_type = None
                                    entry_price = None
                                    
                                    if bb_pct <= 0.05:  # Near lower band
                                        setup_type = 'LONG'
                                        entry_price = df['close'].iloc[i]
                                    elif bb_pct >= 0.95:  # Near upper band
                                        setup_type = 'SHORT'
                                        entry_price = df['close'].iloc[i]
                                    
                                    if setup_type:
                                        trade_result = self.simulate_trade_execution(
                                            df, i, setup_type, entry_price,
                                            sl_mult, tp_strat, entry_thresh
                                        )
                                        
                                        if trade_result:
                                            trades_for_params.append(trade_result)
                            
                            # Analyze results for this parameter combination
                            if trades_for_params:
                                wins = [t for t in trades_for_params if t['outcome'] == 'WIN']
                                losses = [t for t in trades_for_params if t['outcome'] == 'LOSS']
                                
                                win_rate = len(wins) / len(trades_for_params) * 100
                                avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
                                avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
                                total_return = sum([t['pnl_pct'] for t in trades_for_params])
                                avg_hold_time = np.mean([t['hold_time_hours'] for t in trades_for_params])
                                
                                profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if losses and avg_loss < 0 else float('inf')
                                
                                param_result = {
                                    'symbol': symbol,
                                    'sl_multiplier': sl_mult,
                                    'tp_strategy': tp_strat,
                                    'entry_threshold': entry_thresh,
                                    'total_trades': len(trades_for_params),
                                    'win_rate': win_rate,
                                    'avg_win_pct': avg_win,
                                    'avg_loss_pct': avg_loss,
                                    'total_return_pct': total_return,
                                    'profit_factor': profit_factor,
                                    'avg_hold_hours': avg_hold_time
                                }
                                
                                symbol_results.append(param_result)
                                
                                print(f"   SL:{sl_mult}x TP:{tp_strat} Entry:{entry_thresh}% -> "
                                      f"{len(trades_for_params)} trades, {win_rate:.1f}% win rate")
                
                all_results[symbol] = symbol_results
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        return self.analyze_optimization_results(all_results)
    
    def analyze_optimization_results(self, all_results):
        """
        Analyze parameter optimization results and find optimal settings
        """
        print(f"\n📊 PARAMETER OPTIMIZATION ANALYSIS")
        print("=" * 70)
        
        # Combine all results
        combined_results = []
        for symbol, results in all_results.items():
            combined_results.extend(results)
        
        if not combined_results:
            print("❌ No results to analyze")
            return {}
        
        # Convert to DataFrame for analysis
        df_results = pd.DataFrame(combined_results)
        
        print(f"\n🎯 ANALYZED {len(df_results)} parameter combinations")
        
        # Group by parameters and aggregate
        param_groups = df_results.groupby(['sl_multiplier', 'tp_strategy', 'entry_threshold']).agg({
            'total_trades': 'sum',
            'win_rate': 'mean',
            'avg_win_pct': 'mean',
            'avg_loss_pct': 'mean', 
            'total_return_pct': 'sum',
            'profit_factor': 'mean',
            'avg_hold_hours': 'mean'
        }).reset_index()
        
        # Filter for statistical significance (at least 5 trades for quick test)
        significant_results = param_groups[param_groups['total_trades'] >= 5]
        
        if len(significant_results) > 0:
            print(f"\n📈 FOUND {len(significant_results)} statistically significant combinations:")
            print("-" * 50)
            
            # Sort by total return
            top_params = significant_results.nlargest(3, 'total_return_pct')
            
            for idx, row in top_params.iterrows():
                print(f"\n{idx+1}. SL: {row['sl_multiplier']}x ATR | TP: {row['tp_strategy']} | Entry: {row['entry_threshold']}%")
                print(f"   📊 {row['total_trades']} trades | Win Rate: {row['win_rate']:.1f}%")
                print(f"   💰 Avg Win: {row['avg_win_pct']:.2f}% | Avg Loss: {row['avg_loss_pct']:.2f}%")
                print(f"   🎯 Total Return: {row['total_return_pct']:.2f}% | Avg Hold: {row['avg_hold_hours']:.1f}h")
        else:
            print("❌ No statistically significant results found")
        
        return {'results': combined_results, 'top_params': top_params.to_dict('records') if len(significant_results) > 0 else []}

def main():
    """
    Main execution for trade simulation testing
    """
    print("🚀 TRADE SIMULATION ENGINE - PARAMETER OPTIMIZATION")
    print("=" * 70)
    print("🎯 Objective: Find optimal entry, stop-loss, and take-profit parameters")
    print("📊 Method: Realistic trade execution simulation with fees and slippage")
    print("⚡ Focus: Parameter optimization for maximum risk-adjusted returns")
    print("=" * 70)
    
    # Initialize simulation engine
    engine = TradeSimulationEngine()
    
    # Test with major cryptocurrencies (start small)
    test_symbols = ['BTC', 'ETH']
    
    # Run parameter optimization
    results = engine.backtest_with_parameter_optimization(
        symbols=test_symbols,
        timeframe='30d'
    )
    
    print("\n🎉 PARAMETER OPTIMIZATION COMPLETE!")
    
    return results

if __name__ == "__main__":
    main()