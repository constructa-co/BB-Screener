#!/usr/bin/env python3
"""
Run R11 Volume Profile Backtest - Quality-Focused Optimization
"""

import sys
import os

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import importlib.util
spec = importlib.util.spec_from_file_location("VolumeProfileBacktest", "backtest_modules/4_hour_backtest_modules/volume_profile_backtest_4h_r11.py")
VolumeProfileBacktest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VolumeProfileBacktest)
VolumeProfileBacktest = VolumeProfileBacktest.VolumeProfileBacktest

def main():
    print("🚀 Starting R11 Volume Profile Backtest - Quality-Focused Optimization...")
    
    # Initialize backtest
    data_path = "data"  # Adjust path as needed
    output_path = "backtest_results/volume_profile"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize backtest
    backtest = VolumeProfileBacktest(data_path, output_path)
    
    # Run backtest
    print("📊 Running R11 backtest...")
    results = backtest.run_backtest(['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'BNBUSDT'])
    
    print("✅ R11 backtest completed!")
    print(f"📈 Results saved to: {output_path}")
    
    # Print summary
    if results:
        print("\n📊 R11 Performance Summary:")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.1%}")
        print(f"Total P&L: {results.get('total_pnl', 0):.1%}")
        print(f"Avg PnL: {results.get('avg_pnl', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")

if __name__ == "__main__":
    main() 