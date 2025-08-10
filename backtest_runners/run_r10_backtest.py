#!/usr/bin/env python3
"""
Run R10 Volume Profile Backtest
"""

import sys
import os

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import importlib.util
spec = importlib.util.spec_from_file_location("VolumeProfileBacktest", "backtest_modules/4_hour_backtest_modules/volume_profile_backtest_4h_r10.py")
VolumeProfileBacktest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VolumeProfileBacktest)
VolumeProfileBacktest = VolumeProfileBacktest.VolumeProfileBacktest

def main():
    print("🚀 Starting R10 Volume Profile Backtest...")
    
    # Initialize backtest
    data_path = "data"  # Adjust path as needed
    output_path = "backtest_results/volume_profile"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize backtest
    backtest = VolumeProfileBacktest(data_path, output_path)
    
    # Run backtest
    print("📊 Running R10 backtest...")
    results = backtest.run_simple_backtest()
    
    print("✅ R10 backtest completed!")
    print(f"📈 Results saved to: {output_path}")
    
    # Print summary
    if results:
        print("\n📊 R10 Performance Summary:")
        print(f"Total Trades: {results.get('total_trades', 0)}")
        print(f"Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.1%}")
        print(f"Total P&L: {results.get('total_pnl', 0):.1%}")

if __name__ == "__main__":
    main() 