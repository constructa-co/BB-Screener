#!/usr/bin/env python3
"""
FAIR VALUE GAP (FVG) + FIBONACCI SCANNER 4H - R1 WRAPPER
Database integration wrapper for 4-hour FVG scanner
"""

import sys
import os
import importlib.util
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database_and_logging.fair_value_gap_logger import FairValueGapLogger

def load_r0_scanner():
    """Dynamically load the R0 scanner module"""
    try:
        # Path to the R0 scanner
        r0_path = os.path.join(os.path.dirname(__file__), 'fair_value_gap_+_fibonacci_scanner_4h_r0.py')
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("fvg_4h_r0", r0_path)
        r0_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r0_module)
        
        return r0_module.AdvancedFVGScanner4h()
    except Exception as e:
        print(f"❌ Error loading R0 scanner: {e}")
        return None

def extract_fvg_data(scanner_output):
    """Extract FVG data from scanner output for database logging"""
    try:
        if not scanner_output or not isinstance(scanner_output, list) or len(scanner_output) == 0:
            return None
        
        # Extract the first opportunity (highest quality score)
        setup = scanner_output[0]
        
        # Extract basic trade data
        gap = setup['gap']
        trade = setup['trade']
        fib = setup['fibonacci']
        
        fvg_data = {
            'symbol': setup.get('symbol', ''),
            'timeframe': '4h',
            'gap_type': 'BULLISH' if trade.get('direction') == 'LONG' else 'BEARISH',
            'gap_high': float(gap.get('gap_top', 0)),
            'gap_low': float(gap.get('gap_bottom', 0)),
            'gap_size': float(gap.get('gap_size', 0)),
            'gap_size_pct': float(gap.get('gap_size_pct', 0)),
            'current_price': float(setup.get('current_price', 0)),
            'entry_price': float(trade.get('entry_price', 0)),
            'stop_loss': float(trade.get('stop_loss', 0)),
            'target_1': float(trade.get('targets', {}).get('TP1', {}).get('price', 0)),
            'target_2': float(trade.get('targets', {}).get('TP2', {}).get('price', 0)),
            'target_3': float(trade.get('targets', {}).get('TP3', {}).get('price', 0)),
            'risk_reward_1': float(trade.get('targets', {}).get('TP1', {}).get('risk_reward', 0)),
            'risk_reward_2': float(trade.get('targets', {}).get('TP2', {}).get('risk_reward', 0)),
            'risk_reward_3': float(trade.get('targets', {}).get('TP3', {}).get('risk_reward', 0)),
            'setup_score': int(setup.get('quality_score', 0)),
            'volume_at_gap': float(gap.get('volume_at_gap', 0)),
            'volume_confirmation': bool(gap.get('volume_confirmation', False)),
            'momentum_confirmation': bool(gap.get('momentum_confirmation', False)),
            'gap_status': 'OPEN',
            'fill_percentage': float(gap.get('fill_percentage', 0)),
            'gap_age_minutes': int(setup.get('gap_age', 0) * 240),  # Convert 4H candles to minutes
            'entry_timing': trade.get('entry_timing', 'waiting'),
            'current_distance_pct': float(trade.get('current_distance_pct', 0)),
            'risk_pct': float(trade.get('risk_pct', 0)),
        }
        
        # Extract Fibonacci data
        fvg_data.update({
            'fib_level': float(fib.get('fib_level', 0)),
            'fib_confluence': bool(fib.get('confluence', False)),
            'fib_confluence_score': int(fib.get('confluence_score', 0)),
            'swing_high': float(fib.get('swing_high', {}).get('price', 0)),
            'swing_low': float(fib.get('swing_low', {}).get('price', 0)),
            'fib_levels': str(fib.get('fib_levels', [])),
            'target_levels': str(fib.get('target_levels', [])),
        })
        
        # Convert all numeric fields to native Python types
        for key, value in fvg_data.items():
            if isinstance(value, (int, float, bool)):
                fvg_data[key] = type(value)(value)
        
        return fvg_data
        
    except Exception as e:
        print(f"❌ Error extracting FVG data: {e}")
        return None

def main():
    """Main execution function for 4H FVG scanner"""
    print("🚀 Starting 4H FVG Scanner R1...")
    
    try:
        # Load R0 scanner
        scanner = load_r0_scanner()
        if not scanner:
            print("❌ Failed to load R0 scanner")
            return
        
        # Initialize database logger
        logger = FairValueGapLogger()
        
        # Run the scanner
        print("📊 Running 4H FVG scan...")
        scanner_output = scanner.scan_for_fvg_fibonacci_setups()
        
        if not scanner_output:
            print("⚠️ No scanner output received")
            return
        
        # Extract and log FVG data
        fvg_data = extract_fvg_data(scanner_output)
        if fvg_data:
            logger.log_fvg_signal(fvg_data)
            print(f"✅ 4H FVG signal logged for {fvg_data['symbol']}")
        else:
            print("⚠️ No valid FVG data to log")
        
        print("✅ 4H FVG Scanner R1 completed successfully")
        
    except Exception as e:
        print(f"❌ Error in 4H FVG Scanner R1: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
