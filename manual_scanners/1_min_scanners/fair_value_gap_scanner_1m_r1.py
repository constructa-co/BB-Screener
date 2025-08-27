#!/usr/bin/env python3
"""
Fair Value Gap 1M Scanner R1 - Database Integration
Handles special characters in filename safely
"""

import os
import sys
import importlib.util
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Dynamic import to handle '+' in filename
def load_fvg_scanner():
    """Dynamically load scanner with special character in filename"""
    scanner_path = os.path.join(
        project_root,
        "manual_scanners",
        "1_min_scanners", 
        "fair_value_gap_+_fibonacci_scanner_1m_r0.py"
    )
    
    if not os.path.exists(scanner_path):
        print(f"[FVG R1] ERROR: Scanner not found at {scanner_path}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("fvg_fib_scanner", scanner_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try to find the scanner class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and 'scanner' in attr_name.lower():
                    print(f"[FVG R1] Found scanner class: {attr_name}")
                    return attr
            
            print("[FVG R1] Warning: No scanner class found, trying direct execution")
            return module
    except Exception as e:
        print(f"[FVG R1] ERROR loading scanner: {e}")
        return None

# Import logger
try:
    from database_and_logging.fair_value_gap_logger import FairValueGapLogger
    DB_LOGGING_ENABLED = True
except ImportError as e:
    print(f"[FVG R1] Logger not available: {e}")
    DB_LOGGING_ENABLED = False

class FairValueGapScanner1MR1:
    """R1 wrapper for FVG 1M scanner"""
    
    def __init__(self):
        self.timeframe = '1m'
        self.scanner_name = 'FVG 1M Scanner R1'
        self.base_scanner = load_fvg_scanner()
        
        if DB_LOGGING_ENABLED:
            self.db_logger = FairValueGapLogger(timeframe=self.timeframe)
        else:
            self.db_logger = None
    
    def run(self):
        """Execute scanner with database logging"""
        print(f"\n{'='*60}")
        print(f"🎯 {self.scanner_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        if not self.base_scanner:
            print("[FVG R1] ERROR: Base scanner not available")
            return []
        
        try:
            # Try different execution patterns
            results = None
            
            if isinstance(self.base_scanner, type):
                # It's a class, instantiate and run
                scanner_instance = self.base_scanner()
                if hasattr(scanner_instance, 'scan_for_fvg_fibonacci_setups'):
                    results = scanner_instance.scan_for_fvg_fibonacci_setups()
                elif hasattr(scanner_instance, 'run'):
                    results = scanner_instance.run()
                elif hasattr(scanner_instance, 'scan'):
                    results = scanner_instance.scan()
            else:
                # It's a module, try to execute directly
                if hasattr(self.base_scanner, 'main'):
                    results = self.base_scanner.main()
                elif hasattr(self.base_scanner, 'run'):
                    results = self.base_scanner.run()
            
            if results is None:
                print("[FVG R1] No results from scanner")
                return []
            
            # Process results
            logged_count = 0
            quality_signals = []
            
            for result in results if isinstance(results, list) else [results]:
                if not isinstance(result, dict):
                    continue
                
                symbol = result.get('symbol', result.get('ticker', 'UNKNOWN'))
                score = int(result.get('setup_score', result.get('score', 0)))
                
                # Quality filter
                if score >= 60:
                    quality_signals.append(result)
                    
                    if self.db_logger:
                        # Extract FVG data from the combined output
                        fvg_data = self._extract_fvg_data(result)
                        if self.db_logger.log_fvg(symbol, fvg_data):
                            logged_count += 1
                            print(f"  ✅ {symbol}: Score {score}")
                else:
                    print(f"  ⚠️ {symbol}: Score {score} < 60 (skipped)")
            
            # Summary
            print(f"\n{'='*60}")
            print(f"🎯 Found: {len(results) if isinstance(results, list) else 1}")
            print(f"🎯 Quality (60+): {len(quality_signals)}")
            if self.db_logger:
                print(f"🎯 Logged to DB: {logged_count}")
            print(f"{'='*60}\n")
            
            return quality_signals
            
        except Exception as e:
            print(f"[FVG R1] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_fvg_data(self, result):
        """Extract FVG-specific data from the combined scanner output"""
        try:
            # Extract gap data
            gap = result.get('gap', {})
            fib = result.get('fibonacci', {})
            trade = result.get('trade', {})
            
            # Map the data to our database schema
            fvg_data = {
                'gap_type': trade.get('direction', 'UNKNOWN').upper(),
                'gap_high': gap.get('gap_top'),
                'gap_low': gap.get('gap_bottom'),
                'gap_size_pct': gap.get('gap_size_pct'),
                'current_price': result.get('current_price'),
                'entry_price': trade.get('entry_price'),
                'stop_loss': trade.get('stop_loss'),
                'target_1': trade.get('targets', {}).get('TP1', {}).get('price'),
                'target_2': trade.get('targets', {}).get('TP2', {}).get('price'),
                'target_3': trade.get('targets', {}).get('TP3', {}).get('price'),
                'risk_reward_1': trade.get('targets', {}).get('TP1', {}).get('risk_reward'),
                'risk_reward_2': trade.get('targets', {}).get('TP2', {}).get('risk_reward'),
                'risk_reward_3': trade.get('targets', {}).get('TP3', {}).get('risk_reward'),
                'fib_level': fib.get('confluence_level'),
                'fib_confluence': fib.get('confluence_score', 0) > 5,
                'fib_confluence_score': fib.get('confluence_score'),
                'setup_score': result.get('quality_score', result.get('setup_score', 0)),
                'volume_at_gap': gap.get('volume'),
                'volume_confirmation': gap.get('volume_confirmed', False),
                'momentum_confirmation': gap.get('momentum_confirmed', False),
                'gap_age': result.get('gap_age'),
                'source_scanner': 'fvg_1m_r1',
                'scanner_version': '1.0.0'
            }
            
            return fvg_data
            
        except Exception as e:
            print(f"[FVG R1] Error extracting FVG data: {e}")
            return {}

def main():
    scanner = FairValueGapScanner1MR1()
    scanner.run()

if __name__ == "__main__":
    main()
