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

from database_and_logging.fair_value_gap_logger import FairValueGapLogger

def load_fvg_scanner():
    """Dynamically load the FVG scanner with special characters in filename"""
    try:
        # Path to the R0 scanner with special characters
        scanner_path = os.path.join(
            project_root, 
            "manual_scanners", 
            "1_min_scanners", 
            "fair_value_gap_+_fibonacci_scanner_1m_r0.py"
        )
        
        if not os.path.exists(scanner_path):
            print(f"[FVG R1] ERROR: Scanner file not found: {scanner_path}")
            return None
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("fvg_scanner", scanner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the scanner class
        scanner_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '__name__') and 'Scanner' in attr.__name__:
                scanner_class = attr
                break
        
        if scanner_class:
            print(f"[FVG R1] Found scanner class: {scanner_class.__name__}")
            return scanner_class
        else:
            print("[FVG R1] ERROR: No scanner class found in module")
            return None
            
    except Exception as e:
        print(f"[FVG R1] ERROR loading scanner: {e}")
        return None

class FairValueGapScanner1MR1:
    """R1 wrapper for Fair Value Gap 1M scanner with database integration"""
    
    def __init__(self):
        """Initialize the scanner with database logger"""
        self.base_scanner = load_fvg_scanner()
        if self.base_scanner:
            self.db_logger = FairValueGapLogger(timeframe='1m')
        else:
            self.db_logger = None
            print("[FVG R1] ERROR: Failed to load base scanner")
    
    def run(self):
        """Run the scanner and log results to database"""
        try:
            print("=" * 60)
            print(f"🎯 FVG 1M Scanner R1 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            if not self.base_scanner:
                print("[FVG R1] ERROR: No base scanner available")
                return []
            
            # Execute the base scanner
            if hasattr(self.base_scanner, 'scan_for_fvg_fibonacci_setups'):
                # It's a class, instantiate and call the method
                scanner_instance = self.base_scanner()
                results = scanner_instance.scan_for_fvg_fibonacci_setups()
            elif hasattr(self.base_scanner, 'scan'):
                # It's a class, instantiate and call the method
                scanner_instance = self.base_scanner()
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
                score = int(result.get('quality_score', result.get('setup_score', result.get('score', 0))))
                
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
                'gap_type': 'BULLISH' if trade.get('direction') == 'LONG' else 'BEARISH',
                'gap_high': float(gap.get('gap_top', 0)) if gap.get('gap_top') is not None else None,
                'gap_low': float(gap.get('gap_bottom', 0)) if gap.get('gap_bottom') is not None else None,
                'gap_size_pct': float(gap.get('gap_size_pct', 0)) if gap.get('gap_size_pct') is not None else None,
                'current_price': float(result.get('current_price', 0)) if result.get('current_price') is not None else None,
                'entry_price': float(trade.get('entry_price', 0)) if trade.get('entry_price') is not None else None,
                'stop_loss': float(trade.get('stop_loss', 0)) if trade.get('stop_loss') is not None else None,
                'target_1': float(trade.get('targets', {}).get('TP1', {}).get('price', 0)) if trade.get('targets', {}).get('TP1', {}).get('price') is not None else None,
                'target_2': float(trade.get('targets', {}).get('TP2', {}).get('price', 0)) if trade.get('targets', {}).get('TP2', {}).get('price') is not None else None,
                'target_3': float(trade.get('targets', {}).get('TP3', {}).get('price', 0)) if trade.get('targets', {}).get('TP3', {}).get('price') is not None else None,
                'risk_reward_1': float(trade.get('targets', {}).get('TP1', {}).get('risk_reward', 0)) if trade.get('targets', {}).get('TP1', {}).get('risk_reward') is not None else None,
                'risk_reward_2': float(trade.get('targets', {}).get('TP2', {}).get('risk_reward', 0)) if trade.get('targets', {}).get('TP2', {}).get('risk_reward') is not None else None,
                'risk_reward_3': float(trade.get('targets', {}).get('TP3', {}).get('risk_reward', 0)) if trade.get('targets', {}).get('TP3', {}).get('risk_reward') is not None else None,
                'fib_level': float(fib.get('confluence_level', 0)) if fib.get('confluence_level') is not None else None,
                'fib_confluence': bool(fib.get('confluence_score', 0) > 5),
                'fib_confluence_score': int(fib.get('confluence_score', 0)) if fib.get('confluence_score') is not None else 0,
                'setup_score': int(result.get('quality_score', 0)) if result.get('quality_score') is not None else 0,
                'volume_at_gap': float(gap.get('volume', 0)) if gap.get('volume') is not None else None,
                'volume_confirmation': bool(gap.get('volume_confirmed', False)),
                'momentum_confirmation': bool(gap.get('momentum_confirmed', False)),
                'gap_age': int(result.get('gap_age', 0)) if result.get('gap_age') is not None else 0,
                'entry_timing': str(trade.get('entry_timing', 'unknown')).upper(),
                'current_distance_pct': float(trade.get('current_distance_pct', 0)) if trade.get('current_distance_pct') is not None else None,
                'risk_pct': float(trade.get('risk_pct', 0)) if trade.get('risk_pct') is not None else None,
                'swing_high': float(fib.get('swing_high', {}).get('price', 0)) if fib.get('swing_high', {}).get('price') is not None else None,
                'swing_low': float(fib.get('swing_low', {}).get('price', 0)) if fib.get('swing_low', {}).get('price') is not None else None,
                'fib_levels': str(fib.get('fib_levels', {})),
                'target_levels': str(fib.get('target_levels', {})),
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
