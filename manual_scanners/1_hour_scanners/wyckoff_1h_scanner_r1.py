#!/usr/bin/env python3
"""
Wyckoff 1H Scanner R1 - Database Integration Wrapper
File: manual_scanners/1_hour_scanners/wyckoff_1h_scanner_r1.py
Purpose: Add database logging to R0 scanner without modifying original code
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import base scanner (R0)
try:
    from wyckoff_1h_scanner_r0 import WyckoffScanner1H
    BASE_SCANNER_AVAILABLE = True
    print("✅ Base Wyckoff scanner (R0) imported successfully")
except ImportError as e:
    print(f"❌ Base scanner not available: {e}")
    BASE_SCANNER_AVAILABLE = False
    WyckoffScanner1H = object  # Fallback

# Import logger
try:
    from database_and_logging.wyckoff_logger import WyckoffLogger
    DB_LOGGING_ENABLED = True
    print("✅ Database logging enabled for Wyckoff scanner")
except ImportError as e:
    print(f"❌ Database logging not available: {e}")
    DB_LOGGING_ENABLED = False

class WyckoffScanner1HR1(WyckoffScanner1H):
    """Enhanced Wyckoff Scanner with database integration"""
    
    def __init__(self):
        if BASE_SCANNER_AVAILABLE:
            super().__init__()
        self.timeframe = '1h'
        self.scanner_name = 'Wyckoff Scanner R1'
        
        # Initialize logger if available
        if DB_LOGGING_ENABLED:
            try:
                self.db_logger = WyckoffLogger(timeframe=self.timeframe)
                print(f"[{self.scanner_name}] Database logger initialized")
            except Exception as e:
                print(f"[{self.scanner_name}] Logger initialization failed: {e}")
                self.db_logger = None
                DB_LOGGING_ENABLED = False
        else:
            self.db_logger = None
            print(f"[{self.scanner_name}] Running without database logging")
    
    def run(self):
        """Main execution with database logging"""
        if not BASE_SCANNER_AVAILABLE:
            print(f"[{self.scanner_name}] ERROR: Base scanner not available")
            return []
        
        print(f"\n{'='*60}")
        print(f"⚡ {self.scanner_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Run base scanner
        try:
            # Call parent's scan method to get Wyckoff setups
            opportunities = super().scan_for_wyckoff_setups()
            
            if not opportunities:
                print("No Wyckoff setups found in this scan")
                return []
            
            # Process and log setups
            logged_count = 0
            quality_setups = []
            
            print(f"\n📊 Processing {len(opportunities)} setups for database logging...")
            
            for setup in opportunities:
                # Extract symbol from setup
                symbol = setup.get('symbol', 'UNKNOWN')
                setup_score = setup.get('setup_score', 0)
                
                # Quality filter (60+ score only)
                if setup_score >= 60:
                    quality_setups.append(setup)
                    
                    # Log to database if available
                    if self.db_logger and DB_LOGGING_ENABLED:
                        try:
                            if self.db_logger.log_setup(symbol, setup):
                                logged_count += 1
                                print(f"  ✅ Logged: {symbol} - {setup.get('pattern', 'Unknown')} (Score: {setup_score})")
                            else:
                                print(f"  ⚠️  Failed to log: {symbol} - Logger returned False")
                        except Exception as e:
                            print(f"  ❌ Failed to log {symbol}: {e}")
                else:
                    print(f"  ⏭️  Skipped: {symbol} - Score {setup_score} below threshold (60)")
            
            # Summary
            print(f"\n{'='*60}")
            print(f"📊 Scan Summary:")
            print(f"  Total setups found: {len(opportunities)}")
            print(f"  Quality setups (60+): {len(quality_setups)}")
            if self.db_logger and DB_LOGGING_ENABLED:
                print(f"  Logged to database: {logged_count}")
                print(f"  Database logging: {'✅ ENABLED' if logged_count > 0 else '⚠️  NO QUALITY SETUPS'}")
            else:
                print(f"  Database logging: ❌ DISABLED")
            print(f"{'='*60}\n")
            
            return quality_setups
            
        except Exception as e:
            print(f"[{self.scanner_name}] ERROR during scan: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'db_logger') and self.db_logger:
            print(f"[{self.scanner_name}] Cleaning up resources...")

def main():
    """Main execution"""
    scanner = WyckoffScanner1HR1()
    opportunities = scanner.run()
    
    # Display results using R0 method if available
    if BASE_SCANNER_AVAILABLE and opportunities:
        try:
            scanner.display_results(opportunities)
        except Exception as e:
            print(f"Warning: Could not display results: {e}")
    
    return opportunities

if __name__ == "__main__":
    # If running via cron, ensure the environment is loaded
    if not os.environ.get("DATABASE_URL"):
        print("WARNING: DATABASE_URL not set; will run scanner without DB logging.")
    
    main()
