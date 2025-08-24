#!/usr/bin/env python3
"""
Wyckoff 4H Scanner R1 - Database Integration Wrapper
File: manual_scanners/4_hour_scanners/wyckoff_scanner_4h_r1.py
Purpose: Add database logging to R0 scanner without modifying original code
"""
import os, sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from wyckoff_scanner_4h_r0 import WyckoffScanner
    BASE_SCANNER_AVAILABLE = True
    print("✅ Base Wyckoff 4H scanner (R0) imported successfully")
except ImportError as e:
    print(f"❌ Base scanner not available: {e}")
    BASE_SCANNER_AVAILABLE = False
    WyckoffScanner = object

try:
    from database_and_logging.wyckoff_logger import WyckoffLogger
    DB_LOGGING_ENABLED = True # Global variable for initial check
    print("✅ Database logging enabled for Wyckoff 4H scanner")
except ImportError as e:
    print(f"❌ Database logging not available: {e}")
    DB_LOGGING_ENABLED = False

class WyckoffScanner4HR1(WyckoffScanner):
    """Enhanced Wyckoff 4H Scanner with database integration"""
    def __init__(self):
        if BASE_SCANNER_AVAILABLE:
            super().__init__()
        self.timeframe = '4h'
        self.scanner_name = 'Wyckoff Scanner 4H R1'

        # Initialize logger if available
        if DB_LOGGING_ENABLED: # Reference global DB_LOGGING_ENABLED here
            try:
                self.db_logger = WyckoffLogger(timeframe=self.timeframe)
                print(f"[{self.scanner_name}] Database logger initialized")
                self._db_logging_enabled = True # Set instance variable
            except Exception as e:
                print(f"[{self.scanner_name}] Logger initialization failed: {e}")
                self.db_logger = None
                self._db_logging_enabled = False # Set instance variable on failure
        else:
            self.db_logger = None
            self._db_logging_enabled = False # Set instance variable if global is False
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
                    if self.db_logger and self._db_logging_enabled: # Reference instance variable
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
            if self.db_logger and self._db_logging_enabled: # Reference instance variable
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
    scanner = WyckoffScanner4HR1()
    opportunities = scanner.run()
    if BASE_SCANNER_AVAILABLE and opportunities:
        try:
            scanner.display_results(opportunities) # Uses R0's display method
        except Exception as e:
            print(f"Warning: Could not display results: {e}")
    return opportunities

if __name__ == "__main__":
    if not os.environ.get("DATABASE_URL"):
        print("WARNING: DATABASE_URL not set; will run scanner without DB logging.")
    main()
