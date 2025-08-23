#!/usr/bin/env python3
"""
Trend Following Scanner 1H R1 - Enhanced with Database Logging
Extends the base trend_following_scanner.py with database integration
File: manual_scanners/1_hour_scanners/trend_following_scanner_1h_r1.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the base scanner (R0 version)
try:
    from trend_following_scanner import TrendFollowingScanner
    BASE_SCANNER_AVAILABLE = True
    print("✅ Base Trend Following Scanner imported successfully")
except ImportError as e:
    print(f"❌ Error importing base scanner: {e}")
    BASE_SCANNER_AVAILABLE = False

# Import database logger if available
try:
    from database_and_logging.trend_following_logger import TrendFollowingLogger
    DB_LOGGING_ENABLED = True
    print("✅ Database logging enabled for Trend Following scanner")
except ImportError as e:
    print(f"⚠️ Database logging not available: {e}")
    DB_LOGGING_ENABLED = False

class TrendFollowingScannerR1(TrendFollowingScanner):
    """
    Enhanced Trend Following Scanner with database integration
    Extends the base scanner without modifying existing functionality
    """
    
    def __init__(self, timeframe='1h'):
        # Initialize base scanner
        super().__init__()
        self.timeframe = timeframe
        self.scanner_name = f"trend_following_1h_r1"
        
        # Initialize database logger if available
        if DB_LOGGING_ENABLED:
            try:
                self.db_logger = TrendFollowingLogger(timeframe=timeframe)
                print(f"✅ Database logger initialized for {timeframe}")
            except Exception as e:
                print(f"❌ Failed to initialize database logger: {e}")
                self.db_logger = None
                DB_LOGGING_ENABLED = False
        else:
            self.db_logger = None
        
        # Enhanced configuration
        self.quality_threshold = 60  # Minimum quality score for database logging
        self.max_symbols_per_scan = 100  # Limit symbols to prevent overload
        
        print(f"🎯 Trend Following Scanner R1 initialized")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Database Logging: {'Enabled' if DB_LOGGING_ENABLED else 'Disabled'}")
        print(f"   Quality Threshold: {self.quality_threshold}")
    
    def scan_and_log_opportunities(self):
        """
        Enhanced scanning method with database logging
        Returns the same opportunities as base scanner plus logging status
        """
        print(f"\n🎯 TREND FOLLOWING SCANNER R1 - {self.timeframe.upper()}")
        print(f"📊 Enhanced scan with database logging")
        print(f"⏰ Scan time: {datetime.now()}")
        print("=" * 70)
        
        if not BASE_SCANNER_AVAILABLE:
            print("❌ Base scanner not available - cannot proceed")
            return {
                'opportunities': [],
                'total_found': 0,
                'total_logged': 0,
                'errors': ['Base scanner import failed']
            }
        
        # Run base scanner scan
        try:
            opportunities = self.scan_for_trend_setups()
            total_found = len(opportunities)
            total_logged = 0
            errors = []
            
            print(f"\n📈 Base scan complete: {total_found} opportunities found")
            
            # Log opportunities to database if enabled
            if DB_LOGGING_ENABLED and self.db_logger and opportunities:
                print(f"💾 Logging opportunities to database...")
                
                for i, opportunity in enumerate(opportunities, 1):
                    try:
                        # Check quality threshold
                        if opportunity.get('quality_score', 0) >= self.quality_threshold:
                            # Log to database
                            signal_id = self.db_logger.log_opportunity(
                                opportunity['symbol'], 
                                opportunity, 
                                primary_tf=self.timeframe
                            )
                            
                            if signal_id:
                                total_logged += 1
                                print(f"  ✅ {i:2}. {opportunity['symbol']:8} | "
                                      f"Score: {opportunity['quality_score']:2}/100 | "
                                      f"Logged: {signal_id}")
                            else:
                                errors.append(f"Failed to log {opportunity['symbol']}")
                                print(f"  ❌ {i:2}. {opportunity['symbol']:8} | "
                                      f"Score: {opportunity['quality_score']:2}/100 | "
                                      f"Logging failed")
                        else:
                            print(f"  ⚠️ {i:2}. {opportunity['symbol']:8} | "
                                  f"Score: {opportunity['quality_score']:2}/100 | "
                                  f"Below threshold")
                    
                    except Exception as e:
                        error_msg = f"Error processing {opportunity.get('symbol', 'Unknown')}: {e}"
                        errors.append(error_msg)
                        print(f"  ❌ {i:2}. {opportunity.get('symbol', 'Unknown'):8} | Error: {e}")
                        continue
                
                print(f"\n💾 Database logging complete:")
                print(f"   Total opportunities: {total_found}")
                print(f"   Successfully logged: {total_logged}")
                print(f"   Errors: {len(errors)}")
                
            else:
                print(f"💾 Database logging not available - opportunities saved locally only")
            
            # Display results summary
            self._display_enhanced_results(opportunities, total_logged, errors)
            
            return {
                'opportunities': opportunities,
                'total_found': total_found,
                'total_logged': total_logged,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Scanner execution failed: {e}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'opportunities': [],
                'total_found': 0,
                'total_logged': 0,
                'errors': [error_msg]
            }
    
    def _display_enhanced_results(self, opportunities, total_logged, errors):
        """Display enhanced results with database logging status"""
        print(f"\n{'=' * 85}")
        print(f"🎯 TREND FOLLOWING SCANNER R1 - SCAN RESULTS")
        print(f"{'=' * 85}")
        
        # Summary metrics
        print(f"📊 SCAN SUMMARY:")
        print(f"   Total opportunities found: {len(opportunities)}")
        print(f"   Successfully logged to DB: {total_logged}")
        print(f"   Database logging: {'Enabled' if DB_LOGGING_ENABLED else 'Disabled'}")
        print(f"   Quality threshold: {self.quality_threshold}")
        
        if errors:
            print(f"   Errors encountered: {len(errors)}")
        
        # Top opportunities
        if opportunities:
            print(f"\n🏆 TOP OPPORTUNITIES (by quality score):")
            top_opportunities = sorted(opportunities, key=lambda x: x.get('quality_score', 0), reverse=True)[:5]
            
            for i, opp in enumerate(top_opportunities, 1):
                symbol = opp.get('symbol', 'Unknown')
                score = opp.get('quality_score', 0)
                trend = opp.get('trend', {})
                trade = opp.get('trade', {})
                
                print(f"   {i}. {symbol:8} | Score: {score:2}/100 | "
                      f"Trend: {trend.get('direction', 'Unknown')} | "
                      f"Direction: {trade.get('direction', 'Unknown')}")
        
        # Database status
        if DB_LOGGING_ENABLED:
            print(f"\n💾 DATABASE STATUS:")
            print(f"   Logger: Active")
            print(f"   Schema: other_scanners.trend_following_signals")
            print(f"   Timeframe: {self.timeframe}")
            print(f"   Logged signals: {total_logged}")
        else:
            print(f"\n⚠️ DATABASE STATUS:")
            print(f"   Logger: Not available")
            print(f"   Opportunities saved locally only")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if total_logged > 0:
            print(f"   ✅ {total_logged} high-quality signals logged to database")
            print(f"   📊 Monitor database for signal performance tracking")
        else:
            print(f"   ⚠️ No signals met quality threshold ({self.quality_threshold})")
            print(f"   🔍 Consider adjusting quality parameters or wait for better setups")
        
        if errors:
            print(f"   ❌ {len(errors)} errors encountered - check logs for details")
    
    def get_scanner_status(self):
        """Get current scanner status for monitoring"""
        return {
            'scanner_name': self.scanner_name,
            'timeframe': self.timeframe,
            'base_scanner_available': BASE_SCANNER_AVAILABLE,
            'database_logging_enabled': DB_LOGGING_ENABLED,
            'logger_initialized': self.db_logger is not None,
            'quality_threshold': self.quality_threshold,
            'last_scan_time': getattr(self, '_last_scan_time', None)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.db_logger:
            try:
                self.db_logger.close()
                print("✅ Database logger closed")
            except Exception as e:
                print(f"⚠️ Error closing database logger: {e}")

def main():
    """Main execution function"""
    try:
        print("🚀 Starting Trend Following Scanner R1...")
        
        # Initialize scanner
        scanner = TrendFollowingScannerR1(timeframe='1h')
        
        # Display status
        status = scanner.get_scanner_status()
        print(f"\n📊 Scanner Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Run enhanced scan
        results = scanner.scan_and_log_opportunities()
        
        # Final summary
        print(f"\n🎯 SCAN COMPLETE!")
        print(f"   Opportunities found: {results['total_found']}")
        print(f"   Successfully logged: {results['total_logged']}")
        print(f"   Errors: {len(results['errors'])}")
        
        # Cleanup
        scanner.cleanup()
        
        return results
        
    except Exception as e:
        print(f"❌ Fatal error in main execution: {e}")
        traceback.print_exc()
        return {
            'opportunities': [],
            'total_found': 0,
            'total_logged': 0,
            'errors': [f"Fatal error: {e}"]
        }

if __name__ == "__main__":
    main()
