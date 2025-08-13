#!/usr/bin/env python3
"""
Standalone module to sync Excel output to database
Reads the generated Excel file and pushes ALL data to database
Safe - doesn't touch any working code!
"""

import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_logger import TradeLogger

class ExcelToDatabaseSync:
    def __init__(self):
        self.logger = TradeLogger()
        
    def find_latest_excel(self, scanner_type='bb'):
        """Find the most recent Excel file from the scanner"""
        excel_dir = Path('outputs/excel_reports')
        
        # Pattern for BB scanner files
        pattern = f"{scanner_type}_analysis_*.xlsx"
        
        excel_files = list(excel_dir.glob(pattern))
        if not excel_files:
            print(f"No Excel files found matching {pattern}")
            return None
            
        # Get most recent file
        latest_file = max(excel_files, key=os.path.getctime)
        print(f"Found latest Excel: {latest_file}")
        return latest_file
        
    def sync_excel_to_database(self, excel_path=None):
        """Read Excel and sync ALL data to database"""
        
        # Find Excel file if not provided
        if excel_path is None:
            excel_path = self.find_latest_excel()
            if excel_path is None:
                return False
        
        try:
            # Read ALL sheets from Excel
            excel_file = pd.ExcelFile(excel_path)
            print(f"Excel sheets found: {excel_file.sheet_names}")
            
            # Read main analysis sheet (usually first sheet)
            df = pd.read_excel(excel_path, sheet_name=0)
            print(f"Found {len(df)} trades in Excel")
            
            # Filter for actual trades (probability > 0)
            trades_df = df[df['probability'] > 0].copy() if 'probability' in df.columns else df
            print(f"Found {len(trades_df)} valid trades to sync")
            
            if len(trades_df) == 0:
                print("No valid trades to sync")
                return False
            
            # Start database logging
            if not self.logger.connection:
                print("❌ Database connection failed")
                return False
                
            scan_id = self.logger.log_scan_start('bb_scanner', version='1.1')
            print(f"Started database sync with scan_id: {scan_id}")
            
            # Log each trade with ALL columns
            success_count = 0
            for idx, row in trades_df.iterrows():
                try:
                    # Convert row to dict
                    trade_data = row.to_dict()
                    
                    # Clean NaN values
                    for key, value in trade_data.items():
                        if pd.isna(value):
                            trade_data[key] = None
                    
                    # Define main database columns
                    main_columns = [
                        'symbol', 'exchange', 'probability', 
                        'entry_price', 'stop_loss', 'target_1'
                    ]
                    
                    # Everything else goes in scanner_specific_data
                    scanner_specific = {}
                    for key, value in trade_data.items():
                        if key not in main_columns:
                            # Convert to string for JSON serialization
                            if value is not None:
                                if isinstance(value, (int, float)):
                                    scanner_specific[key] = value
                                else:
                                    scanner_specific[key] = str(value)
                    
                    # Create database record
                    record = {
                        'symbol': str(trade_data.get('symbol', '')),
                        'exchange': str(trade_data.get('exchange', '')),
                        'probability': float(trade_data.get('probability', 0)),
                        'entry_price': float(trade_data.get('entry_price', 0)) if trade_data.get('entry_price') else 0,
                        'stop_loss': float(trade_data.get('stop_loss', 0)) if trade_data.get('stop_loss') else 0,
                        'target_1': float(trade_data.get('target_1', 0)) if trade_data.get('target_1') else 0,
                        'scanner_specific_data': json.dumps(scanner_specific)
                    }
                    
                    # Log to database
                    self.logger.log_trade_opportunity(scan_id, record)
                    success_count += 1
                    
                    # Show progress
                    if success_count == 1:
                        print(f"First trade logged: {record['symbol']}")
                        print(f"Number of extra fields in scanner_specific_data: {len(scanner_specific)}")
                        print(f"Sample fields: {list(scanner_specific.keys())[:10]}")
                        
                except Exception as e:
                    print(f"Error logging trade {idx}: {e}")
                    continue
            
            # Complete the scan
            high_prob_count = len(trades_df[trades_df['probability'] > 70]) if 'probability' in trades_df.columns else 0
            self.logger.complete_scan(scan_id, success_count, high_prob_count, 0)
            
            print(f"✅ Successfully synced {success_count}/{len(trades_df)} trades to database")
            print(f"   High probability trades: {high_prob_count}")
            
            # Also sync other sheets if they contain data
            if 'Market_Regime_Analysis' in excel_file.sheet_names:
                self._sync_market_regime(excel_path)
                
            self.logger.close()
            return True
            
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _sync_market_regime(self, excel_path):
        """Optionally sync market regime data"""
        try:
            regime_df = pd.read_excel(excel_path, sheet_name='Market_Regime_Analysis')
            print(f"Also found Market Regime data to sync")
            # Add logic here if you want to store market regime data
        except:
            pass

    def run_after_scan(self):
        """Run this after BB scanner completes"""
        print("="*60)
        print("EXCEL TO DATABASE SYNC")
        print("="*60)
        
        success = self.sync_excel_to_database()
        
        if success:
            print("✅ Database sync completed successfully!")
        else:
            print("❌ Database sync failed - check logs")
        
        return success


if __name__ == "__main__":
    # Run the sync
    syncer = ExcelToDatabaseSync()
    syncer.run_after_scan()
