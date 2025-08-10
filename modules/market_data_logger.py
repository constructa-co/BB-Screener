#!/usr/bin/env python3
"""
Market Data Logger - Logs market-level data from secondary sheets
Separate from trade data for ML cross-referencing
"""

import pandas as pd
import json
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trade_logger import TradeLogger

class MarketDataLogger:
    def __init__(self):
        self.logger = TradeLogger()
        
    def log_market_regime(self, scan_id, regime_df):
        """Log market regime analysis data"""
        try:
            for idx, row in regime_df.iterrows():
                regime_data = row.to_dict()
                
                # Clean NaN values
                for key, value in regime_data.items():
                    if pd.isna(value):
                        regime_data[key] = None
                
                # Create market regime record
                record = {
                    'scan_id': scan_id,
                    'data_type': 'market_regime',
                    'timestamp': datetime.now(),
                    'market_data': json.dumps(regime_data)
                }
                
                self.logger.cursor.execute("""
                    INSERT INTO market_data (scan_id, data_type, timestamp, market_data)
                    VALUES (%(scan_id)s, %(data_type)s, %(timestamp)s, %(market_data)s)
                """, record)
                
            print(f"✅ Logged {len(regime_df)} market regime records")
            
        except Exception as e:
            print(f"❌ Error logging market regime: {e}")
    
    def log_market_overview(self, scan_id, overview_df):
        """Log market overview data"""
        try:
            for idx, row in overview_df.iterrows():
                overview_data = row.to_dict()
                
                # Clean NaN values
                for key, value in overview_data.items():
                    if pd.isna(value):
                        overview_data[key] = None
                
                # Create market overview record
                record = {
                    'scan_id': scan_id,
                    'data_type': 'market_overview',
                    'timestamp': datetime.now(),
                    'market_data': json.dumps(overview_data)
                }
                
                self.logger.cursor.execute("""
                    INSERT INTO market_data (scan_id, data_type, timestamp, market_data)
                    VALUES (%(scan_id)s, %(data_type)s, %(timestamp)s, %(market_data)s)
                """, record)
                
            print(f"✅ Logged {len(overview_df)} market overview records")
            
        except Exception as e:
            print(f"❌ Error logging market overview: {e}")
    
    def log_market_metadata(self, scan_id, metadata_df):
        """Log market metadata"""
        try:
            for idx, row in metadata_df.iterrows():
                metadata = row.to_dict()
                
                # Clean NaN values
                for key, value in metadata.items():
                    if pd.isna(value):
                        metadata[key] = None
                
                # Create market metadata record
                record = {
                    'scan_id': scan_id,
                    'data_type': 'market_metadata',
                    'timestamp': datetime.now(),
                    'market_data': json.dumps(metadata)
                }
                
                self.logger.cursor.execute("""
                    INSERT INTO market_data (scan_id, data_type, timestamp, market_data)
                    VALUES (%(scan_id)s, %(data_type)s, %(timestamp)s, %(market_data)s)
                """, record)
                
            print(f"✅ Logged {len(metadata_df)} market metadata records")
            
        except Exception as e:
            print(f"❌ Error logging market metadata: {e}")
    
    def create_market_data_table(self):
        """Create market_data table if it doesn't exist"""
        try:
            self.logger.cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    scan_id INTEGER REFERENCES scan_results(id),
                    data_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    market_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.logger.connection.commit()
            print("✅ Market data table ready")
            
        except Exception as e:
            print(f"❌ Error creating market data table: {e}")
    
    def close(self):
        """Close database connection"""
        if self.logger.connection:
            self.logger.connection.commit()
            self.logger.close()
