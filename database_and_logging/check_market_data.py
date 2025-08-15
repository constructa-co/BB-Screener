#!/usr/bin/env python3
"""
Check market data logging
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def check_market_data():
    logger = TradeLogger()
    
    # Check if market_data table exists
    logger.cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'market_data'
        )
    """)
    
    table_exists = logger.cursor.fetchone()['exists']
    print(f"📊 Market data table exists: {table_exists}")
    
    if table_exists:
        # Check market data counts by type
        logger.cursor.execute("""
            SELECT data_type, COUNT(*) as count
            FROM market_data
            GROUP BY data_type
            ORDER BY data_type
        """)
        
        counts = logger.cursor.fetchall()
        print(f"\n📈 Market Data Counts:")
        for count in counts:
            print(f"   • {count['data_type']}: {count['count']} records")
        
        # Check latest market data
        logger.cursor.execute("""
            SELECT data_type, timestamp, market_data
            FROM market_data
            ORDER BY timestamp DESC
            LIMIT 3
        """)
        
        latest = logger.cursor.fetchall()
        print(f"\n🔍 Latest Market Data:")
        for record in latest:
            # Handle JSONB data (might already be dict)
            if isinstance(record['market_data'], str):
                data = json.loads(record['market_data'])
            else:
                data = record['market_data']
            print(f"   • {record['data_type']} at {record['timestamp']}: {len(data)} fields")
            print(f"     Sample fields: {list(data.keys())[:5]}")
    
    logger.close()

if __name__ == "__main__":
    check_market_data()
