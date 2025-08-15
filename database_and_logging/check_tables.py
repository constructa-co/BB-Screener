#!/usr/bin/env python3
"""
Check what market tables exist
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def check_tables():
    logger = TradeLogger()
    
    # Check market tables
    logger.cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE 'market_%'
        ORDER BY table_name
    """)
    
    tables = logger.cursor.fetchall()
    print("📊 Market Tables Found:")
    for table in tables:
        print(f"   • {table['table_name']}")
    
    logger.close()

if __name__ == "__main__":
    check_tables()
