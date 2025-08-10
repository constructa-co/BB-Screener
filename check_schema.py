#!/usr/bin/env python3
"""
Check database schema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def check_schema():
    logger = TradeLogger()
    
    # Check scan_results columns
    logger.cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'scan_results' 
        ORDER BY ordinal_position
    """)
    
    cols = logger.cursor.fetchall()
    print("📊 Scan Results columns:")
    for col in cols:
        print(f"   • {col['column_name']}")
    
    # Check trade_opportunities columns
    logger.cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'trade_opportunities' 
        ORDER BY ordinal_position
    """)
    
    cols = logger.cursor.fetchall()
    print(f"\n📊 Trade Opportunities columns:")
    for col in cols:
        print(f"   • {col['column_name']}")
    
    logger.close()

if __name__ == "__main__":
    check_schema()
