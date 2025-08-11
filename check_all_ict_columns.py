#!/usr/bin/env python3
"""
Check All ICT Columns in Database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        logger = TradeLogger()
        
        # Check for all ICT-related columns
        logger.cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trade_opportunities' 
            AND (column_name LIKE '%fib%' 
                 OR column_name LIKE '%gap%' 
                 OR column_name LIKE '%swing%' 
                 OR column_name LIKE '%order%'
                 OR column_name LIKE '%liquidity%'
                 OR column_name LIKE '%imbalance%')
            ORDER BY column_name
        """)
        
        ict_cols = logger.cursor.fetchall()
        print("🔍 ICT-related columns in database:")
        for col in ict_cols:
            print(f"  ✅ {col['column_name']}")
        
        print(f"\n📊 Total ICT columns: {len(ict_cols)}")
        
        # Check total columns
        logger.cursor.execute("""
            SELECT COUNT(*) as count
            FROM information_schema.columns 
            WHERE table_name = 'trade_opportunities'
        """)
        total_cols = logger.cursor.fetchone()['count']
        print(f"📊 Total columns in trade_opportunities: {total_cols}")
        
        logger.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
