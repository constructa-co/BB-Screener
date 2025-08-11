#!/usr/bin/env python3
"""
Check existing columns in trade_opportunities table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        logger = TradeLogger()
        
        # Check for ICT columns
        logger.cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trade_opportunities' 
            AND column_name IN ('gap_high', 'gap_low', 'fib_618', 'swing_high', 'order_block_high')
        """)
        existing_cols = logger.cursor.fetchall()
        
        print("🔍 Checking existing ICT columns...")
        if existing_cols:
            print("✅ Found existing ICT columns:")
            for col in existing_cols:
                print(f"  - {col['column_name']}")
        else:
            print("❌ No ICT columns found - need to add them")
        
        # Check total columns
        logger.cursor.execute("""
            SELECT COUNT(*) as count
            FROM information_schema.columns 
            WHERE table_name = 'trade_opportunities'
        """)
        total_cols = logger.cursor.fetchone()['count']
        print(f"\n📊 Total columns in trade_opportunities: {total_cols}")
        
        logger.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
