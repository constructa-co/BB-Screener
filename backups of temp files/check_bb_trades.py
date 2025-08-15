#!/usr/bin/env python3
"""
Check Recent BB Scanner Trades
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Check recent BB scanner trades
        logger.cursor.execute("""
            SELECT t.symbol, t.probability, s.scan_type, s.scan_timestamp 
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h') 
            ORDER BY s.scan_timestamp DESC 
            LIMIT 10
        """)
        
        results = logger.cursor.fetchall()
        print(f"📊 Found {len(results)} recent BB scanner trades:")
        
        for row in results:
            print(f"  • {row['symbol']} - {row['probability']}% ({row['scan_type']}) - {row['scan_timestamp']}")
        
        # Check if any BB trades meet dashboard criteria (probability >= 70)
        logger.cursor.execute("""
            SELECT COUNT(*) as count
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h') 
            AND t.probability >= 70
            AND t.trade_taken = FALSE
        """)
        
        count_result = logger.cursor.fetchone()
        print(f"\n🎯 BB trades with probability >= 70%: {count_result['count']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    logger.close()

if __name__ == "__main__":
    main()
