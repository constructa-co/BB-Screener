#!/usr/bin/env python3
"""
Debug BB Scanner Filter - Why BB trades not showing in dashboard
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
        # Check BB scanner trade status
        print("🔍 BB Scanner Trade Status Analysis:")
        
        # Total BB trades with probability >= 70
        logger.cursor.execute("""
            SELECT COUNT(*) as total
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
        """)
        total = logger.cursor.fetchone()['total']
        
        # BB trades that are taken
        logger.cursor.execute("""
            SELECT COUNT(*) as taken
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
            AND t.trade_taken = TRUE
        """)
        taken = logger.cursor.fetchone()['taken']
        
        # BB trades that are available
        logger.cursor.execute("""
            SELECT COUNT(*) as available
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
            AND t.trade_taken = FALSE
        """)
        available = logger.cursor.fetchone()['available']
        
        print(f"📊 BB trades >= 70%: {total} total")
        print(f"  • Taken: {taken}")
        print(f"  • Available: {available}")
        
        # Check if there are any BB trades with NULL trade_taken
        logger.cursor.execute("""
            SELECT COUNT(*) as null_taken
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
            AND t.trade_taken IS NULL
        """)
        null_taken = logger.cursor.fetchone()['null_taken']
        print(f"  • NULL trade_taken: {null_taken}")
        
        # Show some sample BB trades
        print(f"\n🔍 Sample BB Scanner Trades:")
        logger.cursor.execute("""
            SELECT t.symbol, t.probability, t.trade_taken, s.scan_type, s.scan_timestamp
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
            ORDER BY s.scan_timestamp DESC 
            LIMIT 5
        """)
        
        samples = logger.cursor.fetchall()
        for sample in samples:
            taken_status = "✅ Taken" if sample['trade_taken'] else "❌ Available" if sample['trade_taken'] is False else "❓ NULL"
            print(f"  • {sample['symbol']} - {sample['probability']}% - {taken_status} ({sample['scan_type']})")
        
        # Check if BB scanner is logging trade_taken correctly
        print(f"\n🔍 BB Scanner Trade Logging Check:")
        logger.cursor.execute("""
            SELECT 
                COUNT(*) as total_bb_trades,
                COUNT(CASE WHEN t.trade_taken IS NOT NULL THEN 1 END) as with_trade_taken,
                COUNT(CASE WHEN t.trade_taken IS NULL THEN 1 END) as without_trade_taken
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
        """)
        
        logging_check = logger.cursor.fetchone()
        print(f"  • Total BB trades: {logging_check['total_bb_trades']}")
        print(f"  • With trade_taken: {logging_check['with_trade_taken']}")
        print(f"  • Without trade_taken: {logging_check['without_trade_taken']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    logger.close()

if __name__ == "__main__":
    main()
