#!/usr/bin/env python3
"""
Check Dashboard Data - What the dashboard should be showing
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
        # Check what the dashboard query should return
        print("🔍 Dashboard Query Results (probability >= 70, trade_taken = FALSE):")
        logger.cursor.execute("""
            SELECT t.symbol, t.probability, s.scan_type, s.scan_timestamp 
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE t.probability >= 70 
            AND t.trade_taken = FALSE 
            AND s.scan_type != 'BB_Backtest_R10'
            ORDER BY t.probability DESC 
            LIMIT 15
        """)
        
        results = logger.cursor.fetchall()
        print(f"📊 Found {len(results)} high-probability trades:")
        
        bb_count = 0
        ict_count = 0
        other_count = 0
        
        for row in results:
            scan_type = row['scan_type']
            if 'bb_scanner' in scan_type:
                bb_count += 1
                prefix = "📊 BB"
            elif 'ict_scanner' in scan_type:
                ict_count += 1
                prefix = "📈 ICT"
            else:
                other_count += 1
                prefix = "📉 Other"
            
            print(f"  {prefix} • {row['symbol']} - {row['probability']}% ({scan_type}) - {row['scan_timestamp']}")
        
        print(f"\n📈 Summary:")
        print(f"  • BB Scanner trades: {bb_count}")
        print(f"  • ICT Scanner trades: {ict_count}")
        print(f"  • Other scanner trades: {other_count}")
        
        # Check recent BB scanner activity
        print(f"\n🔍 Recent BB Scanner Activity:")
        logger.cursor.execute("""
            SELECT COUNT(*) as count, MAX(scan_timestamp) as last_scan
            FROM scan_results 
            WHERE scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND scan_timestamp > NOW() - INTERVAL '24 hours'
        """)
        
        bb_activity = logger.cursor.fetchone()
        print(f"  • BB scans in last 24h: {bb_activity['count']}")
        print(f"  • Last BB scan: {bb_activity['last_scan']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    logger.close()

if __name__ == "__main__":
    main()
