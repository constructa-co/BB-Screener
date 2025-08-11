#!/usr/bin/env python3
"""
Test Balanced Dashboard Query
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
        print("🔍 Testing Balanced Dashboard Query:")
        
        # Test the balanced query
        logger.cursor.execute("""
            (
                -- Get top BB scanner trades
                SELECT 
                    t.symbol, t.probability, t.risk_reward_ratio, s.scan_type, s.scan_timestamp
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE t.probability >= 70
                AND t.trade_taken = FALSE
                AND s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
                ORDER BY t.probability DESC, s.scan_timestamp DESC
                LIMIT 10
            )
            UNION ALL
            (
                -- Get top ICT scanner trades
                SELECT 
                    t.symbol, t.probability, t.risk_reward_ratio, s.scan_type, s.scan_timestamp
                FROM trade_opportunities t
                JOIN scan_results s ON t.scan_id = s.id
                WHERE t.probability >= 70
                AND t.trade_taken = FALSE
                AND s.scan_type LIKE '%ict%'
                ORDER BY t.probability DESC, s.scan_timestamp DESC
                LIMIT 10
            )
            ORDER BY probability DESC, scan_timestamp DESC
            LIMIT 20
        """)
        
        results = logger.cursor.fetchall()
        print(f"📊 Balanced Query Results: {len(results)} trades")
        
        bb_count = 0
        ict_count = 0
        other_count = 0
        
        for trade in results:
            scan_type = trade['scan_type']
            if 'bb_scanner' in scan_type:
                bb_count += 1
                prefix = "📊 BB"
            elif 'ict_scanner' in scan_type:
                ict_count += 1
                prefix = "📈 ICT"
            else:
                other_count += 1
                prefix = "📉 Other"
            
            rr_status = "NULL" if trade['risk_reward_ratio'] is None else f"{trade['risk_reward_ratio']:.2f}"
            print(f"  {prefix} • {trade['symbol']} - {trade['probability']}% - RR: {rr_status} - {trade['scan_timestamp']}")
        
        print(f"\n📈 Summary:")
        print(f"  • BB Scanner: {bb_count}")
        print(f"  • ICT Scanner: {ict_count}")
        print(f"  • Other Scanner: {other_count}")
        
        if bb_count > 0:
            print(f"\n✅ SUCCESS! BB Scanner trades are now showing in the dashboard!")
            print(f"   Balanced mix: {bb_count} BB + {ict_count} ICT = {bb_count + ict_count} total")
        else:
            print(f"\n❌ BB Scanner trades still not showing. Need further investigation.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    logger.close()

if __name__ == "__main__":
    main()
