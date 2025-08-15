#!/usr/bin/env python3
"""
Test Dashboard Query - Exact same query as dashboard
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
        # Test the exact dashboard query
        print("🔍 Testing Dashboard Query (get_best_opportunities):")
        
        # Query 1: Without time filter (like dashboard default)
        logger.cursor.execute("""
            SELECT 
                t.*,
                s.scan_type,
                s.scan_timestamp,
                'Day Trading' as trading_style,
                '4H' as timeframe
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.probability >= 70
            AND t.trade_taken = FALSE
            AND s.scan_type != 'BB_Backtest_R10'
            ORDER BY t.probability DESC, t.risk_reward_ratio DESC
            LIMIT 50
        """)
        
        results1 = logger.cursor.fetchall()
        print(f"📊 Query 1 (no time filter): {len(results1)} results")
        
        bb_count1 = sum(1 for r in results1 if 'bb_scanner' in r['scan_type'])
        ict_count1 = sum(1 for r in results1 if 'ict_scanner' in r['scan_type'])
        print(f"  • BB Scanner: {bb_count1}")
        print(f"  • ICT Scanner: {ict_count1}")
        
        # Query 2: With 24-hour time filter
        logger.cursor.execute("""
            SELECT 
                t.*,
                s.scan_type,
                s.scan_timestamp,
                'Day Trading' as trading_style,
                '4H' as timeframe
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.probability >= 70
            AND t.trade_taken = FALSE
            AND s.scan_type != 'BB_Backtest_R10'
            AND s.scan_timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY t.probability DESC, t.risk_reward_ratio DESC
            LIMIT 50
        """)
        
        results2 = logger.cursor.fetchall()
        print(f"📊 Query 2 (24h time filter): {len(results2)} results")
        
        bb_count2 = sum(1 for r in results2 if 'bb_scanner' in r['scan_type'])
        ict_count2 = sum(1 for r in results2 if 'ict_scanner' in r['scan_type'])
        print(f"  • BB Scanner: {bb_count2}")
        print(f"  • ICT Scanner: {ict_count2}")
        
        # Show recent BB scanner timestamps
        print(f"\n🔍 Recent BB Scanner Timestamps:")
        logger.cursor.execute("""
            SELECT scan_timestamp, scan_type
            FROM scan_results 
            WHERE scan_type IN ('bb_scanner', 'bb_scanner_4h')
            ORDER BY scan_timestamp DESC 
            LIMIT 5
        """)
        
        timestamps = logger.cursor.fetchall()
        for ts in timestamps:
            hours_ago = (logger.cursor.fetchone() or {}).get('hours_ago', 'N/A')
            print(f"  • {ts['scan_timestamp']} ({ts['scan_type']})")
        
        # Check if BB scanner trades are recent enough
        print(f"\n🔍 BB Scanner Recent Trades Check:")
        logger.cursor.execute("""
            SELECT COUNT(*) as recent_count
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
            AND t.trade_taken = FALSE
            AND s.scan_timestamp > NOW() - INTERVAL '24 hours'
        """)
        
        recent_count = logger.cursor.fetchone()['recent_count']
        print(f"  • BB trades in last 24h: {recent_count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    logger.close()

if __name__ == "__main__":
    main()
