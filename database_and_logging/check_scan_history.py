#!/usr/bin/env python3
"""
Check scan history to see what data we have
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def check_scan_history():
    logger = TradeLogger()
    
    # Check scan results
    logger.cursor.execute("""
        SELECT id, scan_type, scan_timestamp, total_coins_analyzed, premium_trades_found
        FROM scan_results 
        WHERE scan_type = 'bb_scanner' 
        ORDER BY id DESC 
        LIMIT 10
    """)
    
    scans = logger.cursor.fetchall()
    print("📊 BB Scanner Scan History:")
    for scan in scans:
        print(f"   • Scan {scan['id']}: {scan['total_coins_analyzed']} analyzed, {scan['premium_trades_found']} premium at {scan['scan_timestamp']}")
    
    # Check total trades by scan
    logger.cursor.execute("""
        SELECT scan_id, COUNT(*) as trade_count
        FROM trade_opportunities t
        JOIN scan_results s ON t.scan_id = s.id
        WHERE s.scan_type = 'bb_scanner'
        GROUP BY scan_id
        ORDER BY scan_id DESC
    """)
    
    trade_counts = logger.cursor.fetchall()
    print(f"\n📈 Trade Count by Scan:")
    for tc in trade_counts:
        print(f"   • Scan {tc['scan_id']}: {tc['trade_count']} trades")
    
    # Check total unique symbols
    logger.cursor.execute("""
        SELECT COUNT(DISTINCT symbol) as unique_symbols
        FROM trade_opportunities t
        JOIN scan_results s ON t.scan_id = s.id
        WHERE s.scan_type = 'bb_scanner'
    """)
    
    unique_symbols = logger.cursor.fetchone()['unique_symbols']
    print(f"\n🎯 Total unique symbols: {unique_symbols}")
    
    # Check probability distribution
    logger.cursor.execute("""
        SELECT probability, COUNT(*) as count
        FROM trade_opportunities t
        JOIN scan_results s ON t.scan_id = s.id
        WHERE s.scan_type = 'bb_scanner'
        GROUP BY probability
        ORDER BY probability
    """)
    
    prob_dist = logger.cursor.fetchall()
    print(f"\n📊 Probability Distribution:")
    for prob in prob_dist:
        print(f"   • {prob['probability']}%: {prob['count']} trades")
    
    logger.close()

if __name__ == "__main__":
    check_scan_history()
