#!/usr/bin/env python3
"""
Final Diagnostic - Find exact issue with BB scanner in dashboard
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
        print("🔍 Final Diagnostic - BB Scanner Dashboard Issue:")
        
        # Check BB scanner trades for NULL risk_reward_ratio
        logger.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN t.risk_reward_ratio IS NULL THEN 1 END) as null_rr,
                COUNT(CASE WHEN t.risk_reward_ratio IS NOT NULL THEN 1 END) as not_null_rr
            FROM trade_opportunities t 
            JOIN scan_results s ON t.scan_id = s.id 
            WHERE s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            AND t.probability >= 70
            AND t.trade_taken = FALSE
        """)
        
        rr_check = logger.cursor.fetchone()
        print(f"📊 BB trades >= 70% (available): {rr_check['total']}")
        print(f"  • NULL risk_reward_ratio: {rr_check['null_rr']}")
        print(f"  • NOT NULL risk_reward_ratio: {rr_check['not_null_rr']}")
        
        # Test query without ORDER BY
        logger.cursor.execute("""
            SELECT 
                t.symbol, t.probability, t.risk_reward_ratio, s.scan_type
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.probability >= 70
            AND t.trade_taken = FALSE
            AND s.scan_type != 'BB_Backtest_R10'
            AND s.scan_type IN ('bb_scanner', 'bb_scanner_4h')
            LIMIT 10
        """)
        
        bb_results = logger.cursor.fetchall()
        print(f"\n🔍 BB Scanner Trades (no ORDER BY): {len(bb_results)} results")
        
        for trade in bb_results:
            rr_status = "NULL" if trade['risk_reward_ratio'] is None else f"{trade['risk_reward_ratio']:.2f}"
            print(f"  • {trade['symbol']} - {trade['probability']}% - RR: {rr_status}")
        
        # Test with ORDER BY probability only
        logger.cursor.execute("""
            SELECT 
                t.symbol, t.probability, t.risk_reward_ratio, s.scan_type
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.probability >= 70
            AND t.trade_taken = FALSE
            AND s.scan_type != 'BB_Backtest_R10'
            ORDER BY t.probability DESC
            LIMIT 20
        """)
        
        all_results = logger.cursor.fetchall()
        print(f"\n🔍 All Scanner Trades (ORDER BY probability): {len(all_results)} results")
        
        bb_count = 0
        ict_count = 0
        for trade in all_results:
            if 'bb_scanner' in trade['scan_type']:
                bb_count += 1
                prefix = "📊 BB"
            elif 'ict_scanner' in trade['scan_type']:
                ict_count += 1
                prefix = "📈 ICT"
            else:
                prefix = "📉 Other"
            
            rr_status = "NULL" if trade['risk_reward_ratio'] is None else f"{trade['risk_reward_ratio']:.2f}"
            print(f"  {prefix} • {trade['symbol']} - {trade['probability']}% - RR: {rr_status}")
        
        print(f"\n📈 Summary:")
        print(f"  • BB Scanner: {bb_count}")
        print(f"  • ICT Scanner: {ict_count}")
        
        # Test with original ORDER BY
        logger.cursor.execute("""
            SELECT 
                t.symbol, t.probability, t.risk_reward_ratio, s.scan_type
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE t.probability >= 70
            AND t.trade_taken = FALSE
            AND s.scan_type != 'BB_Backtest_R10'
            ORDER BY t.probability DESC, t.risk_reward_ratio DESC
            LIMIT 20
        """)
        
        ordered_results = logger.cursor.fetchall()
        print(f"\n🔍 All Scanner Trades (original ORDER BY): {len(ordered_results)} results")
        
        bb_count_ordered = 0
        ict_count_ordered = 0
        for trade in ordered_results:
            if 'bb_scanner' in trade['scan_type']:
                bb_count_ordered += 1
                prefix = "📊 BB"
            elif 'ict_scanner' in trade['scan_type']:
                ict_count_ordered += 1
                prefix = "📈 ICT"
            else:
                prefix = "📉 Other"
            
            rr_status = "NULL" if trade['risk_reward_ratio'] is None else f"{trade['risk_reward_ratio']:.2f}"
            print(f"  {prefix} • {trade['symbol']} - {trade['probability']}% - RR: {rr_status}")
        
        print(f"\n📈 Summary (with ORDER BY):")
        print(f"  • BB Scanner: {bb_count_ordered}")
        print(f"  • ICT Scanner: {ict_count_ordered}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    logger.close()

if __name__ == "__main__":
    main()
