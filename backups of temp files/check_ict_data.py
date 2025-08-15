#!/usr/bin/env python3
"""
Check ICT Scanner Data - See what's currently being stored
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        logger = TradeLogger()
        
        # Get recent ICT trades
        logger.cursor.execute("""
            SELECT t.symbol, t.scanner_specific_data, t.gap_high, t.gap_low, t.fib_618, t.swing_high
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type LIKE 'ict%'
            ORDER BY t.id DESC
            LIMIT 5
        """)
        
        trades = logger.cursor.fetchall()
        print(f"📊 Found {len(trades)} recent ICT trades\n")
        
        if not trades:
            print("❌ No ICT trades found in database")
            print("   Check if ICT scanner is running and logging to database")
            return
        
        for i, trade in enumerate(trades, 1):
            print(f"Trade {i}: {trade['symbol']}")
            print(f"  Gap High (column): {trade.get('gap_high', 'NOT SET')}")
            print(f"  Gap Low (column): {trade.get('gap_low', 'NOT SET')}")
            print(f"  Fib 618 (column): {trade.get('fib_618', 'NOT SET')}")
            print(f"  Swing High (column): {trade.get('swing_high', 'NOT SET')}")
            
            if trade['scanner_specific_data']:
                try:
                    if isinstance(trade['scanner_specific_data'], str):
                        data = json.loads(trade['scanner_specific_data'])
                    else:
                        data = trade['scanner_specific_data']
                    
                    # Check for ICT fields in JSON
                    ict_fields_in_json = []
                    for field in ['gap_high', 'gap_low', 'fib_618', 'swing_high', 'order_block_high', 'liquidity_sweep']:
                        if field in data:
                            ict_fields_in_json.append(f"{field}: {data[field]}")
                    
                    if ict_fields_in_json:
                        print(f"  ⚠️  ICT data found in JSON instead of columns!")
                        for field_info in ict_fields_in_json[:3]:  # Show first 3
                            print(f"     {field_info}")
                        if len(ict_fields_in_json) > 3:
                            print(f"     ... and {len(ict_fields_in_json) - 3} more fields")
                    else:
                        print(f"  ✅ No ICT fields found in JSON (good!)")
                        
                except Exception as e:
                    print(f"  ❌ Error parsing JSON: {e}")
            else:
                print(f"  📝 No scanner_specific_data")
            print()
        
        # Check total ICT trades
        logger.cursor.execute("""
            SELECT COUNT(*) as count
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type LIKE 'ict%'
        """)
        total_ict = logger.cursor.fetchone()['count']
        print(f"📈 Total ICT trades in database: {total_ict}")
        
        logger.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
