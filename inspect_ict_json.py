#!/usr/bin/env python3
"""
Inspect ICT Scanner JSON Data Structure
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        print("🔍 Inspecting ICT Scanner JSON Data Structure...")
        logger = TradeLogger()
        
        # Get a few ICT trades to see the actual data structure
        logger.cursor.execute("""
            SELECT t.id, t.symbol, t.scanner_specific_data
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type LIKE 'ict%'
            AND t.scanner_specific_data IS NOT NULL
            AND t.scanner_specific_data != '{}'
            AND t.scanner_specific_data != 'null'
            LIMIT 3
        """)
        
        ict_trades = logger.cursor.fetchall()
        print(f"📊 Inspecting {len(ict_trades)} ICT trades\n")
        
        for i, trade in enumerate(ict_trades, 1):
            print(f"Trade {i}: {trade['symbol']}")
            print(f"  ID: {trade['id']}")
            
            if trade['scanner_specific_data']:
                try:
                    if isinstance(trade['scanner_specific_data'], str):
                        data = json.loads(trade['scanner_specific_data'])
                    else:
                        data = trade['scanner_specific_data']
                    
                    print(f"  JSON Keys ({len(data)} fields):")
                    for key, value in list(data.items())[:10]:  # Show first 10
                        print(f"    {key}: {value}")
                    
                    if len(data) > 10:
                        print(f"    ... and {len(data) - 10} more fields")
                    
                    # Look for ICT-related fields
                    ict_related = []
                    for key in data.keys():
                        if any(term in key.lower() for term in ['gap', 'fib', 'swing', 'order', 'liquidity', 'imbalance']):
                            ict_related.append(key)
                    
                    if ict_related:
                        print(f"  🎯 ICT-related fields found: {ict_related}")
                    else:
                        print(f"  ❌ No obvious ICT fields found")
                        
                except Exception as e:
                    print(f"  ❌ Error parsing JSON: {e}")
            else:
                print(f"  📝 No scanner_specific_data")
            print()
        
        # Check total field count
        logger.cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN scanner_specific_data IS NOT NULL AND scanner_specific_data != '{}' THEN 1 END) as with_json,
                COUNT(CASE WHEN scanner_specific_data IS NULL OR scanner_specific_data = '{}' THEN 1 END) as without_json
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type LIKE 'ict%'
        """)
        
        stats = logger.cursor.fetchone()
        print(f"📈 ICT Scanner Statistics:")
        print(f"  Total ICT trades: {stats['total_trades']}")
        print(f"  With JSON data: {stats['with_json']}")
        print(f"  Without JSON data: {stats['without_json']}")
        
        logger.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
