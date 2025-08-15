#!/usr/bin/env python3
"""
Check what data is being captured in the database
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def check_data_capture():
    logger = TradeLogger()
    
    # Check latest scan data
    logger.cursor.execute("""
        SELECT symbol, scanner_specific_data 
        FROM trade_opportunities t 
        JOIN scan_results s ON t.scan_id = s.id 
        WHERE s.scan_type = 'bb_scanner' AND s.id = 211 
        LIMIT 1
    """)
    
    result = logger.cursor.fetchone()
    if result and result['scanner_specific_data']:
        # Handle JSONB data (might already be dict)
        if isinstance(result['scanner_specific_data'], str):
            data = json.loads(result['scanner_specific_data'])
        else:
            data = result['scanner_specific_data']
        print(f"📊 Symbol: {result['symbol']}")
        print(f"📊 Fields captured: {len(data)}")
        print(f"📊 Sample fields (first 20):")
        for i, key in enumerate(list(data.keys())[:20]):
            print(f"   • {key}: {data[key]}")
        print(f"📊 Sample fields (last 20):")
        for i, key in enumerate(list(data.keys())[-20:]):
            print(f"   • {key}: {data[key]}")
    
    # Check if we have entry/exit data
    logger.cursor.execute("""
        SELECT symbol, entry_price, stop_loss, target_1, target_2, target_3
        FROM trade_opportunities t 
        JOIN scan_results s ON t.scan_id = s.id 
        WHERE s.scan_type = 'bb_scanner' AND s.id = 211 
        LIMIT 3
    """)
    
    trades = logger.cursor.fetchall()
    print(f"\n🎯 Entry/Exit Data Sample:")
    for trade in trades:
        print(f"   • {trade['symbol']}: Entry={trade['entry_price']}, SL={trade['stop_loss']}, T1={trade['target_1']}, T2={trade['target_2']}, T3={trade['target_3']}")
    
    logger.close()

if __name__ == "__main__":
    check_data_capture()
