#!/usr/bin/env python3
"""
Simple database verification
"""

from trade_logger import TradeLogger

def verify():
    logger = TradeLogger()
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Check BB scanner trades
        logger.cursor.execute("""
            SELECT COUNT(*) FROM trade_opportunities 
            WHERE scan_id IN (
                SELECT id FROM scan_results WHERE scan_type = 'bb_scanner'
            )
        """)
        
        count = logger.cursor.fetchone()['count']
        print(f"📊 Total BB scanner trades in database: {count}")
        
        # Check recent trades
        logger.cursor.execute("""
            SELECT symbol, probability, exchange 
            FROM trade_opportunities 
            WHERE scan_id IN (
                SELECT id FROM scan_results WHERE scan_type = 'bb_scanner'
            )
            ORDER BY id DESC 
            LIMIT 5
        """)
        
        trades = logger.cursor.fetchall()
        print("🎯 Recent BB scanner trades:")
        for trade in trades:
            print(f"   • {trade['symbol']} ({trade['exchange']}): {trade['probability']}% probability")
        
        # Check scanner_specific_data
        logger.cursor.execute("""
            SELECT symbol, scanner_specific_data 
            FROM trade_opportunities 
            WHERE scan_id IN (
                SELECT id FROM scan_results WHERE scan_type = 'bb_scanner'
            )
            AND scanner_specific_data IS NOT NULL
            LIMIT 1
        """)
        
        sample = logger.cursor.fetchone()
        if sample and sample['scanner_specific_data']:
            import json
            try:
                data = json.loads(sample['scanner_specific_data'])
                print(f"\n📊 Sample scanner_specific_data for {sample['symbol']}:")
                print(f"   • Number of fields: {len(data)}")
                print(f"   • Sample fields: {list(data.keys())[:10]}")
            except:
                print(f"   • Data type: {type(sample['scanner_specific_data'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()

if __name__ == "__main__":
    verify()
