#!/usr/bin/env python3
"""
Verify database sync worked
"""

from trade_logger import TradeLogger

def verify_sync():
    logger = TradeLogger()
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Check recent BB scanner scans
        logger.cursor.execute("""
            SELECT id, scan_type, created_at, total_coins, premium_trades 
            FROM scan_results 
            WHERE scan_type = 'bb_scanner' 
            ORDER BY created_at DESC 
            LIMIT 5
        """)
        
        scans = logger.cursor.fetchall()
        print("📊 Recent BB Scanner Scans:")
        for scan in scans:
            print(f"   • Scan {scan[0]}: {scan[2]} - {scan[3]} coins, {scan[4]} trades")
        
        # Check trades for the most recent scan
        if scans:
            latest_scan_id = scans[0][0]
            logger.cursor.execute("""
                SELECT COUNT(*) FROM trade_opportunities 
                WHERE scan_id = %s
            """, (latest_scan_id,))
            
            trade_count = logger.cursor.fetchone()[0]
            print(f"\n📊 Trades in latest scan ({latest_scan_id}): {trade_count}")
            
            # Show sample trades
            logger.cursor.execute("""
                SELECT symbol, probability, exchange 
                FROM trade_opportunities 
                WHERE scan_id = %s 
                ORDER BY probability DESC 
                LIMIT 5
            """, (latest_scan_id,))
            
            trades = logger.cursor.fetchall()
            print("🎯 Sample trades:")
            for trade in trades:
                print(f"   • {trade[0]} ({trade[2]}): {trade[1]}% probability")
        
        # Check scanner_specific_data
        logger.cursor.execute("""
            SELECT symbol, scanner_specific_data 
            FROM trade_opportunities 
            WHERE scan_id = %s 
            LIMIT 1
        """, (latest_scan_id,))
        
        sample = logger.cursor.fetchone()
        if sample and sample[1]:
            import json
            try:
                data = json.loads(sample[1])
                print(f"\n📊 Sample scanner_specific_data for {sample[0]}:")
                print(f"   • Number of fields: {len(data)}")
                print(f"   • Sample fields: {list(data.keys())[:10]}")
            except:
                print(f"   • Data type: {type(sample[1])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        logger.close()

if __name__ == "__main__":
    verify_sync()
