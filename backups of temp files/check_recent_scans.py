#!/usr/bin/env python3

from trade_logger import TradeLogger

def check_recent_scans():
    logger = TradeLogger()
    
    print("🔍 CHECKING RECENT BB SCANNER SCANS")
    print("="*50)
    
    # Check recent scans
    logger.cursor.execute("""
        SELECT id, scanner_type, scan_timestamp, total_coins, opportunities_found
        FROM scans 
        WHERE scanner_type = 'bb_scanner'
        ORDER BY scan_timestamp DESC 
        LIMIT 5
    """)
    
    scans = logger.cursor.fetchall()
    print(f"\n📊 Recent BB Scanner Scans:")
    for scan in scans:
        print(f"   Scan {scan['id']}: {scan['total_coins']} coins, {scan['opportunities_found']} opportunities")
    
    # Check trades from recent scans
    logger.cursor.execute("""
        SELECT COUNT(*) as total_trades, 
               COUNT(CASE WHEN entry_price > 0 THEN 1 END) as trades_with_data,
               COUNT(CASE WHEN entry_price = 0 OR entry_price IS NULL THEN 1 END) as trades_without_data
        FROM trade_opportunities t
        JOIN scans s ON t.scan_id = s.id
        WHERE s.scanner_type = 'bb_scanner'
        AND s.scan_timestamp > NOW() - INTERVAL '1 hour'
    """)
    
    result = logger.cursor.fetchone()
    print(f"\n📈 Recent BB Scanner Trades (last hour):")
    print(f"   Total trades: {result['total_trades']}")
    print(f"   Trades with data: {result['trades_with_data']}")
    print(f"   Trades without data: {result['trades_without_data']}")
    
    # Show sample trades
    logger.cursor.execute("""
        SELECT t.symbol, t.entry_price, t.stop_loss, t.target_1, t.probability, s.scan_timestamp
        FROM trade_opportunities t
        JOIN scans s ON t.scan_id = s.id
        WHERE s.scanner_type = 'bb_scanner'
        AND s.scan_timestamp > NOW() - INTERVAL '1 hour'
        ORDER BY t.created_at DESC
        LIMIT 10
    """)
    
    trades = logger.cursor.fetchall()
    print(f"\n📋 Sample Recent Trades:")
    for trade in trades:
        entry = trade['entry_price'] if trade['entry_price'] else 0
        stop = trade['stop_loss'] if trade['stop_loss'] else 0
        target = trade['target_1'] if trade['target_1'] else 0
        prob = trade['probability'] if trade['probability'] else 0
        
        if entry > 0:
            print(f"   ✅ {trade['symbol']}: Entry={entry:.8f}, Stop={stop:.8f}, Target={target:.8f}, Prob={prob}%")
        else:
            print(f"   ❌ {trade['symbol']}: Entry={entry}, Stop={stop}, Target={target}, Prob={prob}%")
    
    logger.close()

if __name__ == "__main__":
    check_recent_scans()
