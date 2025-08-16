#!/usr/bin/env python3
from trade_logger import TradeLogger

def check_database():
    logger = TradeLogger()
    
    # Check current trades with JSONB
    logger.cursor.execute("SELECT COUNT(*) as count FROM trade_opportunities WHERE scanner_specific_data IS NOT NULL")
    result = logger.cursor.fetchone()
    print(f"Current trades with JSONB: {result['count']}")
    
    # Check latest trade field count
    logger.cursor.execute("""
        SELECT symbol, scanner_specific_data 
        FROM trade_opportunities 
        WHERE scanner_specific_data IS NOT NULL
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    
    latest = logger.cursor.fetchone()
    if latest and latest['scanner_specific_data']:
        field_count = len(latest['scanner_specific_data'])
        print(f"Latest trade ({latest['symbol']}) has {field_count} fields in JSONB")
        
        if field_count >= 100:
            print("✅ Enhanced logger is working correctly!")
        else:
            print(f"❌ Only {field_count} fields (expected 100+)")
    else:
        print("❌ No trades with JSONB data found")
    
    logger.close()

if __name__ == "__main__":
    check_database()
