#!/usr/bin/env python3

import psycopg2
from psycopg2.extras import RealDictCursor

def check_db():
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        database="bb_screener",
        user="bb_user",
        password="bb_password"
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("🔍 DATABASE CHECK")
    print("="*50)
    
    # Check tables
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
    tables = cursor.fetchall()
    print("\n📊 Available tables:")
    for table in tables:
        print(f"   - {table['table_name']}")
    
    # Check trade_opportunities
    if any(t['table_name'] == 'trade_opportunities' for t in tables):
        cursor.execute("SELECT COUNT(*) as count FROM trade_opportunities")
        result = cursor.fetchone()
        print(f"\n📈 Total trades in database: {result['count']}")
        
        cursor.execute("SELECT COUNT(*) as count FROM trade_opportunities WHERE entry_price > 0")
        result = cursor.fetchone()
        print(f"   Trades with entry data: {result['count']}")
        
        cursor.execute("SELECT symbol, entry_price, stop_loss, target_1, probability FROM trade_opportunities ORDER BY created_at DESC LIMIT 5")
        trades = cursor.fetchall()
        print(f"\n📋 Recent trades:")
        for trade in trades:
            entry = trade['entry_price'] if trade['entry_price'] else 0
            stop = trade['stop_loss'] if trade['stop_loss'] else 0
            target = trade['target_1'] if trade['target_1'] else 0
            prob = trade['probability'] if trade['probability'] else 0
            
            if entry > 0:
                print(f"   ✅ {trade['symbol']}: Entry={entry:.8f}, Stop={stop:.8f}, Target={target:.8f}, Prob={prob}%")
            else:
                print(f"   ❌ {trade['symbol']}: Entry={entry}, Stop={stop}, Target={target}, Prob={prob}%")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_db()
