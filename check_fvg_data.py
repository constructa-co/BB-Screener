#!/usr/bin/env python3
"""
Check FVG database for timeframes
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_fvg_data():
    """Check what timeframes are in the FVG database"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set")
        return
    
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check timeframes
        cur.execute("SELECT DISTINCT timeframe, COUNT(*) FROM other_scanners.fvg_signals GROUP BY timeframe ORDER BY timeframe")
        timeframes = cur.fetchall()
        
        print("📊 Timeframes in FVG database:")
        for timeframe, count in timeframes:
            print(f"   {timeframe}: {count} signals")
        
        # Check recent signals
        cur.execute("SELECT symbol, timeframe, detected_at, gap_type FROM other_scanners.fvg_signals ORDER BY detected_at DESC LIMIT 10")
        recent = cur.fetchall()
        
        print("\n📈 Recent FVG signals:")
        for symbol, timeframe, detected_at, gap_type in recent:
            print(f"   {symbol} ({timeframe}) - {gap_type} - {detected_at}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_fvg_data()
