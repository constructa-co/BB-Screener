#!/usr/bin/env python3
# File: /opt/bb-screener/test_fvg_fix.py
import pandas as pd
import psycopg2
import os
from datetime import datetime
import pytz

# Load environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print('❌ DATABASE_URL environment variable not set')
    exit(1)

print("=== TESTING FVG TIMEZONE FIX ===")
print(f"Time: {datetime.now()}")
print("")

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Database connection successful")
    
    # Test current FVG data retrieval
    query = "SELECT detected_at FROM other_scanners.fvg_signals LIMIT 5"
    df = pd.read_sql(query, conn)
    
    print(f"✅ FVG data retrieved: {len(df)} rows")
    print(f"Original data type: {df['detected_at'].dtype}")
    
    if hasattr(df['detected_at'], 'dt'):
        print(f"Is timezone aware? {df['detected_at'].dt.tz is not None}")
        if df['detected_at'].dt.tz is not None:
            print(f"Timezone: {df['detected_at'].dt.tz}")
    else:
        print("Not datetime - converting...")
        df['detected_at'] = pd.to_datetime(df['detected_at'])
        print(f"After conversion - Is timezone aware? {df['detected_at'].dt.tz is not None}")
    
    print("")
    print("=== TESTING THE FIX ===")
    
    # Test the fix
    df['detected_at'] = pd.to_datetime(df['detected_at'])
    
    if df['detected_at'].dt.tz is None:
        df['detected_at'] = df['detected_at'].dt.tz_localize('UTC')
        print("✅ Applied tz_localize (data was not timezone aware)")
    else:
        df['detected_at'] = df['detected_at'].dt.tz_convert('UTC')
        print("✅ Applied tz_convert (data was already timezone aware)")
    
    print(f"Fixed data type: {df['detected_at'].dtype}")
    print(f"Final timezone: {df['detected_at'].dt.tz}")
    
    # Test UAE timezone conversion (what the dashboard needs)
    print("")
    print("=== TESTING UAE TIMEZONE CONVERSION ===")
    
    uae_tz = pytz.timezone('Asia/Dubai')
    
    # Test the specific operation that was failing
    try:
        uae_time = df['detected_at'].dt.tz_convert(uae_tz).dt.strftime('%H:%M')
        print("✅ UAE timezone conversion successful")
        print(f"Sample UAE times: {uae_time.head().tolist()}")
    except Exception as e:
        print(f"❌ UAE timezone conversion failed: {e}")
    
    # Test the age calculation that was also failing
    try:
        now_uae = datetime.now(uae_tz)
        age_calc = df['detected_at'].apply(
            lambda x: f"{(now_uae - x.tz_convert(uae_tz)).total_seconds() / 3600:.1f}h"
        )
        print("✅ Age calculation successful")
        print(f"Sample ages: {age_calc.head().tolist()}")
    except Exception as e:
        print(f"❌ Age calculation failed: {e}")
    
    conn.close()
    print("")
    print("=== TEST COMPLETE ===")
    print("If all tests pass, the fix is ready to implement")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Check database connection and table structure")
