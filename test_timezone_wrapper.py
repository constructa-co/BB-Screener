#!/usr/bin/env python3
# File: /opt/bb-screener/test_timezone_wrapper.py
# Test the timezone wrapper with each table type

import psycopg2
import pandas as pd
import os

# Load environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print('❌ DATABASE_URL environment variable not set')
    exit(1)

print("=== TESTING TIMEZONE WRAPPER WITH ALL TABLE TYPES ===")
print(f"Time: {pd.Timestamp.now()}")
print("")

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Database connection successful")
    print("")
    
    # Test each table
    tables = [
        ('other_scanners.fvg_signals', 'timezone-aware'),
        ('other_scanners.supply_demand_zones', 'timezone-naive'),
        ('other_scanners.trend_following_signals', 'timezone-naive')
    ]
    
    for table, expected in tables:
        print(f"=== TESTING {table.upper()} ===")
        print(f"Expected: {expected}")
        
        try:
            # Test direct query first
            query = f"SELECT detected_at FROM {table} LIMIT 1"
            df = pd.read_sql(query, conn)
            
            if df.empty:
                print(f"  ⚠️ No data in {table}")
                continue
                
            df['detected_at'] = pd.to_datetime(df['detected_at'])
            print(f"  Raw detected_at: {df['detected_at'].iloc[0]}")
            print(f"  Has timezone: {df['detected_at'].dt.tz is not None}")
            print(f"  Timezone: {df['detected_at'].dt.tz}")
            
            # Apply timezone fix
            if df['detected_at'].dt.tz is None:
                # Timezone-naive - add UTC
                df['detected_at'] = df['detected_at'].dt.tz_localize('UTC')
                print(f"  ✅ Applied tz_localize (was timezone-naive)")
            else:
                # Already timezone-aware - ensure it's UTC
                df['detected_at'] = df['detected_at'].dt.tz_convert('UTC')
                print(f"  ✅ Applied tz_convert (was already timezone-aware)")
            
            # Test UAE conversion
            df['uae_time'] = df['detected_at'].dt.tz_convert('Asia/Dubai')
            print(f"  UAE conversion: {'✅ SUCCESS' if not df['uae_time'].isna().any() else '❌ FAILED'}")
            print(f"  UAE time: {df['uae_time'].iloc[0]}")
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
        
        print("")
    
    conn.close()
    print("=== TIMEZONE WRAPPER TEST COMPLETE ===")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
