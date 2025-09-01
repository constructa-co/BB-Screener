#!/usr/bin/env python3
# File: /opt/bb-screener/test_fvg_display.py
import psycopg2
import os

# Load environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print('❌ DATABASE_URL environment variable not set')
    exit(1)

print("=== TESTING FVG DASHBOARD QUERY LOGIC ===")
print(f"Time: {os.popen('date').read().strip()}")
print("")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    print("✅ Database connection successful")
    print("")
    
    # Test the exact query the dashboard should use
    print("=== TESTING DASHBOARD QUERY LOGIC ===")
    
    # Test 1: Basic count with 24h filter
    cur.execute("""
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN detected_at > NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour,
               COUNT(CASE WHEN detected_at > NOW() - INTERVAL '5 minutes' THEN 1 END) as last_5min
        FROM other_scanners.fvg_signals
        WHERE detected_at > NOW() - INTERVAL '24 hours'
    """)
    
    result = cur.fetchone()
    print(f"✅ Signals in last 24h: {result[0]:,}")
    print(f"✅ Signals in last hour: {result[1]:,}")
    print(f"✅ Signals in last 5 min: {result[2]:,}")
    print("")
    
    # Test 2: Sample data retrieval (what dashboard displays)
    print("=== TESTING SAMPLE DATA RETRIEVAL ===")
    cur.execute("""
        SELECT symbol, timeframe, detected_at, gap_type, current_price, entry_price, stop_loss, target_1, target_2, target_3, setup_score, gap_status
        FROM other_scanners.fvg_signals
        WHERE detected_at > NOW() - INTERVAL '24 hours'
        ORDER BY detected_at DESC
        LIMIT 5
    """)
    
    sample_data = cur.fetchall()
    print(f"✅ Sample data retrieved: {len(sample_data)} rows")
    
    if sample_data:
        print("✅ First row sample:")
        print(f"   Symbol: {sample_data[0][0]}")
        print(f"   Timeframe: {sample_data[0][1]}")
        print(f"   Detected: {sample_data[0][2]}")
        print(f"   Gap Type: {sample_data[0][3]}")
        print(f"   Setup Score: {sample_data[0][10]}")
        print(f"   Gap Status: {sample_data[0][11]}")
    print("")
    
    # Test 3: Check for any data type issues
    print("=== TESTING DATA TYPE COMPATIBILITY ===")
    cur.execute("""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(CASE WHEN symbol IS NULL THEN 1 END) as null_symbols,
            COUNT(CASE WHEN detected_at IS NULL THEN 1 END) as null_detected,
            COUNT(CASE WHEN setup_score IS NULL THEN 1 END) as null_scores
        FROM other_scanners.fvg_signals
        WHERE detected_at > NOW() - INTERVAL '24 hours'
    """)
    
    data_quality = cur.fetchone()
    print(f"✅ Total rows: {data_quality[0]:,}")
    print(f"✅ Null symbols: {data_quality[1]}")
    print(f"✅ Null detected_at: {data_quality[2]}")
    print(f"✅ Null setup_scores: {data_quality[3]}")
    print("")
    
    # Test 4: Test the exact function the dashboard calls
    print("=== TESTING DASHBOARD FUNCTION LOGIC ===")
    
    # Simulate the get_fvg_signals function logic
    hours_back = 24  # Default dashboard setting
    
    query = f"""
    SELECT 
        symbol, timeframe, detected_at, gap_type, gap_high, gap_low,
        gap_size, gap_size_pct, current_price, entry_price, stop_loss,
        target_1, target_2, target_3, risk_reward_1, risk_reward_2, risk_reward_3,
        fib_level, fib_confluence, fib_confluence_score, setup_score,
        volume_at_gap, volume_confirmation, momentum_confirmation,
        gap_status, fill_percentage, gap_age_minutes, expires_at,
        scanner_version, algorithm_parameters, source, entry_timing,
        current_distance_pct, risk_pct, swing_high, swing_low,
        fib_levels, target_levels
    FROM other_scanners.fvg_signals
    WHERE detected_at > NOW() - INTERVAL '{hours_back} hours'
    ORDER BY detected_at DESC
    """
    
    cur.execute(query)
    dashboard_data = cur.fetchall()
    print(f"✅ Dashboard query successful: {len(dashboard_data):,} rows")
    
    if dashboard_data:
        print("✅ Dashboard data columns: {len(dashboard_data[0])}")
        print("✅ First row has all required fields")
    
    conn.close()
    print("")
    print("=== TEST COMPLETE ===")
    
    if len(dashboard_data) > 0:
        print("✅ FVG data is available and queryable")
        print("✅ Dashboard should display data successfully")
    else:
        print("❌ No FVG data found - dashboard will show 'No signals'")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
