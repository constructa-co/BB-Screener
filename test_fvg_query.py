#!/usr/bin/env python3
# File: /opt/bb-screener/test_fvg_query.py
import pandas as pd
import psycopg2
import os

# Load environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print('❌ DATABASE_URL environment variable not set')
    exit(1)

print("=== TESTING FVG QUERY AND DATA TYPES ===")
print(f"Time: {pd.Timestamp.now()}")
print("")

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Database connection successful")
    
    # Test the exact query the dashboard uses
    query = """
    SELECT * FROM other_scanners.fvg_signals 
    WHERE detected_at > NOW() - INTERVAL '24 hours'
    ORDER BY detected_at DESC
    LIMIT 10
    """
    
    df = pd.read_sql(query, conn)
    print(f"✅ Retrieved {len(df)} rows")
    print(f"✅ Columns: {df.columns.tolist()}")
    print(f"✅ detected_at dtype: {df['detected_at'].dtype}")
    
    # Check data types for problematic columns
    print("")
    print("=== DATA TYPE ANALYSIS ===")
    for col in df.columns:
        print(f"{col:20}: {df[col].dtype} | Sample: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
    
    # Test timezone conversion (what the dashboard needs)
    print("")
    print("=== TESTING TIMEZONE CONVERSION ===")
    try:
        df['detected_at'] = pd.to_datetime(df['detected_at'])
        print(f"✅ After to_datetime: {df['detected_at'].dtype}")
        
        if hasattr(df['detected_at'], 'dt'):
            print(f"✅ Has timezone: {df['detected_at'].dt.tz}")
            
            # Test UAE timezone conversion
            import pytz
            uae_tz = pytz.timezone('Asia/Dubai')
            uae_time = df['detected_at'].dt.tz_convert(uae_tz).dt.strftime('%H:%M')
            print(f"✅ UAE timezone conversion successful: {uae_time.head().tolist()}")
            
            # Test age calculation
            now_uae = pd.Timestamp.now(uae_tz)
            age_calc = df['detected_at'].apply(
                lambda x: f"{(now_uae - x.tz_convert(uae_tz)).total_seconds() / 3600:.1f}h"
            )
            print(f"✅ Age calculation successful: {age_calc.head().tolist()}")
            
    except Exception as e:
        print(f"❌ Timezone conversion failed: {e}")
    
    # Test PyArrow serialization (what's causing the dashboard error)
    print("")
    print("=== TESTING PYARROW SERIALIZATION ===")
    try:
        import pyarrow as pa
        table = pa.Table.from_pandas(df)
        print(f"✅ PyArrow serialization successful: {len(table)} rows, {len(table.columns)} columns")
    except Exception as e:
        print(f"❌ PyArrow serialization failed: {e}")
        print("This is what's causing the dashboard error!")
        
        # Try to identify problematic columns
        for col in df.columns:
            try:
                pa.array(df[col])
                print(f"✅ Column '{col}' serializes OK")
            except Exception as col_error:
                print(f"❌ Column '{col}' fails: {col_error}")
    
    conn.close()
    print("")
    print("=== TEST COMPLETE ===")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
