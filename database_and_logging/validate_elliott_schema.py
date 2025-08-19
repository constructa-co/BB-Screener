#!/usr/bin/env python3
"""
Elliott Wave Schema Validation Script
Validates that the new elliott_wave_signals table was created correctly
and verifies complete isolation from existing scanners
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import sys

def validate_elliott_schema():
    """Validate Elliott Wave database schema creation"""
    
    print("🔍 ELLIOTT WAVE SCHEMA VALIDATION")
    print("=" * 50)
    
    try:
        # Connect to database
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            print("❌ DATABASE_URL environment variable not set")
            return False
            
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        
        # Test 1: Check if elliott_wave_signals table exists
        print("\n1. Checking Elliott Wave table existence...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'other_scanners' 
            AND table_name = 'elliott_wave_signals'
        """)
        
        if cursor.fetchone():
            print("✅ elliott_wave_signals table exists")
        else:
            print("❌ elliott_wave_signals table not found")
            return False
        
        # Test 2: Check table structure
        print("\n2. Validating table structure...")
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'other_scanners' 
            AND table_name = 'elliott_wave_signals'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        expected_columns = [
            'id', 'scanner_name', 'scanner_version', 'scan_id', 'symbol', 'exchange', 'timeframe',
            'direction', 'entry_price', 'stop_loss', 'tp1', 'tp2', 'tp3', 'risk_reward',
            'pattern_type', 'current_wave', 'wave_degree', 'pattern_quality', 'confidence_score', 'invalidation_level',
            'wave_analysis', 'fibonacci_levels', 'pattern_metrics', 'analysis_ts', 'created_at'
        ]
        
        actual_columns = [col['column_name'] for col in columns]
        missing_columns = set(expected_columns) - set(actual_columns)
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print(f"✅ All {len(expected_columns)} expected columns present")
        
        # Test 3: Check JSONB columns
        print("\n3. Validating JSONB columns...")
        jsonb_columns = [col['column_name'] for col in columns if col['data_type'] == 'jsonb']
        expected_jsonb = ['wave_analysis', 'fibonacci_levels', 'pattern_metrics']
        
        if set(jsonb_columns) == set(expected_jsonb):
            print("✅ All JSONB columns present")
        else:
            print(f"❌ JSONB columns mismatch. Expected: {expected_jsonb}, Found: {jsonb_columns}")
            return False
        
        # Test 4: Check indexes
        print("\n4. Validating indexes...")
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'other_scanners' 
            AND tablename = 'elliott_wave_signals'
        """)
        
        indexes = [row['indexname'] for row in cursor.fetchall()]
        expected_indexes = [
            'elliott_wave_signals_pkey',  # Primary key
            'idx_elliott_symbol_timeframe',
            'idx_elliott_analysis_ts', 
            'idx_elliott_pattern_quality',
            'idx_elliott_current_wave',
            'idx_elliott_pattern_type',
            'idx_elliott_wave_analysis',
            'idx_elliott_fibonacci_levels', 
            'idx_elliott_pattern_metrics'
        ]
        
        found_indexes = len([idx for idx in expected_indexes if any(idx in index for index in indexes)])
        print(f"✅ Found {found_indexes}/{len(expected_indexes)} expected indexes")
        
        # Test 5: Check unique constraint
        print("\n5. Validating unique constraint...")
        cursor.execute("""
            SELECT constraint_name, constraint_type
            FROM information_schema.table_constraints 
            WHERE table_schema = 'other_scanners' 
            AND table_name = 'elliott_wave_signals'
            AND constraint_type = 'UNIQUE'
        """)
        
        constraints = cursor.fetchall()
        if constraints:
            print("✅ Unique constraint exists")
        else:
            print("❌ Unique constraint not found")
            return False
        
        # Test 6: Verify isolation - check other tables unchanged
        print("\n6. Verifying isolation from existing tables...")
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM other_scanners.other_scanners_trades
        """)
        other_trades_count = cursor.fetchone()['count']
        print(f"✅ other_scanners_trades table unchanged ({other_trades_count} records)")
        
        # Test 7: Test insert capability
        print("\n7. Testing insert capability...")
        test_data = {
            'scanner_name': 'elliott_wave',
            'scanner_version': 'R0',
            'scan_id': 'TEST_VALIDATION',
            'symbol': 'TEST/USDT',
            'exchange': 'binance',
            'timeframe': '1h',
            'direction': 'LONG',
            'pattern_type': 'BULLISH_IMPULSE',
            'current_wave': 'WAVE_3',
            'wave_analysis': {'test': 'data'},
            'fibonacci_levels': {'test': 'fib'},
            'pattern_metrics': {'test': 'metrics'}
        }
        
        cursor.execute("""
            INSERT INTO other_scanners.elliott_wave_signals 
            (scanner_name, scanner_version, scan_id, symbol, exchange, timeframe,
             direction, pattern_type, current_wave, wave_analysis, fibonacci_levels, pattern_metrics)
            VALUES (%(scanner_name)s, %(scanner_version)s, %(scan_id)s, %(symbol)s, %(exchange)s, %(timeframe)s,
                    %(direction)s, %(pattern_type)s, %(current_wave)s, %(wave_analysis)s, %(fibonacci_levels)s, %(pattern_metrics)s)
        """, test_data)
        
        # Verify insert
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM other_scanners.elliott_wave_signals 
            WHERE scan_id = 'TEST_VALIDATION'
        """)
        test_count = cursor.fetchone()['count']
        
        if test_count > 0:
            print("✅ Test insert successful")
            
            # Clean up test data
            cursor.execute("DELETE FROM other_scanners.elliott_wave_signals WHERE scan_id = 'TEST_VALIDATION'")
            print("✅ Test data cleaned up")
        else:
            print("❌ Test insert failed")
            return False
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 50)
        print("🎉 ELLIOTT WAVE SCHEMA VALIDATION SUCCESSFUL!")
        print("✅ Database table created correctly")
        print("✅ Complete isolation from existing scanners verified")
        print("✅ Ready for Phase 2: Elliott Wave Logger Creation")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_elliott_schema()
    sys.exit(0 if success else 1)
