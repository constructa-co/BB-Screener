#!/usr/bin/env python3
"""
Elliott Wave Integration Validation Script
Comprehensive validation of Phase 3 integration
Tests database logging, isolation, and data integrity
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json

def validate_elliott_integration():
    """Comprehensive validation of Phase 3 integration"""
    
    try:
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SET search_path TO other_scanners")
        
        print("=" * 80)
        print("ELLIOTT WAVE INTEGRATION VALIDATION")
        print("=" * 80)
        
        # Test 1: Check elliott_wave_signals table
        cur.execute("SELECT COUNT(*) as count FROM elliott_wave_signals")
        elliott_count = cur.fetchone()['count']
        print(f"✅ Elliott signals in database: {elliott_count}")
        
        # Test 2: Check for contamination in other_scanners_trades
        cur.execute("""
            SELECT COUNT(*) as count 
            FROM other_scanners_trades 
            WHERE scanner_name LIKE '%elliott%' 
               OR scanner_name LIKE '%wave%'
        """)
        contamination = cur.fetchone()['count']
        if contamination == 0:
            print("✅ No Elliott data in other_scanners_trades (perfect isolation)")
        else:
            print(f"❌ WARNING: Found {contamination} Elliott entries in wrong table!")
        
        # Test 3: Check recent Elliott signals
        cur.execute("""
            SELECT symbol, timeframe, current_wave, pattern_type, 
                   pattern_quality, entry_price, created_at
            FROM elliott_wave_signals
            WHERE created_at > NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 5
        """)
        recent = cur.fetchall()
        
        if recent:
            print(f"\n✅ Recent Elliott Patterns ({len(recent)} found):")
            for r in recent:
                print(f"  - {r['symbol']} ({r['timeframe']}): Wave {r['current_wave']} "
                      f"{r['pattern_type']} - Quality: {r['pattern_quality']:.1f}")
        else:
            print("\n⚠️ No recent Elliott patterns found (run scanner first)")
        
        # Test 4: Validate JSONB structure
        cur.execute("""
            SELECT 
                jsonb_typeof(wave_analysis) as wave_type,
                jsonb_typeof(fibonacci_levels) as fib_type,
                jsonb_typeof(pattern_metrics) as metrics_type
            FROM elliott_wave_signals
            LIMIT 1
        """)
        json_types = cur.fetchone()
        if json_types:
            print(f"\n✅ JSONB Structure Validation:")
            print(f"  - wave_analysis: {json_types['wave_type']}")
            print(f"  - fibonacci_levels: {json_types['fib_type']}")
            print(f"  - pattern_metrics: {json_types['metrics_type']}")
        else:
            print("\n⚠️ No JSONB data to validate (run scanner first)")
        
        # Test 5: Check scan_id grouping
        cur.execute("""
            SELECT scan_id, COUNT(*) as patterns_per_scan
            FROM elliott_wave_signals
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY scan_id
            ORDER BY scan_id DESC
            LIMIT 3
        """)
        scans = cur.fetchall()
        if scans:
            print(f"\n✅ Scan Grouping (last 3 runs):")
            for s in scans:
                print(f"  - Scan {s['scan_id']}: {s['patterns_per_scan']} patterns")
        else:
            print("\n⚠️ No scan grouping data (run scanner first)")
        
        # Test 6: Verify no duplicate entries
        cur.execute("""
            SELECT symbol, timeframe, current_wave, COUNT(*) as dupe_count
            FROM elliott_wave_signals
            GROUP BY symbol, timeframe, current_wave, scan_id
            HAVING COUNT(*) > 1
        """)
        dupes = cur.fetchall()
        if not dupes:
            print("\n✅ No duplicate entries found (upsert logic working)")
        else:
            print(f"\n⚠️ Found {len(dupes)} potential duplicates")
        
        # Test 7: Check data quality
        cur.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN pattern_quality > 0 THEN 1 END) as quality_records,
                COUNT(CASE WHEN confidence_score > 0 THEN 1 END) as confidence_records,
                COUNT(CASE WHEN wave_analysis != '{}' THEN 1 END) as wave_data_records
            FROM elliott_wave_signals
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)
        quality = cur.fetchone()
        if quality['total_records'] > 0:
            print(f"\n✅ Data Quality Check:")
            print(f"  - Total records: {quality['total_records']}")
            print(f"  - Quality scores: {quality['quality_records']}")
            print(f"  - Confidence scores: {quality['confidence_records']}")
            print(f"  - Wave analysis data: {quality['wave_data_records']}")
        
        # Test 8: Check timeframes
        cur.execute("""
            SELECT timeframe, COUNT(*) as count
            FROM elliott_wave_signals
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY timeframe
        """)
        timeframes = cur.fetchall()
        if timeframes:
            print(f"\n✅ Timeframe Distribution:")
            for tf in timeframes:
                print(f"  - {tf['timeframe']}: {tf['count']} patterns")
        
        # Test 9: Check pattern types
        cur.execute("""
            SELECT pattern_type, COUNT(*) as count
            FROM elliott_wave_signals
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY pattern_type
        """)
        patterns = cur.fetchall()
        if patterns:
            print(f"\n✅ Pattern Type Distribution:")
            for p in patterns:
                print(f"  - {p['pattern_type']}: {p['count']} patterns")
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        
        # Summary
        if elliott_count > 0 and contamination == 0:
            print("\n🎉 SUCCESS: Elliott Wave integration is working perfectly!")
            print("✅ Patterns are being logged to correct table")
            print("✅ Complete isolation from existing scanners maintained")
            print("✅ Ready for production deployment")
            return True
        elif elliott_count == 0:
            print("\n⚠️ No Elliott patterns logged yet - run scanner first")
            print("💡 Run: python manual_scanners/1_hour_scanners/elliot_waves_scanner_1h_r1.py")
            return False
        else:
            print("\n❌ Issues detected - check validation results above")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

if __name__ == "__main__":
    success = validate_elliott_integration()
    exit(0 if success else 1)
