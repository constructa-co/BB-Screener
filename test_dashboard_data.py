#!/usr/bin/env python3
# File: /opt/bb-screener/test_dashboard_data.py

import psycopg2
import os
from datetime import datetime, timedelta

# Load environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

def test_dashboard_data():
    print('=== Dashboard Data Check ===')
    print(f'Time: {datetime.now()}')
    print('')
    
    if not DATABASE_URL:
        print('❌ DATABASE_URL environment variable not set')
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Test tables that should have recent data
        tables = [
            ('FVG Signals', 'other_scanners.fvg_signals'),
            ('Flagpole', 'other_scanners.flagpole_signals'),
            ('Supply/Demand', 'other_scanners.supply_demand_zones'),
            ('Trend Following', 'other_scanners.trend_following_signals'),
            ('Fibonacci', 'other_scanners.fibonacci_signals'),
            ('Wyckoff', 'other_scanners.wyckoff_signals'),
            ('ICT', 'other_scanners.ict_signals'),
            ('Elliott Wave', 'elliott_wave.signals')
        ]
        
        print('Signals in last hour:')
        print('-' * 50)
        
        total_signals = 0
        for name, table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE detected_at > NOW() - INTERVAL '1 hour'")
                count = cur.fetchone()[0]
                print(f'{name:20}: {count:4} signals')
                total_signals += count
            except Exception as e:
                print(f'{name:20}: ERROR - {e}')
        
        print('-' * 50)
        print(f'Total signals: {total_signals}')
        
        # Test specific data retrieval (like dashboard would do)
        print('')
        print('=== Testing Dashboard-Style Queries ===')
        
        # Test FVG data retrieval
        try:
            cur.execute("""
                SELECT symbol, timeframe, detected_at, gap_type, current_price, entry_price, stop_loss, target_1, target_2, target_3, setup_score, gap_status
                FROM other_scanners.fvg_signals
                WHERE detected_at > NOW() - INTERVAL '24 hours'
                ORDER BY detected_at DESC
                LIMIT 5
            """)
            fvg_data = cur.fetchall()
            print(f'FVG recent data: {len(fvg_data)} rows retrieved')
            if fvg_data:
                print(f'  Sample: {fvg_data[0][:3]}...')
        except Exception as e:
            print(f'FVG query error: {e}')
        
        # Test Flagpole data retrieval
        try:
            cur.execute("""
                SELECT symbol, timeframe, detected_at, pattern_type, direction, current_price, breakout_level, target_price, stop_loss, potential_pct, risk_reward, score
                FROM other_scanners.flagpole_signals
                WHERE detected_at > NOW() - INTERVAL '24 hours'
                ORDER BY detected_at DESC
                LIMIT 5
            """)
            flagpole_data = cur.fetchall()
            print(f'Flagpole recent data: {len(flagpole_data)} rows retrieved')
            if flagpole_data:
                print(f'  Sample: {flagpole_data[0][:3]}...')
        except Exception as e:
            print(f'Flagpole query error: {e}')
        
        conn.close()
        
        print('')
        print('=== Dashboard Data Test Complete ===')
        if total_signals > 0:
            print('✅ Data is available - Dashboard should work')
        else:
            print('❌ No recent data - Check scanner execution')
            
    except Exception as e:
        print(f'Database connection error: {e}')

if __name__ == "__main__":
    test_dashboard_data()
