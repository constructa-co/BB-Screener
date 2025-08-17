#!/usr/bin/env python3
"""
Monitor Other Scanners Activity
Shows summary of all scanners in the other_scanners database
"""

import psycopg2
import os
from datetime import datetime, timedelta

def get_database_url():
    """Get the database URL for other scanners"""
    DATABASE_URL = os.getenv('OTHER_SCANNERS_DATABASE_URL')
    
    if not DATABASE_URL:
        # Fallback to main DATABASE_URL with schema
        main_db_url = os.getenv('DATABASE_URL')
        if main_db_url:
            if '?' in main_db_url:
                DATABASE_URL = main_db_url + '&options=-csearch_path=other_scanners'
            else:
                DATABASE_URL = main_db_url + '?options=-csearch_path=other_scanners'
        else:
            print("❌ No database URL found")
            return None
    
    return DATABASE_URL

def monitor_scanners():
    """Monitor all scanners in the other_scanners database"""
    DATABASE_URL = get_database_url()
    if not DATABASE_URL:
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Set schema
        cur.execute("SET search_path TO other_scanners")
        
        # Get scanner activity summary
        cur.execute("""
            SELECT 
                scanner_name,
                COUNT(*) as total_trades,
                MAX(created_at) as last_run,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 1 END) as trades_24h,
                COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as trades_1h,
                AVG(CAST(execution_metadata->>'final_quality' AS FLOAT)) as avg_quality
            FROM other_scanners_trades
            GROUP BY scanner_name
            ORDER BY last_run DESC;
        """)
        
        print("\n" + "="*80)
        print("OTHER SCANNERS ACTIVITY MONITOR")
        print("="*80)
        print(f"{'Scanner':<20} {'Total':<8} {'24h':<6} {'1h':<4} {'Avg Quality':<12} {'Last Run':<20}")
        print("-"*80)
        
        total_scanners = 0
        total_trades = 0
        total_24h = 0
        
        for row in cur.fetchall():
            scanner_name, total, last_run, trades_24h, trades_1h, avg_quality = row
            total_scanners += 1
            total_trades += total
            total_24h += trades_24h or 0
            
            avg_quality_str = f"{avg_quality:.1f}" if avg_quality else "N/A"
            last_run_str = last_run.strftime("%Y-%m-%d %H:%M") if last_run else "Never"
            
            print(f"{scanner_name:<20} {total:<8} {trades_24h or 0:<6} {trades_1h or 0:<4} {avg_quality_str:<12} {last_run_str:<20}")
        
        print("-"*80)
        print(f"SUMMARY: {total_scanners} scanners, {total_trades} total trades, {total_24h} trades in last 24h")
        
        # Get recent activity
        cur.execute("""
            SELECT 
                scanner_name,
                symbol,
                timeframe,
                created_at,
                execution_metadata->>'final_quality' as quality
            FROM other_scanners_trades
            WHERE created_at > NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC
            LIMIT 10;
        """)
        
        recent_trades = cur.fetchall()
        if recent_trades:
            print(f"\n📊 RECENT ACTIVITY (Last Hour):")
            print(f"{'Scanner':<15} {'Symbol':<12} {'TF':<4} {'Quality':<8} {'Time':<20}")
            print("-"*60)
            for row in recent_trades:
                scanner, symbol, tf, created_at, quality = row
                quality_str = f"{float(quality):.1f}" if quality else "N/A"
                time_str = created_at.strftime("%H:%M:%S")
                print(f"{scanner:<15} {symbol:<12} {tf:<4} {quality_str:<8} {time_str:<20}")
        
        # Check for any errors
        cur.execute("""
            SELECT COUNT(*) as error_count
            FROM other_scanners_trades
            WHERE status = 'ERROR'
            AND created_at > NOW() - INTERVAL '24 hours';
        """)
        
        error_count = cur.fetchone()[0]
        if error_count > 0:
            print(f"\n⚠️  WARNING: {error_count} trades with ERROR status in last 24 hours")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")

def check_specific_scanner(scanner_name):
    """Check details for a specific scanner"""
    DATABASE_URL = get_database_url()
    if not DATABASE_URL:
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Set schema
        cur.execute("SET search_path TO other_scanners")
        
        # Get scanner details
        cur.execute("""
            SELECT 
                symbol,
                timeframe,
                side,
                entry_price,
                stop_loss,
                take_profit,
                created_at,
                execution_metadata->>'final_quality' as quality,
                execution_metadata->>'risk_reward' as risk_reward
            FROM other_scanners_trades
            WHERE scanner_name = %s
            ORDER BY created_at DESC
            LIMIT 20;
        """, (scanner_name,))
        
        trades = cur.fetchall()
        if trades:
            print(f"\n📈 DETAILS FOR {scanner_name.upper()}:")
            print(f"{'Symbol':<12} {'TF':<4} {'Side':<6} {'Entry':<10} {'SL':<10} {'TP':<10} {'Quality':<8} {'R/R':<6} {'Time':<20}")
            print("-"*90)
            for row in trades:
                symbol, tf, side, entry, sl, tp, created_at, quality, rr = row
                quality_str = f"{float(quality):.1f}" if quality else "N/A"
                rr_str = f"{float(rr):.1f}" if rr else "N/A"
                time_str = created_at.strftime("%Y-%m-%d %H:%M")
                print(f"{symbol:<12} {tf:<4} {side:<6} {entry:<10.4f} {sl:<10.4f} {tp:<10.4f} {quality_str:<8} {rr_str:<6} {time_str:<20}")
        else:
            print(f"❌ No trades found for {scanner_name}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check specific scanner
        scanner_name = sys.argv[1]
        check_specific_scanner(scanner_name)
    else:
        # Show general overview
        monitor_scanners()
