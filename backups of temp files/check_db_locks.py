#!/usr/bin/env python3
"""
Check Database Locks and Active Processes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        logger = TradeLogger()
        
        print("🔍 Checking database locks and active processes...")
        
        # Check active queries
        logger.cursor.execute("""
            SELECT pid, state, query_start, query 
            FROM pg_stat_activity 
            WHERE state = 'active' 
            AND query NOT LIKE '%pg_stat_activity%'
        """)
        active = logger.cursor.fetchall()
        
        print(f"📊 Active queries: {len(active)}")
        for row in active:
            print(f"  PID {row['pid']}: {row['state']} - {row['query'][:50]}...")
        
        # Check locks on trade_opportunities table
        logger.cursor.execute("""
            SELECT l.pid, l.mode, l.granted, a.query
            FROM pg_locks l
            JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation = 'trade_opportunities'::regclass
        """)
        locks = logger.cursor.fetchall()
        
        print(f"\n🔒 Locks on trade_opportunities: {len(locks)}")
        for row in locks:
            print(f"  PID {row['pid']}: {row['mode']} ({'granted' if row['granted'] else 'waiting'})")
            if row['query']:
                print(f"    Query: {row['query'][:50]}...")
        
        # Check if scanner is running
        logger.cursor.execute("""
            SELECT COUNT(*) as count
            FROM pg_stat_activity 
            WHERE query LIKE '%main_scanner%' OR query LIKE '%ict_scanner%'
        """)
        scanner_running = logger.cursor.fetchone()['count']
        print(f"\n🤖 Scanner processes running: {scanner_running}")
        
        logger.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
