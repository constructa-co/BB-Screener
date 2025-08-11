#!/usr/bin/env python3
"""
Direct SQL Add - Bypass hanging issues with direct connection
"""

import psycopg2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        print("🔧 Direct SQL Add ICT Fields...")
        
        # Get DB URL from trade_logger
        logger = TradeLogger()
        db_url = logger.db_url
        logger.close()
        
        if not db_url:
            print("❌ No database URL found")
            return
        
        # Parse URL
        from urllib.parse import urlparse
        url = urlparse(db_url)
        
        # Connect with timeout
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            database=url.path[1:],
            user=url.username,
            password=url.password,
            connect_timeout=10,
            options='-c statement_timeout=30000'  # 30 second timeout
        )
        
        # Set autocommit to avoid transaction locks
        conn.autocommit = True
        cur = conn.cursor()
        
        print("✅ Connected to database")
        
        # Try adding just one column first
        try:
            print("Testing with gap_high...")
            cur.execute("ALTER TABLE trade_opportunities ADD COLUMN IF NOT EXISTS gap_high DECIMAL(20,8)")
            print("✅ Successfully added gap_high")
            
            # If that works, add the rest
            ict_fields = [
                "gap_low DECIMAL(20,8)",
                "gap_size_pct DECIMAL(6,2)",
                "swing_high DECIMAL(20,8)",
                "swing_low DECIMAL(20,8)",
                "order_block_high DECIMAL(20,8)",
                "order_block_low DECIMAL(20,8)",
                "fib_618 DECIMAL(20,8)",
                "fib_500 DECIMAL(20,8)",
                "fib_382 DECIMAL(20,8)",
                "liquidity_sweep_level DECIMAL(20,8)",
                "imbalance_high DECIMAL(20,8)",
                "imbalance_low DECIMAL(20,8)"
            ]
            
            added = 1  # gap_high already added
            for field_def in ict_fields:
                field_name = field_def.split()[0]
                try:
                    cur.execute(f"ALTER TABLE trade_opportunities ADD COLUMN IF NOT EXISTS {field_def}")
                    print(f"  ✅ Added {field_name}")
                    added += 1
                except Exception as e:
                    if "already exists" in str(e):
                        print(f"  ⏭️  {field_name} already exists")
                    else:
                        print(f"  ❌ Failed {field_name}: {e}")
            
            print(f"\n✅ Added {added} ICT fields successfully!")
            
        except Exception as e:
            print(f"❌ Failed to add gap_high: {e}")
        
        cur.close()
        conn.close()
        print("✅ Database connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
