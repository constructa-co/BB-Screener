#!/usr/bin/env python3
"""
Force Add ICT Scanner Fields - Bypass hanging issues
"""

import psycopg2
import os
from urllib.parse import urlparse

def main():
    try:
        # Use same approach as trade_logger
        from trade_logger import TradeLogger
        logger = TradeLogger()
        
        if not logger.db_url:
            print("❌ No database URL found")
            return
            
        # Parse DATABASE_URL
        url = urlparse(logger.db_url)
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            database=url.path[1:],
            user=url.username,
            password=url.password,
            connect_timeout=5
        )
        conn.set_session(autocommit=True)  # Avoid transaction locks
        cur = conn.cursor()

        print("🔧 Adding ICT Scanner Fields...")

        # Essential ICT fields
        ict_fields = [
            "gap_high DECIMAL(20,8)",
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

        added = 0
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

        # Verify
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trade_opportunities' 
            AND column_name IN ('gap_high', 'gap_low', 'fib_618', 'swing_high')
        """)
        verified = cur.fetchall()
        print(f"🔍 Verified fields: {[v[0] for v in verified]}")

        cur.close()
        conn.close()
        print("✅ Database connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
