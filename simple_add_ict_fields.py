#!/usr/bin/env python3
"""
Simple Add ICT Fields - Use existing trade_logger connection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        print("🔧 Adding ICT Scanner Fields...")
        logger = TradeLogger()
        
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
                query = f"ALTER TABLE trade_opportunities ADD COLUMN IF NOT EXISTS {field_def}"
                print(f"Executing: {query}")
                logger.cursor.execute(query)
                logger.connection.commit()
                print(f"  ✅ Added {field_name}")
                added += 1
            except Exception as e:
                if "already exists" in str(e):
                    print(f"  ⏭️  {field_name} already exists")
                else:
                    print(f"  ❌ Failed {field_name}: {e}")

        print(f"\n✅ Added {added} ICT fields successfully!")

        # Verify
        logger.cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'trade_opportunities' 
            AND column_name IN ('gap_high', 'gap_low', 'fib_618', 'swing_high')
        """)
        verified = logger.cursor.fetchall()
        print(f"🔍 Verified fields: {[v['column_name'] for v in verified]}")

        logger.close()
        print("✅ Database connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
