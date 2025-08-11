#!/usr/bin/env python3
"""
Aggressive Add ICT Columns - With timeouts and error handling
"""

import sys
import os
import signal
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def main():
    try:
        print("🔧 Aggressively Adding ICT Scanner Fields...")
        
        # Set timeout for each operation
        signal.signal(signal.SIGALRM, timeout_handler)
        
        logger = TradeLogger()
        
        # Set aggressive timeouts
        logger.cursor.execute("SET statement_timeout = '10000'")  # 10 seconds
        logger.cursor.execute("SET lock_timeout = '5000'")        # 5 seconds
        
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
        for i, field_def in enumerate(ict_fields, 1):
            field_name = field_def.split()[0]
            print(f"Adding {i}/13: {field_name}...")
            
            try:
                # Set 15 second timeout for this operation
                signal.alarm(15)
                
                query = f"ALTER TABLE trade_opportunities ADD COLUMN IF NOT EXISTS {field_def}"
                logger.cursor.execute(query)
                logger.connection.commit()
                
                signal.alarm(0)  # Cancel timeout
                print(f"  ✅ Added {field_name}")
                added += 1
                
            except TimeoutError:
                print(f"  ⏰ TIMEOUT on {field_name} - skipping")
                signal.alarm(0)
                continue
            except Exception as e:
                if "already exists" in str(e):
                    print(f"  ⏭️  {field_name} already exists")
                    added += 1
                else:
                    print(f"  ❌ Failed {field_name}: {e}")
                signal.alarm(0)
                continue
            
            # Small delay between operations
            time.sleep(0.5)

        print(f"\n✅ Added {added}/13 ICT fields successfully!")

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
