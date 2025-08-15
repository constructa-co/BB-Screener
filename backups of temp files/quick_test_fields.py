#!/usr/bin/env python3
"""
Quick test - add just a few key scanner fields
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    print("🔧 Quick Test - Adding Key Scanner Fields")
    print("=" * 40)
    
    try:
        logger = TradeLogger()
        print("✅ Connected to database")
        
        # Test with just 3 key fields
        fields = [
            "gap_high DECIMAL(20,8)",
            "gap_low DECIMAL(20,8)", 
            "fib_618 DECIMAL(20,8)"
        ]
        
        for i, field in enumerate(fields, 1):
            try:
                query = f"ALTER TABLE trade_opportunities ADD COLUMN IF NOT EXISTS {field}"
                print(f"Executing: {query}")
                logger.cursor.execute(query)
                logger.connection.commit()
                print(f"✅ {i}/3: Added {field.split()[0]}")
            except Exception as e:
                print(f"⚠️ {i}/3: {field.split()[0]} - {e}")
        
        logger.close()
        print("✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
