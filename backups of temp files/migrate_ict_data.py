#!/usr/bin/env python3
"""
Migrate ICT Data from JSON to New Columns
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def main():
    try:
        print("🔄 Migrating ICT Data from JSON to New Columns...")
        logger = TradeLogger()
        
        # Get all ICT trades with JSON data
        logger.cursor.execute("""
            SELECT t.id, t.symbol, t.scanner_specific_data
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type LIKE 'ict%'
            AND t.scanner_specific_data IS NOT NULL
            AND t.scanner_specific_data != '{}'
            AND t.scanner_specific_data != 'null'
        """)
        
        ict_trades = logger.cursor.fetchall()
        print(f"📊 Found {len(ict_trades)} ICT trades with JSON data to migrate")
        
        if not ict_trades:
            print("✅ No ICT trades need migration")
            return
        
        # ICT fields to migrate
        ict_fields = [
            'gap_high', 'gap_low', 'gap_size_pct', 'gap_type',
            'swing_high', 'swing_low', 'order_block_high', 'order_block_low',
            'breaker_block_high', 'breaker_block_low',
            'fvg_high', 'fvg_low', 'liquidity_sweep_level',
            'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786',
            'equilibrium_level', 'imbalance_high', 'imbalance_low'
        ]
        
        migrated = 0
        for trade in ict_trades:
            try:
                # Parse JSON data
                if isinstance(trade['scanner_specific_data'], str):
                    data = json.loads(trade['scanner_specific_data'])
                else:
                    data = trade['scanner_specific_data']
                
                # Extract ICT fields
                updates = {}
                for field in ict_fields:
                    if field in data and data[field] is not None:
                        updates[field] = data[field]
                
                if not updates:
                    continue
                
                # Build UPDATE query
                set_clauses = []
                values = []
                for field, value in updates.items():
                    set_clauses.append(f"{field} = %s")
                    values.append(value)
                
                values.append(trade['id'])  # WHERE clause
                
                query = f"""
                    UPDATE trade_opportunities 
                    SET {', '.join(set_clauses)}
                    WHERE id = %s
                """
                
                logger.cursor.execute(query, values)
                migrated += 1
                
                if migrated % 100 == 0:
                    print(f"  ✅ Migrated {migrated}/{len(ict_trades)} trades...")
                    logger.connection.commit()
                
            except Exception as e:
                print(f"  ❌ Error migrating trade {trade['id']}: {e}")
                continue
        
        # Final commit
        logger.connection.commit()
        print(f"\n✅ Migration complete! Updated {migrated} ICT trades")
        
        # Verify migration
        logger.cursor.execute("""
            SELECT COUNT(*) as count
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type LIKE 'ict%'
            AND t.gap_high IS NOT NULL
        """)
        result = logger.cursor.fetchone()
        print(f"🔍 ICT trades with gap_high data: {result['count']}")
        
        logger.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
