#!/usr/bin/env python3
"""
Simple Scanner Integration - Execute statements individually
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def add_scanner_fields():
    """Add scanner-specific fields one by one"""
    logger = TradeLogger()
    
    # ICT Scanner Gap fields
    ict_fields = [
        "ADD COLUMN IF NOT EXISTS gap_high DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS gap_low DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS gap_size_pct DECIMAL(6,2)",
        "ADD COLUMN IF NOT EXISTS gap_type VARCHAR(20)",
        "ADD COLUMN IF NOT EXISTS swing_high DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS swing_low DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS order_block_high DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS order_block_low DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fvg_high DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fvg_low DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fib_236 DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fib_382 DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fib_500 DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fib_618 DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS fib_786 DECIMAL(20,8)",
    ]
    
    # Volume Scanner fields
    volume_fields = [
        "ADD COLUMN IF NOT EXISTS volume_surge_multiplier DECIMAL(6,2)",
        "ADD COLUMN IF NOT EXISTS relative_volume DECIMAL(10,2)",
        "ADD COLUMN IF NOT EXISTS vwap DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS vwap_deviation DECIMAL(6,2)",
    ]
    
    # Pattern fields
    pattern_fields = [
        "ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(50)",
        "ADD COLUMN IF NOT EXISTS pattern_target DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS pattern_reliability DECIMAL(5,2)",
    ]
    
    # Support/Resistance fields
    sr_fields = [
        "ADD COLUMN IF NOT EXISTS major_resistance_1 DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS major_support_1 DECIMAL(20,8)",
        "ADD COLUMN IF NOT EXISTS pivot_point DECIMAL(20,8)",
    ]
    
    all_fields = ict_fields + volume_fields + pattern_fields + sr_fields
    
    print("🔧 Adding scanner-specific fields...")
    for i, field in enumerate(all_fields, 1):
        try:
            query = f"ALTER TABLE trade_opportunities {field}"
            logger.cursor.execute(query)
            print(f"✅ {i}/{len(all_fields)}: Added {field.split()[-1]}")
        except Exception as e:
            print(f"⚠️ {i}/{len(all_fields)}: {field.split()[-1]} - {e}")
    
    logger.connection.commit()
    print("✅ All scanner fields added!")
    
    # Create indexes
    indexes = [
        ("idx_gap_levels", "gap_high, gap_low"),
        ("idx_gap_size", "gap_size_pct"),
        ("idx_swing_levels", "swing_high, swing_low"),
        ("idx_fib_618", "fib_618"),
        ("idx_volume_surge", "volume_surge_multiplier"),
        ("idx_pattern_type", "pattern_type"),
        ("idx_vwap", "vwap"),
    ]
    
    print("\n🔧 Creating indexes...")
    for name, columns in indexes:
        try:
            query = f"CREATE INDEX IF NOT EXISTS {name} ON trade_opportunities({columns})"
            logger.cursor.execute(query)
            print(f"✅ Created index: {name}")
        except Exception as e:
            print(f"⚠️ Index {name}: {e}")
    
    logger.connection.commit()
    logger.close()
    print("✅ Indexes created!")

def create_views():
    """Create analysis views"""
    logger = TradeLogger()
    
    views = [
        ("all_trades_with_scanner", """
            CREATE OR REPLACE VIEW all_trades_with_scanner AS
            SELECT t.*, s.scan_type, s.scan_timestamp, s.version as scanner_version
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
        """),
        ("bb_trades", """
            CREATE OR REPLACE VIEW bb_trades AS
            SELECT t.*
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scan_type = 'bb_scanner'
        """),
        ("ict_trades", """
            CREATE OR REPLACE VIEW ict_trades AS
            SELECT t.*
            FROM trade_opportunities t
            JOIN scan_results s ON t.scan_id = s.id
            WHERE s.scanner_type LIKE 'ict_scanner%'
        """),
    ]
    
    print("\n🔧 Creating analysis views...")
    for name, query in views:
        try:
            logger.cursor.execute(query)
            print(f"✅ Created view: {name}")
        except Exception as e:
            print(f"⚠️ View {name}: {e}")
    
    logger.connection.commit()
    logger.close()
    print("✅ Views created!")

def main():
    print("🚀 Simple Scanner Integration")
    print("=" * 40)
    
    add_scanner_fields()
    create_views()
    
    print("\n✅ Scanner integration complete!")
    print("\n📋 Test with:")
    print("SELECT * FROM all_trades_with_scanner LIMIT 5;")

if __name__ == "__main__":
    main()
