#!/usr/bin/env python3
"""
Execute all scanner integration changes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def run_sql_file(filename):
    """Execute SQL file"""
    try:
        with open(filename, 'r') as f:
            sql_content = f.read()
        
        logger = TradeLogger()
        
        # Split by semicolon and execute each statement
        statements = sql_content.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                logger.cursor.execute(statement)
                print(f"✅ Executed: {statement[:50]}...")
        
        logger.connection.commit()
        logger.close()
        print(f"✅ Successfully executed {filename}")
        
    except Exception as e:
        print(f"❌ Error executing {filename}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 Implementing Comprehensive Scanner Integration")
    print("=" * 60)
    
    # Step 1: Add all scanner fields
    print("\n📊 Step 1: Adding scanner-specific fields...")
    run_sql_file('add_all_scanner_fields.sql')
    
    # Step 2: Create analysis views
    print("\n📊 Step 2: Creating analysis views...")
    run_sql_file('create_analysis_views.sql')
    
    print("\n✅ Scanner integration complete!")
    print("\n📋 Next steps:")
    print("1. trade_logger.py has been updated with enhanced log_trade_opportunity method")
    print("2. enhanced_export.py has been updated with proper labeling")
    print("3. Test with: SELECT * FROM all_trades_with_scanner LIMIT 5;")

if __name__ == "__main__":
    main()
