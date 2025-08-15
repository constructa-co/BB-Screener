#!/usr/bin/env python3
"""
Create enhanced market tables
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade_logger import TradeLogger

def create_enhanced_tables():
    logger = TradeLogger()
    
    # Read SQL file
    with open('create_market_tables.sql', 'r') as f:
        sql = f.read()
    
    # Split into individual statements
    statements = sql.split(';')
    
    for statement in statements:
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                logger.cursor.execute(statement)
                print(f"✅ Executed: {statement[:50]}...")
            except Exception as e:
                print(f"⚠️ Statement failed: {e}")
                print(f"Statement: {statement[:100]}...")
    
    logger.connection.commit()
    logger.close()
    print("✅ Enhanced market tables created!")

if __name__ == "__main__":
    create_enhanced_tables()
