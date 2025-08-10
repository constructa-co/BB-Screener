#!/usr/bin/env python3
"""
Debug Database Insertion
Test database insertion with minimal data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_logger import TradeLogger
from datetime import datetime

def test_simple_insert():
    """Test simple database insertion"""
    
    print("🔍 Testing Simple Database Insertion...")
    
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Create a scan result
        scan_id = logger.log_scan_start('Test_Scanner', '1.0')
        print(f"✅ Created scan_id: {scan_id}")
        
        # Test minimal trade data
        trade_data = {
            'symbol': 'BTCUSDT',
            'exchange': 'Binance',
            'timeframe': '4h',
            'bb_score': 85.0,
            'probability': 85.0,
            'risk_reward_ratio': 2.5,
            'current_price': 115000.0,
            'entry_price': 115000.0,
            'stop_loss': 110000.0,
            'target_1': 120000.0,
            'target_2': 125000.0,
            'target_3': 130000.0,
            'rsi': 50.0,
            'mfi': 60.0,
            'stochastic_k': 45.0,
            'volume_surge': 1.5,
            'macd_signal': 'bullish',
            'pattern_type': 'Test Pattern',
            'pattern_quality': 'high',
            'confluence_score': 80.0,
            'historical_win_rate': 75.0,
            'category_win_rate': 70.0,
            'similar_setups_count': 10,
            'market_cap': 1000000000.0,
            'volume_24h': 50000000.0,
            'price_change_24h': 2.5,
            'scanner_type': 'Test_Scanner'
        }
        
        print("📝 Attempting to insert trade...")
        print(f"   Symbol: {trade_data['symbol']}")
        print(f"   Probability: {trade_data['probability']}")
        print(f"   Entry: {trade_data['entry_price']}")
        
        # Try to insert
        success = logger.log_trade_opportunity(scan_id, trade_data)
        
        if success:
            print("✅ Trade inserted successfully!")
        else:
            print("❌ Trade insertion failed")
            
            # Check what's in the database
            logger.cursor.execute("SELECT COUNT(*) FROM trade_opportunities")
            count = logger.cursor.fetchone()[0]
            print(f"   Current trades in database: {count}")
            
            # Check scan results
            logger.cursor.execute("SELECT COUNT(*) FROM scan_results")
            scan_count = logger.cursor.fetchone()[0]
            print(f"   Current scans in database: {scan_count}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.close()

def check_database_schema():
    """Check database schema"""
    
    print("\n🔍 Checking Database Schema...")
    
    logger = TradeLogger()
    
    if not logger.connection:
        print("❌ Database connection failed")
        return
    
    try:
        # Check trade_opportunities table
        logger.cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'trade_opportunities'
            ORDER BY ordinal_position
        """)
        
        columns = logger.cursor.fetchall()
        print("📋 trade_opportunities columns:")
        for col in columns:
            print(f"   {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
            
    except Exception as e:
        print(f"❌ Error checking schema: {e}")
    
    finally:
        logger.close()

if __name__ == "__main__":
    print("🔧 Database Insertion Debug")
    print("=" * 50)
    
    check_database_schema()
    test_simple_insert() 