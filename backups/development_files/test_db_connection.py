#!/usr/bin/env python3
"""
Test script for database connection and TradeLogger functionality
"""

from trade_logger import TradeLogger

def test_database_connection():
    """Test the database connection and basic functionality"""
    
    print("🔍 Testing Database Connection...")
    
    # Test connection
    logger = TradeLogger()
    
    if logger.connection:
        print("✅ Database connected!")
        
        # Test logging a scan
        scan_id = logger.log_scan_start('test_scan', 'v1.0')
        if scan_id:
            print(f"✅ Test scan created with ID: {scan_id}")
            
            # Test logging a trade opportunity
            trade_data = {
                'symbol': 'BTCUSDT',
                'bb_score': 85.5,
                'probability': 78.2,
                'risk_reward_ratio': 2.1,
                'current_price': 45000.0,
                'entry_price': 44800.0,
                'stop_loss': 44200.0,
                'target_1': 45600.0,
                'target_2': 46200.0,
                'target_3': 46800.0,
                'rsi': 65.2,
                'mfi': 58.7,
                'pattern_type': 'bullish_fvg',
                'pattern_quality': 'high',
                'confluence_score': 87.3,
                'scanner_type': 'ict',
                'scanner_specific_data': {
                    'fvg_type': 'bullish',
                    'fvg_size': 120.5,
                    'confluence_indicators': ['rsi_oversold', 'volume_surge']
                }
            }
            
            logger.log_trade_opportunity(scan_id, trade_data)
            print("✅ Test trade opportunity logged successfully")
            
            # Complete the scan
            logger.complete_scan(scan_id, 500, 15, 45.2)
            print("✅ Test scan completed successfully")
            
            # Test retrieving recent trades
            recent_trades = logger.get_recent_trades(limit=5)
            print(f"✅ Retrieved {len(recent_trades)} recent trades")
            
            logger.close()
            print("✅ Database connection closed")
            
        else:
            print("❌ Failed to create test scan")
    else:
        print("❌ Database connection failed")

if __name__ == "__main__":
    test_database_connection() 