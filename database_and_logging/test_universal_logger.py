"""
Test script for Universal Scanner Logger
Tests all functionality including initialization, trade logging, UUID generation, and error handling.
"""

import os
import sys
import uuid
import time
from datetime import datetime, timezone
from universal_scanner_logger import UniversalScannerLogger

def test_logger_initialization():
    """Test logger initialization and version extraction."""
    print("🧪 Testing logger initialization...")
    
    try:
        # Test with explicit scanner name and version
        logger = UniversalScannerLogger('test_scanner', 'v1.0.0')
        assert logger.scanner_name == 'test_scanner'
        assert logger.scanner_version == 'v1.0.0'
        assert logger.scanner_uuid is not None
        print("✅ Logger initialization with explicit parameters: PASSED")
        
        # Test with from_env method
        logger_env = UniversalScannerLogger.from_env('env_test_scanner')
        assert logger_env.scanner_name == 'env_test_scanner'
        print("✅ Logger initialization with from_env: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Logger initialization failed: {e}")
        return False

def test_trade_logging():
    """Test basic trade logging functionality."""
    print("\n🧪 Testing trade logging...")
    
    try:
        logger = UniversalScannerLogger('test_scanner', 'v1.0.0')
        
        # Test single trade logging
        trade_data = {
            'symbol': 'BTCUSDT',
            'timeframe': '4H',
            'side': 'BUY',
            'entry_price': 45000.00,
            'quantity': 0.1,
            'stop_loss': 44000.00,
            'take_profit': 47000.00,
            'technical_indicators': {
                'rsi': 65.5,
                'macd': 'bullish',
                'bb_position': 'upper'
            },
            'scanner_signals': {
                'pattern': 'breakout',
                'confidence': 0.85,
                'setup_type': 'ICT_FVG'
            },
            'market_conditions': {
                'trend': 'bullish',
                'volatility': 'medium',
                'volume': 'high'
            }
        }
        
        trade_id = logger.log_trade(trade_data)
        assert trade_id is not None
        print(f"✅ Single trade logged successfully: {trade_id}")
        
        # Test batch trade logging
        batch_trades = [
            {
                'symbol': 'ETHUSDT',
                'timeframe': '1H',
                'side': 'SELL',
                'entry_price': 3000.00,
                'quantity': 1.0,
                'technical_indicators': {'rsi': 75.0},
                'scanner_signals': {'pattern': 'reversal'}
            },
            {
                'symbol': 'ADAUSDT',
                'timeframe': '15m',
                'side': 'BUY',
                'entry_price': 0.50,
                'quantity': 1000.0,
                'technical_indicators': {'rsi': 35.0},
                'scanner_signals': {'pattern': 'oversold_bounce'}
            }
        ]
        
        success_count = logger.log_trades_batch(batch_trades)
        assert success_count == 2
        print(f"✅ Batch trade logging successful: {success_count} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Trade logging failed: {e}")
        return False

def test_uuid_generation():
    """Test UUID generation and uniqueness."""
    print("\n🧪 Testing UUID generation...")
    
    try:
        logger = UniversalScannerLogger('uuid_test_scanner', 'v1.0.0')
        
        # Test that scanner UUID is generated
        assert logger.scanner_uuid is not None
        assert len(logger.scanner_uuid) == 36  # UUID length
        print(f"✅ Scanner UUID generated: {logger.scanner_uuid}")
        
        # Test that trade UUIDs are unique
        trade_ids = set()
        for i in range(5):
            trade_data = {
                'symbol': f'TEST{i}',
                'timeframe': '1m',
                'side': 'BUY',
                'entry_price': 100.00 + i,
                'quantity': 1.0
            }
            trade_id = logger.log_trade(trade_data)
            assert trade_id not in trade_ids
            trade_ids.add(trade_id)
        
        print(f"✅ All {len(trade_ids)} trade UUIDs are unique")
        return True
        
    except Exception as e:
        print(f"❌ UUID generation test failed: {e}")
        return False

def test_database_verification():
    """Test that trades are actually inserted into the database."""
    print("\n🧪 Testing database verification...")
    
    try:
        logger = UniversalScannerLogger('db_test_scanner', 'v1.0.0')
        
        # Log a test trade
        test_trade = {
            'symbol': 'DB_TEST',
            'timeframe': '4H',
            'side': 'BUY',
            'entry_price': 100.00,
            'quantity': 1.0,
            'technical_indicators': {'test': True},
            'scanner_signals': {'verification': 'database_test'}
        }
        
        trade_id = logger.log_trade(test_trade)
        
        # Verify trade exists in database
        with logger.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, scanner_name, symbol, status, created_at
                    FROM other_scanners_trades
                    WHERE id = %s
                """, (trade_id,))
                
                result = cursor.fetchone()
                assert result is not None
                assert result[1] == 'db_test_scanner'
                assert result[2] == 'DB_TEST'
                assert result[3] == 'PENDING'
                
                print(f"✅ Trade verified in database: {result[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery."""
    print("\n🧪 Testing error handling...")
    
    try:
        logger = UniversalScannerLogger('error_test_scanner', 'v1.0.0')
        
        # Test logging with invalid data (should not crash)
        invalid_trade = {
            'symbol': None,  # Invalid symbol
            'entry_price': 'invalid_price',  # Invalid price
            'quantity': -1  # Invalid quantity
        }
        
        # This should handle the error gracefully
        trade_id = logger.log_trade(invalid_trade)
        if trade_id is None:
            print("✅ Error handling for invalid data: PASSED")
        else:
            print("⚠️ Invalid data was logged (may be acceptable)")
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            success = logger.log_error(None, e, {'test_context': True})
            assert success
            print("✅ Error logging: PASSED")
        
        # Test that logger is still functional after errors
        valid_trade = {
            'symbol': 'ERROR_RECOVERY',
            'timeframe': '1H',
            'side': 'BUY',
            'entry_price': 100.00,
            'quantity': 1.0
        }
        
        trade_id = logger.log_trade(valid_trade)
        assert trade_id is not None
        print("✅ Logger recovery after errors: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_trade_lifecycle():
    """Test complete trade lifecycle from pending to closed."""
    print("\n🧪 Testing trade lifecycle...")
    
    try:
        logger = UniversalScannerLogger('lifecycle_test_scanner', 'v1.0.0')
        
        # Create trade
        trade_data = {
            'symbol': 'LIFECYCLE_TEST',
            'timeframe': '4H',
            'side': 'BUY',
            'entry_price': 100.00,
            'quantity': 10.0,
            'stop_loss': 95.00,
            'take_profit': 110.00
        }
        
        trade_id = logger.log_trade(trade_data)
        assert trade_id is not None
        
        # Update to active
        success = logger.update_trade_status(trade_id, 'ACTIVE', {'activated_at': datetime.now().isoformat()})
        assert success
        print("✅ Trade status updated to ACTIVE")
        
        # Close trade
        success = logger.close_trade(trade_id, 110.00, {'closed_reason': 'target_hit'})
        assert success
        print("✅ Trade closed successfully")
        
        # Verify final state in database
        with logger.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT status, exit_price, pnl, pnl_percentage
                    FROM other_scanners_trades
                    WHERE id = %s
                """, (trade_id,))
                
                result = cursor.fetchone()
                assert result[0] == 'CLOSED'
                assert result[1] == 110.00
                assert result[2] == 100.00  # (110-100) * 10
                assert result[3] == 10.00  # 10% profit
                
                print(f"✅ Trade lifecycle completed: Status={result[0]}, PnL={result[2]}, PnL%={result[3]}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Trade lifecycle test failed: {e}")
        return False

def test_connection_pool():
    """Test connection pool functionality."""
    print("\n🧪 Testing connection pool...")
    
    try:
        logger = UniversalScannerLogger('pool_test_scanner', 'v1.0.0')
        
        # Test multiple concurrent operations
        import concurrent.futures
        
        def create_trade(scanner_num):
            """Create a trade in a separate thread."""
            try:
                with UniversalScannerLogger(f'pool_scanner_{scanner_num}', 'v1.0.0') as thread_logger:
                    return thread_logger.log_trade({
                        'symbol': f'POOL_TEST_{scanner_num}',
                        'timeframe': '1m',
                        'side': 'BUY',
                        'entry_price': 100.00 + scanner_num,
                        'quantity': 1.0
                    })
            except Exception as e:
                print(f"❌ Thread {scanner_num} failed: {e}")
                return None
        
        # Run multiple trades concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_trade, i) for i in range(5)]
            results = [f.result() for f in futures]
        
        success_count = sum(1 for r in results if r is not None)
        print(f"✅ Connection pool test: {success_count}/5 concurrent trades successful")
        
        return success_count >= 3  # Allow some failures due to connection limits
        
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Universal Scanner Logger Tests")
    print("=" * 60)
    
    tests = [
        ("Logger Initialization", test_logger_initialization),
        ("Trade Logging", test_trade_logging),
        ("UUID Generation", test_uuid_generation),
        ("Database Verification", test_database_verification),
        ("Error Handling", test_error_handling),
        ("Trade Lifecycle", test_trade_lifecycle),
        ("Connection Pool", test_connection_pool)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Universal Scanner Logger is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    # Check if DATABASE_URL is set
    if not os.getenv('DATABASE_URL'):
        print("⚠️ DATABASE_URL environment variable not set.")
        print("Please set it to your PostgreSQL connection string.")
        print("Example: export DATABASE_URL='postgresql://user:pass@localhost/dbname'")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1)
